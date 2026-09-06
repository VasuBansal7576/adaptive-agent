"""Factory/wiring tests for LearningRuntime; real-provider smoke is separate."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
core_src = os.environ.get("ADAPTIVE_AGENT_CORE_SRC")
if core_src:
    import adaptive_agent

    adaptive_agent.__path__.append(str(Path(core_src) / "adaptive_agent"))
    adaptive_agent.__path__.append(str(ROOT / "src" / "adaptive_agent"))

try:
    from adaptive_agent.candidate import CandidateManager
    from adaptive_agent.learning import PlannerLearningAdapter
    from adaptive_agent.learning_runtime import LearningRuntime, StoreModelObservationSink
    from adaptive_agent.models import CandidateProposal, SkillBundle, SkillVersion
except ImportError as exc:
    pytest.skip(f"durable core is unavailable in this isolated worker: {exc}", allow_module_level=True)

from test_learning_store_integration import ENVIRONMENT, RUN, _setup_store  # noqa: E402


class FakeClient:
    def __init__(self):
        self.feedback = None

    def invoke(self, *, goal, environment, messages, remaining_deadline=None, cancel=None, token_cap=None):
        self.feedback = environment.get("sanitizedFeedback")
        evidence_id = environment["learningContext"]["developmentEvidence"][0]["sourceId"]
        procedure = "Read the current version before retrying a bounded reconciliation."
        payload = {
            "predictedEffect": "reduce version conflicts",
            "editOperations": [{"path": "skills/runtime-reconciliation/procedure", "operation": "add", "value": procedure}],
            "supportingEvidenceIds": [evidence_id],
            "proposerVersion": "runtime-test",
            "skill": {"procedure": procedure},
        }
        import json

        return {"provider": "openai-codex", "model": "test-luna", "responseId": "resp-runtime-test", "text": json.dumps(payload), "usage": {"totalTokens": 12}}


def test_runtime_composes_durable_learning_and_restart_readback(tmp_path: Path):
    store, manager, evidence_id = _setup_store(tmp_path)
    client = FakeClient()
    runtime = LearningRuntime.build(store=store, manager=manager, model_client=client, token_budget=20_000, wall_seconds=20)
    result = runtime.propose_completed_run(RUN)
    assert result.authoritative_candidate["state"] == "validated"
    assert result.candidate_payload["supportingEvidenceIds"] == [evidence_id]
    restarted_store = type(store)(tmp_path)
    restarted_manager = CandidateManager(restarted_store)
    restarted = LearningRuntime.build(store=restarted_store, manager=restarted_manager, model_client=FakeClient(), token_budget=20_000, wall_seconds=20)
    assert restarted.reload_candidate(result.authoritative_candidate["candidate_id"])["state"] == "validated"


def test_learning_model_observation_sink_never_writes_parent_model_response(tmp_path: Path):
    store, _, _ = _setup_store(tmp_path)
    sink = StoreModelObservationSink(store, RUN)

    sink.record_model_observation(
        {"provider": "test", "model": "learning-model", "responseId": "learning-resp", "usage": {"totalTokens": 2, "cost": {"total": 0.000012}}, "durationSeconds": 0.25},
        trusted_parent=True,
    )

    rows = store.list_evidence(RUN)
    observations = [row for row in rows if row.get("evidence_id") == "learning-model-learning-resp"]
    assert len(observations) == 1
    assert observations[0]["event_type"] == "learning_model_observation"
    assert observations[0]["visibility"] == "operator"
    import json
    payload = store.get_artifact(json.loads(observations[0]["source_ref"])["sha256"])
    assert payload["durationSeconds"] == 0.25
    assert payload["nominalCostUsd"] == 0.000012
    assert payload["economicCostStatus"] == "unknown"
    assert "costMicrounits" not in payload
    assert not any(row.get("event_type") == "model_response" and row.get("evidence_id") == "learning-model-learning-resp" for row in rows)


def test_learning_observation_sink_retains_measured_cost_separately(tmp_path: Path):
    store, _, _ = _setup_store(tmp_path)
    sink = StoreModelObservationSink(store, RUN)

    sink.record_model_observation(
        {"provider": "test", "model": "learning-model", "responseId": "measured-resp", "usage": {"totalTokens": 3}, "durationSeconds": 0.5, "costMicrounits": 17, "economicCostStatus": "measured"},
        trusted_parent=True,
    )

    import json
    row = next(row for row in store.list_evidence(RUN) if row.get("evidence_id") == "learning-model-measured-resp")
    payload = store.get_artifact(json.loads(row["source_ref"])["sha256"])
    assert payload["costMicrounits"] == 17
    assert payload["economicCostStatus"] == "measured"
    assert "nominalCostUsd" not in payload


def test_planner_adapter_forwards_usage_cost_and_measured_duration_without_provider_call():
    captured = []

    class Client:
        def invoke(self, **_kwargs):
            return {"provider": "test", "model": "learning-model", "responseId": "adapter-resp", "text": "not parsed here", "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5, "cost": {"total": 0.000021}}}

    class Sink:
        def record_model_observation(self, evidence, *, trusted_parent=False):
            captured.append((dict(evidence), trusted_parent))

    invocation = PlannerLearningAdapter(Client(), Sink())(goal="learn", environment={}, emit=lambda *_args: None)
    assert invocation.response_id == "adapter-resp"
    assert captured[0][1] is True
    evidence = captured[0][0]
    assert evidence["nominalCostUsd"] == 0.000021
    assert evidence["economicCostStatus"] == "unknown"
    assert evidence["durationSeconds"] >= 0


def test_planner_adapter_persists_malformed_optional_cost_as_unknown():
    captured = []

    class Client:
        def invoke(self, **_kwargs):
            return {"provider": "test", "model": "learning-model", "responseId": "malformed-cost", "text": "malformed proposal", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2, "cost": {"total": "not-a-number"}}}

    class Sink:
        def record_model_observation(self, evidence, *, trusted_parent=False):
            captured.append(dict(evidence))

    PlannerLearningAdapter(Client(), Sink())(goal="learn", environment={}, emit=lambda *_args: None)
    assert captured[0]["economicCostStatus"] == "unknown"
    assert "nominalCostUsd" not in captured[0]
    assert captured[0]["durationSeconds"] >= 0


def test_restart_projection_uses_row_bindings_when_json_omits_them(tmp_path: Path):
    import json
    from adaptive_agent.retrieval import content_hash

    store, manager, evidence_id = _setup_store(tmp_path)
    content = "Persisted broker projection with row-level bindings."
    store.save_learning_record(
        f"learning-evidence-{evidence_id}",
        ENVIRONMENT,
        RUN,
        json.dumps({
            "kind": "live_evidence",
            "sourceId": evidence_id,
            "content": content,
            "contentHash": content_hash(content),
            "partition": "development",
            "visibility": "learner",
            "trustClass": "broker",
            "trustedOutcome": True,
        }),
    )

    records = LearningRuntime.build(store=store, manager=manager, model_client=FakeClient())._materialize_run_records(
        environment_id=ENVIRONMENT,
        run_id=RUN,
    )

    persisted = next(record for record in records if record.get("sourceId") == evidence_id)
    assert persisted["environmentId"] == ENVIRONMENT
    assert persisted["runId"] == RUN


def _mark_run(store, *, status: str, passed: object = 0, save_outcome: bool = True):
    row = store.get_run(RUN)
    row["status"] = status
    store.save_run(RUN, row)
    if save_outcome:
        store.save_outcome("out-durable", {"run_id": RUN, "passed": passed, "score": 0.0, "metadata_json": "{}", "checked_at": "2026-01-01T00:00:00+00:00"})
    else:
        with store._connect() as connection:
            connection.execute("DELETE FROM outcomes WHERE run_id = ?", (RUN,))
            connection.commit()


def test_failed_trusted_development_run_can_produce_candidate(tmp_path: Path):
    store, manager, evidence_id = _setup_store(tmp_path)
    _mark_run(store, status="failed", passed=0)
    client = FakeClient()
    result = LearningRuntime.build(store=store, manager=manager, model_client=client, token_budget=20_000, wall_seconds=20).propose_completed_run(RUN)
    assert result.authoritative_candidate["state"] == "validated"
    assert result.candidate_payload["supportingEvidenceIds"] == [evidence_id]
    assert client.feedback == {"status": "failed"}


def test_failed_development_run_without_trusted_outcome_is_rejected(tmp_path: Path):
    store, manager, _ = _setup_store(tmp_path)
    _mark_run(store, status="failed", save_outcome=False)
    with pytest.raises(ValueError, match="trusted evaluator outcome"):
        LearningRuntime.build(store=store, manager=manager, model_client=FakeClient()).propose_completed_run(RUN)


def test_malformed_failure_outcome_is_rejected(tmp_path: Path):
    store, manager, _ = _setup_store(tmp_path)
    _mark_run(store, status="failed", passed="false")
    with pytest.raises(ValueError, match="trusted evaluator outcome"):
        LearningRuntime.build(store=store, manager=manager, model_client=FakeClient()).propose_completed_run(RUN)


def test_rich_broker_projection_keeps_diagnostics_without_secrets(tmp_path: Path):
    import json
    from adaptive_agent.models import sha256_json

    store, manager, evidence_id = _setup_store(tmp_path)
    call_id = "call-rich"
    store.prepare_tool_call({"call_id": call_id, "run_id": RUN, "step_id": "step-1", "environment_id": ENVIRONMENT, "tool": "read", "arguments_json": json.dumps({"invoiceId": "INV-DEV-000", "apiKey": "sk-secret-value"}), "idempotency_key": "idem-rich"})
    result = {"callId": call_id, "toolVersion": "7", "status": "error", "effect": "none", "output": {"status": "open", "expectedAnswer": "hidden-answer", "detail": "version mismatch"}, "error": {"code": "VERSION_CONFLICT", "retry": "after_refresh", "message": "password=hunter2"}}
    result_ref = store.put_artifact(result)
    store.save_tool_result(call_id, json.dumps(result, sort_keys=True, separators=(",", ":")), "none")
    event = store.get_evidence(evidence_id)
    event.pop("evidence_id", None)
    event["source_ref"] = result_ref.model_dump_json(by_alias=True)
    event["content_hash"] = result_ref.sha256
    store.append_evidence(evidence_id, event)
    LearningRuntime.build(store=store, manager=manager, model_client=FakeClient())._materialize_run_records(environment_id=ENVIRONMENT, run_id=RUN)
    records = store.list_learning_records(environment_id=ENVIRONMENT, run_id=RUN)
    content = next(json.loads(row["record_json"])["content"] for row in records if row["record_id"] in {f"learning-evidence-{evidence_id}", f"learning-broker-{evidence_id}"} and '"tool":"read"' in json.loads(row["record_json"])["content"])
    assert '"tool":"read"' in content
    assert '"code":"VERSION_CONFLICT"' in content
    assert '"retry":"after_refresh"' in content
    assert "apiKey" not in content and "sk-secret-value" not in content
    assert "expectedAnswer" not in content and "hidden-answer" not in content
    assert "hunter2" not in content


def _register_dev_run(store, registry, env_id: str, run_id: str, *, status: str, passed: bool, evidence_ids: list[str]):
    """Persist a real completed development run with trusted evidence."""
    import hashlib
    import json
    from datetime import datetime, timezone

    from adaptive_agent.models import ArtifactRef, EnvironmentManifest, RunRecord, RunStatus, TaskInput, ToolSchema, sha256_json
    from adaptive_agent.retrieval import content_hash

    doc_ref = store.put_artifact(f"Public documentation for {env_id}.")
    manifest = EnvironmentManifest(
        environmentId=env_id,
        version="1",
        docs=[doc_ref],
        toolSchemas=[ToolSchema(name="read", version="1", inputSchema={"type": "object"}, outputSchema={"type": "object"}, effect="read")],
        policyRef=ArtifactRef(id="policy", version="1", sha256="1" * 64),
        evaluatorRef=ArtifactRef(id="evaluator", version="1", sha256="2" * 64),
        resetRef=ArtifactRef(id="reset", version="1", sha256="3" * 64),
    )
    if store.get_environment(env_id) is None:
        store.register_environment(env_id, "1", store.put_artifact(manifest.model_dump(mode="json", by_alias=True)))
    task_id = f"task-{run_id}"
    task = TaskInput(taskId=task_id, environmentRef=ArtifactRef(id=env_id, version="1", sha256="4" * 64), goal=f"goal for {env_id}", partition="development")
    task_ref = store.put_artifact(task.model_dump(mode="json", by_alias=True))
    store.register_task(task_id, env_id, "1", task_ref.model_dump_json(), "development", f"goal for {env_id}")
    record = RunRecord(
        taskRef=task_ref,
        environmentRef=ArtifactRef(id=env_id, version="1", sha256="4" * 64),
        policyRef=ArtifactRef(id="policy", version="1", sha256="1" * 64),
        modelProfileRef=ArtifactRef(id="model", version="1", sha256="5" * 64),
        skillBundleRef=ArtifactRef(id="bundle", version="1", sha256="6" * 64),
        budgetRef=ArtifactRef(id="budget", version="1", sha256="7" * 64),
        runId=run_id,
        status=RunStatus(status),
    )
    store.save_run(run_id, {"parent_run_id": None, "task_id": task_id, "environment_id": env_id, "bundle_id": "base", "status": status, "idempotency_key": f"idem-{run_id}", "last_event_sequence": 1, "created_at": datetime.now(timezone.utc).isoformat(), "run_json": record.model_dump_json(by_alias=True)})
    doc_content = f"Public documentation for {env_id}."
    store.save_learning_record(f"learning-doc-{env_id}", env_id, run_id, json.dumps({"kind": "public_doc", "sourceId": doc_ref.id, "content": doc_content, "contentHash": content_hash(doc_content), "environmentId": env_id, "visibility": "public"}, sort_keys=True, separators=(",", ":")))
    for index, evidence_id in enumerate(evidence_ids):
        content = f"broker observed development evidence {evidence_id}"
        ref = store.put_artifact(content)
        store.append_evidence(evidence_id, {"run_id": run_id, "sequence": index + 1, "event_type": "tool_result", "content_hash": sha256_json(content), "source_ref": ref.model_dump_json(), "trust_class": "broker", "visibility": "learner", "redacted": 1})
        store.save_learning_record(f"learning-evidence-{evidence_id}", env_id, run_id, json.dumps({"kind": "live_evidence", "sourceId": evidence_id, "content": content, "contentHash": content_hash(content), "sourceContentHash": sha256_json(content), "environmentId": env_id, "runId": run_id, "partition": "development", "visibility": "learner", "trustClass": "broker", "trustedOutcome": True, "outcomePassed": passed}, sort_keys=True, separators=(",", ":")))
    store.save_outcome(f"out-{run_id}", {"run_id": run_id, "passed": 1 if passed else 0, "score": 1.0, "metadata_json": "{}", "checked_at": datetime.now(timezone.utc).isoformat()})
    return list(evidence_ids)


class _ScriptedLearningClient:
    """Deterministic model client that cites the requested evidence id."""

    def __init__(self, cited_source_id: str):
        self.cited = cited_source_id
        self.seen_evidence: list[str] = []

    def invoke(self, *, goal, environment, messages, remaining_deadline=None, cancel=None, token_cap=None):
        import json

        evidence = environment["learningContext"]["developmentEvidence"]
        self.seen_evidence = [item["sourceId"] for item in evidence]
        assert self.cited in self.seen_evidence, f"cited source {self.cited} never entered learner context"
        procedure = "Reuse the observed development procedure."
        payload = {
            "predictedEffect": "transfer observed procedure",
            "editOperations": [{"path": "skills/multi-run/procedure", "operation": "add", "value": procedure}],
            "supportingEvidenceIds": [self.cited],
            "proposerVersion": "runtime-test",
            "skill": {"procedure": procedure},
        }
        return {"provider": "openai-codex", "model": "test-luna", "responseId": "resp-multi", "text": json.dumps(payload), "usage": {"totalTokens": 12}}


def test_launch_learning_multi_run_cites_second_run_and_preserves_coverage(tmp_path: Path):
    """Actual DurableRuntime.launch_learning with real durable runs, the real
    retriever/validator/candidate sink, and a scripted model client."""
    pytest.importorskip("fastapi")
    from types import SimpleNamespace

    from adaptive_agent.app import DurableRuntime
    from adaptive_agent.controller import Controller
    from adaptive_agent.environment import EnvironmentRegistry
    from adaptive_agent.store import Store

    store = Store(tmp_path)
    registry = EnvironmentRegistry(store)
    run_a = "run-dev-a"
    run_b = "run-dev-b"
    # >8 evidence records on the primary run plus a later selected failure in
    # a second environment: per-run bounding must keep the failed run's record.
    ids_a = _register_dev_run(store, registry, "env-a", run_a, status="succeeded", passed=True, evidence_ids=[f"ev-a{i}" for i in range(10)])
    ids_b = _register_dev_run(store, registry, "env-b", run_b, status="failed", passed=False, evidence_ids=["ev-b0"])
    controller = Controller(store, registry)
    base = SkillBundle(skills=[SkillVersion(skillId="existing", version="1", procedure="Keep the existing procedure.")])
    controller.candidates.initialize_active_bundle(base)

    runtime = DurableRuntime.__new__(DurableRuntime)
    runtime.controller = controller
    runtime.registry = registry
    runtime.packages = {"env-a": object(), "env-b": object()}
    runtime.model_runner = None
    runtime.learning_model_client = _ScriptedLearningClient(cited_source_id=ids_b[0])
    runtime._learning_runtime = None

    result = runtime.launch_learning(SimpleNamespace(run_id=None, run_ids=[run_b, run_a]))

    assert sorted(result["sourceRunIds"]) == sorted([run_a, run_b])
    assert result["candidate"]["supportingEvidenceIds"] == [ids_b[0]]
    # The failed second run's evidence reached the model context despite the
    # primary run holding more than the per-kind window of evidence records.
    assert ids_b[0] in runtime.learning_model_client.seen_evidence
    assert all(ev_id in runtime.learning_model_client.seen_evidence for ev_id in ids_b)
    # Primary coverage is bounded per run, not erased by the second run.
    assert len([eid for eid in runtime.learning_model_client.seen_evidence if eid in ids_a]) <= 8
    # Anything outside the declared set remains fail-closed.
    store.append_evidence("ev-foreign", {"run_id": "run-foreign", "sequence": 1, "event_type": "tool_result", "content_hash": "x" * 64, "source_ref": "{}", "trust_class": "broker", "visibility": "learner", "redacted": 1})
    foreign = _ScriptedLearningClient.__new__(_ScriptedLearningClient)
    foreign.cited = "broker:ev-foreign"
    foreign.seen_evidence = []
    runtime.learning_model_client = foreign
    runtime._learning_runtime = None
    with pytest.raises(Exception):
        runtime.launch_learning(SimpleNamespace(run_id=None, run_ids=[run_a, run_b]))
