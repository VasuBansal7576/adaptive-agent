"""Factory/wiring tests for LearningRuntime; real-provider smoke is separate."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
core_src = os.environ.get("ADAPTIVE_AGENT_CORE_SRC")
if core_src:
    import adaptive_agent

    adaptive_agent.__path__.append(str(Path(core_src) / "adaptive_agent"))
    adaptive_agent.__path__.append(str(ROOT / "src" / "adaptive_agent"))

try:
    from adaptive_agent.candidate import CandidateManager
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
    runtime = LearningRuntime.build(store=store, manager=manager, model_client=client, token_budget=1000, wall_seconds=20)
    result = runtime.propose_completed_run(RUN)
    assert result.authoritative_candidate["state"] == "validated"
    assert result.candidate_payload["supportingEvidenceIds"] == [evidence_id]
    restarted_store = type(store)(tmp_path)
    restarted_manager = CandidateManager(restarted_store)
    restarted = LearningRuntime.build(store=restarted_store, manager=restarted_manager, model_client=FakeClient(), token_budget=1000, wall_seconds=20)
    assert restarted.reload_candidate(result.authoritative_candidate["candidate_id"])["state"] == "validated"


def test_learning_model_observation_sink_never_writes_parent_model_response(tmp_path: Path):
    store, _, _ = _setup_store(tmp_path)
    sink = StoreModelObservationSink(store, RUN)

    sink.record_model_observation(
        {"provider": "test", "model": "learning-model", "responseId": "learning-resp", "usage": {"totalTokens": 2}},
        trusted_parent=True,
    )

    rows = store.list_evidence(RUN)
    observations = [row for row in rows if row.get("evidence_id") == "learning-model-learning-resp"]
    assert len(observations) == 1
    assert observations[0]["event_type"] == "learning_model_observation"
    assert observations[0]["visibility"] == "operator"
    assert not any(row.get("event_type") == "model_response" and row.get("evidence_id") == "learning-model-learning-resp" for row in rows)


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
    result = LearningRuntime.build(store=store, manager=manager, model_client=client, token_budget=1000, wall_seconds=20).propose_completed_run(RUN)
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
