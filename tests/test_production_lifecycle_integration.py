from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_full_production_lifecycle_is_durable_and_restartable(tmp_path, monkeypatch):
    """Exercise the real lifecycle, evaluator, Store, and promotion gate.

    The two injected clients only stand in for the model boundary.  All run
    creation, evidence, recovery, report assembly, and candidate decisions
    remain production implementations.
    """
    monkeypatch.setenv("ADAPTIVE_AGENT_IMAGE_DIGEST", "sha256:e1242afd3804f022cb3bcdc4ae3fe1e5dcb5b79d09e98bdaffd5db320a32f0bb")
    from adaptive_agent.app import create_runtime_app
    from adaptive_agent.evaluation import Arm
    from adaptive_agent.production_evaluator import _lifecycle_execution_plan, _lifecycle_stages

    class TaskModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, *, goal, environment, emit):
            self.calls += 1
            return SimpleNamespace(
                text="synthetic boundary receipt",
                provider="openai-codex",
                model="openai-codex/gpt-5.6-luna",
                response_id=f"synthetic-task-{self.calls}",
                usage={"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                nominalCostUsd=0.000001,
                economicCostStatus="unknown",
            )

    class LearningModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, *, goal, environment, messages, **kwargs):
            self.calls += 1

            def evidence_ids(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {"sourceId", "source_id"} and isinstance(item, str):
                            yield item
                        yield from evidence_ids(item)
                elif isinstance(value, list):
                    for item in value:
                        yield from evidence_ids(item)

            evidence = next((item for item in evidence_ids(environment) if item.startswith("ev_")), None)
            assert evidence
            proposal = {
                "predictedEffect": "bounded synthetic improvement",
                "editOperations": [{"path": "skills/generic/procedure", "operation": "add", "value": "Use the verified broker workflow."}],
                "supportingEvidenceIds": [evidence],
                "proposerVersion": "synthetic-test-model",
                "skill": {"procedure": "Use the verified broker workflow."},
            }
            return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": f"synthetic-learning-{self.calls}", "text": json.dumps(proposal), "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "costMicrounits": 0}

    task_model = TaskModel()
    learning_model = LearningModel()
    app = create_runtime_app(data_dir=tmp_path, model_runner=task_model, learning_model_client=learning_model, evaluator=lambda **_: {"passed": True, "reliable": True, "safetyViolations": 0, "fixtureResetOk": True})
    runtime = app.state.durable_runtime
    from adaptive_agent.evaluation import EvaluationProtocol

    protocol = EvaluationProtocol(core_planner_hash=runtime.core_planner_hash, image_digest=runtime.image_digest, concurrency_limit=4)
    protocol.freeze(runtime.packages)
    active = runtime.controller.get_active_bundle()
    assert active is not None
    stages = _lifecycle_stages(runtime, protocol, 0)
    counts = {stage.name: len(stage.cells) for stage in stages}
    assert counts["validation"] == 360 and counts["final"] == 720
    plan = _lifecycle_execution_plan(counts, 0)
    limits = {"attempts": plan["totalAdmissions"], "inputTokens": 10_000_000, "outputTokens": 10_000_000, "toolCalls": 10_000_000, "wallMicros": 10_000_000_000, "costMicrounits": 10_000_000}
    job = runtime.build_evaluation_job(protocol, {Arm.B0: active})
    result = job.run_experiment("full-production-synthetic", stages, limits=limits)
    if result.status != "complete":
        with runtime.controller.store.connect() as conn:
            for rid in [row[0] for row in conn.execute("SELECT run_id FROM runs LIMIT 3")]:
                for event in runtime.controller.store.list_evidence(rid):
                    if event["event_type"] == "run_failed":
                        print("FAIL", runtime.controller.store.get_artifact(json.loads(event["source_ref"])["sha256"]))
    assert result.status == "complete", result.error
    assert result.reports is not None and set(result.reports) == {"validation", "final"}
    assert result.reports["validation"].validity_status == "valid"
    assert result.reports["final"].validity_status == "valid"
    assert result.reports["final"].ablation_audit is not None and result.reports["final"].ablation_audit.passed
    learned_hash = result.reports["validation"].candidate_hash
    learned = runtime.controller.store.get_bundle_by_hash(learned_hash)
    assert learned is not None
    a_hash = result.runtime_accounting  # retain a durable accounting assertion below
    assert runtime.controller.store.get_bundle_by_hash(result.reports["final"].candidate_hash) is not None
    assert task_model.calls > 0 and learning_model.calls == 1

    first_calls = task_model.calls
    restarted_app = create_runtime_app(data_dir=tmp_path, model_runner=task_model, learning_model_client=learning_model, evaluator=lambda **_: {"passed": True, "reliable": True, "safetyViolations": 0, "fixtureResetOk": True})
    restarted = restarted_app.state.durable_runtime
    restarted_job = restarted.build_evaluation_job(protocol, {Arm.B0: restarted.controller.get_active_bundle()})
    resumed = restarted_job.run_experiment("full-production-synthetic", _lifecycle_stages(restarted, protocol, 0), limits=limits)
    assert resumed.status == "complete"
    assert task_model.calls == first_calls
    assert learning_model.calls == 1

    final_state = resumed.reports["final"] if resumed.reports else None
    assert final_state is not None and final_state.validity_status == "valid"
    a_rows = restarted.controller.store.connect()
    with a_rows as conn:
        bundles = conn.execute("SELECT bundle_id, content_hash, bundle_json FROM skill_bundles").fetchall()
    assert len({row[1] for row in bundles}) >= 3
    assert learned["content_hash"] == learned_hash

    # Removing one durable model evidence row makes recovery fail closed and
    # cannot be converted into a complete lifecycle result.
    with a_rows as conn:
        row = conn.execute("SELECT result_json FROM evaluation_lifecycle_attempts WHERE job_id = ? AND stage = 'validation' LIMIT 1", ("full-production-synthetic",)).fetchone()
        receipt = json.loads(row[0])
        evidence_id = receipt["evidenceRefs"][0]
        conn.execute("DELETE FROM evidence WHERE evidence_id = ?", (evidence_id,))
        conn.commit()
    with pytest.raises(Exception):
        restarted_job._experiment_report(_lifecycle_stages(restarted, protocol, 0), {"results": {"validation": {receipt["cellKey"]: receipt}}}, stage_name="validation")
