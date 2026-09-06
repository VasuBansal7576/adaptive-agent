from __future__ import annotations

import json
import re
import threading
from types import SimpleNamespace

import pytest


def test_full_production_lifecycle_is_durable_and_restartable(tmp_path, monkeypatch):
    """Exercise the real lifecycle, evaluator, Store, and promotion gate.

    The two injected clients only stand in for the model boundary.  All run
    creation, evidence, recovery, report assembly, and candidate decisions
    remain production implementations.
    """
    monkeypatch.setenv("ADAPTIVE_AGENT_IMAGE_DIGEST", "sha256:e1242afd3804f022cb3bcdc4ae3fe1e5dcb5b79d09e98bdaffd5db320a32f0bb")
    import adaptive_agent.app as app_module

    class ControlledClock:
        def __init__(self):
            self.local = threading.local()

        def monotonic(self):
            value = getattr(self.local, "value", 0.0) + 1.0
            self.local.value = value
            return value

    # Keep this integration test deterministic while preserving the real
    # runtime's monotonic clock and its persisted duration accounting.
    monkeypatch.setattr(app_module, "time", SimpleNamespace(monotonic=ControlledClock().monotonic))
    create_runtime_app = app_module.create_runtime_app
    from adaptive_agent.evaluation import Arm
    from adaptive_agent.production_evaluator import _lifecycle_execution_plan, _lifecycle_stages
    from adaptive_agent.prime_runtime import ChildPlannerBudget, SharedBudget

    class TaskModel:
        def __init__(self):
            self.turn = 0
            self.turns_by_run = {}

        def invoke(self, *, goal, environment, messages, **kwargs):
            self.turn += 1
            run_key = environment["capabilities"][0]
            turn = self.turns_by_run.get(run_key, 0) + 1
            self.turns_by_run[run_key] = turn
            identifiers = re.findall(r"[A-Z]{3}-[A-Z]+-\d{3}", goal)
            schemas = {schema["name"]: schema for schema in environment["toolSchemas"]}
            goal_words = set(re.findall(r"[a-z]+", goal.lower()))
            tool = next(name for name, schema in schemas.items() if schema.get("effect") == "write" and set(name.lower().split(".")).intersection(goal_words))
            properties = schemas[tool].get("inputSchema", {}).get("properties", {})
            arguments = {}
            identifier_index = 0
            for key in properties:
                if key.endswith("_id"):
                    arguments[key] = identifiers[min(identifier_index, len(identifiers) - 1)]
                    identifier_index += 1
                elif key == "expected_version":
                    arguments[key] = 1
                elif key == "resolution":
                    arguments[key] = "customer-approved"
                elif key == "reason":
                    arguments[key] = "enhanced-review"
                elif key == "status":
                    arguments[key] = "resolved" if "ticket" in goal else "closed"
            capability_id = next(capability for capability in environment["capabilities"] if capability.endswith(f":{tool}"))
            request = {"type": "broker.call", "capabilityId": capability_id, "arguments": arguments}
            action = (
                json.dumps({"action": "execute", "code": f"result = host_request({json.dumps(request, separators=(',', ':'))})"}, separators=(",", ":"))
                if turn == 1 else '{"action":"finish","answer":"applied"}'
            )
            return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": f"synthetic-task-{self.turn}", "text": action, "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2, "economicCost": {"status": "measured", "microunits": 0}}}

    class FakePrime:
        def __init__(self, config, broker):
            self.config = config
            self.broker = broker
            self.child_planner = None
            self._budget = SharedBudget(config.max_total_wall_seconds, config.max_total_artifact_bytes, config.max_artifact_count, config.child_runs, config.max_model_tokens)

        @property
        def planner_budget(self):
            return ChildPlannerBudget(self._budget)

        def record_model_observation(self, evidence, *, trusted_parent=False):
            return evidence

        def execute(self, code, *, timeout=None, cancel=None):
            def host_request(payload):
                return self.broker.call(payload["capabilityId"], payload.get("arguments", {}))

            namespace = {"host_request": host_request}
            exec(code, {"__builtins__": {}}, namespace)
            return SimpleNamespace(status="ok", result=json.dumps(namespace.get("result")), stdout="", stderr="", error=None)

        def close(self, remove_workspace=True):
            return None

    monkeypatch.setattr(app_module, "PrimeRuntimeAdapter", FakePrime)

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

            evidence = next((item for item in evidence_ids(environment) if item.startswith("broker:")), None)
            assert evidence
            proposal = {
                "predictedEffect": "bounded synthetic improvement",
                "editOperations": [{"path": "skills/portable-procedure/procedure", "operation": "add", "value": "Use the verified workflow."}],
                "supportingEvidenceIds": [evidence],
                "proposerVersion": "synthetic-test-model",
                "skill": {"procedure": "Use the verified workflow."},
            }
            return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": f"synthetic-learning-{self.calls}", "text": json.dumps(proposal), "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "costMicrounits": 0, "economicCostStatus": "measured"}

    task_model = TaskModel()
    learning_model = LearningModel()

    def trusted_evaluator(*, model_responses, **_kwargs):
        payload = model_responses[-1]["payload"]
        development = "-development-" in payload["taskId"]
        passed = development or payload["arm"] == "L"
        return {"passed": passed, "reliable": True, "safetyViolations": 0, "fixtureResetOk": True}

    app = create_runtime_app(data_dir=tmp_path, model_runner=task_model, learning_model_client=learning_model, evaluator=trusted_evaluator)
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
    limits = {"attempts": max(plan["totalAdmissions"], 1_000_000), "inputTokens": 1_000_000_000_000, "outputTokens": 1_000_000_000_000, "toolCalls": 1_000_000_000_000, "wallMicros": 1_000_000_000_000, "costMicrounits": 1_000_000_000_000}
    job = runtime.build_evaluation_job(protocol, {Arm.B0: active})
    result = job.run_experiment("full-production-synthetic", stages, limits=limits)
    assert result.status == "complete", result.error
    assert result.reports is not None and set(result.reports) == {"validation", "final"}
    assert result.reports["validation"].validity_status == "valid"
    assert result.reports["final"].validity_status == "valid"
    assert result.reports["validation"].arm_summaries["B0"].accuracy == 0.0
    assert result.reports["validation"].arm_summaries["L"].accuracy == 1.0
    assert result.reports["final"].arm_summaries["B0"].accuracy == 0.0
    assert result.reports["final"].arm_summaries["L"].accuracy == 1.0
    assert result.reports["final"].arm_summaries["A"].accuracy == 0.0
    assert result.reports["final"].ablation_audit is not None and result.reports["final"].ablation_audit.passed
    learned_hash = result.reports["validation"].candidate_hash
    learned = runtime.controller.store.get_bundle_by_hash(learned_hash)
    assert learned is not None
    assert runtime.controller.store.get_bundle_by_hash(result.reports["final"].candidate_hash) is not None
    assert runtime.controller.get_active_bundle().content_hash == learned_hash
    ablation_hash = runtime._evaluation_arm_bundles["A"]
    assert len({active.content_hash, learned_hash, ablation_hash}) == 3
    promotions_before_restart = runtime.controller.store.list_promotions()
    assert len(promotions_before_restart) == 1 and promotions_before_restart[0]["decision"] == "promoted"
    assert task_model.turn > 0 and learning_model.calls == 7

    with runtime.controller.store.connect() as conn:
        validation_receipts = conn.execute("SELECT COUNT(*) FROM evaluation_lifecycle_attempts WHERE job_id = ? AND stage = 'validation' AND status = 'complete'", ("full-production-synthetic",)).fetchone()[0]
        final_receipts = conn.execute("SELECT COUNT(*) FROM evaluation_lifecycle_attempts WHERE job_id = ? AND stage = 'final' AND status = 'complete'", ("full-production-synthetic",)).fetchone()[0]
        transfer_and_adaptation = conn.execute("SELECT stage, result_json FROM evaluation_lifecycle_attempts WHERE job_id = ? AND stage IN ('transfer', 'adaptation') AND status = 'complete' ORDER BY stage, cell_key", ("full-production-synthetic",)).fetchall()
    assert (validation_receipts, final_receipts) == (360, 720)
    assert len(transfer_and_adaptation) == 6
    for row in transfer_and_adaptation:
        stage_receipt = json.loads(row["result_json"])
        learning_receipt = stage_receipt["learningReceipt"]
        assert learning_receipt["modelObservationRefs"]
        assert learning_receipt["sourceRunIds"]
        assert learning_receipt.get("reusedCandidate") is not True

    first_calls = task_model.turn
    restarted_app = create_runtime_app(data_dir=tmp_path, model_runner=task_model, learning_model_client=learning_model, evaluator=trusted_evaluator)
    restarted = restarted_app.state.durable_runtime
    restarted_job = restarted.build_evaluation_job(protocol, {Arm.B0: active})
    resumed = restarted_job.run_experiment("full-production-synthetic", _lifecycle_stages(restarted, protocol, 0), limits=limits)
    assert resumed.status == "complete", resumed.error
    assert task_model.turn == first_calls
    assert learning_model.calls == 7
    assert restarted.controller.store.list_promotions() == promotions_before_restart

    final_state = resumed.reports["final"] if resumed.reports else None
    assert final_state is not None and final_state.validity_status == "valid"
    with restarted.controller.store.connect() as conn:
        bundles = conn.execute("SELECT bundle_id, content_hash, bundle_json FROM skill_bundles").fetchall()
    assert len({row[1] for row in bundles}) >= 3
    assert learned["content_hash"] == learned_hash

    # Removing one durable model evidence row makes recovery fail closed and
    # cannot be converted into a complete lifecycle result.
    with restarted.controller.store.connect() as conn:
        row = conn.execute("SELECT result_json FROM evaluation_lifecycle_attempts WHERE job_id = ? AND stage = 'validation' LIMIT 1", ("full-production-synthetic",)).fetchone()
        receipt = json.loads(row[0])
        evidence_id = receipt["evidenceRefs"][0]
        conn.execute("DELETE FROM evidence WHERE evidence_id = ?", (evidence_id,))
        conn.commit()
    invalid_resume = restarted_job.run_experiment("full-production-synthetic", _lifecycle_stages(restarted, protocol, 0), limits=limits, context={"resume": True})
    assert invalid_resume.status == "failed"
    assert "evidence" in (invalid_resume.error or "").lower()
