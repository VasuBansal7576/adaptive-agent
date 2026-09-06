import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import adaptive_agent.app as app_module
from adaptive_agent.evaluation import Arm, BudgetSpec, EvaluationProtocol
from adaptive_agent.production_evaluator import _lifecycle_stages
from adaptive_agent.prime_runtime import ChildPlannerBudget, SharedBudget


def test_evaluated_outcome_merge_preserves_canonical_metrics_and_custom_diagnostics(tmp_path):
    app = app_module.create_runtime_app(data_dir=tmp_path)
    runtime = app.state.durable_runtime
    api = TestClient(app, base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
    task = api.get("/environments/finance/tasks").json()[0]
    run_id = api.post("/runs", json={"goal": task["goal"], "environmentId": "finance", "idempotencyKey": "metadata-merge"}).json()["runId"]
    runtime.controller.record_model_response(run_id, {"responseId": "metadata-response"})

    runtime._record_evaluated_outcome(
        run_id,
        app_module.DurableOutcome(
            runId=run_id,
            passed=False,
            metadata={
                "status": "budget_exhausted",
                "reliable": True,
                "safetyViolations": 4,
                "fixtureResetPassed": False,
                "customEvaluator": "retained",
            },
        ),
    )
    row = runtime.controller.store.get_outcome_by_run_id(run_id)
    metadata = json.loads(row["metadata_json"])
    assert metadata["reliable"] is True
    assert metadata["safetyViolations"] == 4
    assert metadata["fixtureResetOk"] is False
    assert metadata["fixtureResetPassed"] is False
    assert metadata["status"] == "budget_exhausted"
    assert metadata["customEvaluator"] == "retained"


def test_evaluation_replay_reuses_terminal_accounting_and_strict_verifier(tmp_path, monkeypatch):
    class FakePrime:
        def __init__(self, config, broker):
            self.config = config
            self.broker = broker
            self.child_planner = None
            self._budget = SharedBudget(
                config.max_total_wall_seconds,
                config.max_total_artifact_bytes,
                config.max_artifact_count,
                config.child_runs,
                config.max_model_tokens,
            )

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
            time.sleep(1.1)
            return SimpleNamespace(status="ok", result=json.dumps(namespace.get("result")), stdout="", stderr="", error=None)

        def close(self, remove_workspace=True):
            return None

    monkeypatch.setattr(app_module, "PrimeRuntimeAdapter", FakePrime)

    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, *, goal, environment, messages, **kwargs):
            self.calls += 1
            capability = next(value for value in environment["capabilities"] if value.endswith(":finance.invoice.apply_payment"))
            request = {
                "type": "broker.call",
                "capabilityId": capability,
                "arguments": {"invoice_id": "INV-DEV-000", "payment_id": "PAY-DEV-000", "expected_version": 1},
            }
            action = {"action": "execute", "code": f"result = host_request({request!r})"} if self.calls == 1 else {"action": "finish", "answer": "done"}
            return {
                "provider": "openai-codex",
                "model": "openai-codex/gpt-5.6-luna",
                "responseId": f"terminal-accounting-{self.calls}",
                "text": json.dumps(action),
                "usage": {"inputTokens": 11000, "outputTokens": 100, "totalTokens": 11100, "economicCost": {"status": "measured", "microunits": 0}},
            }

    model = Model()
    app = app_module.create_runtime_app(data_dir=tmp_path, model_runner=model)
    runtime = app.state.durable_runtime
    runtime.establish_clean_experiment = lambda _protocol: {"clean": True, "actualDocker": True, "provenanceRef": "clean"}
    protocol = EvaluationProtocol(run_budget=BudgetSpec(wall_time_seconds=1), core_planner_hash=runtime.core_planner_hash, image_digest=runtime.image_digest)
    protocol.freeze(runtime.packages)
    active = runtime.controller.get_active_bundle()
    assert active is not None
    stages = _lifecycle_stages(runtime, protocol, 0)
    stages = (stages[0], replace(stages[1], cells=("finance-development-00",)), replace(stages[2], callback=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop after training"))), *stages[3:])
    job = runtime.build_evaluation_job(protocol, {Arm.B0: active})
    limits = {"attempts": 3, "inputTokens": 60000, "outputTokens": 60000, "toolCalls": 64, "wallMicros": 180000000, "costMicrounits": 200000}

    result = job.run_experiment("terminal-accounting", stages, limits=limits)
    assert result.status == "failed"
    assert "stop after training" in result.error
    assert model.calls == 1

    with runtime.controller.store.connect() as connection:
        ledger_before = dict(connection.execute("SELECT * FROM evaluation_lifecycle_budget").fetchone())
    with runtime.controller.store.connect() as connection:
        run_id = connection.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()["run_id"]
    run = runtime.controller.store.get_run(run_id)
    assert run is not None
    run_payload = json.loads(run["run_json"])
    first_ref = run_payload["finalAccountingRef"]
    first_accounting = runtime.controller.store.get_artifact(first_ref)
    assert first_accounting["toolCalls"] == 1
    assert first_accounting["durationSeconds"] >= 1
    outcome_before = runtime.controller.store.get_outcome_by_run_id(run_id)
    assert outcome_before is not None
    outcome_metadata = json.loads(outcome_before["metadata_json"])
    assert outcome_metadata["status"] == "budget_exhausted"
    assert outcome_metadata["reliable"] is False
    assert outcome_metadata["safetyViolations"] == 0
    assert outcome_metadata["fixtureResetOk"] is True
    tool_events_before = [row for row in runtime.controller.store.list_evidence(run_id) if row["event_type"] == "tool_result"]
    status_before = run["status"]

    fresh = app_module.create_runtime_app(data_dir=tmp_path, model_runner=model).state.durable_runtime
    task = next(task for task in fresh.packages["finance"].tasks_for_partition("development") if task.task_id == "finance-development-00")
    config = SimpleNamespace(arm=Arm.B0, seed=17, attempt=0, protocol=protocol.start_candidate_generation(), bundle_hash=active.content_hash)
    replayed = fresh.execute_evaluation_task(task, config, active)
    replayed_run = fresh.controller.store.get_run(run["run_id"])
    replayed_ref = json.loads(replayed_run["run_json"])["finalAccountingRef"]
    assert replayed_ref == first_ref
    assert replayed.accounting_ref == first_ref
    assert fresh.controller.store.get_artifact(first_ref) == first_accounting
    assert model.calls == 1
    assert replayed.cost_microunits == first_accounting["costMicrounits"]
    assert replayed.latency_seconds == first_accounting["durationSeconds"]
    assert fresh.controller.store.get_run(run["run_id"])["status"] == status_before
    assert fresh.controller.store.get_outcome_by_run_id(run_id) == outcome_before
    assert [row for row in fresh.controller.store.list_evidence(run_id) if row["event_type"] == "tool_result"] == tool_events_before

    from adaptive_agent.experiment_runtime import DefaultExperimentStageRunner

    model_row = next(row for row in fresh.controller.store.list_evidence(run_id) if row["event_type"] == "model_response")
    outcome_row = next(row for row in fresh.controller.store.list_evidence(run_id) if row["event_type"] == "trusted_outcome")
    task_row = fresh.controller.store.get_task(task.task_id)
    fresh.controller.store.register_task(task.task_id, task_row["environment_id"], task_row["version"], task_row["task_ref"], "validation", task_row["goal"])
    recovered = DefaultExperimentStageRunner(fresh, protocol).recover_evaluation_observations({
        "stage": "validation",
        "cellKey": "validation:0",
        "runIds": [run_id],
        "taskIds": [task.task_id],
        "evidenceRefs": [model_row["evidence_id"]],
        "outcomeRefs": [outcome_row["evidence_id"]],
    })
    assert len(recovered) == 1
    assert recovered[0].accounting_ref == first_ref
    assert recovered[0].latency_seconds >= 1

    with fresh.controller.store.connect() as connection:
        assert dict(connection.execute("SELECT * FROM evaluation_lifecycle_budget").fetchone()) == ledger_before

    verifier = fresh.controller.evaluator_adapters[2]
    assert verifier.verify(replayed, config.protocol, fresh.packages["finance"])
    assert verifier.verify(recovered[0], config.protocol, fresh.packages["finance"])

    tampered = dict(first_accounting)
    tampered["durationSeconds"] = first_accounting["durationSeconds"] + 1
    tampered_ref = fresh.controller.store.put_artifact(tampered).sha256
    tampered_run = dict(replayed_run)
    tampered_payload = json.loads(tampered_run["run_json"])
    tampered_payload["finalAccountingRef"] = tampered_ref
    tampered_run["run_json"] = json.dumps(tampered_payload, sort_keys=True)
    fresh.controller.store.save_run(run["run_id"], {key: value for key, value in tampered_run.items() if key != "run_id"})
    assert not verifier.verify(replayed, config.protocol, fresh.packages["finance"])
    tampered_payload["finalAccountingRef"] = "missing-terminal-receipt"
    tampered_run["run_json"] = json.dumps(tampered_payload, sort_keys=True)
    fresh.controller.store.save_run(run["run_id"], {key: value for key, value in tampered_run.items() if key != "run_id"})
    assert not verifier.verify(replayed, config.protocol, fresh.packages["finance"])
    with pytest.raises(app_module.LearningRuntimeError, match="terminal accounting receipt is missing"):
        fresh.execute_evaluation_task(task, config, active)
