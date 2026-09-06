import json

from fastapi.testclient import TestClient

from adaptive_agent.app import create_runtime_app


class Invocation:
    provider = "openai-codex"
    model = "openai-codex/gpt-5.6-luna"
    usage = {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}

    def __init__(self, response_id: str):
        self.response_id = response_id
        self.text = "done"


def _assert_reopenable_identity(runtime, run_id, bundle_hash, expected_status):
    store = runtime.controller.store
    run = store.get_run(run_id)
    assert run is not None
    payload = json.loads(run["run_json"])
    assert payload["arm"] == "B0"
    assert payload["seed"] == 0
    assert payload["bundleHash"] == bundle_hash
    assert payload["finalAccountingRef"]
    assert run["status"] == expected_status

    model_row = next(row for row in store.list_evidence(run_id) if row["event_type"] == "model_response")
    model = store.get_artifact(json.loads(model_row["source_ref"])["sha256"])
    accounting = store.get_artifact(payload["finalAccountingRef"])
    assert model["arm"] == accounting["arm"] == "B0"
    assert model["seed"] == accounting["seed"] == 0
    assert model["bundleHash"] == accounting["bundleHash"] == bundle_hash

    outcome = store.get_outcome_by_run_id(run_id)
    assert outcome is not None
    metadata = json.loads(outcome["metadata_json"])
    assert metadata["arm"] == "B0"
    assert metadata["seed"] == 0
    assert metadata["bundleHash"] == bundle_hash
    return payload["finalAccountingRef"], accounting


def test_ordinary_launch_binds_persisted_identity_and_terminal_accounting(tmp_path):
    app = create_runtime_app(
        data_dir=tmp_path,
        model_runner=lambda **_: Invocation("ordinary-success"),
        evaluator=lambda **_: {"passed": True, "score": 1.0},
    )
    runtime = app.state.durable_runtime
    task = runtime.packages["finance"].tasks_for_partition("development")[0]
    api = TestClient(app, base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
    run_id = api.post("/runs", json={"goal": task.goal, "environmentId": "finance", "idempotencyKey": "ordinary-success"}).json()["runId"]
    bundle_hash = runtime.controller.store.get_run(run_id)["bundle_hash"]

    runtime.launch(run_id)

    first_ref, first_accounting = _assert_reopenable_identity(runtime, run_id, bundle_hash, "succeeded")
    reopened = create_runtime_app(data_dir=tmp_path).state.durable_runtime
    second_ref, second_accounting = _assert_reopenable_identity(reopened, run_id, bundle_hash, "succeeded")
    assert second_ref == first_ref
    assert second_accounting == first_accounting


def test_ordinary_launch_preserves_identity_when_evaluator_reports_budget_exhausted(tmp_path):
    app = create_runtime_app(
        data_dir=tmp_path,
        model_runner=lambda **_: Invocation("ordinary-budget"),
        evaluator=lambda **_: {"passed": False, "status": "budget_exhausted"},
    )
    runtime = app.state.durable_runtime
    task = runtime.packages["finance"].tasks_for_partition("development")[0]
    api = TestClient(app, base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
    run_id = api.post("/runs", json={"goal": task.goal, "environmentId": "finance", "idempotencyKey": "ordinary-budget"}).json()["runId"]
    bundle_hash = runtime.controller.store.get_run(run_id)["bundle_hash"]

    runtime.launch(run_id)

    first_ref, first_accounting = _assert_reopenable_identity(runtime, run_id, bundle_hash, "failed")
    reopened = create_runtime_app(data_dir=tmp_path).state.durable_runtime
    second_ref, second_accounting = _assert_reopenable_identity(reopened, run_id, bundle_hash, "failed")
    assert second_ref == first_ref
    assert second_accounting == first_accounting
