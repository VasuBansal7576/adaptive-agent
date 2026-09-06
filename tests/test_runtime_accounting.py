import json

from fastapi.testclient import TestClient

from adaptive_agent.app import create_runtime_app


def test_model_receipts_rehydrate_without_double_charging_after_restart(tmp_path):
    app = create_runtime_app(data_dir=tmp_path)
    api = TestClient(app, base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
    task = api.get("/environments/finance/tasks").json()[0]
    run = api.post(
        "/runs",
        json={"goal": task["goal"], "environmentId": "finance", "idempotencyKey": "accounting-restart"},
    ).json()
    runtime = app.state.durable_runtime
    package = runtime.packages["finance"]
    bundle_hash = runtime.controller.get_active_bundle().content_hash
    base = {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "arm": "B0", "seed": 17, "bundleHash": bundle_hash}
    runtime._record_model_response(run["runId"], package, {**base, "responseId": "r1", "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12}, "costMicrounits": 3})
    runtime._record_model_response(run["runId"], package, {**base, "responseId": "r2", "usage": {"inputTokens": 20, "outputTokens": 4, "totalTokens": 24}, "costMicrounits": 5})

    restarted = create_runtime_app(data_dir=tmp_path).state.durable_runtime
    restarted._record_model_response(run["runId"], restarted.packages["finance"], {**base, "responseId": "r3", "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6}, "costMicrounits": 2})
    rows = [row for row in restarted.controller.store.list_evidence(run["runId"]) if row["event_type"] == "model_response"]
    payload = restarted.controller.store.get_artifact(json.loads(rows[-1]["source_ref"])["sha256"])
    accounting = restarted.controller.store.get_artifact(payload["accountingRef"]["sha256"])
    assert [receipt["responseId"] for receipt in accounting["receipts"]] == ["r1", "r2", "r3"]
    assert accounting["aggregateUsage"] == {"inputTokens": 35, "outputTokens": 7, "totalTokens": 42}
    assert accounting["costMicrounits"] == 10
