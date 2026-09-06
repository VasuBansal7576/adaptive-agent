import json

from fastapi.testclient import TestClient

from adaptive_agent.app import create_runtime_app
import adaptive_agent.app as app_module


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


def test_failed_stage_reconciles_all_responses_once_and_missing_receipt_stays_unverified(tmp_path):
    app = create_runtime_app(data_dir=tmp_path)
    api = TestClient(app, base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
    task = api.get("/environments/finance/tasks").json()[0]
    run = api.post("/runs", json={"goal": task["goal"], "environmentId": "finance", "idempotencyKey": "failed-stage"}).json()
    runtime = app.state.durable_runtime
    package = runtime.packages["finance"]
    bundle_hash = runtime.controller.get_active_bundle().content_hash
    runtime._record_model_response(run["runId"], package, {
        "provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": "batch",
        "arm": "B0", "seed": 17, "bundleHash": bundle_hash,
        "usage": {"inputTokens": 30, "outputTokens": 6, "totalTokens": 36},
        "receipts": [
            {"responseId": "r1", "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12}, "costMicrounits": 3, "durationSeconds": 1.0},
            {"responseId": "r2", "usage": {"inputTokens": 20, "outputTokens": 4, "totalTokens": 24}, "costMicrounits": 5, "durationSeconds": 2.0},
        ],
    })

    class FailedStage:
        def act(self, _ctx):
            raise RuntimeError("token budget exhausted after broker effect")

    runtime._claim_run(run["runId"])
    runtime._execute_run(
        run["runId"], "finance", object(), FailedStage(),
        lambda: app_module.DurableOutcome(
            runId=run["runId"], passed=False,
            metadata={"status": "budget_exhausted", "arm": "B0", "seed": 17, "bundleHash": bundle_hash},
        ),
    )
    evidence = runtime.controller.store.list_evidence(run["runId"])
    assert [row for row in evidence if row["event_type"] == "trusted_outcome"]
    assert runtime.controller.store.get_outcome_by_run_id(run["runId"])["passed"] == 0

    restarted = create_runtime_app(data_dir=tmp_path).state.durable_runtime
    restarted._record_model_response(run["runId"], restarted.packages["finance"], {
        "provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": "batch",
        "arm": "B0", "seed": 17, "bundleHash": bundle_hash,
        "usage": {"inputTokens": 20, "outputTokens": 4, "totalTokens": 24},
        "receipts": [{"responseId": "r2", "usage": {"inputTokens": 20, "outputTokens": 4, "totalTokens": 24}, "costMicrounits": 5, "durationSeconds": 2.0}],
    })
    rows = [row for row in restarted.controller.store.list_evidence(run["runId"]) if row["event_type"] == "model_response"]
    payload = restarted.controller.store.get_artifact(json.loads(rows[-1]["source_ref"])["sha256"])
    accounting = restarted.controller.store.get_artifact(payload["accountingRef"]["sha256"])
    assert [item["responseId"] for item in accounting["receipts"]] == ["r1", "r2"]
    assert accounting["aggregateUsage"]["totalTokens"] == 36

    missing = api.post("/runs", json={"goal": task["goal"], "environmentId": "finance", "idempotencyKey": "missing-receipt"}).json()
    runtime._claim_run(missing["runId"])
    runtime._execute_run(missing["runId"], "finance", object(), FailedStage(), lambda: app_module.DurableOutcome(runId=missing["runId"], passed=False, metadata={"status": "planner_failed"}))
    assert not [row for row in runtime.controller.store.list_evidence(missing["runId"]) if row["event_type"] == "trusted_outcome"]
