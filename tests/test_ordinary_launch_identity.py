import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import adaptive_agent.app as app_module
from adaptive_agent.app import LearningRuntimeError, create_runtime_app
from adaptive_agent.prime_runtime import ChildPlannerBudget, SharedBudget


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, **_kwargs):
        self.calls += 1
        return self.responses.pop(0)


class ForbiddenClient:
    def __init__(self):
        self.calls = 0

    def invoke(self, **_kwargs):
        self.calls += 1
        raise AssertionError("replayed terminal run dispatched a model request")


class FakePrime:
    def __init__(self, config, broker):
        self.config = config
        self.broker = broker
        self.child_planner = None
        # Let LunaPlanner enforce the test's token cap so this exercises its
        # real budget_exhausted result instead of the adapter ledger boundary.
        self._budget = SharedBudget(config.max_total_wall_seconds, config.max_total_artifact_bytes, config.max_artifact_count, config.child_runs, max(config.max_model_tokens, 100))

    @property
    def planner_budget(self):
        return ChildPlannerBudget(self._budget)

    def record_model_observation(self, evidence, *, trusted_parent=False):
        return evidence

    def execute(self, code, *, timeout=None, cancel=None):
        namespace = {}
        exec(code, {"__builtins__": {}}, namespace)
        return SimpleNamespace(status="ok", result=json.dumps(namespace.get("result")), stdout="", stderr="", error=None)

    def close(self, remove_workspace=True):
        return None


def _create_run(app, key, *, model_tokens=100):
    api = TestClient(app, base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
    task = app.state.durable_runtime.packages["finance"].tasks_for_partition("development")[0]
    response = api.post("/runs", json={"goal": task.goal, "environmentId": "finance", "idempotencyKey": key, "budget": {"modelTokens": model_tokens, "toolCalls": 1, "childRuns": 0, "wallTimeSeconds": 30, "costMicrounits": 1000, "currency": "USD"}})
    assert response.status_code == 201, response.text
    run_id = response.json()["runId"]
    runtime = app.state.durable_runtime
    return runtime, run_id, runtime.controller.store.get_run(run_id)["bundle_hash"]


def _snapshot(runtime, run_id):
    store = runtime.controller.store
    return (store.get_run(run_id), store.list_evidence(run_id), store.get_outcome_by_run_id(run_id))


def _assert_identity(runtime, run_id, bundle_hash, expected_status):
    store = runtime.controller.store
    row = store.get_run(run_id)
    assert row is not None
    payload = json.loads(row["run_json"])
    assert payload["arm"] == "B0"
    assert payload["seed"] == 0
    assert payload["bundleHash"] == bundle_hash
    assert payload["finalAccountingRef"]
    assert row["status"] == expected_status
    model_row = next(item for item in store.list_evidence(run_id) if item["event_type"] == "model_response")
    model = store.get_artifact(json.loads(model_row["source_ref"])["sha256"])
    accounting = store.get_artifact(payload["finalAccountingRef"])
    assert model["arm"] == accounting["arm"] == "B0"
    assert model["seed"] == accounting["seed"] == 0
    assert model["bundleHash"] == accounting["bundleHash"] == bundle_hash
    metadata = json.loads(store.get_outcome_by_run_id(run_id)["metadata_json"])
    assert metadata["arm"] == "B0"
    assert metadata["seed"] == 0
    assert metadata["bundleHash"] == bundle_hash


def test_ordinary_scripted_success_reopens_without_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "PrimeRuntimeAdapter", FakePrime)
    client = ScriptedClient([{"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": "ordinary-success", "text": '{"action":"finish","answer":"done"}', "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}])
    app = create_runtime_app(data_dir=tmp_path, evaluator=lambda **_: {"passed": True}, model_runner=None)
    runtime, run_id, bundle_hash = _create_run(app, "ordinary-success")
    runtime.launch(run_id, model_client_override=client)
    _assert_identity(runtime, run_id, bundle_hash, "succeeded")
    before = _snapshot(runtime, run_id)
    reopened = create_runtime_app(data_dir=tmp_path)
    forbidden = ForbiddenClient()
    reopened.state.durable_runtime.launch(run_id, model_client_override=forbidden)
    assert forbidden.calls == 0
    assert _snapshot(reopened.state.durable_runtime, run_id) == before


def test_ordinary_scripted_planner_token_exhaustion_reopens_without_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "PrimeRuntimeAdapter", FakePrime)
    client = ScriptedClient([{"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": "ordinary-budget", "text": '{"action":"finish","answer":"too late"}', "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}])
    app = create_runtime_app(data_dir=tmp_path, model_runner=None)
    runtime, run_id, bundle_hash = _create_run(app, "ordinary-budget", model_tokens=1)
    runtime.launch(run_id, model_client_override=client)
    _assert_identity(runtime, run_id, bundle_hash, "failed")
    metadata = json.loads(runtime.controller.store.get_outcome_by_run_id(run_id)["metadata_json"])
    assert metadata["status"] == "budget_exhausted"
    before = _snapshot(runtime, run_id)
    reopened = create_runtime_app(data_dir=tmp_path)
    forbidden = ForbiddenClient()
    reopened.state.durable_runtime.launch(run_id, model_client_override=forbidden)
    assert forbidden.calls == 0
    assert _snapshot(reopened.state.durable_runtime, run_id) == before


@pytest.mark.parametrize("kwargs", [{"arm": "L"}, {"seed": 1}])
def test_terminal_replay_rejects_conflicting_identity_without_mutation(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(app_module, "PrimeRuntimeAdapter", FakePrime)
    client = ScriptedClient([{"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": "ordinary-conflict", "text": '{"action":"finish","answer":"done"}', "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}])
    app = create_runtime_app(data_dir=tmp_path, evaluator=lambda **_: {"passed": True}, model_runner=None)
    runtime, run_id, _ = _create_run(app, "ordinary-conflict")
    runtime.launch(run_id, model_client_override=client)
    before = _snapshot(runtime, run_id)
    with pytest.raises(LearningRuntimeError, match="conflicts with persisted"):
        runtime.launch(run_id, model_client_override=ForbiddenClient(), **kwargs)
    assert _snapshot(runtime, run_id) == before


def test_malformed_pinned_bundle_hash_is_rejected_before_dispatch(tmp_path):
    app = create_runtime_app(data_dir=tmp_path, model_runner=None)
    runtime, run_id, bundle_hash = _create_run(app, "ordinary-malformed-bundle")
    store = runtime.controller.store
    row = store.get_run(run_id)
    run_record = runtime.controller.get_run(run_id)
    original = store.get_artifact(run_record.skill_bundle_ref)
    tampered = dict(original)
    tampered["contentHash"] = "not-the-payload-hash"
    tampered_ref = store.put_artifact(tampered)
    run_payload = json.loads(row["run_json"])
    run_payload["skillBundleRef"] = tampered_ref.model_dump(mode="json", by_alias=True)
    row["run_json"] = json.dumps(run_payload, sort_keys=True)
    store.save_run(run_id, {key: value for key, value in row.items() if key != "run_id"})
    with pytest.raises(LearningRuntimeError, match="does not match its persisted content"):
        runtime.launch(run_id, model_client_override=ForbiddenClient())
    assert store.get_run(run_id)["status"] == "queued"
    assert bundle_hash == store.get_run(run_id)["bundle_hash"]
