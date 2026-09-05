from fastapi.testclient import TestClient
from threading import Event, Thread

from adaptive_agent.api import ControlPlane, create_app, make_authenticated_model_runner


def manifest():
    return {
        "environmentId": "neutral",
        "version": "1",
        "toolSchemas": [{"name": "counter.read", "version": "1", "inputSchema": {"type": "object"}, "outputSchema": {"type": "object"}, "effect": "read"}],
        "policyRef": {"id": "policy", "version": "1", "sha256": "p"},
        "evaluatorRef": {"id": "evaluator", "version": "1", "sha256": "e"},
        "resetRef": {"id": "reset", "version": "1", "sha256": "r"},
    }


class Invocation:
    text = "done"
    provider = "openai-codex"
    model = "openai-codex/gpt-5.6-luna"
    response_id = "resp-test-1"
    usage = {"inputTokens": 4, "outputTokens": 1}


def model_runner(**kwargs):
    kwargs["emit"]("tool", "counter.read completed")
    return Invocation()


def evaluator(**kwargs):
    return {"passed": kwargs["model_output"] == "done", "score": 1.0}


def client():
    plane = ControlPlane(model_runner=model_runner, evaluator=evaluator)
    api = TestClient(create_app(plane))
    assert api.post("/environments/register", json=manifest()).status_code == 201
    return api


def test_register_create_and_live_lifecycle():
    api = client()
    response = api.post("/runs", json={"goal": "read the counter", "environmentId": "neutral", "idempotencyKey": "run-1"})
    assert response.status_code == 201
    run = response.json()
    assert run["status"] == "queued"
    assert api.post(f"/runs/{run['runId']}/launch").status_code == 202
    events = api.get(f"/runs/{run['runId']}/events").text
    assert "Authenticated model response received" in events
    assert "Run succeeded" in events


def test_idempotency_replays_and_conflicts():
    api = client()
    body = {"goal": "read the counter", "environmentId": "neutral", "idempotencyKey": "same"}
    first = api.post("/runs", json=body)
    replay = api.post("/runs", json=body)
    assert first.status_code == replay.status_code == 201
    assert first.json()["runId"] == replay.json()["runId"]
    conflict = api.post("/runs", json={**body, "goal": "write the counter"})
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_unknown_manifest_fields_are_rejected():
    api = client()
    invalid = {**manifest(), "privileged": True}
    assert api.post("/environments/register", json=invalid).status_code == 422
    assert api.post("/environments/validate", content=b"not-json", headers={"content-type": "application/json"}).status_code == 422


def test_unverified_model_provenance_fails_closed():
    class Unverified:
        text = "done"
        provider = "simulation"
        model = "fixture"
        response_id = "fixture-response"
        usage = {"outputTokens": 1}

    plane = ControlPlane(model_runner=lambda **_: Unverified(), evaluator=evaluator)
    api = TestClient(create_app(plane))
    assert api.post("/environments/register", json=manifest()).status_code == 201
    run = api.post("/runs", json={"goal": "read", "environmentId": "neutral", "idempotencyKey": "unverified"}).json()
    api.post(f"/runs/{run['runId']}/launch")
    assert api.get(f"/runs/{run['runId']}").json()["status"] == "failed"


def test_cancelled_run_cannot_be_reopened_by_late_model_result():
    started, release = Event(), Event()

    def slow_model(**kwargs):
        started.set()
        release.wait(timeout=2)
        return Invocation()

    plane = ControlPlane(model_runner=slow_model, evaluator=evaluator)
    api = TestClient(create_app(plane))
    api.post("/environments/register", json=manifest())
    run = api.post("/runs", json={"goal": "read", "environmentId": "neutral", "idempotencyKey": "cancel"}).json()
    thread = Thread(target=plane.launch, args=(run["runId"],))
    thread.start()
    assert started.wait(timeout=1)
    assert api.post(f"/runs/{run['runId']}/cancel").status_code == 200
    release.set()
    thread.join(timeout=2)
    assert api.get(f"/runs/{run['runId']}").json()["status"] == "cancelled"


def test_prime_bridge_records_parent_owned_model_observation():
    calls = []

    class Client:
        def invoke(self, **kwargs):
            return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": "resp-1", "text": "answer", "usage": {"outputTokens": 2}}

    class Sink:
        def record_model_observation(self, evidence, *, trusted_parent=False):
            calls.append((evidence, trusted_parent))

    runner = make_authenticated_model_runner(Client(), Sink())
    result = runner(goal="goal", environment={}, emit=lambda *_: None)
    assert result.response_id == "resp-1"
    assert calls[0][1] is True
