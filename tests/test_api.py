from fastapi.testclient import TestClient

from adaptive_agent.api import ControlPlane, create_app


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
