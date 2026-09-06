from fastapi.testclient import TestClient
from threading import Event, Thread
import pytest

from adaptive_agent.api import CandidateProposalRequest, ControlPlane, EvaluationRequest, create_app, make_authenticated_model_runner
from adaptive_agent.app import create_runtime_app
from adaptive_agent.evaluation import Arm, EvaluationProtocol, EvaluationRunner, ModelProvenance, Partition, RunObservation, build_environment_packages
from adaptive_agent.planner import make_luna_model_runner
from adaptive_agent.constants import DEFAULT_MODEL_TOKENS


def manifest():
    return {
        "environmentId": "neutral",
        "version": "1",
        "docs": [{"id": "docs-neutral", "version": "1", "sha256": "d"}],
        "taskGoals": ["read the counter"],
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
    api = TestClient(create_app(plane), base_url="http://127.0.0.1")
    assert api.get("/session/bootstrap").json() == {"status": "ready", "transport": "live"}
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
    evidence = api.get(f"/runs/{run['runId']}/evidence").json()
    model_event = next(event for event in evidence if event.get("evidenceType") == "model_response")
    model_evidence = model_event["evidence"]
    assert model_evidence["runId"] == run["runId"]
    assert model_evidence["environmentId"] == "neutral"
    assert model_evidence["responseId"] == "resp-test-1"
    assert model_evidence["planner"]["corePlannerHash"] == run["skillBundleRef"]["sha256"]
    assert model_evidence["imageDigest"] == "image-unpinned"


def test_model_token_default_is_shared_and_practical():
    api = client()
    assert api.get("/run-options").json()["budgetDefaults"]["modelTokens"] == DEFAULT_MODEL_TOKENS == 20_000


def test_durable_launch_retry_does_not_reinvoke_model(tmp_path):
    calls = 0

    class RuntimeInvocation:
        text = "done"
        provider = "openai-codex"
        model = "openai-codex/gpt-5.6-luna"
        response_id = "durable-response"
        usage = {"inputTokens": 1, "outputTokens": 1}

    def runtime_runner(**_kwargs):
        nonlocal calls
        calls += 1
        return RuntimeInvocation()

    app = create_runtime_app(model_runner=runtime_runner, evaluator=lambda **_kwargs: {"passed": True}, data_dir=tmp_path)
    api = TestClient(app, base_url="http://127.0.0.1")
    assert api.get("/session/bootstrap").status_code == 200
    task = api.get("/environments/finance/tasks").json()[0]
    run = api.post("/runs", json={"goal": task["goal"], "environmentId": "finance", "idempotencyKey": "durable-retry"}).json()
    assert api.post(f"/runs/{run['runId']}/launch").status_code == 202
    retry = api.post(f"/runs/{run['runId']}/launch")
    assert retry.status_code == 202
    assert retry.json() == {"runId": run["runId"], "status": "succeeded"}
    assert calls == 1
    assert api.get(f"/runs/{run['runId']}").json()["status"] == "succeeded"


def test_durable_cancelled_run_cannot_be_reopened(tmp_path):
    calls = 0

    class RuntimeInvocation:
        text = "done"
        provider = "openai-codex"
        model = "openai-codex/gpt-5.6-luna"
        response_id = "cancelled-response"
        usage = {"inputTokens": 1, "outputTokens": 1}

    def runtime_runner(**_kwargs):
        nonlocal calls
        calls += 1
        return RuntimeInvocation()

    app = create_runtime_app(model_runner=runtime_runner, evaluator=lambda **_kwargs: {"passed": True}, data_dir=tmp_path)
    api = TestClient(app, base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
    task = api.get("/environments/finance/tasks").json()[0]
    run = api.post("/runs", json={"goal": task["goal"], "environmentId": "finance", "idempotencyKey": "durable-cancel"}).json()
    assert api.post(f"/runs/{run['runId']}/cancel").json()["status"] == "cancelled"
    retry = api.post(f"/runs/{run['runId']}/launch")
    assert retry.json() == {"runId": run["runId"], "status": "cancelled"}
    assert calls == 0


def test_durable_restart_does_not_reopen_claimed_run(tmp_path):
    first_app = create_runtime_app(data_dir=tmp_path)
    first_api = TestClient(first_app, base_url="http://127.0.0.1")
    first_api.get("/session/bootstrap")
    task = first_api.get("/environments/finance/tasks").json()[0]
    run = first_api.post("/runs", json={"goal": task["goal"], "environmentId": "finance", "idempotencyKey": "durable-restart"}).json()
    claimed, _ = first_app.state.controller.claim_run(run["runId"])
    assert claimed is True

    calls = 0

    class RuntimeInvocation:
        text = "done"
        provider = "openai-codex"
        model = "openai-codex/gpt-5.6-luna"
        response_id = "restart-response"
        usage = {"inputTokens": 1, "outputTokens": 1}

    def runtime_runner(**_kwargs):
        nonlocal calls
        calls += 1
        return RuntimeInvocation()

    restarted_app = create_runtime_app(model_runner=runtime_runner, evaluator=lambda **_kwargs: {"passed": True}, data_dir=tmp_path)
    restarted_api = TestClient(restarted_app, base_url="http://127.0.0.1")
    restarted_api.get("/session/bootstrap")
    retry = restarted_api.post(f"/runs/{run['runId']}/launch")
    assert retry.json() == {"runId": run["runId"], "status": "running"}
    assert calls == 0


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


def test_spec_task_ref_run_request_is_supported():
    api = client()
    response = api.post(
        "/runs",
        json={
            "taskRef": {"id": "task-1", "version": "1", "sha256": "t", "goal": "read the counter", "environmentId": "neutral"},
            "modelProfileRef": {"id": "model", "version": "1", "sha256": "m"},
            "budgetRef": {"id": "budget", "version": "1", "sha256": "b"},
            "idempotencyKey": "spec-run",
        },
    )
    assert response.status_code == 201
    assert response.json()["goal"] == "read the counter"


def test_unknown_manifest_fields_are_rejected():
    api = client()
    invalid = {**manifest(), "privileged": True}
    assert api.post("/environments/register", json=invalid).status_code == 422
    assert api.post("/environments/validate", content=b"not-json", headers={"content-type": "application/json"}).status_code == 422
    assert api.post("/environments/register", json={**manifest(), "executionModes": ["host"]}).status_code == 422


def test_registration_and_run_resolve_complete_trusted_references():
    api = client()
    unknown = {**manifest(), "environmentId": "unknown-ref-env", "evaluatorRef": {"id": "missing-evaluator", "version": "1", "sha256": "e"}}
    assert api.post("/environments/register", json=unknown).status_code == 422
    run = api.post("/runs", json={
        "goal": "read the counter", "environmentId": "neutral", "idempotencyKey": "missing-profile",
        "modelProfileRef": {"id": "model", "version": "1"},
    })
    assert run.status_code == 422
    valid = api.post("/runs", json={"goal": "read the counter", "environmentId": "neutral", "idempotencyKey": "full-profile"})
    assert valid.status_code == 201
    stored = api.get(f"/runs/{valid.json()['runId']}").json()
    assert set(stored["modelProfileRef"]) == {"id", "version", "sha256"}
    assert set(stored["budgetRef"]) == {"id", "version", "sha256"}


def test_unverified_model_provenance_fails_closed():
    class Unverified:
        text = "done"
        provider = "simulation"
        model = "fixture"
        response_id = "fixture-response"
        usage = {"outputTokens": 1}

    plane = ControlPlane(model_runner=lambda **_: Unverified(), evaluator=evaluator)
    api = TestClient(create_app(plane), base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
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
    api = TestClient(create_app(plane), base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
    api.post("/environments/register", json=manifest())
    run = api.post("/runs", json={"goal": "read", "environmentId": "neutral", "idempotencyKey": "cancel"}).json()
    thread = Thread(target=plane.launch, args=(run["runId"],))
    thread.start()
    assert started.wait(timeout=1)
    assert api.post(f"/runs/{run['runId']}/cancel").status_code == 200
    release.set()
    thread.join(timeout=2)
    assert api.get(f"/runs/{run['runId']}").json()["status"] == "cancelled"


def test_candidate_evaluation_decision_and_rollback_boundaries():
    plane = ControlPlane(model_runner=model_runner, evaluator=evaluator)
    api = TestClient(create_app(plane), base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
    api.post("/environments/register", json=manifest())
    candidate = api.post(
        "/candidates",
        json={
            "baseBundleHash": plane.active_bundle_hash,
            "editOperations": ["retry after VERSION_CONFLICT"],
            "changedArtifactHashes": ["artifact-1"],
            "supportingEvidenceIds": ["ev-1"],
            "predictedEffect": "fewer stale writes",
            "proposerVersion": "planner-1",
        },
    )
    assert candidate.status_code == 201
    candidate_id = candidate.json()["candidateId"]
    evaluation = api.post(
        "/evaluations",
        json={"candidateId": candidate_id, "baseBundleHash": plane.active_bundle_hash, "protocolHash": "protocol-1", "partitionRef": {"id": "validation", "version": "1", "sha256": "v"}},
    )
    assert evaluation.status_code == 202
    evaluation_id = evaluation.json()["evaluationId"]
    assert api.post(f"/candidates/{candidate_id}/decision", json={"evaluationId": evaluation_id, "decision": "promoted", "reason": "looks good"}).status_code == 403
    assert api.post(f"/candidates/{candidate_id}/rollback", json={"reason": "operator safety rollback"}).status_code == 200


def test_trusted_evaluation_runner_report_is_pinned_before_decision():
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    runner = EvaluationRunner(protocol, packages, safety_cases={"EVAL-004": True, "EVAL-005": True})

    def execute(arm, package, task, seed):
        return RunObservation(task.task_id, package.environment_id, Partition.VALIDATION, seed, arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL)

    plane = ControlPlane()
    base_hash = plane.active_bundle_hash
    candidate = plane.create_candidate(CandidateProposalRequest(
        baseBundleHash=base_hash, editOperations=["bounded change"], changedArtifactHashes=["artifact"],
        supportingEvidenceIds=["evidence"], predictedEffect="improves accuracy", proposerVersion="test",
    ))
    report = runner.run_validation(base_hash=base_hash, candidate_hash=candidate["candidateId"], execute=execute)
    serialized = report.to_dict()
    assert serialized["promotionEligible"] is True
    assert serialized["exposure"][0]["environment_id"] == "finance"

    evaluation = plane.queue_evaluation(EvaluationRequest(
        candidateId=candidate["candidateId"], baseBundleHash=base_hash, protocolHash=report.protocol_hash,
        partitionRef={"id": "validation", "version": "1", "sha256": "partition"},
    ))
    class ForgedCandidateReport:
        def to_dict(self):
            value = dict(serialized)
            value["candidateHash"] = "another-candidate"
            return value

    with pytest.raises(ValueError, match="candidate hash"):
        plane.record_trusted_evaluation(evaluation["evaluationId"], ForgedCandidateReport())
    recorded = plane.record_trusted_evaluation(evaluation["evaluationId"], report)
    assert recorded["state"] == "valid"
    assert recorded["report"]["evaluatorTrusted"] is True


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


def test_control_plane_runs_generic_luna_python_through_prime_capability_seam():
    class PlannerClient:
        def __init__(self):
            self.turn = 0

        def invoke(self, *, goal, environment, messages):
            self.turn += 1
            text = (
                '{"action":"execute","code":"result = host_request({\\"type\\": \\"broker.call\\", '
                '\\"capabilityId\\": \\"counter.read\\", \\"arguments\\": {}})"}'
                if self.turn == 1 else '{"action":"finish","answer":"done"}'
            )
            return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": f"resp-{self.turn}", "text": text, "usage": {"outputTokens": 3}}

    class PrimeResult:
        status = "ok"
        result = "{\"value\": {\"count\": 1}}"
        stdout = ""
        stderr = ""
        error = None

    class PrimeKernel:
        def __init__(self):
            self.codes = []
            self.capability_calls = []

        def execute(self, code, *, timeout=None, cancel=None):
            self.codes.append(code)
            namespace = {"host_request": self.host_request}
            exec(code, {"__builtins__": {}}, namespace)
            return PrimeResult()

        def host_request(self, payload):
            self.capability_calls.append(payload)
            return {"value": {"count": 1}}

    class Sink:
        def __init__(self):
            self.observations = []

        def record_model_observation(self, evidence, *, trusted_parent=False):
            self.observations.append((evidence, trusted_parent))

    kernel, sink = PrimeKernel(), Sink()
    planner_runner = make_luna_model_runner(PlannerClient(), kernel, sink)
    plane = ControlPlane(model_runner=planner_runner, evaluator=lambda **_: {"passed": True})
    api = TestClient(create_app(plane), base_url="http://127.0.0.1")
    api.get("/session/bootstrap")
    api.post("/environments/register", json=manifest())
    run = api.post("/runs", json={"goal": "read the counter", "environmentId": "neutral", "idempotencyKey": "planner"}).json()
    assert api.post(f"/runs/{run['runId']}/launch").status_code == 202
    assert api.get(f"/runs/{run['runId']}").json()["status"] == "succeeded"
    assert kernel.capability_calls[0]["capabilityId"] == "counter.read"
    assert len(sink.observations) == 2

    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    eval_runner = EvaluationRunner(protocol, packages, safety_cases={"EVAL-004": True, "EVAL-005": True})
    base_hash = plane.active_bundle_hash
    candidate = plane.create_candidate(CandidateProposalRequest(
        baseBundleHash=base_hash, editOperations=["bounded generic planner refinement"], changedArtifactHashes=["artifact-planner"],
        supportingEvidenceIds=[run["runId"]], predictedEffect="improves verified task completion", proposerVersion="luna",
    ))

    def evaluate_row(arm, package, task, seed):
        return RunObservation(task.task_id, package.environment_id, Partition.VALIDATION, seed, arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL)

    report = eval_runner.run_validation(base_hash=base_hash, candidate_hash=candidate["candidateId"], execute=evaluate_row)
    evaluation = plane.queue_evaluation(EvaluationRequest(
        candidateId=candidate["candidateId"], baseBundleHash=base_hash, protocolHash=report.protocol_hash,
        partitionRef={"id": "validation", "version": "1", "sha256": "partition"},
    ))
    recorded = plane.record_trusted_evaluation(evaluation["evaluationId"], report)
    assert recorded["promotionEligible"] is True
    promoted = plane.record_trusted_decision(candidate["candidateId"], evaluation["evaluationId"], "promoted", "independent validation passed")
    assert promoted["state"] == "promoted"


def test_operator_bootstrap_and_loopback_origin_boundary():
    api = TestClient(create_app(ControlPlane()), base_url="http://127.0.0.1")
    # No server-issued cookie or bearer token may mutate or read protected data.
    assert api.get("/runs").status_code == 401
    assert api.get("/session").status_code == 401
    assert api.post("/environments/register", json=manifest()).status_code == 401
    # Rebinding the Host or Origin is rejected before authentication.
    assert api.get("/runs", headers={"host": "unrelated.example"}).status_code == 403
    assert api.get("/runs", headers={"origin": "https://unrelated.example"}).status_code == 403
    # Same-origin bootstrap establishes a cookie-only operator session.
    bootstrap = api.get("/session/bootstrap")
    assert bootstrap.status_code == 200
    assert "adaptive_operator_session" in bootstrap.headers["set-cookie"]
    assert "token" not in bootstrap.text.lower()
    assert api.get("/session").json() == {"authenticated": True, "transport": "live"}
    assert api.get("/runs", headers={"origin": "http://127.0.0.1"}).status_code == 200
