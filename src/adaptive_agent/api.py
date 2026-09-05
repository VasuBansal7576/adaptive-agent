"""Control-plane API for the operator console.

The API owns persistence-facing lifecycle and idempotency boundaries.  Learner
code and the browser never get direct access to model, evaluator, or policy
implementations.  A deployment supplies an authenticated model runner and a
trusted evaluator through :class:`ControlPlane`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


JsonObject = dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ref(identifier: str, version: str = "1", value: Any = None) -> JsonObject:
    return {"id": identifier, "version": version, "sha256": _hash(identifier if value is None else value)}


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ToolSchemaInput(ApiModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    input_schema: JsonObject = Field(alias="inputSchema")
    output_schema: JsonObject = Field(alias="outputSchema")
    effect: str

    @field_validator("effect")
    @classmethod
    def valid_effect(cls, value: str) -> str:
        if value not in {"read", "write"}:
            raise ValueError("effect must be read or write")
        return value


class EnvironmentRegistration(ApiModel):
    schema_version: int = Field(1, alias="schemaVersion")
    environment_id: str = Field(alias="environmentId", min_length=1)
    version: str = Field(min_length=1)
    docs: list[JsonObject] = Field(min_length=1)
    task_goals: list[str] = Field(alias="taskGoals", min_length=1)
    tool_schemas: list[ToolSchemaInput] = Field(alias="toolSchemas", min_length=1)
    policy_ref: JsonObject = Field(alias="policyRef")
    evaluator_ref: JsonObject = Field(alias="evaluatorRef")
    reset_ref: JsonObject = Field(alias="resetRef")
    execution_modes: list[str] = Field(default_factory=lambda: ["interactive"], alias="executionModes")
    capabilities: list[str] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def supported_schema(cls, value: int) -> int:
        if value != 1:
            raise ValueError("unsupported manifest schemaVersion")
        return value

    @field_validator("execution_modes")
    @classmethod
    def supported_modes(cls, value: list[str]) -> list[str]:
        allowed = {"interactive", "batch", "dry_run", "replay"}
        if not value or any(mode not in allowed for mode in value):
            raise ValueError("executionModes must contain interactive, batch, dry_run, or replay")
        return value


class EnvironmentFormRegistration(ApiModel):
    """String-valued shape emitted by the console registry form."""

    environment_id: str = Field(alias="environmentId", min_length=1)
    version: str = Field(min_length=1)
    tool_schemas: str = Field(alias="toolSchemas", min_length=1)
    policy_ref: str = Field(alias="policyRef", min_length=1)
    evaluator_ref: str = Field(alias="evaluatorRef", min_length=1)
    reset_ref: str = Field(alias="resetRef", min_length=1)
    task_goals: list[str] = Field(alias="taskGoals", min_length=1)

    def manifest(self) -> EnvironmentRegistration:
        try:
            schemas = json.loads(self.tool_schemas)
        except json.JSONDecodeError as exc:
            raise ValueError("toolSchemas must be valid JSON") from exc
        if not isinstance(schemas, list):
            raise ValueError("toolSchemas must be a JSON array")
        return EnvironmentRegistration(
            environmentId=self.environment_id,
            version=self.version,
            docs=[{"id": "console-docs", "version": self.version, "sha256": _hash(self.tool_schemas)}],
            taskGoals=self.task_goals,
            toolSchemas=schemas,
            policyRef=_ref(self.policy_ref),
            evaluatorRef=_ref(self.evaluator_ref),
            resetRef=_ref(self.reset_ref),
        )


class CreateRunRequest(ApiModel):
    # The console may send the convenient goal/environment projection, while
    # external clients can submit the canonical SPEC task reference.
    goal: str | None = Field(default=None, min_length=1)
    environment_id: str | None = Field(default=None, alias="environmentId", min_length=1)
    task_ref: JsonObject | None = Field(default=None, alias="taskRef")
    model_profile_ref: JsonObject | None = Field(default=None, alias="modelProfileRef")
    budget_ref: JsonObject | None = Field(default=None, alias="budgetRef")
    idempotency_key: str = Field(alias="idempotencyKey", min_length=1, max_length=256)
    execution_mode: str = Field("interactive", alias="executionMode")

    @model_validator(mode="before")
    @classmethod
    def normalize_task_projection(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        task = normalized.get("taskRef")
        if isinstance(task, dict):
            if not normalized.get("goal") and isinstance(task.get("goal"), str):
                normalized["goal"] = task["goal"]
            if not normalized.get("environmentId"):
                env = task.get("environmentId")
                if not env and isinstance(task.get("environmentRef"), dict):
                    env = task["environmentRef"].get("id")
                if isinstance(env, str):
                    normalized["environmentId"] = env
        if not isinstance(normalized.get("goal"), str) or not normalized["goal"].strip():
            raise ValueError("goal or taskRef.goal is required")
        if not isinstance(normalized.get("environmentId"), str) or not normalized["environmentId"].strip():
            raise ValueError("environmentId or taskRef.environmentId is required")
        return normalized


class LearningRequest(ApiModel):
    run_id: str = Field(alias="runId", min_length=1)
    predicted_effect: str = Field(alias="predictedEffect", min_length=1)
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")


class CandidateProposalRequest(ApiModel):
    base_bundle_hash: str = Field(alias="baseBundleHash", min_length=1)
    edit_operations: list[str] = Field(alias="editOperations", min_length=1)
    changed_artifact_hashes: list[str] = Field(alias="changedArtifactHashes", min_length=1)
    supporting_evidence_ids: list[str] = Field(alias="supportingEvidenceIds", min_length=1)
    predicted_effect: str = Field(alias="predictedEffect", min_length=1)
    proposer_version: str = Field(alias="proposerVersion", min_length=1)

    @model_validator(mode="after")
    def bounded_change(self) -> "CandidateProposalRequest":
        if len(self.changed_artifact_hashes) > 3 or sum(len(edit.splitlines()) for edit in self.edit_operations) > 200:
            raise ValueError("candidate exceeds bounded edit limits")
        return self


class EvaluationRequest(ApiModel):
    candidate_id: str = Field(alias="candidateId", min_length=1)
    base_bundle_hash: str = Field(alias="baseBundleHash", min_length=1)
    protocol_hash: str = Field(alias="protocolHash", min_length=1)
    partition_ref: JsonObject = Field(alias="partitionRef")


class CandidateDecisionRequest(ApiModel):
    evaluation_id: str = Field(alias="evaluationId", min_length=1)
    decision: str
    reason: str = Field(min_length=1)

    @field_validator("decision")
    @classmethod
    def valid_decision(cls, value: str) -> str:
        if value not in {"promoted", "rejected", "quarantined"}:
            raise ValueError("decision must be promoted, rejected, or quarantined")
        return value


class RollbackRouteRequest(ApiModel):
    candidate_id: str = Field(alias="candidateId", min_length=1)
    reason: str = Field(min_length=1)


class ApprovalRequest(ApiModel):
    approve: bool


class RollbackRequest(ApiModel):
    reason: str = Field(min_length=1)


class ModelInvocation(Protocol):
    """Authenticated model result required for a live execution."""

    text: str
    provider: str
    model: str
    response_id: str
    usage: Mapping[str, Any]


class AuthenticatedModelClient(Protocol):
    """Trusted parent client for the configured subscription provider."""

    def invoke(self, *, goal: str, environment: JsonObject) -> Mapping[str, Any]: ...


class ModelEvidenceSink(Protocol):
    """Prime adapter seam for record_model_observation(..., trusted_parent=True)."""

    def record_model_observation(self, evidence: Mapping[str, Any], *, trusted_parent: bool = False) -> Any: ...


@dataclass(frozen=True)
class AuthenticatedInvocation:
    text: str
    provider: str
    model: str
    response_id: str
    usage: Mapping[str, Any]


def make_authenticated_model_runner(client: AuthenticatedModelClient, evidence_sink: ModelEvidenceSink) -> ModelRunner:
    """Bridge a parent-owned model response into the run controller."""
    def runner(*, goal: str, environment: JsonObject, emit: Callable[[str, str, str | None], None]) -> AuthenticatedInvocation:
        del emit
        raw = client.invoke(goal=goal, environment=environment)
        if not isinstance(raw, Mapping):
            raise ModelUnavailableError("model client returned a non-object response")
        provider = raw.get("provider")
        model = raw.get("model")
        response_id = raw.get("responseId", raw.get("response_id"))
        text = raw.get("text")
        usage = raw.get("usage")
        if not all(isinstance(value, str) and value.strip() for value in (provider, model, response_id, text)) or not isinstance(usage, Mapping) or not usage:
            raise ModelUnavailableError("authenticated model response requires provider, model, response id, text, and usage")
        evidence_sink.record_model_observation({"provider": provider, "model": model, "responseId": response_id, "usage": dict(usage)}, trusted_parent=True)
        return AuthenticatedInvocation(text, provider, model, response_id, dict(usage))

    return runner


class ModelRunner(Protocol):
    def __call__(self, *, goal: str, environment: JsonObject, emit: Callable[[str, str, str | None], None]) -> ModelInvocation: ...


class OutcomeEvaluator(Protocol):
    def __call__(self, *, goal: str, model_output: str, environment: JsonObject) -> Mapping[str, Any]: ...


class ModelUnavailableError(RuntimeError):
    pass


def unavailable_model(**_: Any) -> ModelInvocation:
    raise ModelUnavailableError("authenticated model runner is not configured")


def default_evaluator(*, goal: str, model_output: str, environment: JsonObject) -> Mapping[str, Any]:
    """Safe default used only for development registration.

    Production deployments must inject the trusted evaluator.  This evaluator
    deliberately reports an unknown outcome instead of manufacturing a pass.
    """
    return {"passed": False, "status": "outcome_unknown", "reason": "trusted evaluator is not configured"}


class ControlPlane:
    def __init__(self, model_runner: ModelRunner = unavailable_model, evaluator: OutcomeEvaluator = default_evaluator) -> None:
        self.model_runner = model_runner
        self.evaluator = evaluator
        self.environments: dict[str, JsonObject] = {}
        self.runs: dict[str, JsonObject] = {}
        self.events: dict[str, list[JsonObject]] = {}
        self.idempotency: dict[str, tuple[str, str]] = {}
        self.learning_actions: list[JsonObject] = []
        self.candidates: dict[str, JsonObject] = {}
        self.evaluations: dict[str, JsonObject] = {}
        self.active_bundle_hash = _hash("bundle-active")
        self.version_history: list[JsonObject] = [{"bundleHash": self.active_bundle_hash, "state": "active", "at": _now()}]
        self.approvals: dict[tuple[str, str], bool] = {}
        self._lock = threading.RLock()
        # Held by the local operator session and never returned in a JSON
        # response or passed into the learner/runtime process.
        self.operator_token = secrets.token_urlsafe(32)

    def create_candidate(self, payload: CandidateProposalRequest) -> JsonObject:
        with self._lock:
            if payload.base_bundle_hash != self.active_bundle_hash:
                raise IdempotencyConflict("active bundle changed; candidate base is stale")
            candidate_id = f"cand_{uuid.uuid4().hex}"
            candidate = {
                "candidateId": candidate_id,
                "baseBundleHash": payload.base_bundle_hash,
                "state": "validated",
                "editOperations": list(payload.edit_operations),
                "changedArtifactHashes": list(payload.changed_artifact_hashes),
                "supportingEvidenceIds": list(payload.supporting_evidence_ids),
                "predictedEffect": payload.predicted_effect,
                "proposerVersion": payload.proposer_version,
                "createdAt": _now(),
            }
            self.candidates[candidate_id] = candidate
            return dict(candidate)

    def queue_evaluation(self, payload: EvaluationRequest) -> JsonObject:
        with self._lock:
            candidate = self.candidates.get(payload.candidate_id)
            if candidate is None:
                raise KeyError("candidate not found")
            if payload.base_bundle_hash != candidate["baseBundleHash"]:
                raise ValueError("evaluation base does not match candidate")
            evaluation_id = f"eval_{uuid.uuid4().hex}"
            evaluation = {
                "evaluationId": evaluation_id,
                "candidateId": payload.candidate_id,
                "baseBundleHash": payload.base_bundle_hash,
                "protocolHash": payload.protocol_hash,
                "partitionRef": payload.partition_ref,
                "state": "queued",
                "trusted": False,
                "createdAt": _now(),
            }
            self.evaluations[evaluation_id] = evaluation
            candidate["state"] = "evaluating"
            return dict(evaluation)

    def record_trusted_decision(self, candidate_id: str, evaluation_id: str, decision: str, reason: str) -> JsonObject:
        """Apply a decision only from a trusted evaluator integration."""
        with self._lock:
            candidate = self.candidates.get(candidate_id)
            evaluation = self.evaluations.get(evaluation_id)
            if candidate is None or evaluation is None or evaluation["candidateId"] != candidate_id:
                raise KeyError("candidate or evaluation not found")
            if evaluation["state"] != "valid":
                raise ValueError("only a valid trusted evaluation may activate a candidate")
            evaluation["decision"] = decision
            evaluation["reason"] = reason
            evaluation["trusted"] = True
            candidate["state"] = decision
            if decision == "promoted":
                self.active_bundle_hash = _hash(candidate_id)
                self.version_history.append({"bundleHash": self.active_bundle_hash, "candidateId": candidate_id, "state": "active", "at": _now()})
            return dict(candidate)

    def register_environment(self, payload: EnvironmentRegistration) -> JsonObject:
        with self._lock:
            key = f"{payload.environment_id}@{payload.version}"
            manifest = payload.model_dump(by_alias=True, mode="json")
            summary = {
                "environmentId": payload.environment_id,
                "version": payload.version,
                "validationState": "valid",
                "evaluatorReady": payload.evaluator_ref.get("id") not in {None, "", "unconfigured"},
                "toolCount": len(payload.tool_schemas),
                "policyScope": str(payload.policy_ref.get("id", "")),
            }
            self.environments[key] = {"manifest": manifest, "summary": summary}
            return summary

    def list_environments(self) -> list[JsonObject]:
        with self._lock:
            return [dict(item["summary"]) for item in self.environments.values()]

    def create_run(self, payload: CreateRunRequest) -> tuple[JsonObject, bool]:
        with self._lock:
            env = next((entry for key, entry in self.environments.items() if key.startswith(f"{payload.environment_id}@")), None)
            if env is None:
                raise KeyError("environment is not registered")
            if payload.execution_mode not in env["manifest"].get("executionModes", ["interactive"]):
                raise ValueError("execution mode is not declared by environment")
            canonical = _hash(payload.model_dump(by_alias=True, mode="json"))
            prior = self.idempotency.get(payload.idempotency_key)
            if prior:
                prior_run_id, prior_hash = prior
                if prior_hash != canonical:
                    raise IdempotencyConflict(prior_run_id)
                return dict(self.runs[prior_run_id]), True
            run_id = f"run_{uuid.uuid4().hex}"
            env_ref = _ref(payload.environment_id, env["manifest"]["version"], env["manifest"])
            run = {
                "runId": run_id,
                "taskRef": _ref(f"task_{run_id}", "1", {"goal": payload.goal}),
                "environmentRef": env_ref,
                "policyRef": env["manifest"]["policyRef"],
                "modelProfileRef": payload.model_profile_ref or _ref("model-profile", "1"),
                "skillBundleRef": _ref("bundle-active", "1"),
                "budgetRef": payload.budget_ref or _ref("budget-default", "1"),
                "status": "queued",
                "lastEventSequence": 0,
                "environmentId": payload.environment_id,
                "executionMode": payload.execution_mode,
                "goal": payload.goal,
                "budgetUsed": {"calls": 0, "callsCeiling": 100, "wallSeconds": 0, "wallCeiling": 900},
            }
            self.runs[run_id] = run
            self.events[run_id] = []
            self.idempotency[payload.idempotency_key] = (run_id, canonical)
            self._emit(run_id, "status", "Run queued and pinned to active bundle.")
            return dict(run), False

    def _emit(self, run_id: str, kind: str, summary: str, detail: str | None = None, error: JsonObject | None = None) -> JsonObject:
        with self._lock:
            sequence = len(self.events.setdefault(run_id, [])) + 1
            event: JsonObject = {"runId": run_id, "sequence": sequence, "at": _now(), "kind": kind, "summary": summary}
            if detail is not None:
                event["detail"] = detail
            if error is not None:
                event["error"] = error
            self.events[run_id].append(event)
            if run_id in self.runs:
                self.runs[run_id]["lastEventSequence"] = sequence
            return event

    def launch(self, run_id: str) -> None:
        with self._lock:
            run = self.runs.get(run_id)
            if run is None:
                raise KeyError("run not found")
            if run["status"] not in {"queued", "failed"}:
                return
            env = next((entry for entry in self.environments.values() if entry["summary"]["environmentId"] == run["environmentId"]), None)
            if env is None:
                raise KeyError("environment is not registered")
            run["status"] = "running"
            self._emit(run_id, "status", f"Run started in {run['executionMode']} mode with authenticated model runner.")
        try:
            invocation = self.model_runner(goal=run["goal"], environment=env["manifest"], emit=lambda k, s, d=None: self._emit(run_id, k, s, d))
            provider = getattr(invocation, "provider", "")
            model = getattr(invocation, "model", "")
            response_id = getattr(invocation, "response_id", getattr(invocation, "responseId", ""))
            usage = getattr(invocation, "usage", None)
            if provider != "openai-codex" or model != "openai-codex/gpt-5.6-luna" or not isinstance(response_id, str) or not response_id.strip() or not isinstance(usage, Mapping) or not usage:
                raise ModelUnavailableError("model runner did not return authenticated provider, model, response id, and usage")
            with self._lock:
                if run["status"] == "cancelled":
                    return
            provenance = {"provider": invocation.provider, "model": invocation.model, "responseId": response_id, "usage": dict(invocation.usage)}
            self._emit(run_id, "evidence", "Authenticated model response received.", json.dumps(provenance, sort_keys=True))
            outcome = dict(self.evaluator(goal=run["goal"], model_output=invocation.text, environment=env["manifest"]))
            with self._lock:
                if run["status"] == "cancelled":
                    return
                run["outcomeRef"] = _ref(f"outcome_{run_id}", "1", outcome)
                run["status"] = "succeeded" if outcome.get("passed") is True else "failed"
            self._emit(run_id, "evidence", "Trusted evaluator recorded outcome.", json.dumps(outcome, sort_keys=True))
            self._emit(run_id, "status", "Run succeeded." if outcome.get("passed") is True else "Run failed.")
        except ModelUnavailableError as exc:
            with self._lock:
                run["status"] = "failed"
            self._emit(run_id, "status", "Run failed: model unavailable.", str(exc), {"code": "TOOL_UNAVAILABLE", "message": str(exc), "correlationId": uuid.uuid4().hex, "retry": "never"})
        except Exception as exc:  # operational failures become observable run failures
            with self._lock:
                run["status"] = "failed"
            self._emit(run_id, "status", "Run failed: execution error.", str(exc), {"code": "TOOL_UNAVAILABLE", "message": str(exc), "correlationId": uuid.uuid4().hex, "retry": "never"})

    def cancel(self, run_id: str) -> JsonObject:
        with self._lock:
            run = self.runs.get(run_id)
            if run is None:
                raise KeyError("run not found")
            if run["status"] in {"succeeded", "failed", "cancelled", "timed_out"}:
                return dict(run)
            run["status"] = "cancelled"
            self._emit(run_id, "status", "Cancelled by operator; future calls revoked.")
            return dict(run)


class IdempotencyConflict(RuntimeError):
    def __init__(self, run_id: str) -> None:
        super().__init__("idempotency key is already bound to a different request")
        self.run_id = run_id


def _loopback_name(value: str | None) -> bool:
    if not value:
        return False
    # urlsplit handles host:port and bracketed IPv6 literals consistently.
    parsed = urlsplit(f"http://{value}")
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def _same_loopback_origin(request: Request) -> bool:
    host = request.headers.get("host")
    if not _loopback_name(host):
        return False
    origin = request.headers.get("origin")
    if not origin:
        return True
    parsed = urlsplit(origin)
    if parsed.scheme not in {"http", "https"} or not _loopback_name(parsed.netloc):
        return False
    host_parsed = urlsplit(f"//{host}")
    if parsed.hostname != host_parsed.hostname or parsed.scheme != request.url.scheme:
        return False
    origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    host_port = host_parsed.port or (443 if request.url.scheme == "https" else 80)
    return origin_port == host_port


def _has_operator_session(request: Request, plane: ControlPlane) -> bool:
    bearer = request.headers.get("authorization", "")
    token = bearer.removeprefix("Bearer ").strip() if bearer.startswith("Bearer ") else ""
    cookie = request.cookies.get("adaptive_operator_session", "")
    return secrets.compare_digest(token or cookie, plane.operator_token)


def create_app(control: ControlPlane | None = None) -> FastAPI:
    plane = control or ControlPlane()
    app = FastAPI(title="Adaptive Agent Control API", version="0.1.0")
    app.state.control_plane = plane

    @app.middleware("http")
    async def access_boundary(request: Request, call_next: Callable[..., Any]) -> Any:
        if not _same_loopback_origin(request):
            return JSONResponse(status_code=403, content={"code": "FORBIDDEN", "message": "control API requires a loopback Host and same-origin request", "correlationId": uuid.uuid4().hex, "retry": "never"})
        if request.url.path in {"/health", "/session/bootstrap"}:
            return await call_next(request)
        if not _has_operator_session(request, plane):
            return JSONResponse(status_code=401, content={"code": "FORBIDDEN", "message": "operator session required", "correlationId": uuid.uuid4().hex, "retry": "never"})
        return await call_next(request)

    @app.get("/session/bootstrap")
    def session_bootstrap(response: Response) -> JsonObject:
        # The token is delivered only as an HttpOnly cookie. Browser JavaScript
        # receives a readiness marker, never the credential itself.
        response.set_cookie("adaptive_operator_session", plane.operator_token, httponly=True, samesite="strict", secure=False, path="/")
        return {"status": "ready", "transport": "live"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/environments")
    def environments() -> list[JsonObject]:
        return plane.list_environments()

    @app.post("/environments/validate")
    async def validate_environment(request: Request) -> JsonObject:
        """Validate the browser's registration form without registering it.

        The form intentionally carries references and tool schemas as strings.
        Parse those strings at this boundary, then run the same strict manifest
        model used by registration.  Unknown keys are rejected rather than
        silently becoming privileged configuration.
        """
        try:
            value = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=422, detail="request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise HTTPException(status_code=422, detail="manifest must be an object")
        allowed = {"environmentId", "version", "toolSchemas", "policyRef", "evaluatorRef", "resetRef", "taskGoals"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            return {"ok": False, "missingFields": unknown}
        required = ["environmentId", "version", "toolSchemas", "policyRef", "evaluatorRef", "resetRef", "taskGoals"]
        missing = [key for key in required if not isinstance(value.get(key), str) or not value[key].strip()]
        if missing:
            return {"ok": False, "missingFields": missing}
        try:
            schemas = json.loads(value["toolSchemas"])
            if not isinstance(schemas, list):
                raise ValueError("toolSchemas must be a JSON array")
            refs = {
                "id": value["policyRef"],
                "version": "1",
                "sha256": _hash(value["policyRef"]),
            }
            evaluator = {"id": value["evaluatorRef"], "version": "1", "sha256": _hash(value["evaluatorRef"])}
            reset = {"id": value["resetRef"], "version": "1", "sha256": _hash(value["resetRef"])}
            EnvironmentRegistration(
                environmentId=value["environmentId"], version=value["version"], toolSchemas=schemas,
                docs=[{"id": "console-docs", "version": value["version"], "sha256": _hash(value["toolSchemas"])}],
                taskGoals=json.loads(value["taskGoals"]) if value["taskGoals"].lstrip().startswith("[") else [value["taskGoals"]],
                policyRef=refs, evaluatorRef=evaluator, resetRef=reset,
            )
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            return {"ok": False, "missingFields": ["toolSchemas" if "tool" in str(exc).lower() else "manifest"]}
        return {"ok": True, "missingFields": []}

    @app.post("/environments/register", status_code=201)
    def register_environment(payload: EnvironmentRegistration) -> JsonObject:
        return plane.register_environment(payload)

    # Keep the resource-style route as a compatibility alias for clients that
    # model registration as creation.  Both routes use the same strict model.
    @app.post("/environments", status_code=201)
    def register_environment(payload: EnvironmentRegistration) -> JsonObject:
        return plane.register_environment(payload)

    @app.post("/environments/form", status_code=201, include_in_schema=False)
    def register_environment_form(payload: EnvironmentFormRegistration) -> JsonObject:
        try:
            return plane.register_environment(payload.manifest())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/runs")
    def runs() -> list[JsonObject]:
        with plane._lock:
            return [dict(run) for run in plane.runs.values()]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> JsonObject:
        with plane._lock:
            run = plane.runs.get(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="run not found")
            return dict(run)

    @app.get("/runs/{run_id}/evidence")
    def run_evidence(run_id: str) -> list[JsonObject]:
        with plane._lock:
            if run_id not in plane.runs:
                raise HTTPException(status_code=404, detail="run not found")
            # Evaluator-only events are never exposed through this projection.
            return [dict(event) for event in plane.events.get(run_id, []) if event.get("kind") == "evidence" and event.get("visibility") != "evaluator_only"]

    @app.post("/runs", status_code=201)
    def create_run(payload: CreateRunRequest) -> JsonObject:
        try:
            run, existing = plane.create_run(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail={"code": "IDEMPOTENCY_CONFLICT", "message": str(exc), "correlationId": uuid.uuid4().hex, "retry": "never"}) from exc
        return run

    @app.post("/runs/{run_id}/launch", status_code=202)
    def launch_run(run_id: str, background: BackgroundTasks) -> JsonObject:
        if run_id not in plane.runs:
            raise HTTPException(status_code=404, detail="run not found")
        background.add_task(plane.launch, run_id)
        return {"runId": run_id, "status": "accepted"}

    @app.get("/runs/{run_id}/events")
    def run_events(run_id: str, cursor: int = Query(0, ge=0)) -> StreamingResponse:
        if run_id not in plane.runs:
            raise HTTPException(status_code=404, detail="run not found")

        async def stream() -> AsyncIterator[str]:
            sent = cursor
            while True:
                with plane._lock:
                    events = [event for event in plane.events.get(run_id, []) if event["sequence"] > sent]
                    terminal = plane.runs[run_id]["status"] in {"succeeded", "failed", "cancelled", "timed_out"}
                for event in events:
                    sent = event["sequence"]
                    yield f"id: {sent}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"
                if terminal and not events:
                    break
                await asyncio.sleep(0.05)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> JsonObject:
        try:
            return plane.cancel(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/runs/{run_id}/approvals/{approval_id}")
    def submit_approval(run_id: str, approval_id: str, payload: ApprovalRequest) -> JsonObject:
        with plane._lock:
            if run_id not in plane.runs:
                raise HTTPException(status_code=404, detail="run not found")
            key = (run_id, approval_id)
            if key in plane.approvals:
                raise HTTPException(status_code=409, detail={"code": "IDEMPOTENCY_CONFLICT", "message": "approval already consumed", "correlationId": uuid.uuid4().hex, "retry": "never"})
            plane.approvals[key] = payload.approve
            plane._emit(run_id, "approval", "Approval accepted." if payload.approve else "Approval rejected.")
            return {"runId": run_id, "approvalId": approval_id, "approved": payload.approve}

    def _launch_learning(payload: LearningRequest) -> JsonObject:
        if payload.run_id not in plane.runs:
            raise HTTPException(status_code=404, detail="run not found")
        action = {"actionId": f"learn_{uuid.uuid4().hex}", "runId": payload.run_id, "predictedEffect": payload.predicted_effect, "evidenceIds": payload.evidence_ids, "status": "staged", "createdAt": _now()}
        plane.learning_actions.append(action)
        plane._emit(payload.run_id, "evidence", "Evidence-linked learning proposal staged.", payload.predicted_effect)
        return action

    @app.post("/learning/launch", status_code=202)
    def launch_learning(payload: LearningRequest) -> JsonObject:
        return _launch_learning(payload)

    @app.post("/runs/{run_id}/learning", status_code=202, include_in_schema=False)
    def launch_run_learning(run_id: str, payload: LearningRequest) -> JsonObject:
        if payload.run_id != run_id:
            raise HTTPException(status_code=422, detail="runId does not match path")
        return _launch_learning(payload)

    @app.get("/skills")
    def skills() -> list[JsonObject]:
        return []

    @app.get("/candidates")
    def candidates() -> list[JsonObject]:
        with plane._lock:
            return [dict(candidate) for candidate in plane.candidates.values()]

    @app.post("/candidates", status_code=201)
    def create_candidate(payload: CandidateProposalRequest) -> JsonObject:
        try:
            return plane.create_candidate(payload)
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail={"code": "VERSION_CONFLICT", "message": str(exc), "correlationId": uuid.uuid4().hex, "retry": "never"}) from exc

    @app.post("/evaluations", status_code=202)
    def create_evaluation(payload: EvaluationRequest) -> JsonObject:
        try:
            return plane.queue_evaluation(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"code": "VERSION_CONFLICT", "message": str(exc), "correlationId": uuid.uuid4().hex, "retry": "never"}) from exc

    @app.get("/evaluations")
    def evaluations() -> list[JsonObject]:
        with plane._lock:
            return [dict(evaluation) for evaluation in plane.evaluations.values()]

    @app.post("/candidates/{candidate_id}/decision")
    def candidate_decision(candidate_id: str, payload: CandidateDecisionRequest) -> JsonObject:
        # Browser/operator requests cannot forge a trusted report. Decisions
        # remain queued until the evaluator integration calls the trusted method.
        with plane._lock:
            if candidate_id not in plane.candidates or payload.evaluation_id not in plane.evaluations:
                raise HTTPException(status_code=404, detail="candidate or evaluation not found")
            raise HTTPException(status_code=403, detail={"code": "FORBIDDEN", "message": "only the trusted evaluator may decide a candidate", "correlationId": uuid.uuid4().hex, "retry": "never"})

    @app.get("/versions/active")
    def active_versions() -> list[JsonObject]:
        with plane._lock:
            return [dict(version) for version in plane.version_history]

    @app.post("/candidates/{candidate_id}/rollback")
    def rollback(candidate_id: str, payload: RollbackRequest) -> JsonObject:
        return rollback_resource(RollbackRouteRequest(candidateId=candidate_id, reason=payload.reason))

    @app.post("/rollbacks")
    def rollback_resource(payload: RollbackRouteRequest) -> JsonObject:
        with plane._lock:
            candidate = plane.candidates.get(payload.candidate_id)
            if candidate is None:
                raise HTTPException(status_code=404, detail="candidate not found")
            prior = plane.version_history[-2] if len(plane.version_history) > 1 else plane.version_history[0]
            plane.active_bundle_hash = prior["bundleHash"]
            candidate["state"] = "rolled_back"
            plane.version_history.append({"bundleHash": prior["bundleHash"], "candidateId": payload.candidate_id, "state": "rolled_back", "reason": payload.reason, "at": _now()})
            return plane.version_history[-1]

    return app


app = create_app()
