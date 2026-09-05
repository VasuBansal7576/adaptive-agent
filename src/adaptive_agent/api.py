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
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import datetime, timezone
from typing import Any, Protocol

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


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
    docs: list[JsonObject] = Field(default_factory=list)
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


class CreateRunRequest(ApiModel):
    goal: str = Field(min_length=1)
    environment_id: str = Field(alias="environmentId", min_length=1)
    model_profile_ref: JsonObject | None = Field(default=None, alias="modelProfileRef")
    budget_ref: JsonObject | None = Field(default=None, alias="budgetRef")
    idempotency_key: str = Field(alias="idempotencyKey", min_length=1, max_length=256)
    execution_mode: str = Field("interactive", alias="executionMode")


class LearningRequest(ApiModel):
    run_id: str = Field(alias="runId", min_length=1)
    predicted_effect: str = Field(alias="predictedEffect", min_length=1)
    evidence_ids: list[str] = Field(default_factory=list, alias="evidenceIds")


class ApprovalRequest(ApiModel):
    approve: bool


class RollbackRequest(ApiModel):
    reason: str = Field(min_length=1)


class ModelInvocation(Protocol):
    """Authenticated model result required for a live execution."""

    text: str
    provider: str
    model: str
    usage: Mapping[str, Any]


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
        self._lock = threading.RLock()

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
            self._emit(run_id, "status", "Run started with authenticated model runner.")
        try:
            invocation = self.model_runner(goal=run["goal"], environment=env["manifest"], emit=lambda k, s, d=None: self._emit(run_id, k, s, d))
            provenance = {"provider": invocation.provider, "model": invocation.model, "usage": dict(invocation.usage)}
            self._emit(run_id, "evidence", "Authenticated model response received.", json.dumps(provenance, sort_keys=True))
            outcome = dict(self.evaluator(goal=run["goal"], model_output=invocation.text, environment=env["manifest"]))
            with self._lock:
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


def create_app(control: ControlPlane | None = None) -> FastAPI:
    plane = control or ControlPlane()
    app = FastAPI(title="Adaptive Agent Control API", version="0.1.0")
    app.state.control_plane = plane

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
        value = await request.json()
        if not isinstance(value, dict):
            raise HTTPException(status_code=422, detail="manifest must be an object")
        allowed = {"environmentId", "version", "toolSchemas", "policyRef", "evaluatorRef", "resetRef"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            return {"ok": False, "missingFields": unknown}
        required = ["environmentId", "version", "toolSchemas", "policyRef", "evaluatorRef", "resetRef"]
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
                policyRef=refs, evaluatorRef=evaluator, resetRef=reset,
            )
        except (json.JSONDecodeError, ValueError, ValidationError) as exc:
            return {"ok": False, "missingFields": ["toolSchemas" if "tool" in str(exc).lower() else "manifest"]}
        return {"ok": True, "missingFields": []}

    @app.post("/environments/register", status_code=201)
    def register_environment(payload: EnvironmentRegistration) -> JsonObject:
        return plane.register_environment(payload)

    @app.get("/runs")
    def runs() -> list[JsonObject]:
        with plane._lock:
            return [dict(run) for run in plane.runs.values()]

    @app.post("/runs", status_code=201)
    def create_run(payload: CreateRunRequest, background: BackgroundTasks) -> JsonObject:
        try:
            run, existing = plane.create_run(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except IdempotencyConflict as exc:
            raise HTTPException(status_code=409, detail={"code": "IDEMPOTENCY_CONFLICT", "message": str(exc), "correlationId": uuid.uuid4().hex, "retry": "never"}) from exc
        if not existing:
            background.add_task(plane.launch, run["runId"])
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

    @app.post("/learning/launch", status_code=202)
    def launch_learning(payload: LearningRequest) -> JsonObject:
        if payload.run_id not in plane.runs:
            raise HTTPException(status_code=404, detail="run not found")
        action = {"actionId": f"learn_{uuid.uuid4().hex}", "runId": payload.run_id, "predictedEffect": payload.predicted_effect, "evidenceIds": payload.evidence_ids, "status": "staged", "createdAt": _now()}
        plane.learning_actions.append(action)
        plane._emit(payload.run_id, "evidence", "Evidence-linked learning proposal staged.", payload.predicted_effect)
        return action

    @app.get("/skills")
    def skills() -> list[JsonObject]:
        return []

    @app.get("/candidates")
    def candidates() -> list[JsonObject]:
        return []

    @app.post("/candidates/{candidate_id}/rollback")
    def rollback(candidate_id: str, payload: RollbackRequest) -> JsonObject:
        raise HTTPException(status_code=404, detail="candidate not found")

    return app


app = create_app()
