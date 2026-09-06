"""Safe side-effect gateway: tool permissions, schema validation, one-use approvals,
idempotency, and unknown-effect reconciliation.

The broker is the only component that executes environment tools. Every request is
checked against a run/environment-scoped Capability with an expiry and resource
scope, then validated against the registered tool input schema. Writes require a
single-use approval that is consumed atomically together with the prepared call
record. Learner-provided claims are never trusted; the broker emits the only
authoritative tool result envelope.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from adaptive_agent.environment import (
    EnvironmentRegistry,
    SchemaValidationError,
    validate_arguments,
)
from adaptive_agent.models import (
    ArtifactRef,
    ToolError,
    ToolErrorCode,
    ToolRequest,
    ToolResult,
    ToolSchema,
)
from adaptive_agent.store import Store


class ToolProvider(ABC):
    """Environment-supplied tool implementation. Executed only via the broker."""

    @abstractmethod
    def execute(self, run_id: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        ...

    @abstractmethod
    def effect(self, tool: str) -> str:
        """'read' or 'write' for the named tool."""
        ...

    @abstractmethod
    def version(self, tool: str) -> str:
        ...

    def reconcile(self, run_id: str, tool: str, arguments: dict[str, Any], idempotency_key: str) -> str:
        """Ask the provider whether a prepared call actually landed.

        Returns 'confirmed' (effect happened), 'no_effect' (safe to retry), or
        'unknown' (still indeterminate — stays OUTCOME_UNKNOWN).
        """
        return "unknown"


def _scope_match(scope_value: Any, arg_value: Any) -> bool:
    """Resource-scope constraint matching for a single argument.

    Supported forms:
      scalar            -> exact equality
      list/tuple/set    -> membership
      {"min": x, "max": y}                    -> numeric bounds (inclusive)
      {"pattern": "regex"}                    -> string match
      {"enum": [...]}                         -> membership
      {"prefix": "s"}                         -> string startswith
    """
    if isinstance(scope_value, dict):
        if "enum" in scope_value:
            return arg_value in scope_value["enum"]
        if "min" in scope_value and not (isinstance(arg_value, (int, float)) and arg_value >= scope_value["min"]):
            return False
        if "max" in scope_value and not (isinstance(arg_value, (int, float)) and arg_value <= scope_value["max"]):
            return False
        if "pattern" in scope_value:
            import re

            return isinstance(arg_value, str) and bool(re.search(scope_value["pattern"], arg_value))
        if "prefix" in scope_value:
            return isinstance(arg_value, str) and arg_value.startswith(scope_value["prefix"])
        return True
    if isinstance(scope_value, (list, tuple, set)):
        return arg_value in scope_value
    return scope_value == arg_value


@dataclass(frozen=True)
class Capability:
    """A run/environment-scoped, expiring capability.

    - run_id and environment_id must match the request exactly.
    - tool is exact-match (no wildcard authority by default).
    - effect must equal the schema-declared side-effect class.
    - resource_scope constrains the canonical arguments; every scoped key must be
      present in the request arguments and satisfy its constraint.
    - expires_at is a hard wall-clock deadline (UTC).
    """

    run_id: str
    environment_id: str
    tool: str
    effect: str
    resource_scope: dict[str, Any] = field(default_factory=dict)
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(seconds=60)
    )

    def covers(self, env_id: str, request: ToolRequest, schema: ToolSchema) -> ToolError | None:
        """Return None when the capability covers the request, else a ToolError."""
        if datetime.now(timezone.utc) > self.expires_at:
            return ToolError(
                code=ToolErrorCode.FORBIDDEN,
                message="capability expired",
                retry="never",
            )
        if self.run_id != request.run_id:
            return ToolError(
                code=ToolErrorCode.FORBIDDEN,
                message="capability bound to a different run",
                retry="never",
            )
        if self.environment_id != env_id:
            return ToolError(
                code=ToolErrorCode.FORBIDDEN,
                message="capability bound to a different environment",
                retry="never",
            )
        if self.tool != request.tool:
            return ToolError(
                code=ToolErrorCode.FORBIDDEN,
                message=f"capability does not cover tool {request.tool!r}",
                retry="never",
            )
        if schema.effect != self.effect:
            return ToolError(
                code=ToolErrorCode.FORBIDDEN,
                message=f"capability side-effect class {self.effect!r} != schema effect {schema.effect!r}",
                retry="never",
            )
        for key, scope_value in self.resource_scope.items():
            if key not in request.arguments:
                return ToolError(
                    code=ToolErrorCode.FORBIDDEN,
                    message=f"required resource scope argument {key!r} missing",
                    retry="never",
                )
            if not _scope_match(scope_value, request.arguments[key]):
                return ToolError(
                    code=ToolErrorCode.FORBIDDEN,
                    message=f"argument {key!r} outside capability resource scope",
                    retry="never",
                )
        return None


def _error_result(call_id: str, tool_version: str, error: ToolError) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        tool_version=tool_version,
        broker_evidence_ref=ArtifactRef(id="none", version="0", sha256="0" * 64),
        status="error",
        error=error,
        effect="none",
    )


Authorizer = Callable[[str, ToolRequest, ToolSchema], ToolError | None]


class ToolBroker:
    """Enforces policy, approvals, and idempotency for tool calls.

    `authorizer` is an optional authoritative authorization callback injected by
    the trusted parent (e.g. Prime's CapabilityBroker). It is consulted after
    local capability and schema checks, before any prepared record exists.
    Returning a ToolError denies the call (FORBIDDEN-shaped); returning None
    allows dispatch to proceed through the remaining fail-closed checks. When no
    authorizer is injected, all local checks still apply — the broker never
    defaults to permissive.
    """

    def __init__(
        self,
        store: Store,
        registry: EnvironmentRegistry,
        authorizer: Authorizer | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.authorizer = authorizer

    # ------------------------------------------------------------------ approvals
    def issue_approval(
        self,
        env_id: str,
        run_id: str,
        tool: str,
        canonical_args: dict[str, Any],
        idempotency_key: str,
        ttl_seconds: int = 300,
    ) -> str:
        """Operator-issued, single-use approval binding (run, env, tool, args, key)."""
        token = f"appr:{env_id}:{run_id}:{tool}:{idempotency_key}"
        expires = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
        self.store.issue_approval(
            token,
            run_id,
            env_id,
            tool,
            idempotency_key,
            json.dumps(canonical_args, sort_keys=True),
            expires,
        )
        return token

    # ------------------------------------------------------------------ dispatch
    def request_tool_call(
        self,
        env_id: str,
        request: ToolRequest,
        capability: Capability,
        provider: ToolProvider,
        budget_remaining: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Validate, authorize, and dispatch (or replay) a tool call."""
        schema = self.registry.get_tool_schema(env_id, request.tool)
        if schema is None:
            return _error_result(
                request.call_id,
                "unknown",
                ToolError(
                    code=ToolErrorCode.TOOL_UNAVAILABLE,
                    message=f"tool {request.tool!r} not registered for this environment",
                    retry="never",
                ),
            )

        # Capability: run/env/tool/effect/scope/expiry
        cap_err = capability.covers(env_id, request, schema)
        if cap_err is not None:
            return _error_result(request.call_id, schema.version, cap_err)

        # Argument schema validation
        try:
            validate_arguments(schema, request.arguments)
        except SchemaValidationError as exc:
            return _error_result(
                request.call_id,
                schema.version,
                ToolError(
                    code=ToolErrorCode.INVALID_INPUT,
                    message=str(exc),
                    retry="never",
                ),
            )

        # Injected authoritative authorizer (Prime CapabilityBroker seam).
        if self.authorizer is not None:
            auth_err = self.authorizer(env_id, request, schema)
            if auth_err is not None:
                return _error_result(request.call_id, schema.version, auth_err)

        # Budget
        if budget_remaining is not None and budget_remaining.get("tool_calls", 0) <= 0:
            return _error_result(
                request.call_id,
                schema.version,
                ToolError(
                    code=ToolErrorCode.BUDGET_EXHAUSTED,
                    message="tool call budget exhausted",
                    retry="never",
                ),
            )

        canonical_args = json.dumps(request.arguments, sort_keys=True)

        # Idempotent replay / conflict detection
        existing = self.store.get_tool_call_by_idempotency(request.run_id, request.idempotency_key)
        if existing is not None:
            if existing["arguments_json"] != canonical_args or existing["tool"] != request.tool:
                return _error_result(
                    request.call_id,
                    schema.version,
                    ToolError(
                        code=ToolErrorCode.IDEMPOTENCY_CONFLICT,
                        message="same idempotency key used with different content",
                        retry="never",
                    ),
                )
            if existing["result_json"]:
                prior = ToolResult.model_validate_json(existing["result_json"])
                return prior.model_copy(update={"call_id": request.call_id})
            # Prepared but unresolved: do not redispatch.
            if provider.effect(request.tool) == "write":
                return _error_result(
                    request.call_id,
                    schema.version,
                    ToolError(
                        code=ToolErrorCode.OUTCOME_UNKNOWN,
                        message="prepared write call is unresolved; reconcile before retry",
                        retry="after_reconciliation",
                    ),
                )
            # Read prepared-but-unresolved calls are safe to re-execute below.

        # Prepare + atomic approval consumption for writes
        call_data = {
            "call_id": request.call_id,
            "run_id": request.run_id,
            "step_id": request.step_id,
            "environment_id": env_id,
            "tool": request.tool,
            "arguments_json": canonical_args,
            "idempotency_key": request.idempotency_key,
        }
        if schema.effect == "write":
            if not request.approval_token:
                return _error_result(
                    request.call_id,
                    schema.version,
                    ToolError(
                        code=ToolErrorCode.FORBIDDEN,
                        message="write tool call requires a one-use approval token",
                        retry="never",
                    ),
                )
            status, _row = self.store.consume_approval_and_prepare(
                request.approval_token,
                call_data,
                request.run_id,
                request.tool,
                request.idempotency_key,
                canonical_args,
            )
            if status == "conflict":
                return _error_result(
                    request.call_id,
                    schema.version,
                    ToolError(
                        code=ToolErrorCode.IDEMPOTENCY_CONFLICT,
                        message="same idempotency key used with different content",
                        retry="never",
                    ),
                )
            if status != "ok":
                return _error_result(
                    request.call_id,
                    schema.version,
                    ToolError(
                        code=ToolErrorCode.FORBIDDEN,
                        message="approval token missing, consumed, expired, or mismatched",
                        retry="never",
                    ),
                )
        else:
            prepared = self.store.prepare_tool_call(call_data)
            if not prepared and existing is None:
                # Lost the race; fall back to the stored record.
                existing = self.store.get_tool_call_by_idempotency(
                    request.run_id, request.idempotency_key
                )
                if existing and existing["result_json"]:
                    prior = ToolResult.model_validate_json(existing["result_json"])
                    return prior.model_copy(update={"call_id": request.call_id})
                return _error_result(
                    request.call_id,
                    schema.version,
                    ToolError(
                        code=ToolErrorCode.IDEMPOTENCY_CONFLICT,
                        message="concurrent idempotency conflict",
                        retry="never",
                    ),
                )

        # Execute
        try:
            output = provider.execute(request.run_id, request.tool, request.arguments)
            effect = "confirmed" if provider.effect(request.tool) == "write" else "none"
            result = ToolResult(
                call_id=request.call_id,
                tool_version=schema.version,
                observed_at=datetime.now(timezone.utc),
                broker_evidence_ref=ArtifactRef(id="pending", version="0", sha256="0" * 64),
                status="ok",
                output=output,
                effect=effect,
            )
        except Exception as exc:
            result = ToolResult(
                call_id=request.call_id,
                tool_version=schema.version,
                observed_at=datetime.now(timezone.utc),
                broker_evidence_ref=ArtifactRef(id="pending", version="0", sha256="0" * 64),
                status="error",
                error=ToolError(
                    code=ToolErrorCode.OUTCOME_UNKNOWN,
                    message=str(exc),
                    retry="after_reconciliation",
                ),
                effect="unknown",
            )

        evidence_ref = self.store.put_artifact(result.model_dump(mode="json", by_alias=True))
        result = result.model_copy(update={"broker_evidence_ref": evidence_ref})
        result_json = result.model_dump_json(by_alias=True)
        self.store.save_tool_result(request.call_id, result_json, result.effect)
        return result

    # ------------------------------------------------------------------ reconciliation
    def reconcile_run(self, run_id: str, provider: ToolProvider) -> list[str]:
        """Resolve prepared calls that never recorded a result.

        - read calls: mark OUTCOME_UNKNOWN (they had no side effect; a retry is
          allowed only after this reconciliation result is recorded)
        - write calls: ask the provider to reconcile against the idempotency key;
          'confirmed'/'no_effect' record a terminal result, 'unknown' stays
          OUTCOME_UNKNOWN and blocks automatic repetition.
        """
        resolved: list[str] = []
        for call in self.store.list_unreconciled_calls(run_id):
            args = json.loads(call["arguments_json"])
            if provider.effect(call["tool"]) != "write":
                verdict = "no_effect"
            else:
                verdict = provider.reconcile(
                    run_id, call["tool"], args, call["idempotency_key"]
                )
            terminal = verdict in ("confirmed", "no_effect")
            result = ToolResult(
                call_id=call["call_id"],
                tool_version=provider.version(call["tool"]),
                broker_evidence_ref=ArtifactRef(id="reconcile", version="0", sha256="0" * 64),
                status="ok" if verdict == "confirmed" else "error",
                error=None
                if verdict == "confirmed"
                else ToolError(
                    code=ToolErrorCode.OUTCOME_UNKNOWN,
                    message=f"reconciliation verdict: {verdict}",
                    retry="safe_read" if verdict == "no_effect" else "never",
                ),
                effect="confirmed" if verdict == "confirmed" else "none" if verdict == "no_effect" else "unknown",
            )
            evidence_ref = self.store.put_artifact(result.model_dump(mode="json", by_alias=True))
            result = result.model_copy(update={"broker_evidence_ref": evidence_ref})
            self.store.mark_call_outcome_unknown(call["call_id"], result.model_dump_json(by_alias=True))
            if terminal:
                resolved.append(call["call_id"])
        return resolved
