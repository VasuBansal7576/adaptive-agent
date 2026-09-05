"""Authenticated, domain-neutral Luna planning loop.

The trusted parent owns model calls and evidence.  The learner receives only
the goal, public environment contract, and active skills, then emits bounded
Python for the Prime kernel.  Tool access remains a kernel-to-parent broker
callback; this module never implements a fixture workflow or executes tools on
the host.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import time
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable, Mapping, Protocol, Sequence


MODEL_PROVIDER = "openai-codex"
MODEL_NAME = "openai-codex/gpt-5.6-luna"


class PlannerError(RuntimeError):
    """A malformed model response or an execution boundary failure."""


class PlannerBudgetExceeded(PlannerError):
    """The parent-owned planner budget was exhausted."""


class PlannerCancelled(PlannerError):
    """The run was cancelled before the next model or kernel turn."""


class PlannerModelClient(Protocol):
    def invoke(self, *, goal: str, environment: Mapping[str, Any], messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]: ...


class PlannerEvidenceSink(Protocol):
    def record_model_observation(self, evidence: Mapping[str, Any], *, trusted_parent: bool = False) -> Any: ...


class KernelExecutor(Protocol):
    def execute(self, code: str, *, timeout: float | None = None, cancel: Event | None = None) -> Any: ...


@dataclass(frozen=True)
class PlannerLimits:
    max_turns: int = 8
    max_wall_seconds: float = 900.0
    max_model_tokens: int = 4_000
    max_code_chars: int = 32_768
    max_context_chars: int = 32_768
    max_output_chars: int = 16_384

    def __post_init__(self) -> None:
        if min(self.max_turns, self.max_model_tokens, self.max_code_chars, self.max_context_chars, self.max_output_chars) < 1 or self.max_wall_seconds <= 0:
            raise ValueError("planner limits must be positive")


@dataclass(frozen=True)
class PlannerEvent:
    kind: str
    summary: str
    detail: str | None = None


@dataclass(frozen=True)
class PlannerResult:
    status: str
    answer: str | None
    turns: int
    model_tokens: int
    kernel_steps: int
    response_ids: tuple[str, ...]
    events: tuple[PlannerEvent, ...]


def _bounded_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return text[:limit]


def _redact(value: Any) -> Any:
    """Remove common credential-shaped fields before model feedback."""
    if isinstance(value, Mapping):
        hidden = {"authorization", "cookie", "password", "secret", "token", "api_key", "apikey"}
        return {str(key): "[REDACTED]" if str(key).lower() in hidden else _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _usage_tokens(usage: Mapping[str, Any]) -> int:
    for key in ("totalTokens", "total_tokens", "outputTokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    raise PlannerError("model usage must include a non-negative token count")


def _parse_action(text: str, max_code_chars: int) -> tuple[str, str | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlannerError("model response must be a JSON action object") from exc
    if not isinstance(payload, Mapping):
        raise PlannerError("model action must be an object")
    action = payload.get("action")
    if action == "finish":
        answer = payload.get("answer", "")
        if not isinstance(answer, str):
            raise PlannerError("finish answer must be text")
        return "finish", answer
    if action == "execute":
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            raise PlannerError("execute action requires non-empty Python code")
        if len(code) > max_code_chars:
            raise PlannerError("generated Python exceeds the code limit")
        return "execute", code
    raise PlannerError("model action must be execute or finish")


class LunaPlanner:
    """Drive authenticated model turns and task-scoped Prime execution."""

    def __init__(
        self,
        client: PlannerModelClient,
        kernel: KernelExecutor,
        evidence_sink: PlannerEvidenceSink,
        *,
        limits: PlannerLimits = PlannerLimits(),
        emit: Callable[[PlannerEvent], None] | None = None,
    ) -> None:
        self.client = client
        self.kernel = kernel
        self.evidence_sink = evidence_sink
        self.limits = limits
        self.emit = emit

    def run(
        self,
        *,
        goal: str,
        environment: Mapping[str, Any],
        active_skills: Sequence[Mapping[str, Any]] = (),
        cancel: Event | None = None,
    ) -> PlannerResult:
        if not goal.strip():
            raise ValueError("goal is required")
        started = time.monotonic()
        events: list[PlannerEvent] = []
        response_ids: list[str] = []
        messages: list[Mapping[str, str]] = [{"role": "system", "content": self._system_prompt(environment, active_skills)}]
        messages.append({"role": "user", "content": f"Goal: {goal}"})
        model_tokens = 0
        kernel_steps = 0

        def publish(kind: str, summary: str, detail: str | None = None) -> None:
            event = PlannerEvent(kind, summary, detail)
            events.append(event)
            if self.emit:
                self.emit(event)

        for turn in range(1, self.limits.max_turns + 1):
            if cancel and cancel.is_set():
                publish("status", "Planner cancelled before model turn.")
                return PlannerResult("cancelled", None, turn - 1, model_tokens, kernel_steps, tuple(response_ids), tuple(events))
            remaining = self.limits.max_wall_seconds - (time.monotonic() - started)
            if remaining <= 0 or model_tokens >= self.limits.max_model_tokens:
                publish("status", "Planner budget exhausted.")
                return PlannerResult("budget_exhausted", None, turn - 1, model_tokens, kernel_steps, tuple(response_ids), tuple(events))

            raw = self._invoke(goal, environment, messages)
            provider, model, response_id, text, usage = self._validate_response(raw)
            used = _usage_tokens(usage)
            model_tokens += used
            if model_tokens > self.limits.max_model_tokens:
                publish("status", "Model token budget exceeded.")
                return PlannerResult("budget_exhausted", None, turn, model_tokens, kernel_steps, tuple(response_ids), tuple(events))
            self.evidence_sink.record_model_observation({"provider": provider, "model": model, "responseId": response_id, "usage": dict(usage)}, trusted_parent=True)
            response_ids.append(response_id)
            publish("model", "Authenticated Luna response received.", response_id)
            action, value = _parse_action(text, self.limits.max_code_chars)
            messages.append({"role": "assistant", "content": text})
            if action == "finish":
                publish("status", "Planner finished.")
                return PlannerResult("succeeded", value, turn, model_tokens, kernel_steps, tuple(response_ids), tuple(events))

            assert value is not None
            kernel_steps += 1
            publish("execute", "Generated Python submitted to Prime kernel.")
            result = self.kernel.execute(value, timeout=min(remaining, self.limits.max_wall_seconds), cancel=cancel)
            status = getattr(result, "status", "error")
            feedback = self._sanitize_execution(result)
            messages.append({"role": "user", "content": "Prime execution feedback:\n" + feedback})
            publish("kernel", "Prime execution result received.", feedback)
            if status == "aborted":
                final = "cancelled" if cancel and cancel.is_set() else "timed_out"
                return PlannerResult(final, None, turn, model_tokens, kernel_steps, tuple(response_ids), tuple(events))

        publish("status", "Planner turn budget exhausted.")
        return PlannerResult("budget_exhausted", None, self.limits.max_turns, model_tokens, kernel_steps, tuple(response_ids), tuple(events))

    def _system_prompt(self, environment: Mapping[str, Any], active_skills: Sequence[Mapping[str, Any]]) -> str:
        public = _redact({"environment": environment, "activeSkills": list(active_skills)})
        contract = _bounded_text(public, self.limits.max_context_chars)
        return (
            "You are the authenticated Luna planner. Generate a generic solution for the supplied goal. "
            "Use public docs, tool schemas, policy, and active skills as contracts. "
            "Return exactly one JSON object: {\\\"action\\\":\\\"execute\\\",\\\"code\\\":\\\"...\\\"} "
            "for Python to run in the task-scoped Prime kernel, or {\\\"action\\\":\\\"finish\\\",\\\"answer\\\":\\\"...\\\"}. "
            "Use the provided rlm host_request broker bridge for tools; never import credentials, access the host, or invent outcomes. "
            "A tool result or error is feedback for the next turn. The contract context is:\n" + contract
        )

    def _invoke(self, goal: str, environment: Mapping[str, Any], messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        invoke = self.client.invoke
        parameters = inspect.signature(invoke).parameters
        if "messages" in parameters or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return invoke(goal=goal, environment=environment, messages=messages)
        # Compatibility for the existing parent adapter while it is being
        # upgraded to retain conversation messages.
        return invoke(goal=goal, environment={**environment, "plannerMessages": list(messages)})

    @staticmethod
    def _validate_response(raw: Mapping[str, Any]) -> tuple[str, str, str, str, Mapping[str, Any]]:
        if not isinstance(raw, Mapping):
            raise PlannerError("model response must be an object")
        provider, model = raw.get("provider"), raw.get("model")
        response_id = raw.get("responseId", raw.get("response_id"))
        text, usage = raw.get("text"), raw.get("usage")
        if provider != MODEL_PROVIDER or model != MODEL_NAME or not isinstance(response_id, str) or not response_id.strip() or not isinstance(text, str) or not text.strip() or not isinstance(usage, Mapping) or not usage:
            raise PlannerError("authenticated model response requires provider, model, response id, text, and usage")
        return provider, model, response_id, text, usage

    def _sanitize_execution(self, result: Any) -> str:
        payload = {
            "status": getattr(result, "status", "error"),
            "result": getattr(result, "result", None),
            "stdout": getattr(result, "stdout", ""),
            "stderr": getattr(result, "stderr", ""),
            "error": getattr(result, "error", None),
        }
        return _bounded_text(_redact(payload), self.limits.max_output_chars)


def _load_object(path: str) -> Any:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("adapter must use module:object syntax")
    return getattr(importlib.import_module(module_name), attribute)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the authenticated Luna planner loop.")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--environment", required=True, help="JSON object or @path to a JSON object")
    parser.add_argument("--adapter", required=True, help="module:factory returning client, kernel, and evidence sink")
    parser.add_argument("--max-turns", type=int, default=PlannerLimits.max_turns)
    args = parser.parse_args(argv)
    raw_environment = args.environment[1:] if args.environment.startswith("@") else args.environment
    if args.environment.startswith("@"):
        with open(raw_environment, encoding="utf-8") as stream:
            raw_environment = stream.read()
    environment = json.loads(raw_environment)
    if not isinstance(environment, Mapping):
        raise ValueError("environment must be a JSON object")
    components = _load_object(args.adapter)()
    if not isinstance(components, Sequence) or len(components) != 3:
        raise ValueError("adapter factory must return (client, kernel, evidence_sink)")
    result = LunaPlanner(*components, limits=PlannerLimits(max_turns=args.max_turns)).run(goal=args.goal, environment=environment)
    print(json.dumps({"status": result.status, "answer": result.answer, "turns": result.turns, "modelTokens": result.model_tokens, "kernelSteps": result.kernel_steps, "responseIds": list(result.response_ids)}))
    return 0 if result.status == "succeeded" else 1


__all__ = ["KernelExecutor", "LunaPlanner", "MODEL_NAME", "MODEL_PROVIDER", "PlannerError", "PlannerEvent", "PlannerEvidenceSink", "PlannerLimits", "PlannerModelClient", "PlannerResult", "PlannerBudgetExceeded", "PlannerCancelled", "main"]

