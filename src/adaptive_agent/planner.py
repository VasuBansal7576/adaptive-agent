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
import os
import re
import selectors
import subprocess
import time
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable, Mapping, Protocol, Sequence

from adaptive_agent.constants import DEFAULT_MODEL_TOKENS


MODEL_PROVIDER = "openai-codex"
MODEL_NAME = "openai-codex/gpt-5.6-luna"


class PlannerError(RuntimeError):
    """A malformed model response or an execution boundary failure."""


class PlannerBudgetExceeded(PlannerError):
    """The parent-owned planner budget was exhausted."""


class PlannerCancelled(PlannerError):
    """The run was cancelled before the next model or kernel turn."""


class PlannerTimedOut(PlannerError):
    """The model call exceeded the parent-owned wall-clock deadline."""


class PlannerModelClient(Protocol):
    def invoke(
        self,
        *,
        goal: str,
        environment: Mapping[str, Any],
        messages: Sequence[Mapping[str, str]],
        remaining_deadline: float | None = None,
        cancel: Event | None = None,
        token_cap: int | None = None,
    ) -> Mapping[str, Any]: ...


class PlannerEvidenceSink(Protocol):
    def record_model_observation(self, evidence: Mapping[str, Any], *, trusted_parent: bool = False) -> Any: ...


class KernelExecutor(Protocol):
    def execute(self, code: str, *, timeout: float | None = None, cancel: Event | None = None) -> Any: ...


@dataclass(frozen=True)
class PlannerLimits:
    max_turns: int = 8
    max_wall_seconds: float = 900.0
    max_model_tokens: int = DEFAULT_MODEL_TOKENS
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


@dataclass(frozen=True)
class LunaInvocation:
    """Final answer and authenticated metadata for the control-plane runner."""

    text: str
    provider: str
    model: str
    response_id: str
    usage: Mapping[str, Any]


def _bounded_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return text[:limit]


def _bounded_usage(value: Any, *, max_keys: int = 32, max_value_chars: int = 256) -> Mapping[str, Any]:
    """Retain accounting fields without allowing provider metadata to grow unbounded."""
    if not isinstance(value, Mapping):
        return {}
    bounded: dict[str, Any] = {}
    for index, (key, item) in enumerate(value.items()):
        if index >= max_keys:
            break
        name = _bounded_text(key, 128)
        if isinstance(item, Mapping):
            bounded[name] = dict(_bounded_usage(item, max_keys=8, max_value_chars=max_value_chars))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            bounded[name] = item if not isinstance(item, str) else item[:max_value_chars]
        else:
            bounded[name] = _bounded_text(item, max_value_chars)
    return bounded


def _redact(value: Any) -> Any:
    """Remove common credential-shaped fields before model feedback."""
    if isinstance(value, Mapping):
        hidden = {"authorization", "cookie", "password", "secret", "token", "api_key", "apikey"}
        return {str(key): "[REDACTED]" if str(key).lower() in hidden else _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?i)\b(authorization|cookie|password|secret|token|api[_-]?key)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", value)
    return value


def _usage_tokens(usage: Mapping[str, Any]) -> int:
    for key in ("totalTokens", "total_tokens", "outputTokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    raise PlannerError("model usage must include a non-negative token count")


def _canonical_model(provider: Any, model: Any) -> tuple[str, str]:
    """Validate provider identity and normalize Prime's bare model name."""
    if provider != MODEL_PROVIDER or not isinstance(model, str):
        raise PlannerError("authenticated model response requires provider, model, response id, text, and usage")
    if model == "gpt-5.6-luna":
        model = MODEL_NAME
    if model != MODEL_NAME:
        raise PlannerError("authenticated model response requires provider, model, response id, text, and usage")
    return provider, model


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return ""
    return "".join(
        str(part.get("text", ""))
        for part in content
        if isinstance(part, Mapping) and part.get("type") == "text"
    )


class PrimeCliModelClient:
    """Invoke the authenticated Prime CLI through a bounded one-shot process.

    Prime's JSON mode emits a JSON-lines event stream rather than one response
    object.  The client selects the final assistant ``message_end`` event and
    returns the provider, response ID, text, and usage observed by the trusted
    parent.  No session, tools, extensions, skills, prompt templates, or
    context files are enabled for these calls.
    """

    def __init__(
        self,
        *,
        executable: str = "prime-agent",
        coding_agent_dir: str | os.PathLike[str] | None = None,
        cwd: str | os.PathLike[str] = "/private/tmp",
        thinking: str = "medium",
        max_output_chars: int = 65_536,
    ) -> None:
        self.executable = executable
        self.coding_agent_dir = os.fspath(coding_agent_dir) if coding_agent_dir is not None else None
        self.cwd = os.fspath(cwd)
        self.thinking = thinking
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        self.max_output_chars = max_output_chars

    def invoke(
        self,
        *,
        goal: str,
        environment: Mapping[str, Any],
        messages: Sequence[Mapping[str, str]],
        remaining_deadline: float | None = None,
        cancel: Event | None = None,
        token_cap: int | None = None,
    ) -> Mapping[str, Any]:
        if cancel is not None and cancel.is_set():
            raise PlannerCancelled("model call cancelled before launch")
        if remaining_deadline is not None and remaining_deadline <= 0:
            raise PlannerTimedOut("model call deadline expired before launch")
        coding_agent_dir = self.coding_agent_dir or os.environ.get("PRIME_AGENT_CODING_AGENT_DIR")
        if not coding_agent_dir:
            raise PlannerError("PRIME_AGENT_CODING_AGENT_DIR is required for the Prime CLI model client")
        history = json.dumps(
            {"goal": goal, "environment": environment, "messages": list(messages)},
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        if token_cap is not None:
            history = f"A hard parent token cap of {max(0, token_cap)} applies to this call.\n" + history
        command = [
            self.executable,
            "--print",
            "--mode", "json",
            "--no-tools",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-session",
            "--cwd", self.cwd,
            "--provider", MODEL_PROVIDER,
            "--model", MODEL_NAME,
            "--thinking", self.thinking,
            "--",
            history,
        ]
        # Keep the subprocess environment deliberately small.  Authentication
        # is loaded from the AO-authorized coding-agent directory; parent
        # provider keys and unrelated secrets must not cross this boundary.
        env = {
            key: value
            for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL")
            if (value := os.environ.get(key)) is not None
        }
        env["PRIME_AGENT_CODING_AGENT_DIR"] = coding_agent_dir
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                command,
                cwd=self.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            raise PlannerError(f"unable to launch Prime CLI: {exc}") from exc

        final: dict[str, Any] | None = None
        stderr_tail = bytearray()
        streams: dict[int, tuple[Any, bytearray]] = {}
        selector = selectors.DefaultSelector()
        if process.stdout is not None:
            os.set_blocking(process.stdout.fileno(), False)
            streams[process.stdout.fileno()] = (process.stdout, bytearray())
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        if process.stderr is not None:
            os.set_blocking(process.stderr.fileno(), False)
            streams[process.stderr.fileno()] = (process.stderr, bytearray())
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")

        def consume_stdout_line(line: bytes) -> None:
            nonlocal final
            try:
                event = json.loads(line.decode("utf-8", errors="replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            if not isinstance(event, Mapping) or event.get("type") != "message_end":
                return
            message = event.get("message")
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                return
            text = _message_text(message)
            if not text:
                return
            provider, model = _canonical_model(message.get("provider"), message.get("model"))
            response_id = message.get("responseId", message.get("response_id"))
            if not isinstance(response_id, str) or not response_id.strip():
                return
            final = {
                "provider": provider,
                "model": model,
                "responseId": _bounded_text(response_id, 512),
                "text": _bounded_text(text, self.max_output_chars),
                "usage": dict(_bounded_usage(message.get("usage"))),
            }

        def stop_process() -> None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream, _ in streams.values():
                stream.close()

        try:
            max_line_bytes = max(self.max_output_chars * 8, 1_048_576)
            while streams:
                if cancel is not None and cancel.is_set():
                    stop_process()
                    raise PlannerCancelled("model call cancelled")
                remaining = None if remaining_deadline is None else remaining_deadline - (time.monotonic() - started)
                if remaining is not None and remaining <= 0:
                    stop_process()
                    raise PlannerTimedOut("model call exceeded deadline")
                timeout = min(0.1, remaining) if remaining is not None else 0.1
                for selected, _ in selector.select(timeout):
                    stream = selected.fileobj
                    descriptor = stream.fileno()
                    try:
                        chunk = os.read(descriptor, 65_536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(stream)
                        stream_info = streams.pop(descriptor, None)
                        if stream_info is not None and selected.data == "stdout":
                            buffer = stream_info[1]
                            if buffer:
                                consume_stdout_line(bytes(buffer))
                        continue
                    stream_info = streams.get(descriptor)
                    if stream_info is None:
                        continue
                    buffer = stream_info[1]
                    if selected.data == "stderr":
                        stderr_tail.extend(chunk)
                        del stderr_tail[:-8_192]
                        continue
                    buffer.extend(chunk)
                    if len(buffer) > max_line_bytes:
                        # A pathological single event cannot consume the parent
                        # process. Discard that event and resume at its newline.
                        newline = buffer.rfind(b"\n")
                        if newline >= 0:
                            del buffer[: newline + 1]
                        else:
                            buffer.clear()
                    while True:
                        newline = buffer.find(b"\n")
                        if newline < 0:
                            break
                        line = bytes(buffer[:newline])
                        del buffer[: newline + 1]
                        consume_stdout_line(line)
            process.wait()
        except (PlannerCancelled, PlannerTimedOut):
            stop_process()
            raise
        if process.returncode != 0:
            detail = bytes(stderr_tail).decode("utf-8", errors="replace").strip()[-2_000:]
            raise PlannerError(f"Prime CLI exited with status {process.returncode}: {detail}")
        if final is None:
            raise PlannerError("Prime CLI JSON stream did not contain a final assistant message")
        return final

    @staticmethod
    def _parse_events(stdout: str) -> Mapping[str, Any]:
        final: Mapping[str, Any] | None = None
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, Mapping) or event.get("type") != "message_end":
                continue
            message = event.get("message")
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            text = _message_text(message)
            if text:
                provider, model = _canonical_model(message.get("provider"), message.get("model"))
                final = {"provider": provider, "model": model, "responseId": message.get("responseId"), "text": text, "usage": message.get("usage")}
        if final is None:
            raise PlannerError("Prime CLI JSON stream did not contain a final assistant message")
        return final


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

            try:
                raw = self._invoke(
                    goal,
                    environment,
                    self._bounded_history(messages),
                    remaining_deadline=remaining,
                    cancel=cancel,
                    token_cap=self.limits.max_model_tokens - model_tokens,
                )
            except PlannerCancelled:
                publish("status", "Planner cancelled during model turn.")
                return PlannerResult("cancelled", None, turn - 1, model_tokens, kernel_steps, tuple(response_ids), tuple(events))
            except PlannerTimedOut:
                publish("status", "Planner timed out during model turn.")
                return PlannerResult("timed_out", None, turn - 1, model_tokens, kernel_steps, tuple(response_ids), tuple(events))
            provider, model, response_id, text, usage = self._validate_response(raw)
            used = _usage_tokens(usage)
            model_tokens += used
            # Record the provider's actual usage before enforcing the parent
            # token cap.  Over-cap evidence is still needed for accounting.
            self.evidence_sink.record_model_observation({"provider": provider, "model": model, "responseId": response_id, "usage": dict(usage)}, trusted_parent=True)
            response_ids.append(response_id)
            if cancel and cancel.is_set():
                publish("status", "Planner cancelled after model turn.")
                return PlannerResult("cancelled", None, turn, model_tokens, kernel_steps, tuple(response_ids), tuple(events))
            if time.monotonic() - started >= self.limits.max_wall_seconds:
                publish("status", "Planner timed out after model turn.")
                return PlannerResult("timed_out", None, turn, model_tokens, kernel_steps, tuple(response_ids), tuple(events))
            if model_tokens > self.limits.max_model_tokens:
                publish("status", "Model token budget exceeded.")
                return PlannerResult("budget_exhausted", None, turn, model_tokens, kernel_steps, tuple(response_ids), tuple(events))
            publish("model", "Authenticated Luna response received.", response_id)
            action, value = _parse_action(text, self.limits.max_code_chars)
            messages.append({"role": "assistant", "content": text})
            if action == "finish":
                publish("status", "Planner finished.")
                return PlannerResult("succeeded", value, turn, model_tokens, kernel_steps, tuple(response_ids), tuple(events))

            assert value is not None
            remaining_after_model = self.limits.max_wall_seconds - (time.monotonic() - started)
            if cancel and cancel.is_set():
                publish("status", "Planner cancelled before kernel turn.")
                return PlannerResult("cancelled", None, turn, model_tokens, kernel_steps, tuple(response_ids), tuple(events))
            if remaining_after_model <= 0:
                publish("status", "Planner timed out before kernel turn.")
                return PlannerResult("timed_out", None, turn, model_tokens, kernel_steps, tuple(response_ids), tuple(events))
            kernel_steps += 1
            publish("execute", "Generated Python submitted to Prime kernel.")
            try:
                result = self.kernel.execute(value, timeout=min(remaining_after_model, self.limits.max_wall_seconds), cancel=cancel)
            except Exception as exc:
                # Policy and kernel boundary errors are model feedback.  Keep
                # the authenticated conversation alive so Luna can revise the
                # generated cell instead of converting a recoverable mistake
                # into a failed control-plane run.
                feedback = self._sanitize_execution(type("KernelError", (), {"status": "error", "error": str(exc), "result": None, "stdout": "", "stderr": ""})())
                messages.append({"role": "user", "content": "Prime execution feedback:\n" + feedback})
                publish("kernel", "Prime execution rejected; feedback returned to planner.", feedback)
                continue
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
        prompt = (
            "You are the authenticated Luna planner. Generate a generic solution for the supplied goal. "
            "Use public docs, tool schemas, policy, and active skills as contracts. "
            "Return exactly one JSON object: {\\\"action\\\":\\\"execute\\\",\\\"code\\\":\\\"...\\\"} "
            "for Python to run in the task-scoped Prime kernel, or {\\\"action\\\":\\\"finish\\\",\\\"answer\\\":\\\"...\\\"}. "
            "Use `from rlm import host_request` and await host_request(\"broker.call\", {\"capabilityId\": \"...\", \"arguments\": {...}}) for tools; do not import the rlm module itself. "
            "Never import credentials, access the host, or invent outcomes. "
            "A tool result or error is feedback for the next turn. The first-class `environment` request field is the authoritative task contract, including task context, public docs, capabilities, tool schemas, schema, and budget."
        )
        environment_skills = environment.get("activeSkills")
        explicit_skills = list(active_skills)
        if "activeSkills" not in environment:
            supplemental = explicit_skills
            label = "The caller supplied these active skills because the environment has none:"
        else:
            normalized_environment_skills = list(environment_skills) if isinstance(environment_skills, Sequence) and not isinstance(environment_skills, (str, bytes)) else environment_skills
            environment_keys = {
                json.dumps(skill, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
                for skill in normalized_environment_skills
            } if isinstance(normalized_environment_skills, list) else set()
            supplemental = []
            seen_keys = set(environment_keys)
            for skill in explicit_skills:
                skill_key = json.dumps(skill, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
                if skill_key not in seen_keys:
                    supplemental.append(skill)
                    seen_keys.add(skill_key)
            label = "These additional caller-provided active skills supplement the authoritative environment activeSkills:"
        if supplemental:
            skills = json.dumps(_redact({"activeSkills": supplemental}), ensure_ascii=False, separators=(",", ":"), default=str)
            available = self.limits.max_context_chars - len(prompt) - len(label) - 2
            if len(skills) > available:
                raise PlannerError("supplemental active skill context exceeds planner context limit")
            prompt += f" {label}\n{skills}"
        return prompt

    def _bounded_history(self, messages: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
        """Keep the system prompt and newest turns within the context cap."""
        if not messages:
            return []
        system = messages[0]
        selected: list[Mapping[str, str]] = [system]
        used = len(_bounded_text(system, self.limits.max_context_chars))
        for message in reversed(messages[1:]):
            encoded = _bounded_text(message, self.limits.max_context_chars)
            if used + len(encoded) > self.limits.max_context_chars:
                break
            selected.append(message)
            used += len(encoded)
        return [selected[0], *reversed(selected[1:])]

    def _invoke(
        self,
        goal: str,
        environment: Mapping[str, Any],
        messages: Sequence[Mapping[str, str]],
        *,
        remaining_deadline: float | None,
        cancel: Event | None,
        token_cap: int,
    ) -> Mapping[str, Any]:
        invoke = self.client.invoke
        parameters = inspect.signature(invoke).parameters
        accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        if "messages" in parameters or accepts_kwargs:
            kwargs: dict[str, Any] = {"goal": goal, "environment": environment, "messages": messages}
            if "remaining_deadline" in parameters or accepts_kwargs:
                kwargs["remaining_deadline"] = remaining_deadline
            if "cancel" in parameters or accepts_kwargs:
                kwargs["cancel"] = cancel
            if "token_cap" in parameters or accepts_kwargs:
                kwargs["token_cap"] = token_cap
            return invoke(**kwargs)
        # Compatibility for the existing parent adapter while it is being
        # upgraded to retain conversation messages.
        return invoke(goal=goal, environment={**environment, "plannerMessages": list(messages)})

    @staticmethod
    def _validate_response(raw: Mapping[str, Any]) -> tuple[str, str, str, str, Mapping[str, Any]]:
        if not isinstance(raw, Mapping):
            raise PlannerError("model response must be an object")
        provider, model = _canonical_model(raw.get("provider"), raw.get("model"))
        response_id = raw.get("responseId", raw.get("response_id"))
        text, usage = raw.get("text"), raw.get("usage")
        if not isinstance(response_id, str) or not response_id.strip() or not isinstance(text, str) or not text.strip() or not isinstance(usage, Mapping) or not usage:
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


def make_luna_model_runner(
    client: PlannerModelClient,
    kernel: KernelExecutor,
    evidence_sink: PlannerEvidenceSink,
    *,
    limits: PlannerLimits = PlannerLimits(),
) -> Callable[..., LunaInvocation]:
    """Adapt the multi-turn planner to ``ControlPlane.model_runner``.

    The kernel is supplied by the trusted parent, normally a Prime runtime
    whose host requests are wired to the authoritative broker.  This adapter
    never creates a local tool provider or accepts learner-side credentials.
    """
    def runner(*, goal: str, environment: Mapping[str, Any], emit: Callable[[str, str, str | None], None]) -> LunaInvocation:
        observed: dict[str, Any] = {}

        class RecordingClient:
            def invoke(self, **kwargs: Any) -> Mapping[str, Any]:
                invoke = client.invoke
                parameters = inspect.signature(invoke).parameters
                accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
                if accepts_kwargs:
                    raw = invoke(**kwargs)
                else:
                    forwarded = {key: value for key, value in kwargs.items() if key in parameters}
                    if "messages" not in parameters:
                        forwarded["environment"] = {**kwargs["environment"], "plannerMessages": list(kwargs["messages"])}
                        forwarded.pop("messages", None)
                    raw = invoke(**forwarded)
                if isinstance(raw, Mapping):
                    observed.update(raw)
                    if raw.get("model") == "gpt-5.6-luna":
                        observed["model"] = MODEL_NAME
                return raw

        try:
            result = LunaPlanner(
                RecordingClient(),
                kernel,
                evidence_sink,
                limits=limits,
                emit=lambda event: emit(event.kind, event.summary, event.detail),
            ).run(goal=goal, environment=environment)
        except Exception as exc:
            # A kernel/evaluator failure must not erase the provider usage that
            # was already observed by the trusted parent.
            if observed:
                setattr(exc, "model_observation", {
                    "provider": observed.get("provider"),
                    "model": observed.get("model"),
                    "responseId": observed.get("responseId", observed.get("response_id")),
                    "usage": dict(_bounded_usage(observed.get("usage"))),
                })
            raise
        if result.status != "succeeded":
            raise PlannerError(f"planner ended with status {result.status}")
        provider, model, response_id, usage, answer = observed.get("provider"), observed.get("model"), observed.get("responseId", observed.get("response_id")), observed.get("usage"), result.answer
        if not isinstance(provider, str) or not isinstance(model, str) or not isinstance(response_id, str) or not isinstance(usage, Mapping) or not isinstance(answer, str):
            raise PlannerError("planner did not retain authenticated final response metadata")
        return LunaInvocation(answer, provider, model, response_id, dict(usage))

    return runner


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


__all__ = ["KernelExecutor", "LunaInvocation", "LunaPlanner", "MODEL_NAME", "MODEL_PROVIDER", "PrimeCliModelClient", "PlannerError", "PlannerEvent", "PlannerEvidenceSink", "PlannerLimits", "PlannerModelClient", "PlannerResult", "PlannerBudgetExceeded", "PlannerCancelled", "PlannerTimedOut", "main", "make_luna_model_runner"]
