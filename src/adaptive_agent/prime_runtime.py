"""Prime Agent runtime adapter with a task-scoped execution boundary.

The adapter talks to the installed Prime Agent Python kernel (`rlm.repl`) over
its documented JSON-lines protocol. Every learner task runs in a Docker
container with no host mounts or network. Model configuration and observed
parent model evidence are kept separate; this module never treats requested
configuration as an inference result.
"""
from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


class AdapterError(RuntimeError):
    """A runtime, protocol, or policy failure."""


class SecurityViolation(AdapterError):
    """Learner code attempted an operation outside the execution policy."""


class ExecutionMode(str, Enum):
    PRIME_SUBSCRIPTION = "prime_subscription_kernel"
    PRIME_SUBSCRIPTION_DOCKER = "prime_subscription_kernel_docker"


@dataclass(frozen=True)
class Capability:
    id: str
    tool: str
    version: str
    effect: str
    resource_scope: str
    expires_at: str

    def public(self) -> dict[str, str]:
        return {
            "id": self.id, "tool": self.tool, "version": self.version,
            "effect": self.effect, "resourceScope": self.resource_scope,
            "expiresAt": self.expires_at,
        }


@dataclass(frozen=True)
class CapabilitySet:
    run_id: str
    capabilities: tuple[Capability, ...]
    provenance: str

    def public(self) -> dict[str, Any]:
        return {
            "runId": self.run_id,
            "capabilities": [c.public() for c in self.capabilities],
            "provenance": {"id": "capabilities", "version": self.provenance,
                           "sha256": self.provenance},
        }


class CapabilityBroker:
    """Narrow capability handoff to the authoritative control-plane broker.

    This adapter owns no credentials, schema, approval, expiry, resource-scope,
    or idempotency policy.  ``authorizer`` is supplied by Devin's trusted
    broker and is the only production path to a side effect.  A local handler
    is retained solely for read-only integration fixtures.
    """

    def __init__(self, run_id: str, capabilities: list[Capability] | None = None,
                 authorizer: Callable[[Capability, Mapping[str, Any]], Any] | None = None):
        self.run_id = run_id
        self._caps = {c.id: c for c in (capabilities or [])}
        self._handlers: dict[str, Callable[[Mapping[str, Any]], Any]] = {}
        self._authorizer = authorizer
        self._lock = threading.RLock()

    def register(self, capability: Capability,
                 handler: Callable[[Mapping[str, Any]], Any] | None = None) -> None:
        if capability.id in self._caps:
            raise AdapterError(f"duplicate capability: {capability.id}")
        self._caps[capability.id] = capability
        if handler is not None:
            self._handlers[capability.id] = handler

    def discover(self) -> CapabilitySet:
        payload = json.dumps([c.public() for c in self._caps.values()], sort_keys=True,
                             separators=(",", ":")).encode()
        digest = hashlib.sha256(payload).hexdigest()
        return CapabilitySet(self.run_id, tuple(self._caps.values()), digest)

    def call(self, capability_id: str, arguments: Mapping[str, Any]) -> Any:
        with self._lock:
            cap = self._caps.get(capability_id)
            handler = self._handlers.get(capability_id)
        if cap is None:
            raise SecurityViolation("capability is not granted")
        if not isinstance(arguments, Mapping):
            raise AdapterError("capability arguments must be an object")
        if self._authorizer is not None:
            # The authoritative broker validates expiry, scope, schemas,
            # approvals, and idempotency. Do not reproduce those checks here.
            return self._authorizer(cap, arguments)
        if handler is None:
            raise SecurityViolation("authoritative broker is not configured")
        return handler(arguments)


@dataclass(frozen=True)
class ArtifactRef:
    id: str
    version: str
    sha256: str
    bytes: int
    path: str


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    cell_id: str
    status: str
    result: str | None
    stdout: str
    stderr: str
    error: dict[str, Any] | None
    duration_ms: int
    mode: ExecutionMode
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ModelObservation:
    """Evidence supplied by the trusted parent after a real model response."""

    provider: str
    model: str
    response_id: str
    usage: Mapping[str, Any]
    observed_at: float

    def public(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model,
                "responseId": self.response_id, "usage": dict(self.usage),
                "observedAt": self.observed_at}


@dataclass(frozen=True)
class ChildPlanRequest:
    """Trusted-parent input to the child planner; it contains no credentials."""

    prompt: str
    kwargs: Mapping[str, Any]
    parent_run_id: str
    depth: int
    remaining_seconds: float
    capabilities: tuple[str, ...]
    cancel_event: threading.Event
    budget: "ChildPlannerBudget"


@dataclass(frozen=True)
class ChildPlan:
    """Code selected by the trusted parent model for one isolated child."""

    code: str
    name: str = "child"
    model: str = "openai-codex/gpt-5.6-luna"


@dataclass
class SharedBudget:
    """One decreasing ledger shared by parent and all child adapters."""

    max_wall_seconds: float
    max_artifact_bytes: int
    max_artifact_count: int
    max_child_runs: int = 0
    max_model_tokens: int | None = None
    started_at: float = field(default_factory=time.monotonic)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    artifact_bytes: int = 0
    artifact_count: int = 0
    artifact_refs: dict[tuple[str, str, str], ArtifactRef] = field(default_factory=dict)
    child_runs_used: int = 0
    model_tokens_used: int = 0

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_wall_seconds - (time.monotonic() - self.started_at))

    def reserve_child(self) -> None:
        with self.lock:
            if self.cancel_event.is_set() or self.remaining_seconds() <= 0 or self.child_runs_used >= self.max_child_runs:
                raise SecurityViolation("shared child budget is exhausted or cancelled")
            self.child_runs_used += 1

    def record_model_usage(self, tokens: int) -> None:
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            raise AdapterError("model token usage must be a non-negative integer")
        with self.lock:
            if self.max_model_tokens is not None and self.model_tokens_used + tokens > self.max_model_tokens:
                raise SecurityViolation("shared model token budget exhausted")
            self.model_tokens_used += tokens

    def cancel(self) -> None:
        self.cancel_event.set()


@dataclass(frozen=True)
class ChildPlannerBudget:
    """Trusted planner seam for cancellation and parent-owned accounting."""

    ledger: SharedBudget

    @property
    def cancel_event(self) -> threading.Event:
        return self.ledger.cancel_event

    @property
    def remaining_seconds(self) -> float:
        return self.ledger.remaining_seconds()

    @property
    def remaining_model_tokens(self) -> int | None:
        if self.ledger.max_model_tokens is None:
            return None
        return max(0, self.ledger.max_model_tokens - self.ledger.model_tokens_used)

    def record_model_usage(self, tokens: int) -> None:
        self.ledger.record_model_usage(tokens)


ChildPlanner = Callable[[ChildPlanRequest], ChildPlan | Mapping[str, Any] | str]


@dataclass
class PrimeRuntimeConfig:
    task_id: str
    root_dir: Path | str | None = None
    model: str = "openai-codex/gpt-5.6-luna"
    provider: str = "openai-codex"
    python: str | None = None
    runtime_src: Path | str | None = None
    max_output_chars: int = 65536
    max_protocol_frame_bytes: int = 1024 * 1024
    max_pending_events: int = 128
    max_cell_seconds: float = 30.0
    max_memory_bytes: int = 512 * 1024 * 1024
    max_cpu_seconds: int = 30
    max_processes: int = 8
    max_artifact_bytes: int = 8 * 1024 * 1024
    max_total_artifact_bytes: int = 64 * 1024 * 1024
    max_artifact_count: int = 128
    max_total_wall_seconds: float = 300.0
    max_model_tokens: int | None = None
    max_child_depth: int = 1
    child_runs: int = 0
    require_docker: bool = True
    docker_image: str | None = None
    # Supplied by the trusted parent/AO session, never by learner code.
    ao_session_id: str | None = None


# This is deliberately narrower than Python's full library surface.  A task
# can still use expressions and pure computation; side effects must go through
# the trusted broker.
_ALLOWED_IMPORTS = frozenset({"math", "statistics", "json", "re", "base64", "datetime", "decimal", "itertools", "functools", "collections"})
_BLOCKED_CALLS = frozenset({"open", "eval", "exec", "compile", "__import__", "input", "breakpoint", "getattr", "globals", "locals", "vars", "dir"})
_BLOCKED_NAMES = frozenset({"os", "sys", "subprocess", "socket", "pathlib", "shutil", "resource", "ctypes", "signal", "builtins", "__builtins__", "importlib"})


class _CellPolicy(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in _ALLOWED_IMPORTS:
                self.violations.append(f"import {alias.name} is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # The only learner-facing Prime surface is the broker bridge.  In
        # particular, do not expose rlm.harness or rlm.repl to learner code.
        if node.module == "rlm" and all(alias.name == "host_request" for alias in node.names):
            return
        root = (node.module or "").split(".", 1)[0]
        if root not in _ALLOWED_IMPORTS:
            self.violations.append(f"from {node.module} import ... is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_CALLS:
            self.violations.append(f"call {node.func.id} is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_NAMES:
            self.violations.append(f"name {node.id} is not available")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in {"__subclasses__", "__globals__", "__code__", "__builtins__", "__loader__"}:
            self.violations.append(f"attribute {node.attr} is not available")
        self.generic_visit(node)


def _validate_cell(code: str) -> None:
    if not isinstance(code, str) or not code.strip():
        raise AdapterError("cell code must be a non-empty string")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise AdapterError(f"invalid cell: {exc}") from exc
    policy = _CellPolicy()
    policy.visit(tree)
    if policy.violations:
        raise SecurityViolation("; ".join(policy.violations))


def _find_runtime_src(explicit: Path | str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("PRIME_AGENT_RUNTIME_SRC")
    if env:
        candidates.append(Path(env))
    for exe in (shutil.which("prime-agent"), "/opt/homebrew/bin/prime-agent"):
        if exe:
            # The CLI lives beside dist/; source runtime is bundled under dist.
            candidates.append(Path(exe).resolve().parent.parent / "lib/node_modules/prime-agent/dist/prime-agent-runtime/src")
            candidates.append(Path("/opt/homebrew/lib/node_modules/prime-agent/dist/prime-agent-runtime/src"))
    candidates.append(Path("/opt/homebrew/lib/node_modules/prime-agent/dist/prime-agent-runtime/src"))
    for candidate in candidates:
        if (candidate / "rlm/repl.py").is_file():
            return candidate
    raise AdapterError("installed Prime Agent rlm.repl runtime was not found")



_IMAGE_CACHE: dict[str, str] = {}
_IMAGE_LOCK = threading.Lock()


def _runtime_image(runtime_src: Path, configured: str | None) -> str:
    """Build a minimal image containing only the verified Prime runtime source."""
    if configured:
        # A configured image is accepted only by digest or a local tag supplied
        # by the trusted deployment. It is never learner-controlled.
        image = configured
        check = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
        if check.returncode:
            raise AdapterError(f"configured Prime runtime image is unavailable: {image}")
        return image
    digest = hashlib.sha256()
    for path in sorted((runtime_src / "rlm").rglob("*.py")):
        digest.update(path.relative_to(runtime_src).as_posix().encode())
        digest.update(path.read_bytes())
    tag = "adaptive-prime-runtime:" + digest.hexdigest()[:20]
    with _IMAGE_LOCK:
        if tag in _IMAGE_CACHE:
            return _IMAGE_CACHE[tag]
        if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0:
            _IMAGE_CACHE[digest.hexdigest()] = tag
            return tag
        with tempfile.TemporaryDirectory(prefix="adaptive-prime-image-") as context:
            context_path = Path(context)
            shutil.copytree(runtime_src / "rlm", context_path / "rlm")
            (context_path / "Dockerfile").write_text(
                "FROM python:3.11-slim\n"
                "COPY rlm /opt/prime-runtime/rlm\n"
                "ENV PYTHONPATH=/opt/prime-runtime PYTHONUNBUFFERED=1\n"
                "ENTRYPOINT [\"python\", \"-m\", \"rlm.repl\"]\n"
            )
            build = subprocess.run(
                ["docker", "build", "--pull", "--tag", tag, str(context_path)],
                capture_output=True, text=True, timeout=300,
            )
            if build.returncode:
                raise AdapterError("could not build the Prime runtime image: " + (build.stderr[-2000:] or build.stdout[-2000:]))
        _IMAGE_CACHE[digest.hexdigest()] = tag
        return tag


class _KernelProcess:
    def __init__(self, adapter: "PrimeRuntimeAdapter") -> None:
        self.adapter = adapter
        self.proc: subprocess.Popen[bytes] | None = None
        self.events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=adapter.config.max_pending_events)
        self.container_name = ""
        self.write_lock = threading.Lock()
        self.reader: threading.Thread | None = None
        self.stderr_reader: threading.Thread | None = None
        self.ready = threading.Event()
        self.start_error: Exception | None = None
        self.stderr_chunks: list[str] = []

    def start(self) -> None:
        config = self.adapter.config
        runtime_src = _find_runtime_src(config.runtime_src)
        image = _runtime_image(runtime_src, config.docker_image)
        memory = max(16, config.max_memory_bytes // (1024 * 1024))
        pids = max(4, config.max_processes)
        safe_task = re.sub(r"[^A-Za-z0-9_.-]", "-", config.task_id)[:32] or "task"
        self.container_name = f"adaptive-prime-{safe_task}-{uuid.uuid4().hex[:16]}"
        command = [
            "docker", "run", "--rm", "-i", "--init", "--name", self.container_name,
            "--network=none", "--read-only",
            "--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size={memory}m",
            "--cap-drop=ALL", "--security-opt", "no-new-privileges:true",
            "--pids-limit", str(pids), "--memory", str(config.max_memory_bytes),
            "--cpus", "0.5", "--ulimit", "nofile=64:64", "--ulimit", "fsize=16777216:16777216",
            "--user", "65534:65534", "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
        ]
        command.extend(["--label", self.adapter.cleanup_label])
        command.append(image)
        try:
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, close_fds=True)
        except OSError as exc:
            raise AdapterError(f"could not start Docker Prime kernel: {exc}") from exc
        self.reader = threading.Thread(target=self._read_events, daemon=True)
        self.reader.start()
        self.stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self.stderr_reader.start()
        deadline = time.monotonic() + 20
        while not self.ready.wait(0.05):
            if self.start_error:
                raise AdapterError(str(self.start_error))
            if self.proc.poll() is not None:
                raise AdapterError(f"Docker Prime kernel exited during startup ({self.proc.returncode})")
            if time.monotonic() > deadline:
                self.kill()
                raise AdapterError("Docker Prime kernel ready handshake timed out")

    def _owned_container_cleanup(self) -> None:
        """Stop/remove only this adapter's uniquely named container."""
        if not self.container_name:
            return
        try:
            subprocess.run(
                ["docker", "rm", "-f", self.container_name],
                capture_output=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _protocol_failure(self, message: str) -> None:
        """Fail closed on an oversized/corrupt frame or pending-event flood."""
        error = AdapterError(message)
        self.start_error = error
        self._owned_container_cleanup()
        # The reader itself owns the protocol pipe; also terminate the Docker
        # CLI so no orphaned host process survives a frame violation.
        self.kill()
        fatal = {"event": "fatal", "error": message}
        try:
            self.events.put_nowait(fatal)
        except queue.Full:
            # Retain the control failure rather than allowing unbounded memory.
            try:
                self.events.get_nowait()
                self.events.put_nowait(fatal)
            except queue.Empty:
                pass

    def _read_events(self) -> None:
        assert self.proc and self.proc.stdout
        limit = self.adapter.config.max_protocol_frame_bytes
        while True:
            raw = self.proc.stdout.readline(limit + 1)
            if not raw:
                break
            if len(raw) > limit or not raw.endswith(b"\n"):
                self._protocol_failure("Prime kernel protocol frame exceeded its byte limit")
                break
            try:
                event = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._protocol_failure(f"invalid Prime kernel frame: {exc}")
                break
            try:
                event = self._bound_event(event)
            except AdapterError as exc:
                self._protocol_failure(str(exc))
                break
            if event.get("event") == "ready":
                self.ready.set()
            elif event.get("event") == "host_request":
                self._handle_host_request(event)
            else:
                try:
                    self.events.put_nowait(event)
                except queue.Full:
                    self._protocol_failure("Prime kernel pending event limit exceeded")
                    break
        self.ready.set()

    def _bound_event(self, event: Any) -> dict[str, Any]:
        if not isinstance(event, dict):
            raise AdapterError("Prime kernel event must be an object")
        limit = self.adapter.config.max_output_chars
        kind = event.get("event")
        bounded = dict(event)
        if kind in {"stdout", "stderr", "result"} and isinstance(bounded.get("text"), str):
            bounded["text"] = bounded["text"][:limit]
        if kind == "error":
            for key in ("ename", "evalue"):
                if isinstance(bounded.get(key), str):
                    bounded[key] = bounded[key][:limit]
            trace = bounded.get("traceback")
            if isinstance(trace, list):
                remaining = limit
                bounded_trace = []
                for line in trace:
                    if not isinstance(line, str) or remaining <= 0:
                        break
                    clipped = line[:remaining]
                    bounded_trace.append(clipped)
                    remaining -= len(clipped)
                bounded["traceback"] = bounded_trace
        return bounded

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        limit = self.adapter.config.max_protocol_frame_bytes
        retained = 0
        while True:
            raw = self.proc.stderr.readline(limit + 1)
            if not raw:
                break
            if len(raw) > limit or not raw.endswith(b"\n"):
                self._protocol_failure("Prime kernel stderr exceeded its byte limit")
                break
            text = raw.decode("utf-8", "replace")
            if retained < self.adapter.config.max_output_chars:
                text = text[: self.adapter.config.max_output_chars - retained]
                self.stderr_chunks.append(text)
                retained += len(text)

    def _handle_host_request(self, event: dict[str, Any]) -> None:
        request_id = event.get("id")
        payload = event.get("data")
        try:
            if not isinstance(request_id, str) or not isinstance(payload, dict):
                raise AdapterError("malformed host request")
            result = self.adapter.handle_host_request(payload)
            reply = {"status": "ok", "result": result}
        except Exception as exc:  # host errors become learner-visible, not process failures
            reply = {"status": "error", "error": str(exc)}
        self.send({"type": "host_reply", "id": request_id, "data": reply})

    def send(self, payload: dict[str, Any]) -> None:
        if not self.proc or not self.proc.stdin:
            raise AdapterError("Prime kernel is not running")
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        if len(data) > self.adapter.config.max_protocol_frame_bytes:
            self._protocol_failure("outbound Prime kernel protocol frame exceeded its byte limit")
            raise AdapterError("outbound Prime kernel protocol frame exceeded its byte limit")
        with self.write_lock:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    @staticmethod
    def _append_output(chunks: list[str], text: str, limit: int) -> None:
        used = sum(len(chunk) for chunk in chunks)
        if used >= limit:
            return
        chunks.append(text[: max(0, limit - used)])

    def execute(self, cell_id: str, code: str, timeout: float, max_output_chars: int = 65536) -> dict[str, Any]:
        self.send({"type": "execute", "id": cell_id, "code": code})
        collected: dict[str, Any] = {"stdout": [], "stderr": [], "result": None, "error": None}
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(0.01, deadline - time.monotonic())
            if remaining <= 0.01 and time.monotonic() >= deadline:
                return self._interrupt_and_collect(cell_id)
            try:
                event = self.events.get(timeout=remaining)
            except queue.Empty:
                return self._interrupt_and_collect(cell_id)
            if event.get("id") not in (cell_id, None):
                continue
            kind = event.get("event")
            if kind == "stdout":
                self._append_output(collected["stdout"], str(event.get("text", "")), max_output_chars)
            elif kind == "stderr":
                self._append_output(collected["stderr"], str(event.get("text", "")), max_output_chars)
            elif kind == "result":
                collected["result"] = str(event.get("text", ""))
            elif kind == "error":
                collected["error"] = {k: event.get(k) for k in ("ename", "evalue", "traceback")}
            elif kind == "fatal":
                raise AdapterError(str(event.get("error", "Prime kernel protocol failure")))
            elif kind == "done":
                collected["status"] = event.get("status", "error")
                return collected

    def _interrupt_and_collect(self, cell_id: str) -> dict[str, Any]:
        self.cancel(cell_id)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            try:
                event = self.events.get(timeout=0.05)
            except queue.Empty:
                continue
            if event.get("id") not in (cell_id, None):
                continue
            if event.get("event") == "done":
                return {"status": "aborted", "stdout": [], "stderr": [], "result": None,
                        "error": {"ename": "TimeoutError", "evalue": "cell timeout", "traceback": []}}
        self.kill()
        raise TimeoutError(f"cell {cell_id} exceeded configured timeout")

    def cancel(self, cell_id: str | None = None) -> None:
        try:
            self.send({"type": "interrupt", **({"id": cell_id} if cell_id else {})})
        except AdapterError:
            return

    def shutdown(self) -> None:
        proc = self.proc
        if not proc:
            self._owned_container_cleanup()
            return
        try:
            if self.proc is proc:
                self.send({"type": "shutdown", "id": uuid.uuid4().hex})
            proc.wait(timeout=2)
        except Exception:
            self._owned_container_cleanup()
            try:
                if os.name != "nt":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        finally:
            self._owned_container_cleanup()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except (AttributeError, OSError):
                    pass
            if self.proc is proc:
                self.proc = None

    def kill(self) -> None:
        proc = self.proc
        self._owned_container_cleanup()
        if not proc:
            return
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass
        finally:
            self._owned_container_cleanup()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except (AttributeError, OSError):
                    pass
            if self.proc is proc:
                self.proc = None


class PrimeRuntimeAdapter:
    """One persistent, task-scoped Prime Python kernel."""

    def __init__(self, config: PrimeRuntimeConfig, broker: CapabilityBroker | None = None,
                 child_planner: ChildPlanner | None = None, *,
                 _shared_budget: SharedBudget | None = None,
                 _artifact_store_root: Path | None = None, _depth: int = 0):
        if config.provider != "openai-codex" or config.model != "openai-codex/gpt-5.6-luna":
            raise AdapterError("only the authenticated openai-codex/gpt-5.6-luna path is supported; Prime Inference is disabled")
        if not config.require_docker:
            raise AdapterError("host execution is forbidden; Docker isolation is mandatory")
        if shutil.which("docker") is None:
            raise AdapterError("Docker CLI is required for the Prime learner boundary")
        ao_session_id = config.ao_session_id or os.environ.get("AO_SESSION_ID")
        if not ao_session_id or not re.fullmatch(r"[A-Za-z0-9_.:-]+", ao_session_id):
            raise AdapterError("trusted AO_SESSION_ID is required for Docker cleanup")
        self.config = config
        self._ao_session_id = ao_session_id
        self.root = Path(config.root_dir) if config.root_dir else Path(tempfile.mkdtemp(prefix=f"adaptive-{config.task_id}-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.broker = broker or CapabilityBroker(config.task_id)
        self.child_planner = child_planner
        self._depth = _depth
        if _depth > config.max_child_depth:
            raise AdapterError("child depth exceeds configured maximum")
        self._budget = _shared_budget or SharedBudget(
            config.max_total_wall_seconds, config.max_total_artifact_bytes, config.max_artifact_count,
            config.child_runs, config.max_model_tokens)
        self._artifact_store_root = _artifact_store_root or (self.root / "artifacts")
        self.kernel: _KernelProcess | None = None
        self._lock = threading.RLock()
        self._children = 0
        self._closed = False
        self._started_at = time.time()
        self._model_observation: ModelObservation | None = None
        self._artifact_transfers: dict[str, dict[str, Any]] = {}
        self._mode = ExecutionMode.PRIME_SUBSCRIPTION_DOCKER
        self._runtime_src = _find_runtime_src(config.runtime_src)

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    @property
    def cleanup_label(self) -> str:
        """Trusted AO label propagated to every learner Docker container."""
        return f"ao.session={self._ao_session_id}"

    def provenance(self) -> dict[str, Any]:
        return {
            "adapter": "adaptive-agent.prime-runtime",
            "mode": self._mode.value,
            "requestedProvider": self.config.provider,
            "requestedModel": self.config.model,
            "observedModelInvocation": self._model_observation.public() if self._model_observation else None,
            "kernel": "Prime Agent rlm.repl protocol v3",
            "runtimeSource": str(self._runtime_src),
            "isolation": "per-run Docker container; nonroot, read-only root, tmpfs /tmp, dropped capabilities, no-new-privileges, pids/memory/cpu limits",
            "cleanupLabel": f"ao.session={self._ao_session_id}",
            "network": "none inside learner container; broker bridge uses the parent stdio protocol",
            "credentials": "not inherited by learner process",
            "limits": {"cellSeconds": self.config.max_cell_seconds, "memoryBytes": self.config.max_memory_bytes,
                       "cpuSeconds": self.config.max_cpu_seconds, "processes": self.config.max_processes},
            "limitations": "Docker daemon/image availability is an environment dependency; authenticated model calls stay in the trusted parent, never in the learner container",
        }

    def record_model_observation(self, evidence: Mapping[str, Any], *, trusted_parent: bool = False) -> ModelObservation:
        """Record parent-owned response/usage evidence, never learner claims."""
        if not trusted_parent:
            raise SecurityViolation("only the trusted parent may record model evidence")
        if not isinstance(evidence, Mapping):
            raise AdapterError("model evidence must be an object")
        provider = evidence.get("provider")
        model = evidence.get("model")
        response_id = evidence.get("responseId") or evidence.get("response_id")
        usage = evidence.get("usage")
        if provider != self.config.provider or model != self.config.model:
            raise AdapterError("model evidence does not match requested subscription path")
        if not isinstance(response_id, str) or not response_id.strip() or not isinstance(usage, Mapping) or not usage:
            raise AdapterError("model evidence requires a response id and non-empty usage")
        observation = ModelObservation(provider, model, response_id, dict(usage), time.time())
        self._model_observation = observation
        return observation

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise AdapterError("adapter is closed")
            if self.kernel and self.kernel.proc and self.kernel.proc.poll() is None:
                return
            self.kernel = _KernelProcess(self)
            try:
                self.kernel.start()
            except Exception:
                self.kernel.kill()
                self.kernel = None
                raise

    def execute(self, code: str, *, timeout: float | None = None, cancel: threading.Event | None = None) -> ExecutionResult:
        _validate_cell(code)
        self.start()
        assert self.kernel
        cell_id = uuid.uuid4().hex
        started = time.monotonic()
        requested_timeout = timeout if timeout is not None else self.config.max_cell_seconds
        effective_timeout = min(requested_timeout, self._budget.remaining_seconds())
        if effective_timeout <= 0 or self._budget.cancel_event.is_set() or (cancel and cancel.is_set()):
            self.cancel(cell_id)
            return ExecutionResult(self.config.task_id, cell_id, "aborted", None, "", "", None, 0, self.mode, self.provenance())
        stop_watcher = threading.Event()
        if cancel is not None:
            def watch_cancel() -> None:
                while not stop_watcher.wait(0.02):
                    if cancel.is_set() or self._budget.cancel_event.is_set():
                        self.cancel(cell_id)
                        return
            threading.Thread(target=watch_cancel, daemon=True).start()
        try:
            raw = self.kernel.execute(cell_id, code, effective_timeout, self.config.max_output_chars)
            status = raw.get("status", "error")
            if cancel is not None and cancel.is_set() and status != "ok":
                status = "aborted"
            if status not in {"ok", "error", "aborted"}:
                status = "error"
            return ExecutionResult(self.config.task_id, cell_id, status, raw.get("result"),
                                   "".join(raw.get("stdout", [])),
                                   "".join(raw.get("stderr", [])),
                                   raw.get("error"), int((time.monotonic() - started) * 1000), self.mode, self.provenance())
        except TimeoutError:
            return ExecutionResult(self.config.task_id, cell_id, "aborted", None, "", "", {"ename": "TimeoutError", "evalue": "cell timeout"},
                                   int((time.monotonic() - started) * 1000), self.mode, self.provenance())
        finally:
            stop_watcher.set()

    def _execute_child_reserved(self, code: str, *, timeout: float | None = None) -> ExecutionResult:
        with self._lock:
            self._children += 1
            child_number = self._children
        child_id = f"{self.config.task_id}-child-{child_number}"
        child_root = self.root / "children" / child_id
        child = PrimeRuntimeAdapter(
            PrimeRuntimeConfig(
                task_id=child_id, root_dir=child_root, model=self.config.model,
                provider=self.config.provider, python=self.config.python,
                runtime_src=self.config.runtime_src, max_output_chars=self.config.max_output_chars,
                max_protocol_frame_bytes=self.config.max_protocol_frame_bytes,
                max_pending_events=self.config.max_pending_events,
                max_cell_seconds=min(timeout or self.config.max_cell_seconds, self.config.max_cell_seconds),
                max_memory_bytes=self.config.max_memory_bytes, max_cpu_seconds=self.config.max_cpu_seconds,
                max_processes=self.config.max_processes, max_artifact_bytes=self.config.max_artifact_bytes,
                max_total_artifact_bytes=self.config.max_total_artifact_bytes,
                max_artifact_count=self.config.max_artifact_count,
                max_total_wall_seconds=self.config.max_total_wall_seconds,
                max_model_tokens=self.config.max_model_tokens,
                max_child_depth=self.config.max_child_depth,
                child_runs=0, require_docker=True, docker_image=self.config.docker_image,
                ao_session_id=self.config.ao_session_id,
            ),
            broker=CapabilityBroker(child_id),
            _shared_budget=self._budget,
            _artifact_store_root=self._artifact_store_root,
            _depth=self._depth + 1,
        )
        try:
            result = child.execute(code, timeout=timeout, cancel=self._budget.cancel_event)
            result.provenance["parentRunId"] = self.config.task_id
            result.provenance["childDepth"] = self._depth + 1
            result.provenance["capabilities"] = []
            return result
        finally:
            child.close(remove_workspace=True)

    def execute_child(self, code: str, *, timeout: float | None = None) -> ExecutionResult:
        """Run one bounded child in a fresh Docker kernel with attenuated context."""
        if self._depth >= self.config.max_child_depth:
            raise SecurityViolation("child depth budget exhausted")
        self._budget.reserve_child()
        return self._execute_child_reserved(code, timeout=timeout)

    def cancel(self, cell_id: str | None = None) -> None:
        self._budget.cancel()
        with self._lock:
            if self.kernel:
                self.kernel.cancel(cell_id)

    @staticmethod
    def _sanitize_child_kwargs(value: Any, depth: int = 0) -> Any:
        if depth > 4:
            raise SecurityViolation("child planner kwargs are too deeply nested")
        if isinstance(value, Mapping):
            clean = {}
            for key, item in value.items():
                if not isinstance(key, str) or len(key) > 128:
                    raise SecurityViolation("child planner kwargs contain an invalid key")
                if any(word in key.lower() for word in ("credential", "secret", "password", "token", "api_key")):
                    raise SecurityViolation("credentials are not allowed in child planner kwargs")
                clean[key] = PrimeRuntimeAdapter._sanitize_child_kwargs(item, depth + 1)
            return clean
        if isinstance(value, list):
            if len(value) > 128:
                raise SecurityViolation("child planner kwargs list is too large")
            return [PrimeRuntimeAdapter._sanitize_child_kwargs(item, depth + 1) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            if isinstance(value, str) and len(value) > 8192:
                raise SecurityViolation("child planner kwargs string is too large")
            return value
        raise SecurityViolation("child planner kwargs must be JSON values")

    def _plan_child(self, payload: Mapping[str, Any]) -> tuple[ChildPlan, ChildPlanRequest]:
        if self.child_planner is None:
            raise SecurityViolation("trusted child planner is not configured")
        prompt = payload.get("prompt")
        kwargs = payload.get("kwargs", {})
        if not isinstance(prompt, str) or not prompt or len(prompt) > 16384:
            raise AdapterError("child prompt must be a bounded non-empty string")
        clean_kwargs = self._sanitize_child_kwargs(kwargs)
        planner_budget = ChildPlannerBudget(self._budget)
        request = ChildPlanRequest(prompt, clean_kwargs, self.config.task_id, self._depth,
                                   planner_budget.remaining_seconds, (),
                                   planner_budget.cancel_event, planner_budget)
        planned = self.child_planner(request)
        if isinstance(planned, ChildPlan):
            plan = planned
        elif isinstance(planned, str):
            plan = ChildPlan(planned)
        elif isinstance(planned, Mapping):
            plan = ChildPlan(str(planned.get("code", "")), str(planned.get("name", "child")),
                             str(planned.get("model", self.config.model)))
        else:
            raise AdapterError("child planner returned an invalid plan")
        if not isinstance(plan.code, str) or not plan.code.strip() or len(plan.code) > 65536:
            raise AdapterError("child planner code is invalid or too large")
        if plan.model != self.config.model:
            raise AdapterError("child planner model does not match the pinned model")
        return plan, request

    def handle_host_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request_type = payload.get("type")
        if request_type == "capabilities.discover":
            return self.broker.discover().public()
        if request_type == "broker.call":
            capability_id = payload.get("capabilityId")
            arguments = payload.get("arguments", {})
            if not isinstance(capability_id, str):
                raise AdapterError("capabilityId is required")
            return {"value": self.broker.call(capability_id, arguments)}
        if request_type == "artifact.begin":
            return self._artifact_begin(payload)
        if request_type == "artifact.chunk":
            return self._artifact_chunk(payload)
        if request_type == "artifact.finish":
            return self._artifact_finish(payload)
        if request_type == "artifact.abort":
            transfer_id = payload.get("transferId")
            if isinstance(transfer_id, str):
                self._artifact_transfers.pop(transfer_id, None)
            return {"aborted": True}
        if request_type == "rlm.run":
            if self.child_planner is None:
                raise SecurityViolation("child run budget exhausted: trusted child planner is not configured")
            if self._depth >= self.config.max_child_depth:
                raise SecurityViolation("child depth budget exhausted")
            # Reserve before invoking the potentially paid parent model.
            # Failed planning attempts remain consumed: without a trusted
            # provider usage receipt, refunding would permit repeated paid calls
            # to bypass the shared ledger.
            self._budget.reserve_child()
            plan, request = self._plan_child(payload)
            result = self._execute_child_reserved(plan.code, timeout=min(self.config.max_cell_seconds, request.remaining_seconds))
            return {
                "rlm_child_id": f"{self.config.task_id}-child-{self._children}",
                "name": plan.name,
                "session_dir": str(self.root / "children" / f"{self.config.task_id}-child-{self._children}"),
                "model": plan.model,
                "status": result.status,
                "result": result.result,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": result.error,
                "provenance": result.provenance,
            }
        if request_type in {"harness.write", "policy.write", "evaluator.write", "promotion.write", "credentials.read", "hidden.read"}:
            raise SecurityViolation(f"learner request denied: {request_type}")
        raise SecurityViolation(f"unsupported host request: {request_type}")

    def _artifact_begin(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        artifact_id = payload.get("artifactId")
        version = payload.get("version", "1")
        size = payload.get("size")
        if (not isinstance(artifact_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", artifact_id)
                or not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", version)
                or not isinstance(size, int) or isinstance(size, bool) or size < 0
                or size > self.config.max_artifact_bytes):
            raise AdapterError("invalid artifact metadata")
        if self._artifact_transfers:
            raise AdapterError("only one artifact transfer may be active")
        transfer_id = uuid.uuid4().hex
        self._artifact_transfers[transfer_id] = {
            "artifactId": artifact_id, "version": version, "size": size,
            "offset": 0, "data": bytearray(),
        }
        # Leave room for JSON/base64 framing under max_protocol_frame_bytes.
        max_chunk = max(1024, min(self.config.max_artifact_bytes,
                                  ((self.config.max_protocol_frame_bytes - 4096) * 3) // 4))
        return {"transferId": transfer_id, "maxChunkBytes": max_chunk}

    def _artifact_chunk(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        transfer_id = payload.get("transferId")
        offset = payload.get("offset")
        encoded = payload.get("data")
        transfer = self._artifact_transfers.get(transfer_id) if isinstance(transfer_id, str) else None
        if transfer is None or not isinstance(offset, int) or offset != transfer["offset"] or not isinstance(encoded, str):
            raise AdapterError("invalid artifact chunk")
        try:
            data = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (binascii.Error, ValueError, UnicodeEncodeError):
            raise AdapterError("artifact chunk is not valid base64") from None
        if not data or len(data) > transfer["size"] - transfer["offset"]:
            raise AdapterError("artifact chunk exceeds declared size")
        transfer["data"].extend(data)
        transfer["offset"] += len(data)
        return {"transferId": transfer_id, "offset": transfer["offset"]}

    def _artifact_finish(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        transfer_id = payload.get("transferId")
        expected = payload.get("sha256")
        transfer = self._artifact_transfers.get(transfer_id) if isinstance(transfer_id, str) else None
        if transfer is None or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise AdapterError("invalid artifact completion")
        if transfer["offset"] != transfer["size"]:
            raise AdapterError("artifact is incomplete")
        data = bytes(transfer["data"])
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            self._artifact_transfers.pop(transfer_id, None)
            raise AdapterError("artifact digest mismatch")
        self._artifact_transfers.pop(transfer_id, None)
        ref = self._store_artifact_bytes(transfer["artifactId"], transfer["version"], data)
        return {"artifact": {"id": ref.id, "version": ref.version, "sha256": ref.sha256,
                              "bytes": ref.bytes, "path": ref.path}}

    def _store_artifact_bytes(self, artifact_id: str, version: str, data: bytes) -> ArtifactRef:
        if len(data) > self.config.max_artifact_bytes:
            raise AdapterError("artifact exceeds configured size limit")
        digest = hashlib.sha256(data).hexdigest()
        key = (artifact_id, version, digest)
        with self._budget.lock:
            existing = self._budget.artifact_refs.get(key)
            if existing is not None:
                return existing
            if self._budget.artifact_count >= self._budget.max_artifact_count:
                raise AdapterError("cumulative artifact count limit exceeded")
            if self._budget.artifact_bytes + len(data) > self._budget.max_artifact_bytes:
                raise AdapterError("cumulative artifact byte limit exceeded")
            self._budget.artifact_count += 1
            self._budget.artifact_bytes += len(data)
            dest = self._artifact_store_root / f"{artifact_id}-{digest[:16]}"
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + f".{uuid.uuid4().hex}.tmp")
                tmp.write_bytes(data)
                os.replace(tmp, dest)
                ref = ArtifactRef(artifact_id, version, digest, len(data), str(dest))
                self._budget.artifact_refs[key] = ref
                return ref
            except Exception:
                self._budget.artifact_count -= 1
                self._budget.artifact_bytes -= len(data)
                raise

    def export_artifact(self, source: str | Path, *, artifact_id: str | None = None, version: str = "1") -> ArtifactRef:
        requested = Path(source)
        raw_src = requested if requested.is_absolute() else self.root / requested
        if raw_src.is_symlink():
            raise SecurityViolation("symlink artifacts are not allowed")
        src = raw_src.resolve()
        root = self.root.resolve()
        if src != root and root not in src.parents:
            raise SecurityViolation("artifact must remain inside the task workspace")
        if not src.is_file():
            raise AdapterError("artifact must be a regular file")
        size = src.stat().st_size
        if size > self.config.max_artifact_bytes:
            raise AdapterError("artifact exceeds configured size limit")
        artifact_id = artifact_id or src.name
        # Refuse names that could escape the content-addressed artifact store.
        if not re.fullmatch(r"[A-Za-z0-9._-]+", artifact_id):
            raise AdapterError("invalid artifact id")
        data = src.read_bytes()
        return self._store_artifact_bytes(artifact_id, version, data)

    def close(self, *, remove_workspace: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.kernel:
                self.kernel.shutdown()
                self.kernel = None
        if remove_workspace:
            shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "PrimeRuntimeAdapter":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
