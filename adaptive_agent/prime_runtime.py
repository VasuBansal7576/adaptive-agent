"""Prime Agent runtime adapter with a task-scoped execution boundary.

The adapter talks to the installed Prime Agent Python kernel (`rlm.repl`) over
its documented JSON-lines protocol.  It intentionally does not call Prime
Inference: model provenance is pinned to the authenticated Codex subscription
selector ``openai-codex/gpt-5.6-luna``.

The host process is trusted.  Learner cells are run in a separate process with
an empty credential-free environment, a private working directory, resource
limits, a broker-only host bridge, and a conservative source policy.  Docker
is supported as an optional stronger boundary; the host fallback is explicit in
``ExecutionMode`` and is never presented as container isolation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import queue
import re
import resource
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


@dataclass
class PrimeRuntimeConfig:
    task_id: str
    root_dir: Path | str | None = None
    model: str = "openai-codex/gpt-5.6-luna"
    provider: str = "openai-codex"
    python: str | None = None
    runtime_src: Path | str | None = None
    max_output_chars: int = 65536
    max_cell_seconds: float = 30.0
    max_memory_bytes: int = 512 * 1024 * 1024
    max_cpu_seconds: int = 30
    max_processes: int = 8
    max_artifact_bytes: int = 8 * 1024 * 1024
    child_runs: int = 0
    require_docker: bool = True
    docker_image: str | None = None


# This is deliberately narrower than Python's full library surface.  A task
# can still use expressions and pure computation; side effects must go through
# the trusted broker.
_ALLOWED_IMPORTS = frozenset({"math", "statistics", "json", "re", "datetime", "decimal", "itertools", "functools", "collections"})
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


def _safe_env(root: Path, runtime_src: Path) -> dict[str, str]:
    # Do not inherit provider credentials, cloud credentials, agent tokens, or
    # arbitrary user configuration.  The runtime only needs Python and its
    # bundled source path.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(root / "home"),
        "TMPDIR": str(root / "tmp"),
        "PYTHONPATH": str(runtime_src),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    return env


def _limit_process(max_cpu: int, max_memory: int, max_processes: int) -> None:
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (max_cpu, max_cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
        resource.setrlimit(resource.RLIMIT_NPROC, (max_processes, max_processes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
    except (OSError, ValueError):
        # Windows and some managed hosts do not expose all rlimits.  The
        # provenance records this limitation; process cancellation remains.
        pass


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
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
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
        command = [
            "docker", "run", "--rm", "-i", "--init",
            "--network=none", "--read-only",
            "--tmpfs", f"/tmp:rw,nosuid,nodev,noexec,size={memory}m",
            "--cap-drop=ALL", "--security-opt", "no-new-privileges:true",
            "--pids-limit", str(pids), "--memory", str(config.max_memory_bytes),
            "--cpus", "0.5", "--ulimit", "nofile=64:64", "--ulimit", "fsize=16777216:16777216",
            "--user", "65534:65534", "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
            image,
        ]
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

    def _read_events(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in iter(self.proc.stdout.readline, b""):
            try:
                event = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.start_error = AdapterError(f"invalid Prime kernel frame: {exc}")
                continue
            if event.get("event") == "ready":
                self.ready.set()
            elif event.get("event") == "host_request":
                self._handle_host_request(event)
            else:
                self.events.put(event)
        self.ready.set()

    def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for raw in iter(self.proc.stderr.readline, b""):
            self.stderr_chunks.append(raw.decode("utf-8", "replace"))

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
        with self.write_lock:
            self.proc.stdin.write(data)
            self.proc.stdin.flush()

    def execute(self, cell_id: str, code: str, timeout: float) -> dict[str, Any]:
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
                collected["stdout"].append(str(event.get("text", "")))
            elif kind == "stderr":
                collected["stderr"].append(str(event.get("text", "")))
            elif kind == "result":
                collected["result"] = str(event.get("text", ""))
            elif kind == "error":
                collected["error"] = {k: event.get(k) for k in ("ename", "evalue", "traceback")}
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
        if not self.proc:
            return
        try:
            self.send({"type": "shutdown", "id": uuid.uuid4().hex})
            self.proc.wait(timeout=2)
        except (Exception,):
            self.kill()
        finally:
            for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                try:
                    stream.close()
                except (AttributeError, OSError):
                    pass
            self.proc = None

    def kill(self) -> None:
        if not self.proc:
            return
        proc = self.proc
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except (AttributeError, OSError):
                    pass
            self.proc = None


class PrimeRuntimeAdapter:
    """One persistent, task-scoped Prime Python kernel."""

    def __init__(self, config: PrimeRuntimeConfig, broker: CapabilityBroker | None = None):
        if config.provider != "openai-codex" or config.model != "openai-codex/gpt-5.6-luna":
            raise AdapterError("only the authenticated openai-codex/gpt-5.6-luna path is supported; Prime Inference is disabled")
        if not config.require_docker:
            raise AdapterError("host execution is forbidden; Docker isolation is mandatory")
        if shutil.which("docker") is None:
            raise AdapterError("Docker CLI is required for the Prime learner boundary")
        self.config = config
        self.root = Path(config.root_dir) if config.root_dir else Path(tempfile.mkdtemp(prefix=f"adaptive-{config.task_id}-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.broker = broker or CapabilityBroker(config.task_id)
        self.kernel: _KernelProcess | None = None
        self._lock = threading.RLock()
        self._children = 0
        self._closed = False
        self._started_at = time.time()
        self._mode = ExecutionMode.PRIME_SUBSCRIPTION_DOCKER
        self._runtime_src = _find_runtime_src(config.runtime_src)

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def provenance(self) -> dict[str, Any]:
        return {
            "adapter": "adaptive-agent.prime-runtime",
            "mode": self._mode.value,
            "provider": self.config.provider,
            "model": self.config.model,
            "kernel": "Prime Agent rlm.repl protocol v3",
            "runtimeSource": str(self._runtime_src),
            "isolation": "per-run Docker container; nonroot, read-only root, tmpfs /tmp, dropped capabilities, no-new-privileges, pids/memory/cpu limits",
            "network": "none inside learner container; broker bridge uses the parent stdio protocol",
            "credentials": "not inherited by learner process",
            "limits": {"cellSeconds": self.config.max_cell_seconds, "memoryBytes": self.config.max_memory_bytes,
                       "cpuSeconds": self.config.max_cpu_seconds, "processes": self.config.max_processes},
            "limitations": "Docker daemon/image availability is an environment dependency; authenticated model calls stay in the trusted parent, never in the learner container",
        }

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
        effective_timeout = timeout if timeout is not None else self.config.max_cell_seconds
        if cancel and cancel.is_set():
            self.cancel(cell_id)
            return ExecutionResult(self.config.task_id, cell_id, "aborted", None, "", "", None, 0, self.mode, self.provenance())
        try:
            raw = self.kernel.execute(cell_id, code, effective_timeout)
            status = raw.get("status", "error")
            if status not in {"ok", "error", "aborted"}:
                status = "error"
            return ExecutionResult(self.config.task_id, cell_id, status, raw.get("result"),
                                   "".join(raw.get("stdout", []))[-self.config.max_output_chars:],
                                   "".join(raw.get("stderr", []))[-self.config.max_output_chars:],
                                   raw.get("error"), int((time.monotonic() - started) * 1000), self.mode, self.provenance())
        except TimeoutError:
            return ExecutionResult(self.config.task_id, cell_id, "aborted", None, "", "", {"ename": "TimeoutError", "evalue": "cell timeout"},
                                   int((time.monotonic() - started) * 1000), self.mode, self.provenance())

    def cancel(self, cell_id: str | None = None) -> None:
        with self._lock:
            if self.kernel:
                self.kernel.cancel(cell_id)

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
        if request_type == "rlm.run":
            if self._children >= self.config.child_runs:
                raise SecurityViolation("child run budget exhausted")
            raise SecurityViolation("child runner is not configured")
        if request_type in {"harness.write", "policy.write", "evaluator.write", "promotion.write", "credentials.read", "hidden.read"}:
            raise SecurityViolation(f"learner request denied: {request_type}")
        raise SecurityViolation(f"unsupported host request: {request_type}")

    def export_artifact(self, source: str | Path, *, artifact_id: str | None = None, version: str = "1") -> ArtifactRef:
        src = (self.root / source).resolve() if not Path(source).is_absolute() else Path(source).resolve()
        root = self.root.resolve()
        if src != root and root not in src.parents:
            raise SecurityViolation("artifact must remain inside the task workspace")
        if src.is_symlink() or not src.is_file():
            raise AdapterError("artifact must be a regular file")
        size = src.stat().st_size
        if size > self.config.max_artifact_bytes:
            raise AdapterError("artifact exceeds configured size limit")
        artifact_id = artifact_id or src.name
        # Refuse names that could escape the content-addressed artifact store.
        if not re.fullmatch(r"[A-Za-z0-9._-]+", artifact_id):
            raise AdapterError("invalid artifact id")
        data = src.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        dest = self.root / "artifacts" / f"{artifact_id}-{digest[:16]}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        return ArtifactRef(artifact_id, version, digest, size, str(dest))

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
