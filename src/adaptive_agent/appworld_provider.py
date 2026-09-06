"""Optional AppWorld benchmark adapter.

The adapter keeps AppWorld out of the main runtime dependency graph.  A single
short-lived worker process owns one AppWorld episode; the parent process exposes
only task instructions, public API schemas, and broker-dispatched API calls.
Evaluator output is reduced to aggregate fields before it crosses the process
boundary.  Test splits are never loaded unless an explicit protocol freeze
allows them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from adaptive_agent.broker import ProviderExecutionOutcome, ToolProvider
from adaptive_agent.evaluation import ArtifactRef as EvaluationArtifactRef, Partition, TaskInput as EvaluationTaskInput, sha256_json
from adaptive_agent.models import (
    ArtifactRef,
    EnvironmentManifest,
    TaskInput as DurableTaskInput,
    ToolSchema,
)
from adaptive_agent.environment import validate_arguments, SchemaValidationError


APPWORLD_ENVIRONMENT_ID = "appworld"
PUBLISHED_APPWORLD_VERSION = "0.1.3.post1"
_PUBLIC_DATA_FILES = ("data/version.txt", "data/LICENSE", "data/base_dbs/version.txt")
_ALLOWED_SPLITS = ("train", "dev", "test_normal", "test_challenge")
_TEST_SPLITS = {"test_normal", "test_challenge"}
_BROKER_DOC_SEARCH = "appworld__search_api_docs"
_BROKER_DOC_GET = "appworld__get_api_doc"
_BROKER_READ = "appworld__call_read"
_BROKER_WRITE = "appworld__call_write"


class AppWorldError(RuntimeError):
    """Base error for optional AppWorld integration failures."""


class AppWorldUnavailable(AppWorldError):
    """Raised when the isolated AppWorld Python environment is unavailable."""


class AppWorldProtocolError(AppWorldError):
    """Raised when the worker violates the bounded JSON-lines protocol."""


@dataclass(frozen=True)
class AppWorldConfig:
    root: Path
    python: str = ""
    package_version: str = PUBLISHED_APPWORLD_VERSION
    environment_id: str = APPWORLD_ENVIRONMENT_ID
    experiment_name: str = "adaptive-agent"
    timeout_seconds: float = 30.0
    allow_test: bool = False

    def __post_init__(self) -> None:
        # Accept path-like values at the public boundary while keeping all
        # internal path operations typed and deterministic.
        object.__setattr__(self, "root", Path(self.root))

    def resolved_python(self) -> str:
        return self.python.strip() or os.environ.get("APPWORLD_PYTHON", "appworld")

    @property
    def data_root(self) -> Path:
        return self.root / "data"


@dataclass(frozen=True)
class AppWorldTask:
    task_id: str
    split: str
    instruction: str
    allowed_apps: tuple[str, ...]
    datetime: str
    db_version: str

    def __post_init__(self) -> None:
        if self.split not in _ALLOWED_SPLITS:
            raise AppWorldError(f"unsupported AppWorld split: {self.split}")
        if not self.task_id or not self.instruction:
            raise AppWorldError("AppWorld task requires task_id and instruction")

    @property
    def is_test(self) -> bool:
        return self.split in _TEST_SPLITS


@dataclass(frozen=True)
class AppWorldRuntimeManifest:
    """Hash-pinned public runtime metadata written to run artifacts."""

    environment_id: str
    package_version: str
    python: str
    root: str
    public_data_sha256: str
    dataset_sha256: str
    split_counts: Mapping[str, int]
    split_ids: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "environmentId": self.environment_id,
            "benchmark": "AppWorld",
            "dataClass": "external_published_simulated_benchmark",
            "packageVersion": self.package_version,
            "python": self.python,
            "root": self.root,
            "publicDataSha256": self.public_data_sha256,
            "datasetSha256": self.dataset_sha256,
            "splitCounts": dict(self.split_counts),
            "splitIds": {name: list(ids) for name, ids in self.split_ids.items()},
            "catalog": {
                "publicTaskSpecsLoaded": True,
                "testTaskSpecsLoaded": False,
                "groundTruthLoaded": False,
                "taskReportsLoaded": False,
            },
            "isolatedEvaluator": {
                "groundTruthLoaded": True,
                "groundTruthMode": "minimal",
                "groundTruthVisibility": "evaluator_only",
                "taskReportsLoaded": False,
            },
        }


class _JsonLineProcess:
    """Bounded request/response protocol for one worker process."""

    _MAX_FRAME_BYTES = 1_048_576

    def __init__(self, command: Sequence[str], env: Mapping[str, str], timeout_seconds: float) -> None:
        self._timeout = timeout_seconds
        self._next_id = 0
        self._stdout_buffer = b""
        try:
            self._proc = subprocess.Popen(
                list(command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=dict(env),
            )
        except OSError as exc:
            raise AppWorldUnavailable(f"could not start AppWorld worker: {exc}") from exc
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            if stream is not None:
                os.set_blocking(stream.fileno(), False)

    @property
    def pid(self) -> int:
        return int(self._proc.pid)

    def _stderr_tail(self) -> str:
        if self._proc.stderr is None:
            return ""
        try:
            chunks: list[bytes] = []
            remaining = 2000
            while remaining > 0:
                try:
                    chunk = os.read(self._proc.stderr.fileno(), remaining)
                except BlockingIOError:
                    break
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)[-2000:].decode("utf-8", "replace")
        except (OSError, ValueError):
            return ""

    def _write_frame(self, frame: Mapping[str, Any], deadline: float) -> None:
        if self._proc.stdin is None:
            raise AppWorldProtocolError("AppWorld worker pipes are unavailable")
        data = (json.dumps(frame, separators=(",", ":")) + "\n").encode("utf-8")
        offset = 0
        fd = self._proc.stdin.fileno()
        while offset < len(data):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            _, writable, _ = select.select([], [fd], [], remaining)
            if not writable:
                raise TimeoutError
            try:
                offset += os.write(fd, data[offset:])
            except BlockingIOError:
                continue

    def _readline_until(self, deadline: float) -> bytes:
        """Read one complete frame without ever blocking past deadline."""
        if self._proc.stdout is None:
            raise AppWorldProtocolError("AppWorld worker pipes are unavailable")
        fd = self._proc.stdout.fileno()
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                line, self._stdout_buffer = self._stdout_buffer[:newline], self._stdout_buffer[newline + 1 :]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError
            chunk = os.read(fd, 65536)
            if not chunk:
                if self._stdout_buffer:
                    line, self._stdout_buffer = self._stdout_buffer, b""
                    return line
                raise AppWorldProtocolError(f"AppWorld worker closed: {self._stderr_tail()}")
            self._stdout_buffer += chunk
            if len(self._stdout_buffer) > self._MAX_FRAME_BYTES:
                raise AppWorldProtocolError("AppWorld worker frame exceeds size limit")

    def request(self, operation: str, payload: Mapping[str, Any] | None = None) -> Any:
        if self._proc.poll() is not None:
            raise AppWorldProtocolError(f"AppWorld worker exited ({self._proc.returncode}): {self._stderr_tail()}")
        if self._proc.stdin is None or self._proc.stdout is None:
            raise AppWorldProtocolError("AppWorld worker pipes are unavailable")
        self._next_id += 1
        request_id = self._next_id
        frame = {"id": request_id, "operation": operation, "payload": dict(payload or {})}
        deadline = time.monotonic() + self._timeout
        written = False
        while True:
            try:
                if not written:
                    self._write_frame(frame, deadline)
                    written = True
                line = self._readline_until(deadline)
            except TimeoutError:
                self.close(force=True)
                raise AppWorldProtocolError(f"AppWorld worker timed out during {operation}") from None
            except AppWorldProtocolError:
                self.close(force=True)
                raise
            try:
                response = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise AppWorldProtocolError("AppWorld worker emitted invalid JSON") from exc
            if not isinstance(response, dict) or response.get("id") != request_id:
                raise AppWorldProtocolError("AppWorld worker response id mismatch")
            if response.get("ok") is not True:
                raise AppWorldError(str(response.get("error", "AppWorld worker operation failed")))
            return response.get("result")

    def close(self, *, force: bool = False) -> None:
        if self._proc.poll() is not None:
            return
        try:
            if not force and self._proc.stdin is not None:
                self._next_id += 1
                deadline = time.monotonic() + min(self._timeout, 1.0)
                self._write_frame({"id": self._next_id, "operation": "close", "payload": {}}, deadline)
                self._readline_until(deadline)
        except (BrokenPipeError, OSError, TimeoutError, AppWorldProtocolError):
            pass
        finally:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2)


class AppWorldCatalog:
    """Read public AppWorld task metadata and API documentation only."""

    def __init__(self, config: AppWorldConfig) -> None:
        self.config = config
        self.data_root = config.data_root
        if not self.data_root.is_dir():
            raise AppWorldUnavailable(f"AppWorld data directory does not exist: {self.data_root}")

    def split_ids(self, split: str) -> tuple[str, ...]:
        if split not in _ALLOWED_SPLITS:
            raise AppWorldError(f"unsupported AppWorld split: {split}")
        path = self.data_root / "datasets" / f"{split}.txt"
        if not path.is_file():
            raise AppWorldUnavailable(f"AppWorld split file is missing: {path}")
        ids = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
        if len(ids) != len(set(ids)):
            raise AppWorldError(f"duplicate task id in {split} split")
        return ids

    def task(self, task_id: str, split: str, *, allow_test: bool | None = None) -> AppWorldTask:
        if split in _TEST_SPLITS and not (self.config.allow_test if allow_test is None else allow_test):
            raise AppWorldError("test AppWorld tasks are sealed until protocol freeze")
        if task_id not in set(self.split_ids(split)):
            raise AppWorldError(f"task {task_id!r} is not in AppWorld {split}")
        spec_path = self.data_root / "tasks" / task_id / "specs.json"
        # specs.json is public task instruction metadata.  Ground-truth files
        # are deliberately never opened by this catalog.
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise AppWorldUnavailable(f"invalid AppWorld task spec: {task_id}") from exc
        if not isinstance(spec, dict):
            raise AppWorldError("AppWorld task spec must be an object")
        allowed_apps = spec.get("allowed_apps", spec.get("allowedApps", ()))
        if not isinstance(allowed_apps, list) or not all(isinstance(item, str) for item in allowed_apps):
            allowed_apps = []
        return AppWorldTask(
            task_id=task_id,
            split=split,
            instruction=str(spec.get("instruction", "")),
            allowed_apps=tuple(allowed_apps),
            datetime=str(spec.get("datetime", "")),
            db_version=str(spec.get("db_version", spec.get("dbVersion", ""))),
        )

    def tasks(self, splits: Iterable[str] = ("train", "dev"), *, allow_test: bool | None = None) -> tuple[AppWorldTask, ...]:
        result: list[AppWorldTask] = []
        for split in splits:
            for task_id in self.split_ids(split):
                result.append(self.task(task_id, split, allow_test=allow_test))
        return tuple(result)

    def public_data_hash(self) -> str:
        """Hash public manifests/docs/specs without opening ground-truth files."""
        digest = hashlib.sha256()
        paths: list[Path] = []
        for relative in _PUBLIC_DATA_FILES:
            path = self.config.root / relative
            if path.is_file():
                paths.append(path)
        for path in sorted((self.data_root / "datasets").glob("*.txt")):
            paths.append(path)
        for path in sorted((self.data_root / "api_docs").rglob("*.json")):
            paths.append(path)
        # Task instructions for train/dev are public learner metadata.  Test
        # task specifications are intentionally excluded from the hash and are
        # never opened by setup; only their split ID lists are recorded.
        public_task_ids = set(self.split_ids("train")) | set(self.split_ids("dev"))
        for path in sorted((self.data_root / "tasks").glob("*/specs.json")):
            if path.parent.name in public_task_ids:
                paths.append(path)
        for path in sorted((self.data_root / "base_dbs").glob("*.db")):
            paths.append(path)
        for path in sorted(paths):
            digest.update(str(path.relative_to(self.config.root)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def dataset_hash(self) -> str:
        """Opaque content hash for the complete local benchmark dataset.

        The bytes are never returned or loaded into learner-visible structures;
        this digest only binds evaluator and task artifacts to one installation.
        """
        digest = hashlib.sha256()
        generated = {"__pycache__"}
        for path in sorted(
            path
            for path in self.data_root.rglob("*")
            if path.is_file() and path.suffix not in {".pyc", ".pyo"} and not generated.intersection(path.parts)
        ):
            digest.update(str(path.relative_to(self.config.root)).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def runtime_manifest(self) -> AppWorldRuntimeManifest:
        split_ids = {split: self.split_ids(split) for split in _ALLOWED_SPLITS}
        return AppWorldRuntimeManifest(
            self.config.environment_id,
            self.config.package_version,
            self.config.resolved_python(),
            str(self.config.root),
            self.public_data_hash(),
            self.dataset_hash(),
            {name: len(ids) for name, ids in split_ids.items()},
            split_ids,
        )


def _standard_entries(path: Path) -> dict[str, Mapping[str, Any]]:
    """Load standard docs in either the published list or fixture map form."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AppWorldUnavailable(f"invalid AppWorld standard API docs: {path}") from exc
    entries: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                entries[str(key)] = value
    elif isinstance(raw, list):
        for value in raw:
            if not isinstance(value, Mapping):
                continue
            key = value.get("api_name", value.get("apiName"))
            if isinstance(key, str) and key:
                entries[key] = value
    return entries


def _schema_like(value: Any) -> bool:
    return isinstance(value, Mapping) and (isinstance(value.get("type"), str) or "properties" in value or "items" in value)


def _example_shape(value: Any) -> dict[str, Any]:
    """Describe only the JSON shape demonstrated by one public example."""
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        return {"type": "array", "items": {}}
    return {"type": "object", "additionalProperties": True}


def _response_schema(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON schema while retaining published response examples."""
    raw = entry.get("response_schemas", entry.get("responseSchemas"))
    if not isinstance(raw, Mapping):
        return {"type": "object", "additionalProperties": True}
    if _schema_like(raw):
        return dict(raw)
    for key in ("success", "200", "2xx"):
        candidate = raw.get(key)
        if _schema_like(candidate):
            return dict(candidate)
    # AppWorld's standard format historically called concrete response examples
    # "response_schemas". Preserve those examples and describe their observed
    # JSON shapes instead of claiming every response is an object.
    examples = list(raw.values())
    shapes: list[dict[str, Any]] = []
    for example in examples:
        shape = _example_shape(example)
        if shape not in shapes:
            shapes.append(shape)
    if len(shapes) == 1:
        schema = shapes[0]
    else:
        schema = {"anyOf": shapes}
    schema["examples"] = examples
    return schema


def _schema_from_function_doc(doc: Mapping[str, Any], version: str, standard: Mapping[str, Any] | None = None) -> ToolSchema:
    function = doc.get("function") if isinstance(doc.get("function"), Mapping) else {}
    name = str(function.get("name", ""))
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}
    else:
        parameters = json.loads(json.dumps(parameters))
    properties = parameters.setdefault("properties", {})
    if not isinstance(properties, dict):
        properties = {}
        parameters["properties"] = properties
    required = list(parameters.get("required", [])) if isinstance(parameters.get("required", []), list) else []
    if standard is not None:
        standard_parameters = standard.get("parameters", [])
        if isinstance(standard_parameters, list):
            for parameter in standard_parameters:
                if not isinstance(parameter, Mapping) or not isinstance(parameter.get("name"), str):
                    continue
                parameter_name = parameter["name"]
                property_schema = properties.get(parameter_name)
                if not isinstance(property_schema, dict):
                    property_schema = {}
                    properties[parameter_name] = property_schema
                for key in ("type", "format", "enum", "items", "minimum", "maximum", "pattern"):
                    if key not in property_schema and key in parameter:
                        property_schema[key] = parameter[key]
                for key in ("description", "default"):
                    if key not in property_schema and key in parameter and parameter[key] is not None:
                        property_schema[key] = parameter[key]
                constraints = parameter.get("constraints")
                if isinstance(constraints, list) and constraints and "constraints" not in property_schema:
                    property_schema["constraints"] = constraints
                if parameter.get("required") is True and parameter_name not in required:
                    required.append(parameter_name)
        parameters["required"] = required
        parameters.setdefault("additionalProperties", False)
    _, _, api = name.partition("__")
    method: str | None = None
    if isinstance(standard, Mapping) and isinstance(standard.get("method"), str):
        method = standard["method"].upper()
    else:
        standard_path = Path(str(doc.get("_standard_path", "")))
        if standard_path.is_file():
            entries = _standard_entries(standard_path)
            entry = entries.get(api, {})
            if isinstance(entry.get("method"), str):
                method = entry["method"].upper()
    if method not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}:
        raise AppWorldUnavailable(f"missing authoritative HTTP method for AppWorld API: {name or '<unnamed>'}")
    effect = "read" if method in {"GET", "HEAD"} else "write"
    return ToolSchema(name=name, version=version, inputSchema=parameters, outputSchema=_response_schema(standard or {}), effect=effect)


@dataclass(frozen=True)
class _ApiDocRecord:
    schema: ToolSchema
    description: str
    app: str
    method: str


def _api_doc_records(config: AppWorldConfig) -> tuple[_ApiDocRecord, ...]:
    docs_root = config.data_root / "api_docs" / "function_calling"
    standard_root = config.data_root / "api_docs" / "standard"
    records: list[_ApiDocRecord] = []
    seen: set[str] = set()
    for path in sorted(docs_root.glob("*.json")):
        try:
            entries = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise AppWorldUnavailable(f"invalid AppWorld API docs: {path}") from exc
        if not isinstance(entries, list):
            continue
        standard_entries = _standard_entries(standard_root / path.name)
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            function = entry.get("function") if isinstance(entry.get("function"), Mapping) else {}
            name = function.get("name", "")
            api = str(name).split("__", 1)[-1]
            standard = standard_entries.get(api)
            enriched = dict(entry)
            enriched["_standard_path"] = str(standard_root / path.name)
            schema = _schema_from_function_doc(enriched, config.package_version, standard)
            if not schema.name or schema.name in seen:
                continue
            seen.add(schema.name)
            description = str(function.get("description", "") or (standard or {}).get("description", "") or "").strip()
            method = str((standard or {}).get("method", "")).upper()
            records.append(_ApiDocRecord(schema, description, schema.name.split("__", 1)[0], method))
    if not records:
        raise AppWorldUnavailable("AppWorld API documentation contains no callable schemas")
    return tuple(records)


def public_tool_schemas(config: AppWorldConfig) -> tuple[ToolSchema, ...]:
    return tuple(record.schema for record in _api_doc_records(config))


def broker_tool_schemas(version: str) -> tuple[ToolSchema, ...]:
    """Small learner-facing surface; full API schemas are retrieved lazily."""
    api_name = {"type": "string", "minLength": 1}
    arguments = {"type": "object", "additionalProperties": True}
    return (
        ToolSchema(name=_BROKER_DOC_SEARCH, version=version, inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, "additionalProperties": False}, outputSchema={"type": "object"}, effect="read"),
        ToolSchema(name=_BROKER_DOC_GET, version=version, inputSchema={"type": "object", "required": ["apiName"], "properties": {"apiName": api_name}, "additionalProperties": False}, outputSchema={"type": "object"}, effect="read"),
        ToolSchema(name=_BROKER_READ, version=version, inputSchema={"type": "object", "required": ["apiName", "arguments"], "properties": {"apiName": api_name, "arguments": arguments}, "additionalProperties": False}, outputSchema={"type": "object"}, effect="read"),
        ToolSchema(name=_BROKER_WRITE, version=version, inputSchema={"type": "object", "required": ["apiName", "arguments"], "properties": {"apiName": api_name, "arguments": arguments}, "additionalProperties": False}, outputSchema={"type": "object"}, effect="write"),
    )


def build_manifest(config: AppWorldConfig) -> EnvironmentManifest:
    catalog = AppWorldCatalog(config)
    data_hash = catalog.public_data_hash()
    dataset_hash = catalog.dataset_hash()
    ref = lambda name: ArtifactRef(id=name, version=config.package_version, sha256=sha256_json({"name": name, "version": config.package_version, "publicData": data_hash, "dataset": dataset_hash}))
    docs_ref = ref("appworld-api-docs")
    return EnvironmentManifest(
        environmentId=config.environment_id,
        version=config.package_version,
        docs=[docs_ref],
        toolSchemas=list(broker_tool_schemas(config.package_version)),
        policyRef=ref("appworld-policy"),
        evaluatorRef=ref("appworld-evaluator"),
        resetRef=ref("appworld-reset"),
        executionModes=["interactive", "batch", "replay"],
        capabilities=["appworld", "public_api_docs", "supervisor_completion", "aggregate_evaluation"],
    )


def register_appworld(registry: Any, config: AppWorldConfig, splits: Iterable[str] = ("train", "dev"), *, runtime_partition: str | None = None) -> tuple[ArtifactRef, tuple[ArtifactRef, ...]]:
    """Register public AppWorld task metadata, defaulting to train and dev."""
    manifest = build_manifest(config)
    manifest_ref = registry.register(manifest)
    environment_ref = ArtifactRef(
        id=config.environment_id,
        version=config.package_version,
        sha256=sha256_json(manifest.model_dump(mode="json", by_alias=True)),
    )
    task_refs: list[ArtifactRef] = []
    for task in AppWorldCatalog(config).tasks(splits):
        durable_task = DurableTaskInput(
            taskId=task.task_id,
            environmentRef=environment_ref,
            goal=task.instruction,
            allowedInputRefs=[],
            partition=runtime_partition or {"train": "development", "dev": "validation", "test_normal": "final", "test_challenge": "final"}[task.split],
            provenance={
                "benchmark": "AppWorld",
                "dataClass": "external_published_simulated_benchmark",
                "officialSplit": task.split,
                "runtimePartition": runtime_partition or {"train": "development", "dev": "validation", "test_normal": "final", "test_challenge": "final"}[task.split],
            },
        )
        task_refs.append(registry.register_task(durable_task))
    return manifest_ref, tuple(task_refs)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    for method in ("model_dump", "to_dict"):
        fn = getattr(value, method, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except TypeError:
                continue
    return str(value)


def _public_supervisor(value: Any) -> dict[str, str]:
    """Keep only the public supervisor identity fields used for discovery."""
    if not isinstance(value, Mapping):
        return {}
    fields = {"firstName", "lastName", "email", "phoneNumber"}
    return {key: item for key, item in value.items() if key in fields and isinstance(item, str) and item}


def _public_app_descriptions(value: Any) -> dict[str, str]:
    """Keep public app descriptions bounded to string metadata."""
    if not isinstance(value, Mapping):
        return {}
    return {str(name): description for name, description in value.items() if isinstance(description, str)}


class AppWorldProvider(ToolProvider):
    """Broker provider backed by one isolated AppWorld episode."""

    def __init__(self, config: AppWorldConfig, task: AppWorldTask, run_id: str, seed: int = 0, *, worker_command: Sequence[str] | None = None) -> None:
        if task.is_test and not config.allow_test:
            raise AppWorldError("test AppWorld tasks are sealed until protocol freeze")
        self.config = config
        self.task = task
        self.run_id = run_id
        self.seed = seed
        self._api_records = {record.schema.name: record for record in _api_doc_records(config)}
        self._api_schemas = {name: record.schema for name, record in self._api_records.items()}
        self._schemas = {schema.name: schema for schema in broker_tool_schemas(config.package_version)}
        worker_script = Path(__file__).with_name("appworld_worker.py")
        launcher = "import runpy,sys; script=sys.argv[1]; root=sys.argv[2]; sys.argv=[script, '--root', root]; runpy.run_path(script, run_name='__main__')"
        command = list(worker_command or [config.resolved_python(), "-u", "-c", launcher, str(worker_script), str(config.root)])
        env = os.environ.copy()
        # The worker is a standalone script.  In particular, do not expose
        # this checkout's ``adaptive_agent`` package to AppWorld's pinned
        # Pydantic 1.x environment; the main runtime uses Pydantic 2.x.
        env.pop("PYTHONPATH", None)
        env["APPWORLD_ROOT"] = str(config.root)
        self._process = _JsonLineProcess(command, env, config.timeout_seconds)
        try:
            reset = self._process.request("reset", {"taskId": task.task_id, "seed": seed, "experimentName": f"{config.experiment_name}-{run_id}"})
        except Exception:
            self._process.close(force=True)
            raise
        self._reset = reset if isinstance(reset, dict) else {}
        reset_apps = self._reset.get("allowedApps") if isinstance(self._reset, dict) else None
        self._allowed_apps = {item for item in reset_apps if isinstance(item, str)} if isinstance(reset_apps, list) else set(task.allowed_apps)

    @property
    def process_id(self) -> int:
        return self._process.pid

    def public_context(self) -> dict[str, Any]:
        allowed_apps = self._reset.get("allowedApps") if isinstance(self._reset, dict) else None
        if not isinstance(allowed_apps, list):
            allowed_apps = list(self.task.allowed_apps)
        schemas = list(self._schemas.values())
        return {
            "taskId": self.task.task_id,
            "split": self.task.split,
            "instruction": self.task.instruction,
            "allowedApps": [item for item in allowed_apps if isinstance(item, str)],
            "supervisor": _public_supervisor(self._reset.get("supervisor") if isinstance(self._reset, dict) else None),
            "appDescriptions": _public_app_descriptions(self._reset.get("appDescriptions") if isinstance(self._reset, dict) else None),
            "apiSchemas": [schema.model_dump(mode="json", by_alias=True) for schema in schemas],
        }

    def execute(self, run_id: str, tool: str, arguments: dict[str, Any]) -> ProviderExecutionOutcome:
        if run_id != self.run_id:
            raise AppWorldError("provider is bound to a different run")
        schema = self._schemas.get(tool)
        if schema is None:
            raise AppWorldError(f"unknown AppWorld API: {tool}")
        if tool == _BROKER_DOC_SEARCH:
            query = str(arguments.get("query", ""))
            offset = arguments.get("offset", 0)
            limit = arguments.get("limit", 20)
            if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                raise AppWorldError("search offset must be a non-negative integer")
            if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
                raise AppWorldError("search limit must be an integer between 1 and 100")
            query_tokens = re.findall(r"[a-z0-9]+", query.casefold())
            ranked: list[tuple[int, str, _ApiDocRecord]] = []
            for name, record in self._api_records.items():
                name_tokens = re.findall(r"[a-z0-9]+", name.casefold())
                description_tokens = re.findall(r"[a-z0-9]+", record.description.casefold())
                if not query_tokens:
                    score = 0
                else:
                    name_set = set(name_tokens)
                    description_set = set(description_tokens)
                    matched = [token for token in query_tokens if token in name_set or token in description_set]
                    if not matched:
                        continue
                    score = sum(5 if token in name_set else 3 for token in matched)
                    if len(matched) == len(query_tokens):
                        score += 10
                    phrase = " ".join(query_tokens)
                    if phrase in " ".join(name_tokens + description_tokens):
                        score += 2
                ranked.append((score, name, record))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            total = len(ranked)
            page = ranked[offset : offset + limit]
            hits = [
                {
                    "apiName": name,
                    "app": record.app,
                    "description": record.description,
                    "method": record.method,
                    "effect": record.schema.effect,
                }
                for _, name, record in page
            ]
            next_offset = offset + limit if offset + limit < total else None
            return ProviderExecutionOutcome(output={"results": hits, "total": total, "offset": offset, "limit": limit, "truncated": next_offset is not None, "nextOffset": next_offset}, effect="none")
        if tool == _BROKER_DOC_GET:
            api_name = arguments.get("apiName")
            record = self._api_records.get(api_name) if isinstance(api_name, str) else None
            if record is None:
                raise AppWorldError("unknown AppWorld API documentation name")
            return ProviderExecutionOutcome(output={"apiName": api_name, "app": record.app, "description": record.description, "method": record.method, "schema": record.schema.model_dump(mode="json", by_alias=True)}, effect="none")
        api_name = arguments.get("apiName")
        api_arguments = arguments.get("arguments")
        if not isinstance(api_name, str) or not isinstance(api_arguments, dict):
            raise AppWorldError("dynamic AppWorld calls require apiName and arguments")
        api_schema = self._api_schemas.get(api_name)
        if api_schema is None:
            raise AppWorldError("unknown AppWorld API")
        app = api_name.split("__", 1)[0]
        if self._allowed_apps and app not in self._allowed_apps and app not in {"api_docs", "supervisor"}:
            raise AppWorldError(f"AppWorld API app is not allowed for task: {app}")
        try:
            validate_arguments(api_schema, api_arguments)
        except SchemaValidationError as exc:
            raise AppWorldError(f"invalid arguments for {api_name}: {exc}") from exc
        if tool == _BROKER_READ and api_schema.effect != "read":
            raise AppWorldError("write API must use appworld__call_write")
        if tool == _BROKER_WRITE and api_schema.effect != "write":
            raise AppWorldError("read API must use appworld__call_read")
        result = self._process.request("call", {"tool": api_name, "arguments": api_arguments})
        return ProviderExecutionOutcome(output=result if isinstance(result, dict) else {"value": result}, effect="confirmed" if schema.effect == "write" else "none")

    def effect(self, tool: str) -> str:
        schema = self._schemas.get(tool)
        if schema is None:
            raise AppWorldError(f"unknown AppWorld API: {tool}")
        return schema.effect

    def version(self, tool: str) -> str:
        schema = self._schemas.get(tool)
        if schema is None:
            raise AppWorldError(f"unknown AppWorld API: {tool}")
        return schema.version

    def evaluate_aggregate(self) -> dict[str, Any]:
        result = self._process.request("evaluate")
        if not isinstance(result, dict):
            raise AppWorldProtocolError("AppWorld evaluator returned a non-object")
        return {key: result[key] for key in ("success", "numTests", "passCount", "failCount", "taskCompleted") if key in result}

    def close(self) -> None:
        self._process.close()

    def __enter__(self) -> "AppWorldProvider":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class AppWorldPackage:
    """Runtime package binding AppWorld metadata to the provider factory."""

    def __init__(self, config: AppWorldConfig) -> None:
        self.config = config
        self.catalog = AppWorldCatalog(config)
        self.manifest = build_manifest(config)
        self.environment_id = config.environment_id

    def learner_tasks(self) -> tuple[AppWorldTask, ...]:
        # Only official train tasks are learner-facing.  Dev remains an
        # evaluator-owned validation partition and is never fed to learning.
        return self.catalog.tasks(("train",))

    def _evaluation_tasks(self, splits: Iterable[str], partition: Partition) -> tuple[EvaluationTaskInput, ...]:
        split_tuple = tuple(splits)
        ref = EvaluationArtifactRef(self.environment_id, self.config.package_version, sha256_json(self.manifest.model_dump(mode="json", by_alias=True)))
        return tuple(EvaluationTaskInput(task.task_id, ref, task.instruction, (), partition, f"appworld:{task.split}") for task in self.catalog.tasks(split_tuple, allow_test=self.config.allow_test))

    def tasks_for_partition(self, partition: Partition | str) -> tuple[EvaluationTaskInput, ...]:
        selected = {
            Partition.DEVELOPMENT: (("train",), Partition.DEVELOPMENT),
            Partition.VALIDATION: (("dev",), Partition.VALIDATION),
            Partition.FINAL: (("test_normal", "test_challenge"), Partition.FINAL),
        }.get(Partition(partition))
        if selected and selected[1] is Partition.FINAL and not self.config.allow_test:
            return ()
        return self._evaluation_tasks(*selected) if selected else ()

    def task_families(self, partition: Partition | str) -> tuple[str, ...]:
        return tuple(sorted({task.family for task in self.tasks_for_partition(partition)}))

    def partition_hash(self, partition: Partition | str) -> str:
        tasks = self.tasks_for_partition(partition)
        return sha256_json({"environmentId": self.environment_id, "partition": Partition(partition).value, "taskIds": [task.task_id for task in tasks], "publicData": self.catalog.public_data_hash(), "dataset": self.catalog.dataset_hash()})

    def structural_signature(self, partition: Partition | str) -> str:
        return sha256_json({"family": self.task_families(partition), "count": len(self.tasks_for_partition(partition))})

    def public_fixture_hash(self) -> str:
        return sha256_json({"manifest": self.manifest.model_dump(mode="json", by_alias=True), "publicData": self.catalog.public_data_hash(), "dataset": self.catalog.dataset_hash()})

    def learner_documents(self) -> tuple[Any, ...]:
        return ()

    def task_provenance(self, task_id: str) -> dict[str, Any]:
        for split in ("train", "dev", "test_normal", "test_challenge"):
            if task_id in self.catalog.split_ids(split):
                return {
                    "benchmark": "AppWorld",
                    "dataClass": "external_published_simulated_benchmark",
                    "officialSplit": split,
                    "runtimePartition": {"train": "development", "dev": "validation", "test_normal": "final", "test_challenge": "final"}[split],
                }
        raise AppWorldError(f"unknown AppWorld task: {task_id}")

    def reset(self, task_id: str, seed: int = 0) -> None:
        raise AppWorldError("AppWorld episodes are created through provider_factory")

    def evaluate(self, task_id: str, session: Any) -> Mapping[str, Any]:
        raise AppWorldError("AppWorld evaluation is owned by evaluate_provider")

    def provider_factory(self, task: Any, run_id: str, seed: int = 0) -> AppWorldProvider:
        task_id = getattr(task, "task_id", None) or getattr(task, "taskId", None)
        if not isinstance(task_id, str):
            raise AppWorldError("AppWorld provider factory requires a task ID")
        split = next((candidate for candidate in _ALLOWED_SPLITS if task_id in self.catalog.split_ids(candidate)), None)
        if split is None:
            raise AppWorldError("AppWorld task is not in a published split")
        if split in _TEST_SPLITS and not self.config.allow_test:
            raise AppWorldError("test AppWorld tasks are sealed until protocol freeze")
        return AppWorldProvider(self.config, self.catalog.task(task_id, split), run_id, seed)

    def evaluate_provider(self, provider: AppWorldProvider) -> Mapping[str, Any]:
        result = provider.evaluate_aggregate()
        return {"passed": bool(result.get("success", False)), "score": 1.0 if result.get("success") else 0.0, "reliable": True, "aggregateEvaluation": result, "provenance": "AppWorld external published simulated benchmark"}

def setup_manifest(config: AppWorldConfig, output: Path | None = None) -> dict[str, Any]:
    """Verify the isolated package and write a public run manifest."""
    python = config.resolved_python()
    probe = subprocess.run([python, "-c", "import importlib.metadata as m; print(m.version('appworld'))"], text=True, capture_output=True, check=False)
    if probe.returncode != 0:
        raise AppWorldUnavailable(f"AppWorld Python cannot import appworld: {probe.stderr.strip()}")
    installed = probe.stdout.strip()
    if installed != config.package_version:
        raise AppWorldUnavailable(f"AppWorld version mismatch: expected {config.package_version}, got {installed}")
    manifest = AppWorldCatalog(config).runtime_manifest().to_dict()
    manifest["installedVersion"] = installed
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional AppWorld adapter setup and worker")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="verify isolated AppWorld and write a public manifest")
    setup.add_argument("--root", required=True)
    setup.add_argument("--python", default="")
    setup.add_argument("--package-version", default=PUBLISHED_APPWORLD_VERSION)
    setup.add_argument("--output", default="")
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    config = AppWorldConfig(Path(args.root), python=args.python, package_version=args.package_version)
    manifest = setup_manifest(config, Path(args.output) if args.output else None)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "APPWORLD_ENVIRONMENT_ID",
    "PUBLISHED_APPWORLD_VERSION",
    "AppWorldCatalog",
    "AppWorldConfig",
    "AppWorldError",
    "AppWorldProvider",
    "AppWorldRuntimeManifest",
    "AppWorldTask",
    "AppWorldUnavailable",
    "build_manifest",
    "main",
    "public_tool_schemas",
    "register_appworld",
    "setup_manifest",
]

if __name__ == "__main__":
    raise SystemExit(main())
