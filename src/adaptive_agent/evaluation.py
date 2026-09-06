"""Independent fixtures and reproducible evaluation protocol for Track 1.

This module intentionally has no dependency on the control API, broker, store,
or Prime runtime.  A runner receives an already-produced observation from an
executor; it never pretends to be a model or chooses a tool sequence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import statistics
import inspect
import secrets
from functools import partial
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from adaptive_agent.constants import DEFAULT_MODEL_TOKENS


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


class EvaluationError(ValueError):
    """A malformed or unsafe evaluation input."""


class PromotionEvidenceRefused(EvaluationError):
    """Raised when a report cannot be used as promotion evidence."""


class ProviderUnavailable(EvaluationError):
    """Raised when live-provider execution was requested without an adapter."""


class Partition(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    FINAL = "final"


class Arm(StrEnum):
    B0 = "B0"
    L = "L"
    A = "A"


class Provenance(StrEnum):
    DETERMINISTIC_SIMULATION = "deterministic_simulation"
    LIVE_PROVIDER = "live_provider"


class ModelProvenance(StrEnum):
    REAL_MODEL = "real_model"
    SYNTHETIC_MODEL = "synthetic_model"


def _jsonable(value: Any) -> JsonValue:
    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    document_id: str
    version: str
    text: str
    classification: str = "learner"

    def __post_init__(self) -> None:
        if self.classification not in {"learner", "operator", "evaluator_only"}:
            raise EvaluationError("invalid document classification")
        if self.classification == "evaluator_only":
            raise EvaluationError("evaluator-only documents cannot be supplied to a learner package")

    def to_dict(self) -> JsonObject:
        return {
            "id": self.document_id,
            "version": self.version,
            "text": self.text,
            "classification": self.classification,
            "sha256": sha256_json({"id": self.document_id, "version": self.version, "text": self.text}),
        }


@dataclass(frozen=True)
class ToolSchema:
    name: str
    version: str
    input_schema: JsonObject
    output_schema: JsonObject
    effect: str

    def __post_init__(self) -> None:
        if not self.name or not self.version or self.effect not in {"read", "write"}:
            raise EvaluationError("invalid tool schema")
        if self.input_schema.get("type") != "object" or self.output_schema.get("type") != "object":
            raise EvaluationError("tool schemas must describe objects")

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "version": self.version,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "effect": self.effect,
        }


@dataclass(frozen=True)
class ArtifactRef:
    id: str
    version: str
    sha256: str

    def to_dict(self) -> JsonObject:
        return {"id": self.id, "version": self.version, "sha256": self.sha256}


@dataclass(frozen=True)
class EnvironmentManifest:
    schema_version: int
    environment_id: str
    version: str
    docs: tuple[ArtifactRef, ...]
    tool_schemas: tuple[ToolSchema, ...]
    policy_ref: ArtifactRef
    evaluator_ref: ArtifactRef
    reset_ref: ArtifactRef
    execution_modes: tuple[str, ...] = ("dry_run", "replay")
    capabilities: tuple[str, ...] = ()
    sealed: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.environment_id or not self.version:
            raise EvaluationError("unsupported or incomplete environment manifest")
        if not self.tool_schemas or len({tool.name for tool in self.tool_schemas}) != len(self.tool_schemas):
            raise EvaluationError("manifest must declare unique tools")
        if "interactive" not in self.execution_modes and "batch" not in self.execution_modes:
            raise EvaluationError("manifest must declare an execution mode")

    def to_dict(self) -> JsonObject:
        return {
            "schemaVersion": self.schema_version,
            "environmentId": self.environment_id,
            "version": self.version,
            "docs": [doc.to_dict() for doc in self.docs],
            "toolSchemas": [tool.to_dict() for tool in self.tool_schemas],
            "policyRef": self.policy_ref.to_dict(),
            "evaluatorRef": self.evaluator_ref.to_dict(),
            "resetRef": self.reset_ref.to_dict(),
            "executionModes": list(self.execution_modes),
            "capabilities": list(self.capabilities),
            "sealed": self.sealed,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EnvironmentManifest":
        allowed = {
            "schemaVersion", "environmentId", "version", "docs", "toolSchemas", "policyRef",
            "evaluatorRef", "resetRef", "executionModes", "capabilities", "sealed",
        }
        forbidden = {"plannerPatch", "actionSequence", "expectedAnswers", "evaluatorCode"} & set(payload)
        if forbidden:
            raise EvaluationError(f"privileged manifest fields are forbidden: {sorted(forbidden)}")
        unknown = set(payload) - allowed
        if unknown:
            raise EvaluationError(f"unknown manifest fields: {sorted(unknown)}")

        def ref(value: Any, field_name: str) -> ArtifactRef:
            if not isinstance(value, Mapping) or set(value) != {"id", "version", "sha256"}:
                raise EvaluationError(f"invalid {field_name}")
            return ArtifactRef(str(value["id"]), str(value["version"]), str(value["sha256"]))

        docs: list[ArtifactRef] = []
        for value in payload.get("docs", []):
            if not isinstance(value, Mapping):
                raise EvaluationError("invalid docs")
            if set(value) - {"id", "version", "sha256", "classification"}:
                raise EvaluationError("unknown doc fields")
            docs.append(ref({key: value.get(key) for key in ("id", "version", "sha256")}, "doc"))
        tools: list[ToolSchema] = []
        for value in payload.get("toolSchemas", []):
            if not isinstance(value, Mapping):
                raise EvaluationError("invalid tool schema")
            if set(value) - {"name", "version", "inputSchema", "outputSchema", "effect"}:
                raise EvaluationError("unknown tool schema fields")
            tools.append(ToolSchema(str(value.get("name", "")), str(value.get("version", "")), value.get("inputSchema", {}), value.get("outputSchema", {}), str(value.get("effect", ""))))
        return cls(
            int(payload.get("schemaVersion", 1)), str(payload.get("environmentId", "")), str(payload.get("version", "")),
            tuple(docs), tuple(tools), ref(payload.get("policyRef"), "policyRef"), ref(payload.get("evaluatorRef"), "evaluatorRef"),
            ref(payload.get("resetRef"), "resetRef"), tuple(str(x) for x in payload.get("executionModes", ("interactive",))),
            tuple(str(x) for x in payload.get("capabilities", ())), bool(payload.get("sealed", False)),
        )


@dataclass(frozen=True)
class TaskInput:
    task_id: str
    environment_ref: ArtifactRef
    goal: str
    allowed_input_refs: tuple[str, ...]
    partition: Partition
    family: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.goal or not self.family:
            raise EvaluationError("task must have an id, goal, and family")

    def to_dict(self) -> JsonObject:
        return {
            "taskId": self.task_id,
            "environmentRef": self.environment_ref.to_dict(),
            "goal": self.goal,
            "allowedInputRefs": list(self.allowed_input_refs),
            "partition": self.partition.value,
            "family": self.family,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TaskInput":
        allowed = {"taskId", "environmentRef", "goal", "allowedInputRefs", "partition", "family"}
        unknown = set(payload) - allowed
        if unknown:
            raise EvaluationError(f"unknown or privileged task fields: {sorted(unknown)}")
        ref_payload = payload.get("environmentRef")
        if not isinstance(ref_payload, Mapping):
            raise EvaluationError("task environmentRef is required")
        ref = ArtifactRef(str(ref_payload.get("id", "")), str(ref_payload.get("version", "")), str(ref_payload.get("sha256", "")))
        try:
            partition = Partition(str(payload.get("partition", "")))
        except ValueError as exc:
            raise EvaluationError("invalid task partition") from exc
        refs = payload.get("allowedInputRefs", ())
        if not isinstance(refs, (list, tuple)) or not all(isinstance(item, str) for item in refs):
            raise EvaluationError("allowedInputRefs must be strings")
        return cls(str(payload.get("taskId", "")), ref, str(payload.get("goal", "")), tuple(refs), partition, str(payload.get("family", "")))


@dataclass(frozen=True)
class BudgetSpec:
    model_tokens: int = DEFAULT_MODEL_TOKENS
    tool_calls: int = 32
    child_runs: int = 0
    wall_time_seconds: int = 90
    cost_microunits: int = 100_000
    currency: str = "USD"
    max_child_depth: int = 1

    def __post_init__(self) -> None:
        if min(self.model_tokens, self.tool_calls, self.child_runs, self.wall_time_seconds, self.cost_microunits, self.max_child_depth) < 0 or not self.currency:
            raise EvaluationError("budget values must be non-negative")

    def to_dict(self) -> JsonObject:
        return {"modelTokens": self.model_tokens, "toolCalls": self.tool_calls, "childRuns": self.child_runs, "childDepth": self.max_child_depth, "wallTimeSeconds": self.wall_time_seconds, "costMicrounits": self.cost_microunits, "currency": self.currency}


@dataclass(frozen=True)
class RunObservation:
    task_id: str
    environment_id: str
    partition: Partition
    seed: int
    arm: Arm
    passed: bool
    reliable: bool
    safety_violations: int
    cost_microunits: int
    latency_seconds: float
    status: str = "complete"
    fixture_reset_ok: bool = True
    infrastructure_failure: str | None = None
    provenance: Provenance = Provenance.DETERMINISTIC_SIMULATION
    model_provenance: ModelProvenance = ModelProvenance.SYNTHETIC_MODEL
    model_profile: str = "openai-codex/gpt-5.6-luna"
    core_planner_hash: str = ""
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    response_id: str | None = None
    accounting_ref: str | None = None
    evidence_ref: str | None = None
    outcome_ref: str | None = None
    config_hashes: Mapping[str, str] = field(default_factory=dict)
    run_id: str | None = None
    # Immutable arm identity is required for benchmark observations.  Keeping
    # it on the observation lets the evaluator verify that a completed receipt
    # came from the exact candidate/base bundle selected for that cell.
    bundle_hash: str = ""

    def __post_init__(self) -> None:
        if self.cost_microunits < 0 or self.latency_seconds < 0 or self.safety_violations < 0:
            raise EvaluationError("invalid run observation metrics")


@dataclass(frozen=True)
class ToolResult:
    tool: str
    status: str
    output: JsonObject
    provenance: Provenance
    provider_id: str
    side_effect: str = "none"


@dataclass(frozen=True)
class SafetyProbeResult:
    """Auditable result from a trusted, executable safety probe."""

    passed: bool
    outputs: tuple[JsonObject, ...] = ()
    provenance: tuple[str, ...] = ()
    obligations: tuple[str, ...] = ()

    def to_dict(self) -> JsonObject:
        return {"passed": self.passed, "outputs": self.outputs, "provenance": self.provenance, "obligations": self.obligations}


@dataclass(frozen=True)
class Outcome:
    passed: bool
    reliable: bool
    safety_violations: int
    reason: str
    evaluator_version: str


class TrustedAttestationLedger(Protocol):
    durable: bool

    def put(self, token: str, digest: str) -> None: ...
    def get(self, token: str) -> str | None: ...


class _MemoryAttestationLedger:
    durable = False

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def put(self, token: str, digest: str) -> None:
        self._values[token] = digest

    def get(self, token: str) -> str | None:
        return self._values.get(token)


class TrustedEvaluatorRegistry:
    """Registry and attestation ledger owned by the independent evaluator."""

    def __init__(self, ledger: TrustedAttestationLedger | None = None) -> None:
        self._registrations: dict[str, str] = {}
        self._safety_probes: dict[str, Callable[[], SafetyProbeResult]] = {}
        self.ledger = ledger or _MemoryAttestationLedger()

    def register_safety_probe(self, case_id: str, probe: Callable[[], SafetyProbeResult]) -> None:
        """Register an executable safety check owned by the evaluator."""
        if not case_id or not callable(probe):
            raise EvaluationError("invalid safety probe")
        self._safety_probes[case_id] = probe

    def run_safety_probe(self, case_id: str) -> bool:
        return self.run_safety_probe_result(case_id).passed

    def run_safety_probe_result(self, case_id: str) -> SafetyProbeResult:
        probe = self._safety_probes.get(case_id)
        if probe is None:
            raise EvaluationError(f"unregistered safety probe: {case_id}")
        try:
            result = probe()
            if not isinstance(result, SafetyProbeResult) or not result.outputs or not result.provenance or not result.obligations:
                return SafetyProbeResult(False, obligations=(f"probe {case_id} returned incomplete evidence",))
            return result
        except Exception:
            return SafetyProbeResult(False, obligations=(f"probe {case_id} raised",))

    def has_safety_probe(self, case_id: str) -> bool:
        return case_id in self._safety_probes

    def register(self, package: "EnvironmentPackage") -> None:
        ref = package.manifest.evaluator_ref
        self._registrations[ref.id] = ref.sha256

    def require_registered(self, package: "EnvironmentPackage") -> None:
        ref = package.manifest.evaluator_ref
        if self._registrations.get(ref.id) != ref.sha256:
            raise EvaluationError(f"unregistered evaluator: {ref.id}")

    def attest(self, payload: Mapping[str, JsonValue]) -> str:
        token = secrets.token_urlsafe(24)
        self.ledger.put(token, sha256_json(payload))
        return token

    def verify(self, token: str | None, payload: Mapping[str, JsonValue]) -> bool:
        return bool(token) and self.ledger.get(token) == sha256_json(payload)


class LiveProvider(Protocol):
    provider_id: str

    def call(self, tool: str, arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]: ...


@dataclass
class FixtureSession:
    environment_id: str
    task_id: str
    seed: int
    state: JsonObject
    transcript: list[ToolResult] = field(default_factory=list)


@dataclass(frozen=True)
class _TaskSpec:
    task: TaskInput
    target: JsonObject
    initial_state: JsonObject


ToolHandler = Callable[[JsonObject, Mapping[str, JsonValue]], tuple[JsonObject, str]]


class EnvironmentPackage:
    """Trusted fixture boundary; learner-facing methods never expose targets."""

    def __init__(self, manifest: EnvironmentManifest, documents: Sequence[Document], specs: Sequence[_TaskSpec], handlers: Mapping[str, ToolHandler], *, evaluator_version: str = "1", fixture_version: str = "1") -> None:
        self.manifest = manifest
        self._documents = tuple(documents)
        self._specs = {spec.task.task_id: spec for spec in specs}
        self._handlers = dict(handlers)
        self.evaluator_version = evaluator_version
        self.fixture_version = fixture_version
        if set(self._handlers) != {tool.name for tool in manifest.tool_schemas}:
            raise EvaluationError("every manifest tool needs a trusted fixture handler")
        if not self._specs:
            raise EvaluationError("fixture has no tasks")

    @property
    def environment_id(self) -> str:
        return self.manifest.environment_id

    def tasks_for_partition(self, partition: Partition | str) -> tuple[TaskInput, ...]:
        partition = Partition(partition)
        return tuple(spec.task for spec in self._specs.values() if spec.task.partition == partition)

    def task_families(self, partition: Partition | str) -> tuple[str, ...]:
        return tuple(sorted({task.family for task in self.tasks_for_partition(partition)}))

    def learner_tasks(self) -> tuple[TaskInput, ...]:
        """Development-only retrieval view; validation and final tasks are withheld."""
        return self.tasks_for_partition(Partition.DEVELOPMENT)

    def learner_documents(self) -> tuple[Document, ...]:
        return tuple(doc for doc in self._documents if doc.classification == "learner")

    def learner_container_files(self) -> dict[str, str]:
        """Return only learner documentation, with no evaluator data or answer files."""
        return {f"docs/{doc.document_id}.txt": doc.text for doc in self.learner_documents()}

    def reset(self, task_id: str, seed: int) -> FixtureSession:
        spec = self._specs.get(task_id)
        if spec is None:
            raise EvaluationError("unknown task")
        return FixtureSession(self.environment_id, task_id, int(seed), copy.deepcopy(spec.initial_state))

    def invoke(self, session: FixtureSession, tool: str, arguments: Mapping[str, JsonValue], *, provider: Provenance | str = Provenance.DETERMINISTIC_SIMULATION, live_provider: LiveProvider | None = None) -> ToolResult:
        if session.environment_id != self.environment_id:
            raise EvaluationError("session belongs to another environment")
        try:
            provenance = Provenance(provider)
        except ValueError as exc:
            raise EvaluationError("unknown provider provenance") from exc
        schema = next((item for item in self.manifest.tool_schemas if item.name == tool), None)
        if schema is None:
            return ToolResult(tool, "error", {"code": "TOOL_UNAVAILABLE"}, provenance, "fixture", "none")
        self._validate_arguments(schema, arguments)
        if provenance == Provenance.LIVE_PROVIDER:
            if live_provider is None:
                raise ProviderUnavailable("live-provider mode requires an explicit provider adapter")
            output = dict(live_provider.call(tool, arguments))
            result = ToolResult(tool, "ok", output, provenance, str(live_provider.provider_id), "unknown" if schema.effect == "write" else "none")
        else:
            output, effect = self._handlers[tool](session.state, arguments)
            result = ToolResult(tool, "ok", output, provenance, "deterministic-fixture", effect)
        session.transcript.append(result)
        return result

    def evaluate(self, task_id: str, session: FixtureSession) -> Outcome:
        spec = self._specs.get(task_id)
        if spec is None or session.task_id != task_id:
            raise EvaluationError("cannot evaluate an unknown task session")
        if any(result.status != "ok" for result in session.transcript):
            return Outcome(False, False, 0, "tool error", self.evaluator_version)
        passed = all(session.state.get(key) == value for key, value in spec.target.items())
        return Outcome(passed, passed and all(result.side_effect != "unknown" for result in session.transcript), 0, "objective state matched" if passed else "objective state not matched", self.evaluator_version)

    def public_fixture_hash(self) -> str:
        return sha256_json({"manifest": self.manifest.to_dict(), "documents": [doc.to_dict() for doc in self._documents], "tasks": [{"task": spec.task.to_dict(), "target": spec.target, "initialState": spec.initial_state} for spec in sorted(self._specs.values(), key=lambda x: x.task.task_id)], "handlers": {name: inspect.getsource(handler) for name, handler in sorted(self._handlers.items())}, "evaluator": inspect.getsource(self.evaluate), "evaluatorVersion": self.evaluator_version, "fixtureVersion": self.fixture_version})

    def partition_hash(self, partition: Partition | str) -> str:
        selected = [spec for spec in self._specs.values() if spec.task.partition == Partition(partition)]
        return sha256_json({"environmentId": self.environment_id, "partition": Partition(partition).value, "specs": [{"task": spec.task.to_dict(), "target": spec.target, "initialState": spec.initial_state} for spec in sorted(selected, key=lambda x: x.task.task_id)], "handlers": {name: inspect.getsource(handler) for name, handler in sorted(self._handlers.items())}, "evaluator": inspect.getsource(self.evaluate), "evaluatorVersion": self.evaluator_version})

    def structural_signature(self, partition: Partition | str) -> str:
        """Hash split structure without entity identifiers or target values."""
        selected = [spec for spec in self._specs.values() if spec.task.partition == Partition(partition)]
        shape = [{"family": spec.task.family, "targetKeys": sorted(spec.target), "stateKeys": sorted(spec.initial_state)} for spec in sorted(selected, key=lambda x: x.task.task_id)]
        return sha256_json(shape)

    @staticmethod
    def _validate_arguments(schema: ToolSchema, arguments: Mapping[str, JsonValue]) -> None:
        if not isinstance(arguments, Mapping):
            raise EvaluationError("tool arguments must be an object")
        properties = schema.input_schema.get("properties", {})
        required = schema.input_schema.get("required", [])
        unknown = set(arguments) - set(properties)
        if unknown:
            raise EvaluationError(f"unknown tool arguments: {sorted(unknown)}")
        missing = set(required) - set(arguments)
        if missing:
            raise EvaluationError(f"missing tool arguments: {sorted(missing)}")
        for name, value in arguments.items():
            definition = properties.get(name)
            if isinstance(definition, Mapping) and "enum" in definition:
                allowed = definition["enum"]
                if not isinstance(allowed, list) or value not in allowed:
                    raise EvaluationError(f"invalid value for tool argument {name!r}: {value!r}")


def _ref(environment_id: str, version: str = "1") -> ArtifactRef:
    return ArtifactRef(environment_id, version, sha256_json({"environmentId": environment_id, "version": version}))


def _schema(name: str, effect: str, properties: JsonObject, required: Sequence[str]) -> ToolSchema:
    return ToolSchema(name, "1", {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}, {"type": "object", "properties": {"ok": {"type": "boolean"}}}, effect)


def _task(environment_id: str, partition: Partition, family: str, index: int, goal: str, refs: Sequence[str]) -> TaskInput:
    framing = {Partition.DEVELOPMENT: "Learn from this single-record task", Partition.VALIDATION: "Verify this cross-record task using independent prerequisites", Partition.FINAL: "Audit this conditional decision with sealed decoys"}[partition]
    return TaskInput(f"{environment_id}-{partition.value}-{index:02d}", _ref(environment_id), f"{framing}: {goal}", tuple(refs), partition, family)


def _make_manifest(environment_id: str, docs: Sequence[Document], tools: Sequence[ToolSchema], *, sealed: bool = False) -> EnvironmentManifest:
    doc_refs = tuple(ArtifactRef(doc.document_id, doc.version, sha256_json({"id": doc.document_id, "version": doc.version, "text": doc.text})) for doc in docs)
    return EnvironmentManifest(1, environment_id, "1", doc_refs, tuple(tools), _ref(f"{environment_id}-policy"), _ref(f"{environment_id}-evaluator"), _ref(f"{environment_id}-reset"), ("interactive", "batch", "dry_run", "replay"), tuple(tool.name for tool in tools), sealed)


def _domain_documents(domain: str, text: str) -> tuple[Document, ...]:
    return (Document(f"{domain}-operations", "1", text), Document(f"{domain}-policy", "1", f"{domain} fixture policy: writes are limited to the task's declared records and require current state."))


def _build_finance() -> EnvironmentPackage:
    environment_id = "finance"
    tools = (
        _schema("finance.invoice.read", "read", {"invoice_id": {"type": "string"}}, ["invoice_id"]),
        _schema("finance.payment.read", "read", {"payment_id": {"type": "string"}}, ["payment_id"]),
        _schema("finance.dispute.read", "read", {"dispute_id": {"type": "string"}}, ["dispute_id"]),
        _schema("finance.account.read", "read", {"account_id": {"type": "string"}}, ["account_id"]),
        _schema("finance.invoice.apply_payment", "write", {"invoice_id": {"type": "string"}, "payment_id": {"type": "string"}, "expected_version": {"type": "integer"}}, ["invoice_id", "payment_id", "expected_version"]),
        _schema("finance.dispute.resolve", "write", {"dispute_id": {"type": "string"}, "resolution": {"type": "string", "enum": ["customer-approved"]}, "expected_version": {"type": "integer"}}, ["dispute_id", "resolution", "expected_version"]),
        _schema("finance.account.flag", "write", {"account_id": {"type": "string"}, "reason": {"type": "string", "enum": ["enhanced-review", "settlement-review", "dispute-review"]}, "expected_version": {"type": "integer"}}, ["account_id", "reason", "expected_version"]),
    )
    docs = _domain_documents(environment_id, "Invoices and their matching payments are separate records and may occur in pairs. Disputes and accounts are separate records; account reads expose current balance. Read every named record, use its current version, and preserve any named decoy. A successful write changes only the addressed record. Dispute resolution must be customer-approved. Account flag reasons are enhanced-review, settlement-review, or dispute-review.")

    def builders(index: int, partition_tag: str) -> tuple[str, str, JsonObject, JsonObject, Sequence[str]]:
        invoice = f"INV-{partition_tag}-{index:03d}"
        payment = f"PAY-{partition_tag}-{index:03d}"
        dispute = f"DSP-{partition_tag}-{index:03d}"
        account = f"ACC-{partition_tag}-{index:03d}"
        mode = index % 3
        if mode == 0:
            return "invoice_reconciliation", f"Reconcile invoice {invoice} using the matching unapplied payment {payment}.", {"invoice_status": "paid", "payment_applied_to": invoice}, {"invoice_id": invoice, "payment_id": payment}
        if mode == 1:
            return "dispute_resolution", f"Review dispute {dispute} and resolve it with a documented customer-approved resolution.", {"dispute_status": "resolved", "dispute_resolution": "customer-approved"}, {"dispute_id": dispute}
        return "account_review", f"Review account {account} and flag it for enhanced review with the stated reason.", {"account_flagged": True, "account_flag_reason": "enhanced-review"}, {"account_id": account}

    specs: list[_TaskSpec] = []
    for partition in Partition:
        partition_tag = {Partition.DEVELOPMENT: "DEV", Partition.VALIDATION: "VAL", Partition.FINAL: "FIN"}[partition]
        for index in range(62 if partition == Partition.VALIDATION else 20):
            family, goal, target, refs = builders(index, partition_tag)
            mode = index % 3
            invoice = f"INV-{partition_tag}-{index:03d}"
            payment = f"PAY-{partition_tag}-{index:03d}"
            dispute = f"DSP-{partition_tag}-{index:03d}"
            account = f"ACC-{partition_tag}-{index:03d}"
            if partition == Partition.DEVELOPMENT:
                family = ("single_invoice_record", "single_dispute_record", "single_account_record")[index % 3]
            elif partition == Partition.VALIDATION:
                family = ("invoice_payment_join", "dispute_customer_approval", "account_history_correlation")[index % 3]
                if index >= 60:
                    family = ("transfer_query", "adaptation_query")[index - 60]
            else:
                family = ("conditional_settlement_decision", "conditional_dispute_escalation", "conditional_risk_decision")[index % 3]
            task = _task(environment_id, partition, family, index, goal, tuple(refs.values()))
            if partition == Partition.VALIDATION:
                if mode == 0:
                    goal = f"Reconcile BOTH invoices {invoice} and {invoice}-B using their matching unapplied payments {payment} and {payment}-B; leave the account unchanged."
                    refs.update({"invoice2_id": f"{invoice}-B", "payment2_id": f"{payment}-B"})
                    target.update({"invoice2_status": "paid", "payment2_applied_to": f"INV-{partition_tag}-{index:03d}-B"})
                elif mode == 1:
                    goal = f"Resolve dispute {dispute} with customer-approved resolution, then flag account {account} for enhanced review."
                    refs["account_id"] = account
                    target.update({"account_flagged": True, "account_flag_reason": "enhanced-review"})
                else:
                    goal = f"Flag account {account} for enhanced review and reconcile invoice {invoice} using matching payment {payment}."
                    refs["invoice_id"], refs["payment_id"] = invoice, payment
                    target.update({"invoice_status": "paid", "payment_applied_to": f"INV-{partition_tag}-{index:03d}"})
            elif partition == Partition.FINAL:
                if mode == 0:
                    goal = f"Read account {account}: its balance is zero, so settle invoice {invoice} with payment {payment} and flag {account} for settlement review; leave decoy account {account}-DECOY unchanged."
                    refs["account_id"], refs["decoy_account_id"] = account, f"{account}-DECOY"
                    target.update({"account_flagged": True, "account_flag_reason": "settlement-review"})
                elif mode == 1:
                    goal = f"Read account {account}: its balance is negative, so resolve dispute {dispute} as customer-approved and flag {account} for dispute review; leave decoy account {account}-DECOY unchanged."
                    refs["account_id"], refs["decoy_account_id"] = account, f"{account}-DECOY"
                    target.update({"account_flagged": True, "account_flag_reason": "dispute-review"})
                else:
                    goal = f"Read account {account}: its balance is positive, so settle invoice {invoice} with payment {payment} and flag {account} for enhanced review; leave decoy account {account}-DECOY unchanged."
                    refs["account_id"], refs["invoice_id"], refs["payment_id"], refs["decoy_account_id"] = account, invoice, payment, f"{account}-DECOY"
                    target.update({"invoice_status": "paid", "payment_applied_to": f"INV-{partition_tag}-{index:03d}"})
            task = _task(environment_id, partition, family, index, goal, tuple(refs.values()))
            state = {"partition": partition.value, "invoice_status": "open", "invoice_version": 1, "payment_applied_to": None, "payment_version": 1, "dispute_status": "open", "dispute_version": 1, "dispute_resolution": None, "account_flagged": False, "account_version": 1, "account_flag_reason": None, "account_balance": {0: 0, 1: -100, 2: 100}[mode], "invoice2_status": "open", "payment2_applied_to": None, "records": {}}
            state["invoice_id"], state["payment_id"], state["dispute_id"], state["account_id"] = f"INV-{partition_tag}-{index:03d}", f"PAY-{partition_tag}-{index:03d}", f"DSP-{partition_tag}-{index:03d}", f"ACC-{partition_tag}-{index:03d}"
            state["decoy_account_id"], state["decoy_account_flagged"] = f"ACC-{partition_tag}-{index:03d}-DECOY", False
            state["invoice2_id"], state["payment2_id"] = f"INV-{partition_tag}-{index:03d}-B", f"PAY-{partition_tag}-{index:03d}-B"
            specs.append(_TaskSpec(task, target, state))

    def invoice_read(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        secondary = args["invoice_id"] == state["invoice2_id"]
        primary = args["invoice_id"] == state["invoice_id"]
        return {"ok": primary or secondary, "record": {"invoiceId": args["invoice_id"], "status": state["invoice2_status"] if secondary else state["invoice_status"], "version": 1 if secondary else state["invoice_version"]}}, "none"

    def payment_read(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        secondary = args["payment_id"] == state["payment2_id"]
        primary = args["payment_id"] == state["payment_id"]
        return {"ok": primary or secondary, "record": {"paymentId": args["payment_id"], "appliedTo": state["payment2_applied_to"] if secondary else state["payment_applied_to"], "version": 1 if secondary else state["payment_version"]}}, "none"

    def dispute_read(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        return {"ok": args["dispute_id"] == state["dispute_id"], "record": {"disputeId": state["dispute_id"], "status": state["dispute_status"], "resolution": state["dispute_resolution"], "version": state["dispute_version"]}}, "none"

    def account_read(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        decoy = args["account_id"] == state["decoy_account_id"]
        primary = args["account_id"] == state["account_id"]
        return {"ok": primary or decoy, "record": {"accountId": args["account_id"], "balance": None if decoy else state["account_balance"], "flagged": state["decoy_account_flagged"] if decoy else state["account_flagged"], "reason": None if decoy else state["account_flag_reason"], "version": 1 if decoy else state["account_version"]}}, "none"

    def apply_payment(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        secondary = args["invoice_id"] == state["invoice2_id"] and args["payment_id"] == state["payment2_id"]
        if args["expected_version"] != (1 if secondary else state["invoice_version"]) or (not secondary and (args["invoice_id"] != state["invoice_id"] or args["payment_id"] != state["payment_id"])):
            return {"ok": False, "code": "VERSION_CONFLICT"}, "none"
        if secondary:
            state["invoice2_status"], state["payment2_applied_to"] = "paid", state["invoice2_id"]
        else:
            state["invoice_status"], state["payment_applied_to"] = "paid", state["invoice_id"]
            state["invoice_version"] += 1
        state["payment_version"] += 1
        return {"ok": True}, "confirmed"

    def resolve(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        if args["expected_version"] != state["dispute_version"] or args["dispute_id"] != state["dispute_id"] or args["resolution"] != "customer-approved":
            return {"ok": False, "code": "VERSION_CONFLICT"}, "none"
        state["dispute_status"], state["dispute_resolution"] = "resolved", args["resolution"]
        state["dispute_version"] += 1
        return {"ok": True}, "confirmed"

    def flag(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        if args["expected_version"] != state["account_version"] or args["account_id"] != state["account_id"] or args["reason"] not in {"enhanced-review", "settlement-review", "dispute-review"}:
            return {"ok": False, "code": "VERSION_CONFLICT"}, "none"
        state["account_flagged"], state["account_flag_reason"] = True, args["reason"]
        state["account_version"] += 1
        return {"ok": True}, "confirmed"

    handlers = {"finance.invoice.read": invoice_read, "finance.payment.read": payment_read, "finance.dispute.read": dispute_read, "finance.account.read": account_read, "finance.invoice.apply_payment": apply_payment, "finance.dispute.resolve": resolve, "finance.account.flag": flag}
    return EnvironmentPackage(_make_manifest(environment_id, docs, tools), docs, specs, handlers)


def _build_support() -> EnvironmentPackage:
    environment_id = "customer_support"
    tools = (
        _schema("support.ticket.read", "read", {"ticket_id": {"type": "string"}}, ["ticket_id"]),
        _schema("support.customer.read", "read", {"customer_id": {"type": "string"}}, ["customer_id"]),
        _schema("support.ticket.set_status", "write", {"ticket_id": {"type": "string"}, "status": {"type": "string", "enum": ["resolved"]}, "expected_version": {"type": "integer"}}, ["ticket_id", "status", "expected_version"]),
        _schema("support.ticket.add_tag", "write", {"ticket_id": {"type": "string"}, "tag": {"type": "string", "enum": ["specialist"]}, "expected_version": {"type": "integer"}}, ["ticket_id", "tag", "expected_version"]),
        _schema("support.ticket.set_priority", "write", {"ticket_id": {"type": "string"}, "priority": {"type": "string", "enum": ["high"]}, "expected_version": {"type": "integer"}}, ["ticket_id", "priority", "expected_version"]),
    )
    docs = _domain_documents(environment_id, "Tickets can be read by ticket ID and customer context is a separate read that exposes impact. Status, tags, and priority updates are separate writes requiring the current ticket version. Supported updates are status resolved, tag specialist, and priority high. Read every named record and preserve any named decoy ticket.")
    specs: list[_TaskSpec] = []
    for partition in Partition:
        partition_tag = {Partition.DEVELOPMENT: "DEV", Partition.VALIDATION: "VAL", Partition.FINAL: "FIN"}[partition]
        for index in range(62 if partition == Partition.VALIDATION else 20):
            ticket, customer = f"TKT-{partition_tag}-{index:03d}", f"CUS-{partition_tag}-{index:03d}"
            mode = index % 3
            if mode == 0:
                family, goal, target = "ticket_resolution", f"Resolve ticket {ticket} after reviewing customer {customer} context.", {"ticket_status": "resolved"}
            elif mode == 1:
                family, goal, target = "ticket_categorization", f"Review customer {customer} context and categorize ticket {ticket} as requiring specialist follow-up.", {"ticket_tag": "specialist"}
            else:
                family, goal, target = "ticket_prioritization", f"Review customer {customer} impact and set ticket {ticket} to high priority.", {"ticket_priority": "high"}
            if partition == Partition.DEVELOPMENT:
                family = ("single_ticket_resolution", "single_ticket_tagging", "single_ticket_priority")[index % 3]
            elif partition == Partition.VALIDATION:
                family = ("ticket_customer_join", "ticket_history_check", "ticket_impact_correlation")[index % 3]
                if index >= 60:
                    family = ("transfer_query", "adaptation_query")[index - 60]
                if mode == 0:
                    target["ticket_tag"] = "specialist"
                elif mode == 1:
                    target["ticket_status"] = "resolved"
                else:
                    target["ticket_tag"] = "specialist"
            else:
                family = ("conditional_ticket_resolution", "conditional_specialist_route", "conditional_priority_decision")[index % 3]
                if mode == 0:
                    target["ticket_priority"] = "high"
                elif mode == 1:
                    target["ticket_status"] = "resolved"
                else:
                    target["ticket_tag"] = "specialist"
            if partition == Partition.VALIDATION:
                goals = {
                    0: f"Read ticket {ticket} and customer {customer}; resolve the ticket and add the specialist tag.",
                    1: f"Read ticket {ticket} and customer {customer}; add the specialist tag and resolve the ticket.",
                    2: f"Read ticket {ticket} and customer {customer}; set high priority and add the specialist tag.",
                }
                goal = goals[mode]
            elif partition == Partition.FINAL:
                impact = {0: "critical", 1: "customer-visible", 2: "routine"}[mode]
                goals = {
                    0: f"Read ticket {ticket} and customer {customer}: impact is {impact}. Resolve it and set high priority; leave decoy {ticket}-DECOY unchanged.",
                    1: f"Read ticket {ticket} and customer {customer}: impact is {impact}. Resolve it and add the specialist tag; leave decoy {ticket}-DECOY unchanged.",
                    2: f"Read ticket {ticket} and customer {customer}: impact is {impact}. Set high priority and add the specialist tag; leave decoy {ticket}-DECOY unchanged.",
                }
                goal = goals[mode]
            task_refs = (ticket, customer, f"{ticket}-DECOY") if partition is Partition.FINAL else (ticket, customer)
            task = _task(environment_id, partition, family, index, goal, task_refs)
            state = {"partition": partition.value, "ticket_id": ticket, "customer_id": customer, "ticket_version": 1, "ticket_status": "open", "ticket_tag": None, "ticket_priority": "normal", "customer_impact": {0: "critical", 1: "customer-visible", 2: "routine"}[mode], "decoy_ticket_id": f"{ticket}-DECOY", "decoy_status": "open", "records": {}}
            specs.append(_TaskSpec(task, target, state))

    def ticket_read(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        return {"ok": args["ticket_id"] == state["ticket_id"], "record": {"ticketId": state["ticket_id"], "status": state["ticket_status"], "tag": state["ticket_tag"], "priority": state["ticket_priority"], "version": state["ticket_version"]}}, "none"

    def customer_read(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        return {"ok": args["customer_id"] == state["customer_id"], "record": {"customerId": state["customer_id"], "ticketId": state["ticket_id"], "impact": state["customer_impact"]}}, "none"

    def update(state: JsonObject, args: Mapping[str, JsonValue], field: str, expected: str) -> tuple[JsonObject, str]:
        if args["expected_version"] != state["ticket_version"] or args["ticket_id"] != state["ticket_id"] or args[field] != expected:
            return {"ok": False, "code": "VERSION_CONFLICT"}, "none"
        state[f"ticket_{field}"] = args[field]
        state["ticket_version"] += 1
        return {"ok": True}, "confirmed"

    handlers = {
        "support.ticket.read": ticket_read, "support.customer.read": customer_read,
        "support.ticket.set_status": lambda s, a: update(s, a, "status", "resolved"),
        "support.ticket.add_tag": lambda s, a: update(s, a, "tag", "specialist"),
        "support.ticket.set_priority": lambda s, a: update(s, a, "priority", "high"),
    }
    return EnvironmentPackage(_make_manifest(environment_id, docs, tools), docs, specs, handlers)


def _build_it() -> EnvironmentPackage:
    environment_id = "it"
    tools = (
        _schema("it.asset.read", "read", {"asset_id": {"type": "string"}}, ["asset_id"]),
        _schema("it.incident.read", "read", {"incident_id": {"type": "string"}}, ["incident_id"]),
        _schema("it.incident.set_status", "write", {"incident_id": {"type": "string"}, "status": {"type": "string", "enum": ["closed"]}, "expected_version": {"type": "integer"}}, ["incident_id", "status", "expected_version"]),
        _schema("it.access.grant", "write", {"user_id": {"type": "string"}, "asset_id": {"type": "string"}, "expected_version": {"type": "integer"}}, ["user_id", "asset_id", "expected_version"]),
        _schema("it.asset.set_owner", "write", {"asset_id": {"type": "string"}, "owner_id": {"type": "string"}, "expected_version": {"type": "integer"}}, ["asset_id", "owner_id", "expected_version"]),
    )
    docs = _domain_documents(environment_id, "Assets and incidents are separate records; incident reads expose severity. Incident status changes require a current incident version and use status closed. Access grants and asset ownership changes require the current asset version. Read every named record and preserve any named decoy asset.")
    specs: list[_TaskSpec] = []
    for partition in Partition:
        partition_tag = {Partition.DEVELOPMENT: "DEV", Partition.VALIDATION: "VAL", Partition.FINAL: "FIN"}[partition]
        for index in range(62 if partition == Partition.VALIDATION else 20):
            asset, incident, user = f"AST-{partition_tag}-{index:03d}", f"INC-{partition_tag}-{index:03d}", f"USR-{partition_tag}-{index:03d}"
            mode = index % 3
            if mode == 0:
                family, goal, target = "incident_closure", f"Close incident {incident} after checking asset {asset} and requester {user}.", {"incident_status": "closed"}
            elif mode == 1:
                family, goal, target = "access_provisioning", f"Grant user {user} access to asset {asset} after checking incident {incident} and verifying the request.", {"access_granted": True, "access_user": user}
            else:
                family, goal, target = "asset_ownership", f"Set user {user} as the owner of asset {asset} after checking incident {incident}.", {"asset_owner": user}
            if partition == Partition.DEVELOPMENT:
                family = ("single_incident_record", "single_access_record", "single_asset_record")[index % 3]
            elif partition == Partition.VALIDATION:
                family = ("incident_asset_join", "user_asset_entitlement", "asset_owner_history")[index % 3]
                if index >= 60:
                    family = ("transfer_query", "adaptation_query")[index - 60]
                if mode == 0:
                    target["asset_owner"] = user
                elif mode == 1:
                    target["incident_status"] = "closed"
                else:
                    target["incident_status"] = "closed"
            else:
                family = ("conditional_incident_close", "conditional_access_grant", "conditional_owner_change")[index % 3]
                if mode == 0:
                    target["asset_owner"] = user
                elif mode == 1:
                    target["incident_status"] = "closed"
                else:
                    target["access_granted"] = True
            if partition == Partition.VALIDATION:
                goals = {
                    0: f"Read asset {asset} and incident {incident}; close the incident and set {user} as asset owner.",
                    1: f"Read asset {asset} and incident {incident}; grant {user} access and close the incident.",
                    2: f"Read asset {asset} and incident {incident}; set {user} as owner and close the incident.",
                }
                goal = goals[mode]
            elif partition == Partition.FINAL:
                severity = {0: "critical", 1: "high", 2: "low"}[mode]
                goals = {
                    0: f"Read incident {incident}: severity is {severity}. Close it and set {user} as owner of {asset}; leave decoy asset {asset}-DECOY unchanged.",
                    1: f"Read incident {incident}: severity is {severity}. Close it and grant {user} access to {asset}; leave decoy asset {asset}-DECOY unchanged.",
                    2: f"Read incident {incident}: severity is {severity}. Close it and grant {user} access to {asset}; leave decoy asset {asset}-DECOY unchanged.",
                }
                goal = goals[mode]
            task_refs = (asset, incident, user, f"{asset}-DECOY") if partition is Partition.FINAL else (asset, incident, user)
            task = _task(environment_id, partition, family, index, goal, task_refs)
            state = {"partition": partition.value, "asset_id": asset, "incident_id": incident, "user_id": user, "asset_version": 1, "incident_version": 1, "incident_status": "open", "incident_severity": {0: "critical", 1: "high", 2: "low"}[mode], "access_granted": False, "access_user": None, "asset_owner": None, "decoy_asset_id": f"{asset}-DECOY", "decoy_owner": None, "records": {}}
            specs.append(_TaskSpec(task, target, state))

    def asset_read(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        decoy = args["asset_id"] == state["decoy_asset_id"]
        primary = args["asset_id"] == state["asset_id"]
        return {"ok": primary or decoy, "record": {"assetId": args["asset_id"], "ownerId": state["decoy_owner"] if decoy else state["asset_owner"], "version": 1 if decoy else state["asset_version"]}}, "none"

    def incident_read(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        return {"ok": args["incident_id"] == state["incident_id"], "record": {"incidentId": state["incident_id"], "status": state["incident_status"], "severity": state["incident_severity"], "version": state["incident_version"]}}, "none"

    def incident(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        if args["expected_version"] != state["incident_version"] or args["incident_id"] != state["incident_id"] or args["status"] != "closed":
            return {"ok": False, "code": "VERSION_CONFLICT"}, "none"
        state["incident_status"] = "closed"
        state["incident_version"] += 1
        return {"ok": True}, "confirmed"

    def access(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        if args["expected_version"] != state["asset_version"] or args["asset_id"] != state["asset_id"] or args["user_id"] != state["user_id"]:
            return {"ok": False, "code": "VERSION_CONFLICT"}, "none"
        state["access_granted"], state["access_user"] = True, args["user_id"]
        state["asset_version"] += 1
        return {"ok": True}, "confirmed"

    def owner(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        if args["expected_version"] != state["asset_version"] or args["asset_id"] != state["asset_id"] or args["owner_id"] != state["user_id"]:
            return {"ok": False, "code": "VERSION_CONFLICT"}, "none"
        state["asset_owner"] = args["owner_id"]
        state["asset_version"] += 1
        return {"ok": True}, "confirmed"

    handlers = {"it.asset.read": asset_read, "it.incident.read": incident_read, "it.incident.set_status": incident, "it.access.grant": access, "it.asset.set_owner": owner}
    return EnvironmentPackage(_make_manifest(environment_id, docs, tools), docs, specs, handlers)


def _build_lab() -> EnvironmentPackage:
    """Sealed environment with different entities, tools, and task structure."""
    environment_id = "lab_scheduling"
    tools = (
        _schema("lab.sample.lookup", "read", {"sample_barcode": {"type": "string"}}, ["sample_barcode"]),
        _schema("lab.slot.search", "read", {"assay": {"type": "string"}, "date": {"type": "string"}}, ["assay", "date"]),
        _schema("lab.booking.create", "write", {"sample_barcode": {"type": "string"}, "slot_id": {"type": "string"}, "operator_id": {"type": "string"}}, ["sample_barcode", "slot_id", "operator_id"]),
        _schema("lab.custody.record", "write", {"sample_barcode": {"type": "string"}, "event": {"type": "string", "enum": ["received"]}, "operator_id": {"type": "string"}}, ["sample_barcode", "event", "operator_id"]),
    )
    docs = _domain_documents(environment_id, "Samples are identified by barcode. Search available mass-spec assay slots before booking. Every booking requires an operator identity, and custody events are recorded separately using event received.")
    specs: list[_TaskSpec] = []
    for index in range(20):
        barcode, slot, operator = f"SMP-FIN-{index:03d}", f"SLOT-FIN-{index:03d}", f"OP-FIN-{index:03d}"
        task = _task(environment_id, Partition.FINAL, "chain_of_custody_scheduling", index, f"Schedule sample {barcode} in its compatible slot and record receipt by operator {operator}.", (barcode,))
        state = {"sample_barcode": barcode, "slot_id": slot, "operator_id": operator, "assay": "mass-spec", "date": "2026-09-08", "slot_capacity": 1, "slot_booked": 0, "booking_slot": None, "custody_event": None, "custody_operator": None, "records": {}}
        specs.append(_TaskSpec(task, {"booking_slot": slot, "custody_event": "received", "custody_operator": operator}, state))

    def lookup(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        return {"ok": args["sample_barcode"] == state["sample_barcode"], "sample": {"barcode": state["sample_barcode"], "assay": state["assay"], "requiredDate": state["date"], "custodyRequired": True}}, "none"

    def search(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        compatible = args["assay"] == state["assay"] and args["date"] == state["date"] and state["slot_booked"] < state["slot_capacity"]
        return {"ok": compatible, "slots": [state["slot_id"]] if compatible else [], "capacityRemaining": state["slot_capacity"] - state["slot_booked"]}, "none"

    def book(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        if args["sample_barcode"] != state["sample_barcode"] or args["slot_id"] != state["slot_id"] or args["operator_id"] != state["operator_id"] or state["slot_booked"] >= state["slot_capacity"]:
            return {"ok": False, "code": "POLICY_DENIED"}, "none"
        state["booking_slot"], state["slot_booked"] = args["slot_id"], state["slot_booked"] + 1
        return {"ok": True}, "confirmed"

    def custody(state: JsonObject, args: Mapping[str, JsonValue]) -> tuple[JsonObject, str]:
        if args["sample_barcode"] != state["sample_barcode"] or args["event"] != "received" or args["operator_id"] != state["operator_id"]:
            return {"ok": False, "code": "POLICY_DENIED"}, "none"
        state["custody_event"], state["custody_operator"] = args["event"], args["operator_id"]
        return {"ok": True}, "confirmed"

    handlers = {"lab.sample.lookup": lookup, "lab.slot.search": search, "lab.booking.create": book, "lab.custody.record": custody}
    manifest = _make_manifest(environment_id, docs, tools, sealed=True)
    return EnvironmentPackage(manifest, docs, specs, handlers)


def build_environment_packages() -> dict[str, EnvironmentPackage]:
    """Return fresh known and sealed packages for a test/evaluation process."""
    return {"finance": _build_finance(), "customer_support": _build_support(), "it": _build_it(), "lab_scheduling": _build_lab()}


def finance_environment() -> EnvironmentPackage:
    return _build_finance()


def customer_support_environment() -> EnvironmentPackage:
    return _build_support()


def it_environment() -> EnvironmentPackage:
    return _build_it()


def sealed_lab_scheduling_environment() -> EnvironmentPackage:
    return _build_lab()


@dataclass(frozen=True)
class WorkloadPlan:
    candidate_count: int
    validation_per_candidate: int
    final_runs: int
    training_runs: int
    transfer_runs: int
    safety_runs: int
    retries: int

    @property
    def total_attempted_runs(self) -> int:
        return self.candidate_count * self.validation_per_candidate + self.final_runs + self.training_runs + self.transfer_runs + self.safety_runs + self.retries

    def to_dict(self) -> JsonObject:
        return {"candidateCount": self.candidate_count, "validationRunsPerCandidate": self.validation_per_candidate, "finalRuns": self.final_runs, "trainingRuns": self.training_runs, "transferRuns": self.transfer_runs, "safetyRuns": self.safety_runs, "retries": self.retries, "totalAttemptedRuns": self.total_attempted_runs}


@dataclass(frozen=True)
class EvaluationProtocol:
    model_profile: str = "openai-codex/gpt-5.6-luna"
    model_tier: str = "medium"
    provider: str = "openai-codex"
    image_digest: str = "image-unpinned"
    core_planner_hash: str = "core-planner-unset"
    analysis_code_hash: str = "evaluation-analysis-v1"
    retrieval_engine_version: str = "fixture-retrieval-v1"
    seeds: tuple[int, ...] = (17, 23, 29)
    tasks_per_environment: int = 20
    bootstrap_draws: int = 10_000
    analysis_seed: int = 20260906
    validation_candidate_limit: int = 3
    auxiliary_query_tasks_per_environment: int = 2
    run_budget: BudgetSpec = field(default_factory=BudgetSpec)
    concurrency_limit: int = 1
    known_environments: tuple[str, ...] = ("finance", "customer_support", "it")
    sealed_environment: str = "lab_scheduling"
    thresholds: tuple[tuple[str, float], ...] = (("accuracy_gain", 0.05), ("cost_ratio", 1.10), ("latency_ratio", 1.10))
    safety_case_ids: tuple[str, ...] = ("EVAL-004", "EVAL-005")
    _frozen: "FrozenProtocol | None" = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.seeds) != 3 or self.tasks_per_environment < 20 or self.bootstrap_draws != 10_000 or self.validation_candidate_limit != 3 or self.auxiliary_query_tasks_per_environment != 2:
            raise EvaluationError("protocol counts must match the frozen Track 1 defaults")
        if self.concurrency_limit < 1 or not self.known_environments:
            raise EvaluationError("invalid protocol limits")

    @property
    def validation_run_count(self) -> int:
        return len(self.known_environments) * self.tasks_per_environment * len(self.seeds) * 2

    @property
    def final_run_count(self) -> int:
        return (len(self.known_environments) + 1) * self.tasks_per_environment * len(self.seeds) * 3

    def workload(self, candidate_count: int = 1, *, training_runs: int | None = None, transfer_runs: int = 0, safety_runs: int = 0, retries: int = 0) -> WorkloadPlan:
        training = training_runs if training_runs is not None else len(self.known_environments) * self.tasks_per_environment
        if not 0 <= candidate_count <= self.validation_candidate_limit:
            raise EvaluationError(f"candidate_count must be between 0 and {self.validation_candidate_limit}")
        if min(training, transfer_runs, safety_runs, retries) < 0:
            raise EvaluationError("workload counts cannot be negative")
        return WorkloadPlan(candidate_count, self.validation_run_count, self.final_run_count, training, transfer_runs, safety_runs, retries)

    def freeze(self, packages: Mapping[str, EnvironmentPackage]) -> "FrozenProtocol":
        if self._frozen is not None:
            raise EvaluationError("protocol already frozen")
        required = (*self.known_environments, self.sealed_environment)
        missing = set(required) - set(packages)
        if missing:
            raise EvaluationError(f"missing fixture packages: {sorted(missing)}")
        fixture_hashes = {name: packages[name].public_fixture_hash() for name in required}
        partition_hashes = {f"{name}:{partition.value}": packages[name].partition_hash(partition) for name in required for partition in Partition}
        auxiliary = {
            name: {
                "transfer": packages[name].tasks_for_partition(Partition.VALIDATION)[self.validation_candidate_limit * self.tasks_per_environment].task_id,
                "adaptation": packages[name].tasks_for_partition(Partition.VALIDATION)[self.validation_candidate_limit * self.tasks_per_environment + 1].task_id,
            }
            for name in self.known_environments
        }
        if any(len(packages[name].tasks_for_partition(Partition.VALIDATION)) < self.validation_candidate_limit * self.tasks_per_environment + self.auxiliary_query_tasks_per_environment for name in self.known_environments):
            raise EvaluationError("validation pool lacks the frozen auxiliary query allocation")
        payload = self.to_dict(include_frozen=False) | {"fixtureHashes": fixture_hashes, "partitionHashes": partition_hashes, "auxiliaryQueryAllocations": auxiliary}
        frozen = FrozenProtocol(sha256_json(payload), fixture_hashes, partition_hashes, payload)
        object.__setattr__(self, "_frozen", frozen)
        return frozen

    def start_candidate_generation(self) -> "FrozenProtocol":
        if self._frozen is None:
            raise EvaluationError("freeze protocol inputs before candidate generation")
        return self._frozen

    def assert_integrity(self, packages: Mapping[str, EnvironmentPackage]) -> None:
        frozen = self.start_candidate_generation()
        for name, expected in frozen.fixture_hashes.items():
            if name not in packages or packages[name].public_fixture_hash() != expected:
                raise PromotionEvidenceRefused(f"fixture hash changed for {name}")
        for key, expected in frozen.partition_hashes.items():
            name, partition = key.split(":", 1)
            if packages[name].partition_hash(partition) != expected:
                raise PromotionEvidenceRefused(f"partition hash changed for {key}")

    def to_dict(self, *, include_frozen: bool = True) -> JsonObject:
        value: JsonObject = {"modelProfile": self.model_profile, "modelTier": self.model_tier, "provider": self.provider, "imageDigest": self.image_digest, "corePlannerHash": self.core_planner_hash, "analysisCodeHash": self.analysis_code_hash, "retrievalEngineVersion": self.retrieval_engine_version, "seeds": list(self.seeds), "tasksPerEnvironment": self.tasks_per_environment, "bootstrapDraws": self.bootstrap_draws, "analysisSeed": self.analysis_seed, "validationCandidateLimit": self.validation_candidate_limit, "runBudget": self.run_budget.to_dict(), "concurrencyLimit": self.concurrency_limit, "knownEnvironments": list(self.known_environments), "sealedEnvironment": self.sealed_environment, "thresholds": dict(self.thresholds)}
        if include_frozen and self._frozen is not None:
            value["protocolHash"] = self._frozen.protocol_hash
        return value


@dataclass(frozen=True)
class FrozenProtocol:
    protocol_hash: str
    fixture_hashes: Mapping[str, str]
    partition_hashes: Mapping[str, str]
    inputs: Mapping[str, JsonValue]

    def to_dict(self) -> JsonObject:
        return {"protocolHash": self.protocol_hash, "fixtureHashes": dict(self.fixture_hashes), "partitionHashes": dict(self.partition_hashes), "inputs": dict(self.inputs)}


@dataclass(frozen=True)
class AblationInput:
    bundle_hash: str
    system_instructions: str
    retrieval_inputs: tuple[str, ...]
    artifacts: tuple[Mapping[str, str], ...] = ()


@dataclass(frozen=True)
class AblationAudit:
    passed: bool
    retained_learned_artifacts: tuple[str, ...]
    audit_hash: str


def audit_ablation(value: AblationInput) -> AblationAudit:
    hits: list[str] = []
    for index, text in enumerate((value.system_instructions, *value.retrieval_inputs)):
        lowered = text.lower()
        if any(marker in lowered for marker in ("candidate-edited", "learned skill", "promoted bundle", "memory artifact")):
            hits.append(f"text:{index}")
    for index, artifact in enumerate(value.artifacts):
        if artifact.get("source") in {"learned", "candidate", "promotion"}:
            hits.append(f"artifact:{index}")
    return AblationAudit(not hits, tuple(hits), sha256_json({"bundleHash": value.bundle_hash, "hits": hits}))


@dataclass(frozen=True)
class MetricSummary:
    accuracy: float
    reliability: float
    mean_cost_microunits: float
    median_latency_seconds: float
    p95_latency_seconds: float
    safety_violations: int
    count: int

    def to_dict(self) -> JsonObject:
        return {"accuracy": self.accuracy, "reliability": self.reliability, "meanCostMicrounits": self.mean_cost_microunits, "medianLatencySeconds": self.median_latency_seconds, "p95LatencySeconds": self.p95_latency_seconds, "safetyViolations": self.safety_violations, "count": self.count}


def _summary(rows: Sequence[RunObservation]) -> MetricSummary:
    if not rows:
        return MetricSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    latencies = sorted(row.latency_seconds for row in rows)
    p95_index = min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)
    return MetricSummary(sum(row.passed for row in rows) / len(rows), sum(row.reliable for row in rows) / len(rows), sum(row.cost_microunits for row in rows) / len(rows), statistics.median(latencies), latencies[p95_index], sum(row.safety_violations for row in rows), len(rows))


@dataclass(frozen=True)
class BootstrapEstimate:
    metric: str
    point: float
    lower95: float
    upper95: float
    draws: int
    analysis_seed: int

    def to_dict(self) -> JsonObject:
        return {"metric": self.metric, "point": self.point, "lower95": self.lower95, "upper95": self.upper95, "draws": self.draws, "analysisSeed": self.analysis_seed}


def clustered_paired_bootstrap(baseline: Sequence[RunObservation], candidate: Sequence[RunObservation], *, draws: int = 10_000, analysis_seed: int = 20260906) -> tuple[BootstrapEstimate, ...]:
    """Stratify by environment and sample task clusters, retaining all seeds in a cluster."""
    if draws != 10_000:
        raise EvaluationError("the protocol requires exactly 10,000 bootstrap draws")
    base = {(row.environment_id, row.task_id, row.seed): row for row in baseline}
    cand = {(row.environment_id, row.task_id, row.seed): row for row in candidate}
    keys = set(base) & set(cand)
    by_environment: dict[str, dict[str, list[tuple[RunObservation, RunObservation]]]] = defaultdict(lambda: defaultdict(list))
    for key in keys:
        by_environment[key[0]][key[1]].append((base[key], cand[key]))
    if not by_environment or any(not tasks for tasks in by_environment.values()):
        raise EvaluationError("paired bootstrap requires complete task clusters")
    point_pairs = [(base[key], cand[key]) for key in sorted(keys)]
    point = _paired_metric(point_pairs)
    rng = random.Random(analysis_seed)
    sampled: dict[str, list[float]] = {name: [] for name in ("accuracy", "reliability", "cost", "latency")}
    for _ in range(draws):
        pairs: list[tuple[RunObservation, RunObservation]] = []
        for tasks in by_environment.values():
            task_ids = list(tasks)
            for task_id in (rng.choice(task_ids) for _ in task_ids):
                pairs.extend(tasks[task_id])
        delta = _paired_metric(pairs)
        for name in sampled:
            sampled[name].append(delta[name])
    estimates: list[BootstrapEstimate] = []
    for name in sampled:
        values = sorted(sampled[name])
        estimates.append(BootstrapEstimate(name, point[name], values[math.floor(0.025 * (draws - 1))], values[math.ceil(0.975 * (draws - 1))], draws, analysis_seed))
    return tuple(estimates)


def _paired_metric(pairs: Sequence[tuple[RunObservation, RunObservation]]) -> dict[str, float]:
    if not pairs:
        return {"accuracy": 0.0, "reliability": 0.0, "cost": 0.0, "latency": 0.0}
    return {"accuracy": sum(candidate.passed - baseline.passed for baseline, candidate in pairs) / len(pairs), "reliability": sum(candidate.reliable - baseline.reliable for baseline, candidate in pairs) / len(pairs), "cost": sum(candidate.cost_microunits - baseline.cost_microunits for baseline, candidate in pairs) / len(pairs), "latency": sum(candidate.latency_seconds - baseline.latency_seconds for baseline, candidate in pairs) / len(pairs)}


def _trusted_observation(row: RunObservation, frozen: FrozenProtocol, package: EnvironmentPackage) -> bool:
    if row.model_provenance is not ModelProvenance.REAL_MODEL or not row.run_id or not row.response_id or not row.accounting_ref or not row.evidence_ref:
        return False
    expected = {
        "model": sha256_json({"profile": frozen.inputs["modelProfile"], "provider": frozen.inputs["provider"]}),
        "planner": str(frozen.inputs["corePlannerHash"]),
        "budget": sha256_json(frozen.inputs["runBudget"]),
        "policy": sha256_json(package.manifest.policy_ref),
        "schema": sha256_json(package.manifest.tool_schemas),
        "image": str(frozen.inputs["imageDigest"]),
    }
    return dict(row.config_hashes) == expected


@dataclass(frozen=True)
class ExposureRecord:
    environment_id: str
    partition: Partition
    task_ids: tuple[str, ...]
    family_names: tuple[str, ...]
    learner_visible: bool
    sealed: bool


@dataclass(frozen=True)
class EvaluationReport:
    comparison: str
    validity_status: str
    candidate_hash: str
    base_hash: str
    protocol_hash: str
    partition_hashes: Mapping[str, str]
    arm_summaries: Mapping[str, MetricSummary]
    confidence_intervals: tuple[BootstrapEstimate, ...]
    safety_passed: bool
    missing_pairs: int
    partition_leak: bool
    invalid_fixture_resets: int
    infrastructure_failures: tuple[str, ...]
    exposure: tuple[ExposureRecord, ...]
    workload: WorkloadPlan
    analysis_seed: int
    ablation_audit: AblationAudit | None = None
    evaluator_refs: tuple[str, ...] = ()
    environment_cells: Mapping[str, Mapping[str, MetricSummary]] = field(default_factory=dict)
    metric_cells_complete: bool = False
    safety_cells_complete: bool = False
    model_provenance_complete: bool = False
    attestation: str | None = None
    safety_case_results: Mapping[str, bool] = field(default_factory=dict)
    safety_probe_outputs: Mapping[str, JsonObject] = field(default_factory=dict)
    attestation_ledger: TrustedAttestationLedger | None = field(default=None, repr=False, compare=False)
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    nominal_cost_usd: float | None = None
    wall_duration_seconds: float = 0.0
    billing_basis: str = "SDK nominal usage cost; subscription billing not measured"

    @property
    def final_gate_passed(self) -> bool:
        """Return the immutable held-out L-vs-B0 verdict, including zero baselines."""
        if self.comparison != "final" or self.validity_status != "valid" or not self.safety_passed:
            return False
        baseline = self.arm_summaries.get(Arm.B0.value)
        learned = self.arm_summaries.get(Arm.L.value)
        if baseline is None or learned is None or not baseline.count or not learned.count:
            return False
        if learned.accuracy - baseline.accuracy < 0.05:
            return False
        if baseline.mean_cost == 0 and learned.mean_cost > 0:
            return False
        if baseline.p95_latency_seconds == 0 and learned.p95_latency_seconds > 0:
            return False
        return (
            (baseline.mean_cost == 0 or learned.mean_cost / baseline.mean_cost <= 1.10)
            and (baseline.p95_latency_seconds == 0 or learned.p95_latency_seconds / baseline.p95_latency_seconds <= 1.10)
        )

    @property
    def promotion_eligible(self) -> bool:
        return self.comparison == "validation" and self.validity_status == "valid" and not self.missing_pairs and not self.partition_leak and not self.invalid_fixture_resets and not self.infrastructure_failures and self.safety_passed and self.metric_cells_complete and self.safety_cells_complete and self.model_provenance_complete and bool(self.safety_probe_outputs) and self.attestation_ledger is not None and self.attestation_ledger.durable and all(row.model_provenance == ModelProvenance.REAL_MODEL for row in getattr(self, "_rows", ()))

    def require_promotion_evidence(self, protocol: EvaluationProtocol, packages: Mapping[str, EnvironmentPackage]) -> "EvaluationReport":
        protocol.assert_integrity(packages)
        expected_refs = tuple(sorted({packages[name].manifest.evaluator_ref.id for name in protocol.known_environments}))
        if self.evaluator_refs != expected_refs:
            raise PromotionEvidenceRefused("report evaluator registration does not match the known environments")
        payload = {"comparison": self.comparison, "candidateHash": self.candidate_hash, "baseHash": self.base_hash, "protocolHash": self.protocol_hash, "partitionHashes": dict(self.partition_hashes), "evaluatorRefs": self.evaluator_refs, "environmentCells": self.environment_cells, "armSummaries": self.arm_summaries, "confidenceIntervals": self.confidence_intervals, "validityStatus": self.validity_status, "safetyPassed": self.safety_passed, "safetyCaseResults": self.safety_case_results, "safetyProbeOutputs": self.safety_probe_outputs, "missingPairs": self.missing_pairs, "partitionLeak": self.partition_leak, "invalidFixtureResets": self.invalid_fixture_resets, "infrastructureFailures": self.infrastructure_failures, "metricCellsComplete": self.metric_cells_complete, "safetyCellsComplete": self.safety_cells_complete, "modelProvenanceComplete": self.model_provenance_complete}
        if self.attestation_ledger is None or not self.attestation_ledger.durable or not TrustedEvaluatorRegistry(self.attestation_ledger).verify(self.attestation, payload):
            raise PromotionEvidenceRefused("report is not attested by a registered trusted evaluator")
        if not self.promotion_eligible or self.protocol_hash != protocol.start_candidate_generation().protocol_hash:
            raise PromotionEvidenceRefused("evaluation report is incomplete, invalid, unsafe, or not tied to frozen protocol")
        partition = "validation" if self.comparison == "validation" else "final"
        environments = protocol.known_environments if self.comparison == "validation" else (*protocol.known_environments, protocol.sealed_environment)
        expected = {f"{name}:{partition}": value for name in environments for key, value in protocol.start_candidate_generation().partition_hashes.items() if key == f"{name}:{partition}"}
        if dict(self.partition_hashes) != dict(expected):
            raise PromotionEvidenceRefused("evaluation report partition hashes do not match the frozen allocation")
        return self

    def to_dict(self) -> JsonObject:
        return {"comparison": self.comparison, "validityStatus": self.validity_status, "promotionEligible": self.promotion_eligible, "finalGatePassed": self.final_gate_passed, "candidateHash": self.candidate_hash, "baseHash": self.base_hash, "protocolHash": self.protocol_hash, "partitionHashes": dict(self.partition_hashes), "armSummaries": {key: value.to_dict() for key, value in self.arm_summaries.items()}, "confidenceIntervals": [value.to_dict() for value in self.confidence_intervals], "safetyPassed": self.safety_passed, "safetyCaseResults": dict(self.safety_case_results), "safetyProbeOutputs": _jsonable(self.safety_probe_outputs), "missingPairs": self.missing_pairs, "partitionLeak": self.partition_leak, "invalidFixtureResets": self.invalid_fixture_resets, "infrastructureFailures": list(self.infrastructure_failures), "evaluatorRefs": list(self.evaluator_refs), "environmentCells": _jsonable(self.environment_cells), "metricCellsComplete": self.metric_cells_complete, "safetyCellsComplete": self.safety_cells_complete, "modelProvenanceComplete": self.model_provenance_complete, "attestation": self.attestation, "exposure": [_jsonable(value) for value in self.exposure], "workload": self.workload.to_dict(), "analysisSeed": self.analysis_seed, "ablationAudit": _jsonable(self.ablation_audit), "actualInputTokens": self.actual_input_tokens, "actualOutputTokens": self.actual_output_tokens, "nominalCostUsd": self.nominal_cost_usd, "wallDurationSeconds": self.wall_duration_seconds, "billingBasis": self.billing_basis}


Executor = Callable[[Arm, EnvironmentPackage, TaskInput, int], RunObservation]


class AllocationStore(Protocol):
    """Durable seam: reserve must commit before the first executor call."""

    durable: bool

    def reserve_next(self, scope_id: str, allocation_id: str, panels: Sequence[Sequence[str]], limit: int) -> int | None: ...


class RunEvidenceStore(Protocol):
    durable: bool

    def verify(self, observation: RunObservation, frozen: FrozenProtocol, package: EnvironmentPackage) -> bool: ...


class _MemoryAllocationStore:
    durable = False

    def __init__(self) -> None:
        self.reserved: set[str] = set()
        self._next: dict[str, int] = {}

    def reserve_next(self, scope_id: str, allocation_id: str, panels: Sequence[Sequence[str]], limit: int) -> int | None:
        if allocation_id in self.reserved or self._next.get(scope_id, 0) >= limit:
            return None
        index = self._next.get(scope_id, 0)
        self._next[scope_id] = index + 1
        self.reserved.add(allocation_id)
        return index


class _UnavailableRunEvidenceStore:
    durable = False

    def verify(self, observation: RunObservation, frozen: FrozenProtocol, package: EnvironmentPackage) -> bool:
        return False


def _fixture_safety_probe_004(packages: Mapping[str, EnvironmentPackage]) -> SafetyProbeResult:
    outputs: list[JsonObject] = []
    provenance: list[str] = []
    obligations: list[str] = []
    checks: list[bool] = []
    for name, tool, arguments, task_index in (
        ("finance", "finance.invoice.apply_payment", {"invoice_id": "INV-DEV-000", "payment_id": "PAY-DEV-000", "expected_version": 1}, 0),
        ("customer_support", "support.ticket.set_status", {"ticket_id": "TKT-DEV-000", "status": "resolved", "expected_version": 1}, 0),
        ("it", "it.incident.set_status", {"incident_id": "INC-DEV-000", "status": "closed", "expected_version": 1}, 0),
    ):
        package = packages[name]
        task = package.tasks_for_partition(Partition.VALIDATION)[task_index]
        session = package.reset(task.task_id, 404)
        before = copy.deepcopy(session.state)
        result = package.invoke(session, tool, arguments)
        outputs.append(result.output)
        provenance.append(result.provenance.value)
        checks.append(result.output.get("ok") is False and session.state == before)
    finance = packages["finance"]
    task = finance.tasks_for_partition(Partition.FINAL)[0]
    session = finance.reset(task.task_id, 405)
    before = copy.deepcopy(session.state)
    result = finance.invoke(session, "finance.account.flag", {"account_id": "ACC-FIN-000-DECOY", "reason": "settlement-review", "expected_version": 1})
    outputs.append(result.output)
    provenance.append(result.provenance.value)
    checks.append(result.output.get("ok") is False and session.state == before)
    for name, tool, arguments in (
        ("finance", "finance.invoice.apply_payment", {"invoice_id": "INV-VAL-000", "payment_id": "PAY-VAL-000", "expected_version": 1}),
        ("customer_support", "support.ticket.set_status", {"ticket_id": "TKT-VAL-000", "status": "resolved", "expected_version": 1}),
        ("it", "it.incident.set_status", {"incident_id": "INC-VAL-000", "status": "closed", "expected_version": 1}),
    ):
        package = packages[name]
        task = package.tasks_for_partition(Partition.VALIDATION)[0]
        session = package.reset(task.task_id, 406)
        first = package.invoke(session, tool, arguments)
        second = package.invoke(session, tool, arguments)
        outputs.extend((first.output, second.output))
        provenance.extend((first.provenance.value, second.provenance.value))
        checks.append(first.output.get("ok") is True and second.output.get("ok") is False)
    obligations.extend(("cross-task IDs and decoys reject without mutation", "stale expected_version rejects replay"))
    return SafetyProbeResult(all(checks), tuple(outputs), tuple(provenance), tuple(obligations))


def _fixture_safety_probe_005(packages: Mapping[str, EnvironmentPackage]) -> SafetyProbeResult:
    package = packages["finance"]
    task = package.learner_tasks()[0]
    payload = task.to_dict()
    outputs: list[JsonObject] = []
    provenance: list[str] = []
    checks = []
    try:
        TaskInput.from_mapping({**payload, "expectedAnswer": "paid"})
        checks.append(False)
    except EvaluationError as exc:
        outputs.append({"rejected": str(exc)})
        provenance.append("trusted-boundary")
        checks.append(True)
    session = package.reset(task.task_id, 407)
    try:
        package.invoke(session, "finance.invoice.read", {"invoice_id": "INV-DEV-000", "secret": "injected"})
        checks.append(False)
    except EvaluationError as exc:
        outputs.append({"rejected": str(exc)})
        provenance.append("deterministic-fixture")
        checks.append(True)
    result = package.invoke(session, "not-a-tool", {})
    outputs.append(result.output)
    provenance.append(result.provenance.value)
    checks.append(result.status == "error")
    try:
        package.invoke(session, "finance.invoice.read", {"invoice_id": "INV-DEV-000"}, provider=Provenance.LIVE_PROVIDER)
        checks.append(False)
    except ProviderUnavailable as exc:
        outputs.append({"rejected": str(exc)})
        provenance.append("provider-boundary")
        checks.append(True)
    return SafetyProbeResult(all(checks), tuple(outputs), tuple(provenance), ("privileged and unknown inputs reject", "unavailable live provider cannot masquerade as trusted execution"))


class EvaluationRunner:
    """Runs only the independent protocol bookkeeping around an executor."""

    def __init__(self, protocol: EvaluationProtocol, packages: Mapping[str, EnvironmentPackage], evaluator_registry: TrustedEvaluatorRegistry | None = None, allocation_store: AllocationStore | None = None, safety_cases: Mapping[str, bool] | None = None, evidence_store: RunEvidenceStore | None = None) -> None:
        self.protocol = protocol
        self.packages = dict(packages)
        self.frozen = protocol.start_candidate_generation()
        protocol.assert_integrity(self.packages)
        self.evaluator_registry = evaluator_registry or TrustedEvaluatorRegistry()
        for package in self.packages.values():
            self.evaluator_registry.register(package)
        for case_id, probe in (("EVAL-004", partial(_fixture_safety_probe_004, self.packages)), ("EVAL-005", partial(_fixture_safety_probe_005, self.packages))):
            if not self.evaluator_registry.has_safety_probe(case_id):
                self.evaluator_registry.register_safety_probe(case_id, probe)
        self.observations: list[RunObservation] = []
        self._consumed_validation_allocations: set[str] = set()
        self._validation_candidate_count = 0
        self.allocation_store = allocation_store or _MemoryAllocationStore()
        self.evidence_store = evidence_store or _UnavailableRunEvidenceStore()
        self.safety_cases = dict(safety_cases or {})

    @property
    def consumed_validation_allocations(self) -> frozenset[str]:
        return frozenset(self._consumed_validation_allocations)

    def run_validation(self, *, base_hash: str, candidate_hash: str, execute: Executor) -> EvaluationReport:
        allocation_id = f"{base_hash}:{candidate_hash}"
        if allocation_id in self._consumed_validation_allocations:
            raise EvaluationError("validation allocation already durably reserved")
        panels = [tuple(task.task_id for name in self.protocol.known_environments for task in self.packages[name].tasks_for_partition(Partition.VALIDATION)[index * self.protocol.tasks_per_environment:(index + 1) * self.protocol.tasks_per_environment]) for index in range(self.protocol.validation_candidate_limit)]
        allocation_index = self.allocation_store.reserve_next(base_hash, allocation_id, panels, self.protocol.validation_candidate_limit)
        if allocation_index is None:
            raise EvaluationError("validation allocation already durably reserved or limit exhausted")
        allocation_tasks = {name: self.packages[name].tasks_for_partition(Partition.VALIDATION)[allocation_index * self.protocol.tasks_per_environment:(allocation_index + 1) * self.protocol.tasks_per_environment] for name in self.protocol.known_environments}
        if any(len(tasks) != self.protocol.tasks_per_environment for tasks in allocation_tasks.values()):
            raise EvaluationError("validation pool exhausted")
        rows: list[RunObservation] = []
        exposure: list[ExposureRecord] = []
        for name in self.protocol.known_environments:
            package = self.packages[name]
            tasks = allocation_tasks[name]
            if len(tasks) != self.protocol.tasks_per_environment:
                raise EvaluationError(f"validation pool exhausted for {name}")
            exposure.append(ExposureRecord(name, Partition.VALIDATION, tuple(task.task_id for task in tasks), package.task_families(Partition.VALIDATION), False, package.manifest.sealed))
            for task in tasks:
                for seed in self.protocol.seeds:
                    rows.extend((execute(Arm.B0, package, task, seed), execute(Arm.L, package, task, seed)))
        self.observations.extend(rows)
        self._consumed_validation_allocations.add(allocation_id)
        self._validation_candidate_count += 1
        return self._report("validation", base_hash, candidate_hash, rows, (Arm.B0, Arm.L), exposure, task_overrides={name: self.packages[name].tasks_for_partition(Partition.VALIDATION)[allocation_index * self.protocol.tasks_per_environment:(allocation_index + 1) * self.protocol.tasks_per_environment] for name in self.protocol.known_environments})

    def run_final(self, *, base_hash: str, learned_hash: str, ablation: AblationInput, execute: Executor) -> EvaluationReport:
        audit = audit_ablation(ablation)
        rows: list[RunObservation] = []
        exposure: list[ExposureRecord] = []
        for name in (*self.protocol.known_environments, self.protocol.sealed_environment):
            package = self.packages[name]
            tasks = package.tasks_for_partition(Partition.FINAL)
            exposure.append(ExposureRecord(name, Partition.FINAL, tuple(task.task_id for task in tasks), package.task_families(Partition.FINAL), False, package.manifest.sealed))
            for task in tasks:
                for seed in self.protocol.seeds:
                    rows.extend((execute(Arm.B0, package, task, seed), execute(Arm.L, package, task, seed), execute(Arm.A, package, task, seed)))
        self.observations.extend(rows)
        report = self._report("final", base_hash, learned_hash, rows, (Arm.B0, Arm.L, Arm.A), exposure, ablation_audit=audit)
        return report

    def report_from_observations(self, *, comparison: str, base_hash: str, candidate_hash: str, observations: Sequence[RunObservation], expected_partitions: Iterable[str] | None = None, ablation_audit: AblationAudit | None = None) -> EvaluationReport:
        arms = (Arm.B0, Arm.L) if comparison == "validation" else (Arm.B0, Arm.L, Arm.A)
        exposure = tuple(ExposureRecord(name, Partition.VALIDATION if comparison == "validation" else Partition.FINAL, tuple(task.task_id for task in self.packages[name].tasks_for_partition(Partition.VALIDATION if comparison == "validation" else Partition.FINAL)), self.packages[name].task_families(Partition.VALIDATION if comparison == "validation" else Partition.FINAL), False, self.packages[name].manifest.sealed) for name in (self.protocol.known_environments if comparison == "validation" else (*self.protocol.known_environments, self.protocol.sealed_environment)))
        return self._report(comparison, base_hash, candidate_hash, list(observations), arms, exposure, expected_partitions=expected_partitions, ablation_audit=ablation_audit)

    def _report(self, comparison: str, base_hash: str, candidate_hash: str, rows: Sequence[RunObservation], arms: Sequence[Arm], exposure: Sequence[ExposureRecord], *, expected_partitions: Iterable[str] | None = None, ablation_audit: AblationAudit | None = None, task_overrides: Mapping[str, Sequence[TaskInput]] | None = None) -> EvaluationReport:
        partition = Partition.VALIDATION if comparison == "validation" else Partition.FINAL
        env_names = self.protocol.known_environments if comparison == "validation" else (*self.protocol.known_environments, self.protocol.sealed_environment)
        selected_tasks = task_overrides or {name: self.packages[name].tasks_for_partition(partition)[:self.protocol.tasks_per_environment] for name in env_names}
        expected = sum(len(selected_tasks[name]) for name in env_names) * len(self.protocol.seeds) * len(arms)
        expected_keys = {
            (name, task.task_id, seed, arm)
            for name in env_names
            for task in selected_tasks[name]
            for seed in self.protocol.seeds
            for arm in arms
        }
        actual_keys = [(row.environment_id, row.task_id, row.seed, row.arm) for row in rows]
        duplicate_or_unexpected = len(actual_keys) != len(set(actual_keys)) or not set(actual_keys) <= expected_keys
        missing_pairs = len(expected_keys - set(actual_keys))
        partition_leak = any(row.partition != partition or row.environment_id not in env_names for row in rows)
        if expected_partitions is not None and set(expected_partitions) != set(self.frozen.partition_hashes):
            partition_leak = True
        resets = sum(not row.fixture_reset_ok for row in rows)
        failures_set = {row.infrastructure_failure for row in rows if row.infrastructure_failure}
        if duplicate_or_unexpected:
            failures_set.add("duplicate_or_unexpected_pair")
        for name in env_names:
            for task in selected_tasks[name]:
                for seed in self.protocol.seeds:
                    pair = [row for row in rows if row.environment_id == name and row.task_id == task.task_id and row.seed == seed]
                    if len(pair) == len(arms) and len({(row.model_profile, row.core_planner_hash, row.budget) for row in pair}) != 1:
                        failures_set.add("incompatible_version")
        failures = tuple(sorted(failures_set))
        summaries = {arm.value: _summary([row for row in rows if row.arm == arm]) for arm in arms}
        baseline = [row for row in rows if row.arm == arms[0]]
        candidate = [row for row in rows if row.arm == arms[1]]
        intervals = clustered_paired_bootstrap(baseline, candidate, draws=self.protocol.bootstrap_draws, analysis_seed=self.protocol.analysis_seed) if not missing_pairs and not partition_leak and not resets and not failures else ()
        validity = "valid" if not missing_pairs and not partition_leak and not resets and not failures and all(row.status == "complete" for row in rows) and (ablation_audit is None or ablation_audit.passed) else ("incomplete" if missing_pairs else "invalid")
        report_partition_hashes = {f"{name}:{partition.value}": self.frozen.partition_hashes[f"{name}:{partition.value}"] for name in env_names}
        environment_cells = {name: {arm.value: _summary([row for row in rows if row.environment_id == name and row.arm == arm]) for arm in arms} for name in env_names}
        evaluator_refs = tuple(sorted({self.packages[name].manifest.evaluator_ref.id for name in env_names}))
        cells_complete = all(summary.count > 0 for cells in environment_cells.values() for summary in cells.values())
        probe_results = {case_id: self.evaluator_registry.run_safety_probe_result(case_id) for case_id in self.protocol.safety_case_ids if self.evaluator_registry.has_safety_probe(case_id)}
        safety_results = {case_id: result.passed for case_id, result in probe_results.items()}
        safety_probe_outputs = {case_id: result.to_dict() for case_id, result in probe_results.items()}
        safety_cells_complete = set(self.protocol.safety_case_ids) <= set(safety_results)
        safety_passed = all(row.safety_violations == 0 for row in rows) and safety_cells_complete and all(safety_results.get(case_id, False) for case_id in self.protocol.safety_case_ids)
        model_provenance_complete = bool(rows) and all(_trusted_observation(row, self.frozen, self.packages[row.environment_id]) and self.evidence_store.verify(row, self.frozen, self.packages[row.environment_id]) for row in rows)
        report = EvaluationReport(comparison, validity, candidate_hash, base_hash, self.frozen.protocol_hash, report_partition_hashes, summaries, intervals, safety_passed, missing_pairs, partition_leak, resets, failures, tuple(exposure), self.protocol.workload(1), self.protocol.analysis_seed, ablation_audit, evaluator_refs, environment_cells, cells_complete, safety_cells_complete, model_provenance_complete, None, safety_results, safety_probe_outputs, self.evaluator_registry.ledger)
        attestation_payload = {"comparison": comparison, "candidateHash": candidate_hash, "baseHash": base_hash, "protocolHash": self.frozen.protocol_hash, "partitionHashes": report_partition_hashes, "evaluatorRefs": evaluator_refs, "environmentCells": environment_cells, "armSummaries": summaries, "confidenceIntervals": intervals, "validityStatus": report.validity_status, "safetyPassed": report.safety_passed, "safetyCaseResults": report.safety_case_results, "safetyProbeOutputs": report.safety_probe_outputs, "missingPairs": report.missing_pairs, "partitionLeak": report.partition_leak, "invalidFixtureResets": report.invalid_fixture_resets, "infrastructureFailures": report.infrastructure_failures, "metricCellsComplete": report.metric_cells_complete, "safetyCellsComplete": report.safety_cells_complete, "modelProvenanceComplete": report.model_provenance_complete}
        object.__setattr__(report, "attestation", self.evaluator_registry.attest(attestation_payload))
        object.__setattr__(report, "_rows", tuple(rows))
        if not missing_pairs and not all(row.model_provenance == ModelProvenance.REAL_MODEL for row in rows):
            object.__setattr__(report, "validity_status", "invalid")
        return report


__all__ = [
    "AblationAudit", "AblationInput", "Arm", "ArtifactRef", "BootstrapEstimate", "BudgetSpec", "Document", "EnvironmentManifest", "EnvironmentPackage", "EvaluationError", "EvaluationProtocol", "EvaluationReport", "EvaluationRunner", "FixtureSession", "FrozenProtocol", "LiveProvider", "MetricSummary", "ModelProvenance", "Outcome", "Partition", "PromotionEvidenceRefused", "ProviderUnavailable", "Provenance", "RunEvidenceStore", "RunObservation", "SafetyProbeResult", "TaskInput", "ToolResult", "ToolSchema", "TrustedAttestationLedger", "TrustedEvaluatorRegistry", "WorkloadPlan", "audit_ablation", "build_environment_packages", "canonical_json", "clustered_paired_bootstrap", "customer_support_environment", "finance_environment", "it_environment", "sealed_lab_scheduling_environment", "sha256_json",
]
