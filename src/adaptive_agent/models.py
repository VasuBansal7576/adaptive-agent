"""Domain-agnostic contracts and data models for the adaptive agent backend core.

These types mirror the contracts in SPEC.md and are intentionally domain-neutral:
no finance, support, or IT workflow is hardcoded.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel as _BaseModel, Field, field_validator, model_validator


class BaseModel(_BaseModel):
    model_config = {"populate_by_name": True, "extra": "forbid"}


Visibility = Literal["learner", "operator", "evaluator_only"]
TrustClass = Literal["learner", "operator", "evaluator", "broker", "system"]
EffectClass = Literal["read", "write", "none", "unknown"]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def sha256_json(data: Any) -> str:
    """Deterministic content hash for JSON-serializable objects."""

    def default(o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_usage(value: Any) -> dict[str, int]:
    """Normalize provider usage into the persisted evidence contract.

    Providers occasionally omit prompt or total counts, or use snake-case
    names.  The trusted parent fills only derivable values and rejects
    contradictory or non-integral counts so response and accounting artifacts
    can share one exact usage object.
    """
    if not isinstance(value, Mapping):
        raise ValueError("model usage must be an object")

    def count(*keys: str) -> int | None:
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
                return candidate
            if candidate is not None:
                raise ValueError(f"model usage field {key} must be a non-negative integer")
        return None

    input_tokens = count("inputTokens", "input_tokens", "promptTokens", "prompt_tokens", "input")
    output_tokens = count("outputTokens", "output_tokens", "completionTokens", "completion_tokens", "output")
    total_tokens = count("totalTokens", "total_tokens")
    if input_tokens is None:
        input_tokens = 0
    if output_tokens is None:
        output_tokens = 0 if total_tokens is None else total_tokens - input_tokens
        if output_tokens < 0:
            raise ValueError("model usage totalTokens is smaller than inputTokens")
    expected_total = input_tokens + output_tokens
    if total_tokens is None:
        total_tokens = expected_total
    if total_tokens != expected_total:
        raise ValueError("model usage totalTokens must equal inputTokens + outputTokens")
    return {"inputTokens": input_tokens, "outputTokens": output_tokens, "totalTokens": total_tokens}


def new_id(prefix: str = "") -> str:
    """Opaque string identifier; prefix is for human readability only."""
    return f"{prefix}{uuid.uuid4().hex}"


class ArtifactRef(BaseModel):
    id: str
    version: str
    sha256: str


class ToolSchema(BaseModel):
    name: str
    version: str
    input_schema: dict[str, Any] = Field(..., alias="inputSchema")
    output_schema: dict[str, Any] = Field(..., alias="outputSchema")
    effect: Literal["read", "write"]

    model_config = {"populate_by_name": True}


class EnvironmentManifest(BaseModel):
    schema_version: Literal[1] = Field(1, alias="schemaVersion")
    environment_id: str = Field(..., alias="environmentId")
    version: str
    docs: list[ArtifactRef] = Field(default_factory=list)
    tool_schemas: list[ToolSchema] = Field(..., alias="toolSchemas")
    policy_ref: ArtifactRef = Field(..., alias="policyRef")
    evaluator_ref: ArtifactRef = Field(..., alias="evaluatorRef")
    reset_ref: ArtifactRef = Field(..., alias="resetRef")
    execution_modes: list[str] = Field(default_factory=lambda: ["interactive"], alias="executionModes")
    capabilities: list[str] = Field(default_factory=list)

    @field_validator("docs", "tool_schemas", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> Any:
        return v if v is not None else []

    @model_validator(mode="after")
    def validate_boundary(self) -> "EnvironmentManifest":
        if not self.docs:
            raise ValueError("at least one document reference is required")
        if not self.tool_schemas:
            raise ValueError("at least one tool schema is required")
        allowed = {"interactive", "batch", "dry_run", "replay"}
        if not self.execution_modes or any(mode not in allowed for mode in self.execution_modes):
            raise ValueError("executionModes contains an unsupported mode")
        return self


class TaskInput(BaseModel):
    task_id: str = Field(..., alias="taskId")
    environment_ref: ArtifactRef = Field(..., alias="environmentRef")
    goal: str
    allowed_input_refs: list[ArtifactRef] = Field(default_factory=list, alias="allowedInputRefs")
    partition: str = "development"  # development | validation | final | training


class Budget(BaseModel):
    budget_id: str = Field(default_factory=lambda: new_id("bud_"))
    max_tool_calls: int = 100
    max_wall_seconds: float = 900.0  # 15 minutes
    max_concurrent_children: int = 2
    max_child_depth: int = 1
    max_model_cost: float | None = None
    max_model_tokens: int | None = None

    @model_validator(mode="after")
    def validate_nonnegative(self) -> "Budget":
        values = (self.max_tool_calls, self.max_wall_seconds, self.max_concurrent_children, self.max_child_depth, self.max_model_cost, self.max_model_tokens)
        if any(value is not None and value < 0 for value in values):
            raise ValueError("budget limits must be non-negative")
        return self


class RunRequest(BaseModel):
    task_ref: ArtifactRef = Field(..., alias="taskRef")
    model_profile_ref: ArtifactRef = Field(..., alias="modelProfileRef")
    budget_ref: ArtifactRef = Field(..., alias="budgetRef")
    idempotency_key: str = Field(..., alias="idempotencyKey")
    parent_run_id: str | None = Field(None, alias="parentRunId")
    execution_mode: str = Field("interactive", alias="executionMode")
    active_skill_refs: list[ArtifactRef] = Field(default_factory=list, alias="activeSkillRefs")
    # Evaluator-owned retries use a fresh immutable attempt identity.  Public
    # runs retain the default zero for backwards compatibility.
    attempt: int = Field(0, ge=0)

    @field_validator("attempt")
    @classmethod
    def validate_attempt(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("attempt must be a non-negative integer")
        return value


class ModelProfile(BaseModel):
    profile_id: str = Field(default_factory=lambda: new_id("mp_"))
    provider: Literal["simulation", "live"]
    model_name: str
    deterministic_seed: int | None = None
    stochastic: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class ExecutionConfig(BaseModel):
    allowed_tools: list[str] = Field(default_factory=list)
    skill_refs: list[ArtifactRef] = Field(default_factory=list)
    instruction_variant: str = "default"
    step_limit: int = 100
    child_count_limit: int = 2
    child_depth_limit: int = 1


class ToolRequest(BaseModel):
    run_id: str = Field(..., alias="runId")
    step_id: str = Field(..., alias="stepId")
    call_id: str = Field(default_factory=lambda: new_id("call_"), alias="callId")
    tool: str
    arguments: dict[str, Any]
    idempotency_key: str = Field(..., alias="idempotencyKey")
    approval_token: str | None = Field(None, alias="approvalToken")


class ToolErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    FORBIDDEN = "FORBIDDEN"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"


class ToolError(BaseModel):
    code: ToolErrorCode
    message: str
    correlation_id: str = Field(default_factory=lambda: new_id("corr_"))
    retry: Literal["never", "safe_read", "after_reconciliation"] = "never"


class ToolResult(BaseModel):
    call_id: str = Field(..., alias="callId")
    tool_version: str = Field(..., alias="toolVersion")
    observed_at: datetime = Field(default_factory=now_utc, alias="observedAt")
    broker_evidence_ref: ArtifactRef = Field(..., alias="brokerEvidenceRef")
    status: Literal["ok", "error"]
    output: Any | None = None
    error: ToolError | None = None
    effect: Literal["none", "confirmed", "unknown"] = "none"


class StepKind(str, Enum):
    retrieve = "retrieve"
    execute = "execute"
    tool = "tool"
    child = "child"
    evaluate = "evaluate"


class StepStatus(str, Enum):
    planned = "planned"
    running = "running"
    awaiting_approval = "awaiting_approval"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    outcome_unknown = "outcome_unknown"


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    awaiting_approval = "awaiting_approval"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    timed_out = "timed_out"


class RunRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: new_id("run_"), alias="runId")
    task_ref: ArtifactRef = Field(..., alias="taskRef")
    environment_ref: ArtifactRef = Field(..., alias="environmentRef")
    policy_ref: ArtifactRef = Field(..., alias="policyRef")
    model_profile_ref: ArtifactRef = Field(..., alias="modelProfileRef")
    skill_bundle_ref: ArtifactRef = Field(..., alias="skillBundleRef")
    budget_ref: ArtifactRef = Field(..., alias="budgetRef")
    execution_mode: str = Field("interactive", alias="executionMode")
    active_skill_refs: list[ArtifactRef] = Field(default_factory=list, alias="activeSkillRefs")
    parent_run_id: str | None = Field(None, alias="parentRunId")
    status: RunStatus = RunStatus.queued
    last_event_sequence: int = Field(0, alias="lastEventSequence")
    outcome_ref: ArtifactRef | None = Field(None, alias="outcomeRef")
    created_at: datetime = Field(default_factory=now_utc, alias="createdAt")
    completed_at: datetime | None = Field(None, alias="completedAt")
    # Benchmark execution identity, present only for evaluator-owned runs.
    arm: str | None = None
    seed: int | None = None
    bundle_hash: str | None = Field(None, alias="bundleHash")
    arm_bundles: dict[str, str] = Field(default_factory=dict, alias="armBundles")
    attempt: int = Field(0, ge=0)


class StepRecord(BaseModel):
    step_id: str = Field(default_factory=lambda: new_id("step_"), alias="stepId")
    run_id: str = Field(..., alias="runId")
    sequence: int
    kind: StepKind
    status: StepStatus = StepStatus.planned
    input_refs: list[ArtifactRef] = Field(default_factory=list, alias="inputRefs")
    output_refs: list[ArtifactRef] = Field(default_factory=list, alias="outputRefs")
    call_id: str | None = None
    error: ToolError | None = None


class EvidenceRecord(BaseModel):
    evidence_id: str = Field(default_factory=lambda: new_id("ev_"))
    run_id: str = Field(..., alias="runId")
    sequence: int
    event_type: str = Field(..., alias="eventType")
    content_hash: str = Field(..., alias="contentHash")
    source_ref: ArtifactRef = Field(..., alias="sourceRef")
    trust_class: TrustClass = Field(..., alias="trustClass")
    visibility: Visibility = "learner"
    redacted: bool = True


class SkillVersion(BaseModel):
    skill_id: str = Field(..., alias="skillId")
    version: str
    parent: str | None = None
    applicability: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    procedure: str  # Python source or declarative procedure text
    expected_tool_contracts: list[ArtifactRef] = Field(default_factory=list, alias="expectedToolContracts")
    failure_handling: list[str] = Field(default_factory=list, alias="failureHandling")
    evidence_refs: list[ArtifactRef] = Field(default_factory=list, alias="evidenceRefs")
    content_hash: str = Field(default="", alias="contentHash")

    @model_validator(mode="after")
    def compute_hash(self) -> "SkillVersion":
        if not self.content_hash:
            payload = self.model_dump(mode="json", by_alias=True, exclude={"content_hash"})
            self.content_hash = sha256_json(payload)
        return self


class SkillBundle(BaseModel):
    bundle_id: str = Field(default_factory=lambda: new_id("bundle_"))
    parent: str | None = None
    skills: list[SkillVersion] = Field(default_factory=list)
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig, alias="executionConfig")
    created_at: datetime = Field(default_factory=now_utc, alias="createdAt")
    content_hash: str = Field(default="", alias="contentHash")

    @model_validator(mode="after")
    def compute_hash(self) -> "SkillBundle":
        if not self.content_hash:
            payload = self.model_dump(mode="json", by_alias=True, exclude={"content_hash"})
            self.content_hash = sha256_json(payload)
        return self


class CandidateState(str, Enum):
    draft = "draft"
    validated = "validated"
    evaluating = "evaluating"
    promoted = "promoted"
    rejected = "rejected"
    quarantined = "quarantined"
    superseded = "superseded"
    rolled_back = "rolled_back"


class CandidateProposal(BaseModel):
    candidate_id: str = Field(default_factory=lambda: new_id("cand_"))
    base_bundle_hash: str = Field(..., alias="baseBundleHash")
    candidate_bundle_hash: str | None = Field(None, alias="candidateBundleHash")
    edit_operations: list[str] = Field(default_factory=list, alias="editOperations")
    changed_artifact_hashes: list[str] = Field(default_factory=list, alias="changedArtifactHashes")
    supporting_evidence_ids: list[str] = Field(default_factory=list, alias="supportingEvidenceIds")
    predicted_effect: str = Field(..., alias="predictedEffect")
    proposer_version: str = Field(..., alias="proposerVersion")
    state: CandidateState = CandidateState.draft
    created_at: datetime = Field(default_factory=now_utc, alias="createdAt")


class EvaluationState(str, Enum):
    queued = "queued"
    running = "running"
    valid = "valid"
    invalid = "invalid"
    cancelled = "cancelled"


class MetricAggregate(BaseModel):
    accuracy: float | None = None
    reliability: float | None = None
    mean_cost: float | None = Field(None, alias="meanCost")
    p95_latency_ms: float | None = Field(None, alias="p95LatencyMs")
    safety_violations: int = Field(0, alias="safetyViolations")


class EvaluationReport(BaseModel):
    report_id: str = Field(default_factory=lambda: new_id("rpt_"))
    candidate_hash: str = Field(..., alias="candidateHash")
    base_hash: str | None = Field(None, alias="baseHash")
    protocol_hash: str = Field(..., alias="protocolHash")
    partition_ref: ArtifactRef = Field(..., alias="partitionRef")
    paired_run_ids: list[tuple[str, str]] = Field(default_factory=list, alias="pairedRunIds")
    metrics: MetricAggregate
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    safety_results: dict[str, bool] = Field(default_factory=dict, alias="safetyResults")
    validity: EvaluationState = EvaluationState.queued
    evaluator_provenance: str = Field(..., alias="evaluatorProvenance")


class PromotionDecision(BaseModel):
    decision_id: str = Field(default_factory=lambda: new_id("dec_"))
    candidate_hash: str = Field(..., alias="candidateHash")
    base_hash: str = Field(..., alias="baseHash")
    report_ref: ArtifactRef = Field(..., alias="reportRef")
    gate_version: str = Field(..., alias="gateVersion")
    decision: Literal["promoted", "rejected", "quarantined"]
    reason: str
    prior_active_hash: str = Field(..., alias="priorActiveHash")
    new_active_hash: str | None = Field(None, alias="newActiveHash")
    timestamp: datetime = Field(default_factory=now_utc)


class Outcome(BaseModel):
    outcome_id: str = Field(default_factory=lambda: new_id("out_"))
    run_id: str = Field(..., alias="runId")
    passed: bool
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=now_utc, alias="checkedAt")


class PromotionGate(BaseModel):
    """Frozen protocol gate. All thresholds must be declared before any candidate sees results."""

    gate_id: str = Field(default_factory=lambda: new_id("gate_"))
    protocol_hash: str = Field(..., alias="protocolHash")
    min_balanced_accuracy_gain: float = Field(0.05, alias="minBalancedAccuracyGain")
    ci_lower_bound: float = Field(0.0, alias="ciLowerBound")
    max_cost_ratio: float = Field(1.10, alias="maxCostRatio")
    max_latency_ratio: float = Field(1.10, alias="maxLatencyRatio")
    require_per_environment_non_regression: bool = Field(True, alias="requirePerEnvironmentNonRegression")
    frozen_at: datetime = Field(default_factory=now_utc, alias="frozenAt")
