/**
 * Runtime contract types mirrored from SPEC.md "Boundary contracts".
 * Single source of UI truth; keep in sync with the control API contract.
 */
export type ArtifactRef = { id: string; version: string; sha256: string };

export type RunStatus =
  | "queued"
  | "running"
  | "awaiting_approval"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "timed_out";

export type ToolErrorCode =
  | "INVALID_INPUT"
  | "FORBIDDEN"
  | "VERSION_CONFLICT"
  | "IDEMPOTENCY_CONFLICT"
  | "BUDGET_EXHAUSTED"
  | "TOOL_UNAVAILABLE"
  | "OUTCOME_UNKNOWN";

export type ToolError = {
  code: ToolErrorCode;
  message: string;
  correlationId: string;
  retry: "never" | "safe_read" | "after_reconciliation";
};

export type RunRecord = {
  runId: string;
  taskRef: ArtifactRef;
  environmentRef: ArtifactRef;
  policyRef: ArtifactRef;
  modelProfileRef: ArtifactRef;
  skillBundleRef: ArtifactRef;
  budgetRef: ArtifactRef;
  status: RunStatus;
  lastEventSequence: number;
  /** recorded by POST /runs and echoed in run-started events */
  executionMode?: "dry_run" | "interactive" | "batch" | "replay";
  outcomeRef?: ArtifactRef;
  /** console presentation fields supplied by the API projection */
  environmentId?: string;
  goal?: string;
  /** declared modes the environment manifest supports (when projected) */
  executionModes?: string[];
  budgetUsed?: { calls: number; callsCeiling: number; wallSeconds: number; wallCeiling: number };
};

export type EvidenceProvenance = {
  evidenceId?: string;
  sourceRefId?: string;
  contentHash?: string;
  trustClass?: string;
  visibility?: string;
  redacted?: boolean;
};

export type RunEvent = {
  runId: string;
  sequence: number;
  at: string;
  kind: "status" | "step" | "tool" | "evidence" | "budget" | "approval";
  summary: string;
  detail?: string; // untrusted text; rendered escaped only
  error?: ToolError;
  approval?: ApprovalRequest;
  /** provenance from the durable evidence row; expandable in the UI */
  evidence?: EvidenceProvenance;
  /** authoritative status transition derived from the validated event TYPE
   *  (never from display text); absent for non-lifecycle events */
  runStatus?: RunStatus;
  /** the durable event type this row was projected from (e.g. run_failed,
   *  outcome_recorded); used for honest failure classification */
  lifecycleType?: string;
  /** the API projected no status field for this event (e.g. bare status or
   *  outcome rows): the authoritative RunRecord must be refreshed */
  needsRecordRefresh?: boolean;
};

export type ApprovalRequest = {
  approvalId: string;
  tool: string;
  toolVersion: string;
  canonicalArguments: string; // pretty JSON, exact arguments
  resourceScope: string;
  effect: "read" | "write";
  expiresAt: string;
};

export type SkillVersionSummary = {
  skillId: string;
  version: string;
  parentVersion: string | null;
  state: "active" | "proposed" | "rejected" | "quarantined" | "rolled_back";
  applicability: string;
  evidenceRefs: string[];
  contentHash: string;
};

export type CandidateDiff = {
  candidateId: string;
  baseBundleRef: ArtifactRef;
  candidateBundleRef: ArtifactRef;
  state: "draft" | "validated" | "evaluating" | "promoted" | "rejected" | "quarantined" | "superseded" | "rolled_back";
  /** unified diff text; rendered as escaped preformatted text */
  diff: string;
  predictedEffect: string;
  /** present only when a trusted evaluation exists; never conflated with prediction */
  measured?: {
    accuracyGainPp: number;
    reliabilityDelta: number;
    costRatio: number;
    p95LatencyRatio: number;
    gateDecision: "promoted" | "rejected" | "quarantined";
    gateReasons: string[];
    evaluationRef: ArtifactRef;
  };
  audit?: RollbackAudit[];
};

export type RollbackAudit = {
  rollbackId: string;
  fromRef: ArtifactRef;
  toRef: ArtifactRef;
  reason: string;
  at: string;
  affectedRuns: string[];
};

export type EnvironmentPackageSummary = {
  environmentId: string;
  version: string;
  validationState: "valid" | "invalid" | "unchecked";
  evaluatorReady: boolean;
  toolCount: number;
  policyScope: string;
  /** declared execution modes projected from the manifest (optional) */
  executionModes?: string[];
  missingFields?: string[];
};

/** GET /run-options: authoritative model profile and bounded budget controls. */
export type RunOptions = {
  modelProfiles: Array<{ ref: { id: string; version: string; sha256: string }; label: string; provider?: string; model?: string }>;
  budgetDefaults: {
    modelTokens: number;
    toolCalls: number;
    childRuns?: number;
    wallTimeSeconds: number;
    costMicrounits?: number;
    currency?: string;
  };
  /** authoritative trusted budget reference advertised by the control plane
   *  (01f2462 top level; 2b3fc75 nested form tolerated); submitted verbatim */
  budgetRef?: { id: string; version: string; sha256: string };
};

/** GET /environments/{id}/tasks: registered task goals a run may target. */
export type TaskOption = {
  taskId: string;
  goal: string;
  executionModes: string[];
};
