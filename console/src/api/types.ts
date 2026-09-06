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
  /** server-declared learning eligibility (trusted development attempts;
   *  heldout excluded). Falls back to succeeded-only when not projected. */
  learningEligible?: boolean;
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
  /** plane projection: artifact refs + unified diff */
  baseBundleRef?: ArtifactRef;
  candidateBundleRef?: ArtifactRef;
  /** durable projection: authoritative hashes + edit operations */
  baseBundleHash?: string;
  candidateBundleHash?: string;
  editOperations?: string[];
  changedArtifactHashes?: string[];
  supportingEvidenceIds?: string[];
  proposerVersion?: string;
  state: "draft" | "validated" | "evaluating" | "promoted" | "rejected" | "quarantined" | "superseded" | "rolled_back";
  /** plane projection: unified diff text; rendered as escaped preformatted text */
  diff?: string;
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
  /** manifest capability metadata projected for the operator (optional) */
  capabilities?: string[];
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

/** GET /evaluations row: durable evaluation job status per candidate. */
/** Authoritative arm metrics from EvaluationReport.to_dict (per arm B0/L/A). */
export type ArmSummary = {
  accuracy: number;
  reliability: number;
  meanCostMicrounits: number;
  medianLatencySeconds: number;
  p95LatencySeconds: number;
  safetyViolations: number;
  count: number;
};

/** Normalized trusted EvaluationReport projection (a864316-lineage; the
 *  console renders only these fields — never fabricated metrics). */
export type EvaluationReportProjection = {
  comparison: "validation" | "final" | string;
  validityStatus: string;
  promotionEligible: boolean;
  candidateHash?: string;
  baseHash?: string;
  protocolHash?: string;
  armSummaries: Record<string, ArmSummary>;
  confidenceIntervals?: Array<{ metric: string; point: number; lower95: number; upper95: number; draws?: number; analysisSeed?: number }>;
  safetyPassed?: boolean;
  missingPairs?: number;
  metricCellsComplete?: boolean;
  safetyCellsComplete?: boolean;
  modelProvenanceComplete?: boolean;
  infrastructureFailures?: string[];
  analysisSeed?: number;
  nominalCostUsd?: number | null;
  actualInputTokens?: number | null;
  actualOutputTokens?: number | null;
  wallDurationSeconds?: number | null;
  /** explicit unknown-billing marker from the backend */
  billingBasis?: string;
};

export type EvaluationJob = {
  evaluationId: string;
  /** legacy QA rows may lack the binding; preserved as null = unverified */
  candidateId: string | null;
  state: "queued" | "running" | "valid" | "invalid" | "cancelled" | "completed" | "decided" | "unverified";
  /** true only when candidateId binds AND the state is a recognized enum */
  verified: boolean;
  trusted?: boolean;
  reason?: string;
  report?: EvaluationReportProjection;
};

/** GET /environments/{id}/tasks: registered task goals a run may target. */
export type TaskOption = {
  taskId: string;
  goal: string;
  executionModes: string[];
};
