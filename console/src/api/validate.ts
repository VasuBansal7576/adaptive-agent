/**
 * Boundary-schema validation for every payload crossing the transport.
 *
 * The console never trusts API payloads: unknown fields are dropped and
 * required fields/enums are checked before data enters the UI. A payload that
 * fails validation raises SchemaError instead of flowing through unchecked.
 */
import type {
  ApprovalRequest,
  ArtifactRef,
  CandidateDiff,
  DiagnosticRecord,
  EnvironmentPackageSummary,
  RunEvent,
  RunOptions,
  RunRecord,
  RunStatus,
  SkillVersionSummary,
  TaskOption,
  ToolError,
  ToolErrorCode,
} from "./types";

export class SchemaError extends Error {
  constructor(field: string) {
    super(`schema violation at ${field}`);
    this.name = "SchemaError";
  }
}

type Unknown = Record<string, unknown>;

function obj(value: unknown, field: string): Unknown {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new SchemaError(field);
  return value as Unknown;
}

export function str(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0) throw new SchemaError(field);
  return value;
}

function optStr(value: unknown, field: string): string | undefined {
  if (value === undefined || value === null) return undefined;
  return str(value, field);
}

function num(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new SchemaError(field);
  return value;
}

function arr(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) throw new SchemaError(field);
  return value;
}

function oneOf<T extends string>(value: unknown, allowed: readonly T[], field: string): T {
  if (typeof value !== "string" || !allowed.includes(value as T)) throw new SchemaError(field);
  return value as T;
}

const RUN_STATUSES = ["queued", "running", "awaiting_approval", "succeeded", "failed", "cancelled", "timed_out"] as const;
const ERROR_CODES = ["INVALID_INPUT", "FORBIDDEN", "VERSION_CONFLICT", "IDEMPOTENCY_CONFLICT", "BUDGET_EXHAUSTED", "TOOL_UNAVAILABLE", "OUTCOME_UNKNOWN"] as const;
const EVENT_KINDS = ["status", "step", "tool", "evidence", "budget", "approval"] as const;

export function parseArtifactRef(value: unknown, field: string): ArtifactRef {
  const o = obj(value, field);
  return { id: str(o.id, `${field}.id`), version: str(o.version, `${field}.version`), sha256: str(o.sha256, `${field}.sha256`) };
}

export function parseToolError(value: unknown, field: string): ToolError {
  const o = obj(value, field);
  return {
    code: oneOf(o.code, ERROR_CODES, `${field}.code`),
    message: str(o.message, `${field}.message`),
    correlationId: str(o.correlationId, `${field}.correlationId`),
    retry: oneOf(o.retry, ["never", "safe_read", "after_reconciliation"] as const, `${field}.retry`),
  };
}

function parseApproval(value: unknown, field: string): ApprovalRequest {
  const o = obj(value, field);
  const effect = oneOf(o.effect, ["read", "write"] as const, `${field}.effect`);
  return {
    approvalId: str(o.approvalId, `${field}.approvalId`),
    tool: str(o.tool, `${field}.tool`),
    toolVersion: str(o.toolVersion, `${field}.toolVersion`),
    canonicalArguments: str(o.canonicalArguments, `${field}.canonicalArguments`),
    resourceScope: str(o.resourceScope, `${field}.resourceScope`),
    effect,
    expiresAt: str(o.expiresAt, `${field}.expiresAt`),
  };
}

export function parseRunEvent(value: unknown, field: string): RunEvent {
  const o = obj(value, field);
  const event: RunEvent = {
    runId: str(o.runId, `${field}.runId`),
    sequence: num(o.sequence, `${field}.sequence`),
    at: str(o.at, `${field}.at`),
    kind: oneOf(o.kind, EVENT_KINDS, `${field}.kind`),
    summary: str(o.summary, `${field}.summary`),
  };
  if (o.detail !== undefined) event.detail = str(o.detail, `${field}.detail`);
  if (o.error !== undefined) event.error = parseToolError(o.error, `${field}.error`);
  if (o.approval !== undefined) event.approval = parseApproval(o.approval, `${field}.approval`);
  return event;
}

/** Operator-readable summaries per durable event type; distinguishes runtime
 *  failures from outcome-check results so a dry_run preview is not mistaken
 *  for a runtime error. */
const SUMMARY_BY_TYPE: Record<string, string> = {
  run_created: "Run created",
  run_started: "Run started",
  run_failed: "Runtime failure recorded",
  run_cancelled: "Run cancelled",
  run_completed: "Run completed",
  run_succeeded: "Run completed",
  run_timed_out: "Run timed out",
  outcome_recorded: "Trusted outcome check recorded",
  trusted_outcome: "Trusted outcome recorded",
  model_observation: "Model observation recorded",
  model_response: "Model response recorded",
  step_started: "Step started",
  step_completed: "Step completed",
  approval: "Approval recorded",
};

/** Durable lifecycle event types -> authoritative RunStatus transitions.
 *  Status is taken ONLY from the validated event type, never display text. */
const EVENT_TYPE_TO_STATUS: Record<string, RunStatus> = {
  run_created: "queued",
  run_started: "running",
  run_completed: "succeeded",
  run_succeeded: "succeeded",
  run_failed: "failed",
  run_cancelled: "cancelled",
  run_timed_out: "timed_out",
};

/** Durable evidence event types -> console event kinds. */
const EVENT_TYPE_TO_KIND: Record<string, RunEvent["kind"]> = {
  run_created: "status",
  run_started: "status",
  run_completed: "status",
  run_succeeded: "status",
  run_failed: "status",
  run_cancelled: "status",
  run_timed_out: "status",
  step_started: "step",
  step_completed: "step",
  tool_called: "tool",
  tool_result: "tool",
  approval: "approval",
  approval_requested: "approval",
  approval_decided: "approval",
  outcome_recorded: "evidence",
  trusted_outcome: "evidence",
  model_observation: "evidence",
  model_response: "evidence",
  evidence: "evidence",
  budget: "budget",
  budget_reserved: "budget",
  budget_recorded: "budget",
};

/**
 * Normalize both SSE wire shapes into the console RunEvent:
 * - plane projection: {runId, sequence, at, kind, summary, ...}
 * - durable envelope: {id, event, data:{runId, sequence, eventType, ...}}
 * Unknown or partial payloads raise SchemaError instead of entering UI state.
 */
export function normalizeSseEvent(value: unknown, field: string): RunEvent {
  const o = obj(value, field);
  if (typeof o.kind === "string" && typeof o.summary === "string" && typeof o.runId === "string") {
    return parseRunEvent(o, field);
  }
  // durable evidence envelope (snake_case on the wire)
  const envelopeId = typeof o.id === "number" ? o.id : undefined;
  const data = obj(o.data ?? o, `${field}.data`);
  const runId = str(data.runId ?? data.run_id, `${field}.data.runId`);
  const sequence = typeof data.sequence === "number"
    ? num(data.sequence, `${field}.data.sequence`)
    : envelopeId !== undefined
      ? num(envelopeId, `${field}.id`)
      : (() => { throw new SchemaError(`${field}.sequence`); })();
  const eventType = str(data.eventType ?? data.event_type ?? o.event, `${field}.eventType`);
  const kind = EVENT_TYPE_TO_KIND[eventType] ?? "step";
  // source_ref may be an embedded JSON string on the durable wire
  let sourceId: string | null = null;
  const sourceRaw = data.sourceRef ?? data.source_ref;
  if (typeof sourceRaw === "string") {
    try {
      const parsed = JSON.parse(sourceRaw) as Record<string, unknown>;
      if (typeof parsed.id === "string") sourceId = parsed.id;
    } catch {
      sourceId = null;
    }
  } else if (typeof sourceRaw === "object" && sourceRaw !== null && typeof (sourceRaw as Record<string, unknown>).id === "string") {
    sourceId = (sourceRaw as Record<string, unknown>).id as string;
  }
  const contentHashRaw = data.contentHash ?? data.content_hash;
  const contentHash = typeof contentHashRaw === "string" ? contentHashRaw : null;
  // safe operator payload fields when the projection includes them
  // (visibility-gated server-side; evaluator_only rows never reach the client)
  const payloadSummaryRaw = data.summary ?? data.payload_summary;
  const payloadDetailRaw = data.detail ?? data.payload_detail;
  const detail = sourceId
    ? `evidence artifact ${sourceId}`
    : contentHash
      ? `content ${contentHash.slice(0, 12)}`
      : undefined;
  const event: RunEvent = {
    runId,
    sequence,
    at: typeof data.at === "string" ? data.at : "",
    kind,
    summary:
      typeof payloadSummaryRaw === "string" && payloadSummaryRaw
        ? payloadSummaryRaw
        : (SUMMARY_BY_TYPE[eventType] ?? `[${eventType}] evidence recorded`),
  };
  if (typeof payloadDetailRaw === "string" && payloadDetailRaw) event.detail = payloadDetailRaw;
  else if (typeof detail === "string") event.detail = detail;
  else if (payloadDetailRaw !== undefined) event.detail = String(payloadDetailRaw);
  const evidenceIdRaw = data.evidenceId ?? data.evidence_id;
  event.evidence = {
    evidenceId: typeof evidenceIdRaw === "string" ? evidenceIdRaw : undefined,
    sourceRefId: sourceId ?? undefined,
    contentHash: contentHash ?? undefined,
    trustClass: typeof data.trustClass === "string" ? data.trustClass : typeof data.trust_class === "string" ? data.trust_class : undefined,
    visibility: typeof data.visibility === "string" ? data.visibility : undefined,
    redacted: typeof data.redacted === "boolean" ? data.redacted : typeof data.redacted === "number" ? data.redacted === 1 : undefined,
  };
  event.lifecycleType = eventType;
  // lifecycle status comes from the validated event type only
  const statusTransition = EVENT_TYPE_TO_STATUS[eventType];
  if (statusTransition) event.runStatus = statusTransition;
  // bare status and outcome rows carry no status field server-side: the
  // authoritative RunRecord must be refreshed to learn the real state
  if (eventType === "status" || eventType === "outcome_recorded") event.needsRecordRefresh = true;
  return event;
}

export function parseRun(value: unknown, field: string): RunRecord {
  const o = obj(value, field);
  const run: RunRecord = {
    runId: str(o.runId, `${field}.runId`),
    taskRef: parseArtifactRef(o.taskRef, `${field}.taskRef`),
    environmentRef: parseArtifactRef(o.environmentRef, `${field}.environmentRef`),
    policyRef: parseArtifactRef(o.policyRef, `${field}.policyRef`),
    modelProfileRef: parseArtifactRef(o.modelProfileRef, `${field}.modelProfileRef`),
    skillBundleRef: parseArtifactRef(o.skillBundleRef, `${field}.skillBundleRef`),
    budgetRef: parseArtifactRef(o.budgetRef, `${field}.budgetRef`),
    status: oneOf(o.status, RUN_STATUSES, `${field}.status`) as RunStatus,
    lastEventSequence: num(o.lastEventSequence, `${field}.lastEventSequence`),
  };
  if (o.outcomeRef !== undefined) run.outcomeRef = parseArtifactRef(o.outcomeRef, `${field}.outcomeRef`);
  if (o.executionMode !== undefined) {
    run.executionMode = oneOf(o.executionMode, ["dry_run", "interactive", "batch", "replay"] as const, `${field}.executionMode`);
  }
  if (o.environmentId !== undefined) run.environmentId = optStr(o.environmentId, `${field}.environmentId`);
  if (o.goal !== undefined) run.goal = optStr(o.goal, `${field}.goal`);
  if (o.executionModes !== undefined) {
    run.executionModes = arr(o.executionModes, `${field}.executionModes`).map((m) => str(m, `${field}.executionModes[]`));
  }
  if (o.learningEligible !== undefined) {
    if (typeof o.learningEligible !== "boolean") throw new SchemaError(`${field}.learningEligible`);
    run.learningEligible = o.learningEligible;
  }
  if (o.budgetUsed !== undefined) {
    const b = obj(o.budgetUsed, `${field}.budgetUsed`);
    run.budgetUsed = {
      calls: num(b.calls, `${field}.budgetUsed.calls`),
      callsCeiling: num(b.callsCeiling, `${field}.budgetUsed.callsCeiling`),
      wallSeconds: num(b.wallSeconds, `${field}.budgetUsed.wallSeconds`),
      wallCeiling: num(b.wallCeiling, `${field}.budgetUsed.wallCeiling`),
    };
  }
  return run;
}

export function parseRuns(value: unknown): RunRecord[] {
  return arr(value, "runs").map((item, i) => parseRun(item, `runs[${i}]`));
}

export function parseEvents(value: unknown): RunEvent[] {
  return arr(value, "events").map((item, i) => parseRunEvent(item, `events[${i}]`));
}

const SKILL_STATES = ["active", "proposed", "rejected", "quarantined", "rolled_back"] as const;

export function parseSkills(value: unknown): SkillVersionSummary[] {
  return arr(value, "skills").map((item, i) => {
    const o = obj(item, `skills[${i}]`);
    return {
      skillId: str(o.skillId, `skills[${i}].skillId`),
      version: str(o.version, `skills[${i}].version`),
      parentVersion: o.parentVersion === null ? null : str(o.parentVersion, `skills[${i}].parentVersion`),
      state: oneOf(o.state, SKILL_STATES, `skills[${i}].state`),
      applicability: str(o.applicability, `skills[${i}].applicability`),
      evidenceRefs: arr(o.evidenceRefs, `skills[${i}].evidenceRefs`).map((r) => str(r, `skills[${i}].evidenceRefs[]`)),
      contentHash: str(o.contentHash, `skills[${i}].contentHash`),
    };
  });
}

const CANDIDATE_STATES = ["draft", "validated", "evaluating", "promoted", "rejected", "quarantined", "superseded", "rolled_back"] as const;
const GATE_DECISIONS = ["promoted", "rejected", "quarantined"] as const;

/**
 * Both candidate projections are accepted and validated:
 * - plane: {baseBundleRef, candidateBundleRef, diff, ...}
 * - durable: {baseBundleHash, candidateBundleHash, editOperations,
 *   changedArtifactHashes, supportingEvidenceIds, proposerVersion, ...}
 * Unknown/missing required fields raise SchemaError instead of silently
 * hiding real candidates behind an empty state.
 */
export function parseCandidates(value: unknown): CandidateDiff[] {
  return arr(value, "candidates").map((item, i) => {
    const o = obj(item, `candidates[${i}]`);
    const field = `candidates[${i}]`;
    const candidateId = str(o.candidateId, `${field}.candidateId`);
    const state = oneOf(o.state, CANDIDATE_STATES, `${field}.state`);
    const predictedEffect = str(o.predictedEffect ?? "", `${field}.predictedEffect`);
    const cand: CandidateDiff = { candidateId, state, predictedEffect };
    if (o.baseBundleRef !== undefined) {
      cand.baseBundleRef = parseArtifactRef(o.baseBundleRef, `${field}.baseBundleRef`);
      cand.candidateBundleRef = parseArtifactRef(o.candidateBundleRef, `${field}.candidateBundleRef`);
      cand.diff = str(o.diff, `${field}.diff`);
    } else if (typeof o.baseBundleHash === "string") {
      // durable projection
      cand.baseBundleHash = str(o.baseBundleHash, `${field}.baseBundleHash`);
      if (o.candidateBundleHash !== undefined && o.candidateBundleHash !== null) {
        cand.candidateBundleHash = str(o.candidateBundleHash, `${field}.candidateBundleHash`);
      }
      cand.editOperations = arr(o.editOperations, `${field}.editOperations`).map((op, j) => str(op, `${field}.editOperations[${j}]`));
      cand.changedArtifactHashes = arr(o.changedArtifactHashes, `${field}.changedArtifactHashes`).map((h) => str(h, `${field}.changedArtifactHashes[]`));
      cand.supportingEvidenceIds = arr(o.supportingEvidenceIds, `${field}.supportingEvidenceIds`).map((e) => str(e, `${field}.supportingEvidenceIds[]`));
      if (o.proposerVersion !== undefined) cand.proposerVersion = str(o.proposerVersion, `${field}.proposerVersion`);
    } else {
      throw new SchemaError(`${field}.projection`);
    }
    if (o.measured !== undefined) {
      const m = obj(o.measured, `${field}.measured`);
      cand.measured = {
        accuracyGainPp: num(m.accuracyGainPp, `${field}.measured.accuracyGainPp`),
        reliabilityDelta: num(m.reliabilityDelta, `${field}.measured.reliabilityDelta`),
        costRatio: num(m.costRatio, `${field}.measured.costRatio`),
        p95LatencyRatio: num(m.p95LatencyRatio, `${field}.measured.p95LatencyRatio`),
        gateDecision: oneOf(m.gateDecision, GATE_DECISIONS, `${field}.measured.gateDecision`),
        gateReasons: arr(m.gateReasons, `${field}.measured.gateReasons`).map((r) => str(r, "gateReasons[]")),
        evaluationRef: parseArtifactRef(m.evaluationRef, `${field}.measured.evaluationRef`),
      };
    }
    if (o.audit !== undefined) {
      cand.audit = arr(o.audit, `${field}.audit`).map((entry, j) => {
        const a = obj(entry, `${field}.audit[${j}]`);
        return {
          rollbackId: str(a.rollbackId, "rollbackId"),
          fromRef: parseArtifactRef(a.fromRef, "fromRef"),
          toRef: parseArtifactRef(a.toRef, "toRef"),
          reason: str(a.reason, "reason"),
          at: str(a.at, "at"),
          affectedRuns: arr(a.affectedRuns, "affectedRuns").map((r) => str(r, "affectedRuns[]")),
        };
      });
    }
    return cand;
  });
}

const ENV_VALIDATION = ["valid", "invalid", "unchecked"] as const;
export function parseEnvironments(value: unknown): EnvironmentPackageSummary[] {
  return arr(value, "environments").map((item, i) => {
    const o = obj(item, `environments[${i}]`);
    const env: EnvironmentPackageSummary = {
      environmentId: str(o.environmentId, `environments[${i}].environmentId`),
      version: str(o.version, `environments[${i}].version`),
      validationState: oneOf(o.validationState, ENV_VALIDATION, `environments[${i}].validationState`),
      evaluatorReady: typeof o.evaluatorReady === "boolean" ? o.evaluatorReady : (() => { throw new SchemaError(`environments[${i}].evaluatorReady`); })(),
      toolCount: num(o.toolCount, `environments[${i}].toolCount`),
      policyScope: str(o.policyScope, `environments[${i}].policyScope`),
    };
    if (o.missingFields !== undefined) {
      env.missingFields = arr(o.missingFields, `environments[${i}].missingFields`).map((f) => str(f, "missingFields[]"));
    }
    // advertised by the durable runtime (51d476c): the console mode filter
    // consumes these instead of offering undeclared modes
    if (o.executionModes !== undefined) {
      env.executionModes = arr(o.executionModes, `environments[${i}].executionModes`).map((m) => str(m, `environments[${i}].executionModes[]`));
    }
    if (o.capabilities !== undefined) {
      env.capabilities = arr(o.capabilities, `environments[${i}].capabilities`).map((c) => str(c, `environments[${i}].capabilities[]`));
    }
    return env;
  });
}

const DIAG_STATES = ["queued", "running", "completed", "failed", "cancelled"] as const;

export function parseDiagnostics(value: unknown): DiagnosticRecord[] {
  return arr(value, "diagnostics").map((item, i) => {
    const o = obj(item, `diagnostics[${i}]`);
    const field = `diagnostics[${i}]`;
    const diagnosticId = str(o.diagnosticId, `${field}.diagnosticId`);
    const candidateId = str(o.candidateId, `${field}.candidateId`);
    const baseBundleHash = str(o.baseBundleHash, `${field}.baseBundleHash`);
    const candidateBundleHash = str(o.candidateBundleHash, `${field}.candidateBundleHash`);
    const state = oneOf(o.state, DIAG_STATES, `${field}.state`);
    const completedCells = num(o.completedCells, `${field}.completedCells`);
    const totalCells = num(o.totalCells, `${field}.totalCells`);
    if (totalCells !== 6) throw new SchemaError(`${field}.totalCells`);
    // startedAt is null while queued; never coerced to a timestamp
    const startedAt: string | null =
      o.startedAt === null || o.startedAt === undefined ? null : str(o.startedAt, `${field}.startedAt`);
    const updatedAt = str(o.updatedAt, `${field}.updatedAt`);
    const arms = arr(o.armSummaries, `${field}.armSummaries`).map((raw, j) => {
      const a = obj(raw, `${field}.armSummaries[${j}]`);
      const armName = oneOf(a.arm, ["B0", "L"] as const, `${field}.armSummaries[${j}].arm`);
      // meanScore is null until a task completes for this arm; never coerced to 0
      const meanScore: number | null =
        a.meanScore === null || a.meanScore === undefined ? null : num(a.meanScore, `${field}.armSummaries[${j}].meanScore`);
      const arm: DiagnosticRecord["armSummaries"][number] = {
        arm: armName,
        completed: num(a.completed, `${field}.armSummaries[${j}].completed`),
        successes: num(a.successes, `${field}.armSummaries[${j}].successes`),
        meanScore,
        totalTokens: num(a.totalTokens, `${field}.armSummaries[${j}].totalTokens`),
        wallDurationSeconds: num(a.wallDurationSeconds, `${field}.armSummaries[${j}].wallDurationSeconds`),
      };
      if (a.infrastructureErrors !== undefined) {
        arm.infrastructureErrors = num(a.infrastructureErrors, `${field}.armSummaries[${j}].infrastructureErrors`);
      }
      return arm;
    });
    if (o.error !== undefined && o.error !== null && typeof o.error !== "string") {
      throw new SchemaError(`${field}.error`);
    }
    if (o.promotionEligible !== false) throw new SchemaError(`${field}.promotionEligible`);
    const resumable = o.resumable;
    const row: DiagnosticRecord = {
      diagnosticId,
      candidateId,
      baseBundleHash,
      candidateBundleHash,
      state,
      completedCells,
      totalCells,
      startedAt,
      updatedAt,
      armSummaries: arms,
      error: (o.error as string | undefined) ?? null,
      promotionEligible: false,
    };
    if (typeof resumable === "boolean") row.resumable = resumable;
    return row;
  });
}

export function parseRunOptions(value: unknown): RunOptions {
  const o = obj(value, "runOptions");
  const profiles = arr(o.modelProfiles, "runOptions.modelProfiles").map((item, i) => {
    const p = obj(item, `runOptions.modelProfiles[${i}]`);
    const ref = obj(p.ref, `runOptions.modelProfiles[${i}].ref`);
    const out: RunOptions["modelProfiles"][number] = {
      ref: {
        id: str(ref.id, `runOptions.modelProfiles[${i}].ref.id`),
        version: str(ref.version, `runOptions.modelProfiles[${i}].ref.version`),
        sha256: str(ref.sha256, `runOptions.modelProfiles[${i}].ref.sha256`),
      },
      label: str(p.label, `runOptions.modelProfiles[${i}].label`),
    };
    if (p.provider !== undefined) out.provider = str(p.provider, `runOptions.modelProfiles[${i}].provider`);
    if (p.model !== undefined) out.model = str(p.model, `runOptions.modelProfiles[${i}].model`);
    return out;
  });
  const budget = obj(o.budgetDefaults, "runOptions.budgetDefaults");
  return {
    modelProfiles: profiles,
    budgetDefaults: {
      modelTokens: num(budget.modelTokens, "runOptions.budgetDefaults.modelTokens"),
      toolCalls: num(budget.toolCalls, "runOptions.budgetDefaults.toolCalls"),
      wallTimeSeconds: num(budget.wallTimeSeconds, "runOptions.budgetDefaults.wallTimeSeconds"),
      childRuns: typeof budget.childRuns === "number" ? budget.childRuns : undefined,
      costMicrounits: typeof budget.costMicrounits === "number" ? budget.costMicrounits : undefined,
      currency: typeof budget.currency === "string" ? budget.currency : undefined,
    },
    // trusted budget ref: top level since 01f2462, nested since 2b3fc75
    budgetRef:
      budget.budgetRef && typeof budget.budgetRef === "object" && !Array.isArray(budget.budgetRef)
        ? parseArtifactRef(budget.budgetRef, "runOptions.budgetDefaults.budgetRef")
        : o.budgetRef && typeof o.budgetRef === "object" && !Array.isArray(o.budgetRef)
          ? parseArtifactRef(o.budgetRef, "runOptions.budgetRef")
          : undefined,
  };
}

export function parseTasks(value: unknown): TaskOption[] {
  return arr(value, "tasks").map((item, i) => {
    const t = obj(item, `tasks[${i}]`);
    return {
      taskId: str(t.taskId, `tasks[${i}].taskId`),
      goal: str(t.goal, `tasks[${i}].goal`),
      executionModes: arr(t.executionModes, `tasks[${i}].executionModes`).map((m) => str(m, `tasks[${i}].executionModes[]`)),
    };
  });
}
