import type {
  ApprovalRequest,
  CandidateDiff,
  DiagnosticRecord,
  EnvironmentPackageSummary,
  EvaluationJob,
  RunEvent,
  RunOptions,
  RunRecord,
  SkillVersionSummary,
  TaskOption,
} from "./types";

/** Canonical SPEC reference object: {id, version, sha256}. */
export type CanonicalRef = { id: string; version: string; sha256: string };

/**
 * Transport abstraction: the console speaks this interface only.
 * - `rest` implements it against the SPEC control API when the backend is reachable.
 * - `simulation` implements it with deterministic fixtures, ALWAYS labeled as
 *   simulated in the UI (persistent banner) and available only as an explicit
 *   dev-only opt-in. Never presented as live inference.
 */
export interface ConsoleTransport {
  readonly mode: "simulation" | "live";
  listEnvironments(): Promise<EnvironmentPackageSummary[]>;
  /** Authoritative model profiles and bounded budget defaults (GET /run-options). */
  getRunOptions(): Promise<RunOptions>;
  /** Registered task goals for an environment (durable runs must match one). */
  getEnvironmentTasks(environmentId: string): Promise<TaskOption[]>;
  listRuns(): Promise<RunRecord[]>;
  listSkills(): Promise<SkillVersionSummary[]>;
  listCandidates(): Promise<CandidateDiff[]>;
  /** Subscribe to a run's event stream starting after `fromCursor` (exclusive). */
  openRunStream(
    runId: string,
    fromCursor: number,
    handlers: {
      onEvent: (event: RunEvent) => void;
      onState: (state: "live" | "stale" | "reconnecting" | "disconnected" | "closed") => void;
    },
  ): () => void;
  cancelRun(runId: string): Promise<void>;
  submitApproval(runId: string, approvalId: string, approve: boolean): Promise<void>;
  requestRollback(candidateId: string, reason: string): Promise<void>;
  /** Re-run the session handshake after a disconnect; simulation resolves. */
  reconnect(): Promise<void>;
  /** Create a run via the canonical taskRef; the server pins the active version. */
  createRun(input: CreateRunInput): Promise<RunRecord>;
  /** Launch a created run (POST /runs/{id}/launch, 202 accepted). */
  launchRun(runId: string): Promise<void>;
  /** Register an environment: strict full manifest (docs + taskGoals + modes). */
  registerEnvironment(manifest: EnvironmentRegistration): Promise<EnvironmentPackageSummary>;
  /** Stage an evidence-linked learning proposal (POST /learning/launch). */
  launchLearningCycle(input: LearningCycleInput): Promise<{ actionId: string; runId: string; status: string }>;
  /** Queue a trusted evaluation for a validated candidate (POST /evaluations). */
  launchEvaluation(input: { candidateId: string; baseBundleHash: string }): Promise<{ evaluationId: string; state: string }>;
  /** Evaluation job statuses (GET /evaluations). */
  listEvaluations(): Promise<EvaluationJob[]>;
  /** Quick development comparison: launch (POST /diagnostics/launch). */
  launchDiagnostic(input: { candidateId: string; baseBundleHash: string }): Promise<{ diagnosticId: string; state: string }>;
  /** Quick development comparison rows (GET /diagnostics). */
  listDiagnostics(): Promise<DiagnosticRecord[]>;
  /** Cancel a quick comparison (POST /diagnostics/{id}/cancel). */
  cancelDiagnostic(diagnosticId: string): Promise<void>;
  validateEnvironmentPackage(fields: EnvironmentPackageForm): Promise<{ ok: boolean; missingFields: string[] }>;
}

export type CreateRunInput = {
  environmentId: string;
  goal: string;
  /** registered task id from getEnvironmentTasks; sent as the canonical taskRef */
  taskId?: string;
  /** server-provided authoritative model ref from run-options (preferred) */
  modelProfileRef?: { id: string; version: string; sha256: string };
  /** server-provided authoritative budget ref from run-options (2b3fc75);
   *  submitted verbatim alongside the validated budget object */
  budgetRef?: { id: string; version: string; sha256: string };
  modelProfile: string;
  /** idempotency key supplied by the console; reused verbatim on retry */
  idempotencyKey: string;
  budget: {
    toolCallCeiling: number;
    wallSecondsCeiling: number;
    /** nonzero model cost/token cap required before execution (SPEC) */
    modelTokenCeiling: number;
  };
  executionMode: "dry_run" | "interactive" | "batch" | "replay";
};

/** runId-only request: the durable learning pipeline generates the proposal
 *  and evidence from the run's development attempts; the operator never
 *  supplies predicted effects or evidence ids (the runtime ignores them). */
export type LearningCycleInput = {
  runId: string;
};

/**
 * Strict full manifest sent to POST /environments (commit 7d3c2b5):
 * requires docs[], taskGoals[], and canonical reference objects.
 */
export type EnvironmentRegistration = {
  schemaVersion?: 1;
  environmentId: string;
  version: string;
  docs: Array<{ id: string; version: string; sha256: string }>;
  taskGoals: string[];
  toolSchemas: Array<{ name: string; version: string; inputSchema: unknown; outputSchema: unknown; effect: "read" | "write" }>;
  policyRef: CanonicalRef;
  evaluatorRef: CanonicalRef;
  resetRef: CanonicalRef;
  executionModes: Array<"dry_run" | "interactive" | "batch" | "replay">;
  capabilities: string[];
};

export const EXECUTION_MODES: Array<{ value: EnvironmentRegistration["executionModes"][number]; label: string }> = [
  { value: "dry_run", label: "dry_run — validate without dispatching side effects" },
  { value: "interactive", label: "interactive — permits approval pauses" },
  { value: "batch", label: "batch — fails approval-required writes instead of waiting" },
  { value: "replay", label: "replay — reads recorded envelopes only" },
];

/** String projection used by the browser form and POST /environments/validate
 *  and POST /environments/form (commit 7d3c2b5 compatibility). */
export type EnvironmentPackageForm = {
  environmentId: string;
  version: string;
  docs: string;
  taskGoals: string;
  toolSchemas: string;
  policyRef: string;
  evaluatorRef: string;
  resetRef: string;
  executionModes: EnvironmentRegistration["executionModes"];
};

export const REQUIRED_PACKAGE_FIELDS: Array<{ key: keyof EnvironmentPackageForm; label: string }> = [
  { key: "environmentId", label: "Environment ID" },
  { key: "version", label: "Package version" },
  { key: "docs", label: "Documentation (JSON array with hashes)" },
  { key: "taskGoals", label: "Task goals (JSON array of strings)" },
  { key: "toolSchemas", label: "Tool schemas (JSON array)" },
  { key: "policyRef", label: "Policy reference" },
  { key: "evaluatorRef", label: "Trusted evaluator reference" },
  { key: "resetRef", label: "Reset fixture reference" },
];

export const EMPTY_PACKAGE_FORM: EnvironmentPackageForm = {
  environmentId: "",
  version: "",
  docs: "",
  taskGoals: "",
  toolSchemas: "",
  policyRef: "",
  evaluatorRef: "",
  resetRef: "",
  executionModes: [],
};

/** sha256 hex; falls back to a deterministic non-cryptographic digest where
 *  WebCrypto is unavailable (test environments). The server re-hashes at the
 *  boundary, so this only needs to be stable and non-empty. */
export async function sha256Hex(value: string): Promise<string> {
  if (typeof crypto !== "undefined" && crypto.subtle) {
    const bytes = new TextEncoder().encode(value);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
  }
  let h = 0x811c9dc5;
  for (let i = 0; i < value.length; i++) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return `fnv${h.toString(16).padStart(8, "0")}`.padEnd(64, "0");
}

async function canonicalRef(identifier: string): Promise<CanonicalRef> {
  return { id: identifier, version: "1", sha256: await sha256Hex(identifier) };
}

/** Client-side pre-validation mirroring the server's required-field checks. */
export function validatePackageFields(fields: EnvironmentPackageForm): string[] {
  const missing: string[] = [];
  for (const { key } of REQUIRED_PACKAGE_FIELDS) {
    const value: unknown = fields[key];
    if (typeof value === "string" && !value.trim()) missing.push(key);
  }
  if (fields.executionModes.length === 0) missing.push("executionModes");
  for (const key of ["docs", "toolSchemas", "taskGoals"] as const) {
    try {
      const parsed = JSON.parse(fields[key]) as unknown;
      if (key === "taskGoals") {
        const goals = Array.isArray(parsed) ? parsed : [parsed];
        if (!goals.every((g) => typeof g === "string" && g.length > 0)) missing.push(key);
      } else if (!Array.isArray(parsed)) {
        missing.push(key);
      } else if (key === "docs") {
        const valid = parsed.every(
          (d) =>
            typeof d === "object" && d !== null && "id" in d && "sha256" in d &&
            ((d as Record<string, unknown>).classification === "learner" ||
              (d as Record<string, unknown>).classification === "operator" ||
              typeof (d as Record<string, unknown>).sha256 === "string"),
        );
        if (!valid) missing.push(key);
      }
    } catch {
      missing.push(key);
    }
  }
  return [...new Set(missing)];
}

/** Map the string form to the strict full manifest (commit 7d3c2b5). */
export async function formToRegistration(fields: EnvironmentPackageForm): Promise<EnvironmentRegistration> {
  const goalsRaw = JSON.parse(fields.taskGoals) as unknown;
  const taskGoals = Array.isArray(goalsRaw) ? (goalsRaw as string[]) : [goalsRaw as string];
  const docsRaw = JSON.parse(fields.docs) as Array<Record<string, unknown>>;
  const docs = docsRaw.map((d) => ({
    id: String(d.id),
    version: String(d.version ?? "1"),
    sha256: String(d.sha256),
  }));
  const toolSchemasRaw = JSON.parse(fields.toolSchemas) as Array<Record<string, unknown>>;
  return {
    environmentId: fields.environmentId,
    version: fields.version,
    docs,
    taskGoals,
    toolSchemas: toolSchemasRaw.map((t) => ({
      name: String(t.name),
      version: String(t.version ?? "1"),
      inputSchema: t.inputSchema ?? { type: "object" },
      outputSchema: t.outputSchema ?? { type: "object" },
      effect: t.effect === "write" ? ("write" as const) : ("read" as const),
    })),
    policyRef: await canonicalRef(fields.policyRef),
    evaluatorRef: await canonicalRef(fields.evaluatorRef),
    resetRef: await canonicalRef(fields.resetRef),
    executionModes: fields.executionModes,
    capabilities: [],
  };
}

/** String projection for /environments/validate and /environments/form. */
export function formToStringPayload(fields: EnvironmentPackageForm): Record<string, string> {
  return {
    environmentId: fields.environmentId,
    version: fields.version,
    toolSchemas: fields.toolSchemas,
    policyRef: fields.policyRef,
    evaluatorRef: fields.evaluatorRef,
    resetRef: fields.resetRef,
    taskGoals: fields.taskGoals,
  };
}

export function newIdempotencyKey(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `idem-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
}
