import type {
  ApprovalRequest,
  CandidateDiff,
  EnvironmentPackageSummary,
  RunEvent,
  RunRecord,
  SkillVersionSummary,
} from "./types";

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
  listRuns(): Promise<RunRecord[]>;
  listSkills(): Promise<SkillVersionSummary[]>;
  listCandidates(): Promise<CandidateDiff[]>;
  /** Subscribe to a run's event stream starting after `fromCursor` (exclusive). */
  openRunStream(
    runId: string,
    fromCursor: number,
    handlers: {
      onEvent: (event: RunEvent) => void;
      onState: (state: "live" | "stale" | "reconnecting" | "closed") => void;
    },
  ): () => void;
  cancelRun(runId: string): Promise<void>;
  submitApproval(runId: string, approvalId: string, approve: boolean): Promise<void>;
  requestRollback(candidateId: string, reason: string): Promise<void>;
  /** Create a run: the server pins the active version; the console never picks it. */
  createRun(input: CreateRunInput): Promise<RunRecord>;
  /** Register an environment package (full manifest) with server-side validation. */
  registerEnvironment(manifest: EnvironmentRegistration): Promise<EnvironmentPackageSummary>;
  /** Submit the evidence-linked candidate from the latest development attempts for evaluation. */
  runLearningCycle(): Promise<{ candidate: CandidateDiff }>;
  validateEnvironmentPackage(fields: EnvironmentPackageForm): Promise<{ ok: boolean; missingFields: string[] }>;
}

export type CreateRunInput = {
  environmentId: string;
  goal: string;
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

export type EnvironmentRegistration = {
  environmentId: string;
  version: string;
  docs: Array<{ id: string; sha256: string; classification: "learner" | "operator" }>;
  toolSchemas: Array<Record<string, unknown>>;
  policyRef: string;
  evaluatorRef: string;
  resetRef: string;
  executionModes: Array<"dry_run" | "interactive" | "batch" | "replay">;
  capabilities: string[];
};

export const EXECUTION_MODES: Array<{ value: EnvironmentRegistration["executionModes"][number]; label: string }> = [
  { value: "dry_run", label: "dry_run — validate without dispatching side effects" },
  { value: "interactive", label: "interactive — permits approval pauses" },
  { value: "batch", label: "batch — fails approval-required writes instead of waiting" },
  { value: "replay", label: "replay — reads recorded envelopes only" },
];

export type EnvironmentPackageForm = {
  environmentId: string;
  version: string;
  docs: string;
  toolSchemas: string;
  policyRef: string;
  evaluatorRef: string;
  resetRef: string;
  executionModes: EnvironmentRegistration["executionModes"];
  capabilities: string;
};

export const REQUIRED_PACKAGE_FIELDS: Array<{ key: keyof EnvironmentPackageForm; label: string }> = [
  { key: "environmentId", label: "Environment ID" },
  { key: "version", label: "Package version" },
  { key: "docs", label: "Documentation (JSON array with hashes and classification)" },
  { key: "toolSchemas", label: "Tool schemas (JSON array)" },
  { key: "policyRef", label: "Policy reference" },
  { key: "evaluatorRef", label: "Trusted evaluator reference" },
  { key: "resetRef", label: "Reset fixture reference" },
  { key: "capabilities", label: "Capability metadata" },
];

export const EMPTY_PACKAGE_FORM: EnvironmentPackageForm = {
  environmentId: "",
  version: "",
  docs: "",
  toolSchemas: "",
  policyRef: "",
  evaluatorRef: "",
  resetRef: "",
  executionModes: [],
  capabilities: "",
};

/** Client-side pre-validation mirroring the SPEC registration contract. */
export function validatePackageFields(fields: EnvironmentPackageForm): string[] {
  const missing: string[] = [];
  for (const { key } of REQUIRED_PACKAGE_FIELDS) {
    const value = fields[key];
    if (typeof value === "string" && !value.trim()) missing.push(key);
  }
  if (fields.executionModes.length === 0) missing.push("executionModes");
  for (const key of ["docs", "toolSchemas", "capabilities"] as const) {
    if (typeof fields[key] === "string" && (fields[key] as string).trim()) {
      try {
        const parsed = JSON.parse(fields[key] as string);
        if (!Array.isArray(parsed)) missing.push(key);
      } catch {
        missing.push(key);
      }
    }
  }
  if (fields.docs.trim()) {
    try {
      const docs = JSON.parse(fields.docs) as unknown[];
      const valid = docs.every(
        (d) =>
          typeof d === "object" && d !== null && "id" in d && "sha256" in d &&
          ((d as Record<string, unknown>).classification === "learner" ||
            (d as Record<string, unknown>).classification === "operator"),
      );
      if (!valid) missing.push("docs");
    } catch {
      missing.push("docs");
    }
  }
  return [...new Set(missing)];
}

export function formToRegistration(fields: EnvironmentPackageForm): EnvironmentRegistration {
  return {
    environmentId: fields.environmentId,
    version: fields.version,
    docs: JSON.parse(fields.docs) as EnvironmentRegistration["docs"],
    toolSchemas: JSON.parse(fields.toolSchemas) as Array<Record<string, unknown>>,
    policyRef: fields.policyRef,
    evaluatorRef: fields.evaluatorRef,
    resetRef: fields.resetRef,
    executionModes: fields.executionModes,
    capabilities: JSON.parse(fields.capabilities) as string[],
  };
}

export function newIdempotencyKey(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `idem-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
}
