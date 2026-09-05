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
 *   simulated in the UI (persistent banner). Never presented as live inference.
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
  validateEnvironmentPackage(fields: EnvironmentPackageForm): Promise<{ ok: boolean; missingFields: string[] }>;
}

export type EnvironmentPackageForm = {
  environmentId: string;
  version: string;
  toolSchemas: string;
  policyRef: string;
  evaluatorRef: string;
  resetRef: string;
};

export const REQUIRED_PACKAGE_FIELDS: Array<{ key: keyof EnvironmentPackageForm; label: string }> = [
  { key: "environmentId", label: "Environment ID" },
  { key: "version", label: "Package version" },
  { key: "toolSchemas", label: "Tool schemas (JSON)" },
  { key: "policyRef", label: "Policy reference" },
  { key: "evaluatorRef", label: "Trusted evaluator reference" },
  { key: "resetRef", label: "Reset fixture reference" },
];

/** Client-side pre-validation mirroring the SPEC registration contract. */
export function validatePackageFields(fields: EnvironmentPackageForm): string[] {
  const missing: string[] = [];
  for (const { key } of REQUIRED_PACKAGE_FIELDS) {
    if (!fields[key] || !String(fields[key]).trim()) missing.push(key);
  }
  if (fields.toolSchemas && fields.toolSchemas.trim()) {
    try {
      const parsed = JSON.parse(fields.toolSchemas);
      if (!Array.isArray(parsed)) missing.push("toolSchemas");
    } catch {
      missing.push("toolSchemas");
    }
  }
  return missing;
}
