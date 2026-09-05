import type {
  CandidateDiff,
  EnvironmentPackageSummary,
  RunEvent,
  RunRecord,
  SkillVersionSummary,
} from "./types";
import type { ConsoleTransport, EnvironmentPackageForm } from "./transport";
import { validatePackageFields } from "./transport";

/**
 * DETERMINISTIC SIMULATION TRANSPORT — development fixture only.
 *
 * This module never contacts a backend and never represents live inference.
 * The UI renders a persistent "SIMULATED" banner whenever this transport is
 * active (see App.tsx). All sequences are fixed and reproducible.
 */

export const SIMULATION_LABEL = "SIMULATED — development fixture, not live inference";

type ScriptedEvent = Omit<RunEvent, "runId" | "at">;

const baseRun: RunRecord = {
  runId: "run-sim-1001",
  taskRef: { id: "task-finance-007", version: "1", sha256: "sim-task-hash-0000000000000000000000000000000000000001" },
  environmentRef: { id: "env-finance", version: "1.2.0", sha256: "sim-env-hash-000000000000000000000000000000000000002" },
  policyRef: { id: "policy-finance", version: "3", sha256: "sim-pol-hash-0000000000000000000000000000000000000003" },
  modelProfileRef: { id: "profile-subscription", version: "1", sha256: "sim-model-hash-00000000000000000000000000000000000004" },
  skillBundleRef: { id: "bundle-active", version: "7", sha256: "sim-bundle-hash-00000000000000000000000000000000000005" },
  budgetRef: { id: "budget-default", version: "1", sha256: "sim-budget-hash-00000000000000000000000000000000000006" },
  status: "running",
  lastEventSequence: 0,
  environmentId: "finance-sim",
  goal: "Reconcile the Q3 ledger batch and record the outcome.",
  budgetUsed: { calls: 5, callsCeiling: 100, wallSeconds: 96, wallCeiling: 900 },
};

const runCatalog: RunRecord[] = [
  baseRun,
  {
    ...baseRun,
    runId: "run-sim-1002",
    status: "awaiting_approval",
    lastEventSequence: 4,
    budgetUsed: { calls: 7, callsCeiling: 100, wallSeconds: 210, wallCeiling: 900 },
  },
  {
    ...baseRun,
    runId: "run-sim-1003",
    status: "failed",
    lastEventSequence: 6,
    outcomeRef: { id: "outcome-1003", version: "1", sha256: "sim-outcome-hash-00000000000000000000000000000000007" },
  },
  {
    ...baseRun,
    runId: "run-sim-1004",
    status: "succeeded",
    lastEventSequence: 8,
    outcomeRef: { id: "outcome-1004", version: "1", sha256: "sim-outcome-hash-00000000000000000000000000000000008" },
  },
  {
    ...baseRun,
    runId: "run-sim-1005",
    status: "cancelled",
    lastEventSequence: 3,
  },
  {
    ...baseRun,
    runId: "run-sim-1006",
    status: "timed_out",
    lastEventSequence: 5,
  },
];

const scripted: Record<string, ScriptedEvent[]> = {
  "run-sim-1001": [
    { sequence: 1, kind: "status", summary: "Run queued and pinned to bundle v7." },
    { sequence: 2, kind: "step", summary: "Retrieved 4 finance documentation chunks." },
    { sequence: 3, kind: "tool", summary: "inventory.read completed (read effect confirmed)." },
    { sequence: 4, kind: "tool", summary: "ledger.append completed (write effect confirmed)." },
    {
      sequence: 5,
      kind: "tool",
      summary: "ledger.append returned VERSION_CONFLICT.",
      detail: "Record changed between inspection and update; the agent rechecked the resource version and retried once.",
      error: { code: "VERSION_CONFLICT", message: "Expected version 12, found 13", correlationId: "corr-sim-5a", retry: "after_reconciliation" },
    },
    { sequence: 6, kind: "budget", summary: "Budget checkpoint: 5/100 calls, 96s/900s wall." },
  ],
  "run-sim-1002": [
    { sequence: 1, kind: "status", summary: "Run queued and pinned to bundle v7." },
    { sequence: 2, kind: "step", summary: "Retrieved 3 documentation chunks." },
    { sequence: 3, kind: "tool", summary: "inventory.read completed." },
    {
      sequence: 4,
      kind: "approval",
      summary: "Approval required for ledger.append (write).",
      approval: {
        approvalId: "aprv-0001",
        tool: "ledger.append",
        toolVersion: "1.4.0",
        canonicalArguments: JSON.stringify({ batchId: "Q3-2026-114", entries: 42, mode: "reconcile" }, null, 2),
        resourceScope: "finance/ledger/Q3-2026",
        effect: "write",
        expiresAt: "2026-09-06T23:59:00Z",
      },
    },
  ],
  "run-sim-1003": [
    { sequence: 1, kind: "status", summary: "Run queued and pinned to bundle v7." },
    { sequence: 2, kind: "tool", summary: "inventory.read completed." },
    {
      sequence: 3,
      kind: "tool",
      summary: "ledger.append returned OUTCOME_UNKNOWN after timeout.",
      error: { code: "OUTCOME_UNKNOWN", message: "Dispatch timed out before confirmation", correlationId: "corr-sim-3b", retry: "after_reconciliation" },
    },
    { sequence: 4, kind: "step", summary: "Next state-changing step blocked pending reconciliation." },
    { sequence: 5, kind: "status", summary: "Run failed: unresolved external effect after reconciliation window." },
    { sequence: 6, kind: "evidence", summary: "Trusted evaluator recorded outcome as not achieved." },
  ],
  "run-sim-1004": [
    { sequence: 1, kind: "status", summary: "Run queued and pinned to bundle v7." },
    { sequence: 2, kind: "tool", summary: "inventory.read completed." },
    { sequence: 3, kind: "tool", summary: "ledger.append completed." },
    { sequence: 4, kind: "step", summary: "All required outcome checks passed." },
    { sequence: 5, kind: "evidence", summary: "Trusted evaluator recorded outcome as achieved." },
    { sequence: 6, kind: "status", summary: "Run succeeded." },
  ],
  "run-sim-1005": [
    { sequence: 1, kind: "status", summary: "Run queued and pinned to bundle v7." },
    { sequence: 2, kind: "tool", summary: "inventory.read completed." },
    { sequence: 3, kind: "status", summary: "Cancelled by operator; future calls revoked. Dispatched external actions were not undone." },
    {
      sequence: 4,
      kind: "tool",
      summary: "Retry attempt rejected: idempotency key reused with changed arguments.",
      detail: "Operation-level reconciliation of the prior dispatch continues; the run stays closed.",
      error: { code: "IDEMPOTENCY_CONFLICT", message: "Key run-sim-1005:op-2 was used with a different canonical payload", correlationId: "corr-sim-5d", retry: "never" },
    },
  ],
  "run-sim-1006": [
    { sequence: 1, kind: "status", summary: "Run queued and pinned to bundle v7." },
    { sequence: 2, kind: "tool", summary: "inventory.read completed." },
    { sequence: 3, kind: "tool", summary: "reconciliation.sweep dispatched." },
    {
      sequence: 4,
      kind: "tool",
      summary: "Tool timeout with unconfirmed effect.",
      error: { code: "OUTCOME_UNKNOWN", message: "Wall clock exceeded 15m budget during dispatch", correlationId: "corr-sim-6c", retry: "never" },
    },
    { sequence: 5, kind: "status", summary: "Run timed out; reconciliation pending." },
  ],
};

const skillCatalog: SkillVersionSummary[] = [
  { skillId: "recheck-before-update", version: "3", parentVersion: "2", state: "active", applicability: "tools with optimistic versioning", evidenceRefs: ["ev-sim-31", "ev-sim-32"], contentHash: "sim-skill-hash-1" },
  { skillId: "batch-reconcile", version: "5", parentVersion: "4", state: "proposed", applicability: "multi-entry ledger batches", evidenceRefs: ["ev-sim-44"], contentHash: "sim-skill-hash-2" },
  { skillId: "unsafe-bulk-write", version: "1", parentVersion: null, state: "quarantined", applicability: "—", evidenceRefs: ["ev-sim-51"], contentHash: "sim-skill-hash-3" },
  { skillId: "stale-cache-read", version: "2", parentVersion: "1", state: "rejected", applicability: "—", evidenceRefs: ["ev-sim-61"], contentHash: "sim-skill-hash-4" },
  { skillId: "batch-reconcile", version: "4", parentVersion: "3", state: "rolled_back", applicability: "multi-entry ledger batches", evidenceRefs: ["ev-sim-40"], contentHash: "sim-skill-hash-5" },
];

const candidateCatalog: CandidateDiff[] = [
  {
    candidateId: "cand-sim-202",
    baseBundleRef: { id: "bundle-active", version: "7", sha256: "sim-bundle-hash-7" },
    candidateBundleRef: { id: "bundle-cand-202", version: "8", sha256: "sim-bundle-hash-8" },
    state: "promoted",
    diff: [
      "--- bundle-active@7/skills/recheck-before-update.py",
      "+++ bundle-cand-202@8/skills/recheck-before-update.py",
      "@@ -12,7 +12,10 @@",
      " def propose_update(record):",
      "-    return update(record.id, record.payload)",
      "+    current = read(record.id)",
      "+    if current.version != record.version:",
      "+        record = refresh(record)",
      "+    return update(record.id, record.payload)",
    ].join("\n"),
    predictedEffect: "Fewer stale updates on versioned records (prediction, not a score).",
    measured: {
      accuracyGainPp: 7.4,
      reliabilityDelta: 0.06,
      costRatio: 1.02,
      p95LatencyRatio: 1.04,
      gateDecision: "promoted",
      gateReasons: ["Accuracy gain ≥ 5pp with CI lower bound > 0", "No environment regressed", "Safety suite passed"],
      evaluationRef: { id: "eval-sim-88", version: "1", sha256: "sim-eval-hash-8" },
    },
    audit: [
      {
        rollbackId: "rb-sim-01",
        fromRef: { id: "bundle-cand-101", version: "7", sha256: "sim-bundle-hash-7a" },
        toRef: { id: "bundle-active", version: "6", sha256: "sim-bundle-hash-6" },
        reason: "Safety suite flagged unbounded child fan-out.",
        at: "2026-09-06T12:40:00Z",
        affectedRuns: ["run-sim-1003", "run-sim-1005"],
      },
    ],
  },
  {
    candidateId: "cand-sim-203",
    baseBundleRef: { id: "bundle-active", version: "7", sha256: "sim-bundle-hash-7" },
    candidateBundleRef: { id: "bundle-cand-203", version: "8", sha256: "sim-bundle-hash-9" },
    state: "rejected",
    diff: [
      "--- bundle-active@7/config/execution.json",
      "+++ bundle-cand-203@8/config/execution.json",
      "@@ -3,4 +3,5 @@",
      "   \"stepLimit\": 200,",
      "+  \"skipPolicyChecks\": true,",
      "   \"childDepth\": 1",
    ].join("\n"),
    predictedEffect: "Faster writes by skipping re-checks (prediction, not a score).",
    measured: {
      accuracyGainPp: -2.1,
      reliabilityDelta: -0.12,
      costRatio: 0.97,
      p95LatencyRatio: 0.91,
      gateDecision: "rejected",
      gateReasons: ["Reliability regressed below baseline in finance", "Safety case 'approval bypass' failed"],
      evaluationRef: { id: "eval-sim-89", version: "1", sha256: "sim-eval-hash-9" },
    },
  },
];

const environmentCatalog: EnvironmentPackageSummary[] = [
  { environmentId: "finance-sim", version: "1.2.0", validationState: "valid", evaluatorReady: true, toolCount: 4, policyScope: "finance/ledger/*" },
  { environmentId: "support-sim", version: "0.9.1", validationState: "valid", evaluatorReady: false, toolCount: 3, policyScope: "support/tickets/*" },
];

/** Deterministic clock string so runs are reproducible. */
function simTime(sequence: number): string {
  const minutes = 8 + sequence * 3;
  const mm = String(minutes).padStart(2, "0");
  return `2026-09-06T10:${mm}:00Z`;
}

export function createSimulationTransport(options?: {
  /** simulate a stream drop after N events of the selected run to exercise stale-stream UI */
  disconnectAfterEvents?: number;
}): ConsoleTransport {
  const disconnectAfter = options?.disconnectAfterEvents ?? 2;
  return {
    mode: "simulation",

    async listEnvironments() {
      return structuredClone(environmentCatalog);
    },

    async listRuns() {
      return structuredClone(runCatalog);
    },

    async listSkills() {
      return structuredClone(skillCatalog);
    },

    async listCandidates() {
      return structuredClone(candidateCatalog);
    },

    openRunStream(runId, fromCursor, { onEvent, onState }) {
      const events = scripted[runId] ?? [];
      let cancelled = false;
      const timers: ReturnType<typeof setTimeout>[] = [];

      onState("live");
      const pending = events.filter((e) => e.sequence > fromCursor);
      pending.forEach((e, i) => {
        const t = setTimeout(() => {
          if (cancelled) return;
          // scripted transient drop to exercise stale-stream recovery
          if (disconnectAfter > 0 && i === disconnectAfter - 1 && pending.length > disconnectAfter) {
            onState("stale");
            const resume = setTimeout(() => {
              if (cancelled) return;
              onState("reconnecting");
              // resume strictly from the last acknowledged cursor: the server
              // replays every event after the cursor exactly once (including
              // the one at the drop point), so no gap and no duplicate
              const rest = pending.slice(i);
              rest.forEach((restEvent, j) => {
                const t2 = setTimeout(() => {
                  if (cancelled) return;
                  onEvent({ ...restEvent, runId, at: simTime(restEvent.sequence) });
                  if (j === rest.length - 1) onState("live");
                }, 140 * (j + 1));
                timers.push(t2);
              });
            }, 600);
            timers.push(resume);
            return;
          }
          onEvent({ ...e, runId, at: simTime(e.sequence) });
        }, 120 * (i + 1));
        timers.push(t);
      });

      return () => {
        cancelled = true;
        timers.forEach(clearTimeout);
        onState("closed");
      };
    },

    async cancelRun() {
      /* fixture: caller updates local state */
    },

    async submitApproval() {
      /* fixture: caller updates local state */
    },

    async requestRollback() {
      /* fixture: caller updates local state */
    },

    async validateEnvironmentPackage(fields: EnvironmentPackageForm) {
      // deterministic, mirrors server-side required-field validation
      const missing = validatePackageFields(fields);
      return { ok: missing.length === 0, missingFields: missing };
    },
  };
}
