import type {
  CandidateDiff,
  EnvironmentPackageSummary,
  RunEvent,
  RunOptions,
  RunRecord,
  SkillVersionSummary,
  TaskOption,
} from "./types";
import type {
  ConsoleTransport,
  CreateRunInput,
  EnvironmentPackageForm,
  EnvironmentRegistration,
  LearningCycleInput,
} from "./transport";
import { validatePackageFields, sha256Hex } from "./transport";
import { parseRun } from "./validate";

/**
 * DETERMINISTIC SIMULATION TRANSPORT — development fixture only.
 *
 * This module never contacts a backend and never represents live inference.
 * The UI renders a persistent "SIMULATED" banner whenever this transport is
 * active (see App.tsx), and the option exists only as an explicit dev-only
 * opt-in. All sequences are fixed and reproducible.
 */

export const SIMULATION_LABEL = "SIMULATED — development fixture, not live inference";

type ScriptedEvent = Omit<RunEvent, "runId" | "at">;

function hash(pad: number): string {
  return `sim-hash-${String(pad).padStart(40, "0")}`;
}

const baseRun: RunRecord = {
  runId: "run-sim-1001",
  taskRef: { id: "task-finance-007", version: "1", sha256: hash(1) },
  environmentRef: { id: "env-finance", version: "1.2.0", sha256: hash(2) },
  policyRef: { id: "policy-finance", version: "3", sha256: hash(3) },
  modelProfileRef: { id: "profile-subscription", version: "1", sha256: hash(4) },
  skillBundleRef: { id: "bundle-active", version: "7", sha256: hash(5) },
  budgetRef: { id: "budget-default", version: "1", sha256: hash(6) },
  status: "running",
  lastEventSequence: 0,
  environmentId: "finance-sim",
  goal: "Reconcile the Q3 ledger batch and record the outcome.",
  budgetUsed: { calls: 5, callsCeiling: 100, wallSeconds: 96, wallCeiling: 900 },
};

let runCatalog: RunRecord[] = [
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
    outcomeRef: { id: "outcome-1003", version: "1", sha256: hash(7) },
  },
  {
    ...baseRun,
    runId: "run-sim-1004",
    status: "succeeded",
    lastEventSequence: 8,
    outcomeRef: { id: "outcome-1004", version: "1", sha256: hash(8) },
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

/** Events for runs created through createRun in this fixture session. */
let learningActionCount = 0;

function scriptForNewRun(runId: string, input: CreateRunInput): ScriptedEvent[] {
  return [
    { sequence: 1, kind: "status", summary: `Run queued and pinned to the current active bundle (${input.executionMode} mode).` },
    { sequence: 2, kind: "step", summary: "Retrieved 2 documentation chunks for the goal." },
    { sequence: 3, kind: "tool", summary: "inventory.read completed (read effect confirmed)." },
    { sequence: 4, kind: "budget", summary: "Budget checkpoint recorded against the reserved ledger." },
  ];
}

let skillCatalog: SkillVersionSummary[] = [
  { skillId: "recheck-before-update", version: "3", parentVersion: "2", state: "active", applicability: "tools with optimistic versioning", evidenceRefs: ["ev-sim-31", "ev-sim-32"], contentHash: hash(11) },
  { skillId: "batch-reconcile", version: "5", parentVersion: "4", state: "proposed", applicability: "multi-entry ledger batches", evidenceRefs: ["ev-sim-44"], contentHash: hash(12) },
  { skillId: "unsafe-bulk-write", version: "1", parentVersion: null, state: "quarantined", applicability: "—", evidenceRefs: ["ev-sim-51"], contentHash: hash(13) },
  { skillId: "stale-cache-read", version: "2", parentVersion: "1", state: "rejected", applicability: "—", evidenceRefs: ["ev-sim-61"], contentHash: hash(14) },
  { skillId: "batch-reconcile", version: "4", parentVersion: "3", state: "rolled_back", applicability: "multi-entry ledger batches", evidenceRefs: ["ev-sim-40"], contentHash: hash(15) },
];

let candidateCatalog: CandidateDiff[] = [
  {
    candidateId: "cand-sim-204",
    baseBundleHash: "a".repeat(64),
    candidateBundleHash: "b".repeat(64),
    editOperations: [
      '{"operation":"modify","path":"skills/batch-reconcile/procedure","value":"Verify batch totals before appending; skip already-applied entries."}',
    ],
    changedArtifactHashes: ["c".repeat(64)],
    supportingEvidenceIds: ["broker:ev_sim_77"],
    proposerVersion: "1",
    state: "validated" as const,
    predictedEffect: "Fewer duplicate ledger entries per batch (prediction, not a score).",
  },
  {
    candidateId: "cand-sim-202",
    baseBundleRef: { id: "bundle-active", version: "7", sha256: hash(21) },
    candidateBundleRef: { id: "bundle-cand-202", version: "8", sha256: hash(22) },
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
      evaluationRef: { id: "eval-sim-88", version: "1", sha256: hash(23) },
    },
    audit: [
      {
        rollbackId: "rb-sim-01",
        fromRef: { id: "bundle-cand-101", version: "7", sha256: hash(24) },
        toRef: { id: "bundle-active", version: "6", sha256: hash(25) },
        reason: "Safety suite flagged unbounded child fan-out.",
        at: "2026-09-06T12:40:00Z",
        affectedRuns: ["run-sim-1003", "run-sim-1005"],
      },
    ],
  },
  {
    candidateId: "cand-sim-203",
    baseBundleRef: { id: "bundle-active", version: "7", sha256: hash(26) },
    candidateBundleRef: { id: "bundle-cand-203", version: "8", sha256: hash(27) },
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
      evaluationRef: { id: "eval-sim-89", version: "1", sha256: hash(28) },
    },
  },
  {
    candidateId: "cand-sim-205",
    baseBundleHash: "d".repeat(64),
    candidateBundleHash: "e".repeat(64),
    editOperations: [
      '{"operation":"modify","path":"skills/recheck-before-update/procedure","value":"Final-sealed candidate procedure."}',
    ],
    changedArtifactHashes: ["f".repeat(64)],
    supportingEvidenceIds: ["broker:ev_sim_99"],
    proposerVersion: "1",
    state: "rejected" as const,
    predictedEffect: "Sealed-final candidate (prediction, not a score).",
  },
];

let environmentCatalog: EnvironmentPackageSummary[] = [
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

    async getRunOptions(): Promise<RunOptions> {
      // mirrors the live /run-options projection
      return {
        modelProfiles: [
          {
            ref: { id: "model-profile", version: "1", sha256: await sha256Hex(JSON.stringify("model-profile")) },
            label: "Luna",
            provider: "openai-codex",
            model: "openai-codex/gpt-5.6-luna",
          },
        ],
        budgetDefaults: {
          modelTokens: 20000,
          toolCalls: 32,
          childRuns: 0,
          wallTimeSeconds: 90,
          costMicrounits: 100000,
          currency: "USD",
        },
        budgetRef: { id: "budget-default", version: "1", sha256: await sha256Hex(JSON.stringify("budget-default")) },
      };
    },

    async getEnvironmentTasks(environmentId: string): Promise<TaskOption[]> {
      // deterministic registered tasks per environment
      const tasks: TaskOption[] = [
        { taskId: `task-${environmentId}-007`, goal: "Reconcile the Q3 ledger batch and record the outcome.", executionModes: ["interactive", "dry_run"] },
        { taskId: `task-${environmentId}-011`, goal: "Replay the dispute workflow and summarize the outcome.", executionModes: ["interactive", "replay"] },
      ];
      return structuredClone(tasks);
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

    async createRun(input: CreateRunInput) {
      const runId = `run-sim-${1007 + runCatalog.length - 6}`;
      const run: RunRecord = parseRun(
        {
          runId,
          taskRef: { id: input.taskId ?? `task-${input.environmentId}`, version: "1", sha256: hash(31 + runCatalog.length) },
          environmentRef: { id: `env-${input.environmentId}`, version: "1", sha256: hash(41) },
          policyRef: { id: `policy-${input.environmentId}`, version: "1", sha256: hash(42) },
          modelProfileRef: input.modelProfileRef ?? { id: input.modelProfile, version: "1", sha256: hash(43) },
          skillBundleRef: { id: "bundle-active", version: "7", sha256: hash(44) },
          budgetRef: { id: `budget-${input.idempotencyKey.slice(0, 8)}`, version: "1", sha256: hash(45) },
          status: "queued",
          lastEventSequence: 0,
          executionMode: input.executionMode,
          environmentId: input.environmentId,
          goal: input.goal,
          budgetUsed: { calls: 0, callsCeiling: input.budget.toolCallCeiling, wallSeconds: 0, wallCeiling: input.budget.wallSecondsCeiling },
        },
        "createdRun",
      );
      runCatalog = [...runCatalog, run];
      scripted[runId] = scriptForNewRun(runId, input);
      return structuredClone(run);
    },

    async launchRun(runId: string) {
      scripted[runId] = [
        ...(scripted[runId] ?? []),
        { sequence: (scripted[runId]?.length ?? 0) + 1, kind: "status" as const, summary: `Run started in ${runCatalog.find((r) => r.runId === runId)?.executionMode ?? "interactive"} mode with authenticated model runner.` },
      ];
    },

    async registerEnvironment(manifest: EnvironmentRegistration) {
      const summary: EnvironmentPackageSummary = {
        environmentId: manifest.environmentId,
        version: manifest.version,
        validationState: "valid",
        evaluatorReady: true,
        toolCount: manifest.toolSchemas.length,
        policyScope: manifest.policyRef.id,
      };
      environmentCatalog = [...environmentCatalog, summary];
      return structuredClone(summary);
    },

    async launchEvaluation(input: { candidateId: string; baseBundleHash: string }) {
      const evaluationId = `eval-sim-${(learningActionCount += 1).toString().padStart(3, "0")}`;
      return { evaluationId, state: "queued" as const };
    },

    async listEvaluations() {
      // raw rows in the API projection shape; the console parser classifies
      // them (verified reports vs unverified legacy). Representative validation
      // and final report shapes from the durable projection (96a84d3/8fcb18a).
      const armSummary = (accuracy: number) => ({
        accuracy,
        reliability: accuracy + 0.02,
        meanCostMicrounits: 41250,
        medianLatencySeconds: 41.2,
        p95LatencySeconds: 88.4,
        safetyViolations: 0,
        count: 60,
      });
      return [
        {
          evaluationId: "eval-sim-001",
          candidateId: "cand-sim-204",
          baseBundleHash: "2647d69d89ff03689c5427b675472699ec144d842a58a726ca5d8b59074f74cc",
          state: "valid",
          trusted: true,
          report: {
            comparison: "validation",
            validityStatus: "valid",
            promotionEligible: true,
            candidateHash: "60d7e19903203cdc430847fc2fae6224539c038bd78a4894db4f6de583f63ced",
            baseHash: "2647d69d89ff03689c5427b675472699ec144d842a58a726ca5d8b59074f74cc",
            protocolHash: "p".repeat(64),
            armSummaries: { B0: armSummary(0.55), L: armSummary(0.65) },
            confidenceIntervals: [
              { metric: "accuracy_gain", point: 0.1, lower95: 0.06, upper95: 0.14, draws: 10000, analysisSeed: 20260906 },
            ],
            safetyPassed: true,
            missingPairs: 0,
            metricCellsComplete: true,
            safetyCellsComplete: true,
            modelProvenanceComplete: true,
            infrastructureFailures: [],
            analysisSeed: 20260906,
            nominalCostUsd: 1.25,
            actualInputTokens: 48120,
            actualOutputTokens: 6240,
            wallDurationSeconds: 841,
            billingBasis: "nominal-usd-pinned",
          },
        },
        {
          // final B0/L/A report shape (completion/decided states recognized)
          evaluationId: "eval-sim-002",
          candidateId: "cand-sim-205",
          state: "decided",
          trusted: true,
          report: {
            comparison: "final",
            validityStatus: "valid",
            promotionEligible: false,
            armSummaries: { B0: armSummary(0.58), L: armSummary(0.61), A: armSummary(0.56) },
            confidenceIntervals: [
              { metric: "accuracy_gain", point: 0.03, lower95: -0.01, upper95: 0.07, draws: 10000, analysisSeed: 20260906 },
            ],
            safetyPassed: true,
            missingPairs: 2,
            metricCellsComplete: false,
            infrastructureFailures: ["fixture cell timed out"],
            billingBasis: "unknown-unreported",
          },
        },
        // legacy malformed metadata: preserved explicitly unverified
        { evaluationId: "eval-sim-legacy", state: "completed", trusted: true, validity: "valid" },
      ] as never;
    },

    async reconnect() {
      /* fixture transport never disconnects */
    },

    async launchLearningCycle(input: LearningCycleInput) {
      // fixture: the durable pipeline generates proposal + evidence from the
      // completed development run
      const actionId = `learn-sim-${(learningActionCount += 1).toString().padStart(3, "0")}`;
      return { actionId, runId: input.runId, status: "staged" as const };
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

    async cancelRun(runId: string) {
      runCatalog = runCatalog.map((r) => (r.runId === runId && (r.status === "queued" || r.status === "running" || r.status === "awaiting_approval") ? { ...r, status: "cancelled" as const } : r));
      scripted[runId] = [
        ...(scripted[runId] ?? []),
        { sequence: (scripted[runId]?.length ?? 0) + 1, kind: "status" as const, summary: "Cancelled by operator; future calls revoked. Dispatched external actions were not undone." },
      ];
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
