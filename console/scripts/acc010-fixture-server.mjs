/**
 * ACC-010 CLASSIFIED UI-FIXTURE SERVER — deterministic, isolated, GET-only.
 *
 * This server is a labeled development fixture for browser acceptance
 * (viewport widths, keyboard navigation, loading/empty/error states). It
 * serves the BUILT console from console/dist and deterministic JSON fixtures
 * on the direct API routes. It never proxies to the real backend, never
 * touches any real Store, and performs no model/learning/evaluation calls.
 * Every response carries the X-UI-Fixture header for unambiguous
 * classification in network inspection.
 *
 * Usage: BASE_PORT=8910 node console/scripts/acc010-fixture-server.mjs
 */
import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { extname, join } from "node:path";

const PORT = Number(process.env.BASE_PORT ?? 8910);
const DIST = process.env.DIST ?? new URL("../dist/", import.meta.url).pathname;

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png" };

const arm = (accuracy) => ({
  accuracy, reliability: Math.min(1, accuracy + 0.03),
  meanCostMicrounits: 41250, medianLatencySeconds: 41.2, p95LatencySeconds: 88.4,
  safetyViolations: 0, count: 60,
});

const run = (runId, status, goal, learningEligible, extra = {}) => ({
  runId, status, lastEventSequence: extra.lastEventSequence ?? 4,
  taskRef: { id: `task-${runId.slice(4, 10)}`, version: "1", sha256: "f".repeat(64) },
  environmentRef: { id: "finance", version: "1.2.0", sha256: "e".repeat(64) },
  policyRef: { id: "policy-finance", version: "3", sha256: "d".repeat(64) },
  modelProfileRef: { id: "model-profile", version: "1", sha256: "c".repeat(64) },
  skillBundleRef: { id: "bundle-active", version: "7", sha256: "b".repeat(64) },
  budgetRef: { id: "budget-default", version: "1", sha256: "a".repeat(64) },
  environmentId: "finance", goal, learningEligible, executionMode: "interactive",
  budgetUsed: { calls: 8, callsCeiling: 32, wallSeconds: 120, wallCeiling: 900 },
  ...extra,
});

const RUNS = [
  run("run_acc010_succeeded_1", "succeeded", "Reconcile invoice INV-DEV-003 against payment PAY-DEV-003 and apply the matching payment.", true, { outcomeRef: { id: "o1", version: "1", sha256: "1".repeat(64) } }),
  run("run_acc010_failed_with_long_identifier_2", "failed", "Dispute the duplicated charge on statement period 2026-08 and record the customer-visible resolution outcome for the support transcript.", true),
  run("run_acc010_running_3", "running", "Sweep the reconciliation batch for stale entries.", false),
  run("run_acc010_running_deny_6", "running", "Export the settled batch; cancellation is refused after dispatch.", false),
  run("run_acc010_awaiting_approval_4", "awaiting_approval", "Apply the verified payment batch after operator approval.", false),
  run("run_acc010_timed_out_5", "timed_out", "Retry the idempotent export after the broker timeout window.", false),
];

const REPORT_VALIDATION = {
  comparison: "validation", validityStatus: "valid", promotionEligible: true,
  candidateHash: "60d7e19903203cdc430847fc2fae6224539c038bd78a4894db4f6de583f63ced",
  baseHash: "2647d69d89ff03689c5427b675472699ec144d842a58a726ca5d8b59074f74cc",
  protocolHash: "p".repeat(64),
  armSummaries: { B0: arm(0.55), L: arm(0.65) },
  confidenceIntervals: [{ metric: "accuracy_gain", point: 0.1, lower95: 0.06, upper95: 0.14, draws: 10000, analysisSeed: 20260906 }],
  safetyPassed: true, missingPairs: 0, metricCellsComplete: true, safetyCellsComplete: true,
  modelProvenanceComplete: true, infrastructureFailures: [], analysisSeed: 20260906,
  nominalCostUsd: 1.25, actualInputTokens: 48120, actualOutputTokens: 6240,
  wallDurationSeconds: 841, billingBasis: "nominal-usd-pinned",
};

const CANDIDATES = [
  { candidateId: "cand_acc010_validated", baseBundleHash: "2".repeat(64), candidateBundleHash: "3".repeat(64), editOperations: ['{"operation":"add","path":"skills/reconcile-invoice-payment/procedure","value":"For an invoice-payment reconciliation task, read the named invoice and payment records first. Verify the invoice is eligible for reconciliation, the payment is unapplied, and both records are current. Then call the finance operation that applies the payment to the invoice using the named record identifiers. Do not modify unrelated records, and handle read or write errors without retrying blindly."}'], changedArtifactHashes: ["4".repeat(64)], supportingEvidenceIds: ["broker:ev_acc010_1", "broker:ev_acc010_2"], predictedEffect: "Improve reconciliation by reading both records first, confirming the payment is unapplied and the invoice is current, then applying the matching payment.", proposerVersion: "1", state: "validated" },
  { candidateId: "cand_acc010_promoted", baseBundleRef: { id: "bundle-active", version: "7", sha256: "7".repeat(64) }, candidateBundleRef: { id: "bundle-cand", version: "8", sha256: "8".repeat(64) }, diff: "--- bundle-active@7\n+++ bundle-cand@8\n@@ -1,3 +1,4 @@\n read invoice and payment\n+verify eligibility before update\n apply payment", state: "promoted", predictedEffect: "Fewer stale updates (prediction, not a score).", measured: { accuracyGainPp: 7.4, reliabilityDelta: 0.06, costRatio: 1.02, p95LatencyRatio: 1.04, gateDecision: "promoted", gateReasons: ["Accuracy gain ≥ 5pp", "No environment regressed"], evaluationRef: { id: "eval-1", version: "1", sha256: "9".repeat(64) } } },
  { candidateId: "cand_acc010_evaluating", baseBundleHash: "5".repeat(64), candidateBundleHash: "6".repeat(64), editOperations: ['{"operation":"modify","path":"skills/batch-reconcile/procedure","value":"Verify batch totals before appending."}'], changedArtifactHashes: ["5".repeat(64)], supportingEvidenceIds: [], predictedEffect: "Fewer duplicate entries (prediction, not a score).", proposerVersion: "1", state: "evaluating" },
];

const EVALUATIONS = [
  { evaluationId: "eval_acc010_validation", candidateId: "cand_acc010_promoted", baseBundleHash: "7".repeat(64), state: "valid", trusted: true, report: REPORT_VALIDATION },
  // legacy malformed metadata: preserved explicitly unverified
  { evaluationId: "eval_acc010_legacy", state: "completed", trusted: true, validity: "valid" },
];

const FIXTURES = {
  "/session": { authenticated: true, transport: "fixture" },
  "/run-options": { modelProfiles: [{ ref: { id: "model-profile", version: "1", sha256: "c".repeat(64) }, label: "Luna", provider: "openai-codex", model: "openai-codex/gpt-5.6-luna" }], budgetDefaults: { modelTokens: 20000, toolCalls: 32, childRuns: 0, wallTimeSeconds: 90, costMicrounits: 100000, currency: "USD", budgetRef: { id: "budget-default", version: "1", sha256: "a".repeat(64) } }, budgetRef: { id: "budget-default", version: "1", sha256: "a".repeat(64) } },
  "/environments": [
    { environmentId: "finance", version: "1.2.0", validationState: "valid", evaluatorReady: true, toolCount: 4, policyScope: "finance/ledger/*", executionModes: ["interactive", "dry_run"], capabilities: ["read"] },
  ],
  "/skills": [],
  "/runs": RUNS,
  "/candidates": CANDIDATES,
  "/evaluations": EVALUATIONS,
};

const server = createServer((req, res) => {
  const url = new URL(req.url ?? "/", `http://127.0.0.1:${PORT}`);
  const path = url.pathname;
  res.setHeader("X-UI-Fixture", "acc010-classified-fixture");

  if (path === "/session/bootstrap") {
    res.setHeader("set-cookie", "acc010_fixture_session=fake; HttpOnly; SameSite=Strict; Path=/");
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ status: "ready", transport: "fixture" }));
    return;
  }
  if (FIXTURES[path] !== undefined) {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(FIXTURES[path]));
    return;
  }
  if (req.method === "POST" && /^\/runs\/[^/]+\/cancel$/.test(path)) {
    const runId = decodeURIComponent(path.split("/")[2]);
    if (runId.includes("deny")) {
      // classified failure fixture: broker refuses cancellation
      res.writeHead(409, { "content-type": "application/json" });
      res.end(JSON.stringify({ detail: { code: "FORBIDDEN", message: "cancellation window closed; effect already dispatched", correlationId: "corr-acc010-deny", retry: "never" } }));
      return;
    }
    const found = RUNS.find((r) => r.runId === runId);
    if (found && ["queued", "running", "awaiting_approval"].includes(found.status)) {
      found.status = "cancelled";
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ ok: true }));
      return;
    }
    res.writeHead(404, { "content-type": "application/json" });
    res.end(JSON.stringify({ detail: "run not found" }));
    return;
  }
  if (path.startsWith("/runs/") && path.endsWith("/events")) {
    // two visible frames then server EOF (terminal close); frames echo the
    // requested run id so the console applies them to the selected run
    const streamRunId = decodeURIComponent(path.slice("/runs/".length, -"/events".length));
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
    const frames = [
      { id: 1, event: "run_created", data: { run_id: streamRunId, sequence: 1, event_type: "run_created", content_hash: "h1", source_ref: '{"id":"a1","version":"1","sha256":"h1"}', trust_class: "system", visibility: "operator", redacted: 0 } },
      { id: 2, event: "run_started", data: { run_id: streamRunId, sequence: 2, event_type: "run_started", content_hash: "h2", source_ref: '{"id":"a2","version":"1","sha256":"h2"}', trust_class: "system", visibility: "operator", redacted: 0 } },
    ];
    for (const frame of frames) {
      res.write(`id: ${frame.id}\ndata: ${JSON.stringify(frame)}\n\n`);
    }
    res.end();
    return;
  }
  if (path.startsWith("/runs/")) {
    const runId = decodeURIComponent(path.slice("/runs/".length));
    const found = RUNS.find((r) => r.runId === runId);
    if (found) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(found));
      return;
    }
  }
  if (path === "/health") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ status: "ok" }));
    return;
  }
  if (path.startsWith("/api/")) {
    res.writeHead(404, { "content-type": "application/json" });
    res.end(JSON.stringify({ detail: "/api is a dev-only proxy prefix" }));
    return;
  }

  // static built console
  let file = path === "/" ? "index.html" : path.slice(1);
  const full = join(DIST, file);
  if (existsSync(full)) {
    res.writeHead(200, { "content-type": MIME[extname(full)] ?? "application/octet-stream" });
    res.end(readFileSync(full));
    return;
  }
  res.writeHead(404, { "content-type": "text/plain" });
  res.end("fixture 404");
});

server.listen(PORT, "127.0.0.1", () => console.log(`acc010 fixture server on http://127.0.0.1:${PORT} (CLASSIFIED UI FIXTURE)`));
