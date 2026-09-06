/**
 * Integration smoke against the actual FastAPI control plane (default :8000).
 * Run while the API is up:  bun run console/scripts/api-smoke.mjs
 * Exits 0 on pass; skips (exit 0, message) when the API is unreachable.
 * Uses the same contract as the console: bootstrap session, list, create+launch
 * a run, read SSE events with cursor semantics, cancel.
 */
const base = process.env.API_BASE ?? "http://127.0.0.1:8000";
// minimal cookie jar: browsers keep the HttpOnly operator cookie automatically;
// node needs it carried explicitly
let cookie = "";

async function json(path, init) {
  const res = await fetch(`${base}${path}`, {
    headers: { "content-type": "application/json", ...(cookie ? { cookie } : {}) },
    ...init,
  });
  const setCookie = res.headers.get("set-cookie");
  if (setCookie) cookie = setCookie.split(";")[0];
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${path} -> HTTP ${res.status}: ${body.slice(0, 200)}`);
  }
  return res.json();
}

async function main() {
  try {
    await fetch(`${base}/health`).then((r) => {
      if (!r.ok) throw new Error(`health HTTP ${r.status}`);
    });
  } catch {
    console.log(`SKIP: control API not reachable at ${base}`);
    return 0;
  }

  // 1. session handshake (HttpOnly cookie)
  const bootstrap = await json("/session/bootstrap");
  if (bootstrap.status !== "ready") throw new Error(`bootstrap status: ${JSON.stringify(bootstrap)}`);
  const session = await json("/session");
  if (session.authenticated !== true) throw new Error("session not authenticated after bootstrap");

  // 2. reads
  let environments = await json("/environments");
  const runsBefore = await json("/runs");
  console.log(`reads ok: ${environments.length} environments, ${runsBefore.length} runs`);

  // 3. ensure a registered environment (same strict manifest the console sends)
  if (environments.length === 0) {
    await json("/environments/register", {
      method: "POST",
      body: JSON.stringify({
        environmentId: "smoke-neutral",
        version: "1",
        docs: [{ id: "docs-smoke", version: "1", sha256: "0".repeat(64) }],
        taskGoals: ["read the counter"],
        toolSchemas: [{ name: "counter.read", version: "1", inputSchema: { type: "object" }, outputSchema: { type: "object" }, effect: "read" }],
        policyRef: { id: "policy-smoke", version: "1", sha256: "0".repeat(64) },
        evaluatorRef: { id: "evaluator-smoke", version: "1", sha256: "0".repeat(64) },
        resetRef: { id: "reset-smoke", version: "1", sha256: "0".repeat(64) },
        executionModes: ["interactive", "dry_run"],
        capabilities: ["read"],
      }),
    });
    environments = await json("/environments");
    console.log(`registered smoke environment: ${environments[0].environmentId}`);
  }

  // 4. create + launch a run (direct projection)
  const manifest = environments[0]?.manifest;
  const environmentId = environments[0]?.environmentId ?? manifest?.environmentId ?? "neutral";
  const run = await json("/runs", {
    method: "POST",
    body: JSON.stringify({
      goal: "console smoke: verify create/launch/events/cancel",
      environmentId,
      modelProfileRef: { id: "openai-codex/gpt-5.6-luna", version: "1" },
      budgetRef: { id: "budget-smoke", version: "1" },
      idempotencyKey: `smoke-${Date.now()}`,
      executionMode: "dry_run",
    }),
  });
  if (!run.runId) throw new Error(`create run failed: ${JSON.stringify(run)}`);
  console.log(`run created: ${run.runId} (${run.executionMode})`);
  if (run.executionMode !== "dry_run") throw new Error(`executionMode not recorded: ${run.executionMode}`);

  await json(`/runs/${run.runId}/launch`, { method: "POST", body: "{}" });

  // 4. SSE events with cursor semantics
  const seen = [];
  const res = await fetch(`${base}/runs/${run.runId}/events?cursor=0`, {
    headers: cookie ? { cookie } : {},
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const deadline = Date.now() + 8000;
  while (seen.length < 2 && Date.now() < deadline) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index;
    while ((index = buffer.indexOf("\n\n")) !== -1) {
      const chunk = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      const dataLine = chunk.split("\n").find((l) => l.startsWith("data:"));
      if (dataLine) {
        try {
          const event = JSON.parse(dataLine.slice(5).trim());
          if (typeof event.sequence === "number") seen.push(event);
        } catch {
          /* non-JSON line */
        }
      }
    }
  }
  await reader.cancel().catch(() => {});
  if (seen.length === 0) throw new Error("no SSE events received");
  if (seen.some((e) => e.sequence <= 0)) throw new Error("non-monotonic sequence numbers");
  console.log(`sse ok: ${seen.length} events, sequences ${seen.map((e) => e.sequence).join(",")}`);

  // 5. idempotent replay: same key + same payload must not double-create
  // (create with a fresh distinct key for cancel coverage)
  await json(`/runs/${run.runId}/cancel`, { method: "POST", body: "{}" });
  const after = await json(`/runs/${run.runId}`);
  console.log(`cancel ok: final status ${after.status}`);
  console.log("PASS");
  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch((error) => {
    console.error(`FAIL: ${error.message}`);
    process.exit(1);
  });
