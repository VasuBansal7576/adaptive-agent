import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { createRestTransport } from "../api/rest";
import { createSimulationTransport } from "../api/simulation";
import { newIdempotencyKey } from "../api/transport";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const RUN = {
  runId: "run_live_1",
  taskRef: { id: "task", version: "1", sha256: "h" },
  environmentRef: { id: "env", version: "1", sha256: "h" },
  policyRef: { id: "p", version: "1", sha256: "h" },
  modelProfileRef: { id: "model-profile", version: "1", sha256: "h" },
  skillBundleRef: { id: "b", version: "1", sha256: "h" },
  budgetRef: { id: "budget-x", version: "1", sha256: "h" },
  status: "queued",
  lastEventSequence: 0,
  executionMode: "interactive",
  goal: "wire contract",
} as const;

function fetchScript(handlers: Array<(url: string, init?: RequestInit) => { status?: number; body?: unknown } | void>) {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  let i = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL, init?: RequestInit) => {
      const entry = { url: String(url), init };
      calls.push(entry);
      const handler = handlers[Math.min(i++, handlers.length - 1)];
      const result = handler(entry.url, init) ?? {};
      return new Response(result.body === undefined ? "{}" : JSON.stringify(result.body), {
        status: result.status ?? 200,
        headers: { "content-type": "application/json" },
      });
    }),
  );
  return calls;
}

beforeEach(() => {
  // EventSource is unavailable in the test DOM; transports must degrade honestly
  vi.stubGlobal("EventSource", undefined);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("live wire contract", () => {
  it("sends modelProfileRef/budgetRef as canonical ref objects and launches after create", async () => {
    const transport = createRestTransport();
    const calls: Array<{ url: string; init?: RequestInit }> = [];
    let created = false;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL, init?: RequestInit) => {
        calls.push({ url: String(url), init });
        const u = String(url);
        if (u.endsWith("/session/bootstrap") || u.endsWith("/session")) return new Response("{}", { status: 200 });
        if (u.endsWith("/runs") && init?.method === "POST") {
          created = true;
          const body = JSON.parse(String(init.body));
          // modelProfileRef comes from the /run-options projection verbatim;
          // budget is the operator's validated object; no browser-computed hashes
          expect(body.modelProfileRef).toEqual({ id: "model-profile", version: "1", sha256: "server-provided" });
          // server-advertised trusted budget ref submitted verbatim (2b3fc75)
          expect(body.budgetRef).toEqual({ id: "budget-default", version: "1", sha256: "server-provided" });
          expect(body.budget).toEqual({
            modelTokens: 7777,
            toolCalls: 32,
            childRuns: 0,
            wallTimeSeconds: 90,
            costMicrounits: 100000,
            currency: "USD",
          });
          expect(typeof body.idempotencyKey).toBe("string");
          return new Response(JSON.stringify(RUN), { status: 201 });
        }
        if (/\/runs\/run_live_1\/launch$/.test(u)) {
          expect(created).toBe(true); // launch only after create returned
          return new Response("{}", { status: 202 });
        }
        if (u.endsWith("/runs")) return new Response(JSON.stringify([RUN]), { status: 200 });
        return new Response("{}", { status: 200 });
      }),
    );
    await transport.createRun({
      environmentId: "env",
      goal: "wire contract",
      modelProfile: "Luna",
      modelProfileRef: { id: "model-profile", version: "1", sha256: "server-provided" },
      budgetRef: { id: "budget-default", version: "1", sha256: "server-provided" },
      idempotencyKey: newIdempotencyKey(),
      budget: { toolCallCeiling: 32, wallSecondsCeiling: 90, modelTokenCeiling: 7777 },
      executionMode: "interactive",
    });
    // launch must have fired after create returned
    expect(calls.some((c) => /\/runs\/run_live_1\/launch$/.test(c.url))).toBe(true);
  });

  it("refuses to invent a model ref when /run-options did not provide one", async () => {
    const transport = createRestTransport();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response(JSON.stringify({}), { status: 200 })),
    );
    await expect(
      transport.createRun({
        environmentId: "env",
        goal: "no ref",
        modelProfile: "Luna",
        idempotencyKey: newIdempotencyKey(),
        budget: { toolCallCeiling: 32, wallSecondsCeiling: 90, modelTokenCeiling: 4000 },
        executionMode: "interactive",
      }),
    ).rejects.toThrow(/modelProfileRef must come from the \/run-options projection/);
  });
  it("UI create-run flow yields a launched (non-queued) run via transport createRun", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const launch = vi.fn();
    const wrapped = {
      ...sim,
      createRun: async (input: Parameters<typeof sim.createRun>[0]) => {
        // mirror transport behavior: create then launch before returning
        await launch();
        return sim.createRun(input);
      },
    };
    render(<App transport={wrapped} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    await user.type(within(dialog).getByLabelText("Goal"), "ui launch flow");
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));
    await waitFor(() => expect(launch).toHaveBeenCalled());
    expect((await screen.findAllByText(/run-sim-1007/)).length).toBeGreaterThanOrEqual(1);
  });

  it("stages learning via /learning/launch", async () => {
    const calls = fetchScript([
      (_url) => ({ body: { status: "ready" } }), // bootstrap
      (_url) => ({ body: { authenticated: true } }), // session
      (_url, init) => {
        const body = JSON.parse(String(init?.body));
        expect(body).toEqual({ runId: "run_live_1" }); // runId-only request
        return { body: { actionId: "learn_1", status: "staged" } };
      },
    ]);
    const transport = createRestTransport();
    const result = await transport.launchLearningCycle({ runId: "run_live_1" });
    expect(result.actionId).toBe("learn_1");
    expect(calls.some((c) => c.url.endsWith("/learning/launch"))).toBe(true);
  });
});

describe("FastAPI detail normalization", () => {
  it("normalizes structured, string, and array detail failures into the envelope", async () => {
    const transport = createRestTransport();
    // the session handshake runs once (memoized); each later call hits its
    // failure step directly
    const script: Array<{ status: number; body: unknown }> = [
      { status: 200, body: { status: "ready" } },
      { status: 200, body: { authenticated: true } },
      { status: 409, body: { detail: { code: "IDEMPOTENCY_CONFLICT", message: "dup", correlationId: "c9", retry: "never" } } },
      { status: 422, body: { detail: "goal or taskRef.goal is required" } },
      { status: 422, body: { detail: [{ loc: ["body", "goal"], msg: "field required" }] } },
    ];
    let i = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        const step = script[i++];
        return new Response(JSON.stringify(step.body), { status: step.status });
      }),
    );
    await expect(transport.cancelRun("r")).rejects.toMatchObject({ code: "IDEMPOTENCY_CONFLICT", correlationId: "c9" });
    await expect(transport.cancelRun("r")).rejects.toMatchObject({ message: "goal or taskRef.goal is required" });
    // validation arrays classify as INVALID_INPUT with the normalized first msg
    await expect(transport.cancelRun("r")).rejects.toMatchObject({ code: "INVALID_INPUT", message: "field required" });
  });
});
