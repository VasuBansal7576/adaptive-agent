import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { createSimulationTransport } from "../api/simulation";
import type { ConsoleTransport } from "../api/transport";
import { normalizeSseEvent, SchemaError } from "../api/validate";

beforeEach(() => {
  vi.stubGlobal("EventSource", undefined);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("durable SSE event normalization", () => {
  it("accepts the plane projection unchanged", () => {
    const event = normalizeSseEvent(
      { runId: "r1", sequence: 3, at: "2026-09-06T10:00:00Z", kind: "status", summary: "Run queued." },
      "sse",
    );
    expect(event.sequence).toBe(3);
    expect(event.kind).toBe("status");
    expect(event.summary).toBe("Run queued.");
  });

  it("maps the durable evidence envelope {id, event, data} into RunEvent", () => {
    const event = normalizeSseEvent(
      {
        id: 7,
        event: "run_started",
        data: {
          runId: "r1",
          sequence: 7,
          eventType: "run_started",
          contentHash: "a".repeat(64),
          sourceRef: { id: "art-1", version: "1", sha256: "b".repeat(64) },
          trustClass: "system",
          visibility: "operator",
        },
      },
      "sse",
    );
    expect(event.runId).toBe("r1");
    expect(event.sequence).toBe(7);
    expect(event.kind).toBe("status");
    expect(event.summary).toBe("Run started");
    expect(event.detail).toContain("art-1");
  });

  it("falls back to the envelope id when the row lacks a sequence and defaults unknown types to step", () => {
    const event = normalizeSseEvent(
      { id: 4, event: "custom_probe", data: { runId: "r1", eventType: "custom_probe" } },
      "sse",
    );
    expect(event.sequence).toBe(4);
    expect(event.kind).toBe("step");
  });

  it("maps the exact durable snake_case wire row observed against the live plane", () => {
    const event = normalizeSseEvent(
      {
        id: 1,
        event: "run_created",
        data: {
          evidence_id: "ev_34a1",
          run_id: "run_9811",
          sequence: 1,
          event_type: "run_created",
          content_hash: "9baec1f5358fbce8fa7d8a27fc4be5cb89d28857c67daf0b7ed83cdf8fe610b5",
          source_ref: '{"id":"art_9baec1f5358fbce8","version":"1","sha256":"9baec1f5358fbce8"}',
          trust_class: "system",
          visibility: "learner",
          redacted: 1,
        },
      },
      "sse",
    );
    expect(event.runId).toBe("run_9811");
    expect(event.sequence).toBe(1);
    expect(event.kind).toBe("status");
    expect(event.summary).toBe("Run created");
    expect(event.detail).toContain("art_9baec1f5358fbce8");
  });

  it("renders 333ae4d payload projections: run_failed error detail and model usage", () => {
    const failed = normalizeSseEvent(
      {
        id: 12,
        event: "run_failed",
        data: {
          run_id: "r1",
          sequence: 12,
          event_type: "run_failed",
          content_hash: "h12",
          source_ref: '{"id":"art_12","version":"1","sha256":"h12"}',
          summary: "Runtime failure recorded",
          detail: "Prime CLI exited status 130: Daemon worker client closed",
          trust_class: "system",
          visibility: "operator",
        },
      },
      "sse",
    );
    expect(failed.summary).toBe("Runtime failure recorded");
    expect(failed.detail).toBe("Prime CLI exited status 130: Daemon worker client closed");
    expect(failed.runStatus).toBe("failed");

    const model = normalizeSseEvent(
      {
        id: 5,
        event: "model_response",
        data: {
          run_id: "r1",
          sequence: 5,
          event_type: "model_response",
          content_hash: "h5",
          source_ref: '{"id":"art_5","version":"1","sha256":"h5"}',
          summary: "Model response recorded",
          detail: '{"model":"openai-codex/gpt-5.6-luna","provider":"openai-codex","usage":{"inputTokens":120,"outputTokens":48,"totalTokens":168}}',
          trust_class: "system",
          visibility: "operator",
        },
      },
      "sse",
    );
    expect(model.kind).toBe("evidence");
    expect(model.summary).toBe("Model response recorded");
    expect(model.detail).toContain("totalTokens");
  });

  it("raises SchemaError on garbage instead of passing it through", () => {
    expect(() => normalizeSseEvent({ id: 1 }, "sse")).toThrow(SchemaError);
  });
});

describe("run-options and registered tasks in the new-run dialog", () => {
  it("applies /run-options budget defaults and Luna profile from the API", async () => {
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    await waitFor(() => expect(within(dialog).getByLabelText("Model")).toHaveValue("Luna"));
    await waitFor(() => expect(within(dialog).getByLabelText("Tool-call limit")).toHaveValue(32));
    await waitFor(() => expect(within(dialog).getByLabelText("Time limit (s)")).toHaveValue(90));
    await waitFor(() => expect(within(dialog).getByLabelText("Token budget")).toHaveValue(20000));
  });

  it("renders the ACTUAL trusted report projection: validation B0/L arms, gate, IDs, billing; decided/final B0/L/A with unknown billing", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const sent: Array<{ budgetRef?: { id: string } }> = [];
    const transport = {
      ...sim,
      createRun: async (input: Parameters<typeof sim.createRun>[0]) => {
        sent.push({ budgetRef: input.budgetRef ? { id: input.budgetRef.id } : undefined });
        return sim.createRun(input);
      },
      getRunOptions: async () => {
        const options = await sim.getRunOptions();
        return { ...options, budgetDefaults: { ...options.budgetDefaults }, budgetRef: { id: "budget-default", version: "1", sha256: "top-level" } };
      },
      listEvaluations: async () => {
        const evaluations = await sim.listEvaluations();
        return evaluations;
      },
      listCandidates: async () => {
        const candidates = await sim.listCandidates();
        return candidates;
      },
    };
    render(<App transport={transport as never} />);
    await user.click(await screen.findByRole("tab", { name: "Candidates" }));
    // validation report: B0/L arms with actual metrics and CI
    // validation arm table rendered (B0/L rows with actual metrics)
    expect(await screen.findAllByText("B0").then((els) => els.length)).toBeGreaterThanOrEqual(1);
    expect(await screen.findByText("65.0%")).toBeInTheDocument(); // L accuracy from the report
    expect(await screen.findByText(/accuracy_gain: 0\.100 \[0\.060, 0\.140]/)).toBeInTheDocument();
    expect(await screen.findByText(/validity valid · promotionEligible YES/)).toBeInTheDocument();
    // decided/final report renders B0/L/A and reports the explicit billing unknown
    expect(await screen.findByText("A")).toBeInTheDocument(); // final arm A rendered
    expect(await screen.findByText(/UNKNOWN — no nominal cost reported/)).toBeInTheDocument();
    expect(await screen.findByText(/missing pairs 2/)).toBeInTheDocument();
    expect(await screen.findByText(/INCOMPLETE/)).toBeInTheDocument();
    // legacy row stays unverified and out of the measured area
    expect(await screen.findByText(/legacy evaluation record\(s\) preserved as UNVERIFIED/)).toBeInTheDocument();
  });

  it("consumes executionModes advertised on environment summaries (51d476c)", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const envs = await sim.listEnvironments();
    envs[0].executionModes = ["dry_run"];
    const transport: ConsoleTransport = { ...sim, listEnvironments: async () => envs };
    render(<App transport={transport} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    await waitFor(() => expect(within(dialog).getByLabelText("Mode")).toHaveValue("dry_run"));
    expect(within(within(dialog).getByLabelText("Mode")).getAllByRole("option")).toHaveLength(1);
  });

  it("parses the REAL durable candidate projection (hashes + edit operations)", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const realShape = [
      {
        candidateId: "cand_405ac49df3c845f8be97490490a41350",
        baseBundleHash: "2647d69d89ff03689c5427b675472699ec144d842a58a726ca5d8b59074f74cc",
        candidateBundleHash: "60d7e19903203cdc430847fc2fae6224539c038bd78a4894db4f6de583f63ced",
        editOperations: [
          '{"operation":"add","path":"skills/reconcile-invoice-payment/procedure","value":"For an invoice-payment reconciliation task, read the named invoice and payment records first."}',
        ],
        changedArtifactHashes: ["5a92230eab60bf54c2a832424ce9ac7c3e8fc6c53fe52e6153c0cf9dc0ec1f29"],
        supportingEvidenceIds: ["broker:ev_2efcf195b4ab49519a4970c604d043bf"],
        predictedEffect: "Improve reconciliation by reading both records first.",
        proposerVersion: "1",
        state: "evaluating",
      },
    ];
    const transport: ConsoleTransport = { ...sim, listCandidates: async () => realShape as never };
    render(<App transport={transport} />);
    await user.click(await screen.findByRole("tab", { name: "Candidates" }));
    expect(await screen.findByText("cand_405ac49df3c845f8be97490490a41350")).toBeInTheDocument();
    expect(await screen.findByText("Evaluating")).toBeInTheDocument();
    expect(await screen.findByText(/2647d69d89ff0368/)).toBeInTheDocument();
    expect(await screen.findByText(/broker:ev_2efcf195/)).toBeInTheDocument();
    expect(await screen.findByText(/read the named invoice and payment records first/)).toBeInTheDocument();
  });

  it("accepts the top-level budgetRef from /run-options (01f2462 shape)", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const sent: Array<{ budgetRef?: { id: string } }> = [];
    const transport = {
      ...sim,
      getRunOptions: async () => {
        const options = await sim.getRunOptions();
        return { ...options, budgetDefaults: { ...options.budgetDefaults }, budgetRef: { id: "budget-default", version: "1", sha256: "top-level" } };
      },
      createRun: async (input: Parameters<typeof sim.createRun>[0]) => {
        sent.push({ budgetRef: input.budgetRef ? { id: input.budgetRef.id } : undefined });
        return sim.createRun(input);
      },
    };
    render(<App transport={transport} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    await waitFor(() => expect(within(dialog).getByLabelText("Model")).toHaveValue("Luna"));
    fireEvent.change(within(dialog).getByLabelText("Goal"), { target: { value: "top-level ref probe" } });
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));
    await waitFor(() => expect(sent.length).toBe(1));
    expect(sent[0].budgetRef?.id).toBe("budget-default");
  });

  it("sources the goal from a registered task and records its id", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const created: Array<{ taskId?: string; goal: string }> = [];
    const transport = {
      ...sim,
      createRun: async (input: Parameters<typeof sim.createRun>[0]) => {
        created.push({ taskId: input.taskId, goal: input.goal });
        return sim.createRun(input);
      },
    };
    render(<App transport={transport} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    const taskSelect = await within(dialog).findByLabelText("Registered task (required)");
    await user.selectOptions(taskSelect, "task-finance-sim-007");
    await waitFor(() =>
      expect(within(dialog).getByLabelText("Goal")).toHaveValue("Reconcile the Q3 ledger batch and record the outcome."),
    );
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));
    await waitFor(() => expect(created.length).toBe(1));
    expect(created[0].taskId).toBe("task-finance-sim-007");
  });
});
