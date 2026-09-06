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
