import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { createSimulationTransport } from "../api/simulation";
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
    expect(event.summary).toContain("run_started");
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
    await waitFor(() => expect(within(dialog).getByLabelText("Token budget")).toHaveValue(4000));
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
    const taskSelect = await within(dialog).findByLabelText("Registered task");
    await user.selectOptions(taskSelect, "task-finance-sim-007");
    await waitFor(() =>
      expect(within(dialog).getByLabelText("Goal")).toHaveValue("Reconcile the Q3 ledger batch and record the outcome."),
    );
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));
    await waitFor(() => expect(created.length).toBe(1));
    expect(created[0].taskId).toBe("task-finance-sim-007");
  });
});
