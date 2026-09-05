import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { createSimulationTransport } from "../api/simulation";
import { initialConsoleState, reducer, type ConsoleState } from "../state/consoleStore";
import { applyEvent } from "../state/consoleStore";

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network unavailable")));
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("ghost-data regression: mode switch clears prior-mode state", () => {
  it("reducer transport reset clears runs, events, candidates, and cursors", () => {
    let state: ConsoleState = { ...initialConsoleState, transportMode: "simulation", runs: [{ runId: "run-sim-1001" } as never], candidates: [{ candidateId: "cand-sim-202" } as never] };
    state = applyEvent(state, { runId: "run-sim-1001", sequence: 1, at: "t", kind: "status", summary: "s" });
    expect(state.events["run-sim-1001"]).toHaveLength(1);
    const switched = reducer(state, { type: "transport", mode: "live" });
    expect(switched.runs).toHaveLength(0);
    expect(switched.events).toEqual({});
    expect(switched.cursors).toEqual({});
    expect(switched.candidates).toHaveLength(0);
    expect(switched.transportMode).toBe("live");
  });

  it("switching simulation -> live removes run-sim-* and fixture events from the DOM", async () => {
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getByRole("button", { name: "Switch to live control API" }));
    // live API is unreachable in this test: no simulation banner, no ghost runs
    expect(await screen.findByText(/Failed to load console data/)).toBeInTheDocument();
    expect(screen.queryAllByText(/run-sim-/)).toHaveLength(0);
    expect(screen.queryByText(/SIMULATED — development fixture/)).not.toBeInTheDocument();
  });

  it("selection cannot carry a stale run id across modes", () => {
    let state: ConsoleState = { ...initialConsoleState, transportMode: "simulation", selectedRunId: "run-sim-1001" };
    state = reducer(state, {
      type: "loadOk",
      environments: [],
      runs: [{ runId: "run-live-1" } as never],
      skills: [],
      candidates: [],
    });
    expect(state.selectedRunId).toBe("run-live-1");
  });
});

describe("live default and honest stream badges", () => {
  it("defaults to the live transport with no simulation banner (dev-only opt-in)", async () => {
    render(<App />);
    expect(await screen.findByText(/Failed to load console data/)).toBeInTheDocument();
    expect(screen.queryByText(/SIMULATED — development fixture/)).not.toBeInTheDocument();
  });

  it("simulation mode never shows a green stream-connected badge implying inference", async () => {
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    expect(await screen.findByText(/Simulated stream connected \(fixture\)/)).toBeInTheDocument();
    expect(screen.queryByText("Stream connected")).not.toBeInTheDocument();
  });
});

describe("createRun, learning cycle, and registration", () => {
  it("creates a run with goal/environment/model/budget/idempotency and selects it", async () => {
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    await user.type(within(dialog).getByLabelText("Goal"), "Sim end-to-end goal");
    await user.type(within(dialog).getByLabelText("Model profile"), "profile-subscription");
    await user.type(within(dialog).getByLabelText("Model token cap"), "5000");
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));
    expect((await screen.findAllByText(/run-sim-1007/)).length).toBeGreaterThanOrEqual(1);
  });

  it("requires a nonzero model token cap before creating a run", async () => {
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    await user.type(within(dialog).getByLabelText("Goal"), "Goal");
    await user.type(within(dialog).getByLabelText("Model profile"), "profile");
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));
    expect(await within(dialog).findByText(/nonzero model token\/cost cap is required/)).toBeInTheDocument();
  });

  it("runs a learning cycle that stages an evidence-linked proposal", async () => {
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await user.click(await screen.findByRole("tab", { name: "Candidates" }));
    await user.click(await screen.findByRole("button", { name: "Run learning cycle" }));
    const dialog = await screen.findByRole("dialog", { name: "Run learning cycle" });
    await user.selectOptions(within(dialog).getByLabelText("Run (development attempt)"), "run-sim-1001");
    await user.type(within(dialog).getByLabelText("Predicted effect"), "fewer stale updates (prediction, not a score)");
    await user.type(within(dialog).getByLabelText("Evidence IDs (comma-separated)"), "ev-sim-31");
    await user.click(within(dialog).getByRole("button", { name: "Stage learning cycle" }));
    expect(await screen.findByText(/Learning cycle staged: action learn-sim-\d+ \(staged\)/)).toBeInTheDocument();
  });

  it("registers a complete environment manifest", async () => {
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await user.click(await screen.findByRole("tab", { name: "Environment registry" }));
    await user.type(await screen.findByLabelText("Environment ID"), "it-sim");
    await user.type(screen.getByLabelText("Package version"), "1.0.0");
    await user.type(screen.getByLabelText("Policy reference"), "policy-it");
    await user.type(screen.getByLabelText("Trusted evaluator reference"), "evaluator-it");
    await user.type(screen.getByLabelText("Reset fixture reference"), "reset-it");
    await user.click(screen.getByRole("checkbox", { name: /dry_run/ }));
    fireEvent.change(screen.getByLabelText(/Documentation \(JSON array/), { target: { value: '[{"id":"doc1","sha256":"a1","classification":"learner"}]' } });
    fireEvent.change(screen.getByLabelText(/Task goals \(JSON array/), { target: { value: '["read the counter"]' } });
    fireEvent.change(screen.getByLabelText(/Tool schemas \(JSON array/), { target: { value: '[{"name":"it.read","effect":"read"}]' } });
    await user.click(screen.getByRole("button", { name: "Register environment" }));
    expect(await screen.findByText(/Environment registered\./)).toBeInTheDocument();
    expect(await screen.findByText("it-sim")).toBeInTheDocument();
  });
});
