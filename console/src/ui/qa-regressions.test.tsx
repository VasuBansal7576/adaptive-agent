import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { createSimulationTransport } from "../api/simulation";
import type { ConsoleTransport } from "../api/transport";

beforeEach(() => {
  vi.stubGlobal("EventSource", undefined);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("qa regressions: createRun recovery and honesty", () => {
  function failingThenOkTransport(): { transport: ConsoleTransport; keys: string[] } {
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const keys: string[] = [];
    let failed = false;
    const transport: ConsoleTransport = {
      ...sim,
      createRun: async (input) => {
        keys.push(input.idempotencyKey);
        if (!failed) {
          failed = true;
          // first attempt fails AFTER the request left the console (QA scenario)
          throw new Error("schema violation at runs[0].modelProfileRef.sha256");
        }
        return sim.createRun(input);
      },
    };
    return { transport, keys };
  }

  it("preserves form choices and reuses the same idempotency key across recovery", async () => {
    const user = userEvent.setup();
    const { transport, keys } = failingThenOkTransport();
    render(<App transport={transport} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    await user.type(within(dialog).getByLabelText("Goal"), "roundtrip probe goal");
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));

    // failure: dialog stays open, choices preserved, uncertainty stated
    expect(dialog).toBeInTheDocument();
    expect(await within(dialog).findByText(/API may have accepted the run/)).toBeInTheDocument();
    await waitFor(() => expect(within(dialog).getByLabelText("Goal")).toHaveValue("roundtrip probe goal"));
    // global banner also states honest uncertainty
    expect(await screen.findByText(/API may have accepted this change/)).toBeInTheDocument();

    // retry with the SAME idempotency key
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));
    await waitFor(() => expect(keys.length).toBe(2));
    expect(keys[0]).toBe(keys[1]);
    // success closes the dialog and shows the run
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect((await screen.findAllByText(/run-sim-1007/)).length).toBeGreaterThanOrEqual(1);
  });

  it("restricts mode choices to the selected environment's declared modes", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const envs = await sim.listEnvironments();
    envs[0].executionModes = ["dry_run"];
    const transport: ConsoleTransport = { ...sim, listEnvironments: async () => envs };
    render(<App transport={transport} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    const modeSelect = within(dialog).getByLabelText("Mode") as HTMLSelectElement;
    // default and only option is the declared mode
    expect(modeSelect).toHaveValue("dry_run");
    expect(within(modeSelect).getAllByRole("option")).toHaveLength(1);
    expect(screen.queryByText(/interactive/)).not.toBeInTheDocument();
  });

  it("acknowledges successful package validation visibly", async () => {
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await user.click(await screen.findByRole("tab", { name: "Environment registry" }));
    await user.type(await screen.findByLabelText("Environment ID"), "ack-probe");
    await user.type(screen.getByLabelText("Package version"), "1");
    await user.type(screen.getByLabelText("Policy reference"), "p");
    await user.type(screen.getByLabelText("Trusted evaluator reference"), "e");
    await user.type(screen.getByLabelText("Reset fixture reference"), "r");
    await user.click(screen.getByRole("checkbox", { name: /dry_run/ }));
    const { fireEvent } = await import("@testing-library/react");
    fireEvent.change(screen.getByLabelText(/Documentation \(JSON array/), { target: { value: '[{"id":"d","sha256":"a","classification":"learner"}]' } });
    fireEvent.change(screen.getByLabelText(/Task goals \(JSON array/), { target: { value: '["g"]' } });
    fireEvent.change(screen.getByLabelText(/Tool schemas \(JSON array/), { target: { value: '[{"name":"x.read","effect":"read"}]' } });
    await user.click(screen.getByRole("button", { name: "Validate package" }));
    expect(await screen.findByText(/Package validated/)).toBeInTheDocument();
  });

  it("shows a connected stream badge (never Connecting) in live mode with no runs", async () => {
    // QA scenario: bootstrap succeeds, but the plane has no runs yet. The
    // header badge must report a connected transport, never an endless Connecting.
    const responses: Record<string, { status: number; body: unknown }> = {
      "/api/session/bootstrap": { status: 200, body: { status: "ready" } },
      "/api/session": { status: 200, body: { authenticated: true } },
      "/api/environments": { status: 200, body: [] },
      "/api/runs": { status: 200, body: [] },
      "/api/skills": { status: 200, body: [] },
      "/api/candidates": { status: 200, body: [] },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL) => {
        const hit = responses[String(url).replace(/^https?:\/\/[^/]+/, "")] ?? responses[String(url)];
        if (!hit) throw new TypeError(`unexpected fetch ${String(url)}`);
        return new Response(JSON.stringify(hit.body), { status: hit.status });
      }),
    );
    render(<App />);
    expect(await screen.findByText("Stream connected")).toBeInTheDocument();
    expect(screen.queryByText("Connecting")).not.toBeInTheDocument();
    expect(await screen.findByText("No runs yet")).toBeInTheDocument();
  });
});
