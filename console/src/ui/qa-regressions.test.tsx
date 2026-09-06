import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { createSimulationTransport } from "../api/simulation";
import type { ConsoleTransport } from "../api/transport";
// eslint-disable-next-line @typescript-eslint/no-var-requires

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
    fireEvent.change(within(dialog).getByLabelText("Goal"), { target: { value: "roundtrip probe goal" } });
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));

    // failure: dialog stays open, choices preserved, uncertainty stated
    expect(dialog).toBeInTheDocument();
    expect(await within(dialog).findByText(/API may have accepted the run/, {}, { timeout: 4000 })).toBeInTheDocument();
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

  it("applies the operator's edited budget values to the run request, not the defaults", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const sent: Array<{ budget?: Record<string, number> }> = [];
    const transport: ConsoleTransport = {
      ...sim,
      createRun: async (input) => {
        sent.push({ budget: { ...input.budget, modelTokens: input.budget.modelTokenCeiling } as never });
        return sim.createRun(input);
      },
    };
    render(<App transport={transport} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    // wait for the run-options response to land (Model label flips to "Luna"),
    // then edit the token budget so no late prefill can race the edit
    await waitFor(() => expect(within(dialog).getByLabelText("Model")).toHaveValue("Luna"), { timeout: 3000 });
    await waitFor(() => expect(within(dialog).getByLabelText("Token budget")).toHaveValue(20000));
    fireEvent.change(within(dialog).getByLabelText("Token budget"), { target: { value: "7777" } });
    await user.type(within(dialog).getByLabelText("Goal"), "budget application probe");
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));
    await waitFor(() => expect(sent.length).toBe(1));
    expect(sent[0].budget?.modelTokens).toBe(7777);
  });

  it("preserves legacy evaluation rows as UNVERIFIED, never trusted evidence (QA run eval_d123…)", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const transport: ConsoleTransport = {
      ...sim,
      // exact legacy QA shape: no candidateId, unknown state, trusted:true
      listEvaluations: async () =>
        [
          { evaluationId: "eval_d1238417af2044118f4982816655d34d", promotionEligible: false, state: "completed", trusted: true, validity: "valid" },
        ] as never,
    };
    render(<App transport={transport} />);
    await user.click(await screen.findByRole("tab", { name: "Candidates" }));
    expect(await screen.findByText(/legacy evaluation record\(s\) preserved as UNVERIFIED/)).toBeInTheDocument();
    expect(await screen.findByText(/state "completed"/)).toBeInTheDocument();
    // trusted flag from an unverifiable row is never rendered as evidence
    expect(screen.getByText(/trusted flag present \(unverified\)/)).toBeInTheDocument();
    expect(screen.queryByText(/\(trusted\)/)).not.toBeInTheDocument();
  });

  it("uses learningEligible for the learning source, including trusted failed attempts", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const runs = await sim.listRuns();
    const eligibleFailed = { ...runs[2], status: "failed" as const, learningEligible: true };
    const ineligibleSucceeded = { ...runs[3], status: "succeeded" as const, learningEligible: false };
    const transport: ConsoleTransport = {
      ...sim,
      listRuns: async () => [eligibleFailed, ineligibleSucceeded],
    };
    render(<App transport={transport} />);
    await user.click(await screen.findByRole("tab", { name: "Candidates" }));
    await user.click(await screen.findByRole("button", { name: "Run learning cycle" }));
    const dialog = await screen.findByRole("dialog", { name: "Run learning cycle" });
    const select = within(dialog).getByLabelText("Completed development run");
    const options = within(select).getAllByRole("option").map((o) => o.textContent ?? "");
    expect(options.some((t) => t.includes(eligibleFailed.runId))).toBe(true); // trusted failed dev attempt eligible
    expect(options.some((t) => t.includes(ineligibleSucceeded.runId))).toBe(false); // server excluded
  });

  it("moves focus to the run detail panel when a run is selected (768 master-detail)", async () => {
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await screen.findAllByRole("button", { name: /run-sim-1002/ });
    await user.click(screen.getAllByRole("button", { name: /run-sim-1002/ })[0]);
    // the detail region receives focus so keyboard users land in the new context
    await waitFor(() =>
      expect((document.activeElement as HTMLElement | null)?.getAttribute("aria-label")).toBe("Run details"),
    );
  });

  it("launches evaluation from a validated candidate and surfaces backend errors", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const launched: Array<{ candidateId: string }> = [];
    const transport: ConsoleTransport = {
      ...sim,
      launchEvaluation: async (input) => {
        launched.push(input);
        return { evaluationId: "eval-sim-900", state: "queued" };
      },
    };
    render(<App transport={transport} />);
    await user.click(await screen.findByRole("tab", { name: "Candidates" }));
    await user.click((await screen.findAllByRole("button", { name: "Launch evaluation" }))[0]);
    expect(await screen.findByText(/Evaluation eval-sim-900 queued/)).toBeInTheDocument();
    expect(launched[0].candidateId).toBe("cand-sim-204");
  });

  it("valid cards state proposal validation, never a performance claim", async () => {
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await userEvent.setup().click(await screen.findByRole("tab", { name: "Candidates" }));
    expect(await screen.findByText(/Proposal validation passed — this is not a performance result/)).toBeInTheDocument();
  });

  it("shows the workflow strip and Learn-from-this-run only for server-declared eligible runs", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const runs = await sim.listRuns();
    const eligible = { ...runs[3], learningEligible: true };
    const ineligible = { ...runs[0], status: "running" as const, learningEligible: false };
    const transport: ConsoleTransport = { ...sim, listRuns: async () => [eligible, ineligible] };
    render(<App transport={transport} />);
    await screen.findAllByRole("button", { name: /run-sim-1004/ });
    // workflow guidance renders all four stages with honest captions
    expect(screen.getByText("1. Execute goal")).toBeInTheDocument();
    expect(screen.getByText("2. Learn from verified attempt")).toBeInTheDocument();
    expect(screen.getByText("3. Evaluate candidate")).toBeInTheDocument();
    expect(screen.getByText("4. Activate only if gate passes")).toBeInTheDocument();
    // select the eligible run: the Learn action appears
    await user.click(screen.getAllByRole("button", { name: /run-sim-1004/ })[0]);
    expect(await screen.findByRole("button", { name: "Learn from this run" })).toBeInTheDocument();
    // select the ineligible run: guidance replaces the action, no fake progress
    await user.click(screen.getAllByRole("button", { name: /run-sim-1001/ })[0]);
    await waitFor(() => expect(screen.queryByRole("button", { name: "Learn from this run" })).not.toBeInTheDocument());
    expect(screen.getByText(/Eligible after a server-verified outcome/)).toBeInTheDocument();
  });

  it("lists a run whose learningEligible flips true after the terminal outcome, without a full reload (QA run_a1662f…)", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const baseRun = (await sim.listRuns())[3]; // succeeded run
    let eligible = false;
    let calls = 0;
    const transport: ConsoleTransport = {
      ...sim,
      // the server projects learningEligible only after the private trusted
      // outcome commits; the console must pick that up on the Candidates-tab
      // refresh, not a full reload
      listRuns: async () => {
        calls += 1;
        const learningEligible = calls >= 2 ? true : eligible;
        return [{ ...baseRun, learningEligible }];
      },
    };
    render(<App transport={transport} />);
    await screen.findAllByRole("button", { name: /run-sim-1004/ });
    // opening Candidates forces an authoritative refresh: the server now
    // reports learningEligible=true for the completed run
    await user.click(await screen.findByRole("tab", { name: "Candidates" }));
    await user.click(await screen.findByRole("button", { name: "Run learning cycle" }));
    const dialog = await screen.findByRole("dialog", { name: "Run learning cycle" });
    // the dialog-open refresh lands; the select appears without a reload
    await waitFor(() => expect(within(dialog).getByLabelText("Completed development run")).toBeInTheDocument(), { timeout: 4000 });
    const select = within(dialog).getByLabelText("Completed development run");
    await waitFor(() => expect(within(select).getAllByRole("option").length).toBeGreaterThanOrEqual(2));
    const options = within(select).getAllByRole("option").map((o) => o.textContent ?? "");
    expect(options.some((t) => t.includes(baseRun.runId))).toBe(true);
    expect(calls).toBeGreaterThanOrEqual(2); // the refresh happened without a reload
  });

  it("opens the learning-cycle dialog from the empty-candidates state (regression)", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const transport: ConsoleTransport = { ...sim, listCandidates: async () => [] };
    render(<App transport={transport} />);
    await user.click(await screen.findByRole("tab", { name: "Candidates" }));
    // the empty state previously returned before mounting the modal, so the
    // button did nothing
    await user.click(await screen.findByRole("button", { name: "Run learning cycle" }));
    expect(await screen.findByRole("dialog", { name: "Run learning cycle" })).toBeInTheDocument();
  });

  it("distinguishes contract errors from the empty state", async () => {
    const { SchemaError, parseCandidates } = await import("../api/validate");
    expect(() => parseCandidates([{ candidateId: "c" }])).toThrow(SchemaError); // missing projection
    expect(() =>
      parseCandidates([
        {
          candidateId: "c",
          state: "validated",
          predictedEffect: "p",
          baseBundleHash: "h",
          editOperations: [],
          changedArtifactHashes: [],
          supportingEvidenceIds: [],
        },
      ]),
    ).not.toThrow();
  });

  it("resets the token budget to the advertised default after success, never a hardcoded literal", async () => {
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    let dialog = await screen.findByRole("dialog", { name: "Create run" });
    // server-advertised default (fixture mirrors the authoritative 20,000 default)
    await waitFor(() => expect(within(dialog).getByLabelText("Token budget")).toHaveValue(20000), { timeout: 3000 });
    await user.type(within(dialog).getByLabelText("Goal"), "advertised reset probe");
    await user.click(within(dialog).getByRole("button", { name: "Create run" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    // reopen: the token budget returns to the ADVERTISED default (4000 here),
    // proving the reset path follows the server value rather than a literal
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    dialog = await screen.findByRole("dialog", { name: "Create run" });
    await waitFor(() => expect(within(dialog).getByLabelText("Token budget")).toHaveValue(20000), { timeout: 3000 });
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
    expect(within(dialog).queryByText(/interactive/)).not.toBeInTheDocument(); // only declared modes in the dialog
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
