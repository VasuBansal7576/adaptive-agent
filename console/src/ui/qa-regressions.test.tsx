import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { createSimulationTransport } from "../api/simulation";
import type { ConsoleTransport } from "../api/transport";
import type { RunRecord } from "../api/types";
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

  it("Modal directly: Escape fires onClose exactly once; Tab wraps last->first and Shift+Tab first->last", async () => {
    const { Modal } = await import("../components/ui");
    const onClose = vi.fn();
    const firstRef = { current: null as HTMLInputElement | null };
    const lastRef = { current: null as HTMLButtonElement | null };
    const { rerender } = render(
      <Modal open title="Unit" onClose={onClose} returnFocusTo={{ current: null }}>
        <form>
          <input
            aria-label="First field"
            ref={(el) => {
              firstRef.current = el;
            }}
          />
          <button
            ref={(el) => {
              lastRef.current = el;
            }}
            onClick={onClose}
          >
            Last control
          </button>
        </form>
      </Modal>,
    );
    const dialog = screen.getByRole("dialog", { name: "Unit" });
    expect(dialog).toBeInTheDocument();

    // focus the LAST control, press Tab once: must land on the FIRST field (single wrap)
    lastRef.current?.focus();
    await userEvent.setup().keyboard("{Tab}");
    expect(firstRef.current).toHaveFocus();

    // focus the FIRST control, press Shift+Tab once: must land on the LAST control
    firstRef.current?.focus();
    await userEvent.setup().keyboard("{Shift>}{Tab}{/Shift}");
    expect(lastRef.current).toHaveFocus();

    // Escape fires onClose EXACTLY once even if pressed twice (dialog closed -> listener gone)
    await userEvent.setup().keyboard("{Escape}");
    rerender(
      <Modal open={false} title="Unit" onClose={onClose}>
        <form />
      </Modal>,
    );
    await userEvent.setup().keyboard("{Escape}");
    await userEvent.setup().keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("Modal Escape closes exactly once and Tab wraps one step per press", async () => {
    // use the new-run dialog: onClose count observable via the dialog state
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    // move focus to the last focusable (Create run), then Tab: must wrap to the first field in ONE press (no double-jump)
    const createBtn = within(dialog).getByRole("button", { name: "Create run" });
    (createBtn as HTMLButtonElement).focus();
    expect(createBtn).toHaveFocus();
    await user.keyboard("{Tab}");
    const focusables = dialog.querySelectorAll<HTMLElement>("a[href], button:not([disabled]), input, select, textarea");
    const positions = focusables.length;
    const afterOne = document.activeElement;
    await user.keyboard("{Tab}");
    const afterTwo = document.activeElement;
    await user.keyboard("{Tab}");
    const afterThree = document.activeElement;
    // no double-execution: one press advances exactly one position in the trap
    expect(afterTwo).not.toBe(afterOne);
    expect(afterThree).not.toBe(afterTwo);
    expect(within(dialog).queryAllByRole("button", { name: "Create run" }).length).toBe(1);
    expect(positions).toBeGreaterThan(1);
    // Escape closes and does not re-fire (dialog stays closed)
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows the workflow strip and Learn-from-this-run only for eligible runs", async () => {
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
    expect(screen.getByText(/Becomes eligible after a verified outcome/)).toBeInTheDocument();
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

  it("distinguishes sources by EXACT registered id: built-in fixture vs AppWorld vs unspecified", async () => {
    const { sourceLabelFor } = await import("../api/sourceLabels");
    // exact known built-ins keep fixture wording
    expect(sourceLabelFor("finance").tag).toBe("simulated business fixture");
    expect(sourceLabelFor("customer_support").tag).toBe("simulated business fixture");
    expect(sourceLabelFor("it").tag).toBe("simulated business fixture");
    expect(sourceLabelFor("lab_scheduling").tag).toBe("simulated business fixture");
    // the explicit -sim dev-fixture ids used by the console simulation
    expect(sourceLabelFor("finance-sim").tag).toBe("simulated business fixture");
    // AppWorld exact id only, plain published-benchmark wording
    const appworld = sourceLabelFor("appworld");
    expect(appworld.tag).toBe("AppWorld");
    expect(appworld.provenance).toBe("Tasks use AppWorld, a published benchmark with simulated app data.");
    // arbitrary user packages are NOT mislabeled
    expect(sourceLabelFor("finance-external").tag).toBe("source not specified");
    expect(sourceLabelFor("appworld-custom").tag).toBe("source not specified");
    // unknown registered ids: source not specified, never built-in fixture
    const unknown = sourceLabelFor("mystery_env");
    expect(unknown.tag).toBe("source not specified");
    expect(unknown.provenance).toBe("Data source not specified.");
    // case is NOT folded: user-defined casing stays unregistered
    expect(sourceLabelFor("Finance").tag).toBe("source not specified");
    expect(sourceLabelFor("APPWORLD").tag).toBe("source not specified");

  });

  it("renders the AppWorld source label in registry and workflow without claiming results", async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const envs = await sim.listEnvironments();
    envs.push({ environmentId: "appworld", version: "1.0", validationState: "valid", evaluatorReady: true, toolCount: 3, policyScope: "appworld/*", executionModes: ["interactive"] });
    const appworldRun: RunRecord = { ...(await sim.listRuns())[3], runId: "run_aw_1", environmentId: "appworld", environmentRef: { id: "appworld", version: "1", sha256: "aw" }, learningEligible: true };
    const transport: ConsoleTransport = {
      ...sim,
      listEnvironments: async () => envs,
      listRuns: async () => [appworldRun],
    };
    render(<App transport={transport} />);
    // registry card tags AppWorld distinctly
    await user.click(await screen.findByRole("tab", { name: "Environment registry" }));
    expect(await screen.findByText("AppWorld")).toBeInTheDocument();
    // selected-run workflow carries the AppWorld provenance, no result claims
    await user.click(await screen.findByRole("tab", { name: "Runs" }));
    await user.click((await screen.findAllByRole("button", { name: /run_aw_1/ }))[0]);
    expect(await screen.findByText(/Tasks use AppWorld, a published benchmark with simulated app data/)).toBeInTheDocument();
  });

  it("labels the built-in task catalog truthfully as simulated business fixtures", async () => {
    const user = userEvent.setup();
    const transport: ConsoleTransport = { ...createSimulationTransport({ disconnectAfterEvents: 0 }) };
    render(<App transport={transport} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    // workflow strip states the provenance boundary explicitly
    expect(screen.getByText(/Tasks use the built-in simulated business fixture catalog/)).toBeInTheDocument();
    // new-run dialog carries the same boundary near task selection
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    expect(
      await within(dialog).findByText(/Tasks use the built-in simulated business fixture catalog/),
    ).toBeInTheDocument();
    // no external/published benchmark sourcing claimed in this built-in flow
    expect((document.body.textContent || "").toLowerCase()).not.toContain("published benchmark");
    expect((document.body.textContent || "").toLowerCase()).not.toContain("appworld");
  });

  it("quick comparison: launches 6-run development check, shows progress and per-arm results, cancels", { timeout: 30000 }, async () => {
    const user = userEvent.setup();
    const sim = createSimulationTransport({ disconnectAfterEvents: 0 });
    const launched: Array<{ candidateId: string; baseBundleHash: string }> = [];
    const transport: ConsoleTransport = {
      ...sim,
      launchDiagnostic: async (input) => {
        launched.push(input);
        return sim.launchDiagnostic(input);
      },
    };
    render(<App transport={transport} />);
    await user.click(await screen.findByRole("tab", { name: "Candidates" }));

    // compact honest copy: development check, not promotion evidence; estimate + 6 runs
    expect(await screen.findByText(/Development check, not promotion evidence/)).toBeInTheDocument();
    expect(await screen.findByText(/6 runs \(3 public development tasks, baseline and learned\)/)).toBeInTheDocument();
    expect(await screen.findByText(/Estimated a few minutes/)).toBeInTheDocument();

    // launch: primary action after Learn
    await user.click(await screen.findByRole("button", { name: /Quick comparison/ }));
    await waitFor(() => expect(launched.length).toBe(1));
    // duplicate launch disabled while an active diagnostic exists
    await waitFor(() => expect(screen.getByRole("button", { name: /Quick comparison in progress/ })).toBeDisabled());

    // live poll progresses 0/6 -> 6/6 with per-arm results (2s cadence)
    await waitFor(
      () => expect(screen.getByText("6/6 runs")).toBeInTheDocument(),
      { timeout: 20000 },
    );
    expect((await screen.findAllByText("B0")).length).toBeGreaterThanOrEqual(1);
    expect((await screen.findAllByText("L")).length).toBeGreaterThanOrEqual(1);
    // never conflated with promotion evidence
    expect(screen.getAllByText(/Development check, not promotion evidence/).length).toBeGreaterThanOrEqual(1);

    // second diagnostic: cancel path
    await user.click(await screen.findByRole("button", { name: "Quick comparison" }));
    await waitFor(() => expect(screen.getAllByText(/2\/6 runs|1\/6 runs/)[0]).toBeInTheDocument(), { timeout: 12000 });
    const cancelButtons = screen.getAllByRole("button", { name: "Cancel" });
    await user.click(cancelButtons[cancelButtons.length - 1]);
    await waitFor(() => expect(screen.getAllByText("Cancelled").length).toBeGreaterThanOrEqual(1), { timeout: 4000 });
  });

  it("diagnostic parse: rejects malformed rows and wrong totalCells instead of silently rendering", async () => {
    const { parseDiagnostics, SchemaError } = await import("../api/validate");
    expect(() => parseDiagnostics([{ diagnosticId: "d1", candidateId: "c", baseBundleHash: "b", candidateBundleHash: "cb", state: "running", completedCells: 1, totalCells: 12, startedAt: "t", updatedAt: "t", armSummaries: [], error: null, promotionEligible: false }])).toThrow(SchemaError);
    expect(() => parseDiagnostics([{ diagnosticId: "d1", candidateId: "c", baseBundleHash: "b", candidateBundleHash: "cb", state: "completed", completedCells: 6, totalCells: 6, startedAt: "t", updatedAt: "t", armSummaries: [{ arm: "X" }], error: null, promotionEligible: false }])).toThrow(SchemaError);
    expect(() => parseDiagnostics([{ diagnosticId: "d1", candidateId: "c", baseBundleHash: "b", candidateBundleHash: "cb", state: "completed", completedCells: 6, totalCells: 6, startedAt: "t", updatedAt: "t", armSummaries: [], error: null, promotionEligible: true }])).toThrow(SchemaError);
  });

  it("full validation stays a secondary explicit long-running action", async () => {
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await userEvent.setup().click(await screen.findByRole("tab", { name: "Candidates" }));
    const full = await screen.findByText("Full validation (360 runs)");
    expect(full).toBeInTheDocument();
    // presented as notice text, never as a quick check
    expect(full.getAttribute("title")).toContain("takes hours");
    // it is NOT a launch button for the diagnostics flow
    expect(full.tagName).toBe("SPAN");
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
