import { describe, expect, it } from "vitest";
import { applyEvent, initialConsoleState } from "./consoleStore";
import type { RunEvent, RunRecord } from "../api/types";

const run: RunRecord = {
  runId: "r1",
  taskRef: { id: "t", version: "1", sha256: "h1" },
  environmentRef: { id: "e", version: "1", sha256: "h2" },
  policyRef: { id: "p", version: "1", sha256: "h3" },
  modelProfileRef: { id: "m", version: "1", sha256: "h4" },
  skillBundleRef: { id: "b", version: "1", sha256: "h5" },
  budgetRef: { id: "bu", version: "1", sha256: "h6" },
  status: "running",
  lastEventSequence: 0,
};

const state = { ...initialConsoleState, runs: [run] };

function ev(sequence: number, summary = `event ${sequence}`): RunEvent {
  return { runId: "r1", sequence, at: "2026-09-06T10:00:00Z", kind: "status", summary };
}

describe("event cursor semantics", () => {
  it("applies in-order events and advances the cursor", () => {
    const s1 = applyEvent(state, ev(1));
    const s2 = applyEvent(s1, ev(2));
    expect(s2.cursors["r1"]).toBe(2);
    expect(s2.events["r1"]).toHaveLength(2);
  });

  it("drops duplicate replays after reconnect (no duplicate application)", () => {
    const s1 = applyEvent(state, ev(1));
    const s2 = applyEvent(s1, ev(2));
    const replayed = applyEvent(s2, ev(1));
    const replayedAgain = applyEvent(replayed, ev(2));
    expect(replayedAgain.cursors["r1"]).toBe(2);
    expect(replayedAgain.events["r1"]).toHaveLength(2);
  });

  it("advances the cursor past hidden evaluator rows on a sequence gap without a false stale warning", () => {
    const s1 = applyEvent(state, ev(3));
    // hidden evaluator_only rows create legitimate gaps: the cursor advances
    // past them (server resume ledger is complete), no stale flag is raised,
    // and no fabricated event content is applied
    expect(s1.cursors["r1"]).toBe(3);
    expect(s1.connection).toBe(initialConsoleState.connection);
    expect(s1.events["r1"]).toBeUndefined();
  });

  it("status changes ONLY via validated lifecycle transitions or record refresh", () => {
    // durable lifecycle events carry runStatus from the event TYPE
    let s = applyEvent(state, { ...ev(1), runStatus: "queued" } as RunEvent);
    expect(s.runs[0].status).toBe("queued");
    s = applyEvent(s, { ...ev(2), runStatus: "running" } as RunEvent);
    expect(s.runs[0].status).toBe("running");
    s = applyEvent(s, { ...ev(3), runStatus: "failed" } as RunEvent);
    expect(s.runs[0].status).toBe("failed");
    // plane-shape status events with no validated field NEVER change status,
    // regardless of their display text
    s = applyEvent(s, ev(4, "Run started in interactive mode with authenticated model runner."));
    expect(s.runs[0].status).toBe("failed");
  });
});
