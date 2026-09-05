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

  it("marks the connection stale on a sequence gap instead of applying", () => {
    const s1 = applyEvent(state, ev(3));
    expect(s1.cursors["r1"]).toBeUndefined();
    expect(s1.connection).toBe("stale");
    expect(s1.events["r1"]).toBeUndefined();
  });

  it("derives run status from status events", () => {
    let s = applyEvent(state, ev(1, "Run queued and pinned to bundle v7."));
    expect(s.runs[0].status).toBe("running"); // 'queued' only applies while already queued
    s = applyEvent(s, ev(2, "Approval required for ledger.append (write)."));
    expect(s.runs[0].status).toBe("awaiting_approval");
    s = applyEvent(s, ev(3, "Run succeeded."));
    expect(s.runs[0].status).toBe("succeeded");
  });
});
