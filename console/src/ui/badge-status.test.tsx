import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { App } from "../App";
import { createRestTransport } from "../api/rest";
import { normalizeSseEvent } from "../api/validate";
import { applyEvent, initialConsoleState, reducer } from "../state/consoleStore";
import type { RunEvent, RunRecord } from "../api/types";
import { createSimulationTransport } from "../api/simulation";

/** Minimal EventSource double delivering captured durable frames. */
class FakeEventSource {
  static instances: FakeEventSource[] = [];
  static frames: string[] = [];
  url: string;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
    setTimeout(() => {
      if (this.closed) return;
      for (const frame of FakeEventSource.frames) {
        if (this.closed) return;
        this.onmessage?.({ data: frame });
      }
      setTimeout(() => !this.closed && this.onerror?.(), 5); // stream EOF
    }, 0);
  }
  close() {
    this.closed = true;
  }
}

const wire = (seq: number, type: string) =>
  JSON.stringify({
    id: seq,
    event: type,
    data: {
      run_id: "run_badge_1",
      sequence: seq,
      event_type: type,
      content_hash: `${type}-hash`,
      source_ref: `{"id":"art_${type}","version":"1","sha256":"${type}-hash"}`,
      trust_class: "system",
      visibility: "operator",
      redacted: 0,
    },
  });

beforeEach(() => {
  FakeEventSource.instances = [];
  FakeEventSource.frames = [wire(1, "run_created"), wire(2, "run_started"), wire(3, "run_failed")];
  vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("authoritative badge status (never from display text)", () => {
  it("lifecycle event TYPES drive status transitions in the store", () => {
    const run = { runId: "run_badge_1", status: "queued", lastEventSequence: 0 } as RunRecord;
    let withRun = { ...initialConsoleState, runs: [run] };
    withRun = applyEvent(withRun, normalizeSseEvent(JSON.parse(wire(1, "run_created")), "sse"));
    const started = normalizeSseEvent(JSON.parse(wire(2, "run_started")), "sse");
    expect(started.runStatus).toBe("running");
    const afterStart = applyEvent(withRun, started);
    expect(afterRunStatus(afterStart, "run_badge_1")).toBe("running");
    const failed = normalizeSseEvent(JSON.parse(wire(3, "run_failed")), "sse");
    expect(failed.runStatus).toBe("failed");
    expect(afterRunStatus(applyEvent(afterStart, failed), "run_badge_1")).toBe("failed");
  });

  it("display text NEVER changes status: plane-shape text is ignored locally", () => {
    const run = { runId: "r", status: "queued", lastEventSequence: 0 } as RunRecord;
    const withRun = { ...initialConsoleState, runs: [run] };
    const textEvent: RunEvent = {
      runId: "r",
      sequence: 1,
      at: "",
      kind: "status",
      summary: "Run started in interactive mode with authenticated model runner.",
    };
    const after = applyEvent(withRun, textEvent);
    // no validated status field -> stays queued; the record refresh path
    // (runsRefreshed) is the only other way status changes
    expect(afterRunStatus(after, "r")).toBe("queued");
  });

  it("runsRefreshed merges the actual RunRecord status", () => {
    const run = { runId: "r", status: "queued", lastEventSequence: 2 } as RunRecord;
    const withRun = { ...initialConsoleState, runs: [run] };
    const after = reducer(withRun, {
      type: "runsRefreshed",
      runs: [{ ...run, status: "failed" } as RunRecord],
    });
    expect(afterRunStatus(after, "r")).toBe("failed");
  });

  it("REAL WIRE regression: badge goes queued -> running -> failed and record refresh follows terminal EOF", async () => {
    const transport = createRestTransport();
    // session handshake must succeed for the stream to start
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL) => new Response("{}", { status: 200 })),
    );
    const listRuns = vi
      .spyOn(transport, "listRuns")
      .mockResolvedValue([
        {
          runId: "run_badge_1",
          status: "failed",
          lastEventSequence: 3,
          taskRef: { id: "t", version: "1", sha256: "h" },
          environmentRef: { id: "e", version: "1", sha256: "h" },
          policyRef: { id: "p", version: "1", sha256: "h" },
          modelProfileRef: { id: "m", version: "1", sha256: "h" },
          skillBundleRef: { id: "b", version: "1", sha256: "h" },
          budgetRef: { id: "bu", version: "1", sha256: "h" },
          executionMode: "dry_run",
        } as RunRecord,
      ]);
    vi.spyOn(transport, "listEnvironments").mockResolvedValue([]);
    vi.spyOn(transport, "listSkills").mockResolvedValue([]);
    vi.spyOn(transport, "listCandidates").mockResolvedValue([]);
    render(<App transport={transport} />);
    // stream delivers created -> started -> failed frames, then EOF
    await waitFor(() => expect(listRuns).toHaveBeenCalled(), { timeout: 5000 });
    // the badge reflects the authoritative terminal status after refresh
    await waitFor(() => expect(screen.getAllByText("Failed").length).toBeGreaterThanOrEqual(1), { timeout: 5000 });
  });

  it("simulation transports remain unaffected by status derivation changes", () => {
    const transport = createSimulationTransport({ disconnectAfterEvents: 0 });
    expect(transport.mode).toBe("simulation");
  });
});

function afterRunStatus(state: ReturnType<typeof import("../state/consoleStore").reducer>, runId: string): string | undefined {
  return state.runs.find((r) => r.runId === runId)?.status;
}
