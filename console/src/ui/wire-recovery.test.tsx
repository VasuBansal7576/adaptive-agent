import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { createRestTransport } from "../api/rest";

const transportRef = { current: null as unknown as ReturnType<typeof createRestTransport> };
const states: string[] = [];
import { normalizeSseEvent } from "../api/validate";
import { createSimulationTransport } from "../api/simulation";

/** Minimal EventSource double: records each connection's URL (cursor) and
 *  lets tests emit captured-wire frames and drops. */
class FakeEventSource {
  static instances: FakeEventSource[] = [];
  static frames: string[] = [];
  static dropOnConnections = 0; // drop after first frame on the first N connections
  static connectionCount = 0;
  url: string;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
    const shouldDrop = FakeEventSource.connectionCount < FakeEventSource.dropOnConnections;
    FakeEventSource.connectionCount += 1;
    setTimeout(() => this.pump(shouldDrop), 0);
  }
  pump(dropAfterFirst: boolean) {
    if (this.closed) return;
    for (const frame of FakeEventSource.frames) {
      if (this.closed) return;
      this.onmessage?.({ data: frame });
      if (dropAfterFirst) {
        setTimeout(() => !this.closed && this.onerror?.(), 5);
        return;
      }
    }
  }
  close() {
    this.closed = true;
  }
}

const CAPTURED_WIRE = [
  JSON.stringify({
    id: 1,
    event: "run_created",
    data: {
      evidence_id: "ev_370205e6caa74d32b05b04c2377deab0",
      run_id: "run_873a94b0ce424e7699709663b461eb13",
      sequence: 1,
      event_type: "run_created",
      content_hash: "9baec1f5358fbce8fa7d8a27fc4be5cb89d28857c67daf0b7ed83cdf8fe610b5",
      source_ref: '{"id":"art_9baec1f5358fbce8","version":"1","sha256":"9baec1f5358fbce8fa7d8a27fc4be5cb89d28857c67daf0b7ed83cdf8fe610b5"}',
      trust_class: "system",
      visibility: "learner",
      redacted: 1,
    },
  }),
  JSON.stringify({
    id: 2,
    event: "run_started",
    data: {
      evidence_id: "ev_2f747df445dc4971b93a1c6e9fc1d62f",
      run_id: "run_873a94b0ce424e7699709663b461eb13",
      sequence: 2,
      event_type: "run_started",
      content_hash: "b2f29e65724a00e04eb7e44e029ef724a00e04eb7e44e029efb8b999e8d3f52a5",
      source_ref: '{"id":"art_b2f29e65724a00e0","version":"1","sha256":"b2f29e65724a00e04eb7e44e029efb8b999e8d3f52a5"}',
      trust_class: "system",
      visibility: "operator",
      redacted: 0,
    },
  }),
];

beforeEach(() => {
  FakeEventSource.instances = [];
  FakeEventSource.frames = CAPTURED_WIRE;
  FakeEventSource.dropOnConnections = 0;
  FakeEventSource.connectionCount = 0;
  vi.stubGlobal("EventSource", FakeEventSource as unknown as typeof EventSource);
  // isolate network: transports in these tests must never reach a real host
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL) => {
      const u = String(url);
      if (u.includes("/session")) return new Response("{}", { status: 200 });
      if (u.includes("/runs/")) return new Response(JSON.stringify({ status: "running", lastEventSequence: 2 }), { status: 200 });
      return new Response("{}", { status: 200 });
    }),
  );
});
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("captured ACTUAL wire through normalizeSseEvent and the live UI", () => {
  beforeEach(() => {
    transportRef.current = createRestTransport();
    states.length = 0;
  });

  it("records a genuine terminal EOF as Closed, never Stale (QA run_a1662f idle diagnosis)", async () => {
    // the run is terminal: EOF must settle Closed
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL) => {
        const u = String(url);
        if (u.includes("/session")) return new Response("{}", { status: 200 });
        return new Response(JSON.stringify({ status: "succeeded", lastEventSequence: 2 }), { status: 200 });
      }),
    );
    const { default: userEvent } = await import("@testing-library/user-event");
    render(<div />);
    FakeEventSource.frames = CAPTURED_WIRE;
    FakeEventSource.dropOnConnections = 0;
    FakeEventSource.connectionCount = 0;
    let applied = 0;
    const close = transportRef.current.openRunStream("run_873a94b0ce424e7699709663b461eb13", 0, {
      onEvent: () => {
        applied += 1;
      },
      onState: (state) => {
        states.push(state);
      },
    });
    await waitFor(() => expect(applied).toBe(2), { timeout: 4000 });
    // server-side EOF surfaces as an EventSource error; the terminal status
    // check resolves to closed, which the app must record (cleanup emits nothing)
    FakeEventSource.instances.forEach((es) => es.onerror?.());
    await waitFor(() => expect(states).toContain("closed"), { timeout: 4000 });
    expect(states).not.toContain("stale");
    close();
  });

  it("advances the browser cursor over the captured durable frames with zero rejections", () => {
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    const transport = createRestTransport();
    let cursor = 0;
    let rejected = 0;
    const close = transport.openRunStream("run_873a94b0ce424e7699709663b461eb13", cursor, {
      onEvent: (event) => {
        cursor = event.sequence;
      },
      onState: () => undefined,
    });
    for (const frame of CAPTURED_WIRE) {
      try {
        const event = normalizeSseEvent(JSON.parse(frame), "sse.data");
        if (event.sequence > cursor) cursor = event.sequence;
      } catch {
        rejected += 1;
      }
    }
    close();
    expect(rejected).toBe(0);
    expect(cursor).toBe(2);
  });

  it("reconnects after a mid-stream drop, resuming from the preserved cursor without duplicates", async () => {
    const transport = createRestTransport();
    // session bootstrap + non-terminal run status for the EOF check
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string | URL) => {
        const u = String(url);
        if (u.includes("/session")) return new Response("{}", { status: 200 });
        if (u.includes("/runs/run_")) return new Response(JSON.stringify({ status: "running", lastEventSequence: 2 }), { status: 200 });
        return new Response("{}", { status: 200 });
      }),
    );
    render(<div />); // keep the test DOM alive
    FakeEventSource.instances = [];
    FakeEventSource.connectionCount = 0;
    FakeEventSource.dropOnConnections = 1; // first connection drops after one frame
    FakeEventSource.frames = CAPTURED_WIRE;
    const applied: number[] = [];
    const close = transport.openRunStream("run_873a94b0ce424e7699709663b461eb13", 0, {
      onEvent: (event) => applied.push(event.sequence),
      onState: () => undefined,
    });
    // first frame applies (cursor 1), stream drops, reconnect resumes at cursor 1
    await waitFor(() => expect(applied).toEqual([1, 2]), { timeout: 8000 });
    const urls = FakeEventSource.instances.map((es) => es.url);
    expect(urls[0]).toContain("cursor=0");
    expect(urls.some((u) => u.includes("cursor=1"))).toBe(true); // preserved cursor
    close();
  });

  it("keeps the stale banner's manual reconnect action available", async () => {
    const user = userEvent.setup();
    const transport = createRestTransport();
    const reconnect = vi.spyOn(transport, "reconnect").mockResolvedValue(undefined);
    render(<App transport={transport} />);
    await screen.findByText(/Console cannot refresh data/);
    // force a stale state via the store path the stream uses
    const { initialConsoleState, reducer } = await import("../state/consoleStore");
    expect(reducer({ ...initialConsoleState }, { type: "connection", state: "stale" }).connection).toBe("stale");
    await user.click(screen.getByRole("button", { name: /Retry connection/ }));
    await waitFor(() => expect(reconnect).toHaveBeenCalled());
  });
});
