import type { ApprovalRequest, CandidateDiff, EnvironmentPackageSummary, RunEvent, RunRecord, SkillVersionSummary } from "../api/types";
import type { ConsoleTransport } from "../api/transport";

export type ConnectionState = "connecting" | "live" | "stale" | "reconnecting" | "closed";

export type ConsoleState = {
  transportMode: "simulation" | "live";
  loading: boolean;
  loadError: string | null;
  environments: EnvironmentPackageSummary[];
  runs: RunRecord[];
  skills: SkillVersionSummary[];
  candidates: CandidateDiff[];
  selectedRunId: string | null;
  /** acknowledged cursor per run: highest contiguous applied sequence */
  cursors: Record<string, number>;
  /** applied events keyed by runId, ordered by sequence */
  events: Record<string, RunEvent[]>;
  connection: ConnectionState;
};

export const initialConsoleState: ConsoleState = {
  transportMode: "simulation",
  loading: true,
  loadError: null,
  environments: [],
  runs: [],
  skills: [],
  candidates: [],
  selectedRunId: null,
  cursors: {},
  events: {},
  connection: "connecting",
};

export type ConsoleAction =
  | { type: "transport"; mode: "simulation" | "live" }
  | { type: "loadStart" }
  | { type: "loadOk"; environments: EnvironmentPackageSummary[]; runs: RunRecord[]; skills: SkillVersionSummary[]; candidates: CandidateDiff[] }
  | { type: "loadError"; message: string }
  | { type: "selectRun"; runId: string | null }
  | { type: "connection"; state: ConnectionState }
  | { type: "event"; event: RunEvent }
  | { type: "runUpdated"; run: RunRecord };

/**
 * Apply an event with cursor semantics:
 * - sequence <= cursor: duplicate (replay after reconnect) -> dropped
 * - sequence == cursor + 1: applied, cursor advances
 * - sequence > cursor + 1: gap -> buffered? No: we request reconnection from
 *   cursor; out-of-order events are dropped and the stream restarts from the
 *   acknowledged cursor, guaranteeing no duplicate application.
 */
export function applyEvent(state: ConsoleState, event: RunEvent): ConsoleState {
  const cursor = state.cursors[event.runId] ?? 0;
  if (event.sequence <= cursor) return state; // duplicate
  if (event.sequence > cursor + 1) {
    // gap: treat as stale, expect transport to reconnect from cursor
    return { ...state, connection: "stale" };
  }
  const events = state.events[event.runId] ?? [];
  const run = state.runs.find((r) => r.runId === event.runId);
  const runs = run
    ? state.runs.map((r) =>
        r.runId === event.runId
          ? {
              ...r,
              lastEventSequence: Math.max(r.lastEventSequence, event.sequence),
              status: event.kind === "status" ? derivedStatus(event.summary, r.status) : r.status,
            }
          : r,
      )
    : state.runs;
  return {
    ...state,
    runs,
    events: { ...state.events, [event.runId]: [...events, event] },
    cursors: { ...state.cursors, [event.runId]: event.sequence },
  };
}

function derivedStatus(summary: string, current: RunRecord["status"]): RunRecord["status"] {
  const s = summary.toLowerCase();
  if (s.includes("run succeeded")) return "succeeded";
  if (s.includes("run failed")) return "failed";
  if (s.includes("cancelled")) return "cancelled";
  if (s.includes("timed out")) return "timed_out";
  if (s.includes("approval required")) return "awaiting_approval";
  if (s.includes("queued")) return current === "queued" ? "queued" : current;
  return current;
}

export function reducer(state: ConsoleState, action: ConsoleAction): ConsoleState {
  switch (action.type) {
    case "transport":
      return { ...initialConsoleState, transportMode: action.mode };
    case "loadStart":
      return { ...state, loading: true, loadError: null };
    case "loadOk":
      return {
        ...state,
        loading: false,
        loadError: null,
        environments: action.environments,
        runs: action.runs,
        skills: action.skills,
        candidates: action.candidates,
        selectedRunId: state.selectedRunId ?? action.runs[0]?.runId ?? null,
      };
    case "loadError":
      return { ...state, loading: false, loadError: action.message };
    case "selectRun":
      return { ...state, selectedRunId: action.runId };
    case "connection":
      return { ...state, connection: action.state };
    case "event":
      return applyEvent(state, action.event);
    case "runUpdated":
      return { ...state, runs: state.runs.map((r) => (r.runId === action.run.runId ? action.run : r)) };
    default:
      return state;
  }
}

/** Test helper: replay a batch of events and assert dedup behavior. */
export function replayEvents(state: ConsoleState, events: RunEvent[]): ConsoleState {
  return events.reduce<ConsoleState>((acc, event) => reducer(acc, { type: "event", event }), state);
}
