import type { ApprovalRequest, CandidateDiff, EnvironmentPackageSummary, RunEvent, RunRecord, SkillVersionSummary } from "../api/types";
import type { ConsoleTransport } from "../api/transport";

export type ConnectionState = "connecting" | "live" | "stale" | "reconnecting" | "disconnected" | "closed";

export type ConsoleState = {
  transportMode: "simulation" | "live";
  loading: boolean;
  loadError: string | null;
  /** last failed operator action, surfaced until dismissed or retried */
  actionError: { message: string; correlationId: string | null; mayHaveCommitted?: boolean } | null;
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
  actionError: null,
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
  | { type: "runUpdated"; run: RunRecord }
  | { type: "runAdded"; run: RunRecord }
  | { type: "environmentAdded"; environment: EnvironmentPackageSummary }
  | { type: "candidateAdded"; candidate: CandidateDiff }
  | { type: "actionError"; message: string; correlationId?: string | null; mayHaveCommitted?: boolean }
  | { type: "actionErrorCleared" };

/**
 * Apply an event with cursor semantics:
 * - sequence <= cursor: duplicate (replay after reconnect) -> dropped
 * - sequence == cursor + 1: applied, cursor advances
 * - sequence > cursor + 1: gap -> request reconnection from cursor; the
 *   out-of-order event is dropped and the stream restarts from the
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

/** Reset ALL per-mode state. Switching transports must never leak runs, events,
 *  candidates, or cursors from the previous mode (ghost data). */
function resetForMode(mode: ConsoleState["transportMode"]): ConsoleState {
  return { ...initialConsoleState, transportMode: mode };
}

export function reducer(state: ConsoleState, action: ConsoleAction): ConsoleState {
  switch (action.type) {
    case "transport":
      return resetForMode(action.mode);
    case "loadStart":
      // keep actionError visible across background refreshes; it is dismissed
      // explicitly. A refresh with existing data stays silent so open dialogs
      // and their state are never torn down mid-flow.
      return { ...state, loading: state.runs.length === 0, loadError: null };
    case "loadOk": {
      // the selected run must exist in the CURRENT mode's runs; never carry a
      // stale selection (and its events) across a mode boundary
      const selection = action.runs.some((r) => r.runId === state.selectedRunId)
        ? state.selectedRunId
        : action.runs[0]?.runId ?? null;
      return {
        ...state,
        loading: false,
        loadError: null,
        environments: action.environments,
        runs: action.runs,
        skills: action.skills,
        candidates: action.candidates,
        selectedRunId: selection,
        // with no runs to stream, the transport itself is what the badge reports:
        // successful load means connected, never an indefinite Connecting state
        connection: action.runs.length === 0 ? "live" : state.connection,
      };
    }
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
    case "runAdded":
      return {
        ...state,
        runs: [action.run, ...state.runs.filter((r) => r.runId !== action.run.runId)],
        selectedRunId: action.run.runId,
        actionError: null,
      };
    case "environmentAdded":
      return {
        ...state,
        environments: [
          ...state.environments.filter((e) => e.environmentId !== action.environment.environmentId),
          action.environment,
        ],
        actionError: null,
      };
    case "candidateAdded":
      return {
        ...state,
        candidates: [action.candidate, ...state.candidates.filter((c) => c.candidateId !== action.candidate.candidateId)],
        actionError: null,
      };
    case "actionError":
      return {
        ...state,
        actionError: {
          message: action.message,
          correlationId: action.correlationId ?? null,
          mayHaveCommitted: action.mayHaveCommitted ?? false,
        },
      };
    case "actionErrorCleared":
      return { ...state, actionError: null };
    default:
      return state;
  }
}

/** Test helper: replay a batch of events and assert dedup behavior. */
export function replayEvents(state: ConsoleState, events: RunEvent[]): ConsoleState {
  return events.reduce<ConsoleState>((acc, event) => reducer(acc, { type: "event", event }), state);
}
