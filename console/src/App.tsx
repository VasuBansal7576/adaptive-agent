import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { createRestTransport } from "./api/rest";
import { SIMULATION_LABEL, createSimulationTransport } from "./api/simulation";
import type { ConsoleTransport } from "./api/transport";
import { initialConsoleState, reducer } from "./state/consoleStore";
import { StatusBadge } from "./components/StatusBadge";
import { Banner } from "./components/ui";
import { RegistryView } from "./views/RegistryView";
import { RunView } from "./views/RunView";
import { SkillsView } from "./views/SkillsView";
import { CandidatesView } from "./views/CandidatesView";

const TABS = [
  { id: "registry", label: "Environment registry" },
  { id: "runs", label: "Runs" },
  { id: "skills", label: "Skill library" },
  { id: "candidates", label: "Candidates" },
] as const;

type TabId = (typeof TABS)[number]["id"];

/** Dev-only simulation opt-in: `?sim=1` in a dev build, or the explicit toggle. */
function simulationOptedIn(): boolean {
  return import.meta.env.DEV && new URLSearchParams(window.location.search).has("sim");
}

/** Stream badge copy that never implies model inference. A green "live" badge
 *  only means the event stream is connected — and in simulation mode it never
 *  shows as live inference at all. */
function streamBadge(mode: "simulation" | "live", connection: string): { status: string } {
  if (mode === "simulation") {
    if (connection === "live") return { status: "sim_stream_connected" };
    if (connection === "stale") return { status: "sim_stream_stale" };
    if (connection === "reconnecting") return { status: "sim_stream_reconnecting" };
    if (connection === "closed") return { status: "sim_stream_closed" };
    if (connection === "disconnected") return { status: "disconnected" };
    return { status: "sim_stream_connecting" };
  }
  return { status: connection };
}

export function App({ transport: transportProp }: { transport?: ConsoleTransport }) {
  const [transport, setTransport] = useState<ConsoleTransport>(() => {
    if (transportProp) return transportProp;
    return simulationOptedIn() ? createSimulationTransport() : createRestTransport();
  });
  const [state, dispatch] = useReducer(reducer, {
    ...initialConsoleState,
    transportMode: transport.mode,
  });
  const [activeTab, setActiveTab] = useState<TabId>("runs");
  const [streamNonce, setStreamNonce] = useState(0);
  const [learningRequest, setLearningRequest] = useState<{ runId: string } | null>(null);
  const closeStreamRef = useRef<(() => void) | null>(null);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const retryConnection = async () => {
    dispatch({ type: "loadStart" });
    try {
      await transport.reconnect();
    } catch {
      /* the load below renders the disconnected state again */
    }
    void load();
  };

  const load = useCallback(async () => {
    dispatch({ type: "loadStart" });
    try {
      const [environments, runs, skills, candidates] = await Promise.all([
        transport.listEnvironments(),
        transport.listRuns(),
        transport.listSkills(),
        transport.listCandidates(),
      ]);
      dispatch({ type: "loadOk", environments, runs, skills, candidates });
    } catch (error) {
      dispatch({ type: "loadError", message: error instanceof Error ? error.message : "Unknown load failure" });
      // never leave the header spinner in an indefinite Connecting state
      if (transport.mode === "live") dispatch({ type: "connection", state: "disconnected" });
    }
  }, [transport]);

  // load (and reload) whenever the transport changes; the reducer's transport
  // action clears ALL prior-mode state (runs, events, candidates, cursors)
  useEffect(() => {
    closeStreamRef.current?.();
    closeStreamRef.current = null;
    dispatch({ type: "transport", mode: transport.mode });
    void load();
    return () => closeStreamRef.current?.();
  }, [load]);

  // stream lifecycle for the selected run: open from the acknowledged cursor,
  // close when the selection or transport changes; reconnectNonce forces a
  // manual stream reopen (stale banner) while preserving the cursor
  const lastRecordRefresh = useRef(0);
  const acknowledgedRef = useRef(0);
  const refreshCandidates = useCallback(async () => {
    try {
      const candidates = await transport.listCandidates();
      dispatch({ type: "candidatesRefreshed", candidates });
    } catch {
      /* load/stale paths surface connection issues */
    }
  }, [transport]);

  const refreshRunRecords = useCallback(async (force = false) => {
    // throttle background record refreshes; forced refreshes (tab open, user
    // intent) always run
    if (!force && Date.now() - lastRecordRefresh.current < 2000) return;
    lastRecordRefresh.current = Date.now();
    try {
      const runs = await transport.listRuns();
      dispatch({ type: "runsRefreshed", runs });
    } catch {
      /* keep the current view; the stale/retry paths surface connection issues */
    }
  }, [transport]);

  useEffect(() => {
    // after a stream closes (terminal or restart), refresh the actual records
    if (state.connection === "closed") void refreshRunRecords();
  }, [state.connection, refreshRunRecords]);

  useEffect(() => {
    closeStreamRef.current?.();
    closeStreamRef.current = null;
    if (!state.selectedRunId) return;
    const runId = state.selectedRunId;
    const cursor = state.cursors[runId] ?? 0;
    dispatch({ type: "connection", state: "connecting" });
    // acknowledged cursor + burst-deduped refresh schedulers, local to this stream
    acknowledgedRef.current = cursor;
    const pendingRefresh = { timer: null as ReturnType<typeof setTimeout> | null };
    const pendingTerminal = { timer: null as ReturnType<typeof setTimeout> | null };
    const scheduleRecordRefresh = () => {
      if (pendingRefresh.timer) return;
      pendingRefresh.timer = setTimeout(() => {
        pendingRefresh.timer = null;
        void refreshRunRecords(true);
      }, 300);
    };
    const scheduleTerminalRefresh = () => {
      if (pendingTerminal.timer) return;
      pendingTerminal.timer = setTimeout(() => {
        pendingTerminal.timer = null;
        void refreshRunRecords(true);
      }, 1200);
    };
    const close = transport.openRunStream(runId, cursor, {
      onEvent: (event) => {
        // acknowledged cursor local to THIS stream: updated per event so gap
        // detection never reads a stale closure and never forces per-event
        // authoritative GETs
        const gapSkipped = event.sequence > acknowledgedRef.current + 1;
        acknowledgedRef.current = Math.max(acknowledgedRef.current, event.sequence);
        dispatch({ type: "event", event });
        // refresh the authoritative record when the event carries no validated
        // status (bare status/outcome rows) or when plane-shape status/approval
        // events arrive — never parse display text. Bursts (gaps, bare rows)
        // dedupe into one scheduled refresh.
        if (gapSkipped || event.needsRecordRefresh || (!event.runStatus && (event.kind === "status" || event.kind === "approval"))) {
          scheduleRecordRefresh();
        }
        // a terminal lifecycle event schedules ONE delayed authoritative
        // refresh (the private outcome commits server-side around this
        // transition). Normal terminal EOF closes the stream — no manual close
        // here so later evidence always drains first; timers dedupe across
        // replayed historical terminal events.
        const TERMINAL_STATES = ["succeeded", "failed", "cancelled", "timed_out"];
        if (event.runStatus && TERMINAL_STATES.includes(event.runStatus)) {
          scheduleTerminalRefresh();
        }
      },
      onState: (connection) => {
        // record every state the transport reports, including a genuine
        // terminal closed (EOF after a terminal run); teardown cleanup no
        // longer emits closed, so nothing is mislabeled
        dispatch({ type: "connection", state: connection });
      },
    });
    closeStreamRef.current = close;
    return close;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.selectedRunId, transport, state.loading, streamNonce]);

  useEffect(() => {
    if (activeTab === "candidates") void refreshRunRecords(true);
  }, [activeTab, refreshRunRecords]);

  const manualReconnect = async () => {
    try {
      await transport.reconnect();
    } catch {
      /* reconnect surfaces its own state; the stream reopen below still runs */
    }
    setStreamNonce((n) => n + 1);
  };

  const failAction = (error: unknown) => {
    const correlationId = (error as { correlationId?: string } | null)?.correlationId ?? null;
    const message = error instanceof Error ? error.message : "Unknown action failure";
    dispatch({ type: "actionError", message, correlationId });
  };

  const cancelRun = async (runId: string) => {
    try {
      await transport.cancelRun(runId);
      const run = state.runs.find((r) => r.runId === runId);
      if (run) dispatch({ type: "runUpdated", run: { ...run, status: "cancelled" } });
    } catch (error) {
      failAction(error); // visible, with correlation id when the API provides one
    }
  };

  const createRun = async (input: Parameters<ConsoleTransport["createRun"]>[0]) => {
    try {
      // transport-level createRun includes the launch step so runs never stay queued
      const run = await transport.createRun(input);
      dispatch({ type: "runAdded", run });
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown create failure";
      const correlationId = (error as { correlationId?: string } | null)?.correlationId ?? null;
      // honest uncertainty: the POST may have been accepted before the failure
      dispatch({
        type: "actionError",
        message: `Run creation did not complete: ${message}`,
        correlationId,
        mayHaveCommitted: true,
      });
      // refresh the list so a run the API accepted becomes visible
      void load();
      return false;
    }
  };

  const onStreamBadge = streamBadge(transport.mode, state.connection);

  const onTabKeyDown = (event: React.KeyboardEvent, index: number) => {
    let next: number | null = null;
    if (event.key === "ArrowRight") next = (index + 1) % TABS.length;
    if (event.key === "ArrowLeft") next = (index - 1 + TABS.length) % TABS.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = TABS.length - 1;
    if (next !== null) {
      event.preventDefault();
      tabRefs.current[next]?.focus();
      selectTab(TABS[next].id);
    }
  };

  const selectTab = (id: TabId) => {
    setActiveTab(id);
    // narrow viewports scroll the tab row horizontally: keep the active tab visible
    const index = TABS.findIndex((t) => t.id === id);
    requestAnimationFrame(() => tabRefs.current[index]?.scrollIntoView({ block: "nearest", inline: "nearest" }));
  };

  const switchTransport = () => {
    if (transport.mode === "simulation") {
      // leaving the dev fixture: URL opt-out keeps reloads consistent
      if (import.meta.env.DEV && window.location.search.includes("sim=1")) {
        const url = new URL(window.location.href);
        url.searchParams.delete("sim");
        window.history.replaceState(null, "", url);
      }
      setTransport(createRestTransport());
    } else if (import.meta.env.DEV) {
      // simulation is a dev-only, explicit opt-in
      const url = new URL(window.location.href);
      url.searchParams.set("sim", "1");
      window.history.replaceState(null, "", url);
      setTransport(createSimulationTransport());
    }
  };

  const showSimToggle = transport.mode === "simulation" || import.meta.env.DEV;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      <header className="border-b border-slate-800 bg-slate-900/80">
        <div className="mx-auto max-w-7xl px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-lg font-semibold text-slate-100">Adaptive Agent — Operator Console</h1>
              <p className="text-xs text-slate-500">Runs, evidence, skills, and promotion under operator authority</p>
            </div>
            <div className="flex items-center gap-2">
              {/* badge text states stream connectivity only — never model inference */}
              <span title="Event stream connectivity; does not indicate model inference">
                <StatusBadge status={onStreamBadge.status} />
              </span>
              {showSimToggle && (
                <button
                  type="button"
                  onClick={switchTransport}
                  className="rounded-md border border-slate-600 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800"
                  aria-label={transport.mode === "simulation" ? "Switch to live control API" : "Switch to simulation fixtures (development only)"}
                >
                  {transport.mode === "simulation" ? "Use live API" : "Use simulation (dev)"}
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {transport.mode === "simulation" && (
        <div className="border-b border-amber-800 bg-amber-950/70">
          <p className="mx-auto max-w-7xl px-4 py-2 text-xs font-semibold tracking-wide text-amber-300 sm:px-6 lg:px-8" role="status">
            ⚠ {SIMULATION_LABEL}
          </p>
        </div>
      )}

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        {state.actionError && (
          <div className="mb-4">
            <Banner tone="bad" title="Action failed" role="alert">
              {state.actionError.message}
              {state.actionError.correlationId ? ` — correlation ${state.actionError.correlationId}` : ""}
              {state.actionError.mayHaveCommitted
                ? " The API may have accepted this change before the failure; any form choices are preserved and the operation key is reused on retry."
                : ""}
              <button
                type="button"
                onClick={() => dispatch({ type: "actionErrorCleared" })}
                className="ml-2 underline underline-offset-2"
              >
                Dismiss
              </button>
            </Banner>
          </div>
        )}
        {state.loadError && (
          <div className="mb-4">
            <Banner tone="bad" title="Console cannot refresh data" role="alert">
              {state.loadError} — this is a display/refresh failure; changes already accepted by the API are not
              undone.{" "}
              <button type="button" onClick={() => void retryConnection()} className="underline underline-offset-2">
                Retry connection
              </button>
            </Banner>
          </div>
        )}

        <div role="tablist" aria-label="Console sections" className="mb-5 flex gap-1 overflow-x-auto border-b border-slate-800">
          {TABS.map((tab, index) => (
            <button
              key={tab.id}
              ref={(el) => {
                tabRefs.current[index] = el;
              }}
              role="tab"
              id={`tab-${tab.id}`}
              aria-selected={activeTab === tab.id}
              aria-controls={`panel-${tab.id}`}
              tabIndex={activeTab === tab.id ? 0 : -1}
              onKeyDown={(e) => onTabKeyDown(e, index)}
              onClick={() => selectTab(tab.id)}
              className={`whitespace-nowrap border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? "border-sky-500 text-sky-300"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div role="tabpanel" id={`panel-${activeTab}`} aria-labelledby={`tab-${activeTab}`} tabIndex={-1}>
          {activeTab === "registry" && (
            <RegistryView
              transport={transport}
              environments={state.environments}
              loading={state.loading}
              onRegistered={(environment) => dispatch({ type: "environmentAdded", environment })}
              onActionError={(message, correlationId) => dispatch({ type: "actionError", message, correlationId })}
            />
          )}
          {activeTab === "runs" && (
            <RunView
              transport={transport}
              runs={state.runs}
              events={state.events}
              cursor={state.cursors[state.selectedRunId ?? ""] ?? 0}
              connection={state.connection}
              selectedRunId={state.selectedRunId}
              loading={state.loading}
              environments={state.environments}
              onSelectRun={(runId) => dispatch({ type: "selectRun", runId })}
              onCancel={(run) => void cancelRun(run.runId)}
              onCreateRun={createRun}
              onLearnFromRun={(runId) => {
                // the invoking control unmounts when the tab switches: mark it
                // so the dialog can return focus here on close
                const invoker = document.activeElement;
                if (invoker instanceof HTMLElement) invoker.setAttribute("data-acc010-focus-return", "");
                setLearningRequest({ runId });
                setActiveTab("candidates");
              }}
              onActionError={(message, correlationId) => dispatch({ type: "actionError", message, correlationId })}
              onReconnect={() => void manualReconnect()}
            />
          )}
          {activeTab === "skills" && <SkillsView skills={state.skills} loading={state.loading} />}
          {activeTab === "candidates" && (
            <CandidatesView
              transport={transport}
              candidates={state.candidates}
              runs={state.runs}
              loading={state.loading}
              onActionError={(message, correlationId) => dispatch({ type: "actionError", message, correlationId })}
              onRefreshCandidates={() => void refreshCandidates()}
              onRefreshRuns={() => void refreshRunRecords(true)}
              learningRequest={learningRequest}
              onLearningRequestConsumed={() => setLearningRequest(null)}
            />
          )}
        </div>
      </main>

    </div>
  );
}
