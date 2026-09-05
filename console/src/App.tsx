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
  const closeStreamRef = useRef<(() => void) | null>(null);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

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
  // close when the selection or transport changes
  useEffect(() => {
    closeStreamRef.current?.();
    closeStreamRef.current = null;
    if (!state.selectedRunId) return;
    const runId = state.selectedRunId;
    const cursor = state.cursors[runId] ?? 0;
    dispatch({ type: "connection", state: "connecting" });
    const close = transport.openRunStream(runId, cursor, {
      onEvent: (event) => dispatch({ type: "event", event }),
      onState: (connection) => {
        if (connection !== "closed") dispatch({ type: "connection", state: connection });
      },
    });
    closeStreamRef.current = close;
    return close;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.selectedRunId, transport, state.loading]);

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
      const run = await transport.createRun(input);
      dispatch({ type: "runAdded", run });
      // explicit launch step (POST /runs/{id}/launch)
      await transport.launchRun(run.runId);
    } catch (error) {
      failAction(error);
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
      setActiveTab(TABS[next].id);
    }
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
            <Banner tone="bad" title="Failed to load console data" role="alert">
              {state.loadError} — the console keeps your current view.{" "}
              <button type="button" onClick={() => void load()} className="underline underline-offset-2">
                Retry
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
              onClick={() => setActiveTab(tab.id)}
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
              onCreateRun={(input) => void createRun(input)}
              onActionError={(message, correlationId) => dispatch({ type: "actionError", message, correlationId })}
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
            />
          )}
        </div>
      </main>

      <footer className="border-t border-slate-800 py-4 text-center text-xs text-slate-600">
        Operator actions are audited. The learner never receives the operator token. Evidence renders as escaped text.
      </footer>
    </div>
  );
}
