import { useEffect, useRef, useState } from "react";
import type { ConsoleTransport } from "../api/transport";
import { describeToolError } from "../api/errors";
import type { ApprovalRequest, RunEvent, RunRecord } from "../api/types";
import type { ConnectionState } from "../state/consoleStore";
import { StatusBadge } from "../components/StatusBadge";
import { Banner, EmptyState, LoadingState, Modal } from "../components/ui";

const KIND_LABEL: Record<RunEvent["kind"], string> = {
  status: "Status",
  step: "Step",
  tool: "Tool",
  evidence: "Evidence",
  budget: "Budget",
  approval: "Approval",
};

export function RunView({
  transport,
  runs,
  events,
  cursor,
  connection,
  selectedRunId,
  loading,
  onSelectRun,
  onCancel,
}: {
  transport: ConsoleTransport;
  runs: RunRecord[];
  events: Record<string, RunEvent[]>;
  cursor: number;
  connection: ConnectionState;
  selectedRunId: string | null;
  loading: boolean;
  onSelectRun: (runId: string) => void;
  onCancel: (run: RunRecord) => void;
}) {
  const selected = runs.find((r) => r.runId === selectedRunId) ?? null;
  const [approval, setApproval] = useState<{ request: ApprovalRequest } | null>(null);
  const approvalShown = useRef<string | null>(null);

  // surface the latest pending approval of the selected run as a dialog
  useEffect(() => {
    if (!selected) return;
    const runEvents = events[selected.runId] ?? [];
    const pending = [...runEvents].reverse().find((e) => e.approval && e.approval.approvalId !== approvalShown.current);
    if (pending?.approval && selected.status === "awaiting_approval") {
      approvalShown.current = pending.approval.approvalId;
      setApproval({ request: pending.approval });
    }
  }, [selected, events]);

  if (loading) return <LoadingState label="Loading runs…" />;
  if (runs.length === 0)
    return (
      <EmptyState title="No runs yet">
        Create a run by choosing a registered environment, a goal, a model profile, and a resource budget. The
        active skill version is pinned by the server.
      </EmptyState>
    );

  const runEvents = selected ? events[selected.runId] ?? [] : [];
  const lastError = [...runEvents].reverse().find((e) => e.error);

  return (
    <section aria-labelledby="runs-heading" className="grid gap-6 lg:grid-cols-[minmax(280px,380px)_1fr]">
      <div>
        <h2 id="runs-heading" className="text-base font-semibold text-slate-100">
          Runs
        </h2>
        <ul className="mt-3 space-y-2">
          {runs.map((run) => (
            <li key={run.runId}>
              <button
                type="button"
                onClick={() => onSelectRun(run.runId)}
                aria-current={run.runId === selectedRunId ? "true" : undefined}
                className={`w-full rounded-lg border px-3 py-2.5 text-left text-sm transition-colors ${
                  run.runId === selectedRunId
                    ? "border-sky-600 bg-sky-950/40 text-slate-100"
                    : "border-slate-700 bg-slate-900/60 text-slate-300 hover:border-slate-500"
                }`}
              >
                <span className="flex items-center justify-between gap-2">
                  <span className="font-mono text-[13px]">{run.runId}</span>
                  <StatusBadge status={run.status} />
                </span>
                <span className="mt-1 block truncate text-xs text-slate-400">{run.environmentId} — {run.goal}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="min-w-0">
        {selected ? (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <div className="min-w-0">
                <h3 className="font-mono text-sm text-slate-100">{selected.runId}</h3>
                <p className="mt-0.5 text-[13px] text-slate-400">{selected.goal}</p>
                <p className="mt-1 text-xs text-slate-500">
                  bundle {selected.skillBundleRef.id} v{selected.skillBundleRef.version} · env {selected.environmentRef.id} v{selected.environmentRef.version}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <StatusBadge status={selected.status} />
                {selected.status === "queued" || selected.status === "running" || selected.status === "awaiting_approval" ? (
                  <button
                    type="button"
                    onClick={() => onCancel(selected)}
                    className="rounded-md border border-rose-700 px-3 py-1.5 text-xs font-medium text-rose-300 hover:bg-rose-950/60"
                  >
                    Cancel run
                  </button>
                ) : null}
              </div>
            </div>

            {connection === "stale" && (
              <Banner tone="warn" title="Event stream stale — reconnecting from the last acknowledged cursor" role="alert">
                Status shown may be behind. Events resume from cursor {cursor}; no duplicates will be introduced.
              </Banner>
            )}
            {connection === "reconnecting" && (
              <Banner tone="info" title="Reconnecting to event stream…" />
            )}

            {lastError?.error && (
              <Banner tone={lastError.error.code === "OUTCOME_UNKNOWN" ? "warn" : "bad"} title={`Tool error: ${lastError.error.code}`} role="alert">
                {describeToolError(lastError.error)}
              </Banner>
            )}

            {selected.status === "failed" && !lastError?.error && (
              <Banner tone="bad" title="Run failed">
                The trusted outcome check did not pass. Inspect the evidence below; the active version is unchanged.
              </Banner>
            )}
            {selected.status === "timed_out" && (
              <Banner tone="warn" title="Run timed out">
                An external effect may be unresolved. The broker will reconcile before further state changes.
              </Banner>
            )}
            {selected.status === "cancelled" && (
              <Banner tone="info" title="Run cancelled">
                Future calls were revoked. Already dispatched external actions were not undone; reconciliation occurs on restart.
              </Banner>
            )}

            {selected.budgetUsed && (
              <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4 text-[13px] text-slate-300">
                <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Budget</h4>
                <div className="mt-2 grid gap-3 sm:grid-cols-2">
                  <BudgetBar label="Tool calls" used={selected.budgetUsed.calls} ceiling={selected.budgetUsed.callsCeiling} unit="calls" />
                  <BudgetBar label="Wall time" used={selected.budgetUsed.wallSeconds} ceiling={selected.budgetUsed.wallCeiling} unit="s" />
                </div>
              </div>
            )}

            <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                Event stream <span className="normal-case tracking-normal">(cursor {cursor})</span>
              </h4>
              {runEvents.length === 0 ? (
                <p className="mt-2 text-[13px] text-slate-400">No events applied yet.</p>
              ) : (
                <ol className="mt-2 space-y-2" aria-label="Run events">
                  {runEvents.map((event) => (
                    <li key={event.sequence} className="flex gap-3 rounded-lg bg-slate-800/60 px-3 py-2">
                      <span className="w-10 shrink-0 font-mono text-xs text-slate-500">#{event.sequence}</span>
                      <div className="min-w-0">
                        <p className="text-[13px] text-slate-200">
                          <span className="mr-2 rounded bg-slate-700 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-slate-300">
                            {KIND_LABEL[event.kind]}
                          </span>
                          {event.summary}
                        </p>
                        {event.error && (
                          <p className="mt-1 text-xs text-rose-300">
                            {event.error.code} · retry: {event.error.retry} · correlation {event.error.correlationId}
                          </p>
                        )}
                        {event.detail && <p className="mt-1 text-xs text-slate-400">{event.detail}</p>}
                        <p className="mt-0.5 font-mono text-[10px] text-slate-600">{event.at}</p>
                      </div>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          </div>
        ) : (
          <EmptyState title="Select a run" />
        )}
      </div>

      <ApprovalDialog
        request={approval?.request ?? null}
        transport={transport}
        runId={selected?.runId ?? ""}
        onDecided={() => setApproval(null)}
      />
    </section>
  );
}

function BudgetBar({ label, used, ceiling, unit }: { label: string; used: number; ceiling: number; unit: string }) {
  const pct = Math.min(100, Math.round((used / ceiling) * 100));
  return (
    <div>
      <div className="flex items-center justify-between">
        <span>{label}</span>
        <span className="text-slate-400">
          {used}/{ceiling} {unit} ({pct}%)
        </span>
      </div>
      <div
        role="meter"
        aria-valuenow={used}
        aria-valuemin={0}
        aria-valuemax={ceiling}
        aria-label={`${label} used: ${used} of ${ceiling} ${unit}`}
        className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-700"
      >
        <div className={`h-full ${pct > 90 ? "bg-rose-500" : "bg-sky-500"}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ApprovalDialog({
  request,
  transport,
  runId,
  onDecided,
}: {
  request: ApprovalRequest | null;
  transport: ConsoleTransport;
  runId: string;
  onDecided: () => void;
}) {
  const [busy, setBusy] = useState(false);
  if (!request) return null;
  const decide = async (approve: boolean) => {
    setBusy(true);
    try {
      await transport.submitApproval(runId, request.approvalId, approve);
    } finally {
      setBusy(false);
      onDecided();
    }
  };
  return (
    <Modal open title="Approval required" onClose={() => onDecided()}>
      <div className="space-y-3 text-sm text-slate-300">
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[13px]">
          <dt className="text-slate-500">Tool</dt>
          <dd className="font-mono">{request.tool} <span className="text-slate-500">v{request.toolVersion}</span></dd>
          <dt className="text-slate-500">Effect</dt>
          <dd>
            <StatusBadge status={request.effect === "write" ? "awaiting_approval" : "validated"} /> {request.effect}
          </dd>
          <dt className="text-slate-500">Scope</dt>
          <dd className="font-mono text-[12px]">{request.resourceScope}</dd>
          <dt className="text-slate-500">Expires</dt>
          <dd className="font-mono text-[12px]">{request.expiresAt}</dd>
        </dl>
        <div>
          <p className="text-xs font-medium text-slate-400">Exact canonical arguments</p>
          <pre className="mt-1 max-h-48 overflow-auto rounded-lg bg-slate-800 p-3 font-mono text-xs text-slate-200">
            {request.canonicalArguments}
          </pre>
        </div>
        <div className="flex justify-end gap-2 pt-1">
          <button
            type="button"
            onClick={() => decide(false)}
            disabled={busy}
            className="rounded-md border border-slate-600 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800"
          >
            Deny
          </button>
          <button
            type="button"
            onClick={() => decide(true)}
            disabled={busy}
            className="rounded-md bg-emerald-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-60"
          >
            {busy ? "Submitting…" : "Approve"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
