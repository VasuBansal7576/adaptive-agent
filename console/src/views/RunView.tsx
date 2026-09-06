import { useEffect, useRef, useState } from "react";
import type { ConsoleTransport, CreateRunInput } from "../api/transport";
import { newIdempotencyKey } from "../api/transport";
import { describeToolError } from "../api/errors";
import type { ApprovalRequest, EnvironmentPackageSummary, RunEvent, RunRecord, TaskOption } from "../api/types";
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
  environments,
  onSelectRun,
  onCancel,
  onCreateRun,
  onActionError,
  onReconnect,
}: {
  transport: ConsoleTransport;
  runs: RunRecord[];
  events: Record<string, RunEvent[]>;
  cursor: number;
  connection: ConnectionState;
  selectedRunId: string | null;
  loading: boolean;
  environments: EnvironmentPackageSummary[];
  onSelectRun: (runId: string) => void;
  onCancel: (run: RunRecord) => void;
  onCreateRun: (input: CreateRunInput) => Promise<boolean>;
  onActionError: (message: string, correlationId?: string | null) => void;
  onReconnect: () => void;
}) {
  const selected = runs.find((r) => r.runId === selectedRunId) ?? null;
  const [approval, setApproval] = useState<{ request: ApprovalRequest } | null>(null);
  const [newRunOpen, setNewRunOpen] = useState(false);
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
      <div className="space-y-4">
        <EmptyState title="No runs yet">
          Create a run by choosing a registered environment, a goal, a model profile, and a resource budget. The
          active skill version is pinned by the server.
        </EmptyState>
        <div>
          <button
            type="button"
            onClick={() => setNewRunOpen(true)}
            className="rounded-md bg-sky-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-sky-500"
          >
            New run
          </button>
        </div>
        <NewRunDialog
          open={newRunOpen}
          onClose={() => setNewRunOpen(false)}
          environments={environments}
          transport={transport}
          onCreateRun={onCreateRun}
        />
      </div>
    );

  const runEvents = selected ? events[selected.runId] ?? [] : [];
  const lastError = [...runEvents].reverse().find((e) => e.error);
  // honest failure classification from validated event types: a recorded
  // run_failed lifecycle event is a runtime failure; otherwise the outcome
  // evidence decides — no text heuristics
  const lastRuntimeFailure = [...runEvents].reverse().find((e) => e.lifecycleType === "run_failed");
  const hasOutcomeEvidence = runEvents.some((e) => e.lifecycleType === "outcome_recorded");

  return (
    <section aria-labelledby="runs-heading" className="grid min-w-0 gap-6 lg:grid-cols-[minmax(280px,380px)_minmax(0,1fr)]">
      <div className="min-w-0">
        <div className="flex items-center justify-between gap-2">
          <h2 id="runs-heading" className="text-base font-semibold text-slate-100">
            Runs
          </h2>
          <button
            type="button"
            onClick={() => setNewRunOpen(true)}
            className="rounded-md bg-sky-600 px-3 py-1 text-xs font-medium text-white hover:bg-sky-500"
          >
            New run
          </button>
        </div>
        <ul className="mt-3 space-y-2">
          {runs.map((run) => (
            <li key={run.runId}>
              <button
                type="button"
                onClick={() => onSelectRun(run.runId)}
                aria-current={run.runId === selectedRunId ? "true" : undefined}
                className={`w-full min-w-0 rounded-lg border px-3 py-2.5 text-left text-sm transition-colors ${
                  run.runId === selectedRunId
                    ? "border-sky-600 bg-sky-950/40 text-slate-100"
                    : "border-slate-700 bg-slate-900/60 text-slate-300 hover:border-slate-500"
                }`}
              >
                <span className="flex min-w-0 items-center justify-between gap-2">
                  <span className="min-w-0 break-all font-mono text-[13px]">{run.runId}</span>
                  <span className="shrink-0">
                    <StatusBadge status={run.status} />
                  </span>
                </span>
                <span className="mt-1 block truncate text-xs text-slate-400" title={`${run.environmentId} — ${run.goal}`}>
                  {run.environmentId} — {run.goal}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="min-w-0">
        {selected ? (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-700 bg-slate-900/60 p-4">
              <div className="min-w-0 flex-1">
                <h3 className="break-all font-mono text-sm text-slate-100">{selected.runId}</h3>
                <p className="mt-0.5 text-[13px] text-slate-400 [overflow-wrap:anywhere]">{selected.goal}</p>
                <p className="mt-1 text-xs text-slate-500 [overflow-wrap:anywhere]">
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
                Status shown may be behind. Events resume from cursor {cursor}; no duplicates will be introduced.{" "}
                <button
                  type="button"
                  onClick={() => onReconnect()}
                  className="underline underline-offset-2"
                >
                  Reconnect now
                </button>
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
              lastRuntimeFailure ? (
                <Banner tone="bad" title="Runtime failure" role="alert">
                  The run failed before or during execution (recorded run_failed evidence). The active version is
                  unchanged; inspect the expanded event below for the recorded error.
                </Banner>
              ) : hasOutcomeEvidence ? (
                <Banner tone="bad" title="Run failed" role="alert">
                  The trusted outcome evidence recorded a failure. Inspect the expanded outcome event below; the
                  active version is unchanged.
                </Banner>
              ) : (
                <Banner tone="bad" title="Run failed" role="alert">
                  The run reached a failed state. Inspect the event evidence below for the recorded cause; the
                  active version is unchanged.
                </Banner>
              )
            )}
            {selected.status === "timed_out" && (
              <Banner tone="warn" title="Run timed out">
                An external effect may remain unresolved. Operation-level reconciliation continues without reopening
                the run; a reconciled result cannot trigger a second dispatch.
              </Banner>
            )}
            {selected.status === "cancelled" && (
              <Banner tone="info" title="Run cancelled">
                Future calls were revoked. Already dispatched external actions were not undone; operation-level
                reconciliation continues without reopening the run.
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
                    <li key={event.sequence} className="rounded-lg bg-slate-800/60 px-3 py-2">
                      <details>
                        <summary className="flex cursor-pointer list-none gap-3 [&::-webkit-details-marker]:hidden">
                          <span className="w-10 shrink-0 font-mono text-xs text-slate-500">#{event.sequence}</span>
                          <div className="min-w-0 flex-1">
                            <p className="text-[13px] text-slate-200 [overflow-wrap:anywhere]">
                              <span className="mr-2 rounded bg-slate-700 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-slate-300">
                                {KIND_LABEL[event.kind]}
                              </span>
                              {event.summary}
                              {event.evidence?.evidenceId && (
                                <span className="ml-2 align-middle">
                                  <span aria-hidden="true" className="text-slate-500">▸</span>
                                  <span className="sr-only">— evidence details available, expand to inspect</span>
                                </span>
                              )}
                            </p>
                            {event.error && (
                              <p className="mt-1 break-all text-xs text-rose-300">
                                {event.error.code} · retry: {event.error.retry} · correlation {event.error.correlationId}
                              </p>
                            )}
                            {event.detail && <p className="mt-1 text-xs text-slate-400 [overflow-wrap:anywhere]">{event.detail}</p>}
                            <p className="mt-0.5 font-mono text-[10px] text-slate-600">{event.at}</p>
                          </div>
                        </summary>
                        <div className="mt-2 min-w-0 border-t border-slate-700/60 pt-2 pl-4">
                          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-[11px]">
                            <dt className="text-slate-500">Evidence</dt>
                            <dd className="break-all font-mono text-slate-300">{event.evidence?.evidenceId ?? "—"}</dd>
                            <dt className="text-slate-500">Artifact</dt>
                            <dd className="break-all font-mono text-slate-300">{event.evidence?.sourceRefId ?? "—"}</dd>
                            <dt className="text-slate-500">Trust class</dt>
                            <dd className="text-slate-300">{event.evidence?.trustClass ?? "—"}</dd>
                            <dt className="text-slate-500">Visibility</dt>
                            <dd className="text-slate-300">{event.evidence?.visibility ?? "—"}</dd>
                            <dt className="text-slate-500">Redacted</dt>
                            <dd className="text-slate-300">
                              {event.evidence?.redacted === undefined ? "—" : event.evidence.redacted ? "yes" : "no"}
                            </dd>
                          </dl>
                          {event.evidence?.contentHash && (
                            <p
                              className="mt-1 break-all font-mono text-[10px] text-slate-600"
                              title={`Content fingerprint ${event.evidence.contentHash}`}
                            >
                              fingerprint {event.evidence.contentHash.slice(0, 16)}…
                            </p>
                          )}
                          {event.error && (
                            <p className="mt-1 text-[11px] text-slate-400 [overflow-wrap:anywhere]">
                              {describeToolError(event.error)}
                            </p>
                          )}
                        </div>
                      </details>
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

      <NewRunDialog
        open={newRunOpen}
        onClose={() => setNewRunOpen(false)}
        environments={environments}
        transport={transport}
        onCreateRun={onCreateRun}
      />

      <ApprovalDialog
        request={approval?.request ?? null}
        runId={selected?.runId ?? ""}
        onDecide={async (approve) => {
          if (!approval) return;
          try {
            await transport.submitApproval(selected?.runId ?? "", approval.request.approvalId, approve);
          } catch (error) {
            const message = error instanceof Error ? error.message : "Approval submission failed";
            const correlationId = (error as { correlationId?: string } | null)?.correlationId ?? null;
            onActionError(`Approval failed: ${message}`, correlationId);
          } finally {
            setApproval(null);
          }
        }}
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

/** Fallback model profiles when /run-options is unavailable. The id is the
 *  AUTHORITATIVE trusted-plane model reference ("model-profile"); the Luna
 *  label reflects the configured planner (openai-codex/gpt-5.6-luna). */
export const MODEL_PROFILES = [
  { ref: { id: "model-profile", version: "1", sha256: "" }, label: "GPT-5.6 Luna (subscription)" },
];

const BUDGET_FALLBACK = { modelTokens: 4000, toolCalls: 32, wallTimeSeconds: 90 };

function NewRunDialog({
  open,
  onClose,
  environments,
  transport,
  onCreateRun,
}: {
  open: boolean;
  onClose: () => void;
  environments: EnvironmentPackageSummary[];
  transport: ConsoleTransport;
  onCreateRun: (input: CreateRunInput) => Promise<boolean>;
}) {
  const [goal, setGoal] = useState("");
  const [environmentId, setEnvironmentId] = useState("");
  const [taskId, setTaskId] = useState("");
  const [tasks, setTasks] = useState<TaskOption[]>([]);
  const [modelProfile, setModelProfile] = useState(MODEL_PROFILES[0].label);
  const [modelProfiles, setModelProfiles] = useState<Array<{ ref: { id: string; version: string; sha256: string }; label: string }>>(MODEL_PROFILES);
  const [executionMode, setExecutionMode] = useState<CreateRunInput["executionMode"]>("interactive");
  const [toolCallCeiling, setToolCallCeiling] = useState(String(BUDGET_FALLBACK.toolCalls));
  const [wallSecondsCeiling, setWallSecondsCeiling] = useState(String(BUDGET_FALLBACK.wallTimeSeconds));
  const [modelTokenCeiling, setModelTokenCeiling] = useState(String(BUDGET_FALLBACK.modelTokens));
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  // one idempotency key per dialog session: reused across recovery retries so a
  // transient failure can never produce a duplicate run
  const [idempotencyKey, setIdempotencyKey] = useState(() => newIdempotencyKey());
  const [submitNotice, setSubmitNotice] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const goalRef = useRef<HTMLTextAreaElement>(null);
  // run-options defaults must never clobber operator edits that land while the
  // options request is in flight
  const budgetTouched = useRef(false);

  const selectedEnv = environments.find((e) => e.environmentId === environmentId);
  const selectedTask = tasks.find((t) => t.taskId === taskId);
  // mode support resolution: intersect the selected task's declared modes with
  // the environment summary's modes when both are projected
  const declared = selectedTask?.executionModes ?? selectedEnv?.executionModes ?? ["dry_run", "interactive", "batch", "replay"];
  const envModes = selectedEnv?.executionModes;
  const modeSource = envModes ? declared.filter((m) => envModes.includes(m)) : declared;
  const supportedModes = (modeSource.length > 0 ? modeSource : declared).filter((m): m is CreateRunInput["executionMode"] =>
    ["dry_run", "interactive", "batch", "replay"].includes(m),
  );

  useEffect(() => {
    if (!open) return;
    setEnvironmentId((prev) => prev || environments[0]?.environmentId || "");
    let cancelled = false;
    // authoritative model profiles and bounded budget defaults from the API
    transport
      .getRunOptions()
      .then((options) => {
        if (cancelled) return;
        if (options.modelProfiles.length > 0 && modelProfile === MODEL_PROFILES[0].label) {
          setModelProfiles(options.modelProfiles.map((p) => ({ ref: p.ref, label: p.label })));
          setModelProfile(options.modelProfiles[0].label);
        }
        if (!budgetTouched.current) {
          setToolCallCeiling(String(options.budgetDefaults.toolCalls));
          setWallSecondsCeiling(String(options.budgetDefaults.wallTimeSeconds));
          setModelTokenCeiling(String(options.budgetDefaults.modelTokens));
        }
      })
      .catch(() => {
        /* fallback constants remain; never block the dialog on this */
      });
    // the Modal's initial-focus handler places focus on the first field (the
    // goal textarea); no aggressive polling that could steal focus mid-typing
    return () => {
      cancelled = true;
    };
  }, [open, environments, transport]);

  // registered tasks for the selected environment; durable runs must match one
  useEffect(() => {
    if (!open || !environmentId) return;
    let cancelled = false;
    setTaskId("");
    setTasks([]);
    transport
      .getEnvironmentTasks(environmentId)
      .then((registered) => {
        if (!cancelled) {
          setTasks(registered);
          // registered tasks are required when the environment declares them
          if (registered.length > 0) {
            setTaskId((prev) => prev || registered[0].taskId);
            setGoal((prev) => prev || registered[0].goal);
          }
        }
      })
      .catch(() => {
        /* free-form goal remains available when the environment has no tasks */
      });
    return () => {
      cancelled = true;
    };
  }, [open, environmentId, transport]);

  // keep the chosen mode within the environment's declared modes
  useEffect(() => {
    if (supportedModes.length > 0 && !supportedModes.includes(executionMode)) {
      setExecutionMode(supportedModes.includes("interactive") ? "interactive" : supportedModes[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [environmentId]);

  const validate = (): boolean => {
    const errors: Record<string, string> = {};
    if (!goal.trim()) errors.goal = "Goal is required";
    if (!environmentId) errors.environmentId = "Environment is required";
    if (tasks.length > 0 && !taskId) errors.taskId = "A registered task is required for this environment";
    if (!modelProfile.trim()) errors.modelProfile = "Model profile is required";
    const calls = Number(toolCallCeiling);
    if (!Number.isFinite(calls) || calls <= 0) errors.toolCallCeiling = "Tool-call ceiling must be a positive number";
    const wall = Number(wallSecondsCeiling);
    if (!Number.isFinite(wall) || wall <= 0) errors.wallSecondsCeiling = "Wall-time ceiling must be a positive number";
    const tokens = Number(modelTokenCeiling);
    if (!Number.isFinite(tokens) || tokens <= 0) errors.modelTokenCeiling = "Token budget must be a positive number";
    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;
    setSubmitting(true);
    setSubmitNotice(null);
    const ok = await onCreateRun({
      environmentId,
      goal: goal.trim(),
      taskId: taskId || undefined,
      modelProfileRef: modelProfiles.find((p) => p.label === modelProfile)?.ref,
      modelProfile: modelProfile,
      // same key across recovery retries within this dialog session
      idempotencyKey,
      budget: {
        toolCallCeiling: Number(toolCallCeiling),
        wallSecondsCeiling: Number(wallSecondsCeiling),
        modelTokenCeiling: Number(modelTokenCeiling),
      },
      executionMode,
    });
    setSubmitting(false);
    if (ok) {
      // success: fresh key for the next dialog session
      setGoal("");
      setModelTokenCeiling("20000");
      setFieldErrors({});
      setSubmitNotice(null);
      setIdempotencyKey(newIdempotencyKey());
      onClose();
    } else {
      // transient failure: every choice is preserved; the same key is reused
      setSubmitNotice(
        "Run creation did not complete. Your goal, environment, and budget choices are preserved. The API may have accepted the run; retrying reuses the same idempotency key and cannot create a duplicate.",
      );
    }
  };

  return (
    <Modal open={open} title="Create run" onClose={onClose}>
      <form onSubmit={submit} noValidate className="space-y-3 text-sm text-slate-300">
        <div>
          <label htmlFor="newrun-goal" className="block text-xs font-medium text-slate-400">Goal</label>
          <textarea
            id="newrun-goal"
            ref={goalRef}
            rows={2}
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            aria-invalid={fieldErrors.goal ? true : undefined}
            aria-describedby={fieldErrors.goal ? "newrun-goal-error" : undefined}
            className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100"
            placeholder="e.g., Reconcile the Q3 ledger batch"
          />
          {fieldErrors.goal && <p id="newrun-goal-error" role="alert" className="mt-1 text-xs text-rose-400">{fieldErrors.goal}</p>}
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div>
            <label htmlFor="newrun-env" className="block text-xs font-medium text-slate-400">Environment</label>
            <select
              id="newrun-env"
              value={environmentId}
              onChange={(e) => setEnvironmentId(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100"
            >
              <option value="">Select…</option>
              {environments.map((env) => (
                <option key={env.environmentId} value={env.environmentId}>
                  {env.environmentId} v{env.version}
                </option>
              ))}
            </select>
            {fieldErrors.environmentId && <p role="alert" className="mt-1 text-xs text-rose-400">{fieldErrors.environmentId}</p>}
          </div>
          <div>
            <label htmlFor="newrun-model" className="block text-xs font-medium text-slate-400">Model</label>
            <select
              id="newrun-model"
              value={modelProfile}
              onChange={(e) => setModelProfile(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100"
            >
              {modelProfiles.map((profile) => (
                <option key={profile.label} value={profile.label}>{profile.label}</option>
              ))}
            </select>
          </div>
        </div>
        {tasks.length > 0 && (
          <div>
            <label htmlFor="newrun-task" className="block text-xs font-medium text-slate-400">Registered task (required)</label>
            <select
              id="newrun-task"
              value={taskId}
              onChange={(e) => {
                setTaskId(e.target.value);
                const chosen = tasks.find((t) => t.taskId === e.target.value);
                if (chosen) setGoal(chosen.goal);
              }}
              aria-describedby={taskId ? undefined : "newrun-task-hint"}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100"
            >
              {tasks.map((task) => (
                <option key={task.taskId} value={task.taskId}>{task.goal}</option>
              ))}
            </select>
            {/* free-form goals are not accepted by the durable runtime yet */}
            {fieldErrors.taskId && <p role="alert" className="mt-1 text-xs text-rose-400">{fieldErrors.taskId}</p>}
            <p id="newrun-task-hint" className="mt-1 text-[11px] text-slate-500">
              Select a registered task — this environment accepts registered task goals only.
            </p>
          </div>
        )}
        <div className="grid gap-3 sm:grid-cols-3">
          <div>
            <label htmlFor="newrun-calls" className="block text-xs font-medium text-slate-400">Tool-call limit</label>
            <input id="newrun-calls" type="number" min="1" value={toolCallCeiling} onChange={(e) => { budgetTouched.current = true; setToolCallCeiling(e.target.value); }}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100" />
            {fieldErrors.toolCallCeiling && <p role="alert" className="mt-1 text-xs text-rose-400">{fieldErrors.toolCallCeiling}</p>}
          </div>
          <div>
            <label htmlFor="newrun-wall" className="block text-xs font-medium text-slate-400">Time limit (s)</label>
            <input id="newrun-wall" type="number" min="1" value={wallSecondsCeiling} onChange={(e) => { budgetTouched.current = true; setWallSecondsCeiling(e.target.value); }}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100" />
            {fieldErrors.wallSecondsCeiling && <p role="alert" className="mt-1 text-xs text-rose-400">{fieldErrors.wallSecondsCeiling}</p>}
          </div>
          <div>
            <label htmlFor="newrun-tokens" className="block text-xs font-medium text-slate-400">Token budget</label>
            <input id="newrun-tokens" type="number" min="1" value={modelTokenCeiling} onChange={(e) => { budgetTouched.current = true; setModelTokenCeiling(e.target.value); }}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100" />
            {fieldErrors.modelTokenCeiling && <p role="alert" className="mt-1 text-xs text-rose-400">{fieldErrors.modelTokenCeiling}</p>}
          </div>
        </div>
        <div>
          <label htmlFor="newrun-mode" className="block text-xs font-medium text-slate-400">Mode</label>
          <select
            id="newrun-mode"
            value={executionMode}
            onChange={(e) => setExecutionMode(e.target.value as CreateRunInput["executionMode"])}
            className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100"
          >
            {supportedModes.map((mode) => (
              <option key={mode} value={mode}>{mode}</option>
            ))}
          </select>
          {selectedEnv?.executionModes && (
            <p className="mt-1 text-[11px] text-slate-500">
              Modes declared by {selectedEnv.environmentId}: {selectedEnv.executionModes.join(", ")}
            </p>
          )}
        </div>
        {submitNotice && (
          <Banner tone="warn" title={submitNotice} />
        )}
        <div className="flex justify-end gap-2 pt-1">
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-slate-600 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting}
            className="rounded-md bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-60"
          >
            {submitting ? "Creating…" : "Create run"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

function ApprovalDialog({
  request,
  runId,
  onDecide,
}: {
  request: ApprovalRequest | null;
  runId: string;
  onDecide: (approve: boolean) => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  if (!request) return null;
  const decide = async (approve: boolean) => {
    setBusy(true);
    try {
      await onDecide(approve);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal open title="Approval required" onClose={() => void decide(false)}>
      <div className="space-y-3 text-sm text-slate-300">
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[13px]">
          <dt className="text-slate-500">Tool</dt>
          <dd className="font-mono">{request.tool} <span className="text-slate-500">v{request.toolVersion}</span></dd>
          <dt className="text-slate-500">Effect</dt>
          <dd>
            <StatusBadge status={request.effect === "write" ? "awaiting_approval" : "validated"} /> {request.effect}
          </dd>
          <dt className="text-slate-500">Scope</dt>
          <dd className="break-all font-mono text-[12px]">{request.resourceScope}</dd>
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
            onClick={() => void decide(false)}
            disabled={busy}
            className="rounded-md border border-slate-600 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800"
          >
            Deny
          </button>
          <button
            type="button"
            onClick={() => void decide(true)}
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
