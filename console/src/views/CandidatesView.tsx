import { useEffect, useState } from "react";
import type { ConsoleTransport, LearningCycleInput } from "../api/transport";
import type { CandidateDiff, EvaluationJob, RunRecord } from "../api/types";
import { StatusBadge } from "../components/StatusBadge";
import { Banner, EmptyState, LoadingState, Modal } from "../components/ui";

export function CandidatesView({
  transport,
  candidates,
  runs,
  loading,
  onActionError,
  onRefreshCandidates,
}: {
  transport: ConsoleTransport;
  candidates: CandidateDiff[];
  runs: RunRecord[];
  loading: boolean;
  onActionError: (message: string, correlationId?: string | null) => void;
  onRefreshCandidates: () => void;
}) {
  const [rollbackTarget, setRollbackTarget] = useState<CandidateDiff | null>(null);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [cycleOpen, setCycleOpen] = useState(false);
  const [cycleBusy, setCycleBusy] = useState(false);
  const [cycleRunId, setCycleRunId] = useState("");
  const [notice, setNotice] = useState<{ tone: "good" | "bad"; text: string } | null>(null);
  const [evalJobs, setEvalJobs] = useState<EvaluationJob[] | null>(null);
  const [evalJobsError, setEvalJobsError] = useState<string | null>(null);

  // evaluation job statuses follow the candidate list
  useEffect(() => {
    let cancelled = false;
    setEvalJobsError(null);
    transport
      .listEvaluations()
      .then((jobs) => {
        if (!cancelled) setEvalJobs(jobs);
      })
      .catch((error) => {
        if (!cancelled) setEvalJobsError(error instanceof Error ? error.message : "evaluation status unavailable");
      });
    return () => {
      cancelled = true;
    };
  }, [transport, candidates]);

  const launchEvaluation = async (cand: CandidateDiff) => {
    if (!cand.baseBundleHash) {
      onActionError(`Evaluation launch unavailable for ${cand.candidateId}: the candidate has no base bundle hash.`);
      return;
    }
    try {
      const { evaluationId, state } = await transport.launchEvaluation({ candidateId: cand.candidateId, baseBundleHash: cand.baseBundleHash });
      setNotice({ tone: "good", text: `✓ Evaluation ${evaluationId} queued (${state}) for ${cand.candidateId}.` });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Evaluation launch failed";
      const correlationId = (error as { correlationId?: string } | null)?.correlationId ?? null;
      onActionError(`Evaluation launch failed for ${cand.candidateId}: ${message}`, correlationId);
    }
  };

  // eligible learning source: a COMPLETED development run the model can learn
  // from; the durable pipeline generates proposal + evidence from its attempts
  const eligibleRuns = runs.filter((r) => r.status === "succeeded");

  const runCycle = async (input: LearningCycleInput) => {
    setCycleBusy(true);
    try {
      const { actionId, status } = await transport.launchLearningCycle(input);
      setNotice({
        tone: "good",
        text: `✓ Learning cycle staged: action ${actionId} (${status}) on run ${input.runId}. The runtime generates the proposal and evidence; the candidate list refreshes automatically.`,
      });
      setCycleOpen(false);
      onRefreshCandidates();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Learning cycle failed";
      const correlationId = (error as { correlationId?: string } | null)?.correlationId ?? null;
      onActionError(`Learning cycle failed: ${message}`, correlationId);
    } finally {
      setCycleBusy(false);
    }
  };

  const submitRollback = async () => {
    if (!rollbackTarget) return;
    setBusy(true);
    try {
      await transport.requestRollback(rollbackTarget.candidateId, reason);
      setNotice({ tone: "good", text: `✓ Rollback of ${rollbackTarget.candidateId} recorded; new runs use the previous approved version.` });
      setRollbackTarget(null);
      setReason("");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Rollback failed";
      const correlationId = (error as { correlationId?: string } | null)?.correlationId ?? null;
      // visible failure with correlation id; no partial state was applied
      onActionError(`Rollback failed: ${message}`, correlationId);
      setNotice({ tone: "bad", text: "✗ Rollback request failed. Check connection and retry; no partial state was applied." });
    } finally {
      setBusy(false);
    }
  };

  const hasData = !loading && candidates.length === 0;

  return (
    <section aria-labelledby="candidates-heading" className="space-y-4">
      <div>
        <h2 id="candidates-heading" className="text-base font-semibold text-slate-100">
          Candidate comparison
        </h2>
        <p className="mt-1 text-[13px] text-slate-400">
          Predicted effects and measured performance are labeled separately. Only trusted evaluation reports gate
          promotion; rollback restores the prior approved version and is audited.
        </p>
      </div>

      <LearningCycleButton busy={cycleBusy} onRun={() => setCycleOpen(true)} />

      {loading && <LoadingState label="Loading candidates…" />}

      {hasData && (
        <EmptyState title="No candidates yet">
          Candidates appear when the runtime stages an evidence-linked proposal from a completed development run.
          Predicted effects are never treated as measured results.
        </EmptyState>
      )}

      {notice && <Banner tone={notice.tone} title={notice.text.replace(/^[✓✗] /, "")} />}

      {candidates.map((cand) => (
        <article key={cand.candidateId} className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="break-all font-mono text-sm text-slate-100">{cand.candidateId}</h3>
            <div className="flex items-center gap-2">
              <StatusBadge status={cand.state} />
              {cand.state === "validated" && (
                <button
                  type="button"
                  onClick={() => void launchEvaluation(cand)}
                  className="rounded-md border border-sky-700 px-3 py-1.5 text-xs font-medium text-sky-300 hover:bg-sky-950/60"
                  title="Queue the trusted evaluation for this candidate"
                >
                  Launch evaluation
                </button>
              )}
            </div>
          </div>
          {cand.state === "validated" && (
            <p className="mt-1 text-[11px] text-slate-500">
              Proposal validation passed — this is not a performance result.
            </p>
          )}
          {cand.state === "evaluating" && (
            <p className="mt-1 text-[11px] text-slate-500">
              {evalJobs === null && !evalJobsError
                ? "Evaluation status loading…"
                : evalJobsError
                  ? `Evaluation status unavailable: ${evalJobsError}`
                  : (() => {
                      const jobs = evalJobs?.filter((j) => j.candidateId === cand.candidateId) ?? [];
                      return jobs.length === 0
                        ? "No evaluation job recorded for this candidate."
                        : `Evaluation ${jobs[0].evaluationId}: ${jobs[0].state}${jobs[0].trusted ? " (trusted)" : ""}${jobs[0].reason ? ` — ${jobs[0].reason}` : ""}`;
                    })()}
            </p>
          )}
          <p className="mt-1 text-xs text-slate-500 [overflow-wrap:anywhere]">
            {cand.baseBundleRef
              ? `base ${cand.baseBundleRef.id} v${cand.baseBundleRef.version} → candidate ${cand.candidateBundleRef?.id} v${cand.candidateBundleRef?.version}`
              : `base bundle ${cand.baseBundleHash?.slice(0, 16)}… → candidate bundle ${cand.candidateBundleHash ? `${cand.candidateBundleHash.slice(0, 16)}…` : "pending"}`}
          </p>

          <div className="mt-3 grid gap-4 lg:grid-cols-2">
            <div className="min-w-0">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                {cand.diff ? "Diff (immutable, bounded)" : "Edit operations (bounded)"}
              </h4>
              <pre className="mt-1 max-h-72 overflow-auto rounded-lg bg-slate-800 p-3 font-mono text-[11px] leading-relaxed text-slate-200 [overflow-wrap:anywhere]">
                {cand.diff ?? renderEditOperations(cand)}
              </pre>
            </div>
            <div className="min-w-0 space-y-3">
              <div className="rounded-lg bg-slate-800/60 p-3">
                <p className="text-xs font-semibold text-sky-300">Predicted effect</p>
                <p className="mt-1 text-[13px] text-slate-300 [overflow-wrap:anywhere]">{cand.predictedEffect}</p>
              </div>
              {cand.supportingEvidenceIds && cand.supportingEvidenceIds.length > 0 && (
                <div className="rounded-lg bg-slate-800/60 p-3">
                  <p className="text-xs font-semibold text-slate-300">Supporting evidence</p>
                  <ul className="mt-1 list-disc pl-5 font-mono text-[11px] text-slate-400">
                    {cand.supportingEvidenceIds.map((id) => (
                      <li key={id} className="break-all">{id}</li>
                    ))}
                  </ul>
                </div>
              )}
              {cand.measured ? (
                <div className="rounded-lg bg-slate-800/60 p-3">
                  <p className="text-xs font-semibold text-emerald-300">Measured result (trusted evaluation)</p>
                  <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-[13px]">
                    <dt className="text-slate-500">Accuracy gain</dt>
                    <dd className="text-slate-200">{cand.measured.accuracyGainPp.toFixed(1)} pp</dd>
                    <dt className="text-slate-500">Reliability Δ</dt>
                    <dd className="text-slate-200">{cand.measured.reliabilityDelta >= 0 ? "+" : ""}{cand.measured.reliabilityDelta.toFixed(2)}</dd>
                    <dt className="text-slate-500">Cost ratio</dt>
                    <dd className="text-slate-200">{cand.measured.costRatio.toFixed(2)}×</dd>
                    <dt className="text-slate-500">p95 latency ratio</dt>
                    <dd className="text-slate-200">{cand.measured.p95LatencyRatio.toFixed(2)}×</dd>
                  </dl>
                  <p className="mt-2 text-xs text-slate-400">
                    Gate: <StatusBadge status={cand.measured.gateDecision === "promoted" ? "promoted" : "rejected"} />
                  </p>
                  <ul className="mt-1 list-disc pl-5 text-xs text-slate-400">
                    {cand.measured.gateReasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                  <p className="mt-1 break-all font-mono text-[10px] text-slate-600">report {cand.measured.evaluationRef.id} · {cand.measured.evaluationRef.sha256}</p>
                </div>
              ) : (
                <div className="rounded-lg bg-slate-800/60 p-3">
                  <p className="text-xs font-semibold text-slate-400">Measured result</p>
                  <p className="mt-1 text-[13px] text-slate-500">Awaiting a trusted evaluation report.</p>
                </div>
              )}
            </div>
          </div>

          {cand.audit && cand.audit.length > 0 && (
            <div className="mt-3 rounded-lg border border-slate-700 bg-slate-800/40 p-3">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Rollback history</h4>
              <ul className="mt-2 space-y-2">
                {cand.audit.map((rb) => (
                  <li key={rb.rollbackId} className="text-[13px] text-slate-300">
                    <span className="font-mono text-xs text-slate-400">{rb.at}</span> — {rb.reason}
                    <span className="block break-all text-xs text-slate-500">
                      {rb.fromRef.id} v{rb.fromRef.version} → {rb.toRef.id} v{rb.toRef.version} · affected runs: {rb.affectedRuns.join(", ")}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {cand.state === "promoted" && (
            <div className="mt-3">
              <button
                type="button"
                onClick={() => setRollbackTarget(cand)}
                className="rounded-md border border-amber-700 px-3 py-1.5 text-xs font-medium text-amber-300 hover:bg-amber-950/60"
              >
                Request rollback
              </button>
            </div>
          )}
        </article>
      ))}

      <LearningCycleModal
        open={cycleOpen}
        onClose={() => setCycleOpen(false)}
        runs={eligibleRuns}
        busy={cycleBusy}
        onSubmit={(input) => void runCycle(input)}
      />

      <Modal open={rollbackTarget !== null} title="Request rollback" onClose={() => setRollbackTarget(null)}>
        <div className="space-y-3 text-sm text-slate-300">
          <p className="text-[13px]">
            Rollback atomically restores a previously approved immutable version and records the reason and affected
            runs. External systems already changed by tools are not reverted.
          </p>
          <div>
            <label htmlFor="rollback-reason" className="block text-xs font-medium text-slate-400">
              Reason (recorded in audit history)
            </label>
            <input
              id="rollback-reason"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100"
              placeholder="e.g., safety suite flagged unbounded child fan-out"
            />
          </div>
          <div className="flex justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={() => setRollbackTarget(null)}
              className="rounded-md border border-slate-600 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={submitRollback}
              disabled={busy || !reason.trim()}
              className="rounded-md bg-amber-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-amber-500 disabled:opacity-60"
            >
              {busy ? "Submitting…" : "Confirm rollback"}
            </button>
          </div>
        </div>
      </Modal>
    </section>
  );
}

/** Render durable edit operations (JSON-encoded strings) as readable blocks. */
function renderEditOperations(cand: CandidateDiff): string {
  if (!cand.editOperations || cand.editOperations.length === 0) return "(no edit operations)";
  return cand.editOperations
    .map((op, i) => {
      try {
        const parsed = JSON.parse(op) as { operation?: string; path?: string; value?: string };
        return [
          `# operation ${i + 1}: ${parsed.operation ?? "?"} ${parsed.path ?? ""}`,
          parsed.value ?? op,
        ].join("\n");
      } catch {
        return `# operation ${i + 1}\n${op}`;
      }
    })
    .join("\n\n");
}

function LearningCycleButton({ busy, onRun }: { busy: boolean; onRun: () => void }) {
  return (
    <div>
      <button
        type="button"
        onClick={onRun}
        disabled={busy}
        className="rounded-md bg-sky-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-60"
        title="Select a completed development run; the runtime generates the proposal and evidence"
      >
        Run learning cycle
      </button>
    </div>
  );
}

function LearningCycleModal({
  open,
  onClose,
  runs,
  busy,
  onSubmit,
}: {
  open: boolean;
  onClose: () => void;
  runs: RunRecord[];
  busy: boolean;
  onSubmit: (input: LearningCycleInput) => void;
}) {
  const [runId, setRunId] = useState("");
  return (
    <Modal open={open} title="Run learning cycle" onClose={onClose}>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!runId) return;
          onSubmit({ runId });
        }}
        className="space-y-3 text-sm text-slate-300"
      >
        <p className="text-xs text-slate-500">
          The runtime generates the proposal and supporting evidence from the selected run's completed development
          attempts. Predicted effects are model-generated and never treated as scores; only the trusted evaluator
          gates promotion.
        </p>
        <div>
          <label htmlFor="cycle-run" className="block text-xs font-medium text-slate-400">
            Completed development run
          </label>
          {runs.length === 0 ? (
            <p role="status" className="mt-1 text-[13px] text-slate-400">
              No completed development runs are eligible yet. Complete a run successfully first, then stage a
              learning cycle.
            </p>
          ) : (
            <select
              id="cycle-run"
              value={runId}
              onChange={(e) => setRunId(e.target.value)}
              className="mt-1 w-full rounded-md border border-slate-600 bg-slate-800 px-2.5 py-1.5 text-sm text-slate-100"
            >
              <option value="">Select a run…</option>
              {runs.map((run) => (
                <option key={run.runId} value={run.runId}>
                  {run.runId} — {run.goal}
                </option>
              ))}
            </select>
          )}
        </div>
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
            disabled={busy || !runId}
            className="rounded-md bg-sky-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-sky-500 disabled:opacity-60"
          >
            {busy ? "Staging…" : "Stage learning cycle"}
          </button>
        </div>
      </form>
    </Modal>
  );
}
