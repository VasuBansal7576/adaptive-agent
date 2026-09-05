import { useState } from "react";
import type { ConsoleTransport } from "../api/transport";
import type { CandidateDiff } from "../api/types";
import { StatusBadge } from "../components/StatusBadge";
import { Banner, EmptyState, LoadingState, Modal } from "../components/ui";

export function CandidatesView({
  transport,
  candidates,
  loading,
}: {
  transport: ConsoleTransport;
  candidates: CandidateDiff[];
  loading: boolean;
}) {
  const [rollbackTarget, setRollbackTarget] = useState<CandidateDiff | null>(null);
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<{ tone: "good" | "bad"; text: string } | null>(null);

  if (loading) return <LoadingState label="Loading candidates…" />;
  if (candidates.length === 0)
    return (
      <EmptyState title="No candidates yet">
        Candidates appear when the learner submits an evidence-linked proposal from a completed development
        attempt. Predicted effects are never treated as measured results.
      </EmptyState>
    );

  const submitRollback = async () => {
    if (!rollbackTarget) return;
    setBusy(true);
    try {
      await transport.requestRollback(rollbackTarget.candidateId, reason);
      setNotice({ tone: "good", text: `✓ Rollback of ${rollbackTarget.candidateId} recorded; new runs use the previous approved version.` });
      setRollbackTarget(null);
      setReason("");
    } catch {
      setNotice({ tone: "bad", text: "✗ Rollback request failed. Check connection and retry; no partial state was applied." });
    } finally {
      setBusy(false);
    }
  };

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

      {notice && <Banner tone={notice.tone} title={notice.text.replace(/^[✓✗] /, "")} />}

      <div className="space-y-4">
        {candidates.map((cand) => (
          <article key={cand.candidateId} className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="font-mono text-sm text-slate-100">{cand.candidateId}</h3>
              <StatusBadge status={cand.state} />
            </div>
            <p className="mt-1 text-xs text-slate-500">
              base {cand.baseBundleRef.id} v{cand.baseBundleRef.version} → candidate {cand.candidateBundleRef.id} v{cand.candidateBundleRef.version}
            </p>

            <div className="mt-3 grid gap-4 lg:grid-cols-2">
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500">Diff (immutable, bounded)</h4>
                <pre className="mt-1 overflow-auto rounded-lg bg-slate-800 p-3 font-mono text-[11px] leading-relaxed text-slate-200">
                  {cand.diff}
                </pre>
              </div>
              <div className="space-y-3">
                <div className="rounded-lg bg-slate-800/60 p-3">
                  <p className="text-xs font-semibold text-sky-300">Predicted effect</p>
                  <p className="mt-1 text-[13px] text-slate-300">{cand.predictedEffect}</p>
                </div>
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
                    <p className="mt-1 font-mono text-[10px] text-slate-600">report {cand.measured.evaluationRef.id} · {cand.measured.evaluationRef.sha256}</p>
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
                      <span className="block text-xs text-slate-500">
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
      </div>

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
