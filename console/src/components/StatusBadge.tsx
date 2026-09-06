/** Non-color status indicators: every status pairs a glyph with a text label. */
export const STATUS_META: Record<
  string,
  { glyph: string; label: string; tone: "good" | "warn" | "bad" | "info" | "neutral" }
> = {
  queued: { glyph: "○", label: "Queued", tone: "info" },
  running: { glyph: "◐", label: "Running", tone: "info" },
  awaiting_approval: { glyph: "⏸", label: "Awaiting approval", tone: "warn" },
  succeeded: { glyph: "✓", label: "Succeeded", tone: "good" },
  failed: { glyph: "✗", label: "Failed", tone: "bad" },
  cancelled: { glyph: "⊘", label: "Cancelled", tone: "neutral" },
  timed_out: { glyph: "⏱", label: "Timed out", tone: "bad" },
  active: { glyph: "✓", label: "Active", tone: "good" },
  proposed: { glyph: "◐", label: "Proposed", tone: "info" },
  rejected: { glyph: "✗", label: "Rejected", tone: "bad" },
  quarantined: { glyph: "⚠", label: "Quarantined", tone: "warn" },
  rolled_back: { glyph: "↩", label: "Rolled back", tone: "neutral" },
  evaluating: { glyph: "◐", label: "Evaluating", tone: "info" },
  validated: { glyph: "✓", label: "Validated", tone: "good" },
  draft: { glyph: "○", label: "Draft", tone: "neutral" },
  superseded: { glyph: "↳", label: "Superseded", tone: "neutral" },
  promoted: { glyph: "✓", label: "Promoted", tone: "good" },
  valid: { glyph: "✓", label: "Valid", tone: "good" },
  invalid: { glyph: "✗", label: "Invalid", tone: "bad" },
  "live": { glyph: "●", label: "Stream connected", tone: "good" },
  "ready": { glyph: "✓", label: "Ready", tone: "good" },
  "stale": { glyph: "◌", label: "Stale", tone: "warn" },
  "reconnecting": { glyph: "◌", label: "Reconnecting", tone: "info" },
  "connecting": { glyph: "◌", label: "Connecting", tone: "info" },
  "disconnected": { glyph: "⨯", label: "Disconnected", tone: "bad" },
  "closed": { glyph: "○", label: "Closed", tone: "neutral" },
  // simulated-stream badges deliberately avoid the green "connected" tone so a
  // fixture stream is never mistaken for live inference
  "sim_stream_connected": { glyph: "◐", label: "Simulated stream connected (fixture)", tone: "warn" },
  "sim_stream_stale": { glyph: "◌", label: "Simulated stream stale", tone: "warn" },
  "sim_stream_reconnecting": { glyph: "◌", label: "Simulated stream reconnecting", tone: "info" },
  "sim_stream_connecting": { glyph: "◌", label: "Simulated stream connecting", tone: "info" },
  "sim_stream_closed": { glyph: "○", label: "Simulated stream closed", tone: "neutral" },
};

const TONE_CLASS: Record<string, string> = {
  good: "bg-emerald-950 text-emerald-300 ring-emerald-700",
  warn: "bg-amber-950 text-amber-300 ring-amber-700",
  bad: "bg-rose-950 text-rose-300 ring-rose-700",
  info: "bg-sky-950 text-sky-300 ring-sky-700",
  neutral: "bg-slate-800 text-slate-300 ring-slate-600",
};

export function StatusBadge({ status }: { status: string }) {
  const meta = STATUS_META[status] ?? { glyph: "•", label: status, tone: "neutral" as const };
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${TONE_CLASS[meta.tone]}`}
    >
      <span aria-hidden="true">{meta.glyph}</span>
      <span>{meta.label}</span>
    </span>
  );
}
