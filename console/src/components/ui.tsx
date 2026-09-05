import { useEffect, useRef, type ReactNode } from "react";

export function Banner({
  tone,
  title,
  children,
  role = "status",
}: {
  tone: "warn" | "bad" | "info" | "good";
  title: string;
  children?: ReactNode;
  role?: "status" | "alert";
}) {
  const tones = {
    warn: "border-amber-700 bg-amber-950/60 text-amber-200",
    bad: "border-rose-700 bg-rose-950/60 text-rose-200",
    info: "border-sky-700 bg-sky-950/60 text-sky-200",
    good: "border-emerald-700 bg-emerald-950/60 text-emerald-200",
  } as const;
  const glyphs = { warn: "⚠", bad: "✗", info: "ℹ", good: "✓" } as const;
  return (
    <div role={role} aria-live={role === "alert" ? "assertive" : "polite"} className={`rounded-lg border px-4 py-3 text-sm ${tones[tone]}`}>
      <p className="flex items-center gap-2 font-medium">
        <span aria-hidden="true">{glyphs[tone]}</span>
        <span>{title}</span>
      </p>
      {children ? <div className="mt-1 text-[13px] leading-relaxed opacity-90">{children}</div> : null}
    </div>
  );
}

export function LoadingState({ label }: { label: string }) {
  return (
    <div role="status" aria-live="polite" className="flex items-center gap-3 p-8 text-sm text-slate-400">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-600 border-t-sky-400" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-slate-700 bg-slate-900/40 p-6">
      <p className="text-sm font-semibold text-slate-300">{title}</p>
      {children ? <div className="mt-2 text-[13px] leading-relaxed text-slate-400">{children}</div> : null}
    </div>
  );
}

/** Modal with focus trap, Escape to cancel, and focus restoration on close. */
export function Modal({
  open,
  title,
  onClose,
  children,
}: {
  open: boolean;
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    restoreRef.current = document.activeElement as HTMLElement | null;
    const panel = panelRef.current;
    panel?.querySelector<HTMLElement>("input, button, [tabindex]")?.focus();

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !panel) return;
      const focusables = Array.from(
        panel.querySelectorAll<HTMLElement>("a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex='-1'])"),
      );
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      restoreRef.current?.focus();
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" role="presentation">
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="w-full max-w-lg rounded-xl border border-slate-700 bg-slate-900 p-5 shadow-2xl"
      >
        <h2 className="text-base font-semibold text-slate-100">{title}</h2>
        <div className="mt-3">{children}</div>
      </div>
    </div>
  );
}
