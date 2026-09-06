import type { SkillVersionSummary } from "../api/types";
import { StatusBadge } from "../components/StatusBadge";
import { EmptyState, LoadingState } from "../components/ui";

export function SkillsView({ skills, loading }: { skills: SkillVersionSummary[]; loading: boolean }) {
  if (loading) return <LoadingState label="Loading skill library…" />;
  if (skills.length === 0)
    return (
      <EmptyState title="Skill library is empty">
        The initial active bundle contains only generic execution and safety instructions. Learned skills appear
        here after passing the promotion gate.
      </EmptyState>
    );

  const grouped = new Map<string, SkillVersionSummary[]>();
  for (const skill of skills) {
    const list = grouped.get(skill.skillId) ?? [];
    list.push(skill);
    grouped.set(skill.skillId, list);
  }

  return (
    <section aria-labelledby="skills-heading">
      <h2 id="skills-heading" className="text-base font-semibold text-slate-100">
        Skill library
      </h2>
      <p className="mt-1 text-[13px] text-slate-400">
        Active, proposed, rejected, quarantined, and rolled-back versions are distinguished. Active skills never
        bypass policy checks; their preconditions are informational only.
      </p>
      <div className="mt-4 grid gap-4 md:grid-cols-2">
        {[...grouped.entries()].map(([skillId, versions]) => (
          <div key={skillId} className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
            <h3 className="font-mono text-sm text-slate-100">{skillId}</h3>
            <ul className="mt-3 space-y-2">
              {versions.map((v) => (
                <li key={`${v.skillId}@${v.version}`} className="rounded-lg bg-slate-800/60 px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs text-slate-300">
                      v{v.version}
                      {v.parentVersion ? <span className="text-slate-600"> ← v{v.parentVersion}</span> : null}
                    </span>
                    <StatusBadge status={v.state} />
                  </div>
                  <p className="mt-1 text-xs text-slate-400">applies to: {v.applicability}</p>
                  <p className="mt-0.5 font-mono text-[10px] text-slate-600">
                    {v.contentHash} · evidence: {v.evidenceRefs.join(", ") || "—"}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}
