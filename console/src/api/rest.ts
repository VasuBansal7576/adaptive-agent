import type { RunEvent, RunRecord, SkillVersionSummary, CandidateDiff, EnvironmentPackageSummary } from "./types";
import type { ConsoleTransport, EnvironmentPackageForm } from "./transport";
import { validatePackageFields } from "./transport";

/**
 * Live transport against the SPEC control API.
 * - REST for reads/mutations; SSE for run event streams with cursor resume.
 * - The server pins active versions and enforces policy; the console only
 *   displays and relays operator actions.
 *
 * Expected control API surface (proposed to adaptive-agent-2, see coordination):
 *   GET  /environments            -> EnvironmentPackageSummary[]
 *   GET  /runs                    -> RunRecord[]
 *   GET  /skills                  -> SkillVersionSummary[]
 *   GET  /candidates              -> CandidateDiff[]
 *   GET  /runs/{id}/events?cursor -> SSE: data: RunEvent (monotonic sequence)
 *   POST /runs/{id}/cancel
 *   POST /runs/{id}/approvals/{approvalId}  body: {approve: boolean}
 *   POST /candidates/{id}/rollback          body: {reason}
 *   POST /environments/validate             body: EnvironmentPackageForm
 * Errors use {code, message, correlationId, retry} per SPEC.
 */
export function createRestTransport(baseUrl = ""): ConsoleTransport {
  async function json<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${baseUrl}${path}`, {
      headers: { "content-type": "application/json" },
      ...init,
    });
    if (!res.ok) {
      let payload: unknown = null;
      try {
        payload = await res.json();
      } catch {
        /* non-JSON error body */
      }
      const err = new Error(`HTTP ${res.status}`);
      (err as Error & { payload?: unknown; status?: number }).payload = payload;
      (err as Error & { payload?: unknown; status?: number }).status = res.status;
      throw err;
    }
    return res.json() as Promise<T>;
  }

  return {
    mode: "live",

    listEnvironments: () => json<EnvironmentPackageSummary[]>("/environments"),
    listRuns: () => json<RunRecord[]>("/runs"),
    listSkills: () => json<SkillVersionSummary[]>("/skills"),
    listCandidates: () => json<CandidateDiff[]>("/candidates"),

    openRunStream(runId, fromCursor, { onEvent, onState }) {
      let es: EventSource | null = null;
      let retryTimer: ReturnType<typeof setTimeout> | null = null;
      let closed = false;
      let lastSequence = fromCursor;

      const connect = () => {
        if (closed) return;
        onState(lastSequence === fromCursor ? "live" : "reconnecting");
        es = new EventSource(`${baseUrl}/runs/${encodeURIComponent(runId)}/events?cursor=${lastSequence}`);
        es.onmessage = (message) => {
          try {
            const event = JSON.parse(message.data) as RunEvent;
            if (event.sequence > lastSequence) {
              lastSequence = event.sequence;
              onEvent(event);
            }
            // stale duplicates (sequence <= cursor) are dropped client-side too
          } catch {
            onState("stale");
          }
        };
        es.onerror = () => {
          es?.close();
          onState("stale");
          retryTimer = setTimeout(connect, 1500);
        };
      };
      connect();

      return () => {
        closed = true;
        if (retryTimer) clearTimeout(retryTimer);
        es?.close();
        onState("closed");
      };
    },

    cancelRun: (runId) => json<void>(`/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST", body: "{}" }),

    submitApproval: (runId, approvalId, approve) =>
      json<void>(`/runs/${encodeURIComponent(runId)}/approvals/${encodeURIComponent(approvalId)}`, {
        method: "POST",
        body: JSON.stringify({ approve }),
      }),

    requestRollback: (candidateId, reason) =>
      json<void>(`/candidates/${encodeURIComponent(candidateId)}/rollback`, {
        method: "POST",
        body: JSON.stringify({ reason }),
      }),

    async validateEnvironmentPackage(fields: EnvironmentPackageForm) {
      const missing = validatePackageFields(fields);
      if (missing.length > 0) return { ok: false, missingFields: missing };
      try {
        const result = await json<{ ok: boolean; missingFields: string[] }>("/environments/validate", {
          method: "POST",
          body: JSON.stringify(fields),
        });
        return result;
      } catch (error) {
        return { ok: false, missingFields: ["server rejected package"] };
      }
    },
  };
}
