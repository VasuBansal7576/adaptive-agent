import type { RunEvent, RunRecord, SkillVersionSummary, CandidateDiff, EnvironmentPackageSummary } from "./types";
import type { ConsoleTransport, CreateRunInput, EnvironmentPackageForm, EnvironmentRegistration } from "./transport";
import { validatePackageFields, formToRegistration } from "./transport";
import {
  SchemaError,
  parseCandidates,
  parseEnvironments,
  parseRun,
  parseRunEvent,
  parseRuns,
  parseSkills,
} from "./validate";

/** Structured API error carrying the SPEC error envelope for UI surfacing. */
export class ApiError extends Error {
  code: string;
  correlationId: string | null;
  retry: string | null;
  constructor(payload: { code?: unknown; message?: unknown; correlationId?: unknown; retry?: unknown }, status: number) {
    const code = typeof payload.code === "string" ? payload.code : "UNKNOWN";
    const message = typeof payload.message === "string" ? payload.message : `HTTP ${status}`;
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.correlationId = typeof payload.correlationId === "string" ? payload.correlationId : null;
    this.retry = typeof payload.retry === "string" ? payload.retry : null;
  }
  describe(): string {
    const corr = this.correlationId ? ` (correlation ${this.correlationId})` : "";
    return `${this.code}: ${this.message}${corr}`;
  }
}

/**
 * Live transport against the SPEC control API.
 * - REST for reads/mutations; SSE for run event streams with cursor resume.
 * - Every payload passes boundary-schema validation before entering UI state.
 *
 * Expected control API surface (proposed to the Python API owner):
 *   GET  /environments            -> EnvironmentPackageSummary[]
 *   POST /environments            -> register full manifest -> EnvironmentPackageSummary
 *   POST /environments/validate
 *   GET  /runs                    -> RunRecord[]
 *   POST /runs (CreateRunInput)   -> RunRecord (server pins active version)
 *   GET  /runs/{id}/events?cursor -> SSE: data: RunEvent (monotonic sequence)
 *   POST /runs/{id}/cancel
 *   POST /runs/{id}/approvals/{approvalId}  body: {approve: boolean}
 *   POST /candidates/{id}/rollback          body: {reason}
 *   POST /learning-cycles         -> {candidate: CandidateDiff}
 */
export function createRestTransport(baseUrl = ""): ConsoleTransport {
  async function json(path: string, init?: RequestInit): Promise<unknown> {
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
      throw new ApiError((payload ?? {}) as Record<string, unknown>, res.status);
    }
    return res.json();
  }

  function validated<T>(promise: Promise<unknown>, parse: (value: unknown) => T): Promise<T> {
    return promise.then(
      (value) => {
        try {
          return parse(value);
        } catch (error) {
          if (error instanceof SchemaError) throw error;
          throw new SchemaError("response envelope");
        }
      },
      (error) => {
        if (error instanceof ApiError || error instanceof SchemaError) throw error;
        throw error;
      },
    );
  }

  return {
    mode: "live",

    listEnvironments: () => validated(json("/environments"), parseEnvironments),
    listRuns: () => validated(json("/runs"), parseRuns),
    listSkills: () => validated(json("/skills"), parseSkills),
    listCandidates: () => validated(json("/candidates"), parseCandidates),

    openRunStream(runId, fromCursor, { onEvent, onState }) {
      // environments without EventSource support (some test DOMs) surface an
      // honest stale state rather than crashing
      if (typeof EventSource === "undefined") {
        onState("stale");
        return () => undefined;
      }
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
            // boundary validation before the event enters reducer state
            const event = parseRunEvent(JSON.parse(message.data), "sse.data");
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

    cancelRun: (runId) => json(`/runs/${encodeURIComponent(runId)}/cancel`, { method: "POST", body: "{}" }).then(() => undefined),

    submitApproval: (runId, approvalId, approve) =>
      json(`/runs/${encodeURIComponent(runId)}/approvals/${encodeURIComponent(approvalId)}`, {
        method: "POST",
        body: JSON.stringify({ approve }),
      }).then(() => undefined),

    requestRollback: (candidateId, reason) =>
      json(`/candidates/${encodeURIComponent(candidateId)}/rollback`, {
        method: "POST",
        body: JSON.stringify({ reason }),
      }).then(() => undefined),

    createRun: (input: CreateRunInput) =>
      validated(
        json("/runs", { method: "POST", body: JSON.stringify(input) }),
        (value) => parseRun(value, "run"),
      ),

    async registerEnvironment(manifest: EnvironmentRegistration) {
      await json("/environments", { method: "POST", body: JSON.stringify(manifest) });
      const environments = await validated(json("/environments"), parseEnvironments);
      const created = environments.find((e) => e.environmentId === manifest.environmentId);
      if (!created) throw new ApiError({ code: "UNKNOWN", message: "registered environment missing from list" }, 200);
      return created;
    },

    runLearningCycle: () =>
      validated(json("/learning-cycles", { method: "POST", body: "{}" }), (value) => {
        const o = (value ?? {}) as Record<string, unknown>;
        const candidates = parseCandidates([o.candidate]);
        return { candidate: candidates[0] };
      }),

    async validateEnvironmentPackage(fields: EnvironmentPackageForm) {
      const missing = validatePackageFields(fields);
      if (missing.length > 0) return { ok: false, missingFields: missing };
      try {
        const result = (await json("/environments/validate", {
          method: "POST",
          body: JSON.stringify(formToRegistration(fields)),
        })) as { ok: boolean; missingFields: string[] };
        return result;
      } catch {
        return { ok: false, missingFields: ["server rejected package"] };
      }
    },
  };
}
