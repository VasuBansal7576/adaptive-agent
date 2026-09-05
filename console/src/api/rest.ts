import type { RunEvent, RunRecord, SkillVersionSummary, CandidateDiff, EnvironmentPackageSummary } from "./types";
import type {
  ConsoleTransport,
  CreateRunInput,
  EnvironmentPackageForm,
  EnvironmentRegistration,
  LearningCycleInput,
} from "./transport";
import { validatePackageFields, formToStringPayload, formToRegistration } from "./transport";
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
  constructor(payload: Record<string, unknown>, status: number) {
    // FastAPI wraps structured envelopes in `detail` (409/403); plain 422s use a string detail
    const envelope = (payload.detail && typeof payload.detail === "object" && payload.detail !== null
      ? (payload.detail as Record<string, unknown>)
      : payload) as Record<string, unknown>;
    const code = typeof envelope.code === "string" ? envelope.code : "UNKNOWN";
    const message = typeof envelope.message === "string" ? envelope.message : typeof payload.detail === "string" ? payload.detail : `HTTP ${status}`;
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.correlationId = typeof envelope.correlationId === "string" ? envelope.correlationId : null;
    this.retry = typeof envelope.retry === "string" ? envelope.retry : null;
  }
  describe(): string {
    const corr = this.correlationId ? ` (correlation ${this.correlationId})` : "";
    return `${this.code}: ${this.message}${corr}`;
  }
}

/**
 * Live transport against the SPEC control API (aligned with commit 7d3c2b5).
 * - POST /environments takes the strict full manifest (docs[] + taskGoals[] +
 *   canonical refs + executionModes); the string form lives at /environments/form.
 * - POST /runs accepts the canonical taskRef projection; launch is explicit
 *   via POST /runs/{id}/launch; executionMode is recorded on the run and its events.
 * - POST /learning/launch stages an evidence-linked learning proposal.
 * - Every payload passes boundary-schema validation before entering UI state.
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
    if (res.status === 204) return undefined;
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

    async createRun(input: CreateRunInput) {
      // canonical taskRef projection: {goal, environmentId, environmentRef}
      const body = {
        taskRef: {
          goal: input.goal,
          environmentId: input.environmentId,
          environmentRef: { id: input.environmentId, version: "1" },
        },
        modelProfileRef: { id: input.modelProfile, version: "1" },
        budgetRef: { id: `budget-${input.idempotencyKey.slice(0, 8)}`, version: "1" },
        idempotencyKey: input.idempotencyKey,
        executionMode: input.executionMode,
      };
      const run = await validated(json("/runs", { method: "POST", body: JSON.stringify(body) }), (value) =>
        parseRun(value, "run"),
      );
      return run;
    },

    launchRun: (runId) =>
      json(`/runs/${encodeURIComponent(runId)}/launch`, { method: "POST", body: "{}" }).then(() => undefined),

    async registerEnvironment(manifest: EnvironmentRegistration) {
      await json("/environments", { method: "POST", body: JSON.stringify(manifest) });
      const environments = await validated(json("/environments"), parseEnvironments);
      const created = environments.find((e) => e.environmentId === manifest.environmentId);
      if (!created) throw new ApiError({ code: "UNKNOWN", message: "registered environment missing from list" }, 200);
      return created;
    },

    launchLearningCycle: async (input: LearningCycleInput) => {
      const action = (await json("/learning/launch", {
        method: "POST",
        body: JSON.stringify({
          runId: input.runId,
          predictedEffect: input.predictedEffect,
          evidenceIds: input.evidenceIds,
        }),
      })) as Record<string, unknown>;
      const actionId = typeof action.actionId === "string" ? action.actionId : "";
      const status = typeof action.status === "string" ? action.status : "staged";
      if (!actionId) throw new SchemaError("learningAction.actionId");
      return { actionId, runId: input.runId, status };
    },

    async validateEnvironmentPackage(fields: EnvironmentPackageForm) {
      const missing = validatePackageFields(fields);
      if (missing.length > 0) return { ok: false, missingFields: missing };
      try {
        // the validate boundary takes the string form projection
        const result = (await json("/environments/validate", {
          method: "POST",
          body: JSON.stringify(formToStringPayload(fields)),
        })) as { ok: boolean; missingFields: string[] };
        return result;
      } catch {
        return { ok: false, missingFields: ["server rejected package"] };
      }
    },
  };
}
