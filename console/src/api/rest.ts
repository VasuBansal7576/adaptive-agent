import type { RunEvent, RunRecord, SkillVersionSummary, CandidateDiff, EnvironmentPackageSummary, RunOptions, TaskOption } from "./types";
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
  normalizeSseEvent,
  parseRunOptions,
  parseTasks,
} from "./validate";

/** Structured API error carrying the SPEC error envelope for UI surfacing. */
export class ApiError extends Error {
  code: string;
  correlationId: string | null;
  retry: string | null;
  constructor(payload: Record<string, unknown>, status: number) {
    // FastAPI failure shapes: structured envelope in `detail` (409/403),
    // string `detail` (plain HTTPException), or array `detail` (validation).
    const raw = payload.detail;
    // FastAPI validation errors use an array of {loc, msg, type} objects.
    // Normalize the first actionable message into the same envelope used by
    // structured control-plane errors so callers render one error path.
    const arrayMessage = Array.isArray(raw)
      ? raw.find(
          (entry): entry is Record<string, unknown> =>
            typeof entry === "object" && entry !== null && typeof (entry as Record<string, unknown>).msg === "string",
        )
      : null;
    const envelope: Record<string, unknown> =
      raw && typeof raw === "object" && !Array.isArray(raw)
        ? (raw as Record<string, unknown>)
        : payload; // synthetic envelopes carry code/message at the top level
    const message =
      typeof envelope.message === "string" && envelope.message
        ? envelope.message
        : arrayMessage && typeof arrayMessage.msg === "string" && arrayMessage.msg
          ? arrayMessage.msg
          : typeof raw === "string" && raw
            ? raw
            : `HTTP ${status}`;
    // validation responses classify as INVALID_INPUT unless the control plane
    // provided a specific code
    const code =
      typeof envelope.code === "string"
        ? envelope.code
        : typeof payload.code === "string"
          ? payload.code
          : status === 422
            ? "INVALID_INPUT"
            : "UNKNOWN";
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

/** Friendly operator-facing text; raw JSON/HTML error bodies never reach the UI. */
export const DISCONNECTED_MESSAGE = "API unavailable — the control API did not respond.";

/**
 * Live transport against the local Python control API (same-origin /api proxy).
 * Session handshake: GET /session/bootstrap (HttpOnly cookie) then GET /session
 * run before any data request or SSE connection; failures surface as a clear
 * Disconnected state with a retry action, never an indefinite spinner.
 */
export function createRestTransport(baseUrl = "/api"): ConsoleTransport {
  let sessionPromise: Promise<void> | null = null;

  const ensureSession = (): Promise<void> => {
    if (sessionPromise) return sessionPromise;
    sessionPromise = (async () => {
      try {
        const bootstrap = await fetch(`${baseUrl}/session/bootstrap`);
        if (!bootstrap.ok) {
          throw new ApiError(await safeJson(bootstrap), bootstrap.status);
        }
        await fetch(`${baseUrl}/session`);
      } catch (error) {
        // allow a later retry to re-attempt the handshake
        sessionPromise = null;
        if (error instanceof ApiError) throw error;
        throw new ApiError({ code: "DISCONNECTED", message: DISCONNECTED_MESSAGE }, 0);
      }
    })();
    return sessionPromise;
  };

  async function safeJson(res: Response): Promise<Record<string, unknown>> {
    try {
      return (await res.json()) as Record<string, unknown>;
    } catch {
      return {};
    }
  }

  async function json(path: string, init?: RequestInit): Promise<unknown> {
    await ensureSession();
    let res: Response;
    try {
      res = await fetch(`${baseUrl}${path}`, {
        headers: { "content-type": "application/json" },
        ...init,
      });
    } catch {
      throw new ApiError({ code: "DISCONNECTED", message: DISCONNECTED_MESSAGE }, 0);
    }
    if (res.status === 401) {
      // the HttpOnly cookie may have expired across a server restart: reset the
      // cached session and re-authenticate once (safe GET bootstrap) before
      // failing
      sessionPromise = null;
      await ensureSession();
      try {
        res = await fetch(`${baseUrl}${path}`, {
          headers: { "content-type": "application/json" },
          ...init,
        });
      } catch {
        throw new ApiError({ code: "DISCONNECTED", message: DISCONNECTED_MESSAGE }, 0);
      }
    }
    if (!res.ok) {
      throw new ApiError(await safeJson(res), res.status);
    }
    if (res.status === 204) return undefined;
    try {
      return await res.json();
    } catch {
      throw new SchemaError("response envelope");
    }
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
    getRunOptions: () => validated(json("/run-options"), parseRunOptions),
    getEnvironmentTasks: (environmentId: string) =>
      validated(json(`/environments/${encodeURIComponent(environmentId)}/tasks`), parseTasks),
    listRuns: () => validated(json("/runs"), parseRuns),
    listSkills: () => validated(json("/skills"), parseSkills),
    listCandidates: () => validated(json("/candidates"), parseCandidates),

    async reconnect() {
      sessionPromise = null;
      await ensureSession();
    },

    openRunStream(runId, fromCursor, { onEvent, onState }) {
      let es: EventSource | null = null;
      let retryTimer: ReturnType<typeof setTimeout> | null = null;
      let closed = false;
      let lastSequence = fromCursor;
      let consecutiveFailures = 0;

      const start = () => {
        if (closed) return;
        // environments without EventSource support surface an honest stale state
        if (typeof EventSource === "undefined") {
          onState("stale");
          return;
        }
        onState(lastSequence === fromCursor ? "live" : "reconnecting");
        es = new EventSource(`${baseUrl}/runs/${encodeURIComponent(runId)}/events?cursor=${lastSequence}`);
        es.onmessage = (message) => {
          try {
            // boundary validation before the event enters reducer state; both
            // wire shapes are accepted (plane RunEvent and durable evidence
            // envelope {id, event, data})
            const event = normalizeSseEvent(JSON.parse(message.data), "sse.data");
            if (event.sequence > lastSequence) {
              lastSequence = event.sequence;
              consecutiveFailures = 0;
              onEvent(event);
            }
            // stale duplicates (sequence <= cursor) are dropped client-side too
          } catch {
            onState("stale");
          }
        };
        es.onerror = () => {
          es?.close();
          // EOF handling: a terminal run's stream closes server-side. Check the
          // run's status once; close normally when terminal, otherwise bounded
          // retry with a fresh session (server restart invalidates the cookie)
          void (async () => {
            let terminal = false;
            try {
              const run = (await json(`/runs/${encodeURIComponent(runId)}`)) as { status?: unknown };
              terminal =
                run?.status === "succeeded" ||
                run?.status === "failed" ||
                run?.status === "cancelled" ||
                run?.status === "timed_out";
            } catch {
              terminal = false; // unreachable -> bounded retry path below
            }
            if (terminal) {
              onState("closed");
              return;
            }
            if (consecutiveFailures >= 5) {
              onState("stale");
              return;
            }
            consecutiveFailures += 1;
            onState("reconnecting");
            retryTimer = setTimeout(() => {
              sessionPromise = null; // server restart may have rotated the cookie
              ensureSession()
                .then(start)
                .catch(() => onState("disconnected"));
            }, 1500);
          })();
        };
      };

      // the HttpOnly session cookie must exist before SSE connects
      ensureSession()
        .then(start)
        .catch(() => onState("disconnected"));

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
      // Refs are authoritative and NEVER computed in the browser: the model
      // ref comes from the server's /run-options projection. The budget is
      // sent as the validated object from the operator's controls; the
      // backend hashes and stores it as the run's budget artifact.
      if (!input.modelProfileRef || !input.modelProfileRef.sha256) {
        throw new SchemaError("run modelProfileRef must come from the /run-options projection");
      }
      const taskRef: Record<string, unknown> = { goal: input.goal, environmentId: input.environmentId };
      if (input.taskId) taskRef.id = input.taskId;
      const body: Record<string, unknown> = {
        taskRef,
        modelProfileRef: input.modelProfileRef,
        // authoritative trusted budget ref from /run-options, submitted
        // verbatim (never computed client-side); the budget object carries
        // the operator's validated values for the durable artifact path
        ...(input.budgetRef ? { budgetRef: input.budgetRef } : {}),
        budget: {
          modelTokens: input.budget.modelTokenCeiling,
          toolCalls: input.budget.toolCallCeiling,
          childRuns: 0,
          wallTimeSeconds: input.budget.wallSecondsCeiling,
          costMicrounits: 100000,
          currency: "USD",
        },
        idempotencyKey: input.idempotencyKey,
        executionMode: input.executionMode,
      };
      const run = await validated(json("/runs", { method: "POST", body: JSON.stringify(body) }), (value) =>
        parseRun(value, "run"),
      );
      // explicit launch step: a created run must not remain queued
      await json(`/runs/${encodeURIComponent(run.runId)}/launch`, { method: "POST", body: "{}" });
      return run;
    },

    launchRun: (runId) =>
      json(`/runs/${encodeURIComponent(runId)}/launch`, { method: "POST", body: "{}" }).then(() => undefined),

    async registerEnvironment(manifest: EnvironmentRegistration) {
      await json("/environments/register", { method: "POST", body: JSON.stringify(manifest) });
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
