import type { RunEvent, RunRecord, SkillVersionSummary, CandidateDiff, EnvironmentPackageSummary, RunOptions, TaskOption, EvaluationJob, EvaluationReportProjection } from "./types";
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

const EVAL_STATES = ["queued", "running", "valid", "invalid", "cancelled", "completed", "decided"];

function parseReport(value: unknown, field: string): EvaluationReportProjection {
  const r = (value ?? {}) as Record<string, unknown>;
  const str = (key: string): string | undefined => (typeof r[key] === "string" ? (r[key] as string) : undefined);
  const bool = (key: string): boolean | undefined => (typeof r[key] === "boolean" ? (r[key] as boolean) : undefined);
  const numOrNull = (key: string): number | null | undefined =>
    typeof r[key] === "number" ? (r[key] as number) : r[key] === null ? null : undefined;
  const armSummaries: Record<string, EvaluationReportProjection["armSummaries"][string]> = {};
  if (r.armSummaries && typeof r.armSummaries === "object" && !Array.isArray(r.armSummaries)) {
    for (const [arm, raw] of Object.entries(r.armSummaries as Record<string, unknown>)) {
      const a = (raw ?? {}) as Record<string, unknown>;
      const n = (key: string): number => (typeof a[key] === "number" ? (a[key] as number) : 0);
      armSummaries[arm] = {
        accuracy: n("accuracy"),
        reliability: n("reliability"),
        meanCostMicrounits: n("meanCostMicrounits"),
        medianLatencySeconds: n("medianLatencySeconds"),
        p95LatencySeconds: n("p95LatencySeconds"),
        safetyViolations: n("safetyViolations"),
        count: n("count"),
      };
    }
  } else {
    throw new SchemaError(`${field}.armSummaries`);
  }
  const report: EvaluationReportProjection = {
    comparison: str("comparison") ?? "unknown",
    validityStatus: str("validityStatus") ?? "unknown",
    promotionEligible: bool("promotionEligible") ?? false,
    armSummaries,
  };
  if (r.candidateHash !== undefined) report.candidateHash = str("candidateHash");
  if (r.baseHash !== undefined) report.baseHash = str("baseHash");
  if (r.protocolHash !== undefined) report.protocolHash = str("protocolHash");
  if (Array.isArray(r.confidenceIntervals)) {
    report.confidenceIntervals = (r.confidenceIntervals as Record<string, unknown>[]).map((ci, i) => ({
      metric: typeof ci.metric === "string" ? ci.metric : `metric-${i}`,
      point: typeof ci.point === "number" ? ci.point : 0,
      lower95: typeof ci.lower95 === "number" ? ci.lower95 : 0,
      upper95: typeof ci.upper95 === "number" ? ci.upper95 : 0,
      draws: typeof ci.draws === "number" ? ci.draws : undefined,
      analysisSeed: typeof ci.analysisSeed === "number" ? ci.analysisSeed : undefined,
    }));
  }
  const safetyPassed = bool("safetyPassed");
  if (safetyPassed !== undefined) report.safetyPassed = safetyPassed;
  if (r.missingPairs !== undefined && typeof r.missingPairs === "number") report.missingPairs = r.missingPairs;
  const metricCellsComplete = bool("metricCellsComplete");
  if (metricCellsComplete !== undefined) report.metricCellsComplete = metricCellsComplete;
  const safetyCellsComplete = bool("safetyCellsComplete");
  if (safetyCellsComplete !== undefined) report.safetyCellsComplete = safetyCellsComplete;
  const modelProvenanceComplete = bool("modelProvenanceComplete");
  if (modelProvenanceComplete !== undefined) report.modelProvenanceComplete = modelProvenanceComplete;
  if (Array.isArray(r.infrastructureFailures)) {
    report.infrastructureFailures = (r.infrastructureFailures as unknown[]).map((f, i) =>
      typeof f === "string" ? f : `failure-${i}`,
    );
  }
  if (r.analysisSeed !== undefined && typeof r.analysisSeed === "number") report.analysisSeed = r.analysisSeed;
  report.nominalCostUsd = numOrNull("nominalCostUsd");
  report.actualInputTokens = numOrNull("actualInputTokens");
  report.actualOutputTokens = numOrNull("actualOutputTokens");
  report.wallDurationSeconds = numOrNull("wallDurationSeconds");
  if (r.billingBasis !== undefined && typeof r.billingBasis === "string") report.billingBasis = r.billingBasis;
  return report;
}

function arr2evals(value: unknown): EvaluationJob[] {
  if (!Array.isArray(value)) throw new SchemaError("evaluations");
  return value.map((item, i) => {
    const o = (item ?? {}) as Record<string, unknown>;
    const evaluationId = typeof o.evaluationId === "string" ? o.evaluationId : `unknown-${i}`;
    const candidateId = typeof o.candidateId === "string" && o.candidateId ? o.candidateId : null;
    const rawState = typeof o.state === "string" ? o.state : "";
    // legacy/malformed rows (missing candidateId, unrecognized states) are
    // preserved explicitly UNVERIFIED — trusted:true from an unverifiable row
    // is NEVER normalized into performance evidence
    const verified = candidateId !== null && EVAL_STATES.includes(rawState);
    const job: EvaluationJob = {
      evaluationId,
      candidateId,
      state: verified ? (rawState as EvaluationJob["state"]) : "unverified",
      verified,
    };
    if (typeof o.trusted === "boolean") job.trusted = o.trusted;
    if (typeof o.reason === "string") job.reason = o.reason;
    if (o.validity === "invalid" || o.promotionEligible === false) {
      job.reason = job.reason ?? `legacy record (validity: ${String(o.validity ?? "unknown")})`;
    }
    if (o.error && typeof o.error === "object" && !Array.isArray(o.error)) {
      const err = o.error as Record<string, unknown>;
      if (typeof err.message === "string") job.reason = err.message;
    }
    // canonical trusted report projection (96a84d3/8fcb18a): rendered as actual
    // evaluation details, never fabricated; unbound legacy rows stay unverified
    if (o.report !== undefined) {
      if (!verified) {
        job.reason = job.reason ?? "report present but row is not canonically bound — unverified";
      } else {
        try {
          job.report = parseReport(o.report, `evaluations[${i}].report`);
        } catch (error) {
          job.reason = `report projection incomplete: ${(error as Error).message}`;
        }
      }
    }
    return job;
  });
}

/** Friendly operator-facing text; raw JSON/HTML error bodies never reach the UI. */
export const DISCONNECTED_MESSAGE = "API unavailable — the control API did not respond.";

/**
 * Live transport against the local Python control API (same-origin /api proxy).
 * Session handshake: GET /session/bootstrap (HttpOnly cookie) then GET /session
 * run before any data request or SSE connection; failures surface as a clear
 * Disconnected state with a retry action, never an indefinite spinner.
 */
/**
 * API base selection:
 * - Vite dev server: "/api" (same-origin proxy to 127.0.0.1:8000; the proxy
 *   keeps the console origin so the loopback/same-origin access boundary and
 *   the HttpOnly operator cookie work).
 * - Production build: "" (direct same-origin routes — create_runtime_app
 *   serves the console at / and the API at its own paths).
 * An explicit baseUrl argument always wins (tests / alternative deployments).
 */
export function defaultApiBase(): string {
  return import.meta.env.PROD ? "" : "/api";
}

export function createRestTransport(baseUrl: string = defaultApiBase()): ConsoleTransport {
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
        // cleanup is NOT an EOF event: emit no connection state so the app can
        // distinguish operator/unmount teardown from a genuine terminal close
        closed = true;
        if (retryTimer) clearTimeout(retryTimer);
        es?.close();
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

    launchEvaluation: async (input: { candidateId: string; baseBundleHash: string }) => {
      // protocolHash/partitionRef are control-plane-owned; the backend derives
      // or rejects until the contract relaxation lands (coordinated with 2)
      const action = (await json("/evaluations", {
        method: "POST",
        body: JSON.stringify({ candidateId: input.candidateId, baseBundleHash: input.baseBundleHash }),
      })) as Record<string, unknown>;
      const evaluationId = typeof action.evaluationId === "string" ? action.evaluationId : "";
      const state = typeof action.state === "string" ? action.state : "queued";
      if (!evaluationId) throw new SchemaError("evaluation.evaluationId");
      return { evaluationId, state };
    },

    listEvaluations: () =>
      validated(json("/evaluations"), (value) =>
        arr2evals(value),
      ),

    launchLearningCycle: async (input: LearningCycleInput) => {
      const action = (await json("/learning/launch", {
        method: "POST",
        body: JSON.stringify({ runId: input.runId }),
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
