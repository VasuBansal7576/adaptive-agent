import type { ToolError, ToolErrorCode } from "./types";

/**
 * Safe operator-facing recovery guidance per SPEC error semantics.
 * Never embeds credentials or hidden evaluator data.
 */
export const RECOVERY_GUIDANCE: Record<ToolErrorCode, string> = {
  INVALID_INPUT: "Check the highlighted fields; your input is preserved. Correct and resubmit.",
  FORBIDDEN: "This action is outside the run's policy scope. Request the needed capability from the environment owner.",
  VERSION_CONFLICT: "The base version changed. The proposal was superseded; submit a new candidate against the current active version.",
  IDEMPOTENCY_CONFLICT: "This operation key was already used with different arguments. Review the stored operation and submit a new operation with a fresh key; nothing was dispatched.",
  BUDGET_EXHAUSTED: "The run budget is exhausted. Raise the ceiling in the run profile before restarting.",
  TOOL_UNAVAILABLE: "The tool is temporarily unavailable. Retry after the provider recovers; no side effects occurred.",
  OUTCOME_UNKNOWN: "An external effect could not be confirmed. The broker will reconcile before further state changes; do not repeat the operation manually.",
};

export function describeToolError(error: ToolError): string {
  const guidance = RECOVERY_GUIDANCE[error.code] ?? "Contact the operator.";
  return `${error.message} (correlation ${error.correlationId}) — ${guidance}`;
}
