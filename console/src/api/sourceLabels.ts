/**
 * Truthful source labeling for environment/task data provenance.
 *
 * One small typed mapping used by the registry cards, task-selection dialog,
 * and the selected-run workflow. No backend field is required: the label is
 * derived from the environmentId the server already projects.
 *
 * - Known built-in catalog ids (finance, customer_support, it, lab_scheduling,
 *   including their "-sim" development-fixture variants) are labeled as
 *   built-in simulated business fixtures.
 * - "appworld" is an external published simulated benchmark: results are
 *   never real customer data and no completed benchmark claims are made.
 * - Anything else: source not specified — never labeled as built-in fixture.
 */
export const BUILTIN_CATALOG_IDS = ["finance", "customer_support", "it", "lab_scheduling"] as const;
const APPWORLD_ID = "appworld";

export type SourceLabel = {
  /** compact tag for cards/badges */
  tag: string;
  /** operator-facing provenance sentence */
  provenance: string;
};

function matchesCatalog(id: string, known: string): boolean {
  return id === known || id.startsWith(`${known}-`);
}

export function sourceLabelFor(environmentId: string | undefined | null): SourceLabel {
  const id = (environmentId ?? "").toLowerCase();
  if (!id) {
    return {
      tag: "source not specified",
      provenance: "Environment source not specified; do not treat results as externally sourced.",
    };
  }
  if (id === APPWORLD_ID || id.startsWith(`${APPWORLD_ID}-`)) {
    return {
      tag: "AppWorld",
      provenance:
        "Tasks come from AppWorld, an external published simulated benchmark. Results are never real customer data, and no completed benchmark claims are made.",
    };
  }
  if (BUILTIN_CATALOG_IDS.some((known) => matchesCatalog(id, known))) {
    return {
      tag: "simulated business fixture",
      provenance:
        "Tasks come from the built-in simulated business fixture catalog; results are not from external datasets, and model execution does not change data provenance.",
    };
  }
  return {
    tag: "source not specified",
    provenance: "Environment source not specified; do not treat results as externally sourced.",
  };
}
