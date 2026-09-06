/**
 * Truthful source labeling for environment/task data provenance.
 *
 * One small typed mapping used by the registry cards, task-selection dialog,
 * and the selected-run workflow. No backend field is required: the label is
 * derived from the environmentId the server already projects.
 *
 * Identification is by EXACT registered id only: the known built-in catalog
 * ids (finance, customer_support, it, lab_scheduling) plus the explicit -sim
 * variants used by the console's development fixtures. Arbitrary user
 * packages like finance-external or appworld-custom are NOT labeled as
 * built-in fixture or AppWorld — they fall back to "source not specified".
 */
export const BUILTIN_CATALOG_IDS = ["finance", "customer_support", "it", "lab_scheduling"] as const;
const APPWORLD_ID = "appworld";
/** explicit development-fixture ids used by the console's simulation transport */
const SIM_VARIANTS = ["finance-sim", "customer_support-sim", "it-sim", "lab_scheduling-sim"];

export type SourceLabel = {
  /** compact tag for cards/badges */
  tag: string;
  /** operator-facing provenance sentence */
  provenance: string;
};

export function sourceLabelFor(environmentId: string | undefined | null): SourceLabel {
  const id = (environmentId ?? "").toLowerCase();
  if (!id) {
    return {
      tag: "source not specified",
      provenance: "Data source not specified.",
    };
  }
  if (id === APPWORLD_ID) {
    return {
      tag: "AppWorld",
      provenance: "Tasks use AppWorld, a published benchmark with simulated app data.",
    };
  }
  if (BUILTIN_CATALOG_IDS.includes(id as (typeof BUILTIN_CATALOG_IDS)[number]) || SIM_VARIANTS.includes(id)) {
    return {
      tag: "simulated business fixture",
      provenance: "Tasks use the built-in simulated business fixture catalog.",
    };
  }
  return {
    tag: "source not specified",
    provenance: "Data source not specified.",
  };
}
