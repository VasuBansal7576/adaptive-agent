"""Candidate lifecycle, deterministic promotion gate, rollback, protected active memory.

The promotion service is the only authority that can change the active bundle
pointer. It:

- recomputes all content hashes rather than trusting caller-supplied ones;
- requires supporting evidence bound to verified DEVELOPMENT-partition runs;
- accepts only reports whose protocol was frozen and registered by a trusted
  evaluator identity — callers cannot choose or smuggle a gate;
- applies the frozen gate mechanically inside a compare-and-swap transaction
  covering the active pointer, the candidate state, and the decision record;
- keeps the candidates.state column and candidate_json payload synchronized;
- restricts rollback to hashes present in the recorded active lineage.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from adaptive_agent.models import (
    ArtifactRef,
    CandidateProposal,
    CandidateState,
    EvaluationReport,
    EvaluationState,
    PromotionDecision,
    PromotionGate,
    SkillBundle,
    sha256_json,
)
from adaptive_agent.store import Store


class CandidateValidationError(ValueError):
    pass


class PromotionError(ValueError):
    pass


# Canonical Track 1 frozen protocol (evaluator contract c9fa183). Once the
# canonical protocol is registered, reports under any other protocol hash are
# stale and refused.
TRACK1_PROTOCOL_HASH = (
    "d208150dd3a0b9b6be6edc88292b7832b0893078dae8ced487ef875b09601ec8"
)
TRACK1_FIXTURE_HASHES = {
    "finance": "65cd6f5e4f1c8a37faa90e1e3a1b8ed5aac5b89e409b57237a57134f3c5b8c20",
    "support": "976a7c2cee5c0d589db914068d8b1be9042d0c4eb5042fc4514e4bfc2d714db6",
    "it": "26595f414b83412afc0cef8ba3d73f934c85c2a1173e55d25eb92df18f6fb6dd",
    "lab": "1d26c077503588e80a40b76713728034407b467d2c95254e9ff83c453281e8a2",
}


# A verifier callback injected by the trusted-evaluator owner (session6). It is
# called with the raw report payload and must return True only when the report's
# attestation token verifies against the evaluator's private attestation ledger.
ReportVerifier = Callable[[Mapping[str, Any]], bool]


@dataclass(frozen=True)
class _ReportView:
    """Normalized gate-consumable view of an evaluator report.

    Supports both the local EvaluationReport model and the finalized external
    evaluator contract (session6 `EvaluationReport.to_dict()` payload).
    """

    report_id: str
    protocol_hash: str
    candidate_hash: str
    base_hash: str
    provenance: str
    accuracy: float
    reliability: float
    mean_cost: float
    p95_latency: float
    safety_violations: int
    suspicious: bool
    safety_results: dict[str, bool]
    paired_runs: list[Any]
    partition_ref: ArtifactRef
    uncertainty: dict[str, Any]
    raw: Any


class CandidateManager:
    """Owns candidate validation, evaluation state, promotion, and rollback."""

    MAX_CHANGED_ARTIFACTS = 3
    MAX_CHANGED_LINES = 200
    FORBIDDEN_PATTERNS = (
        "import os",
        "import sys",
        "__import__",
        "subprocess",
        "eval(",
        "exec(",
        "open(",
        "socket",
    )
    TRUSTED_COMPONENTS = ("broker", "evaluator", "promotion", "policy", "store", "coordinator")

    def __init__(self, store: Store, report_verifier: ReportVerifier | None = None) -> None:
        self.store = store
        self.report_verifier = report_verifier

    # ------------------------------------------------------------------ bundles
    def _active_bundle(self) -> SkillBundle | None:
        row = self.store.get_active_bundle()
        if not row:
            return None
        return SkillBundle.model_validate_json(row["bundle_json"])

    def _bundle_by_hash(self, content_hash: str) -> SkillBundle | None:
        row = self.store.get_bundle_by_hash(content_hash)
        if not row:
            return None
        return SkillBundle.model_validate_json(row["bundle_json"])

    def recompute_bundle_hash(self, bundle: SkillBundle) -> str:
        """Canonical content hash over the payload, excluding the hash field itself."""
        payload = bundle.model_dump(mode="json", by_alias=True, exclude={"content_hash"})
        return sha256_json(payload)

    def initialize_active_bundle(self, bundle: SkillBundle) -> None:
        """Seed protected active memory. Records genesis in the active lineage."""
        bundle.content_hash = self.recompute_bundle_hash(bundle)
        self.store.save_bundle(
            bundle.bundle_id,
            bundle.parent,
            bundle.content_hash,
            bundle.model_dump_json(by_alias=True),
            is_active=True,
        )
        self.store.append_active_history(bundle.content_hash, "genesis")

    # ------------------------------------------------------------------ validation
    def _validate_evidence(self, proposal: CandidateProposal) -> None:
        if not proposal.supporting_evidence_ids:
            raise CandidateValidationError("candidate lacks supporting evidence")
        for ev_id in proposal.supporting_evidence_ids:
            prov = self.store.evidence_provenance(ev_id)
            if prov is None:
                raise CandidateValidationError(f"evidence {ev_id!r} not found")
            if prov["partition"] != "development":
                raise CandidateValidationError(
                    f"evidence {ev_id!r} is from partition {prov['partition']!r}, "
                    "only development runs can motivate a candidate"
                )
            if prov["trust_class"] not in ("learner", "broker", "system"):
                raise CandidateValidationError(
                    f"evidence {ev_id!r} has untrusted class {prov['trust_class']!r}"
                )
            if not self.store.has_trusted_outcome(prov["run_id"]):
                raise CandidateValidationError(
                    f"evidence {ev_id!r} comes from run {prov['run_id']!r} with no "
                    "trusted evaluator outcome"
                )

    def validate_candidate(self, proposal: CandidateProposal, candidate_bundle: SkillBundle) -> None:
        """Check evidence provenance, edit bounds, imports, and trusted-component tampering."""
        if proposal.state != CandidateState.draft:
            raise CandidateValidationError("candidate is not in draft state")

        # Recompute content hash; never trust the caller's supplied value.
        computed = self.recompute_bundle_hash(candidate_bundle)
        if candidate_bundle.content_hash and candidate_bundle.content_hash != computed:
            raise CandidateValidationError("candidate bundle content_hash does not match payload")
        candidate_bundle.content_hash = computed
        if proposal.candidate_bundle_hash and proposal.candidate_bundle_hash != computed:
            raise CandidateValidationError("proposal candidate_bundle_hash does not match bundle")

        if len(proposal.changed_artifact_hashes) > self.MAX_CHANGED_ARTIFACTS:
            raise CandidateValidationError("too many changed artifacts")
        for h in proposal.changed_artifact_hashes:
            if not self.store.has_artifact(h):
                raise CandidateValidationError(f"changed artifact {h} not content-addressed")

        for skill in candidate_bundle.skills:
            changed = sum(1 for line in skill.procedure.splitlines() if line.strip())
            if changed > self.MAX_CHANGED_LINES:
                raise CandidateValidationError("procedure exceeds changed-line limit")
            for pattern in self.FORBIDDEN_PATTERNS:
                if pattern in skill.procedure:
                    raise CandidateValidationError(f"forbidden pattern in skill: {pattern}")

        self._validate_evidence(proposal)

        for op in proposal.edit_operations:
            if any(name in op for name in self.TRUSTED_COMPONENTS):
                raise CandidateValidationError("candidate cannot modify trusted components")

        active = self._active_bundle()
        if active is None:
            raise CandidateValidationError("no active bundle to diff against")
        if proposal.base_bundle_hash != active.content_hash:
            raise CandidateValidationError("candidate base does not match active bundle")

    def submit_candidate(self, proposal: CandidateProposal, candidate_bundle: SkillBundle) -> CandidateProposal:
        """Validate a draft candidate, then persist it in `validated` state."""
        self.validate_candidate(proposal, candidate_bundle)
        proposal.candidate_bundle_hash = candidate_bundle.content_hash
        proposal.state = CandidateState.validated
        self._sync_candidate(proposal)
        self.store.save_bundle(
            candidate_bundle.bundle_id,
            candidate_bundle.parent,
            candidate_bundle.content_hash,
            candidate_bundle.model_dump_json(by_alias=True),
            is_active=False,
        )
        return proposal

    def _sync_candidate(self, proposal: CandidateProposal) -> None:
        """Persist state and payload together so candidates.state == candidate_json.state."""
        self.store.update_candidate(
            proposal.candidate_id,
            proposal.state.value,
            proposal.model_dump_json(by_alias=True),
        ) if self.store.get_candidate(proposal.candidate_id) else self.store.save_candidate(
            proposal.candidate_id,
            {
                "base_bundle_hash": proposal.base_bundle_hash,
                "candidate_bundle_hash": proposal.candidate_bundle_hash,
                "candidate_json": proposal.model_dump_json(by_alias=True),
                "state": proposal.state.value,
                "created_at": proposal.created_at.isoformat(),
            },
        )

    def _get_candidate(self, candidate_id: str) -> CandidateProposal:
        row = self.store.get_candidate(candidate_id)
        if not row:
            raise CandidateValidationError("candidate not found")
        proposal = CandidateProposal.model_validate_json(row["candidate_json"])
        # Keep the returned object's state consistent with the column.
        proposal.state = CandidateState(row["state"])
        return proposal

    def start_evaluation(self, candidate_id: str) -> CandidateProposal:
        proposal = self._get_candidate(candidate_id)
        if proposal.state != CandidateState.validated:
            raise CandidateValidationError("candidate must be validated before evaluation")
        proposal.state = CandidateState.evaluating
        self._sync_candidate(proposal)
        return proposal

    def quarantine(self, candidate_id: str, reason: str) -> CandidateProposal:
        proposal = self._get_candidate(candidate_id)
        proposal.state = CandidateState.quarantined
        self._sync_candidate(proposal)
        return proposal

    # ------------------------------------------------------------------ frozen protocol
    def freeze_protocol(
        self,
        gate: PromotionGate,
        evaluator_id: str,
        *,
        evaluator_refs: list[str] | None = None,
        fixture_hashes: dict[str, str] | None = None,
        partition_hashes: dict[str, str] | None = None,
        protocol_inputs: dict[str, Any] | None = None,
        phase_evaluator_refs: dict[str, list[str]] | None = None,
    ) -> str:
        """Register a frozen promotion gate bound to a trusted evaluator identity.

        Records the frozen fixture/partition hashes and registered evaluator refs
        from the finalized evaluator contract so stale or foreign reports are
        rejected. This is the ONLY way a gate can later be used — callers of
        promote() cannot supply a gate object.
        """
        if not gate.protocol_hash:
            gate.protocol_hash = sha256_json(
                gate.model_dump(mode="json", by_alias=True, exclude={"protocol_hash"})
            )
        if gate.protocol_hash == TRACK1_PROTOCOL_HASH and fixture_hashes is not None and dict(fixture_hashes) != TRACK1_FIXTURE_HASHES:
            raise PromotionError(
                "canonical Track 1 protocol must pin the official fixture hashes"
            )
        self.store.save_frozen_protocol(
            gate.protocol_hash,
            gate.model_dump_json(by_alias=True),
            evaluator_id,
            evaluator_refs=evaluator_refs,
            fixture_hashes=fixture_hashes,
            partition_hashes=partition_hashes,
            protocol_inputs=protocol_inputs,
            phase_evaluator_refs=phase_evaluator_refs,
        )
        return gate.protocol_hash

    def freeze_track1_protocol(
        self,
        evaluator_id: str,
        *,
        evaluator_refs: list[str] | None = None,
        partition_hashes: dict[str, str] | None = None,
    ) -> str:
        """Register the canonical Track 1 protocol with its pinned fixture hashes."""
        return self.freeze_protocol(
            PromotionGate(protocolHash=TRACK1_PROTOCOL_HASH),
            evaluator_id,
            evaluator_refs=evaluator_refs,
            fixture_hashes=TRACK1_FIXTURE_HASHES,
            partition_hashes=partition_hashes,
        )

    def _load_frozen_gate(self, view: _ReportView) -> PromotionGate:
        # Once the canonical protocol is registered, any other protocol hash is stale.
        if self.store.get_frozen_protocol(TRACK1_PROTOCOL_HASH) is not None and view.protocol_hash != TRACK1_PROTOCOL_HASH:
            raise PromotionError(
                "stale report: promotion requires the canonical Track 1 protocol"
            )
        row = self.store.get_frozen_protocol(view.protocol_hash)
        if row is None:
            raise PromotionError(
                f"no active frozen protocol registered for protocol_hash "
                f"{view.protocol_hash!r}; stale or unregistered report"
            )
        registered_refs = set(json.loads(row["evaluator_refs_json"] or "[]"))
        report_refs = set(view.uncertainty.get("evaluator_refs") or [])
        if registered_refs:
            if not report_refs or report_refs != registered_refs:
                raise PromotionError(
                    "report evaluator_refs do not match the frozen protocol registration"
                )
        # Identity: exact registered-ref match suffices (refs are the evaluator
        # identity); otherwise the report provenance must equal evaluator_id.
        identity_ok = view.provenance == row["evaluator_id"] or bool(registered_refs)
        if not identity_ok:
            raise PromotionError(
                "report provenance does not match the registered evaluator identity"
            )
        frozen_partitions = json.loads(row["partition_hashes_json"] or "{}")
        report_partitions = view.uncertainty.get("partition_hashes") or {}
        if frozen_partitions and report_partitions and dict(report_partitions) != frozen_partitions:
            raise PromotionError(
                "report partition hashes do not match the frozen allocation"
            )
        return PromotionGate.model_validate_json(row["gate_json"])

    # ------------------------------------------------------------------ report normalization
    def _view_report(self, report: EvaluationReport | Mapping[str, Any]) -> _ReportView:
        """Normalize a report into the gate-consumable view, enforcing the
        finalized evaluator contract's required cells before any gate math."""
        if isinstance(report, EvaluationReport):
            if report.validity != EvaluationState.valid:
                raise PromotionError("evaluation report is not valid")
            if report.paired_run_ids and not all(len(p) == 2 for p in report.paired_run_ids):
                raise PromotionError("paired runs are malformed")
            m = report.metrics
            if None in (m.accuracy, m.reliability, m.mean_cost, m.p95_latency_ms):
                raise PromotionError("report missing required metric cells")
            return _ReportView(
                report_id=report.report_id,
                protocol_hash=report.protocol_hash,
                candidate_hash=report.candidate_hash,
                base_hash=report.base_hash or "",
                provenance=report.evaluator_provenance,
                accuracy=float(m.accuracy or 0.0),
                reliability=float(m.reliability or 0.0),
                mean_cost=float(m.mean_cost or 0.0),
                p95_latency=float(m.p95_latency_ms or 0.0),
                safety_violations=m.safety_violations,
                suspicious=bool(report.safety_results.get("suspicious")),
                safety_results=dict(report.safety_results),
                paired_runs=list(report.paired_run_ids),
                partition_ref=report.partition_ref,
                uncertainty={
                    **report.uncertainty,
                    "evaluator_refs": [report.evaluator_provenance],
                },
                raw=report,
            )

        if not isinstance(report, Mapping):
            raise PromotionError("unsupported report type")

        # Session6 contract: EvaluationReport.to_dict() payload.
        required = (
            "protocolHash", "candidateHash", "baseHash", "validityStatus",
            "promotionEligible", "safetyPassed",
            "metricCellsComplete", "safetyCellsComplete", "modelProvenanceComplete",
            "attestation", "evaluatorRefs", "partitionHashes", "armSummaries",
            "environmentCells", "confidenceIntervals", "missingPairs",
            "partitionLeak", "invalidFixtureResets", "infrastructureFailures",
            "comparison",
        )
        missing = [k for k in required if k not in report]
        if missing:
            raise PromotionError(f"report missing required cells: {missing}")
        if report["validityStatus"] != "valid":
            raise PromotionError("evaluation report is not valid")
        if report.get("modelProvenanceValid") is False:
            raise PromotionError("synthetic-model provenance is not promotable")
        if not report["promotionEligible"]:
            raise PromotionError(
                "report is not promotion-eligible (missing cells, safety, or "
                "synthetic-model provenance)"
            )
        safety_cases = dict(report.get("safetyCaseResults") or {})
        if not report["safetyPassed"] or not (safety_cases or report["safetyCellsComplete"]):
            raise PromotionError("report missing required safety cells")
        if report["missingPairs"] or report["partitionLeak"] or report["invalidFixtureResets"] or report["infrastructureFailures"]:
            raise PromotionError("report contains invalid or leaked evidence")
        if not report["attestation"]:
            raise PromotionError("report is not attested by a trusted evaluator")
        if self.report_verifier is None:
            raise PromotionError("report attestation verifier is not configured")
        if not self.report_verifier(report):
            raise PromotionError("report attestation failed verification")

        arms = report["armSummaries"]
        b0 = arms.get("B0") or {}
        learned = arms.get("L") or {}
        if not learned or "accuracy" not in learned:
            raise PromotionError("report missing learned-arm metric cells")

        # Confidence intervals: use the accuracy lower bound as the gate CI.
        ci_lower = 0.0
        for est in report["confidenceIntervals"]:
            if est.get("metric") == "accuracy":
                ci_lower = est.get("lower95", 0.0)
                break

        env_cells = report["environmentCells"]
        per_env = {}
        for env, cells in env_cells.items():
            per_env[env] = {
                "baseline_accuracy": (cells.get("B0") or {}).get("accuracy", 0.0),
                "candidate_accuracy": (cells.get("L") or {}).get("accuracy", 0.0),
                "baseline_reliability": (cells.get("B0") or {}).get("reliability", 0.0),
                "candidate_reliability": (cells.get("L") or {}).get("reliability", 0.0),
            }

        suspicious = any(
            name == "suspicious" or (isinstance(v, bool) and v is False and "suspicious" in name)
            for name, v in safety_cases.items()
        )
        safety_results = safety_cases or {"safety_cells_complete": True}
        return _ReportView(
            report_id=str(report.get("reportId") or report.get("analysisSeed", "report")),
            protocol_hash=str(report["protocolHash"]),
            candidate_hash=str(report["candidateHash"]),
            base_hash=str(report["baseHash"] or ""),
            provenance="|".join(sorted(report["evaluatorRefs"])) if report["evaluatorRefs"] else "",
            accuracy=float(learned.get("accuracy", 0.0)),
            reliability=float(learned.get("reliability", 0.0)),
            mean_cost=float(learned.get("meanCostMicrounits", 0.0)),
            p95_latency=float(learned.get("p95LatencySeconds", 0.0)),
            safety_violations=int(learned.get("safetyViolations", 0)),
            suspicious=suspicious,
            safety_results=safety_results,
            paired_runs=[[a, b] for a, b in [("B0", "L")]],
            partition_ref=ArtifactRef(
                id="frozen-partition",
                version=str(report.get("analysisSeed", "0")),
                sha256=sha256_json(report["partitionHashes"]),
            ),
            uncertainty={
                "baseline_accuracy": float(b0.get("accuracy", 0.0)),
                "ci_lower": ci_lower,
                "baseline_cost": float(b0.get("meanCostMicrounits", 0.0)),
                "baseline_latency": float(b0.get("p95LatencySeconds", 0.0)),
                "per_environment": per_env,
                "evaluator_refs": sorted(report["evaluatorRefs"]),
                "partition_hashes": dict(report["partitionHashes"]),
                "comparison": report["comparison"],
            },
            raw=report,
        )

    def _validate_report(self, proposal: CandidateProposal, view: _ReportView) -> None:
        if not proposal.candidate_bundle_hash:
            raise PromotionError("candidate bundle hash not recorded")
        if view.candidate_hash != proposal.candidate_bundle_hash:
            raise PromotionError("report candidate hash does not match submitted candidate")
        if view.base_hash != proposal.base_bundle_hash:
            raise PromotionError("report base hash does not match candidate base")
        if not view.safety_results:
            raise PromotionError("report missing required safety cells")
        if not view.paired_runs:
            raise PromotionError("report has no paired runs")
        if not view.provenance:
            raise PromotionError("report missing evaluator provenance")

    # ------------------------------------------------------------------ promotion
    def promote(self, candidate_id: str, report: EvaluationReport | Mapping[str, Any]) -> PromotionDecision:
        """Apply the frozen gate and atomically CAS the active pointer.

        The gate is loaded from the frozen protocol keyed by report protocol_hash;
        callers may not supply a gate. Stale reports (inactive/unknown protocol,
        mismatched evaluator identity/refs, mismatched frozen partition hashes,
        missing attestation or incomplete cells) are refused before any gate math.
        """
        proposal = self._get_candidate(candidate_id)
        if proposal.state not in (CandidateState.evaluating, CandidateState.validated):
            raise PromotionError(f"candidate state {proposal.state.value} cannot be promoted")

        view = self._view_report(report)
        self._validate_report(proposal, view)
        gate = self._load_frozen_gate(view)

        active = self._active_bundle()
        prior_hash = active.content_hash if active else ""
        if view.base_hash != prior_hash:
            proposal.state = CandidateState.superseded
            self._sync_candidate(proposal)
            raise PromotionError("active base changed during evaluation; candidate superseded")

        decision, reason = self._check_gate(view, gate)

        report_payload = (
            report.model_dump(mode="json", by_alias=True)
            if isinstance(report, EvaluationReport)
            else dict(report)
        )
        promotion = PromotionDecision(
            candidate_hash=proposal.candidate_bundle_hash or "",
            base_hash=proposal.base_bundle_hash,
            report_ref=self.store.put_artifact(report_payload),
            gate_version=gate.protocol_hash,
            decision=decision,  # type: ignore[arg-type]
            reason=reason,
            prior_active_hash=prior_hash,
            new_active_hash=None,
        )
        promo_data = {
            "decision_id": promotion.decision_id,
            "candidate_hash": promotion.candidate_hash,
            "base_hash": promotion.base_hash,
            "prior_active_hash": promotion.prior_active_hash,
            "new_active_hash": promotion.new_active_hash,
            "decision": promotion.decision,
            "reason": promotion.reason,
            "timestamp": promotion.timestamp.isoformat(),
        }

        if decision == "promoted":
            target_hash = proposal.candidate_bundle_hash
            if self._bundle_by_hash(target_hash) is None:
                raise PromotionError("candidate bundle not found")
            proposal.state = CandidateState.promoted
            promo_data["new_active_hash"] = target_hash
            promotion.new_active_hash = target_hash
            ok = self.store.cas_active_bundle(
                expected_active_hash=prior_hash,
                new_active_hash=target_hash,
                candidate_id=candidate_id,
                new_candidate_state=proposal.state.value,
                candidate_json=proposal.model_dump_json(by_alias=True),
                promotion=promo_data,
            )
            if not ok:
                proposal.state = CandidateState.superseded
                self._sync_candidate(proposal)
                raise PromotionError("active pointer changed mid-promotion; candidate superseded")
        else:
            proposal.state = (
                CandidateState.quarantined if decision == "quarantined" else CandidateState.rejected
            )
            self.store.save_promotion(promotion.decision_id, promo_data)
            self._sync_candidate(proposal)

        self.store.save_evaluation(
            view.report_id,
            {
                "candidate_hash": view.candidate_hash,
                "base_hash": view.base_hash,
                "protocol_hash": view.protocol_hash,
                "partition_ref": view.partition_ref.model_dump_json(by_alias=True),
                "report_json": json.dumps(report_payload, sort_keys=True, default=str),
                "validity": "valid",
            },
        )
        return promotion

    def _check_gate(self, view: _ReportView, gate: PromotionGate) -> tuple[str, str]:
        if view.safety_violations > 0 or any(v is False for v in view.safety_results.values()):
            return "rejected", "safety check failed"
        if view.suspicious:
            return "quarantined", "suspicious code flagged"

        baseline_accuracy = view.uncertainty.get("baseline_accuracy", 0.0)
        gain = view.accuracy - baseline_accuracy
        if gain < gate.min_balanced_accuracy_gain:
            return "rejected", f"accuracy gain {gain:.3f} below threshold {gate.min_balanced_accuracy_gain}"
        ci = view.uncertainty.get("ci_lower", 0.0)
        if ci <= gate.ci_lower_bound:
            return "rejected", f"CI lower bound {ci:.3f} not above {gate.ci_lower_bound}"

        if gate.require_per_environment_non_regression:
            for env, vals in view.uncertainty.get("per_environment", {}).items():
                if vals.get("candidate_accuracy", 0) < vals.get("baseline_accuracy", 0):
                    return "rejected", f"observed regression in environment {env}"
                if vals.get("candidate_reliability", 0) < vals.get("baseline_reliability", 0):
                    return "rejected", f"observed reliability regression in environment {env}"

        base_cost = view.uncertainty.get("baseline_cost", 0.0) or 1e-9
        base_latency = view.uncertainty.get("baseline_latency", 0.0) or 1e-9
        if view.mean_cost / base_cost > gate.max_cost_ratio:
            return "rejected", f"cost ratio exceeds {gate.max_cost_ratio}"
        if view.p95_latency / base_latency > gate.max_latency_ratio:
            return "rejected", f"latency ratio exceeds {gate.max_latency_ratio}"

        return "promoted", "passed frozen gate"

    # ------------------------------------------------------------------ rollback
    def rollback(self, target_hash: str, reason: str) -> PromotionDecision:
        """Restore the active pointer to a hash present in the approved lineage."""
        if not self.store.is_in_active_lineage(target_hash):
            raise PromotionError(
                f"rollback target {target_hash!r} is not in the approved active lineage"
            )
        target = self._bundle_by_hash(target_hash)
        if target is None:
            raise PromotionError("rollback target bundle not found")
        active = self._active_bundle()
        prior_hash = active.content_hash if active else ""
        if prior_hash == target_hash:
            raise PromotionError("rollback target is already active")

        decision = PromotionDecision(
            candidate_hash=target_hash,
            base_hash=target_hash,
            report_ref=ArtifactRef(id="rollback", version="1", sha256="0" * 64),
            gate_version="rollback",
            decision="promoted",
            reason=f"rollback: {reason}",
            prior_active_hash=prior_hash,
            new_active_hash=target_hash,
        )
        promo_data = {
            "decision_id": decision.decision_id,
            "candidate_hash": decision.candidate_hash,
            "base_hash": decision.base_hash,
            "prior_active_hash": decision.prior_active_hash,
            "new_active_hash": decision.new_active_hash,
            "decision": decision.decision,
            "reason": decision.reason,
            "timestamp": decision.timestamp.isoformat(),
        }
        ok = self.store.cas_active_bundle(
            expected_active_hash=prior_hash,
            new_active_hash=target_hash,
            candidate_id=f"rollback:{target_hash}",
            new_candidate_state="rolled_back",
            candidate_json=promo_data["reason"],
            promotion=promo_data,
        )
        if not ok:
            raise PromotionError("active pointer changed during rollback; retry reconcile")
        return decision

    def get_active_bundle(self) -> SkillBundle | None:
        return self._active_bundle()
