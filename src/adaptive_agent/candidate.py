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

    def __init__(self, store: Store) -> None:
        self.store = store

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
    def freeze_protocol(self, gate: PromotionGate, evaluator_id: str) -> str:
        """Register a frozen promotion gate bound to a trusted evaluator identity.

        Returns the protocol hash. This is the ONLY way a gate can later be used —
        callers of promote() cannot supply a gate object.
        """
        if not gate.protocol_hash:
            gate.protocol_hash = sha256_json(
                gate.model_dump(mode="json", by_alias=True, exclude={"protocol_hash"})
            )
        self.store.save_frozen_protocol(
            gate.protocol_hash,
            gate.model_dump_json(by_alias=True),
            evaluator_id,
        )
        return gate.protocol_hash

    def _load_frozen_gate(self, report: EvaluationReport) -> PromotionGate:
        row = self.store.get_frozen_protocol(report.protocol_hash)
        if row is None:
            raise PromotionError(
                f"no frozen protocol registered for protocol_hash {report.protocol_hash!r}"
            )
        gate = PromotionGate.model_validate_json(row["gate_json"])
        if report.evaluator_provenance != row["evaluator_id"]:
            raise PromotionError(
                "report provenance does not match the registered evaluator identity"
            )
        return gate

    # ------------------------------------------------------------------ report checks
    def _validate_report(self, proposal: CandidateProposal, report: EvaluationReport) -> None:
        if report.validity != EvaluationState.valid:
            raise PromotionError("evaluation report is not valid")
        if not proposal.candidate_bundle_hash:
            raise PromotionError("candidate bundle hash not recorded")
        if report.candidate_hash != proposal.candidate_bundle_hash:
            raise PromotionError("report candidate hash does not match submitted candidate")
        if report.base_hash != proposal.base_bundle_hash:
            raise PromotionError("report base hash does not match candidate base")
        # Required report cells.
        m = report.metrics
        if None in (m.accuracy, m.reliability, m.mean_cost, m.p95_latency_ms):
            raise PromotionError("report missing required metric cells")
        if not report.safety_results:
            raise PromotionError("report missing required safety cells")
        if not report.paired_run_ids:
            raise PromotionError("report has no paired runs")
        if not report.evaluator_provenance:
            raise PromotionError("report missing evaluator provenance")
        if not report.partition_ref.sha256:
            raise PromotionError("report missing partition reference")

    # ------------------------------------------------------------------ promotion
    def promote(self, candidate_id: str, report: EvaluationReport) -> PromotionDecision:
        """Apply the frozen gate and atomically CAS the active pointer.

        The gate is loaded from the frozen protocol keyed by report.protocol_hash;
        callers may not supply a gate. The compare-and-swap covers the active
        pointer, candidate state, and promotion decision in one transaction.
        """
        proposal = self._get_candidate(candidate_id)
        if proposal.state not in (CandidateState.evaluating, CandidateState.validated):
            raise PromotionError(f"candidate state {proposal.state.value} cannot be promoted")

        self._validate_report(proposal, report)
        gate = self._load_frozen_gate(report)

        active = self._active_bundle()
        prior_hash = active.content_hash if active else ""
        if report.base_hash != prior_hash:
            proposal.state = CandidateState.superseded
            self._sync_candidate(proposal)
            raise PromotionError("active base changed during evaluation; candidate superseded")

        decision, reason = self._check_gate(report, gate)

        promotion = PromotionDecision(
            candidate_hash=proposal.candidate_bundle_hash or "",
            base_hash=proposal.base_bundle_hash,
            report_ref=self.store.put_artifact(report.model_dump(mode="json", by_alias=True)),
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
            # Atomic single-transaction write of the decision + state (no pointer swap).
            self.store.save_promotion(promotion.decision_id, promo_data)
            self._sync_candidate(proposal)

        self.store.save_evaluation(
            report.report_id,
            {
                "candidate_hash": report.candidate_hash,
                "base_hash": report.base_hash,
                "protocol_hash": report.protocol_hash,
                "partition_ref": report.partition_ref.model_dump_json(by_alias=True),
                "report_json": report.model_dump_json(by_alias=True),
                "validity": report.validity.value,
            },
        )
        return promotion

    def _check_gate(self, report: EvaluationReport, gate: PromotionGate) -> tuple[str, str]:
        m = report.metrics
        if m.safety_violations > 0 or any(v is False for v in report.safety_results.values()):
            return "rejected", "safety check failed"
        if report.safety_results.get("suspicious"):
            return "quarantined", "suspicious code flagged"

        baseline_accuracy = report.uncertainty.get("baseline_accuracy", 0.0)
        gain = (m.accuracy or 0.0) - baseline_accuracy
        if gain < gate.min_balanced_accuracy_gain:
            return "rejected", f"accuracy gain {gain:.3f} below threshold {gate.min_balanced_accuracy_gain}"
        ci = report.uncertainty.get("ci_lower", 0.0)
        if ci <= gate.ci_lower_bound:
            return "rejected", f"CI lower bound {ci:.3f} not above {gate.ci_lower_bound}"

        if gate.require_per_environment_non_regression:
            for env, vals in report.uncertainty.get("per_environment", {}).items():
                if vals.get("candidate_accuracy", 0) < vals.get("baseline_accuracy", 0):
                    return "rejected", f"observed regression in environment {env}"
                if vals.get("candidate_reliability", 0) < vals.get("baseline_reliability", 0):
                    return "rejected", f"observed reliability regression in environment {env}"

        base_cost = report.uncertainty.get("baseline_cost", 0.0) or 1e-9
        base_latency = report.uncertainty.get("baseline_latency", 0.0) or 1e-9
        if (m.mean_cost or 0.0) / base_cost > gate.max_cost_ratio:
            return "rejected", f"cost ratio exceeds {gate.max_cost_ratio}"
        if (m.p95_latency_ms or 0.0) / base_latency > gate.max_latency_ratio:
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
