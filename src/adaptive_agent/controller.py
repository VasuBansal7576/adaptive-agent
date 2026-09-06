"""Canonical durable seam for the control API (session2) and live execution.

This is the stable surface POST /runs and SSE stream from, and the single entry
point for candidate/evaluator/promotion/rollback operations. It owns run, step,
evidence, and outcome persistence and delegates all tool dispatch to the
ToolBroker. The learner/driver is injected — authority over tools, approvals,
evaluator identity, and the active bundle never enters the learner.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Protocol

from adaptive_agent.broker import Capability, ToolBroker, ToolProvider
from adaptive_agent.candidate import CandidateManager
from adaptive_agent.environment import EnvironmentRegistry
from adaptive_agent.models import (
    ArtifactRef,
    CandidateProposal,
    EvaluationReport,
    EvidenceRecord,
    Outcome,
    PromotionDecision,
    PromotionGate,
    RunRecord,
    RunRequest,
    RunStatus,
    SkillBundle,
    StepKind,
    StepRecord,
    StepStatus,
    TaskInput,
    ToolRequest,
    ToolResult,
    new_id,
    sha256_json,
)
from adaptive_agent.store import Store


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunDriver(Protocol):
    """Learner-side driver, injected by the session that owns the agent loop.

    Receives a DriverContext (broker-gated tool access only) and must not hold
    any reference to the Store, Broker, approvals, or evaluator.
    """

    def act(self, ctx: "DriverContext") -> None: ...


ApprovalProvider = Callable[[ToolRequest], str | None]


class DriverContext:
    """The only surface a driver can see: run identity and brokered tool calls."""

    def __init__(self, controller: "Controller", run_id: str, env_id: str, provider: ToolProvider) -> None:
        self._controller = controller
        self.run_id = run_id
        self.env_id = env_id
        self._provider = provider

    def call_tool(
        self,
        step_id: str,
        tool: str,
        arguments: dict[str, Any],
        idempotency_key: str,
        capability: Capability,
        approval_token: str | None = None,
    ) -> ToolResult:
        req = ToolRequest(
            runId=self.run_id,
            stepId=step_id,
            tool=tool,
            arguments=arguments,
            idempotencyKey=idempotency_key,
            approvalToken=approval_token,
        )
        result = self._controller.dispatch_tool(self.env_id, req, capability, self._provider)
        self._controller.append_event(
            self.run_id,
            "tool_result",
            result.model_dump(mode="json", by_alias=True),
            trust_class="broker",
            visibility="learner",
        )
        return result


class Controller:
    """Durable seam: runs, steps, evidence, outcomes, candidates, promotions."""

    def __init__(
        self,
        store: Store,
        registry: EnvironmentRegistry,
        broker: ToolBroker,
        approval_provider: ApprovalProvider | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.broker = broker
        self.candidates = CandidateManager(store)
        self.approval_provider = approval_provider

    # ------------------------------------------------------------------ runs
    def create_run(self, request: RunRequest, task: TaskInput, skill_bundle: SkillBundle | None = None) -> RunRecord:
        """Idempotent run creation: same idempotency key returns the stored run."""
        existing = self.store.get_run_by_idempotency_key(request.idempotency_key)
        if existing:
            return RunRecord.model_validate_json(existing["run_json"])

        if skill_bundle is None:
            skill_bundle = self.candidates.get_active_bundle() or SkillBundle()
        bundle_ref = self.store.put_artifact(skill_bundle.model_dump(mode="json", by_alias=True))
        manifest = self.registry.get_manifest(task.environment_ref.id)
        if manifest is None:
            raise KeyError(f"environment {task.environment_ref.id!r} not registered")

        run = RunRecord(
            taskRef=request.task_ref,
            environmentRef=task.environment_ref,
            policyRef=manifest.policy_ref,
            modelProfileRef=request.model_profile_ref,
            skillBundleRef=bundle_ref,
            budgetRef=request.budget_ref,
            executionMode=request.execution_mode,
            activeSkillRefs=request.active_skill_refs,
            parentRunId=request.parent_run_id,
            status=RunStatus.queued,
        )
        self.store.save_run(
            run.run_id,
            {
                "parent_run_id": run.parent_run_id,
                "task_id": task.task_id,
                "environment_id": manifest.environment_id,
                "bundle_id": skill_bundle.bundle_id,
                "status": run.status.value,
                "idempotency_key": request.idempotency_key,
                "last_event_sequence": 0,
                "created_at": run.created_at.isoformat(),
                "run_json": run.model_dump_json(by_alias=True),
            },
        )
        self.append_event(run.run_id, "run_created", {"run_id": run.run_id}, "system", "learner")
        return run

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self.store.get_run(run_id)
        if not row:
            return None
        run = RunRecord.model_validate_json(row["run_json"])
        run.status = RunStatus(row["status"])
        run.last_event_sequence = row["last_event_sequence"]
        if row["completed_at"]:
            run.completed_at = datetime.fromisoformat(row["completed_at"])
        return run

    def _set_run_status(self, run_id: str, status: RunStatus) -> None:
        row = self.store.get_run(run_id)
        run = RunRecord.model_validate_json(row["run_json"])
        run.status = status
        if status in (RunStatus.succeeded, RunStatus.failed, RunStatus.cancelled, RunStatus.timed_out):
            run.completed_at = datetime.now(timezone.utc)
        row.pop("run_id", None)  # save_run prepends the key column
        row["status"] = status.value
        row["completed_at"] = run.completed_at.isoformat() if run.completed_at else None
        row["run_json"] = run.model_dump_json(by_alias=True)
        self.store.save_run(run_id, row)

    def cancel_run(self, run_id: str) -> RunRecord | None:
        run = self.get_run(run_id)
        if run is None:
            return None
        if run.status in (RunStatus.succeeded, RunStatus.failed, RunStatus.cancelled):
            return run
        self._set_run_status(run_id, RunStatus.cancelled)
        self.append_event(run_id, "run_cancelled", {}, "operator", "operator")
        return self.get_run(run_id)

    def claim_run(self, run_id: str) -> tuple[bool, RunRecord | None]:
        """Claim queued execution exactly once before side-effect setup."""
        claimed = self.store.claim_run(run_id)
        if claimed is None:
            return False, None
        run = self.get_run(run_id)
        if claimed:
            self.append_event(run_id, "run_started", {"runId": run_id}, "system", "operator")
        return claimed, run

    # ------------------------------------------------------------------ steps
    def begin_step(self, run_id: str, kind: StepKind | str, input_refs: list[ArtifactRef] | None = None) -> StepRecord:
        kind = StepKind(kind)
        seq = self.store.next_event_sequence(run_id)
        step = StepRecord(runId=run_id, sequence=seq, kind=kind, status=StepStatus.running, inputRefs=input_refs or [])
        self.store.save_step(
            step.step_id,
            {
                "run_id": run_id,
                "sequence": seq,
                "kind": kind.value,
                "status": step.status.value,
                "step_json": step.model_dump_json(by_alias=True),
            },
        )
        self.append_event(run_id, "step_started", {"step_id": step.step_id, "kind": kind.value}, "system", "learner")
        return step

    def finish_step(self, step: StepRecord, status: StepStatus | str, error: Any | None = None) -> StepRecord:
        status = StepStatus(status)
        step.status = status
        step.error = error
        self.store.update_step_status(step.step_id, status.value, step.model_dump_json(by_alias=True))
        self.append_event(step.run_id, "step_finished", {"step_id": step.step_id, "status": status.value}, "system", "learner")
        return step

    # ------------------------------------------------------------------ tool dispatch (broker-only)
    def dispatch_tool(self, env_id: str, request: ToolRequest, capability: Capability, provider: ToolProvider, budget_remaining: dict[str, Any] | None = None, dry_run: bool = False) -> ToolResult:
        if request.approval_token is None and self.approval_provider is not None:
            schema = self.registry.get_tool_schema(env_id, request.tool)
            if schema is not None and schema.effect == "write":
                request.approval_token = self.approval_provider(request)
        return self.broker.request_tool_call(env_id, request, capability, provider, budget_remaining=budget_remaining, dry_run=dry_run)

    # ------------------------------------------------------------------ evidence / SSE
    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any], trust_class: str, visibility: str) -> EvidenceRecord:
        seq = self.store.next_event_sequence(run_id)
        ev = EvidenceRecord(
            runId=run_id,
            sequence=seq,
            eventType=event_type,
            contentHash=sha256_json(payload),
            sourceRef=self.store.put_artifact(payload),
            trustClass=trust_class,  # type: ignore[arg-type]
            visibility=visibility,  # type: ignore[arg-type]
            redacted=visibility == "learner",
        )
        self.store.append_evidence(
            ev.evidence_id,
            {
                "run_id": run_id,
                "sequence": seq,
                "event_type": event_type,
                "content_hash": ev.content_hash,
                "source_ref": ev.source_ref.model_dump_json(by_alias=True),
                "trust_class": trust_class,
                "visibility": visibility,
                "redacted": 1 if ev.redacted else 0,
            },
        )
        return ev

    def events(self, run_id: str, after_sequence: int = 0) -> list[dict[str, Any]]:
        """Ordered SSE-ready event payloads: [{id, event, data}]."""
        rows = [r for r in self.store.list_evidence(run_id) if r["sequence"] > after_sequence]
        return [
            {
                "id": r["sequence"],
                "event": r["event_type"],
                "data": r,
            }
            for r in rows
        ]

    def stream_events(self, run_id: str, after_sequence: int = 0, poll_interval: float = 0.5, timeout: float = 300.0) -> Iterator[dict[str, Any]]:
        """Blocking generator for SSE until the run reaches a terminal status."""
        seq = after_sequence
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for ev in self.events(run_id, after_sequence=seq):
                seq = max(seq, int(ev["id"]))
                yield ev
            run = self.get_run(run_id)
            if run is None:
                return
            if run.status in (RunStatus.succeeded, RunStatus.failed, RunStatus.cancelled, RunStatus.timed_out):
                # Drain any events appended at completion.
                for ev in self.events(run_id, after_sequence=seq):
                    yield ev
                return
            time.sleep(poll_interval)

    # ------------------------------------------------------------------ outcomes / reconciliation
    def record_outcome(self, run_id: str, passed: bool, score: float | None = None, metadata: dict[str, Any] | None = None) -> Outcome:
        """Record a trusted evaluator outcome. Only callers holding evaluator
        authority may invoke this; the learner never sees it."""
        outcome = Outcome(runId=run_id, passed=passed, score=score, metadata=metadata or {})
        self.store.save_outcome(
            outcome.outcome_id,
            {
                "run_id": run_id,
                "passed": 1 if passed else 0,
                "score": score,
                "metadata_json": __import__("json").dumps(outcome.metadata),
                "checked_at": outcome.checked_at.isoformat(),
            },
        )
        self.append_event(run_id, "outcome_recorded", {"passed": passed, "score": score}, "evaluator", "operator")
        return outcome

    def reconcile_run(self, run_id: str, provider: ToolProvider) -> list[str]:
        return self.broker.reconcile_run(run_id, provider)

    # ------------------------------------------------------------------ execution loop
    def execute_run(
        self,
        run_id: str,
        env_id: str,
        provider: ToolProvider,
        driver: RunDriver,
        evaluate: Callable[[], Outcome] | None = None,
        claimed: bool = False,
    ) -> RunRecord:
        """Run lifecycle wrapper: running -> driver -> evaluator-derived terminal state.

        The driver sees only a DriverContext; provider errors surface as
        OUTCOME_UNKNOWN evidence, never a crash of the control plane.
        """
        if not claimed:
            acquired, current = self.claim_run(run_id)
            if not acquired:
                return current
        current = self.get_run(run_id)
        if current is None or current.status != RunStatus.running:
            return current
        ctx = DriverContext(self, run_id, env_id, provider)
        try:
            driver.act(ctx)
            current = self.get_run(run_id)
            if current is None or current.status == RunStatus.cancelled:
                return current
            if evaluate is None:
                # Keep the low-level controller compatible with synthetic unit
                # drivers.  Production runtimes pass an evaluator and therefore
                # never infer success from driver completion.
                self._set_run_status(run_id, RunStatus.succeeded)
            else:
                outcome = evaluate()
                self.record_outcome(run_id, outcome.passed, score=outcome.score, metadata=outcome.metadata)
                self._set_run_status(run_id, RunStatus.succeeded if outcome.passed else RunStatus.failed)
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            self.append_event(run_id, "run_failed", {"error": str(exc)}, "system", "operator")
            if (current := self.get_run(run_id)) is not None and current.status != RunStatus.cancelled:
                self._set_run_status(run_id, RunStatus.failed)
        return self.get_run(run_id)

    # ------------------------------------------------------------------ candidate/promotion seam
    def submit_candidate(self, proposal: CandidateProposal, bundle: SkillBundle) -> CandidateProposal:
        return self.candidates.submit_candidate(proposal, bundle)

    def start_evaluation(self, candidate_id: str) -> CandidateProposal:
        return self.candidates.start_evaluation(candidate_id)

    def freeze_protocol(self, gate: PromotionGate, evaluator_id: str) -> str:
        return self.candidates.freeze_protocol(gate, evaluator_id)

    def submit_evaluation_report(self, candidate_id: str, report: EvaluationReport) -> PromotionDecision:
        """Evaluator-owned entry point: applies the frozen gate and CAS."""
        return self.candidates.promote(candidate_id, report)

    def quarantine_candidate(self, candidate_id: str, reason: str) -> CandidateProposal:
        return self.candidates.quarantine(candidate_id, reason)

    def rollback(self, target_hash: str, reason: str) -> PromotionDecision:
        return self.candidates.rollback(target_hash, reason)

    def get_active_bundle(self) -> SkillBundle | None:
        return self.candidates.get_active_bundle()

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return self.store.get_candidate(candidate_id)

    def list_promotions(self) -> list[dict[str, Any]]:
        return self.store.list_promotions()
