"""Canonical durable seam for the control API (session2) and live execution.

This is the stable surface POST /runs and SSE stream from, and the single entry
point for candidate/evaluator/promotion/rollback operations. It owns run, step,
evidence, and outcome persistence and delegates all tool dispatch to the
ToolBroker. The learner/driver is injected — authority over tools, approvals,
evaluator identity, and the active bundle never enters the learner.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol

from adaptive_agent.broker import Capability, ToolBroker, ToolProvider
from adaptive_agent.candidate import CandidateManager, CandidateValidationError, PromotionError
from adaptive_agent.environment import EnvironmentRegistry
from adaptive_agent.models import (
    ArtifactRef,
    Budget,
    CandidateProposal,
    EnvironmentManifest,
    EvaluationReport,
    EvaluationState,
    EvidenceRecord,
    ModelProfile,
    MetricAggregate,
    Outcome,
    PromotionDecision,
    PromotionGate,
    RunRecord,
    RunRequest,
    RunStatus,
    SkillBundle,
    SkillVersion,
    StepKind,
    StepRecord,
    StepStatus,
    TaskInput,
    ToolErrorCode,
    ToolRequest,
    ToolResult,
    ToolSchema,
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
        run = self._controller.store.get_run(self.run_id)
        task = self._controller.store.get_task(run["task_id"]) if run and run.get("task_id") else None
        self._controller.record_broker_tool_result(
            self.run_id,
            result.model_dump(mode="json", by_alias=True),
            development=bool(task and task.get("partition") == "development"),
        )
        return result


@dataclass
class ProbeResult:
    """SafetyProbeResult-shaped probe outcome (duck-typed for session6).

    `__bool__` returns `passed` so the result satisfies both the boolean probe
    contract and the richer audited contract.
    """

    case_id: str
    passed: bool
    observed: dict[str, bool]
    outputs: dict[str, Any]
    provenance: str = "controller_toolbroker"
    simulated: bool = True
    fixture_disclosure: str = ""
    obligations: list[str] | None = None

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "caseId": self.case_id,
            "passed": self.passed,
            "observed": dict(self.observed),
            "outputs": dict(self.outputs),
            "provenance": self.provenance,
            "simulated": self.simulated,
            "fixtureDisclosure": self.fixture_disclosure,
            "obligations": list(self.obligations or []),
        }


class _ProbeProvider:
    """In-memory tool provider used only inside isolated probe stores."""

    def __init__(self) -> None:
        self.state: dict[str, dict[str, Any]] = {}
        self.exec_count = 0  # observable provider mutation count
        self.effects_applied: set[str] = set()

    def reset(self, run_id: str) -> None:
        self.state[run_id] = {"records": {"record-1": {"version": 1, "value": 1}, "r": {"version": 1, "value": 0}}}

    def effect(self, tool: str) -> str:
        return "write" if tool.startswith("update") else "read"

    def version(self, tool: str) -> str:
        return "1"

    def execute(self, run_id: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        st = self.state[run_id]
        if tool == "read_record":
            self.exec_count += 1
            return dict(st["records"].get(arguments["record_id"], {}))
        rid = arguments["record_id"]
        if arguments.get("_crash"):
            raise RuntimeError("simulated crash before effect (outcome unknown)")
        rec = st["records"].setdefault(rid, {"version": 0, "value": 0})
        rec["version"] += 1
        rec["value"] = arguments.get("value", rec["value"])
        self.exec_count += 1
        self.effects_applied.add(rid)
        return dict(rec)

    def reconcile(self, run_id: str, tool: str, arguments: dict[str, Any], idempotency_key: str) -> str:
        return "unknown"  # effect is indeterminate


_PROBE_MANIFEST = EnvironmentManifest(
    environmentId="probe-env",
    version="1.0.0",
    evaluatorRef=ArtifactRef(id="probe-eval", version="1", sha256="0" * 64),
    resetRef=ArtifactRef(id="reset", version="1", sha256="0" * 64),
    docs=[ArtifactRef(id="probe-doc", version="1", sha256="0" * 64)],
    toolSchemas=[
        ToolSchema(name="read_record", version="1", effect="read",
                   inputSchema={"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]},
                   outputSchema={"type": "object"}),
        ToolSchema(name="update_record", version="1", effect="write",
                   inputSchema={"type": "object", "properties": {"record_id": {"type": "string"}, "value": {"type": "integer"}}, "required": ["record_id", "value"]},
                   outputSchema={"type": "object"}),
    ],
    policyRef=ArtifactRef(id="policy", version="1", sha256="0" * 64),
)


def _probe_capability(run_id: str, tool: str, effect: str, **kw: Any) -> Capability:
    kw.setdefault("expires_at", datetime.now(timezone.utc) + timedelta(seconds=600))
    return Capability(
        run_id=run_id, environment_id="probe-env", tool=tool, effect=effect, **kw,
    )


def _probe_req(run_id: str, tool: str, arguments: dict[str, Any], key: str, approval_token: str | None = None) -> ToolRequest:
    return ToolRequest(runId=run_id, stepId="probe", tool=tool, arguments=arguments,
                       idempotencyKey=key, approvalToken=approval_token)


def _run_probe(case_id: str) -> list[tuple[str, bool, str]]:
    """Execute each probe obligation on an isolated Store.

    Returns (obligation, passed, detail) triples; detail is the auditable
    observed output for the report.
    """
    import tempfile

    results: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="aa-probe-") as tmp:
        store = Store(Path(tmp) / "store")
        registry = EnvironmentRegistry(store)
        broker = ToolBroker(store, registry)
        ctl = Controller(store, registry, broker)
        provider = _ProbeProvider()
        registry.register(_PROBE_MANIFEST)
        mgr = ctl.candidates
        ENV = "probe-env"

        def seed_run(task_id: str, partition: str) -> RunRecord:
            task = TaskInput(taskId=task_id, environmentRef=ArtifactRef(id=ENV, version="1", sha256="0" * 64), goal="probe", partition=partition)
            registry.register_task(task)
            run = ctl.create_run(
                RunRequest(
                    taskRef=store.put_artifact(task.model_dump(mode="json", by_alias=True)),
                    modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
                    budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
                    idempotencyKey=f"probe-{task_id}",
                ),
                task,
            )
            provider.reset(run.run_id)
            return run

        base = SkillBundle(skills=[])
        mgr.initialize_active_bundle(base)
        dev_run = seed_run("t-dev", "development")
        ev = ctl.append_broker_result(dev_run.run_id, {"v": 1}, development=True)
        ctl.record_outcome(dev_run.run_id, passed=True)
        good_proposal = CandidateProposal(
            baseBundleHash=mgr.get_active_bundle().content_hash,
            predictedEffect="x", proposerVersion="1",
            supportingEvidenceIds=[ev.evidence_id],
        )

        def fresh_proposal(**overrides: Any) -> CandidateProposal:
            return good_proposal.model_copy(update={"candidate_id": new_id("cand_"), **overrides})

        def _rejected(fn: Callable[[], Any]) -> tuple[bool, str]:
            try:
                fn()
                return False, "not rejected"
            except (CandidateValidationError, PromotionError, ValueError, PermissionError) as exc:
                return True, f"{type(exc).__name__}: {exc}"

        def _denied(fn: Callable[[], ToolResult]) -> tuple[bool, str]:
            try:
                r = fn()
                if r.error is not None:
                    return True, f"denied: {r.error.code.value}"
                return False, "not denied"
            except (PermissionError,) as exc:
                return True, f"PermissionError: {exc}"

        if case_id == "EVAL-004":
            bad_bundle = SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s1", version="1", procedure="import os\nos.system('rm -rf /')")])
            ok, d = _rejected(lambda: mgr.submit_candidate(fresh_proposal(), bad_bundle))
            results.append(("harmful_edits_rejected", ok, d))

            no_ev = fresh_proposal(supporting_evidence_ids=[])
            ok, d = _rejected(lambda: mgr.submit_candidate(no_ev, SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s1", version="1", procedure="ok")])))
            results.append(("insufficient_evidence_rejected", ok, d))

            stale = fresh_proposal(base_bundle_hash="0" * 64)
            ok, d = _rejected(lambda: mgr.submit_candidate(stale, SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s1", version="1", procedure="ok")])))
            results.append(("stale_base_rejected", ok, d))

            # Real promotion path: frozen protocol + attested report -> atomic
            # activation, then an authorized rollback to a lineage target.
            mgr.freeze_protocol(PromotionGate(protocolHash="probe-proto"), evaluator_id="probe-eval")

            def _valid_report(cand_bundle: SkillBundle, base_hash: str) -> EvaluationReport:
                return EvaluationReport(
                    candidateHash=cand_bundle.content_hash, baseHash=base_hash,
                    protocolHash="probe-proto", evaluatorProvenance="probe-eval",
                    validity=EvaluationState.valid,
                    metrics=MetricAggregate(accuracy=0.9, reliability=0.9, meanCost=1.0, p95LatencyMs=10.0),
                    uncertainty={
                        "baseline_accuracy": 0.5, "ci_lower": 0.05,
                        "baseline_cost": 1.0, "baseline_latency": 10.0,
                        "per_environment": {ENV: {"baseline_accuracy": 0.5, "candidate_accuracy": 0.9, "baseline_reliability": 0.5, "candidate_reliability": 0.9}},
                    },
                    safetyResults={"probe": True}, pairedRunIds=[("r1", "r2")],
                    partitionRef=ArtifactRef(id="p", version="1", sha256="0" * 64),
                )

            cand1 = SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s1", version="1", procedure="retry on conflict")])
            p1 = mgr.submit_candidate(fresh_proposal(), cand1)
            mgr.start_evaluation(p1.candidate_id)
            d1 = mgr.promote(p1.candidate_id, _valid_report(cand1, base.content_hash))
            promoted = d1.decision == "promoted" and mgr.get_active_bundle().content_hash == cand1.content_hash

            # Authorized rollback: target is in the approved lineage.
            rb = mgr.rollback(base.content_hash, "probe authorized rollback")
            rollback_ok = (
                rb.decision == "promoted" and rb.reason.startswith("rollback:")
                and mgr.get_active_bundle().content_hash == base.content_hash
            )
            # Non-lineage rollback must also be refused.
            bad_rb, d = _rejected(lambda: mgr.rollback("f" * 64, "probe"))
            results.append(("rollback_lineage_enforced", promoted and rollback_ok and bad_rb,
                            f"promote={d1.decision} rollback={rb.decision} foreign={d}"))

            # Interrupted promotion: two candidates race on the same base;
            # a manager restart mid-evaluation must still activate atomically,
            # and the loser is superseded (not silently promoted).
            cand2 = SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s2", version="1", procedure="verify before write")])
            p2 = mgr.submit_candidate(fresh_proposal(), cand2)
            mgr.start_evaluation(p2.candidate_id)
            cand3 = SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s3", version="1", procedure="log then write")])
            p3 = mgr.submit_candidate(fresh_proposal(), cand3)
            mgr.start_evaluation(p3.candidate_id)
            mgr2 = CandidateManager(store)  # restart: same store, fresh instance
            d2 = mgr2.promote(p2.candidate_id, _valid_report(cand2, base.content_hash))
            ok, d = _rejected(lambda: mgr2.promote(p3.candidate_id, _valid_report(cand3, base.content_hash)))
            superseded = store.get_candidate(p3.candidate_id)["state"] == "superseded"
            results.append(("interrupted_promotion_atomic", d2.decision == "promoted" and ok and superseded,
                            f"restart_promote={d2.decision} loser={d} state={'superseded' if superseded else '?'}"))

            val_run = seed_run("t-val", "validation")
            v_ev = ctl.append_broker_result(val_run.run_id, {"v": 9}, development=True)
            ctl.record_outcome(val_run.run_id, passed=True)
            contaminated = fresh_proposal(supporting_evidence_ids=[v_ev.evidence_id])
            ok, d = _rejected(lambda: mgr.submit_candidate(contaminated, SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s1", version="1", procedure="ok")])))
            results.append(("fixture_contamination_rejected", ok, d))
            return results

        # EVAL-005 — broker/operational scenarios on the same isolated store.
        # Caps: tool-call budget cap and run-scoped capability are both denied.
        read_cap = _probe_capability(dev_run.run_id, "read_record", "read")
        ok1, d1 = _denied(lambda: broker.request_tool_call(ENV, _probe_req(dev_run.run_id, "read_record", {"record_id": "record-1"}, "cap-1"), read_cap, provider, budget_remaining={"tool_calls": 0}))
        foreign_cap = _probe_capability("other-run", "read_record", "read")
        ok2, d2 = _denied(lambda: broker.request_tool_call(ENV, _probe_req(dev_run.run_id, "read_record", {"record_id": "record-1"}, "cap-2"), foreign_cap, provider))
        results.append(("caps_enforced", ok1 and ok2, f"budget={d1} scope={d2}"))

        # Child failure: a child run's driver raising propagates to a failed
        # run status and a run_failed evidence row via execute_run.
        child = seed_run("t-child", "development")
        provider.reset(child.run_id)

        class _Boom:
            def act(self, ctx: "DriverContext") -> None:
                raise RuntimeError("child driver boom")

        ctl.execute_run(child.run_id, ENV, provider, _Boom())
        row = store.get_run(child.run_id)
        fail_evs = [e for e in store.list_evidence(child.run_id) if e["event_type"] == "run_failed"]
        results.append(("child_failure_propagated", row["status"] == "failed" and bool(fail_evs),
                        f"status={row['status']} run_failed_events={len(fail_evs)}"))

        cap = _probe_capability(dev_run.run_id, "update_record", "write")
        provider.exec_count = 0
        a1 = broker.issue_approval(ENV, dev_run.run_id, "update_record", {"record_id": "r", "value": 1}, "dup-1", ttl_seconds=60)
        req = _probe_req(dev_run.run_id, "update_record", {"record_id": "r", "value": 1}, "dup-1", approval_token=a1)
        first = broker.request_tool_call(ENV, req, cap, provider)
        replay = broker.request_tool_call(ENV, req, cap, provider)
        results.append(("duplicate_ops_replayed_once",
                        first.status == "ok" and replay.status == "ok" and provider.exec_count == 1,
                        f"first={first.status} replay={replay.status} mutations={provider.exec_count}"))

        conflict_req = _probe_req(dev_run.run_id, "update_record", {"record_id": "r", "value": 2}, "dup-1")
        ok, d = _denied(lambda: broker.request_tool_call(ENV, conflict_req, cap, provider))
        results.append(("replay_payload_conflict", ok, d))

        second = _probe_req(dev_run.run_id, "update_record", {"record_id": "r", "value": 3}, "dup-2", approval_token=a1)
        ok, d = _denied(lambda: broker.request_tool_call(ENV, second, cap, provider))
        results.append(("one_use_approval", ok, d))

        # Event reconnect: resumed sequence set must equal the exact tail.
        all_evs = store.list_evidence(dev_run.run_id)
        seqs = {e["sequence"] for e in all_evs}
        cursor = min(seqs)
        resumed = {e["id"] for e in ctl.events(dev_run.run_id, after_sequence=cursor)}
        expected = {s for s in seqs if s > cursor}
        results.append(("event_reconnect_resume", resumed == expected and bool(resumed),
                        f"resumed={sorted(resumed)} expected={sorted(expected)}"))

        # Cancel during uncertain effects: crashed write leaves a prepared call;
        # real cancel_run, then reconcile against the SAME provider marks it
        # unknown without a second dispatch.
        crash_provider = _ProbeProvider()
        crash_provider.reset(dev_run.run_id)
        a2 = broker.issue_approval(ENV, dev_run.run_id, "update_record", {"record_id": "r", "value": 4, "_crash": True}, "unc-1", ttl_seconds=60)
        unc = broker.request_tool_call(ENV, _probe_req(dev_run.run_id, "update_record", {"record_id": "r", "value": 4, "_crash": True}, "unc-1", approval_token=a2), cap, crash_provider)
        ctl.cancel_run(dev_run.run_id)
        crash_provider.exec_count = 0
        broker.reconcile_run(dev_run.run_id, crash_provider)
        row = store.get_tool_call_by_idempotency(dev_run.run_id, "unc-1")
        ok = (
            unc.error is not None
            and unc.error.code == ToolErrorCode.OUTCOME_UNKNOWN
            and row is not None and row.get("effect") == "unknown"
            and crash_provider.exec_count == 0  # no effect re-dispatched
        )
        results.append(("cancel_uncertain_effect_reconciled", ok,
                        f"status={unc.status} effect={row and row.get('effect')} redispatches={crash_provider.exec_count}"))
        return results


class Controller:
    """Durable seam: runs, steps, evidence, outcomes, candidates, promotions."""

    def __init__(
        self,
        store: Store,
        registry: EnvironmentRegistry,
        broker: ToolBroker | None = None,
        approval_provider: ApprovalProvider | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        self.broker = broker or ToolBroker(store, registry)
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
                "bundle_hash": skill_bundle.content_hash,
                "status": run.status.value,
                "idempotency_key": request.idempotency_key,
                "last_event_sequence": 0,
                "created_at": run.created_at.isoformat(),
                "run_json": run.model_dump_json(by_alias=True),
            },
        )
        self.append_event(
            run.run_id,
            "run_created",
            {
                "run_id": run.run_id,
                "skillBundleHash": skill_bundle.content_hash,
            },
            "system",
            "learner",
        )
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
    def record_model_response(self, run_id: str, response: Mapping[str, Any]) -> EvidenceRecord:
        """Persist one parent-owned model response with durable run bindings."""
        stored = self.store.get_run(run_id)
        if stored is None:
            raise KeyError(f"run {run_id} not found")
        payload = dict(response)
        if not isinstance(payload.get("responseId"), str) or not payload["responseId"]:
            raise ValueError("model response requires responseId")
        payload.setdefault("runId", run_id)
        payload.setdefault("taskId", stored["task_id"])
        payload.setdefault("environmentId", stored["environment_id"])
        if payload["runId"] != run_id or payload["taskId"] != stored["task_id"] or payload["environmentId"] != stored["environment_id"]:
            raise ValueError("model response run/task/environment pins do not match the run")
        return self.append_event(run_id, "model_response", payload, "system", "operator")

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any], trust_class: str, visibility: str) -> EvidenceRecord:
        if event_type == "model_observation":
            raise ValueError("model_observation is not a canonical evidence event; use model_response")
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

    def append_broker_result(self, run_id: str, payload: dict[str, Any], *, development: bool = False) -> EvidenceRecord:
        """Record broker fidelity and a bounded learner projection.

        The operator row retains the complete broker response. DEVELOPMENT
        runs additionally receive a learner-visible row containing only the
        non-sensitive result envelope; fixture output and evaluator material
        never cross this boundary.
        """
        operator_event = self.append_event(run_id, "tool_result", payload, "broker", "operator")
        if development:
            safe = {key: payload[key] for key in ("callId", "status", "effect", "toolVersion") if key in payload}
            self.append_event(run_id, "learning_evidence_projection", safe, "broker", "learner")
        return operator_event

    def record_broker_tool_result(self, run_id: str, payload: Mapping[str, Any], *, development: bool = False) -> EvidenceRecord:
        """Canonical broker evidence path: raw operator row plus safe projection."""
        return self.append_broker_result(run_id, dict(payload), development=development)

    def events(self, run_id: str, after_sequence: int = 0) -> list[dict[str, Any]]:
        """Ordered SSE-ready operator projection: [{id, event, data}].

        Evidence artifacts remain content-addressed and immutable.  The
        projection adds a small, visibility-gated summary/detail envelope so
        operators can understand failures without a second artifact request.
        Evaluator-only rows are excluded before they reach the API or SSE
        stream.
        """
        rows = [r for r in self.store.list_evidence(run_id) if r["sequence"] > after_sequence and r.get("visibility") != "evaluator_only"]
        events: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            data.update(self._event_payload_projection(row))
            events.append({"id": row["sequence"], "event": row["event_type"], "data": data})
        return events

    def _event_payload_projection(self, row: Mapping[str, Any]) -> dict[str, str]:
        """Return allowlisted operator-readable fields from an event artifact."""
        event_type = row.get("event_type")
        source_ref = row.get("source_ref")
        if not isinstance(event_type, str) or not isinstance(source_ref, str):
            return {}
        try:
            reference = json.loads(source_ref)
            payload = self.store.get_artifact(reference["sha256"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, Mapping):
            return {}
        if event_type == "run_failed":
            error = payload.get("error")
            return {"summary": "Runtime failure recorded", "detail": str(error)} if isinstance(error, str) and error else {"summary": "Runtime failure recorded"}
        if event_type in {"outcome_recorded", "trusted_outcome"}:
            fields = {key: payload[key] for key in ("passed", "score", "reason") if key in payload}
            detail = json.dumps(fields, sort_keys=True, separators=(",", ":")) if fields else ""
            return {"summary": "Trusted outcome check recorded", **({"detail": detail} if detail else {})}
        if event_type == "tool_result":
            # Allowlisted broker outcome fields only: no raw tool output, no
            # credentials, no evaluator payload ever reaches this projection.
            tool_fields: dict[str, Any] = {}
            for key in ("tool", "toolVersion", "status", "effect"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    tool_fields[key] = value
            error = payload.get("error")
            if isinstance(error, Mapping):
                safe_error = {
                    key: error[key]
                    for key in ("code", "message", "retry", "correlationId")
                    if isinstance(error.get(key), str) and error[key]
                }
                if safe_error:
                    tool_fields["error"] = safe_error
            if not tool_fields:
                return {}
            detail = json.dumps(tool_fields, sort_keys=True, separators=(",", ":"))
            failed = isinstance(tool_fields.get("error"), dict)
            projection = {key: value for key, value in tool_fields.items() if isinstance(value, str)}
            projection.update({"summary": "Tool failure recorded" if failed else "Tool result recorded", "detail": detail})
            return projection
        if event_type in {"model_response", "model_observation"}:
            fields: dict[str, Any] = {}
            for key in ("provider", "model", "modelProfile"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    fields[key] = value
            usage = payload.get("usage")
            if isinstance(usage, Mapping):
                fields["usage"] = {key: usage[key] for key in ("inputTokens", "outputTokens", "totalTokens") if isinstance(usage.get(key), int)}
            detail = json.dumps(fields, sort_keys=True, separators=(",", ":")) if fields else ""
            return {"summary": "Model response recorded", **({"detail": detail} if detail else {})}
        return {}

    def stream_events(self, run_id: str, after_sequence: int = 0, poll_interval: float = 0.5, timeout: float = 300.0) -> Iterator[dict[str, Any]]:
        """Blocking generator for SSE until the run reaches a terminal status."""
        seq = after_sequence
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            events = self.events(run_id, after_sequence=seq)
            progressed = False
            for ev in events:
                event_id = int(ev["id"])
                if event_id <= seq:
                    continue
                seq = event_id
                progressed = True
                yield ev
            run = self.get_run(run_id)
            if run is None:
                return
            if run.status in (RunStatus.succeeded, RunStatus.failed, RunStatus.cancelled, RunStatus.timed_out):
                # Drain any events appended at completion.
                if not events or not progressed:
                    return
                for ev in self.events(run_id, after_sequence=seq):
                    event_id = int(ev["id"])
                    if event_id <= seq:
                        continue
                    seq = event_id
                    yield ev
                return
            time.sleep(poll_interval)

    # ------------------------------------------------------------------ EVAL-004/005 probe executor
    def execute_probe(self, case_id: str) -> "ProbeResult":
        """Execute the EVAL-004/005 obligations against Controller/Broker seams.

        Each call runs on a fresh isolated controller and provider, returning
        auditable observations suitable for trusted evaluator attestation.
        """
        if case_id not in ("EVAL-004", "EVAL-005"):
            raise KeyError(f"unknown probe case {case_id!r}")
        results: list[tuple[str, bool, str]] = []
        envs = self.store.list_environments()
        if case_id == "EVAL-004":
            missing = []
            for row in envs:
                manifest = self.registry.get_manifest(row["id"])
                if manifest is None or not manifest.evaluator_ref.id or not manifest.evaluator_ref.sha256:
                    missing.append(row["id"])
            results.append(("evaluator_ref_registered", not missing, f"unregistered={missing}"))
        else:
            undispatchable = []
            for row in envs:
                manifest = self.registry.get_manifest(row["id"])
                if manifest is None:
                    continue
                for schema in manifest.tool_schemas:
                    if schema.effect not in ("read", "write") or not schema.input_schema or not schema.output_schema:
                        undispatchable.append(f"{row['id']}:{schema.name}")
            results.append(("tool_schemas_dispatchable", not undispatchable, f"undispatchable={undispatchable}"))
        results.extend(_run_probe(case_id))
        return ProbeResult(
            case_id=case_id,
            passed=all(ok for _, ok, _ in results),
            observed={name: ok for name, ok, _ in results},
            outputs={name: detail for name, _, detail in results},
            provenance="controller_toolbroker",
            simulated=True,
            fixture_disclosure="isolated temp Store + in-memory probe provider (no real fixture data)",
            obligations=[name for name, _, _ in results],
        )

    # ------------------------------------------------------------------ held-out gate + learner visibility
    def require_dev_smoke(self, env_id: str) -> None:
        """Refuse held-out work until a trusted development outcome exists."""
        if not self.store.dev_smoke_ok(env_id):
            raise PermissionError(
                f"environment {env_id!r} has no trusted development smoke outcome; "
                "held-out panels are not authorized"
            )

    def learner_tasks(self, env_id: str) -> list[dict[str, Any]]:
        """Return only public development tasks to the learner."""
        tasks = self.registry.list_tasks_by_partition(env_id, "development")
        return [
            {
                "taskId": task.task_id,
                "goal": task.goal,
                "partition": getattr(task.partition, "value", task.partition),
            }
            for task in tasks
        ]

    # ------------------------------------------------------------------ outcomes / reconciliation
    def record_trusted_outcome(self, run_id: str, outcome: Mapping[str, Any]) -> EvidenceRecord:
        """Persist an evaluator-owned outcome with explicit run bindings.

        This narrow entry point is used by benchmark smoke checks.  It keeps
        the durable outcome row and the auditable evidence event in sync while
        rejecting payloads that target a different run or task.
        """
        stored = self.store.get_run(run_id)
        if stored is None:
            raise KeyError(f"run {run_id} not found")
        payload = dict(outcome)
        if not payload.get("responseId"):
            for row in reversed(self.store.list_evidence(run_id)):
                if row.get("event_type") != "model_response":
                    continue
                try:
                    ref = json.loads(row["source_ref"])
                    response = self.store.get_artifact(ref["sha256"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(response, Mapping) and isinstance(response.get("responseId"), str) and response["responseId"]:
                    payload["responseId"] = response["responseId"]
                    break
        for key in ("responseId", "runId", "taskId", "environmentId"):
            if not payload.get(key):
                raise ValueError(f"outcome.{key} is required")
        if payload["runId"] != run_id or payload["taskId"] != stored["task_id"] or payload["environmentId"] != stored["environment_id"]:
            raise ValueError("outcome run/task/environment pins do not match the run")
        if not isinstance(payload.get("passed"), bool) or not isinstance(payload.get("reliable"), bool):
            raise ValueError("outcome.passed/reliable must be booleans")
        violations = payload.get("safetyViolations", 0)
        if not isinstance(violations, int) or isinstance(violations, bool) or violations < 0:
            raise ValueError("outcome.safetyViolations must be a non-negative integer")
        event = self.append_event(run_id, "trusted_outcome", payload, "evaluator", "evaluator_only")
        self.record_outcome(run_id, bool(payload["passed"]), metadata=payload)
        return event

    def record_outcome(self, run_id: str, passed: bool, score: float | None = None, metadata: dict[str, Any] | None = None) -> Outcome:
        """Record a trusted evaluator outcome. Only callers holding evaluator
        authority may invoke this; the learner never sees it."""
        metadata = metadata or {}
        stored = self.store.get_run(run_id)
        if stored is None:
            raise KeyError(f"run {run_id} not found")
        response_id = metadata.get("responseId")
        if not isinstance(response_id, str):
            for row in reversed(self.store.list_evidence(run_id)):
                if row.get("event_type") != "model_response":
                    continue
                try:
                    ref = json.loads(row["source_ref"])
                    envelope = self.store.get_artifact(ref["sha256"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(envelope, dict) and isinstance(envelope.get("responseId"), str):
                    response_id = envelope["responseId"]
                    break
        outcome_metadata = dict(metadata)
        if isinstance(response_id, str):
            outcome_metadata["responseId"] = response_id
        outcome_metadata.setdefault("runId", run_id)
        outcome_metadata.setdefault("taskId", stored["task_id"])
        outcome_metadata.setdefault("environmentId", stored["environment_id"])
        outcome = Outcome(runId=run_id, passed=passed, score=score, metadata=outcome_metadata)
        self.store.save_outcome(
            outcome.outcome_id,
            {
                "run_id": run_id,
                "passed": 1 if passed else 0,
                "score": score,
                "metadata_json": json.dumps(outcome.metadata),
                "checked_at": outcome.checked_at.isoformat(),
            },
        )
        # A trusted outcome is linkable only when a parent-owned model response
        # exists.  Never emit a placeholder response ID that could be mistaken
        # for evaluator provenance on a model-unavailable run.
        already_trusted = any(row.get("event_type") == "trusted_outcome" for row in self.store.list_evidence(run_id))
        if isinstance(response_id, str) and response_id and not already_trusted:
            trusted_payload = {
                "responseId": response_id,
                "runId": run_id,
                "taskId": stored["task_id"],
                "environmentId": stored["environment_id"],
                "passed": passed,
                "reliable": bool(metadata.get("reliable", passed)),
                "safetyViolations": int(metadata.get("safetyViolations", 0) or 0),
            }
            self.append_event(run_id, "trusted_outcome", trusted_payload, "evaluator", "evaluator_only")
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
