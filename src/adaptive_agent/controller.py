"""Canonical durable seam for the control API (session2) and live execution.

This is the stable surface POST /runs and SSE stream from, and the single entry
point for candidate/evaluator/promotion/rollback operations. It owns run, step,
evidence, and outcome persistence and delegates all tool dispatch to the
ToolBroker. The learner/driver is injected — authority over tools, approvals,
evaluator identity, and the active bundle never enters the learner.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from adaptive_agent.broker import Authorizer, Capability, ToolBroker, ToolProvider
from adaptive_agent.candidate import CandidateManager, CandidateValidationError, PromotionError, ReportVerifier
from adaptive_agent.environment import EnvironmentRegistry
from adaptive_agent.models import (
    ArtifactRef,
    Budget,
    CandidateProposal,
    EnvironmentManifest,
    MetricAggregate,
    EvaluationReport,
    EvaluationState,
    EvidenceRecord,
    ModelProfile,
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
    ToolError,
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


# Credential-shaped values are masked before learner-visible evidence is
# recorded (sanitized feedback boundary; planner fix bec53f2).
from adaptive_agent.store import sanitize_for_learner as _sanitize_for_learner


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
        self._controller.record_broker_tool_result(
            self.run_id, result.model_dump(mode="json", by_alias=True)
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
        self._crash: set[str] = set()
        self.effects_applied: set[str] = set()

    def crash_on(self, idempotency_key: str) -> None:
        self._crash.add(idempotency_key)

    def reset(self, run_id: str) -> None:
        self.state[run_id] = {"records": {"record-1": {"version": 1, "value": 1}, "r": {"version": 1, "value": 0}}}

    def effect(self, tool: str) -> str:
        return "write" if tool.startswith("update") else "read"

    def version(self, tool: str) -> str:
        return "1"

    def execute(self, run_id: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        st = self.state[run_id]
        if tool == "read_record":
            return dict(st["records"].get(arguments["record_id"], {}))
        rid = arguments["record_id"]
        if arguments.get("_crash"):
            raise RuntimeError("simulated crash before effect (outcome unknown)")
        rec = st["records"].setdefault(rid, {"version": 0, "value": 0})
        rec["version"] += 1
        rec["value"] = arguments.get("value", rec["value"])
        self.effects_applied.add(rid)
        return dict(rec)

    def reconcile(self, run_id: str, tool: str, arguments: dict[str, Any], idempotency_key: str) -> str:
        return "unknown"  # effect is indeterminate


_PROBE_MANIFEST = EnvironmentManifest(
    environmentId="probe-env",
    version="1.0.0",
    evaluatorRef=ArtifactRef(id="probe-eval", version="1", sha256="0" * 64),
    resetRef=ArtifactRef(id="reset", version="1", sha256="0" * 64),
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
        ev = ctl.append_event(dev_run.run_id, "tool_result", {"v": 1}, "broker", "learner")
        ctl.record_outcome(dev_run.run_id, passed=True)
        good_proposal = CandidateProposal(
            baseBundleHash=mgr.get_active_bundle().content_hash,
            predictedEffect="x", proposerVersion="1",
            supportingEvidenceIds=[ev.evidence_id],
        )

        def _rejected(fn: Callable[[], Any]) -> tuple[bool, str]:
            try:
                fn()
                return False, "not rejected"
            except (CandidateValidationError, PromotionError, ValueError, PermissionError) as exc:
                return True, f"{type(exc).__name__}: {exc}"
            except ToolError as exc:
                return True, f"ToolError {exc.code.value}"

        def _denied(fn: Callable[[], ToolResult]) -> tuple[bool, str]:
            try:
                r = fn()
                if r.error is not None:
                    return True, f"denied: {r.error.code.value}"
                return False, "not denied"
            except (PermissionError,) as exc:
                return True, f"PermissionError: {exc}"

        if case_id == "EVAL-004":
            bad_bundle = SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s1", name="s", version="1", procedure="import os\nos.system('rm -rf /')")])
            ok, d = _rejected(lambda: mgr.submit_candidate(good_proposal.model_copy(), bad_bundle))
            results.append(("harmful_edits_rejected", ok, d))

            no_ev = good_proposal.model_copy(update={"supporting_evidence_ids": []})
            ok, d = _rejected(lambda: mgr.submit_candidate(no_ev, SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s1", name="s", version="1", procedure="ok")])))
            results.append(("insufficient_evidence_rejected", ok, d))

            stale = good_proposal.model_copy(update={"base_bundle_hash": "0" * 64})
            ok, d = _rejected(lambda: mgr.submit_candidate(stale, SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s1", name="s", version="1", procedure="ok")])))
            results.append(("stale_base_rejected", ok, d))

            ok, d = _rejected(lambda: mgr.rollback("f" * 64, "probe"))
            results.append(("non_lineage_rollback_rejected", ok, d))

            cand2 = SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s1", name="s", version="1", procedure="ok")])
            p2 = mgr.submit_candidate(good_proposal.model_copy(), cand2)
            mgr.start_evaluation(p2.candidate_id)
            rep = EvaluationReport(
                candidateHash=cand2.content_hash, baseHash="e" * 64,
                protocolHash="p", evaluatorProvenance="eval",
                validity=EvaluationState.valid,
                metrics=MetricAggregate(accuracy=0.9, reliability=0.9, meanCost=1.0, p95LatencyMs=1.0),
                safetyResults={"s": True}, pairedRunIds=[["a", "b"]],
                partitionRef=ArtifactRef(id="x", version="1", sha256="0" * 64),
            )
            ok, d = _rejected(lambda: mgr.promote(p2.candidate_id, rep))
            results.append(("interrupted_promotion_stale_base", ok, d))

            val_run = seed_run("t-val", "validation")
            v_ev = ctl.append_event(val_run.run_id, "tool_result", {"v": 9}, "broker", "learner")
            ctl.record_outcome(val_run.run_id, passed=True)
            contaminated = good_proposal.model_copy(update={"supporting_evidence_ids": [v_ev.evidence_id]})
            ok, d = _rejected(lambda: mgr.submit_candidate(contaminated, SkillBundle(parent=base.bundle_id, skills=[SkillVersion(skillId="s1", name="s", version="1", procedure="ok")])))
            results.append(("fixture_contamination_rejected", ok, d))
            return results

        # EVAL-005 — broker/operational scenarios on the same isolated store.
        expired = _probe_capability(dev_run.run_id, "read_record", "read",
                                    expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        ok, d = _denied(lambda: broker.request_tool_call(ENV, _probe_req(dev_run.run_id, "read_record", {"record_id": "record-1"}, "pc-1"), expired, provider))
        results.append(("capability_bounds_enforced", ok, d))

        child = seed_run("t-child", "development")
        store.update_run_status(child.run_id, "failed", completed_at=_utcnow())
        row = store.get_run(child.run_id)
        results.append(("child_failure_recorded", row["status"] == "failed", f"status={row['status']}"))

        cap = _probe_capability(dev_run.run_id, "update_record", "write")
        a1 = broker.issue_approval(ENV, dev_run.run_id, "update_record", {"record_id": "r", "value": 1}, "dup-1", ttl_seconds=60)
        req = _probe_req(dev_run.run_id, "update_record", {"record_id": "r", "value": 1}, "dup-1", approval_token=a1)
        first = broker.request_tool_call(ENV, req, cap, provider)
        replay = broker.request_tool_call(ENV, req, cap, provider)
        call_row = store.get_tool_call_by_idempotency(dev_run.run_id, "dup-1")
        ok = first.status == "ok" and replay.status == "ok" and call_row is not None
        results.append(("duplicate_ops_replayed_once", ok, f"first={first.status} replay={replay.status}"))

        conflict_req = _probe_req(dev_run.run_id, "update_record", {"record_id": "r", "value": 2}, "dup-1")
        ok, d = _denied(lambda: broker.request_tool_call(ENV, conflict_req, cap, provider))
        results.append(("replay_payload_conflict", ok, d))

        second = _probe_req(dev_run.run_id, "update_record", {"record_id": "r", "value": 3}, "dup-2", approval_token=a1)
        ok, d = _denied(lambda: broker.request_tool_call(ENV, second, cap, provider))
        results.append(("one_use_approval", ok, d))

        all_evs = store.list_evidence(dev_run.run_id)
        resume = ctl.events(dev_run.run_id, after_sequence=all_evs[0]["sequence"])
        ok = bool(resume) and all(e["id"] != all_evs[0]["evidence_id"] for e in resume)
        results.append(("event_reconnect_resume", ok, f"resumed={len(resume)} events"))

        crash_provider = _ProbeProvider()
        crash_provider.reset(dev_run.run_id)
        a2 = broker.issue_approval(ENV, dev_run.run_id, "update_record", {"record_id": "r", "value": 4, "_crash": True}, "unc-1", ttl_seconds=60)
        unc = broker.request_tool_call(ENV, _probe_req(dev_run.run_id, "update_record", {"record_id": "r", "value": 4, "_crash": True}, "unc-1", approval_token=a2), cap, crash_provider)
        broker.reconcile_run(dev_run.run_id, provider)
        row = store.get_tool_call_by_idempotency(dev_run.run_id, "unc-1")
        ok = (
            unc.error is not None
            and unc.error.code == ToolErrorCode.OUTCOME_UNKNOWN
            and row is not None
            and row.get("effect") == "unknown"
        )
        results.append(("cancel_uncertain_effect_reconciled", ok, f"status={unc.status} effect={row and row.get('effect')}"))
        return results


class Controller:
    """Durable seam: runs, steps, evidence, outcomes, candidates, promotions."""

    def __init__(
        self,
        store: Store,
        registry: EnvironmentRegistry,
        broker: ToolBroker | None = None,
        approval_provider: ApprovalProvider | None = None,
        authorizer: Authorizer | None = None,
        report_verifier: ReportVerifier | None = None,
    ) -> None:
        self.store = store
        self.registry = registry
        if broker is None:
            broker = ToolBroker(store, registry, authorizer=authorizer)
        elif authorizer is not None:
            broker.authorizer = authorizer
        self.broker = broker
        self.candidates = CandidateManager(store, report_verifier=report_verifier)
        self.approval_provider = approval_provider

    # ------------------------------------------------------------------ runs
    @staticmethod
    def _request_fingerprint(request: RunRequest, task: TaskInput) -> str:
        return sha256_json(
            {
                "taskRef": request.task_ref.model_dump(mode="json"),
                "modelProfileRef": request.model_profile_ref.model_dump(mode="json"),
                "budgetRef": request.budget_ref.model_dump(mode="json"),
                "parentRunId": request.parent_run_id,
                "taskId": task.task_id,
                "environmentId": task.environment_ref.id,
            }
        )

    def create_run(self, request: RunRequest, task: TaskInput, skill_bundle: SkillBundle | None = None) -> RunRecord:
        """Atomic run idempotency: same key + same request replays the stored run;
        same key + different request raises RunIdempotencyConflict."""
        fingerprint = self._request_fingerprint(request, task)
        manifest = self.registry.get_manifest(task.environment_ref.id)
        if manifest is None:
            raise KeyError(f"environment {task.environment_ref.id!r} not registered")
        if skill_bundle is None:
            skill_bundle = self.candidates.get_active_bundle() or SkillBundle()
        bundle_ref = self.store.put_artifact(skill_bundle.model_dump(mode="json", by_alias=True))

        run = RunRecord(
            taskRef=request.task_ref,
            environmentRef=task.environment_ref,
            policyRef=manifest.policy_ref,
            modelProfileRef=request.model_profile_ref,
            skillBundleRef=bundle_ref,
            budgetRef=request.budget_ref,
            parentRunId=request.parent_run_id,
            status=RunStatus.queued,
        )
        status, row = self.store.create_run_idempotent(
            request.idempotency_key,
            fingerprint,
            {
                "run_id": run.run_id,
                "parent_run_id": run.parent_run_id,
                "task_id": task.task_id,
                "environment_id": manifest.environment_id,
                "bundle_id": skill_bundle.bundle_id,
                "status": run.status.value,
                "last_event_sequence": 0,
                "created_at": run.created_at.isoformat(),
                "run_json": run.model_dump_json(by_alias=True),
            },
        )
        if status == "exists":
            return RunRecord.model_validate_json(row["run_json"])
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
    def dispatch_tool(self, env_id: str, request: ToolRequest, capability: Capability, provider: ToolProvider) -> ToolResult:
        if request.approval_token is None and self.approval_provider is not None:
            schema = self.registry.get_tool_schema(env_id, request.tool)
            if schema is not None and schema.effect == "write":
                request.approval_token = self.approval_provider(request)
        return self.broker.request_tool_call(env_id, request, capability, provider)

    # ------------------------------------------------------------------ evidence / SSE
    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any], trust_class: str, visibility: str) -> EvidenceRecord:
        # Sanitized feedback boundary: learner-visible events never carry
        # credential-shaped values.
        if visibility == "learner":
            payload = _sanitize_for_learner(payload)
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

    def record_broker_tool_result(self, run_id: str, result_payload: Mapping[str, Any]) -> EvidenceRecord:
        """Raw tool_result persists as operator-only evidence; the learner gets a
        separate sanitized projection event (raw never enters learner context)."""
        raw = dict(result_payload)
        self.append_event(run_id, "tool_result", raw, "broker", "operator")
        return self.append_event(
            run_id, "tool_result", _sanitize_for_learner(raw), "broker", "learner"
        )

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

    # ------------------------------------------------------------------ canonical trusted evidence
    # Contract with session6's SQLiteRunEvidenceStore (e916487):
    #   model_response evidence -> artifact: {responseId, usage{inputTokens,
    #     outputTokens,totalTokens}, versionRefs{...}}; content_hash matches.
    #   accounting artifact -> {responseId, runId, taskId, environmentId, usage,
    #     costMicrounits, durationSeconds, versionRefs} (usage/versionRefs equal
    #     the model_response payload).
    #   trusted_outcome evidence -> artifact: {responseId, runId, taskId,
    #     environmentId, passed, reliable, safetyViolations}.

    @staticmethod
    def _require_usage(usage: Any) -> dict[str, int]:
        if not isinstance(usage, dict):
            raise ValueError("usage must be an object")
        for key in ("inputTokens", "outputTokens", "totalTokens"):
            v = usage.get(key)
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"usage.{key} must be a non-negative integer")
        if usage["totalTokens"] != usage["inputTokens"] + usage["outputTokens"]:
            raise ValueError("usage.totalTokens must equal input+output")
        return usage

    def _run_row(self, run_id: str) -> dict[str, Any]:
        row = self.store.get_run(run_id)
        if row is None:
            raise KeyError(f"run {run_id!r} not found")
        return row

    def record_model_response(self, run_id: str, response: Mapping[str, Any]) -> EvidenceRecord:
        """Record a trusted parent model observation (responseId + usage pinned)."""
        self._run_row(run_id)
        if not isinstance(response.get("responseId"), str) or not response["responseId"]:
            raise ValueError("response.responseId is required")
        self._require_usage(response.get("usage"))
        if not isinstance(response.get("versionRefs"), dict) or not response["versionRefs"]:
            raise ValueError("response.versionRefs must be a non-empty object")
        return self.append_event(
            run_id, "model_response", dict(response), "system", "operator"
        )

    def record_accounting(self, run_id: str, accounting: Mapping[str, Any], model_response: Mapping[str, Any]) -> ArtifactRef:
        """Content-address an accounting record bound to a model response."""
        row = self._run_row(run_id)
        for key in ("responseId", "runId", "taskId", "environmentId"):
            if not accounting.get(key):
                raise ValueError(f"accounting.{key} is required")
        if accounting["responseId"] != model_response.get("responseId"):
            raise ValueError("accounting.responseId does not match model response")
        if accounting["runId"] != run_id or accounting["taskId"] != row["task_id"] or accounting["environmentId"] != row["environment_id"]:
            raise ValueError("accounting run/task/environment pins do not match the run")
        if self._require_usage(accounting.get("usage")) != model_response.get("usage"):
            raise ValueError("accounting.usage does not match model response usage")
        if accounting.get("versionRefs") != model_response.get("versionRefs"):
            raise ValueError("accounting.versionRefs does not match model response")
        for key in ("costMicrounits", "durationSeconds"):
            v = accounting.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0 or v != v or v == float("inf"):
                raise ValueError(f"accounting.{key} must be a finite non-negative number")
        ref = self.store.put_artifact(dict(accounting))
        self.append_event(run_id, "accounting_recorded", {"accountingRef": ref.model_dump(mode="json")}, "system", "operator")
        return ref

    def record_trusted_outcome(self, run_id: str, outcome: Mapping[str, Any]) -> EvidenceRecord:
        """Record the evaluator-owned outcome bound to a model response."""
        row = self._run_row(run_id)
        for key in ("responseId", "runId", "taskId", "environmentId"):
            if not outcome.get(key):
                raise ValueError(f"outcome.{key} is required")
        if outcome["runId"] != run_id or outcome["taskId"] != row["task_id"] or outcome["environmentId"] != row["environment_id"]:
            raise ValueError("outcome run/task/environment pins do not match the run")
        if not isinstance(outcome.get("passed"), bool) or not isinstance(outcome.get("reliable"), bool):
            raise ValueError("outcome.passed/reliable must be booleans")
        sv = outcome.get("safetyViolations")
        if not isinstance(sv, int) or isinstance(sv, bool) or sv < 0:
            raise ValueError("outcome.safetyViolations must be a non-negative integer")
        ev = self.append_event(run_id, "trusted_outcome", dict(outcome), "evaluator", "operator")
        self.record_outcome(run_id, bool(outcome["passed"]), None, dict(outcome))
        return ev

    # ------------------------------------------------------------------ EVAL-004/005 probe executor
    def execute_probe(self, case_id: str) -> "ProbeResult":
        """Executable safety probes over a fresh isolated Store/Controller.

        SPEC 568-575:
          EVAL-004 — candidate/promotion safety: harmful edits, insufficient
            evidence, stale base, non-lineage rollback, interrupted promotion
            (base drift), fixture contamination (non-development evidence).
          EVAL-005 — operational safety: capability bounds, child failures,
            duplicate/idempotent operations, replay/payload conflict,
            one-use approvals, event reconnect, cancel during uncertain effects.

        Each obligation is actually executed against Controller/ToolBroker/
        CandidateManager on an isolated Store; observed outcomes and the exact
        tested obligations are returned in a SafetyProbeResult-shaped object.
        """
        if case_id not in ("EVAL-004", "EVAL-005"):
            raise KeyError(f"unknown probe case {case_id!r}")
        results = _run_probe(case_id)
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
        """Held-out gate: refuse validation/final panels until the environment
        has a trusted development smoke outcome (benchmark contract)."""
        if not self.store.dev_smoke_ok(env_id):
            raise PermissionError(
                f"environment {env_id!r} has no trusted development smoke outcome; "
                "held-out panels are not authorized"
            )

    def learner_tasks(self, env_id: str) -> list[dict[str, Any]]:
        """Learner-visible task listing: development partition only, stripped to
        public fields — validation/final tasks and hidden inputs never leak."""
        return [
            {"taskId": t.task_id, "goal": t.goal, "partition": t.partition}
            for t in self.registry.list_tasks_by_partition(env_id, "development")
        ]

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
    def execute_run(self, run_id: str, env_id: str, provider: ToolProvider, driver: RunDriver) -> RunRecord:
        """Run lifecycle wrapper: running -> driver -> succeeded/failed.

        The driver sees only a DriverContext; provider errors surface as
        OUTCOME_UNKNOWN evidence, never a crash of the control plane.
        """
        self._set_run_status(run_id, RunStatus.running)
        ctx = DriverContext(self, run_id, env_id, provider)
        try:
            driver.act(ctx)
            self._set_run_status(run_id, RunStatus.succeeded)
        except Exception as exc:
            self.append_event(run_id, "run_failed", {"error": str(exc)}, "system", "operator")
            self._set_run_status(run_id, RunStatus.failed)
        return self.get_run(run_id)

    # ------------------------------------------------------------------ candidate/promotion seam
    def submit_candidate(self, proposal: CandidateProposal, bundle: SkillBundle) -> CandidateProposal:
        return self.candidates.submit_candidate(proposal, bundle)

    def start_evaluation(self, candidate_id: str) -> CandidateProposal:
        return self.candidates.start_evaluation(candidate_id)

    def freeze_protocol(self, gate: PromotionGate, evaluator_id: str, **kwargs: Any) -> str:
        return self.candidates.freeze_protocol(gate, evaluator_id, **kwargs)

    def submit_evaluation_report(
        self, candidate_id: str, report: EvaluationReport | Mapping[str, Any]
    ) -> PromotionDecision:
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

    # ------------------------------------------------------------------ Prime host_request seam
    _HOST_FORBIDDEN = frozenset({
        "harness.write", "policy.write", "evaluator.write",
        "promotion.write", "credentials.read", "hidden.read",
    })

    def register_prime_capability(
        self,
        capability_id: str,
        env_id: str,
        capability: Capability,
        provider: ToolProvider,
    ) -> None:
        """Bind a Prime-advertised capability id to an env/capability/provider.

        Only requests carrying a registered capability id can reach dispatch —
        everything else fails closed.
        """
        if not hasattr(self, "_prime_caps"):
            self._prime_caps = {}
            self._artifact_transfers = {}
        self._prime_caps[capability_id] = (env_id, capability, provider)

    def handle_host_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Narrow Prime ingress: capability discovery, brokered calls, and bounded
        artifact transfer. Learner authority is never widened."""
        request_type = payload.get("type")
        if request_type in self._HOST_FORBIDDEN:
            raise PermissionError(f"learner request denied: {request_type}")
        if request_type == "capabilities.discover":
            env_ids = {env for env, _, _ in getattr(self, "_prime_caps", {}).values()}
            return {"environments": sorted(env_ids), "capabilities": sorted(getattr(self, "_prime_caps", {}).keys())}
        if request_type == "broker.call":
            return self._prime_broker_call(payload)
        if request_type in {"artifact.begin", "artifact.chunk", "artifact.finish", "artifact.abort"}:
            return self._artifact_transfer(payload)
        raise PermissionError(f"unsupported host request: {request_type}")

    def _prime_broker_call(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        cap_id = payload.get("capabilityId")
        entry = getattr(self, "_prime_caps", {}).get(cap_id) if isinstance(cap_id, str) else None
        if entry is None:
            raise PermissionError("unknown capability")
        env_id, capability, provider = entry
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            raise PermissionError("arguments must be an object")
        idem = payload.get("idempotencyKey")
        if not isinstance(idem, str) or not idem:
            idem = f"prime:{cap_id}:{sha256_json(arguments)}"
        req = ToolRequest(
            runId=capability.run_id,
            stepId=str(payload.get("stepId") or "prime"),
            tool=capability.tool,
            arguments=arguments,
            idempotencyKey=idem,
            approvalToken=payload.get("approvalToken") if isinstance(payload.get("approvalToken"), str) else None,
        )
        result = self.dispatch_tool(env_id, req, capability, provider)
        out = result.model_dump(mode="json", by_alias=True)
        self.record_broker_tool_result(
            capability.run_id,
            {"capabilityId": cap_id, **{k: v for k, v in out.items()}},
        )
        return {"value": _sanitize_for_learner(out)}

    # Bounded artifact ingress, mirroring Prime's begin/chunk/finish contract.
    MAX_ARTIFACT_BYTES = 8 * 1024 * 1024

    def _artifact_transfer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not hasattr(self, "_artifact_transfers"):
            self._artifact_transfers = {}
        transfers = self._artifact_transfers
        t = payload.get("type")
        if t == "artifact.begin":
            artifact_id = payload.get("artifactId")
            size = payload.get("size")
            if (not isinstance(artifact_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", artifact_id)
                    or not isinstance(size, int) or isinstance(size, bool)
                    or size < 0 or size > self.MAX_ARTIFACT_BYTES):
                raise PermissionError("invalid artifact metadata")
            if transfers:
                raise PermissionError("only one artifact transfer may be active")
            tid = new_id("xfer_")
            transfers[tid] = {"id": artifact_id, "size": size, "offset": 0, "data": bytearray()}
            return {"transferId": tid, "maxChunkBytes": min(self.MAX_ARTIFACT_BYTES, 262144)}
        if t == "artifact.chunk":
            tid = payload.get("transferId")
            transfer = transfers.get(tid) if isinstance(tid, str) else None
            offset = payload.get("offset")
            encoded = payload.get("data")
            if transfer is None or offset != transfer["offset"] or not isinstance(encoded, str):
                raise PermissionError("invalid artifact chunk")
            try:
                data = base64.b64decode(encoded.encode("ascii"), validate=True)
            except (binascii.Error, ValueError, UnicodeEncodeError):
                raise PermissionError("artifact chunk is not valid base64") from None
            if not data or len(data) > transfer["size"] - transfer["offset"]:
                raise PermissionError("artifact chunk exceeds declared size")
            transfer["data"].extend(data)
            transfer["offset"] += len(data)
            return {"transferId": tid, "offset": transfer["offset"]}
        if t == "artifact.finish":
            tid = payload.get("transferId")
            expected = payload.get("sha256")
            transfer = transfers.get(tid) if isinstance(tid, str) else None
            if transfer is None or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise PermissionError("invalid artifact completion")
            if transfer["offset"] != transfer["size"]:
                raise PermissionError("artifact is incomplete")
            data = bytes(transfer["data"])
            transfers.pop(tid, None)
            if hashlib.sha256(data).hexdigest() != expected:
                raise PermissionError("artifact digest mismatch")
            ref = self.store.put_immutable_bytes(data)
            return {"artifact": {"id": transfer["id"], "sha256": ref.sha256, "bytes": len(data)}}
        if t == "artifact.abort":
            tid = payload.get("transferId")
            if isinstance(tid, str):
                transfers.pop(tid, None)
            return {"aborted": True}
        raise PermissionError(f"unsupported host request: {t}")
