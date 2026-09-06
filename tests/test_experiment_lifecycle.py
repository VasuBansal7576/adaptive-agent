from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_agent.evaluation import EvaluationProtocol, build_environment_packages
from adaptive_agent.evaluation_job import EvaluationJob, LifecycleStage
from adaptive_agent.experiment_lifecycle import (
    ExperimentLifecycleError,
    InfrastructureFailure,
    LifecyclePolicy,
    lifecycle_json,
    run_experiment_lifecycle,
)
from adaptive_agent.store import Store


def _job(tmp_path: Path) -> EvaluationJob:
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    return EvaluationJob(Store(tmp_path), object(), protocol, packages, {}, lambda *_: None)


def _stages(seen: list[tuple[str, str]]) -> tuple[LifecycleStage, ...]:
    names = ("bootstrap", "training", "learning", "transfer", "adaptation", "safety", "validation", "final")

    def callback(cell: str, context: dict[str, object]) -> dict[str, object]:
        seen.append((str(context["stage"]), cell))
        return {"usage": {"inputTokens": 1, "outputTokens": 1}, "toolCalls": 1, "wallSeconds": 0.01, "costMicrounits": 1}

    return tuple(LifecycleStage(name, (f"{name}-0",), callback) for name in names)


def test_complete_lifecycle_is_ordered_resumable_and_does_not_repeat_success(tmp_path: Path):
    job = _job(tmp_path)
    first_seen: list[tuple[str, str]] = []
    stages = _stages(first_seen)
    limits = {"attempts": 8, "inputTokens": 16, "outputTokens": 16, "toolCalls": 8, "wallMicros": 8_000_000, "costMicrounits": 8}
    first = job.run_experiment("experiment", stages, limits=limits)
    assert first.status == "complete"
    assert [stage for stage, _ in first_seen] == [stage.name for stage in stages]
    second_seen: list[tuple[str, str]] = []
    second = job.run_experiment("experiment", _stages(second_seen), limits=limits)
    assert second.status == "complete"
    assert second_seen == []
    assert job.lifecycle_accounting("experiment")["attempts"] == 8


def test_lifecycle_budget_exhaustion_stops_future_launches(tmp_path: Path):
    job = _job(tmp_path)
    stages = tuple(LifecycleStage(name, (f"{name}-0",), lambda *_: {"costMicrounits": 1}) for name in ("bootstrap", "training", "learning", "transfer", "adaptation", "safety", "validation", "final"))
    result = job.run_experiment("limited", stages, limits={"attempts": 2, "inputTokens": 100, "outputTokens": 100, "toolCalls": 100, "wallMicros": 100_000_000, "costMicrounits": 100})
    assert result.status == "failed"
    assert result.runtime_accounting is not None
    assert result.runtime_accounting["attempts"] == 2
    assert "budget exhausted" in (result.error or "")


def test_lifecycle_rejects_stage_order_that_could_leak_held_out_data(tmp_path: Path):
    job = _job(tmp_path)
    stages = _stages([])
    with pytest.raises(ValueError, match="ordered"):
        job.run_experiment("leak", (stages[-1], *stages[:-1]))


class _LifecycleCallbacks:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def bootstrap_development(self, policy, *, attempt):
        self.calls.append(("bootstrap", attempt))
        return {"clean": True, "pins": {name: name for name in policy.required_pins}, "developmentTaskIds": list(policy.development_task_ids), "exposedEnvironments": [], "executed": True, "runIds": ["bootstrap"], "status": "complete"}

    def run_development(self, bootstrap, policy, *, attempt):
        self.calls.append(("development", attempt))
        return {"executed": True, "runIds": ["development"], "taskIds": list(policy.development_task_ids), "exposedEnvironments": [], "learning": {"actual": True}, "partition": "development", "status": "complete"}

    def generate_candidate(self, development, bootstrap, *, attempt):
        self.calls.append(("candidate", attempt))
        return {"actual": True, "candidateId": "candidate", "baseBundleHash": "activeBundleHash", "sourceRunIds": ["development"]}

    def run_transfer(self, candidate, environment_id, *, reset, exposed, attempt):
        self.calls.append(("transfer", attempt))
        return {"executed": True, "runIds": ["transfer"], "environmentId": environment_id, "resetBefore": reset, "exposedToLearning": exposed, "status": "complete"}

    def run_adaptation(self, candidate, environment_id, support_task_ids, *, reset, attempt):
        self.calls.append(("adaptation", attempt))
        return {"executed": True, "runIds": ["adaptation"], "environmentId": environment_id, "supportTaskIds": list(support_task_ids), "resetBefore": reset, "queryExposedToLearning": False, "status": "complete"}

    def run_registered_safety(self, candidate, case_ids, *, attempt):
        self.calls.append(("safety", attempt))
        return {"registeredCaseIds": list(case_ids), "passedCaseIds": list(case_ids)}

    def run_validation(self, candidate, *, attempt):
        self.calls.append(("validation", attempt))
        return {"executed": True, "runIds": ["validation"], "partition": "validation", "heldOut": True, "exposureLeak": False, "status": "complete"}

    def run_final(self, candidate, *, attempt):
        self.calls.append(("final", attempt))
        return {"executed": True, "runIds": ["final"], "partition": "final", "sealed": True, "exposureLeak": False, "selectedCandidateAfterObservation": False, "status": "complete"}


def _lifecycle_policy() -> LifecyclePolicy:
    return LifecyclePolicy(("dev-1",), "reserved", ("support-1",), infrastructure_retries=1)


def test_bounded_helper_requires_real_ordered_receipts_and_discloses_boundary():
    callbacks = _LifecycleCallbacks()
    result = run_experiment_lifecycle(callbacks, _lifecycle_policy())
    assert result.status == "complete"
    assert [name for name, _ in callbacks.calls] == ["bootstrap", "development", "candidate", "transfer", "adaptation", "safety", "validation", "final"]
    assert "No paid calls" in lifecycle_json(result)


def test_bounded_helper_retries_only_declared_infrastructure_failures():
    callbacks = _LifecycleCallbacks()
    original = callbacks.run_development
    calls = []

    def flaky(bootstrap, policy, *, attempt):
        calls.append(attempt)
        if attempt == 0:
            raise InfrastructureFailure("transient")
        return original(bootstrap, policy, attempt=attempt)

    callbacks.run_development = flaky
    result = run_experiment_lifecycle(callbacks, _lifecycle_policy())
    assert result.status == "complete"
    assert calls == [0, 1]


def test_bounded_helper_rejects_workload_only_receipts():
    callbacks = _LifecycleCallbacks()
    callbacks.run_transfer = lambda *args, **kwargs: {"transferRuns": 100}
    with pytest.raises(ExperimentLifecycleError, match="executed durable receipt"):
        run_experiment_lifecycle(callbacks, _lifecycle_policy())
