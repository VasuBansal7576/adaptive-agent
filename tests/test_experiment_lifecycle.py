from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_agent.evaluation import EvaluationProtocol, build_environment_packages
from adaptive_agent.evaluation_job import EvaluationJob, LifecycleStage
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
        return {"status": "complete", "stage": context["stage"], "cellKey": cell, "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "toolCalls": 1, "wallSeconds": 0.01, "costMicrounits": 1}

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
    def receipt(cell, context):
        return {"status": "complete", "stage": context["stage"], "cellKey": cell, "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, "costMicrounits": 1}
    stages = tuple(LifecycleStage(name, (f"{name}-0",), receipt) for name in ("bootstrap", "training", "learning", "transfer", "adaptation", "safety", "validation", "final"))
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


def test_failed_cell_is_not_reused_or_allowed_to_advance_on_resume(tmp_path: Path):
    job = _job(tmp_path)
    seen: list[str] = []

    def failing(cell, context):
        seen.append(str(context["stage"]))
        raise RuntimeError("infrastructure unavailable")

    stages = list(_stages([]))
    stages[0] = LifecycleStage("bootstrap", ("bootstrap-0",), failing)
    first = job.run_experiment("failed-resume", tuple(stages))
    second = job.run_experiment("failed-resume", tuple(stages))
    assert first.status == second.status == "failed"
    assert seen == ["bootstrap"]
    assert job.lifecycle_accounting("failed-resume")["attempts"] == 1


def test_declared_infrastructure_retry_uses_a_new_attempt(tmp_path: Path):
    job = _job(tmp_path)
    stages = list(_stages([]))
    original = stages[0].callback
    attempts: list[int] = []

    def retrying(cell, context):
        attempts.append(int(context["attempt"]))
        if len(attempts) == 1:
            raise RuntimeError("transient infrastructure failure")
        return original(cell, context)

    stages[0] = LifecycleStage("bootstrap", ("bootstrap-0",), retrying, retries=1)
    result = job.run_experiment("declared-retry", tuple(stages))
    assert result.status == "complete"
    assert attempts == [0, 1]
    assert job.lifecycle_accounting("declared-retry")["attempts"] == 9


def test_malformed_accounting_is_not_charged_or_double_counted(tmp_path: Path):
    job = _job(tmp_path)
    seen = 0

    def malformed(cell, context):
        nonlocal seen
        seen += 1
        return {"status": "complete", "stage": context["stage"], "cellKey": cell, "usage": {"inputTokens": -1, "outputTokens": 1, "totalTokens": 0}, "costMicrounits": 1}

    stages = list(_stages([]))
    stages[0] = LifecycleStage("bootstrap", ("bootstrap-0",), malformed)
    first = job.run_experiment("malformed", tuple(stages))
    second = job.run_experiment("malformed", tuple(stages))
    assert first.status == second.status == "failed"
    assert seen == 1
    assert job.lifecycle_accounting("malformed")["inputTokens"] == 0
    assert job.lifecycle_accounting("malformed")["costMicrounits"] == 0


def test_nested_subcall_admission_retains_failed_usage_once(tmp_path: Path):
    job = _job(tmp_path)
    stages = list(_stages([]))
    original = stages[0].callback
    subcalls = 0

    def with_subcall(cell, context):
        nonlocal subcalls
        subcalls += 1
        admission = context["admitSubcall"]("child-0", estimated_input_tokens=3, estimated_output_tokens=2, estimated_tool_calls=1, estimated_cost_microunits=5)
        context["recordSubcall"](admission["admissionId"], result={"usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}, "toolCalls": 1, "wallSeconds": 0.01, "costMicrounits": 5}, error="child infrastructure failure")
        return original(cell, context)

    stages[0] = LifecycleStage("bootstrap", ("bootstrap-0",), with_subcall)
    limits = {"attempts": 8, "inputTokens": 16, "outputTokens": 16, "toolCalls": 16, "wallMicros": 8_000_000, "costMicrounits": 16}
    first = job.run_experiment("subcalls", tuple(stages), limits=limits)
    second = job.run_experiment("subcalls", tuple(stages), limits=limits)
    assert first.status == second.status == "complete"
    assert subcalls == 1
    assert job.lifecycle_accounting("subcalls")["subcalls"] == 1
    assert job.lifecycle_accounting("subcalls")["inputTokens"] == 11


def test_malformed_nested_subcall_accounting_is_not_charged(tmp_path: Path):
    job = _job(tmp_path)
    job._lifecycle_budget("bad-subcall", {"attempts": 1, "inputTokens": 10, "outputTokens": 10, "toolCalls": 10, "wallMicros": 10_000, "costMicrounits": 10})
    admission = job.admit_lifecycle_subcall("bad-subcall", "bootstrap", "bootstrap-0", "child-0")
    with pytest.raises(ValueError, match="subcall accounting"):
        job.record_lifecycle_subcall(admission["admissionId"], result={"usage": {"inputTokens": -1, "outputTokens": 0, "totalTokens": -1}, "costMicrounits": 1})
    assert job.lifecycle_accounting("bad-subcall")["inputTokens"] == 0
    assert job.lifecycle_accounting("bad-subcall")["costMicrounits"] == 0


def test_unknown_nested_cost_blocks_future_admission(tmp_path: Path):
    job = _job(tmp_path)
    job._lifecycle_budget("unknown-subcall", {"attempts": 2, "inputTokens": 10, "outputTokens": 10, "toolCalls": 10, "wallMicros": 10_000, "costMicrounits": 10})
    admission = job.admit_lifecycle_subcall("unknown-subcall", "transfer", "leave-out:finance", "child-0")
    job.record_lifecycle_subcall(admission["admissionId"], result={"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "economicCostStatus": "unknown"})
    with pytest.raises(ValueError, match="budget exhausted"):
        job.admit_lifecycle_subcall("unknown-subcall", "transfer", "leave-out:finance", "child-1")
