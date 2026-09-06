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
