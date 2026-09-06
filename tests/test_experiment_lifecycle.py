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
