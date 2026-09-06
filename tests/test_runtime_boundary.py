"""Regression coverage for runtime evaluation retry identity.

This test targets the integrated application runtime owned by the runtime
worker.  The evaluator worktree intentionally has no application module, so
it is skipped here and runs when this file is applied to that runtime branch.
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace

import pytest

app_module = pytest.importorskip("adaptive_agent.app")

from adaptive_agent.benchmark import FrozenExecutionConfig  # noqa: E402
from adaptive_agent.evaluation import Arm, EvaluationProtocol, Partition  # noqa: E402


def test_runtime_retry_uses_fresh_run_identity_after_pre_receipt_failure(tmp_path):
    """A failed attempt must remain immutable while a declared retry can run."""

    calls = 0

    class FailThenFinish:
        def __call__(self, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("provider failed before producing a receipt")
            return SimpleNamespace(
                text="finish",
                provider="openai-codex",
                model="openai-codex/gpt-5.6-luna",
                response_id="retry-response",
                usage={"inputTokens": 1, "outputTokens": 1},
            )

    app = app_module.create_runtime_app(
        model_runner=FailThenFinish(),
        evaluator=lambda **_kwargs: {"passed": True},
        data_dir=tmp_path,
    )
    runtime = app.state.durable_runtime
    protocol = EvaluationProtocol()
    protocol.freeze(runtime.packages)
    task = runtime.registry.list_tasks_by_partition("finance", "development")[0]
    bundle = runtime.controller.get_active_bundle()
    assert bundle is not None
    bundle_hash = getattr(bundle, "content_hash", None)
    if not isinstance(bundle_hash, str):
        bundle_hash = bundle["content_hash"]

    def config(attempt: int) -> FrozenExecutionConfig:
        values = {
            "protocol": protocol.start_candidate_generation(),
            "arm": Arm.B0,
            "seed": 17,
            "bundle_hash": bundle_hash,
        }
        if any(field.name == "attempt" for field in fields(FrozenExecutionConfig)):
            values["attempt"] = attempt
        return FrozenExecutionConfig(**values)

    with pytest.raises(ValueError, match="trusted model and outcome evidence"):
        runtime.execute_evaluation_task(task, config(0), bundle)

    with runtime.controller.store.connect() as connection:
        failed_runs = connection.execute(
            "SELECT run_id, status FROM runs ORDER BY created_at"
        ).fetchall()
    assert len(failed_runs) == 1
    assert failed_runs[0]["status"] == "failed"
    failed_run_id = failed_runs[0]["run_id"]

    observation = runtime.execute_evaluation_task(task, config(1), bundle)

    assert observation.run_id is not None
    assert observation.run_id != failed_run_id
    assert runtime.get_run(failed_run_id)["status"] == "failed"
    assert runtime.get_run(observation.run_id)["status"] == "succeeded"
    assert calls == 2
