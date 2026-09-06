from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adaptive_agent.app import create_runtime_app
from adaptive_agent.models import CandidateProposal, SkillBundle
from adaptive_agent.evaluation import (
    Arm,
    BudgetSpec,
    ModelProvenance,
    Partition,
    Provenance,
    RunObservation,
)


def _candidate(runtime, candidate_id: str, base_hash: str) -> str:
    active = runtime.controller.get_active_bundle()
    assert active is not None and active.content_hash == base_hash
    bundle = SkillBundle(
        parent=base_hash,
        skills=list(active.skills),
        executionConfig=active.execution_config,
    )
    runtime.controller.store.save_bundle(
        bundle.bundle_id,
        bundle.parent,
        bundle.content_hash,
        bundle.model_dump_json(by_alias=True),
        is_active=False,
    )
    proposal = CandidateProposal(
        candidate_id=candidate_id,
        baseBundleHash=base_hash,
        candidateBundleHash=bundle.content_hash,
        predictedEffect="diagnostic",
        proposerVersion="test",
        state="validated",
    )
    runtime.controller.store.save_candidate(
        candidate_id,
        {
            "base_bundle_hash": base_hash,
            "candidate_bundle_hash": bundle.content_hash,
            "candidate_json": proposal.model_dump_json(by_alias=True),
            "state": "validated",
            "created_at": proposal.created_at.isoformat(),
        },
    )
    return bundle.content_hash


def _scripted_executor(runtime, passed, before_return=None):
    accounting = runtime.controller.store.put_artifact(
        {"aggregateUsage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3}}
    )
    runtime.verify_evaluation_observation = lambda *_args: True

    def execute(task, config, bundle):
        if before_return is not None:
            before_return()
        return RunObservation(
            task.task_id,
            task.environment_ref.id,
            Partition.DEVELOPMENT,
            config.seed,
            config.arm,
            bool(passed(config) if callable(passed) else passed),
            True,
            0,
            0,
            0.25,
            model_provenance=ModelProvenance.REAL_MODEL,
            accounting_ref=accounting.sha256,
            bundle_hash=config.bundle_hash,
        )

    return execute


def test_diagnostic_api_runs_six_cells_and_is_separate_from_promotion(tmp_path: Path):
    app = create_runtime_app(data_dir=tmp_path)
    runtime = app.state.durable_runtime
    active = runtime.controller.get_active_bundle()
    assert active is not None
    candidate_hash = _candidate(runtime, "cand_diag", active.content_hash)
    calls: list[tuple[str, str]] = []

    scripted = _scripted_executor(runtime, lambda config: config.arm == "L")

    def executor(task, config, bundle):
        calls.append((task.task_id, config.arm))
        return scripted(task, config, bundle)

    runtime.diagnostics.executor = executor
    client = TestClient(app, base_url="http://127.0.0.1")
    assert client.get("/session/bootstrap").status_code == 200
    response = client.post(
        "/diagnostics/launch",
        json={"candidateId": "cand_diag", "baseBundleHash": active.content_hash},
    )
    assert response.status_code == 202
    diagnostic = response.json()
    assert diagnostic["candidateBundleHash"] == candidate_hash
    assert diagnostic["totalCells"] == 6
    assert diagnostic["promotionEligible"] is False
    assert diagnostic["purpose"] == "development_diagnostic"
    repeat = runtime.diagnostics.create("cand_diag", active.content_hash)
    assert repeat["diagnosticId"] == diagnostic["diagnosticId"]
    result = runtime.diagnostics.run(diagnostic["diagnosticId"])
    assert result["state"] == "completed"
    assert result["completedCells"] == 6
    assert len(calls) == 6
    assert {task for task, _ in calls} == {
        "finance-development-00",
        "customer_support-development-00",
        "it-development-00",
    }
    assert [item["completed"] for item in result["armSummaries"]] == [3, 3]
    assert [item["successes"] for item in result["armSummaries"]] == [0, 3]
    with runtime.controller.store.connect() as conn:
        failures = [
            row[0]
            for row in conn.execute(
                "SELECT failure_class FROM diagnostic_cells WHERE diagnostic_id = ? AND arm = 'B0'",
                (diagnostic["diagnosticId"],),
            ).fetchall()
        ]
    assert failures == ["task_failure"] * 3
    assert runtime.controller.store.list_promotions() == []
    assert runtime.controller.get_candidate("cand_diag")["state"] == "validated"
    assert client.get("/diagnostics").json()[0]["state"] == "completed"


def test_diagnostic_rejects_stale_base_and_repeated_run_resumes_without_calls(
    tmp_path: Path,
):
    app = create_runtime_app(data_dir=tmp_path)
    runtime = app.state.durable_runtime
    active = runtime.controller.get_active_bundle()
    assert active is not None
    _candidate(runtime, "cand_stale", active.content_hash)
    stale_bundle = SkillBundle(
        parent=active.content_hash,
        skills=list(active.skills),
        executionConfig=active.execution_config,
    )
    runtime.controller.store.save_bundle(
        stale_bundle.bundle_id,
        stale_bundle.parent,
        stale_bundle.content_hash,
        stale_bundle.model_dump_json(by_alias=True),
        is_active=False,
    )
    runtime.controller.store.set_active_bundle(stale_bundle.content_hash)
    with pytest.raises(ValueError, match="stale"):
        runtime.diagnostics.create("cand_stale", active.content_hash)
    runtime.controller.store.set_active_bundle(active.content_hash)
    calls = 0
    scripted = _scripted_executor(runtime, True)

    def executor(task, config, bundle):
        nonlocal calls
        calls += 1
        return scripted(task, config, bundle)

    runtime.diagnostics.executor = executor
    diagnostic = runtime.diagnostics.create("cand_stale", active.content_hash)
    runtime.diagnostics.run(diagnostic["diagnosticId"])
    assert calls == 6
    runtime.diagnostics.run(diagnostic["diagnosticId"])
    assert calls == 6


def test_diagnostic_cancel_marks_queued_cells_and_does_not_promote(tmp_path: Path):
    app = create_runtime_app(data_dir=tmp_path)
    runtime = app.state.durable_runtime
    active = runtime.controller.get_active_bundle()
    assert active is not None
    _candidate(runtime, "cand_cancel", active.content_hash)
    started = threading.Event()
    release = threading.Event()

    runtime.diagnostics.executor = _scripted_executor(
        runtime, True, lambda: (started.set(), release.wait(timeout=2))
    )
    diagnostic = runtime.diagnostics.create("cand_cancel", active.content_hash)
    worker = threading.Thread(
        target=runtime.diagnostics.run, args=(diagnostic["diagnosticId"],)
    )
    worker.start()
    assert started.wait(timeout=2)
    cancelled = runtime.diagnostics.cancel(diagnostic["diagnosticId"])
    assert cancelled["state"] == "running"
    assert "admitted cells will finish" in cancelled["error"]
    release.set()
    worker.join(timeout=5)
    assert runtime.diagnostics.get(diagnostic["diagnosticId"])["state"] == "cancelled"
    assert runtime.controller.store.list_promotions() == []


def test_diagnostic_uses_aggregate_final_accounting_over_last_turn_usage(
    tmp_path: Path,
):
    app = create_runtime_app(data_dir=tmp_path)
    runtime = app.state.durable_runtime
    active = runtime.controller.get_active_bundle()
    assert active is not None
    candidate_hash = _candidate(runtime, "cand_accounting", active.content_hash)
    accounting = runtime.controller.store.put_artifact(
        {
            "aggregateUsage": {
                "inputTokens": 21923,
                "outputTokens": 3539,
                "totalTokens": 25462,
            },
            "usage": {"inputTokens": 8556, "outputTokens": 419, "totalTokens": 8975},
        }
    )
    seed_observation = RunObservation(
        "placeholder",
        "finance",
        Partition.DEVELOPMENT,
        17,
        Arm.B0,
        True,
        True,
        0,
        0,
        0.1,
        model_provenance=ModelProvenance.REAL_MODEL,
        provenance=Provenance.DETERMINISTIC_SIMULATION,
        budget=BudgetSpec(),
        accounting_ref=accounting.sha256,
    )
    runtime.verify_evaluation_observation = lambda *_args: True
    runtime.diagnostics.executor = lambda task, config, _bundle: replace(
        seed_observation,
        task_id=task.task_id,
        environment_id=task.environment_ref.id,
        arm=config.arm,
        seed=config.seed,
        bundle_hash=config.bundle_hash,
    )
    diagnostic = runtime.diagnostics.create("cand_accounting", active.content_hash)
    result = runtime.diagnostics.run(diagnostic["diagnosticId"])
    assert result["state"] == "completed"
    assert all(item["totalTokens"] == 25462 * 3 for item in result["armSummaries"])
