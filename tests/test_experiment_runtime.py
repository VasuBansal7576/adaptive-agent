from __future__ import annotations

from types import SimpleNamespace

import pytest

from adaptive_agent.experiment_runtime import (
    DefaultExperimentStageRunner,
    ExperimentRuntimeError,
)


class Bundle:
    def __init__(self, content_hash: str):
        self.content_hash = content_hash


class Store:
    def __init__(self):
        self.artifacts = {
            "acct-1": {
                "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
                "toolCalls": 1,
                "durationSeconds": 0.25,
                "costMicrounits": 7,
            }
        }

    def get_bundle_by_hash(self, content_hash):
        return {"bundle_json": "{}", "content_hash": content_hash}

    def get_artifact(self, ref):
        return self.artifacts[ref]

    def get_evidence(self, evidence_id):
        if evidence_id == "model-1":
            return {"event_type": "model_response"}
        if evidence_id == "outcome-1":
            return {"event_type": "trusted_outcome"}
        return None

    def get_run(self, run_id):
        return {"status": "succeeded"} if run_id == "dev-run" else None

    def get_outcome_by_run_id(self, run_id):
        return {"passed": 1} if run_id == "dev-run" else None


class Package:
    def __init__(self, environment_id):
        self.environment_id = environment_id
        self._tasks = {
            "development": [self._task("development", 0)],
            "validation": [self._task("validation", 0)],
            "final": [self._task("final", 0)],
        }

    def _task(self, partition, index):
        return SimpleNamespace(
            task_id=f"{self.environment_id}-{partition}-{index}",
            environment_ref=SimpleNamespace(id=self.environment_id),
            goal=f"{partition} goal",
        )

    def tasks_for_partition(self, partition):
        return tuple(self._tasks[partition])


class Controller:
    def __init__(self):
        self.store = Store()

    def get_active_bundle(self):
        return Bundle("base")


class Runtime:
    core_planner_hash = "core"
    image_digest = "image"

    def __init__(self):
        self.controller = Controller()
        self.packages = {name: Package(name) for name in ("known-a", "known-b", "known-c", "sealed")}
        self.calls = []

    def execute_evaluation_task(self, task, config, bundle):
        self.calls.append((task.task_id, config.arm, config.seed, bundle.content_hash))
        return SimpleNamespace(
            run_id=f"run-{len(self.calls)}",
            evidence_ref="model-1",
            outcome_ref="outcome-1",
            accounting_ref="acct-1",
            response_id=f"response-{len(self.calls)}",
            model_provenance="real_model",
        )


class Protocol:
    known_environments = ("known-a", "known-b", "known-c")
    sealed_environment = "sealed"
    tasks_per_environment = 1
    seeds = (17, 23, 29)
    safety_case_ids = ("EVAL-004",)

    def start_candidate_generation(self):
        return SimpleNamespace(
            protocol_hash="protocol",
            inputs={
                "provider": "openai-codex",
                "modelProfile": "openai-codex/gpt-5.6-luna",
                "corePlannerHash": "core",
                "imageDigest": "image",
                "runBudget": {"modelTokens": 20},
            },
        )


def test_default_runner_executes_real_receipt_bound_training_cell():
    runtime = Runtime()
    runner = DefaultExperimentStageRunner(runtime, Protocol())

    result = runner(cell_key="known-a-development-0", context={"stage": "training", "attempt": 0})

    assert result["status"] == "complete"
    assert result["runIds"] == ["run-1"]
    assert result["usage"] == {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}
    assert runtime.calls == [("known-a-development-0", "B0", 17, "base")]


def test_panel_mapping_keeps_frozen_validation_and_final_counts_without_duplication():
    runtime = Runtime()
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    runner._candidate = lambda _context: Bundle("candidate")
    context = {"stage": "validation", "attempt": 0, "results": {"learning": {"candidate-generation": {"candidateBundleHash": "candidate"}}}}

    first = runner(cell_key="validation:0", context=context)
    second = runner(cell_key="validation:1", context=context)
    assert first["arm"] == "B0" and second["arm"] == "L"
    assert first["environmentId"] == second["environmentId"] == "known-a"
    assert first["seed"] == second["seed"] == 17
    with pytest.raises(ExperimentRuntimeError):
        runner(cell_key="validation:360", context=context)

    final_context = {**context, "stage": "final", "ablationBundleHash": "base"}
    sealed = runner(cell_key="final:27", context=final_context)
    assert sealed["environmentId"] == "sealed"
    assert sealed["sealed"] is True
    with pytest.raises(ExperimentRuntimeError):
        runner(cell_key="final:720", context=final_context)


def test_sealed_environment_is_rejected_before_final():
    runtime = Runtime()
    runner = DefaultExperimentStageRunner(runtime, Protocol())

    with pytest.raises(ExperimentRuntimeError, match="sealed environment"):
        runner(cell_key="leave-out:sealed", context={"stage": "transfer", "results": {}})
