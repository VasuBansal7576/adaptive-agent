from types import SimpleNamespace

from adaptive_agent.appworld_benchmark import AppWorldBenchmarkRunner, AppWorldProtocol
from adaptive_agent.evaluation import Arm, ModelProvenance, Partition, RunObservation


class Package:
    def dataset_content_hash(self):
        return "dataset-hash"

    def task_ids(self, split):
        return [f"{split}-{index}" for index in range(12)]

    def task(self, task_id):
        return SimpleNamespace(task_id=task_id)


def test_freeze_selects_official_final_subset_without_inspecting_answers(tmp_path):
    protocol = AppWorldProtocol.freeze(Package(), model_profile="model", core_planner_hash="core")
    assert len(protocol.sampled_task_ids) == 20
    assert protocol.seeds == (0,)
    assert set(protocol.split_by_task_id) <= {(task, split) for split in ("test_normal", "test_challenge") for task in Package().task_ids(split)}
    assert protocol.to_dict()["scope"] == "published_subset"


def test_runner_binds_cells_and_does_not_redispatch_completed_cells(tmp_path):
    protocol = AppWorldProtocol.freeze(Package(), model_profile="model", core_planner_hash="core", published_count=2)
    calls = []

    def execute(task, arm, seed, bundle_hash, budget):
        calls.append((task.task_id, arm, seed, bundle_hash, budget.model_tokens, budget.tool_calls))
        return RunObservation(task.task_id, "appworld", Partition.FINAL, seed, arm, True, True, 0, 3, 1.0, model_provenance=ModelProvenance.REAL_MODEL, response_id="response", run_id=f"run-{len(calls)}")

    runner = AppWorldBenchmarkRunner(tmp_path, Package(), protocol, execute, lambda observation, task, bundle: True)
    report = runner.run("job", {Arm.B0: "b0", Arm.L: "l", Arm.A: "a"})
    assert len(calls) == 6
    assert report.paired_task_count == 2
    assert report.to_dict()["missingPairs"] == 0
    runner.run("job", {Arm.B0: "b0", Arm.L: "l", Arm.A: "a"})
    assert len(calls) == 6
