from types import SimpleNamespace

from adaptive_agent.appworld_benchmark import AppWorldBenchmarkRunner, AppWorldCellResult, AppWorldProtocol
from adaptive_agent.evaluation import Arm, ModelProvenance, Partition, RunObservation


class Catalog:
    def split_ids(self, split): return tuple(f"{split}-{index}" for index in range(12))
    def dataset_hash(self): return "dataset-hash"
    def task(self, task_id, split, allow_test=False): return SimpleNamespace(task_id=task_id, split=split, instruction="do it")


class Package:
    catalog = Catalog()
    def provider_factory(self, task, run_id, seed=0): raise AssertionError("runtime owns provider construction")
    def evaluate_provider(self, provider): raise AssertionError("runtime owns evaluation")


class Runtime:
    def __init__(self): self.calls, self.results = [], {}
    def run_appworld_cell(self, *, package, task, arm, seed, bundle_hash, budget, run_id):
        self.calls.append(run_id)
        result = AppWorldCellResult(RunObservation(task.task_id, "appworld", Partition.FINAL, seed, arm, arm is not Arm.B0, True, 0, 3, 1.0, model_provenance=ModelProvenance.REAL_MODEL, response_id=run_id, run_id=run_id, bundle_hash=bundle_hash), {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5})
        self.results[run_id] = result
        return result
    def recover_appworld_cell(self, run_id): return self.results.get(run_id)
    def verify_appworld_cell(self, result, **kwargs):
        row = result.observation
        return row.task_id == kwargs["task"].task_id and row.arm == kwargs["arm"] and row.seed == kwargs["seed"] and row.bundle_hash == kwargs["bundle_hash"] and row.model_provenance is ModelProvenance.REAL_MODEL


def test_freeze_selects_official_final_subset_without_inspecting_answers():
    protocol = AppWorldProtocol.freeze(Package(), model_profile="model", core_planner_hash="core")
    assert len(protocol.sampled_task_ids) == 20
    assert set(protocol.split_by_task_id) <= {(task, split) for split in ("test_normal", "test_challenge") for task in Package().catalog.split_ids(split)}


def test_runner_is_durable_and_reports_measured_usage(tmp_path):
    protocol = AppWorldProtocol.freeze(Package(), model_profile="model", core_planner_hash="core", published_count=2)
    runtime = Runtime()
    runner = AppWorldBenchmarkRunner(tmp_path, Package(), protocol, runtime)
    report = runner.run("job", {Arm.B0: "b0", Arm.L: "l", Arm.A: "a"})
    assert len(runtime.calls) == 6
    assert report.paired_task_count == 2
    assert report.arm_summaries["B0"]["inputTokens"] == 4
    runner.run("job", {Arm.B0: "b0", Arm.L: "l", Arm.A: "a"})
    assert len(runtime.calls) == 6
