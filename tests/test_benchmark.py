import tempfile
import unittest
from pathlib import Path

from adaptive_agent.benchmark import ResumableEvaluationDriver
from adaptive_agent.evaluation import EvaluationProtocol, ModelProvenance, Partition, RunObservation, build_environment_packages
from adaptive_agent.store import Store


class BenchmarkDriverTests(unittest.TestCase):
    def test_failed_runtime_is_persisted_and_panel_resumes_without_reselection(self):
        packages = build_environment_packages()
        protocol = EvaluationProtocol()
        protocol.freeze(packages)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            calls = []
            def execute(task, frozen_config, bundle):
                calls.append((task.task_id, frozen_config.arm, frozen_config.seed))
                if task.partition is Partition.DEVELOPMENT:
                    return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL)
                raise RuntimeError("runtime unavailable")
            class TrustedSmokeEvidence:
                durable = True
                def verify(self, observation, frozen, package):
                    return True
            driver = ResumableEvaluationDriver(store, protocol, packages, execute, object(), evidence_store=TrustedSmokeEvidence())
            driver.run("bench-1", Partition.DEVELOPMENT)
            calls.clear()
            first = driver.run("bench-1", Partition.VALIDATION, base_hash="base", candidate_hash="candidate")
            self.assertTrue(first.failed)
            first_panel = {task_id for task_id, _, _ in calls}
            calls.clear()
            resumed = ResumableEvaluationDriver(store, protocol, packages, execute, object(), evidence_store=TrustedSmokeEvidence())
            second = resumed.run("bench-1", Partition.VALIDATION, base_hash="base", candidate_hash="candidate")
            self.assertTrue(second.failed)
            self.assertEqual(first_panel, {task_id for task_id, _, _ in calls})

    def test_synthetic_executor_result_is_failed_not_success(self):
        packages = build_environment_packages()
        protocol = EvaluationProtocol()
        protocol.freeze(packages)
        with tempfile.TemporaryDirectory() as directory:
            def execute(task, frozen_config, bundle):
                return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.SYNTHETIC_MODEL)
            driver = ResumableEvaluationDriver(Store(Path(directory)), protocol, packages, execute, object())
            result = driver.run("bench-2", Partition.DEVELOPMENT)
            self.assertTrue(result.failed)
            self.assertFalse(result.complete)


if __name__ == "__main__":
    unittest.main()
