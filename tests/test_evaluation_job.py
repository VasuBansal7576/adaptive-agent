import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from adaptive_agent.benchmark import BenchmarkSummary
from adaptive_agent.evaluation import Arm, EvaluationError, EvaluationProtocol, Partition, build_environment_packages
from adaptive_agent.evaluation_job import EvaluationJob
from adaptive_agent.store import Store


class _Bundle:
    def __init__(self, value):
        self.content_hash = value


def _bundles():
    return {arm: _Bundle(arm.value) for arm in (Arm.B0, Arm.L, Arm.A)}


class EvaluationJobTests(unittest.TestCase):
    def _job(self, directory, bundles=None):
        packages = build_environment_packages()
        protocol = EvaluationProtocol()
        protocol.freeze(packages)
        controller = Mock()
        controller.execute_probe.return_value = {"passed": True, "outputs": [{"ok": True}], "provenance": ["controller_toolbroker"], "obligations": ["trusted probe"]}
        return EvaluationJob(Store(Path(directory)), controller, protocol, packages, bundles or _bundles(), Mock())

    def test_missing_bundle_preflight_does_not_construct_or_run_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self._job(directory, {Arm.B0: _Bundle("b0")})
            with self.assertRaisesRegex(EvaluationError, "missing expected arm bundles"):
                job.run("job-missing", "validation", base_hash="base", candidate_hash="candidate")
            self.assertIsNone(job.readback("job-missing"))

    def test_incomplete_driver_result_is_persisted_without_second_model_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self._job(directory)
            summary = Mock(spec=BenchmarkSummary)
            summary.complete = False
            summary.statuses = ()
            with patch("adaptive_agent.evaluation_job.ResumableEvaluationDriver") as driver_type:
                driver_type.return_value.run.return_value = summary
                result = job.run("job-incomplete", "validation", base_hash="base", candidate_hash="candidate")
            self.assertEqual(result.status, "incomplete")
            stored = job.readback("job-incomplete")
            self.assertIsNotNone(stored)
            self.assertEqual(stored.status, "incomplete")
            driver_type.return_value.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
