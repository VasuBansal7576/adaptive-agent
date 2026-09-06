import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from adaptive_agent.benchmark import BenchmarkSummary
from adaptive_agent.evaluation import Arm, EvaluationError, EvaluationProtocol, EvaluationReport, Partition, build_environment_packages
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

    def test_lifecycle_reports_are_published_by_phase_idempotently(self):
        with tempfile.TemporaryDirectory() as directory:
            job = self._job(directory)

            def report(phase):
                return EvaluationReport(
                    comparison=phase,
                    validity_status="valid",
                    candidate_hash="candidate-hash",
                    base_hash="base-hash",
                    protocol_hash="protocol-hash",
                    partition_hashes=(
                        {
                            "customer_support:validation": "customer-support-validation-partition",
                            "finance:validation": "finance-validation-partition",
                            "it:validation": "it-validation-partition",
                        }
                        if phase == "validation"
                        else {
                            "customer_support:final": "customer-support-final-partition",
                            "finance:final": "finance-final-partition",
                            "it:final": "it-final-partition",
                            "lab_scheduling:final": "lab-scheduling-final-partition",
                        }
                    ),
                    arm_summaries={},
                    confidence_intervals=(),
                    safety_passed=True,
                    missing_pairs=0,
                    partition_leak=False,
                    invalid_fixture_resets=0,
                    infrastructure_failures=(),
                    exposure=(),
                    workload=job.protocol.workload(1),
                    analysis_seed=job.protocol.analysis_seed,
                )

            reports = {"validation": report("validation"), "final": report("final")}
            job._save("lifecycle", "experiment", "complete", reports=reports)
            job._save("lifecycle", "experiment", "complete", reports=reports)

            rows = job.store.list_evaluations()
            self.assertEqual({row["report_id"] for row in rows}, {"lifecycle:validation", "lifecycle:final"})
            self.assertEqual(len(rows), 2)
            self.assertEqual({json.loads(row["partition_ref"])["id"] for row in rows}, {"validation", "final"})
            self.assertEqual(
                {json.loads(row["partition_ref"])["sha256"] for row in rows},
                {"finance-validation-partition", "finance-final-partition"},
            )


if __name__ == "__main__":
    unittest.main()
