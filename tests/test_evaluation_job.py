import tempfile
import unittest
import json
from types import SimpleNamespace
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

    def test_auxiliary_evidence_reports_distinct_query_support_and_learning_usage(self):
        job = object.__new__(EvaluationJob)
        job.store = SimpleNamespace(get_artifact=lambda ref: {"usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5}} if ref == "query-acct" else {"usage": {"inputTokens": 7, "outputTokens": 11, "totalTokens": 18}})
        query = SimpleNamespace(run_id="query-run", task_id="query-task", accounting_ref="query-acct", cost_microunits=13, latency_seconds=1.0, passed=True, reliable=True, safety_violations=0)
        support = SimpleNamespace(run_id="support-run", task_id="support-task", accounting_ref="support-acct", cost_microunits=17, latency_seconds=2.0, passed=True, reliable=True, safety_violations=0)
        stages = (SimpleNamespace(name="transfer", cells=("leave-out:finance",)), SimpleNamespace(name="adaptation", cells=()))
        state = {"results": {"transfer": {"leave-out:finance": {"queryRunIds": ["query-run"], "supportRunIds": ["support-run"], "trainingSourceRunIds": ["learning-source"], "queryTaskIds": [], "learningReceipt": {"usage": {"inputTokens": 19, "outputTokens": 23, "totalTokens": 42,}, "costMicrounits": 29}}}}}
        evidence = job._auxiliary_evidence(stages, state, lambda *_args, **_kwargs: (query, support))
        self.assertEqual(evidence["exposure"]["transfer"]["taskIds"], ["query-task", "support-task"])
        self.assertEqual(evidence["exposure"]["transfer"]["sourceRunIds"], ["learning-source", "support-run"])
        self.assertEqual(evidence["overhead"]["transfer"]["query"]["totalTokens"], 5)
        self.assertEqual(evidence["overhead"]["transfer"]["support"]["totalTokens"], 18)
        self.assertEqual(evidence["overhead"]["transfer"]["learning"]["totalTokens"], 42)
        self.assertEqual(evidence["overhead"]["transfer"]["support"]["costMicrounits"], 17)
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
