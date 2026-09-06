import tempfile
import threading
import time
import unittest
from pathlib import Path

from adaptive_agent.benchmark import ResumableEvaluationDriver
from adaptive_agent.evaluation import Arm, EvaluationError, EvaluationProtocol, ModelProvenance, Partition, RunObservation, build_environment_packages
from adaptive_agent.evaluation_store import SQLiteAllocationStore
from adaptive_agent.models import SkillBundle
from adaptive_agent.store import Store


class _Bundle:
    def __init__(self, content_hash):
        self.content_hash = content_hash


def _arm_bundles():
    return {arm: _Bundle(f"bundle-{arm.value}") for arm in (Arm.B0, Arm.L, Arm.A)}


class BenchmarkDriverTests(unittest.TestCase):
    def test_missing_arm_bundle_fails_before_executor_call(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            calls = []
            with tempfile.TemporaryDirectory() as directory:
                def execute(task, frozen_config, bundle):
                    calls.append((task.task_id, frozen_config.arm))
                    return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash=frozen_config.bundle_hash)
                class Evidence:
                    durable = True
                    def verify(self, observation, frozen, package):
                        return True
                driver = ResumableEvaluationDriver(Store(Path(directory)), protocol, packages, execute, object(), evidence_store=Evidence(), arm_bundles={Arm.B0: _Bundle("bundle-B0")})
                driver.run_development_smoke("missing-bundles")
                with self.assertRaisesRegex(EvaluationError, "missing expected arm bundles"):
                    driver.run("missing-bundles", Partition.VALIDATION, base_hash="b", candidate_hash="c")
            self.assertEqual(len(calls), 1)

    def test_observation_cannot_relabel_selected_arm_bundle(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                def execute(task, frozen_config, bundle):
                    return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash="wrong")
                driver = ResumableEvaluationDriver(Store(Path(directory)), protocol, packages, execute, object(), arm_bundles=_arm_bundles())
                result = driver.run_development_smoke("wrong-bundle")
                self.assertTrue(result.failed)
                self.assertIn("bundle hash", result.statuses[0].error)

    def test_development_smoke_is_one_trusted_run(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                calls = []
                def execute(task, frozen_config, bundle):
                    calls.append((task.task_id, frozen_config.seed, frozen_config.arm))
                    return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash=frozen_config.bundle_hash)
                class TrustedSmokeEvidence:
                    durable = True
                    def verify(self, observation, frozen, package):
                        return True
                result = ResumableEvaluationDriver(Store(Path(directory)), protocol, packages, execute, object(), evidence_store=TrustedSmokeEvidence(), arm_bundles=_arm_bundles()).run_development_smoke("smoke")
                self.assertTrue(result.complete)
                self.assertEqual(result.expected_count, 1)
                self.assertEqual(len(calls), 1)

    def test_parallel_cells_are_bounded_by_frozen_concurrency(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol(concurrency_limit=4)
            protocol.freeze(packages)
            active = peak = 0
            guard = threading.Lock()

            class TrustedEvidence:
                durable = True

                def verify(self, observation, frozen, package):
                    return True

            def execute(task, frozen_config, bundle):
                nonlocal active, peak
                with guard:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(0.002)
                    return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash=frozen_config.bundle_hash)
                finally:
                    with guard:
                        active -= 1

            with tempfile.TemporaryDirectory() as directory:
                result = ResumableEvaluationDriver(Store(Path(directory)), protocol, packages, execute, object(), evidence_store=TrustedEvidence(), arm_bundles=_arm_bundles()).run("parallel", Partition.DEVELOPMENT)
            self.assertTrue(result.complete)
            self.assertEqual(result.expected_count, 60)
            self.assertLessEqual(peak, 4)

    def test_smoke_identity_authorizes_held_out_run_after_reopen(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                class Evidence:
                    durable = True
                    def verify(self, observation, frozen, package):
                        return True
                def execute(task, frozen_config, bundle):
                    if task.partition is Partition.DEVELOPMENT:
                        return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash=frozen_config.bundle_hash)
                    raise RuntimeError("held-out test executor")
                first = ResumableEvaluationDriver(Store(path), protocol, packages, execute, object(), evidence_store=Evidence(), arm_bundles=_arm_bundles())
                self.assertTrue(first.run_development_smoke("root-gate").complete)
                reopened = ResumableEvaluationDriver(Store(path), protocol, packages, execute, object(), evidence_store=Evidence(), arm_bundles=_arm_bundles())
                self.assertTrue(reopened._development_smoke_complete("root-gate"))
                self.assertTrue(reopened.run("root-gate", Partition.VALIDATION, base_hash="b", candidate_hash="c").failed)

    def test_live_owner_cannot_be_stolen_by_another_driver(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                store = Store(Path(directory))
                driver_a = ResumableEvaluationDriver(store, protocol, packages, lambda *_: None, object(), arm_bundles=_arm_bundles())
                driver_b = ResumableEvaluationDriver(store, protocol, packages, lambda *_: None, object(), arm_bundles=_arm_bundles())
                task = packages["finance"].tasks_for_partition(Partition.DEVELOPMENT)[0]
                self.assertTrue(driver_a._claim("live-owner", task, Arm.B0, 0))
                self.assertFalse(driver_b._claim("live-owner", task, Arm.B0, 0))

    def test_allocation_gap_recovers_original_panel_and_persists_plan(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                store = Store(Path(directory))
                class TrustedSmokeEvidence:
                    durable = True
                    def verify(self, observation, frozen, package):
                        return True
                def execute(task, frozen_config, bundle):
                    if task.partition is Partition.DEVELOPMENT:
                        return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash=frozen_config.bundle_hash)
                    raise RuntimeError("validation unavailable")
                smoke = ResumableEvaluationDriver(store, protocol, packages, execute, object(), evidence_store=TrustedSmokeEvidence(), arm_bundles=_arm_bundles())
                smoke.run("gap", Partition.DEVELOPMENT)
                base = SQLiteAllocationStore(store)
                class CrashAfterAllocation:
                    def __init__(self):
                        self.crashed = False
                    def reserve_next(self, *args):
                        index = base.reserve_next(*args)
                        if not self.crashed:
                            self.crashed = True
                            raise KeyboardInterrupt("crash after allocation")
                        return index
                    def get(self, allocation_id):
                        return base.get(allocation_id)
                crashing = ResumableEvaluationDriver(store, protocol, packages, execute, object(), allocation_store=CrashAfterAllocation(), evidence_store=TrustedSmokeEvidence(), arm_bundles=_arm_bundles())
                with self.assertRaises(KeyboardInterrupt):
                    crashing.run("gap", Partition.VALIDATION, base_hash="base", candidate_hash="candidate")
                resumed = ResumableEvaluationDriver(store, protocol, packages, execute, object(), evidence_store=TrustedSmokeEvidence(), arm_bundles=_arm_bundles())
                result = resumed.run("gap", Partition.VALIDATION, base_hash="base", candidate_hash="candidate")
                self.assertTrue(result.failed)
                with store.connect() as conn:
                    plan = conn.execute("SELECT panel_json FROM benchmark_plans WHERE benchmark_id = 'gap' AND partition = 'validation'").fetchone()
                self.assertIsNotNone(plan)
                self.assertEqual(len(__import__("json").loads(plan["panel_json"])), 60)

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
                        return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash=frozen_config.bundle_hash)
                    raise RuntimeError("runtime unavailable")
                class TrustedSmokeEvidence:
                    durable = True
                    def verify(self, observation, frozen, package):
                        return True
                driver = ResumableEvaluationDriver(store, protocol, packages, execute, object(), evidence_store=TrustedSmokeEvidence(), arm_bundles=_arm_bundles())
                driver.run("bench-1", Partition.DEVELOPMENT)
                calls.clear()
                first = driver.run("bench-1", Partition.VALIDATION, base_hash="base", candidate_hash="candidate")
                self.assertTrue(first.failed)
                first_panel = {task_id for task_id, _, _ in calls}
                calls.clear()
                resumed = ResumableEvaluationDriver(store, protocol, packages, execute, object(), evidence_store=TrustedSmokeEvidence(), arm_bundles=_arm_bundles())
                second = resumed.run("bench-1", Partition.VALIDATION, base_hash="base", candidate_hash="candidate")
                self.assertTrue(second.failed)
                self.assertEqual(first_panel, {task_id for task_id, _, _ in calls})

    def test_synthetic_executor_result_is_failed_not_success(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                def execute(task, frozen_config, bundle):
                    return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.SYNTHETIC_MODEL, bundle_hash=frozen_config.bundle_hash)
                driver = ResumableEvaluationDriver(Store(Path(directory)), protocol, packages, execute, object(), arm_bundles=_arm_bundles())
                result = driver.run("bench-2", Partition.DEVELOPMENT)
                self.assertTrue(result.failed)
                self.assertFalse(result.complete)

    def test_crash_after_first_task_keeps_full_immutable_panel_for_resume(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                store = Store(Path(directory))
                calls = []
                def crash(task, frozen_config, bundle):
                    calls.append(task.task_id)
                    if task.partition is Partition.DEVELOPMENT:
                        return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash=frozen_config.bundle_hash)
                    raise KeyboardInterrupt("simulated process crash")
                class TrustedSmokeEvidence:
                    durable = True
                    def verify(self, observation, frozen, package):
                        return True
                driver = ResumableEvaluationDriver(store, protocol, packages, crash, object(), evidence_store=TrustedSmokeEvidence(), arm_bundles=_arm_bundles())
                driver.run("bench-crash", Partition.DEVELOPMENT)
                calls.clear()
                with self.assertRaises(KeyboardInterrupt):
                    driver.run("bench-crash", Partition.VALIDATION, base_hash="base", candidate_hash="candidate")
                with store._connect() as conn:
                    plan = conn.execute("SELECT panel_json FROM benchmark_plans WHERE benchmark_id = ? AND partition = 'validation'", ("bench-crash",)).fetchone()
                self.assertEqual(len(__import__("json").loads(plan["panel_json"])), 60)
                self.assertEqual(len(calls), 1)

    def test_crash_after_first_validation_cell_resumes_all_360_cells(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                store = Store(Path(directory))
                bundles = _arm_bundles()
                calls = []
                crashed = [False]
                class Evidence:
                    durable = True
                    def verify(self, observation, frozen, package):
                        return True
                def execute(task, frozen_config, bundle):
                    calls.append((task.task_id, frozen_config.arm, frozen_config.seed))
                    if task.partition is Partition.DEVELOPMENT:
                        return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash=frozen_config.bundle_hash)
                    if not crashed[0]:
                        crashed[0] = True
                        raise KeyboardInterrupt("crash after first validation cell")
                    return RunObservation(task.task_id, task.environment_ref.id, Partition.VALIDATION, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash=frozen_config.bundle_hash)
                first = ResumableEvaluationDriver(store, protocol, packages, execute, object(), evidence_store=Evidence(), arm_bundles=bundles)
                first.run_development_smoke("resume-360")
                with self.assertRaises(KeyboardInterrupt):
                    first.run("resume-360", Partition.VALIDATION, base_hash="b", candidate_hash="c")
                calls.clear()
                reopened = ResumableEvaluationDriver(Store(Path(directory)), protocol, packages, execute, object(), evidence_store=Evidence(), arm_bundles=bundles)
                result = reopened.run("resume-360", Partition.VALIDATION, base_hash="b", candidate_hash="c")
                self.assertTrue(result.complete)
                self.assertEqual(result.expected_count, 360)
                self.assertEqual(len(calls), 360)
                self.assertEqual(len(set(calls)), 360)
                with store.connect() as conn:
                    receipt_count = conn.execute("SELECT COUNT(*) AS count FROM task_runs WHERE benchmark_id = 'resume-360' AND partition = 'validation' AND status = 'complete'").fetchone()["count"]
                    attempt_count = conn.execute("SELECT COUNT(*) AS count FROM benchmark_task_attempts WHERE benchmark_id = 'resume-360' AND partition = 'validation'").fetchone()["count"]
                self.assertEqual(receipt_count, 360)
            self.assertEqual(attempt_count, 361)

    def test_benchmark_id_is_fenced_to_frozen_inputs_and_bundle(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                store = Store(Path(directory))
                def execute(task, frozen_config, bundle):
                    return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.SYNTHETIC_MODEL, bundle_hash=frozen_config.bundle_hash)
                first = ResumableEvaluationDriver(store, protocol, packages, execute, {"bundle": "one"}, arm_bundles=_arm_bundles())
                first.run("fenced", Partition.DEVELOPMENT)
                changed = ResumableEvaluationDriver(store, protocol, packages, execute, {"bundle": "two"}, arm_bundles=_arm_bundles())
                with self.assertRaisesRegex(EvaluationError, "different frozen inputs"):
                    changed.run("fenced", Partition.DEVELOPMENT)

    def test_benchmark_cells_use_dedicated_runs_and_resume_without_generic_collision(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                store = Store(Path(directory))
                bundle = SkillBundle(skills=[])
                calls = []

                class TrustedEvidence:
                    durable = True

                    def verify(self, observation, frozen, package):
                        return True

                def execute(task, frozen_config, selected_bundle):
                    calls.append(task.task_id)
                    return RunObservation(
                        task.task_id,
                        task.environment_ref.id,
                        Partition.DEVELOPMENT,
                        frozen_config.seed,
                        frozen_config.arm,
                        True,
                        True,
                        0,
                        1,
                        1.0,
                        model_provenance=ModelProvenance.REAL_MODEL,
                        bundle_hash=selected_bundle.content_hash,
                    )

                first = ResumableEvaluationDriver(store, protocol, packages, execute, bundle, evidence_store=TrustedEvidence(), owner_id="owner-a")
                self.assertTrue(first.run_development_smoke("dedicated").complete)
                self.assertEqual(len(calls), 1)
                with store.connect() as conn:
                    benchmark_rows = conn.execute("SELECT benchmark_id, arm, seed, owner_id, status FROM task_runs WHERE benchmark_id = 'dedicated:smoke'").fetchall()
                    generic_rows = conn.execute("SELECT * FROM task_runs WHERE benchmark_id IS NULL").fetchall()
                self.assertEqual(len(benchmark_rows), 1)
                self.assertEqual(dict(benchmark_rows[0])["owner_id"], "owner-a")
                self.assertEqual(dict(benchmark_rows[0])["status"], "complete")
                self.assertEqual(generic_rows, [])

                # A restart must reuse the durable owner recorded on the cell.
                resumed = ResumableEvaluationDriver(store, protocol, packages, execute, bundle, evidence_store=TrustedEvidence(), owner_id="owner-a")
                self.assertTrue(resumed.run_development_smoke("dedicated").complete)
                self.assertEqual(len(calls), 1)

    def test_retry_uses_fresh_attempt_identity_and_preserves_failed_attempt(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                store = Store(Path(directory))
                bundle = SkillBundle(skills=[])
                attempts = []

                class TrustedEvidence:
                    durable = True

                    def verify(self, observation, frozen, package):
                        return True

                def execute(task, frozen_config, selected_bundle):
                    attempts.append(frozen_config.attempt)
                    if len(attempts) == 1:
                        raise RuntimeError("provider failed before receipt")
                    return RunObservation(
                        task.task_id,
                        task.environment_ref.id,
                        Partition.DEVELOPMENT,
                        frozen_config.seed,
                        frozen_config.arm,
                        True,
                        True,
                        0,
                        1,
                        1.0,
                        model_provenance=ModelProvenance.REAL_MODEL,
                        bundle_hash=selected_bundle.content_hash,
                    )

                driver = ResumableEvaluationDriver(
                    store,
                    protocol,
                    packages,
                    execute,
                    bundle,
                    evidence_store=TrustedEvidence(),
                    owner_id="retry-owner",
                )
                first = driver.run_development_smoke("retry-identity")
                self.assertFalse(first.complete)
                second = driver.run_development_smoke("retry-identity")
                self.assertTrue(second.complete)
                self.assertEqual(attempts, [0, 1])
                with store.connect() as conn:
                    rows = conn.execute(
                        "SELECT status FROM benchmark_task_attempts WHERE benchmark_id = ? ORDER BY rowid",
                        ("retry-identity:smoke",),
                    ).fetchall()
                self.assertEqual([row["status"] for row in rows], ["failed", "complete"])

    def test_interrupted_attempt_is_uncertain_and_same_owner_retry_advances_attempt(self):
            packages = build_environment_packages()
            protocol = EvaluationProtocol()
            protocol.freeze(packages)
            with tempfile.TemporaryDirectory() as directory:
                store = Store(Path(directory))
                bundle = SkillBundle(skills=[])
                attempts = []

                class TrustedEvidence:
                    durable = True

                    def verify(self, observation, frozen, package):
                        return True

                def execute(task, frozen_config, selected_bundle):
                    attempts.append(frozen_config.attempt)
                    if len(attempts) == 1:
                        raise KeyboardInterrupt("worker interrupted")
                    return RunObservation(task.task_id, task.environment_ref.id, Partition.DEVELOPMENT, frozen_config.seed, frozen_config.arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, bundle_hash=selected_bundle.content_hash)

                driver = ResumableEvaluationDriver(store, protocol, packages, execute, bundle, evidence_store=TrustedEvidence(), owner_id="interrupt-owner")
                with self.assertRaises(KeyboardInterrupt):
                    driver.run_development_smoke("interrupt")
                self.assertTrue(driver.run_development_smoke("interrupt").complete)
                self.assertEqual(attempts, [0, 1])
                with store.connect() as conn:
                    rows = conn.execute("SELECT status FROM benchmark_task_attempts WHERE benchmark_id = 'interrupt:smoke' ORDER BY rowid").fetchall()
                self.assertEqual([row["status"] for row in rows], ["uncertain", "complete"])


if __name__ == "__main__":
    unittest.main()
