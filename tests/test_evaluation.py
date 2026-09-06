import dataclasses
import unittest

from adaptive_agent.evaluation import (
    AblationInput,
    Arm,
    ArtifactRef,
    BudgetSpec,
    EnvironmentManifest,
    EvaluationError,
    EvaluationProtocol,
    EvaluationRunner,
    ModelProvenance,
    Partition,
    PromotionEvidenceRefused,
    ProviderUnavailable,
    Provenance,
    RunObservation,
    TaskInput,
    audit_ablation,
    build_environment_packages,
    clustered_paired_bootstrap,
    TrustedEvaluatorRegistry,
)


def _observation(env, task, seed, arm, passed=True, *, partition=Partition.VALIDATION):
    return RunObservation(task.task_id, env, partition, seed, arm, passed, passed, 0, 100 if arm is Arm.B0 else 105, 1.0, core_planner_hash="same")


class EvaluationTests(unittest.TestCase):
  def test_required_argument_schemas_reject_privileged_fields_and_unknown_tools(self):
    packages = build_environment_packages()
    task = packages["finance"].learner_tasks()[0]
    payload = task.to_dict()
    self.assertEqual(TaskInput.from_mapping(payload), task)
    with self.assertRaisesRegex(EvaluationError, "privileged task fields"):
        TaskInput.from_mapping({**payload, "expectedAnswer": "paid"})
    manifest = packages["finance"].manifest.to_dict()
    self.assertEqual(EnvironmentManifest.from_mapping(manifest).environment_id, "finance")
    with self.assertRaisesRegex(EvaluationError, "privileged manifest fields"):
        EnvironmentManifest.from_mapping({**manifest, "actionSequence": ["finance.invoice.read"]})
    session = packages["finance"].reset(task.task_id, 11)
    with self.assertRaisesRegex(EvaluationError, "unknown tool arguments"):
        packages["finance"].invoke(session, "finance.invoice.read", {"invoice_id": "INV-DEV-000", "secret": "no"})

  def test_reset_is_deterministic_and_isolated_between_sessions(self):
    package = build_environment_packages()["finance"]
    task = package.tasks_for_partition(Partition.DEVELOPMENT)[0]
    first = package.reset(task.task_id, 7)
    second = package.reset(task.task_id, 7)
    package.invoke(first, "finance.invoice.apply_payment", {"invoice_id": "INV-DEV-000", "payment_id": "PAY-DEV-000", "expected_version": 1})
    self.assertNotEqual(first.state, second.state)
    self.assertEqual(second.state["invoice_status"], "open")
    self.assertEqual(package.reset(task.task_id, 7).state, package.reset(task.task_id, 7).state)

  def test_partition_retrieval_withholds_validation_and_final_from_learner(self):
    packages = build_environment_packages()
    for package in packages.values():
        learner_ids = {task.task_id for task in package.learner_tasks()}
        self.assertEqual(learner_ids & {task.task_id for task in package.tasks_for_partition(Partition.VALIDATION)}, set())
        self.assertEqual(learner_ids & {task.task_id for task in package.tasks_for_partition(Partition.FINAL)}, set())
        self.assertTrue(all("expected" not in path.lower() for path in package.learner_container_files()))
        self.assertTrue(all("answer" not in document.text.lower() for document in package.learner_documents()))
    sealed = packages["lab_scheduling"]
    self.assertTrue(sealed.manifest.sealed)
    self.assertFalse(sealed.learner_tasks())
    self.assertTrue(all("SLOT-" not in text for text in sealed.learner_container_files().values()))

  def test_task_families_and_entities_are_disjoint_across_splits(self):
    packages = build_environment_packages()
    for package in packages.values():
        family_sets = [set(package.task_families(partition)) for partition in Partition if package.tasks_for_partition(partition)]
        for left, right in zip(family_sets, family_sets[1:]):
            self.assertTrue(left.isdisjoint(right))
        partitions = [package.tasks_for_partition(partition) for partition in Partition if package.tasks_for_partition(partition)]
        for left, right in zip(partitions, partitions[1:]):
            self.assertTrue({ref for task in left for ref in task.allowed_input_refs}.isdisjoint({ref for task in right for ref in task.allowed_input_refs}))
        signatures = [package.structural_signature(partition) for partition in Partition if package.tasks_for_partition(partition)]
        self.assertEqual(len(signatures), len(set(signatures)))
    self.assertTrue(any("join" in family for family in packages["finance"].task_families(Partition.VALIDATION)))
    self.assertTrue(any("conditional" in family for family in packages["customer_support"].task_families(Partition.FINAL)))
    self.assertTrue(any("history" in family for family in packages["it"].task_families(Partition.VALIDATION)))

  def test_fixture_reads_return_authoritative_records_and_versions(self):
    packages = build_environment_packages()
    finance = packages["finance"]
    task = finance.learner_tasks()[0]
    session = finance.reset(task.task_id, 3)
    result = finance.invoke(session, "finance.invoice.read", {"invoice_id": "INV-DEV-000"})
    self.assertEqual(result.output["record"]["version"], 1)
    self.assertEqual(result.output["record"]["status"], "open")
    support = packages["customer_support"]
    support_task = support.learner_tasks()[0]
    support_result = support.invoke(support.reset(support_task.task_id, 3), "support.ticket.read", {"ticket_id": "TKT-DEV-000"})
    self.assertEqual(support_result.output["record"]["version"], 1)
    lab = packages["lab_scheduling"]
    lab_task = lab.tasks_for_partition(Partition.FINAL)[0]
    lab_session = lab.reset(lab_task.task_id, 3)
    sample = lab.invoke(lab_session, "lab.sample.lookup", {"sample_barcode": "SMP-FIN-000"})
    slot = lab.invoke(lab_session, "lab.slot.search", {"assay": sample.output["sample"]["assay"], "date": sample.output["sample"]["requiredDate"]})
    self.assertEqual(slot.output["slots"], ["SLOT-FIN-000"])

  def test_live_provider_and_deterministic_simulation_are_distinct_provenance(self):
    package = build_environment_packages()["finance"]
    task = package.learner_tasks()[0]
    simulated = package.invoke(package.reset(task.task_id, 1), "finance.invoice.read", {"invoice_id": "INV-DEV-000"})
    self.assertIs(simulated.provenance, Provenance.DETERMINISTIC_SIMULATION)
    self.assertEqual(simulated.provider_id, "deterministic-fixture")
    with self.assertRaises(ProviderUnavailable):
        package.invoke(package.reset(task.task_id, 1), "finance.invoice.read", {"invoice_id": "INV-DEV-000"}, provider=Provenance.LIVE_PROVIDER)
    class Provider:
        provider_id = "sandbox-provider"
        def call(self, tool, arguments):
            return {"ok": True, "providerTool": tool}
    live = package.invoke(package.reset(task.task_id, 1), "finance.invoice.read", {"invoice_id": "INV-DEV-000"}, provider=Provenance.LIVE_PROVIDER, live_provider=Provider())
    self.assertIs(live.provenance, Provenance.LIVE_PROVIDER)
    self.assertEqual(live.provider_id, "sandbox-provider")

  def test_heldout_objectives_require_composed_operations(self):
    finance = build_environment_packages()["finance"]
    task = finance.tasks_for_partition(Partition.VALIDATION)[0]
    session = finance.reset(task.task_id, 4)
    finance.invoke(session, "finance.invoice.apply_payment", {"invoice_id": "INV-VAL-000", "payment_id": "PAY-VAL-000", "expected_version": 1})
    self.assertFalse(finance.evaluate(task.task_id, session).passed)
    finance.invoke(session, "finance.invoice.apply_payment", {"invoice_id": "INV-VAL-000-B", "payment_id": "PAY-VAL-000-B", "expected_version": 1})
    self.assertTrue(finance.evaluate(task.task_id, session).passed)

    support = build_environment_packages()["customer_support"]
    task = support.tasks_for_partition(Partition.VALIDATION)[0]
    session = support.reset(task.task_id, 4)
    support.invoke(session, "support.ticket.set_status", {"ticket_id": "TKT-VAL-000", "status": "resolved", "expected_version": 1})
    self.assertFalse(support.evaluate(task.task_id, session).passed)
    support.invoke(session, "support.ticket.add_tag", {"ticket_id": "TKT-VAL-000", "tag": "specialist", "expected_version": 2})
    self.assertTrue(support.evaluate(task.task_id, session).passed)

  def test_safety_is_executable_and_evaluator_owned(self):
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    registry = TrustedEvaluatorRegistry()
    registry.register_safety_probe("EVAL-004", lambda: True)
    registry.register_safety_probe("EVAL-005", lambda: False)
    runner = EvaluationRunner(protocol, packages, registry)
    report = runner.report_from_observations(comparison="validation", base_hash="base", candidate_hash="candidate", observations=[])
    self.assertEqual(report.safety_case_results, {"EVAL-004": True, "EVAL-005": False})
    self.assertFalse(report.safety_passed)

  def test_protocol_freezes_hashes_before_candidate_generation_and_detects_drift(self):
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    frozen = protocol.freeze(packages)
    self.assertTrue(frozen.protocol_hash)
    self.assertEqual(len(frozen.fixture_hashes), 4)
    self.assertEqual(len(frozen.partition_hashes), 12)
    protocol.start_candidate_generation()
    with self.assertRaisesRegex(EvaluationError, "already frozen"):
        protocol.freeze(packages)
    changed = dict(packages)
    changed["finance"] = build_environment_packages()["finance"]
    changed["finance"].manifest = dataclasses.replace(changed["finance"].manifest, version="2")
    with self.assertRaisesRegex(PromotionEvidenceRefused, "fixture hash changed"):
        protocol.assert_integrity(changed)

  def test_clustered_bootstrap_is_seeded_and_clusters_all_seeds_per_task(self):
    packages = build_environment_packages()
    baseline, candidate = [], []
    for env in ("finance", "customer_support", "it"):
        tasks = packages[env].tasks_for_partition(Partition.VALIDATION)[:2]
        for task in tasks:
            for seed in (17, 23, 29):
                baseline.append(_observation(env, task, seed, Arm.B0, True))
                candidate.append(_observation(env, task, seed, Arm.L, task.task_id.endswith("00")))
    one = clustered_paired_bootstrap(baseline, candidate, analysis_seed=9)
    two = clustered_paired_bootstrap(baseline, candidate, analysis_seed=9)
    self.assertEqual(one, two)
    self.assertTrue(all(item.draws == 10_000 and item.analysis_seed == 9 for item in one))
    self.assertLess(next(item for item in one if item.metric == "accuracy").point, 0)


  def test_incomplete_report_cannot_be_promotion_evidence_and_hashes_are_checked(self):
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    frozen = protocol.freeze(packages)
    runner = EvaluationRunner(protocol, packages, safety_cases={"EVAL-004": True, "EVAL-005": True})
    task = packages["finance"].tasks_for_partition(Partition.VALIDATION)[0]
    rows = [_observation("finance", task, 17, Arm.B0), _observation("finance", task, 17, Arm.L)]
    report = runner.report_from_observations(comparison="validation", base_hash="base", candidate_hash="candidate", observations=rows)
    self.assertEqual(report.validity_status, "incomplete")
    self.assertFalse(report.promotion_eligible)
    with self.assertRaises(PromotionEvidenceRefused):
        report.require_promotion_evidence(protocol, packages)
    invalid = dataclasses.replace(report, protocol_hash="wrong")
    with self.assertRaises(PromotionEvidenceRefused):
        invalid.require_promotion_evidence(protocol, packages)
    self.assertTrue(frozen.protocol_hash)

  def test_invalid_duplicate_or_leaked_pair_is_not_a_valid_report(self):
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    runner = EvaluationRunner(protocol, packages, safety_cases={"EVAL-004": True, "EVAL-005": True})
    task = packages["finance"].tasks_for_partition(Partition.VALIDATION)[0]
    rows = [_observation("finance", task, 17, Arm.B0), _observation("finance", task, 17, Arm.B0)]
    report = runner.report_from_observations(comparison="validation", base_hash="base", candidate_hash="candidate", observations=rows)
    self.assertEqual(report.validity_status, "incomplete")
    self.assertIn("duplicate_or_unexpected_pair", report.infrastructure_failures)
    leaked = dataclasses.replace(rows[0], partition=Partition.FINAL)
    leaked_report = runner.report_from_observations(comparison="validation", base_hash="base", candidate_hash="candidate", observations=[leaked])
    self.assertTrue(leaked_report.partition_leak)

  def test_simulated_model_rows_cannot_be_promotion_evidence(self):
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    runner = EvaluationRunner(protocol, packages, safety_cases={"EVAL-004": True, "EVAL-005": True})
    rows = []
    for name in protocol.known_environments:
        for task in packages[name].tasks_for_partition(Partition.VALIDATION)[:20]:
            for seed in protocol.seeds:
                rows.extend((_observation(name, task, seed, Arm.B0), _observation(name, task, seed, Arm.L, True)))
    report = runner.report_from_observations(comparison="validation", base_hash="base", candidate_hash="candidate", observations=rows)
    self.assertEqual(report.validity_status, "invalid")
    self.assertFalse(report.promotion_eligible)

  def test_validation_allocations_are_disjoint_and_candidate_limited(self):
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    runner = EvaluationRunner(protocol, packages, safety_cases={"EVAL-004": True, "EVAL-005": True})
    seen = []
    def execute(arm, package, task, seed):
        seen.append((package.environment_id, task.task_id, arm))
        return RunObservation(task.task_id, package.environment_id, Partition.VALIDATION, seed, arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL)
    for index in range(3):
        report = runner.run_validation(base_hash="base", candidate_hash=f"candidate-{index}", execute=execute)
        self.assertTrue(report.promotion_eligible)
    with self.assertRaisesRegex(EvaluationError, "limit exhausted"):
        runner.run_validation(base_hash="base", candidate_hash="candidate-3", execute=execute)
    allocations = [set(task_id for env, task_id, arm in seen[index * 120:index * 120 + 120]) for index in range(3)]
    self.assertTrue(allocations[0].isdisjoint(allocations[1]))
    self.assertTrue(allocations[1].isdisjoint(allocations[2]))

  def test_validation_reservation_survives_executor_failure_and_blocks_replay(self):
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    runner = EvaluationRunner(protocol, packages)
    def fail_before_result(arm, package, task, seed):
        raise RuntimeError("executor interrupted")
    with self.assertRaises(RuntimeError):
        runner.run_validation(base_hash="base", candidate_hash="candidate", execute=fail_before_result)
    self.assertIn("base:candidate", runner.allocation_store.reserved)
    with self.assertRaisesRegex(EvaluationError, "durably reserved"):
        runner.run_validation(base_hash="base", candidate_hash="candidate", execute=fail_before_result)

  def test_promotion_requires_attested_registered_report_not_caller_gate_fields(self):
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    registry = TrustedEvaluatorRegistry()
    runner = EvaluationRunner(protocol, packages, registry, safety_cases={"EVAL-004": True, "EVAL-005": True})
    def execute(arm, package, task, seed):
        return RunObservation(task.task_id, package.environment_id, Partition.VALIDATION, seed, arm, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL)
    report = runner.run_validation(base_hash="base-v1", candidate_hash="candidate-v1", execute=execute)
    self.assertTrue(report.promotion_eligible)
    self.assertEqual(report.base_hash, "base-v1")
    forged = dataclasses.replace(report, validity_status="valid", safety_passed=True, candidate_hash="candidate-v2")
    with self.assertRaises(PromotionEvidenceRefused):
        forged.require_promotion_evidence(protocol, packages)

  def test_report_dict_matches_control_plane_consumption_contract(self):
    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    runner = EvaluationRunner(protocol, packages)
    report = runner.report_from_observations(comparison="validation", base_hash="base", candidate_hash="candidate", observations=[])
    self.assertTrue({"protocolHash", "baseHash", "candidateHash", "validityStatus", "promotionEligible", "partitionHashes", "evaluatorRefs", "attestation", "modelProvenanceComplete", "metricCellsComplete", "safetyCellsComplete", "exposure", "armSummaries", "confidenceIntervals", "workload"} <= set(report.to_dict()))


  def test_ablation_audit_rejects_retained_learned_material(self):
    clean = audit_ablation(AblationInput("a", "generic safety instructions", ("finance operations",)))
    self.assertTrue(clean.passed)
    dirty = audit_ablation(AblationInput("a", "generic safety instructions", (), ({"source": "learned", "id": "skill-1"},)))
    self.assertFalse(dirty.passed)
    self.assertEqual(dirty.retained_learned_artifacts, ("artifact:0",))


  def test_protocol_counts_and_budget_match_spec_defaults(self):
    protocol = EvaluationProtocol()
    self.assertEqual(protocol.validation_run_count, 360)
    self.assertEqual(protocol.final_run_count, 720)
    workload = protocol.workload(candidate_count=2, training_runs=60, transfer_runs=12, safety_runs=8, retries=4)
    self.assertEqual(workload.total_attempted_runs, 1_524)
    self.assertGreater(BudgetSpec().model_tokens, 0)
