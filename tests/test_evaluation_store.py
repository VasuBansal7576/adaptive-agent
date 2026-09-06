import tempfile
import unittest
import dataclasses
from pathlib import Path

from adaptive_agent.evaluation_store import (
    ControllerSafetyProbeAdapter,
    SQLiteAllocationStore,
    SQLiteRunEvidenceStore,
    SQLiteTrustedAttestationLedger,
)
from adaptive_agent.evaluation import SafetyProbeResult, TrustedEvaluatorRegistry
from adaptive_agent.store import Store
from adaptive_agent.evaluation import Arm, EvaluationProtocol, ModelProvenance, Partition, RunObservation, build_environment_packages, sha256_json
from adaptive_agent.models import SkillBundle


class DurableEvaluatorStoreTests(unittest.TestCase):
    def test_controller_probe_adapter_requires_attested_shape(self):
        class Executor:
            def __init__(self, payload):
                self.payload = payload

            def execute_probe(self, case_id):
                return self.payload

        valid = Executor({"passed": True, "outputs": [{"transcript": "denied"}], "provenance": ["toolbroker"], "obligations": ["reject privileged input"]})
        registry = TrustedEvaluatorRegistry()
        ControllerSafetyProbeAdapter(valid).register(registry)
        result = registry.run_safety_probe_result("EVAL-004")
        self.assertIsInstance(result, SafetyProbeResult)
        self.assertTrue(result.passed)

        incomplete = Executor({"passed": True, "detail": {"ok": True}, "obligations": []})
        registry = TrustedEvaluatorRegistry()
        ControllerSafetyProbeAdapter(incomplete).register(registry)
        result = registry.run_safety_probe_result("EVAL-004")
        self.assertFalse(result.passed)
        self.assertIn("raised", result.obligations[0])

    def test_attestation_survives_store_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = SQLiteTrustedAttestationLedger(Store(path))
            first.put("token-1", "digest-1")
            with self.assertRaises(ValueError):
                first.put("token-1", "digest-tampered")
            reopened = SQLiteTrustedAttestationLedger(Store(path))
            self.assertTrue(reopened.durable)
            self.assertEqual(reopened.get("token-1"), "digest-1")

    def test_next_panel_is_atomic_disjoint_and_survives_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            store = Store(path)
            first = SQLiteAllocationStore(store)
            panels = (("task-0",), ("task-1",), ("task-2",))
            self.assertEqual(first.reserve_next("base", "candidate-0", panels, 3), 0)
            self.assertEqual(first.reserve_next("base", "candidate-1", panels, 3), 1)
            self.assertIsNone(first.reserve_next("base", "candidate-0", panels, 3))
            reopened = SQLiteAllocationStore(Store(path))
            self.assertEqual(reopened.reserve_next("base", "candidate-2", panels, 3), 2)
            self.assertIsNone(reopened.reserve_next("base", "candidate-3", panels, 3))

    def test_run_evidence_verifier_accepts_bound_real_record_and_rejects_pin_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory))
            package = build_environment_packages()["finance"]
            task = package.tasks_for_partition(Partition.VALIDATION)[0]
            protocol = EvaluationProtocol(image_digest="sha256:image-real")
            frozen = protocol.freeze(build_environment_packages())
            run_id, response_id = "run-1", "provider-response-1"
            bundle = SkillBundle()
            store.save_bundle(bundle.bundle_id, bundle.parent, bundle.content_hash, bundle.model_dump_json(by_alias=True))
            run_payload = {"arm": "L", "seed": 17, "armBundles": {"L": bundle.content_hash}}
            store.save_run(run_id, {"task_id": task.task_id, "environment_id": "finance", "bundle_id": bundle.bundle_id, "status": "succeeded", "idempotency_key": "run-key", "last_event_sequence": 0, "created_at": "now", "run_json": __import__("json").dumps(run_payload)})
            version_refs = {"policy": package.manifest.policy_ref.sha256, "schema": sha256_json(package.manifest.tool_schemas), "planner": protocol.core_planner_hash, "budget": sha256_json(frozen.inputs["runBudget"]), "image": protocol.image_digest}
            usage = {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
            response = {"responseId": response_id, "provider": "openai-codex", "modelProfile": protocol.model_profile, "status": "complete", "usage": usage, "versionRefs": version_refs, "arm": "L", "seed": 17, "bundleHash": bundle.content_hash}
            response_ref = store.put_artifact(response)
            store.append_evidence("evidence-1", {"run_id": run_id, "sequence": 1, "event_type": "model_response", "content_hash": response_ref.sha256, "source_ref": response_ref.model_dump_json(by_alias=True), "trust_class": "broker", "visibility": "operator", "redacted": 0})
            accounting_ref = store.put_artifact({"responseId": response_id, "runId": run_id, "taskId": task.task_id, "environmentId": "finance", "usage": usage, "costMicrounits": 1, "durationSeconds": 1.0, "versionRefs": version_refs, "arm": "L", "seed": 17, "bundleHash": bundle.content_hash})
            outcome = {"responseId": response_id, "runId": run_id, "taskId": task.task_id, "environmentId": "finance", "passed": True, "reliable": True, "safetyViolations": 0}
            outcome_ref = store.put_artifact(outcome)
            store.append_evidence("outcome-1", {"run_id": run_id, "sequence": 2, "event_type": "trusted_outcome", "content_hash": outcome_ref.sha256, "source_ref": outcome_ref.model_dump_json(by_alias=True), "trust_class": "evaluator", "visibility": "operator", "redacted": 0})
            expected = {"model": sha256_json({"profile": protocol.model_profile, "provider": protocol.provider}), "planner": protocol.core_planner_hash, "budget": sha256_json(frozen.inputs["runBudget"]), "policy": sha256_json(package.manifest.policy_ref), "schema": sha256_json(package.manifest.tool_schemas), "image": protocol.image_digest}
            row = RunObservation(task.task_id, "finance", Partition.VALIDATION, 17, Arm.L, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, response_id=response_id, accounting_ref=accounting_ref.sha256, evidence_ref="evidence-1", outcome_ref="outcome-1", config_hashes=expected, run_id=run_id, bundle_hash=bundle.content_hash)
            verifier = SQLiteRunEvidenceStore(store)
            self.assertTrue(verifier.verify(row, frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, response_id="wrong"), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, run_id="wrong"), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, accounting_ref="missing"), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, config_hashes={**expected, "image": "sha256:changed"}), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, config_hashes={**expected, "planner": "changed"}), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, passed=False), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, arm=Arm.B0), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, seed=23), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, bundle_hash="wrong"), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, cost_microunits=99), frozen, package))
            bad_response = store.put_artifact({**response, "provider": "untrusted", "usage": None})
            store.append_evidence("bad-response", {"run_id": run_id, "sequence": 4, "event_type": "model_response", "content_hash": bad_response.sha256, "source_ref": bad_response.model_dump_json(by_alias=True), "trust_class": "broker", "visibility": "operator", "redacted": 0})
            self.assertFalse(verifier.verify(dataclasses.replace(row, evidence_ref="bad-response"), frozen, package))
            store.append_evidence("corrupt", {"run_id": run_id, "sequence": 3, "event_type": "model_response", "content_hash": response_ref.sha256, "source_ref": store.put_artifact(["not", "object"]).model_dump_json(by_alias=True), "trust_class": "broker", "visibility": "operator", "redacted": 0})
            self.assertFalse(verifier.verify(dataclasses.replace(row, evidence_ref="corrupt"), frozen, package))


if __name__ == "__main__":
    unittest.main()
