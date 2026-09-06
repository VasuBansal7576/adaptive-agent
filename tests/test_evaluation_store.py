import tempfile
import unittest
import dataclasses
from pathlib import Path

from adaptive_agent.evaluation_store import (
    SQLiteAllocationStore,
    SQLiteRunEvidenceStore,
    SQLiteTrustedAttestationLedger,
)
from adaptive_agent.store import Store
from adaptive_agent.evaluation import Arm, EvaluationProtocol, ModelProvenance, Partition, RunObservation, build_environment_packages, sha256_json


class DurableEvaluatorStoreTests(unittest.TestCase):
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
            store.save_run(run_id, {"task_id": task.task_id, "environment_id": "finance", "bundle_id": "bundle", "status": "succeeded", "idempotency_key": "run-key", "last_event_sequence": 0, "created_at": "now", "run_json": "{}"})
            response = {"responseId": response_id, "provider": "openai-codex", "status": "complete"}
            response_ref = store.put_artifact(response)
            store.append_evidence("evidence-1", {"run_id": run_id, "sequence": 1, "event_type": "model_response", "content_hash": response_ref.sha256, "source_ref": response_ref.model_dump_json(by_alias=True), "trust_class": "broker", "visibility": "operator", "redacted": 0})
            accounting_ref = store.put_artifact({"responseId": response_id, "runId": run_id, "taskId": task.task_id, "environmentId": "finance", "versionRefs": {"policy": "v1"}})
            expected = {"model": sha256_json({"profile": protocol.model_profile, "provider": protocol.provider}), "planner": protocol.core_planner_hash, "budget": sha256_json(frozen.inputs["runBudget"]), "policy": sha256_json(package.manifest.policy_ref), "schema": sha256_json(package.manifest.tool_schemas), "image": protocol.image_digest}
            row = RunObservation(task.task_id, "finance", Partition.VALIDATION, 17, Arm.L, True, True, 0, 1, 1.0, model_provenance=ModelProvenance.REAL_MODEL, response_id=response_id, accounting_ref=accounting_ref.sha256, evidence_ref="evidence-1", config_hashes=expected, run_id=run_id)
            verifier = SQLiteRunEvidenceStore(store)
            self.assertTrue(verifier.verify(row, frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, response_id="wrong"), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, run_id="wrong"), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, accounting_ref="missing"), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, config_hashes={**expected, "image": "sha256:changed"}), frozen, package))
            self.assertFalse(verifier.verify(dataclasses.replace(row, config_hashes={**expected, "planner": "changed"}), frozen, package))


if __name__ == "__main__":
    unittest.main()
