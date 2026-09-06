from pathlib import Path
import json
import tempfile

import pytest

from adaptive_agent.production_evaluator import _lifecycle_execution_plan, _lifecycle_stages, _reject_shared_qa_path, _require_bound_real_receipt
from adaptive_agent.store import Store


def test_evaluator_rejects_shared_root_qa_store_and_children():
    with pytest.raises(RuntimeError, match="shared root QA"):
        _reject_shared_qa_path("/private/tmp/adaptive-agent-browser-api")
    with pytest.raises(RuntimeError, match="shared root QA"):
        _reject_shared_qa_path("/private/tmp/adaptive-agent-browser-api/child")


def test_evaluator_requires_distinct_source_and_target_stores(tmp_path: Path):
    with pytest.raises(RuntimeError, match="isolated"):
        _reject_shared_qa_path(tmp_path, tmp_path)


def test_lifecycle_execution_plan_accounts_for_nested_work_and_retries():
    counts = {"bootstrap": 1, "training": 60, "learning": 1, "transfer": 3, "adaptation": 3, "safety": 2, "validation": 360, "final": 720}
    plan = _lifecycle_execution_plan(counts, retries=1)
    assert plan == {"primaryCells": 1150, "primaryAttempts": 2300, "retryAttempts": 1150, "nestedSubcalls": 32, "totalAdmissions": 2332, "retriesPerCell": 1}


def test_lifecycle_stages_freeze_declared_retry_semantics():
    class Task:
        task_id = "development-task"

    class Package:
        def tasks_for_partition(self, partition):
            return (Task(),)

    class Protocol:
        known_environments = ("finance",)
        safety_case_ids = ("safety-case",)
        validation_run_count = 1
        final_run_count = 1
        sealed_environment = "sealed"

    class Runtime:
        packages = {"finance": Package()}

        def run_experiment_stage(self, **kwargs):
            return kwargs

    stages = _lifecycle_stages(Runtime(), Protocol(), declared_retries=2)
    assert all(stage.retries == 2 for stage in stages)


def test_final_preparer_persists_memory_disabled_ablation_from_learning_bundle(tmp_path: Path):
    from adaptive_agent.evaluation import EvaluationProtocol, build_environment_packages
    from adaptive_agent.models import SkillBundle

    packages = build_environment_packages()
    protocol = EvaluationProtocol()
    protocol.freeze(packages)
    store = Store(tmp_path)
    learned = SkillBundle()
    store.save_bundle(learned.bundle_id, learned.parent, learned.content_hash, learned.model_dump_json(by_alias=True))

    class Controller:
        def __init__(self):
            self.store = store

    class Runtime:
        controller = Controller()
        experiment_stage_runner = None

        def run_experiment_stage(self, **kwargs):
            return kwargs

    Runtime.packages = packages
    final = _lifecycle_stages(Runtime(), protocol, declared_retries=0)[-1]
    prepared = final.final_preparer({"results": {"learning": {"candidate-generation": {"candidateBundleHash": learned.content_hash}}}})
    assert prepared["ablationBundleHash"]
    ablation_row = store.get_bundle_by_hash(prepared["ablationBundleHash"])
    assert ablation_row is not None
    ablation = SkillBundle.model_validate_json(ablation_row["bundle_json"])
    assert ablation.skills == [] and ablation.execution_config.skill_refs == []


def test_bound_private_outcome_requires_trusted_canonical_identity():
    with tempfile.TemporaryDirectory() as directory:
        store = Store(Path(directory))
        run_id = "source-run"
        identity = {"runId": run_id, "taskId": "task-1", "environmentId": "finance", "arm": "B0", "seed": 17, "bundleHash": "a" * 64}
        store.save_run(run_id, {"task_id": "task-1", "environment_id": "finance", "bundle_id": "bundle-1", "status": "succeeded", "idempotency_key": "source-key", "request_fingerprint": "source-fingerprint", "created_at": "2026-09-06T00:00:00+00:00", "run_json": json.dumps({"arm": "B0", "seed": 17, "bundleHash": "a" * 64})})
        response = {**identity, "responseId": "response-1", "provider": "openai-codex"}
        outcome = {**identity, "responseId": "response-1", "passed": True, "reliable": True, "safetyViolations": 0}
        response_ref = store.put_artifact(response)
        outcome_ref = store.put_artifact(outcome)
        store.append_evidence("model", {"run_id": run_id, "sequence": 1, "event_type": "model_response", "trust_class": "broker", "visibility": "operator", "redacted": 0, "content_hash": response_ref.sha256, "source_ref": response_ref.model_dump_json(by_alias=True)})
        store.append_evidence("outcome", {"run_id": run_id, "sequence": 2, "event_type": "trusted_outcome", "trust_class": "evaluator", "visibility": "evaluator_only", "redacted": 0, "content_hash": outcome_ref.sha256, "source_ref": outcome_ref.model_dump_json(by_alias=True)})
        _require_bound_real_receipt(directory, run_id)
