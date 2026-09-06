from pathlib import Path
import json
import tempfile

import pytest

from adaptive_agent.production_evaluator import _lifecycle_execution_plan, _reject_shared_qa_path, _require_bound_real_receipt, build_job
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


def test_public_build_job_resume_restores_original_b0_after_promotion(tmp_path: Path, monkeypatch):
    from adaptive_agent.evaluation import Arm
    from adaptive_agent.experiment_runtime import DefaultExperimentStageRunner
    from adaptive_agent.models import SkillBundle, SkillVersion
    import adaptive_agent.production_evaluator as production_evaluator

    image = "sha256:" + "1" * 64
    monkeypatch.setattr(production_evaluator, "_docker_image_digest", lambda: image)
    first_app, _, first_job = build_job(str(tmp_path), None, None, None, initialize=True, force_job=True, job_id="restart-job")
    runtime = first_app.state.durable_runtime
    original = first_job.arm_bundles[Arm.B0]
    original_hash = original.content_hash

    promoted = SkillBundle(parent=original.bundle_id, skills=[SkillVersion(skillId="promoted-skill", version="1", procedure="new procedure")])
    promoted.content_hash = runtime.controller.candidates.recompute_bundle_hash(promoted)
    runtime.controller.store.save_bundle(promoted.bundle_id, promoted.parent, promoted.content_hash, promoted.model_dump_json(by_alias=True), False)
    runtime.controller.store.set_active_bundle(promoted.content_hash)

    resumed_app, resumed_protocol, resumed_job = build_job(str(tmp_path), None, None, None, initialize=False, force_job=True, job_id="restart-job")
    resumed_runtime = resumed_app.state.durable_runtime
    resumed_b0 = resumed_job.arm_bundles[Arm.B0]

    assert resumed_runtime.controller.get_active_bundle().content_hash == promoted.content_hash
    assert resumed_b0.content_hash == original_hash
    assert DefaultExperimentStageRunner(resumed_runtime, resumed_protocol).base_hash == original_hash


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
        store.append_evidence("model", {"run_id": run_id, "sequence": 1, "event_type": "model_response", "trust_class": "broker", "visibility": "evaluator_only", "redacted": 0, "content_hash": response_ref.sha256, "source_ref": response_ref.model_dump_json(by_alias=True)})
        store.append_evidence("outcome", {"run_id": run_id, "sequence": 2, "event_type": "trusted_outcome", "trust_class": "evaluator", "visibility": "evaluator_only", "redacted": 0, "content_hash": outcome_ref.sha256, "source_ref": outcome_ref.model_dump_json(by_alias=True)})
        _require_bound_real_receipt(directory, run_id)
