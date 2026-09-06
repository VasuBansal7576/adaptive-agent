"""Portable integration tests for the durable Store/CandidateManager seam.

In an integrated checkout these are ordinary package imports.
For an isolated worker, set ``ADAPTIVE_AGENT_CORE_SRC`` to session 4's source
directory; this only extends the package search path for the test process.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
core_src = os.environ.get("ADAPTIVE_AGENT_CORE_SRC")
if core_src:
    import adaptive_agent

    adaptive_agent.__path__.append(str(Path(core_src) / "adaptive_agent"))
    adaptive_agent.__path__.append(str(ROOT / "src" / "adaptive_agent"))

try:
    from adaptive_agent.candidate import CandidateManager
    from adaptive_agent.learning_store import CandidateManagerLearningAdapter, DurableLearningSourceAdapter, LearningStoreError
    from adaptive_agent.models import ArtifactRef, CandidateProposal, EnvironmentManifest, SkillBundle, SkillVersion, ToolSchema, sha256_json
    from adaptive_agent.retrieval import RetrievalError, SourceKind, content_hash
    from adaptive_agent.store import Store
except ImportError as exc:
    pytest.skip(f"durable core is unavailable in this isolated worker: {exc}", allow_module_level=True)


ENVIRONMENT = "finance-development"
RUN = "run-durable"


def _setup_store(tmp_path: Path):
    store = Store(tmp_path)
    doc_ref = store.put_artifact("Public reconciliation documentation.")
    manifest = EnvironmentManifest(
        environmentId=ENVIRONMENT,
        version="1",
        docs=[doc_ref],
        toolSchemas=[ToolSchema(name="read", version="1", inputSchema={"type": "object"}, outputSchema={"type": "object"}, effect="read")],
        policyRef=ArtifactRef(id="policy", version="1", sha256="1" * 64),
        evaluatorRef=ArtifactRef(id="evaluator", version="1", sha256="2" * 64),
        resetRef=ArtifactRef(id="reset", version="1", sha256="3" * 64),
    )
    store.register_environment(ENVIRONMENT, "1", store.put_artifact(manifest.model_dump(mode="json", by_alias=True)))
    task_ref = store.put_artifact({"goal": "reconcile a version conflict"})
    store.register_task("task-durable", ENVIRONMENT, "1", task_ref.model_dump_json(), "development", "reconcile a version conflict")
    store.save_run(RUN, {"parent_run_id": None, "task_id": "task-durable", "environment_id": ENVIRONMENT, "bundle_id": "base", "status": "succeeded", "idempotency_key": "idem-durable", "last_event_sequence": 1, "created_at": datetime.now(timezone.utc).isoformat(), "run_json": "{}"})
    evidence_content = "broker observed a version conflict in development"
    evidence_ref = store.put_artifact(evidence_content)
    evidence_id = "ev-durable"
    store.append_evidence(evidence_id, {"run_id": RUN, "sequence": 1, "event_type": "tool_result", "content_hash": sha256_json(evidence_content), "source_ref": evidence_ref.model_dump_json(), "trust_class": "broker", "visibility": "learner", "redacted": 1})
    store.save_outcome("out-durable", {"run_id": RUN, "passed": 0, "score": 0.0, "metadata_json": "{}", "checked_at": datetime.now(timezone.utc).isoformat()})
    manager = CandidateManager(store)
    base = SkillBundle(skills=[SkillVersion(skillId="existing", version="1", procedure="Keep the existing procedure.")])
    manager.initialize_active_bundle(base)
    return store, manager, evidence_id


def _adapter(store, manager):
    return CandidateManagerLearningAdapter(store, manager, proposal_type=CandidateProposal, bundle_type=SkillBundle, skill_type=SkillVersion)


def _patch(operations):
    skill = {}
    config = {}
    for operation in operations:
        if operation["path"] == "executionConfig/instructionVariant":
            config["instructionVariant"] = operation["value"]
        else:
            skill[operation["path"].split("/")[-1]] = operation["value"]
    value = {"operations": operations, "skill": skill, "executionConfigPatch": config}
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return data, hashlib.sha256(data).hexdigest()


def _submit(store, manager, evidence_id, operations):
    patch_bytes, patch_hash = _patch(operations)
    adapter = _adapter(store, manager)
    adapter.persist_candidate_patch(patch_bytes, patch_hash)
    active = manager.get_active_bundle()
    payload = {
        "baseBundleHash": active.content_hash,
        "editOperations": [json.dumps(operation, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for operation in operations],
        "changedArtifactHashes": [patch_hash],
        "supportingEvidenceIds": [evidence_id],
        "predictedEffect": "reduce conflicts",
        "proposerVersion": "test",
    }
    return adapter.create_candidate(payload), patch_hash


def test_restart_retrieval_and_candidate_submission_bind_exact_persisted_patch(tmp_path: Path):
    store, manager, evidence_id = _setup_store(tmp_path)
    source_adapter = DurableLearningSourceAdapter(store)
    records = source_adapter.records(environment_id=ENVIRONMENT, run_id=RUN)
    assert any(item.kind.value == SourceKind.PUBLIC_DOC.value for item in records)
    retriever = source_adapter.retriever(environment_id=ENVIRONMENT, run_id=RUN)
    assert [item.source_id for item in retriever.search("version conflict", environment_id=ENVIRONMENT, run_id=RUN).evidence] == [evidence_id]
    doc_id = next(item.source_id for item in records if item.kind.value == SourceKind.PUBLIC_DOC.value)
    with pytest.raises(RetrievalError):
        retriever.require_development_evidence([doc_id], environment_id=ENVIRONMENT, run_id=RUN)

    operations = [
        {"operation": "add", "path": "skills/new-reconciliation/procedure", "value": "Read the latest version before retrying."},
        {"operation": "replace", "path": "skills/existing/failureHandling", "value": ["Stop and report a repeated conflict."]},
        {"operation": "replace", "path": "executionConfig/instructionVariant", "value": "reconciliation-v2"},
    ]
    submitted, patch_hash = _submit(store, manager, evidence_id, operations)
    assert submitted["state"] == "validated"
    assert manager.get_active_bundle().content_hash != submitted["candidateBundleHash"]
    assert store.has_artifact(patch_hash)

    restarted = Store(tmp_path)
    restarted_manager = CandidateManager(restarted)
    restarted_adapter = _adapter(restarted, restarted_manager)
    candidate = restarted.get_candidate(submitted["candidate_id"])
    assert candidate["state"] == "validated"
    assert restarted_adapter._load_patch_bytes(patch_hash) == _patch(operations)[0]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("skills/existing/procedure", "Updated procedure."),
        ("skills/existing/applicability", {"environment": "development"}),
        ("skills/existing/preconditions", ["Read current state first."]),
        ("skills/existing/failureHandling", ["Stop after one retry."]),
        ("executionConfig/instructionVariant", "safe-v2"),
    ],
)
def test_every_allowed_path_is_model_validated_and_submitted(tmp_path: Path, path: str, value):
    store, manager, evidence_id = _setup_store(tmp_path)
    submitted, _ = _submit(store, manager, evidence_id, [{"operation": "replace", "path": path, "value": value}])
    assert submitted["state"] == "validated"


def test_persisted_patch_mismatch_is_rejected_before_candidate_manager(tmp_path: Path):
    store, manager, evidence_id = _setup_store(tmp_path)
    actual = [{"operation": "replace", "path": "skills/existing/procedure", "value": "Actual."}]
    patch_bytes, patch_hash = _patch(actual)
    adapter = _adapter(store, manager)
    adapter.persist_candidate_patch(patch_bytes, patch_hash)
    active = manager.get_active_bundle()
    tampered = [{"operation": "replace", "path": "skills/existing/procedure", "value": "Tampered."}]
    payload = {"baseBundleHash": active.content_hash, "editOperations": [json.dumps(tampered[0], sort_keys=True, separators=(",", ":"))], "changedArtifactHashes": [patch_hash], "supportingEvidenceIds": [evidence_id], "predictedEffect": "x", "proposerVersion": "test"}
    with pytest.raises(LearningStoreError, match="do not match"):
        adapter.create_candidate(payload)
    assert manager.get_active_bundle().content_hash == active.content_hash


def test_operator_hidden_and_non_development_records_are_excluded():
    class RecordStore:
        def list_learning_records(self, *, environment_id: str, run_id: str):
            rows = []
            for source_id, visibility in (("doc-public", "public"), ("doc-operator", "operator"), ("doc-hidden", "evaluator_only")):
                text = f"{source_id}"
                rows.append({"kind": "public_doc", "sourceId": source_id, "content": text, "contentHash": content_hash(text), "environmentId": environment_id, "visibility": visibility})
            for source_id, partition, trusted in (("ev-dev", "development", True), ("ev-validation", "validation", True), ("ev-final", "final", True), ("ev-untrusted", "development", False)):
                text = source_id
                rows.append({"kind": "live_evidence", "sourceId": source_id, "content": text, "contentHash": content_hash(text), "environmentId": environment_id, "runId": run_id, "partition": partition, "visibility": "learner", "trustClass": "broker", "trustedOutcome": trusted})
            return rows

    records = DurableLearningSourceAdapter(RecordStore()).records(environment_id=ENVIRONMENT, run_id=RUN)
    assert [record.source_id for record in records] == ["doc-public", "ev-dev"]
