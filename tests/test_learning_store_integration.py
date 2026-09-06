from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SESSION4 = Path("/Users/vasu/.ao/data/worktrees/adaptive-agent/adaptive-agent-4")
pytest.importorskip("pydantic")


def _load_owned_module(name: str):
    path = ROOT / "src" / "adaptive_agent" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"adaptive_agent.{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if str(SESSION4 / "src") not in sys.path:
    sys.path.insert(0, str(SESSION4 / "src"))
retrieval = _load_owned_module("retrieval")
_load_owned_module("learning")
learning_store = _load_owned_module("learning_store")

from adaptive_agent.candidate import CandidateManager  # noqa: E402
from adaptive_agent.models import ArtifactRef, CandidateProposal, EnvironmentManifest, SkillBundle, SkillVersion, ToolSchema, sha256_json  # noqa: E402
from adaptive_agent.store import Store  # noqa: E402


def _setup_store(tmp_path: Path):
    store = Store(tmp_path)
    env_id, run_id = "finance-development", "run-durable"
    doc_ref = store.put_artifact("Public reconciliation documentation.")
    manifest = EnvironmentManifest(
        environmentId=env_id,
        version="1",
        docs=[doc_ref],
        toolSchemas=[ToolSchema(name="read", version="1", inputSchema={"type": "object"}, outputSchema={"type": "object"}, effect="read")],
        policyRef=ArtifactRef(id="policy", version="1", sha256="1" * 64),
        evaluatorRef=ArtifactRef(id="evaluator", version="1", sha256="2" * 64),
        resetRef=ArtifactRef(id="reset", version="1", sha256="3" * 64),
    )
    store.register_environment(env_id, "1", store.put_artifact(manifest.model_dump(mode="json", by_alias=True)))
    task_ref = store.put_artifact({"goal": "reconcile a version conflict"})
    store.register_task("task-durable", env_id, "1", task_ref.model_dump_json(), "development", "reconcile a version conflict")
    store.save_run(run_id, {"parent_run_id": None, "task_id": "task-durable", "environment_id": env_id, "bundle_id": "base", "status": "succeeded", "idempotency_key": "idem-durable", "last_event_sequence": 1, "created_at": datetime.now(timezone.utc).isoformat(), "run_json": "{}"})
    evidence_content = "broker observed a version conflict in development"
    evidence_ref = store.put_artifact(evidence_content)
    evidence_id = "ev-durable"
    store.append_evidence(evidence_id, {"run_id": run_id, "sequence": 1, "event_type": "tool_result", "content_hash": sha256_json(evidence_content), "source_ref": evidence_ref.model_dump_json(), "trust_class": "broker", "visibility": "learner", "redacted": 1})
    store.save_outcome("out-durable", {"run_id": run_id, "passed": 0, "score": 0.0, "metadata_json": "{}", "checked_at": datetime.now(timezone.utc).isoformat()})
    manager = CandidateManager(store)
    manager.initialize_active_bundle(SkillBundle(skills=[]))
    return store, manager, env_id, run_id, evidence_id


def test_store_records_and_candidate_manager_are_the_durable_authority(tmp_path: Path):
    store, manager, env_id, run_id, evidence_id = _setup_store(tmp_path)
    sources = learning_store.DurableLearningSourceAdapter(store)
    records = sources.records(environment_id=env_id, run_id=run_id)
    assert {item.source_id for item in records} >= {evidence_id}
    assert any(item.kind.value == retrieval.SourceKind.PUBLIC_DOC.value for item in records)
    retriever = sources.retriever(environment_id=env_id, run_id=run_id)
    result = retriever.search("version conflict", environment_id=env_id, run_id=run_id)
    assert [item.source_id for item in result.evidence] == [evidence_id]
    doc_id = next(item.source_id for item in records if item.kind.value == retrieval.SourceKind.PUBLIC_DOC.value)
    with pytest.raises(retrieval.RetrievalError):
        retriever.require_development_evidence([doc_id], environment_id=env_id, run_id=run_id)

    adapter = learning_store.CandidateManagerLearningAdapter(store, manager, proposal_type=CandidateProposal, bundle_type=SkillBundle, skill_type=SkillVersion)
    patch = b'{"operations":[]}'
    patch_hash = hashlib.sha256(patch).hexdigest()
    assert adapter.persist_candidate_patch(patch, patch_hash)["immutable"] is True
    active = manager.get_active_bundle()
    payload = {"baseBundleHash": active.content_hash, "editOperations": [json.dumps({"operation": "add", "path": "skills/reconcile/procedure", "value": "Read the latest version before retrying."}, sort_keys=True, separators=(",", ":"))], "changedArtifactHashes": [patch_hash], "supportingEvidenceIds": [evidence_id], "predictedEffect": "reduce conflicts", "proposerVersion": "test"}
    submitted = adapter.create_candidate(payload)
    assert submitted["state"] == "validated"
    assert manager.get_active_bundle().content_hash == active.content_hash
    assert store.get_candidate(submitted["candidate_id"])["state"] == "validated"
