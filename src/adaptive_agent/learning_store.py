"""Durable learning adapters for the Store and CandidateManager seams.

This module owns the construction that turns durable records into learner
context and turns a validated proposal into a draft candidate.  Session 2
supplies only the authenticated model client; it never owns storage, bundle
mutation, or candidate authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .learning import _canonical_operations
from .retrieval import AccessFilteredRetriever, InMemorySourceProvider, SourceKind, SourceRecord, content_hash


class LearningStoreError(ValueError):
    """Raised when durable learning records or candidate persistence are unsafe."""


class DurableStore(Protocol):
    def list_learning_records(self, *, environment_id: str, run_id: str) -> list[Mapping[str, Any]]: ...

    def put_immutable_bytes(self, data: bytes) -> Mapping[str, Any]: ...


class CandidateAuthority(Protocol):
    def get_active_bundle(self) -> Any: ...

    def submit_candidate(self, proposal: Any, candidate_bundle: Any) -> Any: ...


class DurableLearningSourceAdapter:
    """Expose only Store records that are safe for one learning request."""

    def __init__(self, store: DurableStore) -> None:
        self.store = store

    def records(self, *, environment_id: str, run_id: str) -> tuple[SourceRecord, ...]:
        list_records = getattr(self.store, "list_learning_records", None)
        raw_records = list_records(environment_id=environment_id, run_id=run_id) if callable(list_records) else self._legacy_records(environment_id=environment_id, run_id=run_id)
        sources: list[SourceRecord] = []
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                raise LearningStoreError("Store returned a non-object learning record")
            kind = raw.get("kind")
            if kind == SourceKind.PUBLIC_DOC.value:
                if raw.get("environmentId") != environment_id or raw.get("visibility") != "public":
                    raise LearningStoreError("public documentation is not bound to the requested environment")
                sources.append(self._source(raw, kind=SourceKind.PUBLIC_DOC, run_id=None, partition=None, visibility="learner", trust_class="operator"))
                continue
            if kind == SourceKind.LIVE_EVIDENCE.value:
                if raw.get("environmentId") != environment_id or raw.get("runId") != run_id:
                    raise LearningStoreError("development evidence is not bound to the requested run")
                if raw.get("partition") != "development" or raw.get("visibility") != "learner":
                    continue
                if raw.get("trustedOutcome") is not True:
                    continue
                sources.append(self._source(raw, kind=SourceKind.LIVE_EVIDENCE, run_id=run_id, partition="development", visibility="learner", trust_class=str(raw.get("trustClass", "broker"))))
                continue
            if kind == SourceKind.TASK_STATE.value:
                if raw.get("environmentId") != environment_id or raw.get("runId") != run_id or raw.get("visibility") != "learner":
                    continue
                if raw.get("trustedOutcome") is not True:
                    continue
                sources.append(self._source(raw, kind=SourceKind.TASK_STATE, run_id=run_id, partition=None, visibility="learner", trust_class="system"))
                continue
            raise LearningStoreError(f"unsupported durable learning record kind: {kind!r}")
        return tuple(sources)

    def _legacy_records(self, *, environment_id: str, run_id: str) -> list[Mapping[str, Any]]:
        """Read the pre-adapter Store API until session 4 exposes the narrow read seam."""
        get_environment = getattr(self.store, "get_environment", None)
        get_artifact = getattr(self.store, "get_artifact", None)
        if not callable(get_environment) or not callable(get_artifact):
            raise LearningStoreError("Store lacks the durable learning read seam")
        environment = get_environment(environment_id)
        if not isinstance(environment, Mapping):
            raise LearningStoreError("stored environment is missing")
        manifest_ref = environment.get("manifest_ref")
        if not isinstance(manifest_ref, str):
            raise LearningStoreError("stored environment manifest reference is missing")
        manifest_ref_data = json.loads(manifest_ref)
        manifest = get_artifact(manifest_ref_data["sha256"])
        records: list[Mapping[str, Any]] = []
        for doc_ref in manifest.get("docs", []):
            if not isinstance(doc_ref, Mapping) or not isinstance(doc_ref.get("sha256"), str):
                raise LearningStoreError("stored public document reference is malformed")
            content = json.dumps(get_artifact(doc_ref["sha256"]), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            records.append({"kind": "public_doc", "sourceId": str(doc_ref.get("id", doc_ref["sha256"])), "content": content, "contentHash": content_hash(content), "environmentId": environment_id, "visibility": "public"})
        list_evidence = getattr(self.store, "list_evidence", None)
        evidence_provenance = getattr(self.store, "evidence_provenance", None)
        get_outcome = getattr(self.store, "get_outcome_by_run_id", None)
        if not callable(list_evidence) or not callable(evidence_provenance) or not callable(get_outcome):
            raise LearningStoreError("Store lacks durable evidence provenance methods")
        trusted_outcome = get_outcome(run_id) is not None
        for row in list_evidence(run_id):
            provenance = evidence_provenance(row["evidence_id"])
            if not provenance or provenance.get("environment_id") != environment_id or provenance.get("partition") != "development":
                continue
            source_ref = row.get("source_ref")
            if not isinstance(source_ref, str):
                continue
            source_ref_data = json.loads(source_ref)
            content = json.dumps(get_artifact(source_ref_data["sha256"]), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            records.append({"kind": "live_evidence", "sourceId": row["evidence_id"], "content": content, "contentHash": row["content_hash"], "environmentId": environment_id, "runId": run_id, "partition": "development", "visibility": row["visibility"], "trustClass": row["trust_class"], "trustedOutcome": trusted_outcome})
        if trusted_outcome:
            records.append({"kind": "task_state", "sourceId": f"outcome:{run_id}", "content": "A trusted evaluator outcome is stored for this development run.", "contentHash": content_hash("A trusted evaluator outcome is stored for this development run."), "environmentId": environment_id, "runId": run_id, "visibility": "learner", "trustedOutcome": True})
        return records

    def retriever(self, *, environment_id: str, run_id: str) -> AccessFilteredRetriever:
        return AccessFilteredRetriever(InMemorySourceProvider(self.records(environment_id=environment_id, run_id=run_id)))

    @staticmethod
    def _source(raw: Mapping[str, Any], *, kind: SourceKind, run_id: str | None, partition: str | None, visibility: str, trust_class: str) -> SourceRecord:
        source_id = raw.get("sourceId")
        content = raw.get("content")
        supplied_hash = raw.get("contentHash")
        if not isinstance(source_id, str) or not source_id or not isinstance(content, str) or not content or not isinstance(supplied_hash, str):
            raise LearningStoreError("durable learning record lacks id, content, or content hash")
        if supplied_hash != content_hash(content):
            raise LearningStoreError(f"durable learning record hash mismatch: {source_id}")
        return SourceRecord(
            source_id=source_id,
            kind=kind,
            content=content,
            content_hash=supplied_hash,
            environment_id=raw.get("environmentId"),
            run_id=run_id,
            partition=partition,
            visibility=visibility,
            trust_class=trust_class,
            verified=True,
            metadata={"storedVisibility": raw.get("visibility"), "trustedOutcome": raw.get("trustedOutcome", False)},
        )


class CandidateManagerLearningAdapter:
    """Persist exact patch bytes and submit a bundle through CandidateManager."""

    def __init__(self, store: DurableStore, manager: CandidateAuthority, *, proposal_type: Any, bundle_type: Any, skill_type: Any) -> None:
        self.store = store
        self.manager = manager
        self.proposal_type = proposal_type
        self.bundle_type = bundle_type
        self.skill_type = skill_type

    def persist_candidate_patch(self, patch_bytes: bytes, content_hash_value: str) -> Mapping[str, Any]:
        put_bytes = getattr(self.store, "put_immutable_bytes", None)
        if callable(put_bytes):
            persisted = put_bytes(bytes(patch_bytes))
        else:
            try:
                patch_value = json.loads(bytes(patch_bytes).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LearningStoreError("canonical patch bytes are not JSON") from exc
            ref = self.store.put_artifact(patch_value)
            persisted = {"sha256": ref.sha256, "size": len(patch_bytes), "immutable": True}
        if not isinstance(persisted, Mapping) or persisted.get("sha256") != content_hash_value or persisted.get("size") != len(patch_bytes) or persisted.get("immutable") is not True:
            raise LearningStoreError("Store did not attest exact immutable patch bytes")
        return {"sha256": content_hash_value, "size": len(patch_bytes), "stored": True, "immutable": True}

    def create_candidate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        active = self.manager.get_active_bundle()
        if active is None or active.content_hash != payload.get("baseBundleHash"):
            raise LearningStoreError("candidate base is not the pinned active bundle")
        operations = []
        for raw in payload.get("editOperations", []):
            if not isinstance(raw, str):
                raise LearningStoreError("candidate operation is not canonical JSON")
            try:
                operations.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise LearningStoreError("candidate operation is not valid JSON") from exc
        operations = _canonical_operations(operations)
        candidate_bundle = self._apply_operations(active, operations)
        proposal = self.proposal_type(
            baseBundleHash=payload["baseBundleHash"],
            editOperations=[json.dumps(op, sort_keys=True, separators=(",", ":"), ensure_ascii=False) for op in operations],
            changedArtifactHashes=list(payload["changedArtifactHashes"]),
            supportingEvidenceIds=list(payload["supportingEvidenceIds"]),
            predictedEffect=payload["predictedEffect"],
            proposerVersion=payload["proposerVersion"],
        )
        submitted = self.manager.submit_candidate(proposal, candidate_bundle)
        return submitted.model_dump(mode="json", by_alias=True) if hasattr(submitted, "model_dump") else dict(submitted)

    def _apply_operations(self, active: Any, operations: Sequence[Mapping[str, Any]]) -> Any:
        candidate = self.bundle_type(parent=active.bundle_id, skills=[skill.model_copy(deep=True) for skill in active.skills], executionConfig=active.execution_config.model_copy(deep=True))
        candidate.content_hash = ""
        skills = {skill.skill_id: skill for skill in candidate.skills}
        for operation in operations:
            path = operation["path"]
            value = operation["value"]
            if path == "executionConfig/instructionVariant":
                if operation["operation"] == "remove":
                    candidate.execution_config.instruction_variant = "default"
                else:
                    candidate.execution_config.instruction_variant = value
                continue
            _, skill_id, field = path.split("/")
            skill = skills.get(skill_id)
            if skill is None:
                if operation["operation"] != "add" or field != "procedure":
                    raise LearningStoreError("cannot modify a missing skill except by adding its procedure")
                skill = self.skill_type(skillId=skill_id, version="1", procedure=value, parent=active.bundle_id)
                candidate.skills.append(skill)
                skills[skill_id] = skill
                continue
            if operation["operation"] == "add" and field in {"procedure", "applicability", "preconditions", "failureHandling"} and getattr(skill, field, None) not in (None, "", {}, []):
                raise LearningStoreError(f"cannot add an existing skill field: {path}")
            if operation["operation"] == "remove":
                value = {} if field == "applicability" else [] if field in {"preconditions", "failureHandling"} else ""
            setattr(skill, field, value)
        return candidate


def make_durable_learning_adapters(store: DurableStore, manager: CandidateAuthority, *, proposal_type: Any, bundle_type: Any, skill_type: Any) -> tuple[DurableLearningSourceAdapter, CandidateManagerLearningAdapter]:
    """Return the two injected adapters used by the learning service recipe."""
    return DurableLearningSourceAdapter(store), CandidateManagerLearningAdapter(store, manager, proposal_type=proposal_type, bundle_type=bundle_type, skill_type=skill_type)


__all__ = ["CandidateManagerLearningAdapter", "DurableLearningSourceAdapter", "LearningStoreError", "make_durable_learning_adapters"]
