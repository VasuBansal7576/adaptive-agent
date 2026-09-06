"""Runtime construction for learning from completed durable development runs.

The runtime reads the existing Store after a run has completed and exposes only
public documentation, redacted broker observations, and a coarse trusted
outcome marker to ``LearningService``.  It never evaluates a run, promotes a
candidate, or treats model text as an outcome.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .learning import LearningProposal, LearningService, PlannerLearningAdapter
from .learning_store import CandidateManagerLearningAdapter, DurableLearningSourceAdapter, LearningStoreError
from .learning_projection import DurableBrokerLearningProjection, LearningProjectionError
from .retrieval import AccessFilteredRetriever, InMemorySourceProvider, canonical_json, content_hash


class LearningRuntimeError(ValueError):
    """Raised when a completed durable run cannot safely become learning input."""


def _artifact_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return canonical_json(value)


_SECRET_VALUE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|bearer\s+\S+|"
    r"(?:api[_-]?key|token|secret|password|authorization|credential)\s*[:=]\s*\S+)"
)
_HIDDEN_KEY = re.compile(r"(?i)(?:hidden|expected|evaluator|answer[_ -]?key|secret|credential|api[_-]?key|token|password|authorization)")


def _sanitize_learning_value(value: Any) -> Any:
    """Keep operational shape while excluding credentials and evaluator text."""
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_learning_value(item)
            for key, item in value.items()
            if not _HIDDEN_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_learning_value(item) for item in value]
    return value


class StoreModelObservationSink:
    """Persist authenticated model provenance without making it learner evidence."""

    def __init__(self, store: Any, run_id: str) -> None:
        self.store = store
        self.run_id = run_id

    def record_model_observation(self, evidence: Mapping[str, Any], *, trusted_parent: bool = False) -> Any:
        if trusted_parent is not True:
            raise LearningRuntimeError("model observations require trusted parent provenance")
        if not isinstance(evidence.get("responseId"), str) or not isinstance(evidence.get("usage"), Mapping):
            raise LearningRuntimeError("model observation provenance is incomplete")
        usage = dict(evidence["usage"])
        duration = evidence.get("durationSeconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or duration < 0:
            raise LearningRuntimeError("model observation duration is malformed")
        cost = evidence.get("costMicrounits")
        if cost is not None and (isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(float(cost)) or cost < 0):
            raise LearningRuntimeError("model observation economic cost is malformed")
        nominal = evidence.get("nominalCostUsd")
        if nominal is None and isinstance(usage.get("cost"), Mapping):
            nominal = usage["cost"].get("total")
        if nominal is not None and (isinstance(nominal, bool) or not isinstance(nominal, (int, float)) or not math.isfinite(float(nominal)) or nominal < 0):
            raise LearningRuntimeError("model observation nominal cost is malformed")
        status = evidence.get("economicCostStatus", "unknown")
        if not isinstance(status, str) or not status:
            raise LearningRuntimeError("model observation economic status is malformed")
        payload = {
            "provider": evidence.get("provider"),
            "model": evidence.get("model"),
            "responseId": evidence["responseId"],
            "usage": usage,
            "durationSeconds": float(duration),
            "economicCostStatus": status,
            "status": str(evidence.get("status", "complete")),
            **({"costMicrounits": int(round(float(cost)))} if cost is not None else {}),
            **({"nominalCostUsd": float(nominal)} if nominal is not None else {}),
        }
        ref = self.store.put_artifact(payload)
        sequence = self.store.next_event_sequence(self.run_id)
        self.store.append_evidence(
            f"learning-model-{evidence['responseId']}",
            {"run_id": self.run_id, "sequence": sequence, "event_type": "learning_model_observation", "content_hash": ref.sha256, "source_ref": ref.model_dump_json(by_alias=True), "trust_class": "system", "visibility": "operator", "redacted": 0},
        )
        return payload


@dataclass
class LearningRuntime:
    store: Any
    manager: Any
    source_adapter: DurableLearningSourceAdapter
    candidate_adapter: CandidateManagerLearningAdapter
    service: LearningService
    token_budget: int = 20_000
    wall_seconds: float = 120.0

    @classmethod
    def build(
        cls,
        *,
        store: Any,
        manager: Any,
        model_client: Any | None = None,
        evidence_sink_factory: Any = StoreModelObservationSink,
        token_budget: int = 20_000,
        wall_seconds: float = 120.0,
    ) -> "LearningRuntime":
        if token_budget <= 0 or wall_seconds <= 0:
            raise ValueError("learning runtime budgets must be positive")
        from adaptive_agent.models import CandidateProposal, SkillBundle, SkillVersion
        if model_client is None:
            from adaptive_agent.planner import PrimeCliModelClient

            model_client = PrimeCliModelClient(coding_agent_dir=os.environ.get("PRIME_AGENT_CODING_AGENT_DIR"), cwd="/private/tmp")
        source_adapter = DurableLearningSourceAdapter(store)
        candidate_adapter = CandidateManagerLearningAdapter(store, manager, proposal_type=CandidateProposal, bundle_type=SkillBundle, skill_type=SkillVersion)
        # The sink is bound per completed run, so construction is completed in
        # propose_completed_run after the run identity is verified.
        service = LearningService(AccessFilteredRetriever(InMemorySourceProvider()), PlannerLearningAdapter(model_client, evidence_sink_factory(store, "__unbound__")), candidate_adapter, lambda: manager.get_active_bundle().content_hash)
        return cls(store, manager, source_adapter, candidate_adapter, service, token_budget, wall_seconds)

    def _materialize_run_records(self, *, environment_id: str, run_id: str, public_documents: Any = ()) -> list[Mapping[str, Any]]:
        save = getattr(self.store, "save_learning_record", None)

        # A completed run may have been projected by an earlier process whose
        # source artifacts have since been compacted. Keep that projection as
        # the restart source of truth and only fall back to raw CAS joins when
        # no materialized records are available.
        existing_reader = getattr(self.store, "list_learning_records", None)
        existing_rows = existing_reader(environment_id=environment_id, run_id=run_id) if callable(existing_reader) else ()
        existing_records: list[Mapping[str, Any]] = []
        existing_keys: set[tuple[Any, Any]] = set()
        existing_indexes: dict[tuple[Any, Any], int] = {}
        encoded_keys: set[tuple[Any, Any]] = set()
        for row in existing_rows or ():
            if not isinstance(row, Mapping):
                continue
            encoded = row.get("record_json")
            if isinstance(encoded, str):
                try:
                    decoded = json.loads(encoded)
                except json.JSONDecodeError:
                    continue
                if isinstance(decoded, Mapping):
                    key = (decoded.get("kind"), decoded.get("sourceId"))
                    if key not in existing_keys:
                        existing_keys.add(key)
                        existing_indexes[key] = len(existing_records)
                        existing_records.append(decoded)
                    elif key not in encoded_keys:
                        existing_records[existing_indexes[key]] = decoded
                    encoded_keys.add(key)
            elif isinstance(row.get("kind"), str):
                key = (row.get("kind"), row.get("sourceId"))
                if key not in existing_keys:
                    existing_keys.add(key)
                    existing_indexes[key] = len(existing_records)
                    existing_records.append(row)

        # Materialized records remain the restart source of truth, but only
        # after their content, bindings, and derived-evidence provenance have
        # been revalidated.  This prevents a legacy fallback row from becoming
        # learner context merely because it has the right record shape.
        raw_records: list[Mapping[str, Any]] = []
        materialized_keys: set[tuple[Any, Any]] = set()
        for record in existing_records:
            kind = record.get("kind")
            key = (kind, record.get("sourceId"))
            content = record.get("content")
            if not isinstance(content, str) or record.get("contentHash") != content_hash(content):
                continue
            if kind == "public_doc":
                valid = record.get("environmentId") == environment_id and record.get("visibility") == "public"
            elif kind == "task_state":
                valid = record.get("environmentId") == environment_id and record.get("runId") == run_id and record.get("visibility") == "learner" and record.get("trustedOutcome") is True
            elif kind == "live_evidence":
                source_id = record.get("sourceId")
                get_evidence = getattr(self.store, "get_evidence", None)
                derived = get_evidence(source_id) if callable(get_evidence) and isinstance(source_id, str) else None
                # Older persisted projections predate the derived-evidence
                # provenance row and sourceContentHash field. They are still
                # eligible as restart input when the Store returned the
                # encoded, content-hashed materialized record itself. Direct
                # legacy fallback rows never receive this compatibility path.
                legacy_materialized = key in encoded_keys
                valid = (
                    isinstance(source_id, str) and (source_id.startswith("broker:") or legacy_materialized)
                    and record.get("environmentId") == environment_id and record.get("runId") == run_id
                    and record.get("partition") == "development" and record.get("visibility") == "learner"
                    and record.get("trustClass") == "broker" and record.get("trustedOutcome") is True
                    and (
                        legacy_materialized
                        or (
                            isinstance(record.get("sourceContentHash"), str)
                            and isinstance(derived, Mapping) and derived.get("event_type") == "learning_evidence_projection"
                            and derived.get("run_id") == run_id and derived.get("visibility") == "learner"
                            and derived.get("redacted") == 1 and derived.get("trust_class") == "broker"
                        )
                    )
                )
            else:
                valid = False
            if valid:
                # Reuse each durable source once on restart. The materialized
                # record is already canonical and must not be rewritten under
                # a fresh record id.
                raw_records.append(record)
                materialized_keys.add(key)

        def persist(record_id: str, record: Mapping[str, Any]) -> None:
            key = (record.get("kind"), record.get("sourceId"))
            if key in materialized_keys:
                return
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if callable(save):
                save(record_id, environment_id, run_id, encoded)
            raw_records.append(record)
            materialized_keys.add(key)
        environment = self.store.get_environment(environment_id)
        if not isinstance(environment, Mapping):
            raise LearningRuntimeError("completed run environment is not stored")
        public_doc_reader = getattr(self.store, "get_public_docs", None)
        manifest_ref = environment.get("manifest_ref")
        manifest_docs: list[Mapping[str, Any]] = []
        if isinstance(manifest_ref, str):
            try:
                manifest = self.store.get_artifact(json.loads(manifest_ref)["sha256"])
                manifest_docs = [doc for doc in manifest.get("docs", []) if isinstance(doc, Mapping)]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                # A restart can retain the narrow learner projection after
                # source CAS compaction.  Existing records are sufficient in
                # that case; a run with no projection still fails closed below.
                manifest = {}
        if callable(public_doc_reader):
            try:
                docs = public_doc_reader(environment_id)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                docs = []
        else:
            if not isinstance(manifest_ref, str):
                raise LearningRuntimeError("completed run public manifest reference is missing")
            docs = [{**doc, "content": self.store.get_artifact(doc["sha256"])} for doc in manifest.get("docs", [])]
        if not docs:
            docs = [record for record in raw_records if record.get("kind") == "public_doc"]
        supplied_docs = {}
        for supplied in public_documents or ():
            if hasattr(supplied, "document_id"):
                supplied = {"id": supplied.document_id, "version": supplied.version, "text": supplied.text, "classification": getattr(supplied, "classification", "learner")}
            if isinstance(supplied, Mapping) and isinstance(supplied.get("id"), str):
                supplied_docs[supplied["id"]] = dict(supplied)
        if callable(public_doc_reader):
            returned_ids = {doc.get("id") for doc in docs if isinstance(doc, Mapping)}
            for ref in manifest_docs:
                if ref.get("id") in returned_ids:
                    continue
                candidate = supplied_docs.get(ref.get("id"))
                if candidate is None or candidate.get("classification") in {"operator", "evaluator_only"}:
                    continue
                candidate_content = {key: candidate[key] for key in ("id", "version", "text") if key in candidate}
                if content_hash(_artifact_text(candidate_content)) != ref.get("sha256"):
                    raise LearningRuntimeError(f"public document hash mismatch: {ref.get('id')}")
                docs.append({**ref, "content": candidate_content})
        for doc in docs:
            doc_id = doc.get("id") if isinstance(doc, Mapping) else None
            if not isinstance(doc_id, str) and isinstance(doc, Mapping):
                doc_id = doc.get("sourceId")
            if not isinstance(doc, Mapping) or not isinstance(doc_id, str):
                raise LearningRuntimeError("public document projection is malformed")
            if "content" in doc:
                stored_doc = doc["content"]
            else:
                candidate = supplied_docs.get(doc_id)
                if candidate is None:
                    raise LearningRuntimeError(f"public document content is not stored: {doc_id}")
                ref = self.store.put_artifact(candidate)
                if ref.sha256 != doc.get("sha256"):
                    raise LearningRuntimeError(f"public document hash mismatch: {doc_id}")
                stored_doc = self.store.get_artifact(doc["sha256"])
            doc_content = _artifact_text(stored_doc)
            record = {"kind": "public_doc", "sourceId": doc_id, "content": doc_content, "contentHash": content_hash(doc_content), "environmentId": environment_id, "visibility": "public"}
            persist(f"learning-doc-{doc.get('sha256', content_hash(doc_content))}", record)

        trusted_outcome = self.store.get_outcome_by_run_id(run_id)
        if not isinstance(trusted_outcome, Mapping) or "passed" not in trusted_outcome or not isinstance(trusted_outcome.get("passed"), (bool, int)):
            raise LearningRuntimeError("completed development run lacks a trusted evaluator outcome")
        trusted_outcome_present = bool(trusted_outcome)
        outcome_passed = bool(trusted_outcome["passed"])
        run_row = self.store.get_run(run_id)
        try:
            projected = DurableBrokerLearningProjection(self.store).project(
                environment_id=environment_id,
                run_id=run_id,
                task_id=run_row["task_id"] if isinstance(run_row, Mapping) else "",
                outcome_passed=outcome_passed,
            )
        except (LearningProjectionError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # If the raw broker source was compacted, continue with a
            # previously persisted and validated projection.  Without one,
            # the unified evidence seam below reports the missing dependency.
            projected = []
        for record_id, record in projected:
            persist(record_id, record)
        persisted_projection = []
        if not projected:
            for record in existing_records:
                if not isinstance(record, Mapping) or record.get("kind") != "live_evidence" or not isinstance(record.get("sourceId"), str) or not record["sourceId"].startswith("broker:"):
                    continue
                if record.get("environmentId") != environment_id or record.get("runId") != run_id or record.get("partition") != "development" or record.get("visibility") != "learner" or record.get("trustClass") != "broker" or record.get("trustedOutcome") is not True:
                    continue
                content = record.get("content")
                if not isinstance(content, str) or record.get("contentHash") != content_hash(content) or not isinstance(record.get("sourceContentHash"), str):
                    continue
                derived = getattr(self.store, "get_evidence", lambda _id: None)(record["sourceId"])
                if not isinstance(derived, Mapping) or derived.get("event_type") != "learning_evidence_projection" or derived.get("run_id") != run_id or derived.get("visibility") != "learner" or derived.get("redacted") != 1 or derived.get("trust_class") != "broker":
                    continue
                persisted_projection.append(record)
        joined_reader = getattr(self.store, "list_learning_evidence", None)
        if not callable(joined_reader):
            raise LearningRuntimeError("Store lacks unified learning evidence seam")
        events = joined_reader(environment_id=environment_id, run_id=run_id, include_broker_projection=True)
        for event in events if not projected and not persisted_projection else ():
            if event.get("kind") != "broker_call":
                continue
            if event.get("partition") != "development" or event.get("environmentId") != environment_id or event.get("runId") != run_id:
                continue
            evidence_id = event.get("evidenceId")
            if not isinstance(evidence_id, str):
                continue
            safe = {
                key: _sanitize_learning_value(event[key])
                for key in ("callId", "tool", "input", "result", "status", "errorCode", "retry", "version", "effect", "idempotencyKey", "argumentsSha256", "resultSha256", "evidenceContentHash")
                if key in event and event[key] is not None
            }
            content = f"Broker development observation: {canonical_json(safe)}"
            source_id = evidence_id if evidence_id.startswith("broker:") else f"broker:{evidence_id}"
            record = {"kind": "live_evidence", "sourceId": source_id, "content": content, "contentHash": content_hash(content), "sourceContentHash": event.get("evidenceContentHash"), "sourceEvidenceId": evidence_id, "sourceCallId": event.get("callId"), "environmentId": environment_id, "runId": run_id, "partition": "development", "visibility": "learner", "trustClass": "broker", "trustedOutcome": trusted_outcome_present, "outcomePassed": outcome_passed}
            persist(f"learning-broker-{source_id}", record)
        outcome_content = f"A trusted evaluator outcome is recorded for this completed development run; passed={str(outcome_passed).lower()}."
        outcome_record = {"kind": "task_state", "sourceId": f"outcome:{run_id}", "content": outcome_content, "contentHash": content_hash(outcome_content), "environmentId": environment_id, "runId": run_id, "visibility": "learner", "trustedOutcome": trusted_outcome_present, "outcomePassed": outcome_passed}
        persist(f"learning-outcome-{run_id}", outcome_record)
        return raw_records

    def propose_completed_run(self, run_id: str, *, goal: str | None = None, feedback: Mapping[str, Any] | None = None, public_documents: Any = ()) -> LearningProposal:
        stored = self.store.get_run(run_id)
        if not isinstance(stored, Mapping) or stored.get("status") not in {"succeeded", "failed", "cancelled", "timed_out", "outcome_unknown"}:
            raise LearningRuntimeError("learning requires a completed development run")
        environment_id = stored.get("environment_id")
        if not isinstance(environment_id, str):
            raise LearningRuntimeError("completed run environment binding is missing")
        task = self.store.get_task(stored["task_id"])
        if not isinstance(task, Mapping) or task.get("partition") != "development":
            raise LearningRuntimeError("learning requires a DEVELOPMENT task")
        raw_records = self._materialize_run_records(environment_id=environment_id, run_id=run_id, public_documents=public_documents)
        environment = {"environmentId": environment_id, "version": task.get("version", "1")}
        sink = StoreModelObservationSink(self.store, run_id)
        self.service.retriever = self.source_adapter.retriever(environment_id=environment_id, run_id=run_id) if not raw_records else self.source_adapter.retriever_from_raw(raw_records, environment_id=environment_id, run_id=run_id)
        self.service.model_runner = PlannerLearningAdapter(self.service.model_runner.client, sink)
        outcome = self.store.get_outcome_by_run_id(run_id)
        status = "succeeded" if bool(outcome.get("passed")) else "failed"
        return self.service.propose(run_id=run_id, environment_id=environment_id, goal=goal or task["goal"], environment=environment, feedback=feedback or {"status": status}, remaining_deadline=self.wall_seconds, token_cap=self.token_budget, max_repair_attempts=1)

    def reload_candidate(self, candidate_id: str) -> Mapping[str, Any]:
        candidate = self.store.get_candidate(candidate_id)
        if not isinstance(candidate, Mapping):
            raise LearningRuntimeError("candidate is not readable after restart")
        candidate_hash = candidate.get("candidate_bundle_hash")
        if not isinstance(candidate_hash, str) or not self.store.get_bundle_by_hash(candidate_hash):
            raise LearningRuntimeError("candidate bundle is not readable after restart")
        return candidate


__all__ = ["LearningRuntime", "LearningRuntimeError", "StoreModelObservationSink"]
