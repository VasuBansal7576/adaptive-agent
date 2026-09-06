"""Runtime construction for learning from completed durable development runs.

The runtime reads the existing Store after a run has completed and exposes only
public documentation, redacted broker observations, and a coarse trusted
outcome marker to ``LearningService``.  It never evaluates a run, promotes a
candidate, or treats model text as an outcome.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .learning import LearningProposal, LearningService, PlannerLearningAdapter
from .learning_store import CandidateManagerLearningAdapter, DurableLearningSourceAdapter, LearningStoreError
from .learning_projection import DurableBrokerLearningProjection
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
        payload = {"provider": evidence.get("provider"), "model": evidence.get("model"), "responseId": evidence["responseId"], "usage": dict(evidence["usage"])}
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
        raw_records: list[Mapping[str, Any]] = []

        def persist(record_id: str, record: Mapping[str, Any]) -> None:
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if callable(save):
                save(record_id, environment_id, run_id, encoded)
            else:
                raw_records.append(record)
        environment = self.store.get_environment(environment_id)
        if not isinstance(environment, Mapping):
            raise LearningRuntimeError("completed run environment is not stored")
        public_doc_reader = getattr(self.store, "get_public_docs", None)
        manifest_ref = environment.get("manifest_ref")
        manifest_docs: list[Mapping[str, Any]] = []
        if isinstance(manifest_ref, str):
            manifest = self.store.get_artifact(json.loads(manifest_ref)["sha256"])
            manifest_docs = [doc for doc in manifest.get("docs", []) if isinstance(doc, Mapping)]
        if callable(public_doc_reader):
            docs = public_doc_reader(environment_id)
        else:
            if not isinstance(manifest_ref, str):
                raise LearningRuntimeError("completed run public manifest reference is missing")
            docs = [{**doc, "content": self.store.get_artifact(doc["sha256"])} for doc in manifest.get("docs", [])]
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
            if not isinstance(doc, Mapping) or not isinstance(doc.get("id"), str):
                raise LearningRuntimeError("public document projection is malformed")
            if "content" in doc:
                stored_doc = doc["content"]
            else:
                candidate = supplied_docs.get(doc["id"])
                if candidate is None:
                    raise LearningRuntimeError(f"public document content is not stored: {doc['id']}")
                ref = self.store.put_artifact(candidate)
                if ref.sha256 != doc.get("sha256"):
                    raise LearningRuntimeError(f"public document hash mismatch: {doc['id']}")
                stored_doc = self.store.get_artifact(doc["sha256"])
            doc_content = _artifact_text(stored_doc)
            record = {"kind": "public_doc", "sourceId": doc["id"], "content": doc_content, "contentHash": content_hash(doc_content), "environmentId": environment_id, "visibility": "public"}
            persist(f"learning-doc-{doc.get('sha256', content_hash(doc_content))}", record)

        trusted_outcome = self.store.get_outcome_by_run_id(run_id)
        if not isinstance(trusted_outcome, Mapping) or "passed" not in trusted_outcome or not isinstance(trusted_outcome.get("passed"), (bool, int)):
            raise LearningRuntimeError("completed development run lacks a trusted evaluator outcome")
        outcome_passed = bool(trusted_outcome["passed"])
        run_row = self.store.get_run(run_id)
        projected = DurableBrokerLearningProjection(self.store).project(
            environment_id=environment_id,
            run_id=run_id,
            task_id=run_row["task_id"] if isinstance(run_row, Mapping) else "",
        )
        for record_id, record in projected:
            persist(record_id, record)
        joined_reader = getattr(self.store, "list_learning_evidence", None) or getattr(self.store, "list_learner_evidence", None)
        events = joined_reader(environment_id=environment_id, run_id=run_id) if callable(joined_reader) else self.store.list_evidence(run_id)
        for event in events if not projected else ():
            if callable(joined_reader):
                if event.get("partition") != "development" or event.get("environment_id") != environment_id or event.get("run_id") != run_id:
                    continue
                if event.get("visibility") != "learner" or event.get("redacted") != 1:
                    continue
            else:
                provenance = self.store.evidence_provenance(event["evidence_id"])
                if not provenance or provenance.get("environment_id") != environment_id or provenance.get("partition") != "development":
                    continue
                if event.get("trust_class") not in {"broker", "system"} or event.get("visibility") not in {"learner", "operator"}:
                    continue
            # Some session-2 runtimes currently retain broker events as
            # operator-only.  Project only the broker's safe envelope here;
            # never copy its raw payload into learner context.
            if event.get("event_type") != "tool_result":
                continue
            payload: Mapping[str, Any] = {}
            source_event = getattr(self.store, "get_evidence", lambda _id: None)(event["evidence_id"])
            if isinstance(source_event, Mapping) and isinstance(source_event.get("source_ref"), str):
                ref = json.loads(source_event["source_ref"])
                raw_payload = self.store.get_artifact(ref["sha256"])
                if isinstance(raw_payload, Mapping):
                    payload = raw_payload
            safe: dict[str, Any] = {key: payload.get(key) for key in ("status", "effect", "toolVersion") if key in payload}
            # A narrow Store join may already return these allowlisted fields;
            # consume that projection without reopening raw operator evidence.
            for key in ("tool", "input", "result", "error", "callId", "toolVersion", "status", "effect"):
                if key in event:
                    safe[key] = _sanitize_learning_value(event[key])
            call_id = payload.get("callId")
            call_reader = getattr(self.store, "get_tool_call", None)
            call = call_reader(call_id) if callable(call_reader) and isinstance(call_id, str) else None
            if isinstance(call, Mapping) and call.get("run_id") == run_id and call.get("environment_id") == environment_id:
                safe["tool"] = call.get("tool")
                try:
                    safe["input"] = _sanitize_learning_value(json.loads(call.get("arguments_json", "{}")))
                except (TypeError, json.JSONDecodeError):
                    pass
                try:
                    result = json.loads(call.get("result_json")) if call.get("result_json") else {}
                except (TypeError, json.JSONDecodeError):
                    result = {}
                if isinstance(result, Mapping):
                    safe["result"] = _sanitize_learning_value({key: result.get(key) for key in ("status", "effect", "output") if key in result})
                    error = result.get("error")
                    if isinstance(error, Mapping):
                        safe["error"] = {key: _sanitize_learning_value(error.get(key)) for key in ("code", "retry") if key in error}
            content = f"Broker development observation: eventType={event['event_type']}; details={canonical_json(safe)}"
            record = {"kind": "live_evidence", "sourceId": event["evidence_id"], "content": content, "contentHash": content_hash(content), "sourceContentHash": event["content_hash"], "environmentId": environment_id, "runId": run_id, "partition": "development", "visibility": "learner", "trustClass": event.get("trust_class", "broker"), "trustedOutcome": True, "outcomePassed": outcome_passed}
            persist(f"learning-evidence-{event['evidence_id']}", record)
        outcome_content = f"A trusted evaluator outcome is recorded for this completed development run; passed={str(outcome_passed).lower()}."
        outcome_record = {"kind": "task_state", "sourceId": f"outcome:{run_id}", "content": outcome_content, "contentHash": content_hash(outcome_content), "environmentId": environment_id, "runId": run_id, "visibility": "learner", "trustedOutcome": True, "outcomePassed": outcome_passed}
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
