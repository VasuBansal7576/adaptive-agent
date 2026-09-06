"""Runtime construction for learning from completed durable development runs.

The runtime reads the existing Store after a run has completed and exposes only
public documentation, redacted broker observations, and a coarse trusted
outcome marker to ``LearningService``.  It never evaluates a run, promotes a
candidate, or treats model text as an outcome.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .learning import LearningProposal, LearningService, PlannerLearningAdapter
from .learning_store import CandidateManagerLearningAdapter, DurableLearningSourceAdapter, LearningStoreError
from .retrieval import AccessFilteredRetriever, InMemorySourceProvider, canonical_json, content_hash


class LearningRuntimeError(ValueError):
    """Raised when a completed durable run cannot safely become learning input."""


def _artifact_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return canonical_json(value)


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

    def _materialize_run_records(self, *, environment_id: str, run_id: str) -> None:
        save = getattr(self.store, "save_learning_record", None)
        if not callable(save):
            raise LearningRuntimeError("Store lacks save_learning_record")
        environment = self.store.get_environment(environment_id)
        if not isinstance(environment, Mapping):
            raise LearningRuntimeError("completed run environment is not stored")
        manifest_ref = environment.get("manifest_ref")
        if not isinstance(manifest_ref, str):
            raise LearningRuntimeError("completed run public manifest reference is missing")
        manifest = self.store.get_artifact(json.loads(manifest_ref)["sha256"])
        for doc in manifest.get("docs", []):
            doc_content = _artifact_text(self.store.get_artifact(doc["sha256"]))
            record = {"kind": "public_doc", "sourceId": doc["id"], "content": doc_content, "contentHash": content_hash(doc_content), "environmentId": environment_id, "visibility": "public"}
            save(f"learning-doc-{doc['sha256']}", environment_id, run_id, json.dumps(record, sort_keys=True, separators=(",", ":")))

        trusted_outcome = self.store.get_outcome_by_run_id(run_id)
        if not isinstance(trusted_outcome, Mapping) or not trusted_outcome.get("passed"):
            raise LearningRuntimeError("completed development run lacks a trusted passing outcome")
        for event in self.store.list_evidence(run_id):
            provenance = self.store.evidence_provenance(event["evidence_id"])
            if not provenance or provenance.get("environment_id") != environment_id or provenance.get("partition") != "development":
                continue
            if event.get("trust_class") not in {"broker", "system"}:
                continue
            # Some session-2 runtimes currently retain broker events as
            # operator-only.  Project only the broker's safe envelope here;
            # never copy its raw payload into learner context.
            if event.get("visibility") not in {"learner", "operator"} or event.get("event_type") != "tool_result":
                continue
            ref = json.loads(event["source_ref"])
            payload = self.store.get_artifact(ref["sha256"])
            # Only retain safe broker metadata, never raw fixture/evaluator text.
            safe = {key: payload.get(key) for key in ("status", "effect", "toolVersion") if isinstance(payload, Mapping) and key in payload}
            content = f"Broker development observation: eventType={event['event_type']}; details={canonical_json(safe)}"
            record = {"kind": "live_evidence", "sourceId": event["evidence_id"], "content": content, "contentHash": content_hash(content), "sourceContentHash": event["content_hash"], "environmentId": environment_id, "runId": run_id, "partition": "development", "visibility": "learner", "trustClass": event["trust_class"], "trustedOutcome": True}
            save(f"learning-evidence-{event['evidence_id']}", environment_id, run_id, json.dumps(record, sort_keys=True, separators=(",", ":")))
        outcome_content = "A trusted evaluator outcome is recorded for this completed development run."
        outcome_record = {"kind": "task_state", "sourceId": f"outcome:{run_id}", "content": outcome_content, "contentHash": content_hash(outcome_content), "environmentId": environment_id, "runId": run_id, "visibility": "learner", "trustedOutcome": True}
        save(f"learning-outcome-{run_id}", environment_id, run_id, json.dumps(outcome_record, sort_keys=True, separators=(",", ":")))

    def propose_completed_run(self, run_id: str, *, goal: str | None = None, feedback: Mapping[str, Any] | None = None) -> LearningProposal:
        stored = self.store.get_run(run_id)
        if not isinstance(stored, Mapping) or stored.get("status") != "succeeded":
            raise LearningRuntimeError("learning requires a completed successful development run")
        environment_id = stored.get("environment_id")
        if not isinstance(environment_id, str):
            raise LearningRuntimeError("completed run environment binding is missing")
        task = self.store.get_task(stored["task_id"])
        if not isinstance(task, Mapping) or task.get("partition") != "development":
            raise LearningRuntimeError("learning requires a DEVELOPMENT task")
        self._materialize_run_records(environment_id=environment_id, run_id=run_id)
        environment = {"environmentId": environment_id, "version": task.get("version", "1")}
        sink = StoreModelObservationSink(self.store, run_id)
        self.service.retriever = self.source_adapter.retriever(environment_id=environment_id, run_id=run_id)
        self.service.model_runner = PlannerLearningAdapter(self.service.model_runner.client, sink)
        return self.service.propose(run_id=run_id, environment_id=environment_id, goal=goal or task["goal"], environment=environment, feedback=feedback or {"status": "succeeded"}, remaining_deadline=self.wall_seconds, token_cap=self.token_budget, max_repair_attempts=1)

    def reload_candidate(self, candidate_id: str) -> Mapping[str, Any]:
        candidate = self.store.get_candidate(candidate_id)
        if not isinstance(candidate, Mapping):
            raise LearningRuntimeError("candidate is not readable after restart")
        candidate_hash = candidate.get("candidate_bundle_hash")
        if not isinstance(candidate_hash, str) or not self.store.get_bundle_by_hash(candidate_hash):
            raise LearningRuntimeError("candidate bundle is not readable after restart")
        return candidate


__all__ = ["LearningRuntime", "LearningRuntimeError", "StoreModelObservationSink"]
