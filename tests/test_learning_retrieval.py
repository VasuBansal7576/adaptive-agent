import hashlib
import json
import unittest

from adaptive_agent.learning import LearningError, LearningService, PlannerLearningAdapter
from adaptive_agent.retrieval import AccessFilteredRetriever, InMemorySourceProvider, RetrievalError, SourceKind, SourceRecord, content_hash


def source(source_id, kind, text, **kwargs):
    return SourceRecord(source_id, kind, text, content_hash(text), **kwargs)


class Invocation:
    provider = "openai-codex"
    model = "openai-codex/gpt-5.6-luna"
    response_id = "resp-learning-1"
    usage = {"inputTokens": 10, "outputTokens": 20}

    def __init__(self, payload):
        self.text = json.dumps(payload)


class Sink:
    def __init__(self):
        self.payloads = []

    def create_candidate(self, payload):
        self.payloads.append(dict(payload))
        return {"candidateId": "candidate-from-authority", "state": "validated", **payload}


class PlannerClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    def invoke(self, *, goal, environment, messages):
        self.messages = messages
        return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": "planner-response", "text": json.dumps(self.payload), "usage": {"totalTokens": 20}}


class EvidenceSink:
    def __init__(self):
        self.calls = []

    def record_model_observation(self, evidence, *, trusted_parent=False):
        self.calls.append((dict(evidence), trusted_parent))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.sources = [
            source("doc-public", SourceKind.PUBLIC_DOC, "recheck the current version before a write", environment_id="env", trust_class="operator"),
            source("ev-dev", SourceKind.LIVE_EVIDENCE, "broker observed a version conflict", environment_id="env", run_id="run-a", partition="development", trust_class="broker"),
            source("ev-final", SourceKind.LIVE_EVIDENCE, "recheck the current version before a write", environment_id="env", run_id="run-a", partition="final", trust_class="evaluator"),
            source("ev-other-run", SourceKind.LIVE_EVIDENCE, "recheck the current version before a write", environment_id="env", run_id="run-b", partition="development", trust_class="broker"),
            source("ev-hidden", SourceKind.LIVE_EVIDENCE, "recheck the current version before a write", environment_id="env", run_id="run-a", partition="development", trust_class="evaluator", visibility="evaluator_only"),
            source("state", SourceKind.TASK_STATE, "the operation is awaiting reconciliation", environment_id="env", run_id="run-a", trust_class="system"),
            source("skill-active", SourceKind.SKILL, "recheck live version after a conflict", environment_id="env", trust_class="system", metadata={"bundleState": "active", "authority": False}),
        ]
        self.retriever = AccessFilteredRetriever(InMemorySourceProvider(self.sources))

    def test_filter_precedes_ranking_and_keeps_sources_separate(self):
        result = self.retriever.search("recheck current version", environment_id="env", run_id="run-a")
        self.assertEqual([item.source_id for item in result.evidence], ["ev-dev"])
        self.assertEqual([item.source_id for item in result.docs], ["doc-public"])
        self.assertEqual([item.source_id for item in result.task_state], [])
        self.assertEqual([item.source_id for item in result.skills], ["skill-active"])
        self.assertTrue(result.evidence[0].citation.content_hash == content_hash("broker observed a version conflict"))

    def test_hash_and_development_evidence_are_verified(self):
        with self.assertRaises(RetrievalError):
            SourceRecord("bad", SourceKind.PUBLIC_DOC, "text", "0" * 64)
        with self.assertRaisesRegex(RetrievalError, "does not exist"):
            self.retriever.require_development_evidence(["ev-final"], environment_id="env", run_id="run-a")
        with self.assertRaisesRegex(RetrievalError, "does not exist"):
            self.retriever.require_development_evidence(["not-real"], environment_id="env", run_id="run-a")


class LearningTests(unittest.TestCase):
    def setUp(self):
        evidence = source("ev-dev", SourceKind.LIVE_EVIDENCE, "broker observed a version conflict", environment_id="env", run_id="run-a", partition="development", trust_class="broker")
        docs = source("doc", SourceKind.PUBLIC_DOC, "recheck the current version before a write", environment_id="env", trust_class="operator")
        self.retriever = AccessFilteredRetriever(InMemorySourceProvider([evidence, docs]))
        self.sink = Sink()
        self.seen = {}

    def runner_for(self, payload):
        def runner(*, goal, environment, emit):
            self.seen["goal"] = goal
            self.seen["environment"] = environment
            return Invocation(payload)
        return runner

    def service(self, payload):
        return LearningService(self.retriever, self.runner_for(payload), self.sink, lambda: "a" * 64)

    def valid_payload(self):
        return {
            "predictedEffect": "reduce stale updates",
            "editOperations": [{"path": "skills/generic-retry/procedure", "operation": "replace", "value": "Re-read current state after a version conflict before proposing a retry."}],
            "supportingEvidenceIds": ["ev-dev"],
            "proposerVersion": "model-proposal-1",
            "skill": {"procedure": "When a tool reports a version conflict, re-read live state before retrying.", "preconditions": ["the broker has authorized the read"]},
        }

    def test_model_proposal_is_bounded_cited_and_staged_only(self):
        result = self.service(self.valid_payload()).propose(run_id="run-a", environment_id="env", goal="handle a version conflict", feedback={"passed": True, "expectedAnswer": "secret", "status": "failed", "diagnostic": "version conflict"}, environment={"environmentId": "env", "version": "1", "evaluatorRef": {"id": "hidden"}, "toolSchemas": []})
        self.assertEqual(result.base_bundle_hash, "a" * 64)
        self.assertEqual(result.candidate_payload["supportingEvidenceIds"], ["ev-dev"])
        self.assertRegex(result.candidate_payload["changedArtifactHashes"][0], r"^[0-9a-f]{64}$")
        self.assertEqual(result.authoritative_candidate["state"], "validated")
        self.assertNotIn("evaluatorRef", self.seen["environment"])  # learner sees no trusted control reference
        self.assertNotIn("passed", self.seen["environment"]["sanitizedFeedback"])
        self.assertEqual(len(self.sink.payloads), 1)

    def test_fabricated_evidence_is_rejected_before_authoritative_sink(self):
        payload = self.valid_payload()
        payload["supportingEvidenceIds"] = ["fabricated"]
        with self.assertRaisesRegex(LearningError, "does not exist"):
            self.service(payload).propose(run_id="run-a", environment_id="env", goal="learn", environment={})
        self.assertEqual(self.sink.payloads, [])

    def test_declared_hash_mismatch_and_literal_fixture_workflow_are_rejected(self):
        payload = self.valid_payload()
        payload["changedArtifactHashes"] = ["0" * 64]
        with self.assertRaisesRegex(LearningError, "does not match"):
            self.service(payload).propose(run_id="run-a", environment_id="env", goal="learn", environment={})
        payload = self.valid_payload()
        payload["skill"]["procedure"] = "Use INV-DEV-000 and call_tool('update_record') in this exact sequence."
        with self.assertRaises(LearningError):
            self.service(payload).propose(run_id="run-a", environment_id="env", goal="learn", environment={})

    def test_stale_base_is_rejected(self):
        with self.assertRaisesRegex(LearningError, "pinned active"):
            self.service(self.valid_payload()).propose(run_id="run-a", environment_id="env", goal="learn", base_bundle_hash="b" * 64, environment={})

    def test_session2_planner_client_adapter_records_trusted_observation(self):
        client = PlannerClient(self.valid_payload())
        evidence_sink = EvidenceSink()
        service = LearningService(self.retriever, PlannerLearningAdapter(client, evidence_sink), self.sink, lambda: "a" * 64)
        service.propose(run_id="run-a", environment_id="env", goal="learn", environment={})
        self.assertTrue(evidence_sink.calls[0][1])
        self.assertEqual(client.messages[0]["role"], "system")
        self.assertIn("bounded learning patch", client.messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
