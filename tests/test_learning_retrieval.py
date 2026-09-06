import hashlib
import json
import unittest
from threading import Event

from adaptive_agent.learning import LearningError, LearningService, PlannerLearningAdapter, PROPOSAL_CONTRACT
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

    def persist_candidate_patch(self, patch_bytes, content_hash):
        self.patch_bytes = bytes(patch_bytes)
        self.patch_hash = hashlib.sha256(self.patch_bytes).hexdigest()
        return {"sha256": self.patch_hash, "size": len(self.patch_bytes), "stored": True, "immutable": True}


class PlannerClient:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    def invoke(self, *, goal, environment, messages, **kwargs):
        self.messages = messages
        self.forwarded = kwargs
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
        self.assertNotIn("ev-final", result.source_ids)
        self.assertNotIn("ev-hidden", result.source_ids)
        self.assertNotIn("ev-other-run", result.source_ids)

    def test_operator_visibility_and_string_boolean_are_not_learner_inputs(self):
        operator_doc = {"sourceId": "operator-doc", "kind": "public_doc", "content": "private operator note", "contentHash": content_hash("private operator note"), "visibility": "operator", "trustClass": "operator", "verified": True, "active": True}
        with self.assertRaisesRegex(RetrievalError, "boolean"):
            SourceRecord.from_mapping({**operator_doc, "verified": "false"})
        retriever = AccessFilteredRetriever(InMemorySourceProvider([operator_doc]))
        self.assertEqual(retriever.search("private operator note", environment_id="env", run_id="run-a").docs, ())

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
            "editOperations": [
                {"path": "skills/generic-retry/procedure", "operation": "replace", "value": "When a tool reports a version conflict, re-read live state before retrying."},
                {"path": "skills/generic-retry/preconditions", "operation": "replace", "value": ["the broker has authorized the read"]},
            ],
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
        self.assertEqual(self.seen["environment"]["proposalContract"]["type"], PROPOSAL_CONTRACT["type"])
        self.assertEqual(self.seen["environment"]["proposalContract"]["properties"]["supportingEvidenceIds"]["items"], {"enum": ["ev-dev"]})
        self.assertEqual(self.seen["environment"]["proposalLimits"]["maxPatchBytes"], 32_768)
        self.assertEqual(len(self.sink.payloads), 1)
        self.assertEqual(self.sink.patch_hash, result.candidate_payload["changedArtifactHashes"][0])
        self.assertEqual(hashlib.sha256(result.patch_bytes).hexdigest(), result.candidate_payload["changedArtifactHashes"][0])

    def test_learner_claims_and_hidden_evaluator_feedback_never_reach_model(self):
        feedback = {
            "passed": True,
            "score": 1.0,
            "learnerClaim": "the candidate is correct",
            "evaluatorTrace": "hidden evaluator answer and secret fixture",
            "status": "failed",
            "diagnostic": "hidden evaluator trace: expected answer is withheld",
        }
        self.service(self.valid_payload()).propose(run_id="run-a", environment_id="env", goal="learn", feedback=feedback, environment={})
        sanitized = self.seen["environment"]["sanitizedFeedback"]
        self.assertEqual(sanitized, {"status": "failed"})
        rendered = json.dumps(self.seen["environment"], sort_keys=True)
        self.assertNotIn("learnerClaim", rendered)
        self.assertNotIn("hidden evaluator", rendered.casefold())
        self.assertNotIn("expected answer", rendered.casefold())

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

    def test_actual_value_lines_and_exact_skill_binding_are_bounded(self):
        payload = self.valid_payload()
        payload["editOperations"][0]["value"] = "line\n" * 201
        payload["skill"]["procedure"] = payload["editOperations"][0]["value"]
        with self.assertRaisesRegex(LearningError, "200-line"):
            self.service(payload).propose(run_id="run-a", environment_id="env", goal="learn", environment={})
        payload = self.valid_payload()
        payload["editOperations"][0]["value"] = "A generic but different applied value."
        with self.assertRaisesRegex(LearningError, "exact values"):
            self.service(payload).propose(run_id="run-a", environment_id="env", goal="learn", environment={})
        payload = self.valid_payload()
        payload["editOperations"][0]["path"] = "skills/generic-retry/../procedure"
        with self.assertRaisesRegex(LearningError, "unsafe segment"):
            self.service(payload).propose(run_id="run-a", environment_id="env", goal="learn", environment={})

    def test_metadata_only_patch_attestation_is_rejected(self):
        class MetadataOnlySink(Sink):
            def persist_candidate_patch(self, patch_bytes, content_hash):
                return {"sha256": content_hash, "size": len(patch_bytes)}

        service = LearningService(self.retriever, self.runner_for(self.valid_payload()), MetadataOnlySink(), lambda: "a" * 64)
        with self.assertRaisesRegex(LearningError, "exact persisted patch"):
            service.propose(run_id="run-a", environment_id="env", goal="learn", environment={})

    def test_legitimate_learned_tool_orchestration_is_allowed(self):
        payload = self.valid_payload()
        procedure = "Read current state with call_tool('inventory.read'), then use the returned version for the bounded update."
        payload["editOperations"][0]["value"] = procedure
        payload["skill"]["procedure"] = procedure
        result = self.service(payload).propose(run_id="run-a", environment_id="env", goal="learn", environment={})
        self.assertIn("call_tool", result.bundle_patch["skill"]["procedure"])

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

    def test_planner_budget_deadline_and_cancel_are_forwarded(self):
        client = PlannerClient(self.valid_payload())
        service = LearningService(self.retriever, PlannerLearningAdapter(client, EvidenceSink()), self.sink, lambda: "a" * 64)
        cancel = Event()
        service.propose(run_id="run-a", environment_id="env", goal="learn", environment={}, remaining_deadline=12.5, cancel=cancel, token_cap=321)
        self.assertAlmostEqual(client.forwarded["remaining_deadline"], 12.5, delta=0.1)
        self.assertIs(client.forwarded["cancel"], cancel)
        self.assertEqual(client.forwarded["token_cap"], 321)

    def test_model_usage_over_token_cap_is_rejected(self):
        class OverCapInvocation(Invocation):
            usage = {"totalTokens": 11}

        def runner(*, goal, environment, emit):
            return OverCapInvocation(self.valid_payload())

        service = LearningService(self.retriever, runner, self.sink, lambda: "a" * 64)
        with self.assertRaisesRegex(LearningError, "token cap"):
            service.propose(run_id="run-a", environment_id="env", goal="learn", environment={}, token_cap=10)

    def test_contract_enumerates_only_retrieved_development_evidence(self):
        service = self.service(self.valid_payload())
        service.propose(run_id="run-a", environment_id="env", goal="learn", environment={})
        contract = self.seen["environment"]["proposalContract"]
        self.assertEqual(contract["properties"]["supportingEvidenceIds"]["items"], {"enum": ["ev-dev"]})

    def test_malformed_proposal_repair_reuses_deadline_and_remaining_token_ledger(self):
        calls = []
        payload = self.valid_payload()

        class RepairRunner:
            def __call__(self, *, goal, environment, emit, remaining_deadline=None, cancel=None, token_cap=None):
                calls.append((environment["sanitizedFeedback"], remaining_deadline, token_cap))
                if len(calls) == 1:
                    return Invocation({"unexpected": True})
                return Invocation(payload)

        service = LearningService(self.retriever, RepairRunner(), self.sink, lambda: "a" * 64)
        result = service.propose(run_id="run-a", environment_id="env", goal="learn", environment={}, remaining_deadline=12.5, token_cap=50, max_repair_attempts=1)
        self.assertEqual(result.authoritative_candidate["state"], "validated")
        self.assertAlmostEqual(calls[0][1], 12.5, delta=0.1)
        self.assertLess(calls[1][1], calls[0][1])
        self.assertEqual(calls[0][2], 50)
        self.assertEqual(calls[1][2], 30)
        self.assertEqual(calls[1][0]["failureClass"], "malformed_proposal")


if __name__ == "__main__":
    unittest.main()
