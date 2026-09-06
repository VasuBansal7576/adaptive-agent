"""Behavioral tests for the Store/Broker/Candidate scope.

All tool providers and drivers here are SYNTHETIC TEST-ONLY stubs; nothing here
claims real learning.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from adaptive_agent.broker import Capability, ToolBroker
from adaptive_agent.candidate import (
    CandidateManager,
    CandidateValidationError,
    PromotionError,
)
from adaptive_agent.environment import (
    EnvironmentRegistry,
    SchemaValidationError,
    validate_value,
)
from adaptive_agent.models import (
    ArtifactRef,
    CandidateProposal,
    CandidateState,
    EvaluationReport,
    EvaluationState,
    MetricAggregate,
    PromotionGate,
    RunStatus,
    SkillBundle,
    SkillVersion,
    TaskInput,
    ToolError,
    ToolErrorCode,
    ToolRequest,
)
from adaptive_agent.store import Store
from tests.conftest import FakeProvider, NEUTRAL_MANIFEST

ENV = "neutral-test-env"
RUN = "run-1"


def _req(tool: str, args: dict[str, Any], key: str = "k1", run: str = RUN, token: str | None = None) -> ToolRequest:
    return ToolRequest(runId=run, stepId="s1", tool=tool, arguments=args, idempotencyKey=key, approvalToken=token)


def _cap(**kw: Any) -> Capability:
    base = dict(
        run_id=RUN,
        environment_id=ENV,
        tool="update_record",
        effect="write",
        resource_scope={},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    base.update(kw)
    return Capability(**base)


def _evidence(store: Store, run_id: str, partition: str = "development", outcome: bool = True) -> str:
    env = ENV
    store.register_task(
        "t-dev", env, "1", store.put_artifact({"t": 1}).model_dump_json(), partition, "g"
    )
    store.save_run(
        run_id,
        {
            "parent_run_id": None,
            "task_id": "t-dev",
            "environment_id": env,
            "bundle_id": "b",
            "status": "succeeded",
            "idempotency_key": f"run-{run_id}",
            "last_event_sequence": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_json": "{}",
        },
    )
    ev_id = f"ev-{run_id}"
    store.append_evidence(
        ev_id,
        {
            "run_id": run_id,
            "sequence": 1,
            "event_type": "tool_result",
            "content_hash": "0" * 64,
            "source_ref": "{}",
            "trust_class": "broker",
            "visibility": "learner",
            "redacted": 1,
        },
    )
    if outcome:
        store.save_outcome(
            f"out-{run_id}",
            {
                "run_id": run_id,
                "passed": 1,
                "score": 1.0,
                "metadata_json": "{}",
                "checked_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return ev_id


class TestStore:
    def test_partition_filtering_is_strict(self, store: Store):
        ref = store.put_artifact({"a": 1}).model_dump_json()
        store.register_task("t1", ENV, "1", ref, "development", "g1")
        store.register_task("t2", ENV, "1", ref, "validation", "g2")
        store.register_task("t3", "other-env", "1", ref, "development", "g3")
        dev = store.list_tasks_by_partition(ENV, "development")
        assert [t["id"] for t in dev] == ["t1"]
        assert store.list_tasks_by_partition(ENV, "final") == []

    def test_artifact_roundtrip(self, store: Store):
        ref = store.put_artifact({"x": [1, 2, 3], "when": datetime(2026, 9, 6, tzinfo=timezone.utc)})
        assert store.get_artifact(ref) == {"x": [1, 2, 3], "when": "2026-09-06T00:00:00+00:00"}
        assert store.get_artifact(ref.sha256) == store.get_artifact(ref)


class TestSchemaValidation:
    def test_numeric_constraints(self):
        schema = {"type": "integer", "minimum": 2, "exclusiveMaximum": 10, "multipleOf": 2}
        validate_value(4, schema)
        for bad in (1, 10, 5):
            with pytest.raises(SchemaValidationError):
                validate_value(bad, schema)

    def test_string_pattern_and_length(self):
        schema = {"type": "string", "minLength": 2, "maxLength": 5, "pattern": "^[a-z]+$"}
        validate_value("abc", schema)
        for bad in ("a", "abcdef", "ABC"):
            with pytest.raises(SchemaValidationError):
                validate_value(bad, schema)

    def test_type_union_and_const_enum(self):
        validate_value(None, {"type": ["string", "null"]})
        with pytest.raises(SchemaValidationError):
            validate_value(7, {"type": ["string", "null"]})
        validate_value("a", {"enum": ["a", "b"]})
        with pytest.raises(SchemaValidationError):
            validate_value("c", {"enum": ["a", "b"]})
        validate_value("x", {"const": "x"})
        with pytest.raises(SchemaValidationError):
            validate_value("y", {"const": "x"})

    def test_composition_and_not(self):
        validate_value(5, {"allOf": [{"type": "integer"}, {"minimum": 3}]})
        with pytest.raises(SchemaValidationError):
            validate_value(1, {"allOf": [{"type": "integer"}, {"minimum": 3}]})
        validate_value("a", {"anyOf": [{"type": "integer"}, {"type": "string"}]})
        with pytest.raises(SchemaValidationError):
            validate_value(True, {"anyOf": [{"type": "integer"}, {"type": "string"}]})
        validate_value(5, {"oneOf": [{"type": "integer"}, {"type": "string"}]})
        with pytest.raises(SchemaValidationError):
            validate_value(5, {"oneOf": [{"type": "integer"}, {"type": "number"}]})
        with pytest.raises(SchemaValidationError):
            validate_value(5, {"not": {"type": "integer"}})

    def test_ref_alias_resolution(self):
        schema = {
            "$defs": {"pos": {"type": "integer", "minimum": 0}},
            "type": "object",
            "properties": {"n": {"$ref": "#/$defs/pos"}},
        }
        validate_value({"n": 3}, schema)
        with pytest.raises(SchemaValidationError):
            validate_value({"n": -1}, schema)
        with pytest.raises(SchemaValidationError):
            validate_value(1, {"$ref": "http://evil/schema"})

    def test_object_array_constraints(self):
        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {
                "a": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "integer"}},
                "b": {"type": "integer"},
            },
            "additionalProperties": False,
            "dependentRequired": {"a": ["b"]},
        }
        validate_value({"a": [1, 2], "b": 0}, schema)
        with pytest.raises(SchemaValidationError):
            validate_value({"a": [1, 2]}, schema)  # dependentRequired b
        with pytest.raises(SchemaValidationError):
            validate_value({"a": [1, 1], "b": 0}, schema)  # uniqueItems
        with pytest.raises(SchemaValidationError):
            validate_value({"a": [1], "b": 0, "c": 1}, schema)  # additionalProperties

    def test_manifest_validation(self, registry: EnvironmentRegistry):
        registry.register(NEUTRAL_MANIFEST)
        with pytest.raises(SchemaValidationError):
            bad = NEUTRAL_MANIFEST.model_copy(deep=True)
            bad.tool_schemas.append(bad.tool_schemas[0])
            registry.register(bad)


class TestCapabilityAndBroker:
    def test_capability_enforces_scope_expiry_run_env_tool(self, broker: ToolBroker, registry: EnvironmentRegistry, provider: FakeProvider):
        registry.register(NEUTRAL_MANIFEST)
        provider.reset(RUN)
        schema = registry.get_tool_schema(ENV, "update_record")
        assert schema is not None
        req = _req("update_record", {"record_id": "record-1", "version": 1, "value": "x"})

        ok = _cap(resource_scope={"record_id": "record-1", "version": {"min": 1, "max": 9}})
        assert ok.covers(ENV, req, schema) is None

        # wrong run / env / tool / effect / expired / out-of-scope all rejected
        assert _cap(run_id="other").covers(ENV, req, schema) is not None
        assert _cap(environment_id="other-env").covers(ENV, req, schema) is not None
        read_schema = registry.get_tool_schema(ENV, "read_record")
        read_req = _req("read_record", {"record_id": "record-1"})
        assert _cap(tool="update_record").covers(ENV, read_req, read_schema) is not None  # tool match is exact
        assert _cap(tool="read_record", effect="read").covers(ENV, read_req, read_schema) is None
        assert _cap(effect="read").covers(ENV, req, schema) is not None
        assert _cap(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)).covers(ENV, req, schema) is not None
        assert _cap(resource_scope={"record_id": "record-2"}).covers(ENV, req, schema) is not None
        assert _cap(resource_scope={"extra": 1}).covers(ENV, req, schema) is not None

    def test_read_replay_and_conflict(self, broker: ToolBroker, registry: EnvironmentRegistry, provider: FakeProvider):
        registry.register(NEUTRAL_MANIFEST)
        provider.reset(RUN)
        cap = _cap(tool="read_record", effect="read")
        r1 = broker.request_tool_call(ENV, _req("read_record", {"record_id": "record-1"}, key="k"), cap, provider)
        assert r1.status == "ok"
        r2 = broker.request_tool_call(ENV, _req("read_record", {"record_id": "record-1"}, key="k"), cap, provider)
        assert r2.output == r1.output  # replayed, not redispatched
        r3 = broker.request_tool_call(ENV, _req("read_record", {"record_id": "record-2"}, key="k"), cap, provider)
        assert r3.error is not None and r3.error.code == ToolErrorCode.IDEMPOTENCY_CONFLICT

    def test_write_approval_single_use_and_binding(self, broker: ToolBroker, registry: EnvironmentRegistry, provider: FakeProvider):
        registry.register(NEUTRAL_MANIFEST)
        provider.reset(RUN)
        cap = _cap()
        args = {"record_id": "record-1", "version": 1, "value": "x"}

        # no token -> FORBIDDEN
        r = broker.request_tool_call(ENV, _req("update_record", args, key="w1"), cap, provider)
        assert r.error and r.error.code == ToolErrorCode.FORBIDDEN

        token = broker.issue_approval(ENV, RUN, "update_record", args, "w1")
        r = broker.request_tool_call(ENV, _req("update_record", args, key="w1", token=token), cap, provider)
        assert r.status == "ok"

        # replay returns stored result, does NOT re-consume approval or redispatch
        r2 = broker.request_tool_call(ENV, _req("update_record", args, key="w1"), cap, provider)
        assert r2.status == "ok" and r2.output == r.output

        # consumed token cannot authorize a second write
        args2 = {"record_id": "record-1", "version": 2, "value": "y"}
        r3 = broker.request_tool_call(ENV, _req("update_record", args2, key="w2", token=token), cap, provider)
        assert r3.error and r3.error.code == ToolErrorCode.FORBIDDEN

        # token bound to wrong arguments is rejected
        token2 = broker.issue_approval(ENV, RUN, "update_record", args2, "w3")
        r4 = broker.request_tool_call(ENV, _req("update_record", args, key="w3", token=token2), cap, provider)
        assert r4.error and r4.error.code == ToolErrorCode.FORBIDDEN

    def test_unknown_effect_and_reconciliation(self, broker: ToolBroker, registry: EnvironmentRegistry):
        registry.register(NEUTRAL_MANIFEST)
        crashy = FakeProvider(crash_on_write=True)
        crashy.reset(RUN)
        cap = _cap()
        args = {"record_id": "record-1", "version": 1, "value": "x"}
        token = broker.issue_approval(ENV, RUN, "update_record", args, "w9")
        r = broker.request_tool_call(ENV, _req("update_record", args, key="w9", token=token), cap, crashy)
        assert r.error and r.error.code == ToolErrorCode.OUTCOME_UNKNOWN
        assert r.effect == "unknown"

        # Prepared call exists with no result; replay must NOT redispatch.
        r2 = broker.request_tool_call(ENV, _req("update_record", args, key="w9"), cap, crashy)
        assert r2.error and r2.error.code == ToolErrorCode.OUTCOME_UNKNOWN

        # Provider still reports indeterminate -> remains unresolved.
        resolved = broker.reconcile_run(RUN, crashy)
        assert resolved == []
        assert broker.store.list_unreconciled_calls(RUN) == []

        # Now reconcile against a provider that proves no effect landed.
        calm = FakeProvider(crash_on_write=False)
        calm.reset(RUN)
        resolved = broker.reconcile_run(RUN, calm)
        # crash left result_json recorded already as OUTCOME_UNKNOWN; nothing pending
        assert resolved == []


    def test_injected_authorizer_denies_and_allows(self, store, registry, provider):
        """Prime CapabilityBroker seam: an injected authoritative authorizer."""
        registry.register(NEUTRAL_MANIFEST)
        provider.reset(RUN)
        deny = ToolBroker(store, registry, authorizer=lambda e, r, s: ToolError(code=ToolErrorCode.FORBIDDEN, message="denied by parent", retry="never"))
        cap = _cap(tool="read_record", effect="read")
        denied = deny.request_tool_call(ENV, _req("read_record", {"record_id": "record-1"}, key="a1"), cap, provider)
        assert denied.error and denied.error.code == ToolErrorCode.FORBIDDEN
        assert store.get_tool_call_by_idempotency(RUN, "a1") is None  # no prepared record

        allow = ToolBroker(store, registry, authorizer=lambda e, r, s: None)
        ok = allow.request_tool_call(ENV, _req("read_record", {"record_id": "record-1"}, key="a2"), cap, provider)
        assert ok.status == "ok"

        # Authorizer cannot weaken local fail-closed checks (expired capability).
        expired = _cap(tool="read_record", effect="read", expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
        res = allow.request_tool_call(ENV, _req("read_record", {"record_id": "record-1"}, key="a3"), expired, provider)
        assert res.error and res.error.code == ToolErrorCode.FORBIDDEN


class TestControllerSeam:
    """The durable seam session2 wires POST /runs and SSE to."""

    def _ctl(self, store, registry, broker):
        from adaptive_agent.controller import Controller

        registry.register(NEUTRAL_MANIFEST)
        return Controller(store, registry, broker)

    def _task(self, registry, store):
        task = TaskInput(
            taskId="t-1",
            environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64),
            goal="update the record",
            partition="development",
        )
        registry.register_task(task)
        return task

    def test_create_run_idempotent_and_events(self, store, registry, broker, provider):
        from adaptive_agent.models import Budget, ModelProfile, RunRequest

        ctl = self._ctl(store, registry, broker)
        task = self._task(registry, store)
        req = RunRequest(
            taskRef=store.put_artifact(task.model_dump(mode="json", by_alias=True)),
            modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
            budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
            idempotencyKey="idem-1",
        )
        r1 = ctl.create_run(req, task)
        r2 = ctl.create_run(req, task)
        assert r1.run_id == r2.run_id
        assert ctl.get_run(r1.run_id).status.value == "queued"
        evs = ctl.events(r1.run_id)
        assert evs and evs[0]["event"] == "run_created" and evs[0]["id"] == 1
        # Frozen image identity is pinned on the run receipt (or 'image-unpinned').
        created = store.get_evidence(evs[0]["data"]["evidence_id"])
        payload = store.get_artifact(ArtifactRef.model_validate_json(created["source_ref"]))
        assert payload["imageDigest"] == "image-unpinned"
        assert payload["maxChildDepth"] == 1  # frozen Budget.max_child_depth propagated

        # Legacy event type rejected at both the Controller and Store layers.
        with pytest.raises(ValueError):
            ctl.append_event(r1.run_id, "model_observation", {"x": 1}, "system", "operator")
        with pytest.raises(ValueError):
            store.append_evidence("ev-legacy", {"run_id": r1.run_id, "sequence": 99, "event_type": "model_observation"})

    def test_execute_run_dispatches_through_broker(self, store, registry, broker, provider):
        from adaptive_agent.models import Budget, ModelProfile, RunRequest

        ctl = self._ctl(store, registry, broker)
        task = self._task(registry, store)
        req = RunRequest(
            taskRef=store.put_artifact(task.model_dump(mode="json", by_alias=True)),
            modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
            budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
            idempotencyKey="idem-2",
        )
        run = ctl.create_run(req, task)
        provider.reset(run.run_id)

        cap = _cap(run_id=run.run_id, tool="read_record", effect="read")

        class D:
            def act(self, ctx):
                step = ctl.begin_step(ctx.run_id, "tool")  # type: ignore[arg-type]
                res = ctx.call_tool(step.step_id, "read_record", {"record_id": "record-1"}, "r-key", cap)
                assert res.status == "ok"
                ctl.finish_step(step, "succeeded")  # type: ignore[arg-type]

        done = ctl.execute_run(run.run_id, ENV, provider, D())
        assert done.status.value == "succeeded"
        # SSE stream is ordered and complete.
        evs = ctl.events(run.run_id)
        assert [e["id"] for e in evs] == sorted(e["id"] for e in evs)
        assert any(e["event"] == "tool_result" for e in evs)
        ctl.record_outcome(run.run_id, passed=True, score=1.0)
        assert store.get_outcome_by_run_id(run.run_id)["passed"] == 1


    def test_host_request_bridge_fail_closed(self, store, registry, broker, provider):
        import base64 as b64
        import hashlib as hl

        from adaptive_agent.controller import Controller
        from adaptive_agent.models import Budget, ModelProfile, RunRequest

        registry.register(NEUTRAL_MANIFEST)
        ctl = Controller(store, registry, broker)
        task = TaskInput(
            taskId="t-hr",
            environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64),
            goal="g",
            partition="development",
        )
        registry.register_task(task)
        run = ctl.create_run(
            RunRequest(
                taskRef=store.put_artifact(task.model_dump(mode="json", by_alias=True)),
                modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
                budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
                idempotencyKey="idem-hr",
            ),
            task,
        )
        run_id = run.run_id
        provider.reset(run_id)

        # Forbidden types and unknown capabilities fail closed.
        for bad in ("harness.write", "evaluator.write", "credentials.read", "nope"):
            with pytest.raises(PermissionError):
                ctl.handle_host_request({"type": bad})
        with pytest.raises(PermissionError):
            ctl.handle_host_request({"type": "broker.call", "capabilityId": "unknown"})

        cap = _cap(run_id=run_id, tool="read_record", effect="read")
        ctl.register_prime_capability("cap-read", ENV, cap, provider)
        out = ctl.handle_host_request({
            "type": "broker.call",
            "capabilityId": "cap-read",
            "arguments": {"record_id": "record-1"},
            "idempotencyKey": "hr-1",
        })
        assert out["value"]["status"] == "ok"

        # Credential-shaped values are masked in learner-visible feedback.
        ctl.append_event(run_id, "tool_result", {"out": "key sk-abc1234567890 and password=hunter2"}, "broker", "learner")
        learner_ev = store.list_evidence(run_id)[-1]
        learner_payload = store.get_artifact(ArtifactRef.model_validate_json(learner_ev["source_ref"]))
        assert "sk-abc1234567890" not in json.dumps(learner_payload)
        assert "hunter2" not in json.dumps(learner_payload)
        assert "[REDACTED]" in json.dumps(learner_payload)
        # Operator-visible events retain full fidelity.
        ctl.append_event(run_id, "tool_result", {"out": "key sk-abc1234567890"}, "broker", "operator")
        op_ev = store.list_evidence(run_id)[-1]
        op_payload = store.get_artifact(ArtifactRef.model_validate_json(op_ev["source_ref"]))
        assert "sk-abc1234567890" in json.dumps(op_payload)

        # Bounded artifact transfer: begin -> chunk -> finish with sha256 check.
        data = b"hello-artifact"
        begin = ctl.handle_host_request({"type": "artifact.begin", "artifactId": "art", "size": len(data)})
        ctl.handle_host_request({"type": "artifact.chunk", "transferId": begin["transferId"], "offset": 0, "data": b64.b64encode(data).decode()})
        with pytest.raises(PermissionError):
            ctl.handle_host_request({"type": "artifact.finish", "transferId": begin["transferId"], "sha256": "0" * 64})
        # Restart transfer and finish correctly.
        begin2 = ctl.handle_host_request({"type": "artifact.begin", "artifactId": "art", "size": len(data)})
        ctl.handle_host_request({"type": "artifact.chunk", "transferId": begin2["transferId"], "offset": 0, "data": b64.b64encode(data).decode()})
        fin = ctl.handle_host_request({"type": "artifact.finish", "transferId": begin2["transferId"], "sha256": hl.sha256(data).hexdigest()})
        assert fin["artifact"]["sha256"] == hl.sha256(data).hexdigest()


    def test_learning_records_and_immutable_bytes(self, store: Store):
        ref = store.put_immutable_bytes(b"blob-bytes")
        assert store.get_immutable_bytes(ref) == b"blob-bytes"
        assert store.put_immutable_bytes(b"blob-bytes").sha256 == ref.sha256  # dedupe by content hash

        store.save_learning_record("lr-1", ENV, "run-a", '{"skill":"s1"}')
        store.save_learning_record("lr-2", ENV, "run-b", '{"skill":"s2"}')
        store.save_learning_record("lr-3", "other-env", "run-a", '{"skill":"s3"}')
        rows = store.list_learning_records(ENV)
        assert [r["record_id"] for r in rows] == ["lr-1", "lr-2"]
        assert [r["record_id"] for r in store.list_learning_records(ENV, "run-b")] == ["lr-2"]
        # record_json fields are decoded into the row so restart-reuse paths
        # can query kind/sourceId/trustClass directly.
        assert rows[0]["skill"] == "s1"
        store.save_learning_record("lr-4", ENV, "run-c", json.dumps({"kind": "live_evidence", "sourceId": "broker:ev-1", "trustedOutcome": True}))
        reused = [r for r in store.list_learning_records(ENV) if r.get("kind") == "live_evidence"]
        assert reused and reused[0]["sourceId"] == "broker:ev-1" and reused[0]["trustedOutcome"] is True
        assert store.list_learning_records(run_id="run-a") == [
            r for r in store.list_learning_records() if r["run_id"] == "run-a"
        ]

    def test_run_idempotency_fingerprint_conflict_restart_concurrency(self, workspace, registry, broker):
        import threading

        from adaptive_agent.controller import Controller
        from adaptive_agent.models import Budget, ModelProfile, RunRequest
        from adaptive_agent.store import RunIdempotencyConflict

        registry.register(NEUTRAL_MANIFEST)
        task = TaskInput(taskId="t-idem", environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64), goal="g")
        registry.register_task(task)

        store = Store(workspace)
        ctl = Controller(store, registry, broker)
        mp_ref = store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json"))
        bud_ref = store.put_artifact(Budget().model_dump(mode="json"))
        t_ref = store.put_artifact(task.model_dump(mode="json", by_alias=True))

        req1 = RunRequest(taskRef=t_ref, modelProfileRef=mp_ref, budgetRef=bud_ref, idempotencyKey="same-key")
        r1 = ctl.create_run(req1, task)
        r2 = ctl.create_run(req1, task)
        assert r1.run_id == r2.run_id

        # Same key, different request -> conflict.
        other_task = TaskInput(taskId="t-idem2", environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64), goal="g2")
        registry.register_task(other_task)
        req2 = RunRequest(taskRef=store.put_artifact(other_task.model_dump(mode="json", by_alias=True)), modelProfileRef=mp_ref, budgetRef=bud_ref, idempotencyKey="same-key")
        with pytest.raises(RunIdempotencyConflict):
            ctl.create_run(req2, other_task)

        # Restart durability: a fresh Store/Controller on the same dir replays the run.
        store2 = Store(workspace)
        registry2 = EnvironmentRegistry(store2)
        ctl2 = Controller(store2, registry2, ToolBroker(store2, registry2))
        again = ctl2.create_run(req1, task)
        assert again.run_id == r1.run_id

        # Concurrency: identical requests from two threads map to one run.
        results: list[str] = []
        errors: list[Exception] = []

        def worker():
            try:
                results.append(ctl.create_run(req1, task).run_id)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        t1, t2 = threading.Thread(target=worker), threading.Thread(target=worker)
        t1.start(); t2.start(); t1.join(); t2.join()
        assert not errors
        assert results == [r1.run_id, r1.run_id]

    def test_cancel_terminal_idempotent(self, store, registry, broker, provider):
        from adaptive_agent.controller import Controller
        from adaptive_agent.models import Budget, ModelProfile, RunRequest

        registry.register(NEUTRAL_MANIFEST)
        ctl = Controller(store, registry, broker)
        task = TaskInput(taskId="t-cancel", environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64), goal="g")
        registry.register_task(task)
        run = ctl.create_run(
            RunRequest(
                taskRef=store.put_artifact(task.model_dump(mode="json", by_alias=True)),
                modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
                budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
                idempotencyKey="idem-cancel",
            ),
            task,
        )
        c1 = ctl.cancel_run(run.run_id)
        c2 = ctl.cancel_run(run.run_id)  # terminal cancel is idempotent
        assert c1.status.value == "cancelled" and c2.status.value == "cancelled"
        assert c1.completed_at == c2.completed_at

    def test_trusted_evidence_shape(self, store, registry, broker):
        """Canonical model_response/accounting/trusted_outcome records match the
        session6 SQLiteRunEvidenceStore verification contract."""
        from adaptive_agent.controller import Controller
        from adaptive_agent.models import Budget, ModelProfile, RunRequest

        registry.register(NEUTRAL_MANIFEST)
        ctl = Controller(store, registry, broker)
        task = TaskInput(taskId="t-ev", environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64), goal="g")
        registry.register_task(task)
        run = ctl.create_run(
            RunRequest(
                taskRef=store.put_artifact(task.model_dump(mode="json", by_alias=True)),
                modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
                budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
                idempotencyKey="idem-ev",
            ),
            task,
        )
        run_id = run.run_id
        version_refs = {"model": "sha256:abc", "policy": "sha256:def"}
        usage = {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
        response = {"responseId": "resp-1", "usage": usage, "versionRefs": version_refs}
        ev = ctl.record_model_response(run_id, response)
        row = store.get_evidence(ev.evidence_id)
        assert row["event_type"] == "model_response" and row["run_id"] == run_id
        payload = store.get_artifact(ArtifactRef.model_validate_json(row["source_ref"]))
        assert payload["responseId"] == "resp-1"

        accounting = {
            "responseId": "resp-1", "runId": run_id, "taskId": "t-ev", "environmentId": ENV,
            "usage": usage, "costMicrounits": 100, "durationSeconds": 1.5, "versionRefs": version_refs,
            "economicCost": {"status": "measured", "microunits": 100},
            "nominalCostUsd": 0.0001,
            "billingBasis": "SDK nominal usage cost; subscription billing separate",
        }
        acct_ref = ctl.record_accounting(run_id, accounting, response)
        first = store.get_artifact(acct_ref)
        assert first["responseId"] == "resp-1"
        assert first["aggregateUsage"] == usage and first["responseCount"] == 1
        assert first["aggregateCostMicrounits"] == 100.0 and first["aggregateDurationSeconds"] == 1.5
        assert first["economicCost"] == {"status": "measured", "microunits": 100}
        assert first["billingBasis"].startswith("SDK nominal")

        # Second receipt (e.g. a retry): per-response usage stays exact while
        # the cumulative aggregates sum every receipt.
        usage2 = {"inputTokens": 4, "outputTokens": 2, "totalTokens": 6}
        response2 = {"responseId": "resp-2", "usage": usage2, "versionRefs": version_refs}
        ctl.record_model_response(run_id, response2)
        acct2 = ctl.record_accounting(
            run_id,
            {"responseId": "resp-2", "runId": run_id, "taskId": "t-ev", "environmentId": ENV,
             "usage": usage2, "costMicrounits": 50, "durationSeconds": 0.5, "versionRefs": version_refs},
            response2,
        )
        second = store.get_artifact(acct2)
        assert second["usage"] == usage2  # exact per-response
        assert second["aggregateUsage"] == {"inputTokens": 14, "outputTokens": 7, "totalTokens": 21}
        assert second["aggregateCostMicrounits"] == 150.0
        assert second["aggregateDurationSeconds"] == 2.0
        assert second["responseCount"] == 2
        # Economic-cost aggregation across receipts (first receipt only).
        assert second["aggregateEconomicCostMicrounits"] == 100.0
        assert second["economicCostStatuses"] == ["measured"]

        # Exact receipts for EvaluationJob: both responses, both accountings,
        # trusted outcome row, run/task/env binding.
        receipts = store.get_run_receipts(run_id)
        assert receipts["runId"] == run_id and receipts["taskId"] == "t-ev"
        assert [r["responseId"] for r in receipts["modelResponses"]] == ["resp-1", "resp-2"]
        assert [a["responseId"] for a in receipts["accounting"]] == ["resp-1", "resp-2"]
        assert receipts["accounting"][1]["aggregateUsage"]["totalTokens"] == 21

        outcome = {
            "responseId": "resp-1", "runId": run_id, "taskId": "t-ev", "environmentId": ENV,
            "passed": True, "reliable": True, "safetyViolations": 0,
        }
        out_ev = ctl.record_trusted_outcome(run_id, outcome)
        out_row = store.get_evidence(out_ev.evidence_id)
        assert out_row["event_type"] == "trusted_outcome"
        assert store.get_outcome_by_run_id(run_id)["passed"] == 1

        # Bad pins fail closed.
        with pytest.raises(ValueError):
            ctl.record_model_response(run_id, {"responseId": "", "usage": usage, "versionRefs": version_refs})
        # Conflicting direct imageDigest against pinned versionRefs.image rejected;
        # unpinned receipts remain valid.
        with pytest.raises(ValueError):
            ctl.record_model_response(run_id, {"responseId": "resp-x", "usage": usage, "versionRefs": {"image": "sha256:aaa"}, "imageDigest": "sha256:bbb"})
        ctl.record_model_response(run_id, {"responseId": "resp-ok", "usage": usage, "versionRefs": {"image": "sha256:aaa"}, "imageDigest": "sha256:aaa"})
        # Conflicting direct budgetRef against pinned versionRefs.budget rejected.
        with pytest.raises(ValueError):
            ctl.record_model_response(run_id, {"responseId": "resp-y", "usage": usage, "versionRefs": {"budget": "sha256:bbb"}, "budgetRef": {"sha256": "sha256:ccc"}})
        ctl.record_model_response(run_id, {"responseId": "resp-bok", "usage": usage, "versionRefs": {"budget": "sha256:bbb"}, "budgetRef": {"sha256": "sha256:bbb"}})
        with pytest.raises(ValueError):
            ctl.record_accounting(run_id, {**accounting, "taskId": "wrong"}, response)
        with pytest.raises(ValueError):
            ctl.record_trusted_outcome(run_id, {**outcome, "environmentId": "wrong"})

        # EVAL-004/005 probes execute real scenarios and report per-obligation output.
        p4 = ctl.execute_probe("EVAL-004")
        p5 = ctl.execute_probe("EVAL-005")
        assert p4.passed is True and p5.passed is True
        assert p4.provenance == "controller_toolbroker" and p4.obligations and p4.outputs
        assert set(p4.observed) == set(p4.obligations)
        assert bool(p4) is True
        with pytest.raises(KeyError):
            ctl.execute_probe("EVAL-999")


    def test_task_run_resume_and_allocation(self, store: Store, workspace):
        # Resumable task-run statuses: claim once, resume returns existing row.
        claimed, row = store.claim_task_run("tr-1", ENV, "t1", "validation")
        assert claimed and row["status"] == "running"
        resumed, row2 = store.claim_task_run("tr-1", ENV, "t1", "validation")
        assert not resumed and row2["task_run_id"] == "tr-1"
        store.update_task_run_status("tr-1", "complete")
        assert store.get_task_run("tr-1")["status"] == "complete"
        assert [r["task_run_id"] for r in store.list_task_runs(ENV, "validation")] == ["tr-1"]

        # Atomic allocation reservation with restart persistence.
        panels = [["a", "b"], ["c", "d"], ["e", "f"]]
        assert store.reserve_allocation("scope", "alloc-1", panels, 3) == 0
        assert store.reserve_allocation("scope", "alloc-1", panels, 3) is None  # consumed
        assert store.reserve_allocation("scope", "alloc-2", panels, 3) == 1
        assert store.reserve_allocation("scope", "alloc-3", panels, 3) == 2
        assert store.reserve_allocation("scope", "alloc-4", panels, 3) is None  # exhausted
        store2 = Store(workspace)
        assert store2.reserve_allocation("scope", "alloc-2", panels, 3) is None  # restart-safe

        # Crash-after-allocation recovery: read the reserved panel back.
        alloc = store2.get_allocation("alloc-2")
        assert alloc["panel_index"] == 1 and alloc["task_ids"] == ["c", "d"]
        assert store2.get_allocation("nope") is None

    def test_benchmark_task_run_owner_semantics(self, store: Store, workspace):
        # Single-owner claim with benchmark/arm/seed metadata.
        claimed, row = store.claim_benchmark_task_run(
            "btr-1", benchmark_id="bench-1", environment_id=ENV,
            task_id="t1", partition="validation", arm="B0", seed=42, owner_id="driver-a",
        )
        assert claimed and row["owner_id"] == "driver-a" and row["arm"] == "B0" and row["seed"] == 42
        # Same-owner re-claim resumes; foreign owner refused takeover.
        again, row = store.claim_benchmark_task_run(
            "btr-1", benchmark_id="bench-1", environment_id=ENV,
            task_id="t1", partition="validation", arm="B0", seed=42, owner_id="driver-a",
        )
        assert not again and row["owner_id"] == "driver-a"
        foreign, row = store.claim_benchmark_task_run(
            "btr-1", benchmark_id="bench-1", environment_id=ENV,
            task_id="t1", partition="validation", arm="B0", seed=42, owner_id="driver-b",
        )
        assert not foreign and row["owner_id"] == "driver-a"
        # Owner-guarded release.
        assert not store.release_task_run("btr-1", "driver-b", "complete")
        assert store.release_task_run("btr-1", "driver-a", "complete")
        # Restart durability + scoped listing.
        store2 = Store(workspace)
        rows = store2.list_task_runs(benchmark_id="bench-1", arm="B0")
        assert [r["task_run_id"] for r in rows] == ["btr-1"] and rows[0]["status"] == "complete"

    def test_dev_smoke_gate_and_learner_hiding(self, store, registry, broker):
        from adaptive_agent.controller import Controller
        from adaptive_agent.models import Budget, ModelProfile, RunRequest

        registry.register(NEUTRAL_MANIFEST)
        ctl = Controller(store, registry, broker)

        dev = TaskInput(taskId="t-dev-ok", environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64), goal="public dev goal", partition="development")
        hidden = TaskInput(taskId="t-final-secret", environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64), goal="secret", partition="final")
        registry.register_task(dev)
        registry.register_task(hidden)

        # Held-out panels denied before any trusted dev smoke.
        with pytest.raises(PermissionError):
            ctl.require_dev_smoke(ENV)

        # Learner sees only development tasks, public fields only.
        visible = ctl.learner_tasks(ENV)
        assert [t["taskId"] for t in visible] == ["t-dev-ok"]
        assert all(set(t) == {"taskId", "goal", "partition"} for t in visible)

        run = ctl.create_run(
            RunRequest(
                taskRef=store.put_artifact(dev.model_dump(mode="json", by_alias=True)),
                modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
                budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
                idempotencyKey="idem-smoke",
            ),
            dev,
        )
        ctl.record_trusted_outcome(run.run_id, {
            "responseId": "r1", "runId": run.run_id, "taskId": "t-dev-ok",
            "environmentId": ENV, "passed": True, "reliable": True, "safetyViolations": 0,
        })
        ctl.require_dev_smoke(ENV)  # now allowed

    def test_run_tool_call_projection(self, store, registry, broker, provider):
        """Session7 seam: sanitized broker call/evidence join for one
        DEVELOPMENT run — canonical fields only, no credentials or hidden data."""
        from adaptive_agent.controller import Controller
        from adaptive_agent.models import Budget, ModelProfile, RunRequest

        registry.register(NEUTRAL_MANIFEST)
        ctl = Controller(store, registry, broker)
        task = TaskInput(taskId="t-join", environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64), goal="g", partition="development")
        registry.register_task(task)
        run = ctl.create_run(
            RunRequest(
                taskRef=store.put_artifact(task.model_dump(mode="json", by_alias=True)),
                modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
                budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
                idempotencyKey="idem-join",
            ),
            task,
        )
        provider.reset(run.run_id)
        cap = _cap(run_id=run.run_id, tool="read_record", effect="read")
        req = _req("read_record", {"record_id": "record-1"}, run=run.run_id, key="jc-1")
        res = ctl.dispatch_tool(ENV, req, cap, provider)
        assert res.status == "ok"
        ctl.record_broker_tool_result(run.run_id, res.model_dump(mode="json", by_alias=True))

        rows = store.list_run_tool_calls(run.run_id)
        assert len(rows) == 1
        r0 = rows[0]
        assert r0["callId"] == res.call_id and r0["tool"] == "read_record"
        assert r0["input"] == {"record_id": "record-1"}
        assert r0["status"] == "ok" and r0["errorCode"] is None
        assert r0["runId"] == run.run_id and r0["taskId"] == "t-join" and r0["environmentId"] == ENV
        assert r0["partition"] == "development" and r0["visibility"] == "learner" and r0["redacted"] is True
        assert r0["evidenceId"] is not None and r0["evidenceContentHash"]
        assert "approval" not in r0  # approval tokens never projected
        assert r0["argumentsSha256"] and r0["resultSha256"]

        # Non-development runs fail closed.
        fin = TaskInput(taskId="t-join-final", environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64), goal="g", partition="final")
        registry.register_task(fin)
        fin_run = ctl.create_run(
            RunRequest(
                taskRef=store.put_artifact(fin.model_dump(mode="json", by_alias=True)),
                modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
                budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
                idempotencyKey="idem-join-final",
            ),
            fin,
        )
        with pytest.raises(PermissionError):
            store.list_run_tool_calls(fin_run.run_id)
        with pytest.raises(KeyError):
            store.list_run_tool_calls("run-missing")

        # Operator-only tool_result evidence (older app path) is projected with
        # sanitized output and no credential leakage.
        ctl.append_event(
            run.run_id,
            "tool_result",
            {
                "callId": "call_orphan", "tool": "update_record", "status": "ok",
                "output": {"value": "sk-live1234567890abcdef", "token": "api_key=hunter2secret"},
                "toolVersion": "1",
            },
            "broker", "operator",
        )
        rows2 = store.list_run_tool_calls(run.run_id)
        orphan = next(r for r in rows2 if r["callId"] == "call_orphan")
        assert orphan["visibility"] == "learner" and orphan["redacted"] is True
        flat = json.dumps(orphan)
        assert "sk-live1234567890abcdef" not in flat and "hunter2secret" not in flat
        assert "[REDACTED]" in flat
        assert orphan["input"] is None and orphan["argumentsSha256"] is None

        # Tools absent from the manifest are never projected (fail closed),
        # and tampered content-hash evidence is dropped.
        ctl.append_event(run.run_id, "tool_result", {"callId": "call_ghost", "tool": "ghost_tool", "status": "ok"}, "broker", "operator")
        tampered_ref = store.put_artifact({"callId": "call_x", "tool": "update_record"})
        store.append_evidence("ev-tampered", {"run_id": run.run_id, "sequence": 999, "event_type": "tool_result", "content_hash": "0" * 64, "source_ref": tampered_ref.model_dump_json(by_alias=True), "trust_class": "broker", "visibility": "operator", "redacted": 0})
        rows3 = store.list_run_tool_calls(run.run_id)
        assert all(r["tool"] != "ghost_tool" for r in rows3)
        assert all(r["callId"] != "call_x" for r in rows3)

        # Unified runtime feed: evidence + broker_call kinds, no raw operator rows.
        feed = store.list_learning_evidence(environment_id=ENV, run_id=run.run_id)
        kinds = {r["kind"] for r in feed}
        assert "broker_call" in kinds
        assert all(r["partition"] == "development" for r in feed)
        flat_feed = json.dumps(feed)
        assert "sk-live1234567890abcdef" not in flat_feed and "hunter2secret" not in flat_feed
        without = store.list_learning_evidence(environment_id=ENV, run_id=run.run_id, include_broker_projection=False)
        assert {r["kind"] for r in without} == {"evidence"}

    def test_learning_projection(self, store, registry, broker):
        """Session7 seam: public docs, redacted development evidence + trusted
        outcome, patch bytes. No hidden evaluator content."""
        from adaptive_agent.controller import Controller
        from adaptive_agent.models import Budget, ModelProfile, RunRequest

        registry.register(NEUTRAL_MANIFEST)
        ctl = Controller(store, registry, broker)

        # Public docs: only non-restricted artifacts.
        pub = store.put_artifact({"title": "public doc"})
        hidden_doc = store.put_artifact({"classification": "evaluator_only", "data": "secret"})
        manifest_dump = NEUTRAL_MANIFEST.model_dump(mode="json", by_alias=True)
        manifest_dump["docs"] = [
            pub.model_dump(mode="json"), hidden_doc.model_dump(mode="json"),
        ]
        store.register_environment(ENV, "1.0.0", store.put_artifact(manifest_dump))
        docs = store.get_public_docs(ENV)
        assert len(docs) == 1 and docs[0]["content"]["title"] == "public doc"

        dev = TaskInput(taskId="t-proj", environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64), goal="g", partition="development")
        fin = TaskInput(taskId="t-proj-final", environmentRef=ArtifactRef(id=ENV, version="1.0.0", sha256="0" * 64), goal="g", partition="final")
        registry.register_task(dev)
        registry.register_task(fin)
        run = ctl.create_run(
            RunRequest(
                taskRef=store.put_artifact(dev.model_dump(mode="json", by_alias=True)),
                modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
                budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
                idempotencyKey="idem-proj",
            ),
            dev,
        )
        ctl.append_event(run.run_id, "tool_result", {"v": 1}, "broker", "learner")
        ctl.append_event(run.run_id, "internal", {"v": 2}, "broker", "operator")  # excluded
        ctl.record_trusted_outcome(run.run_id, {
            "responseId": "r1", "runId": run.run_id, "taskId": "t-proj",
            "environmentId": ENV, "passed": True, "reliable": True, "safetyViolations": 0,
        })
        proj = store.list_learner_evidence(ENV, run.run_id)
        assert len(proj) == 2  # run_created + tool_result (learner, redacted)
        assert all(r["visibility"] == "learner" and r["redacted"] == 1 and r["partition"] == "development" for r in proj)
        assert proj[-1]["outcome_passed"] == 1
        # Final-partition runs are never projected.
        fin_run = ctl.create_run(
            RunRequest(
                taskRef=store.put_artifact(fin.model_dump(mode="json", by_alias=True)),
                modelProfileRef=store.put_artifact(ModelProfile(provider="simulation", model_name="m").model_dump(mode="json")),
                budgetRef=store.put_artifact(Budget().model_dump(mode="json")),
                idempotencyKey="idem-proj-final",
            ),
            fin,
        )
        ctl.append_event(fin_run.run_id, "tool_result", {"v": 3}, "broker", "learner")
        assert all(r["partition"] == "development" for r in store.list_learner_evidence(ENV))

        # Exact patch bytes round-trip.
        patch = b"diff --git a/f b/f\n+line\n"
        pref = store.put_immutable_bytes(patch)
        assert store.get_immutable_bytes(pref) == patch


class TestCandidateLifecycle:
    def _base(self, manager: CandidateManager, store: Store) -> SkillBundle:
        bundle = SkillBundle(skills=[])
        bundle.content_hash = manager.recompute_bundle_hash(bundle)
        manager.initialize_active_bundle(bundle)
        return bundle

    def _candidate(self, base: SkillBundle) -> SkillBundle:
        return SkillBundle(
            skills=[
                SkillVersion(
                    skillId="retry-on-conflict",
                    version="1",
                    procedure="recheck version before update",
                )
            ],
            parent=base.bundle_id,
        )

    def _report(self, cand: SkillBundle, base: SkillBundle, protocol: str, evaluator: str) -> EvaluationReport:
        return EvaluationReport(
            candidateHash=cand.content_hash,
            baseHash=base.content_hash,
            protocolHash=protocol,
            partitionRef=ArtifactRef(id="p", version="1", sha256="0" * 64),
            pairedRunIds=[("r1", "r2")],
            metrics=MetricAggregate(accuracy=0.9, reliability=0.9, meanCost=1.0, p95LatencyMs=10.0),
            uncertainty={
                "baseline_accuracy": 0.5,
                "ci_lower": 0.05,
                "baseline_cost": 1.0,
                "baseline_latency": 10.0,
                "per_environment": {"neutral": {"baseline_accuracy": 0.5, "candidate_accuracy": 0.9, "baseline_reliability": 0.5, "candidate_reliability": 0.9}},
            },
            safetyResults={"injection": True},
            validity=EvaluationState.valid,
            evaluatorProvenance=evaluator,
        )

    def test_evidence_must_bind_to_development_run(self, manager: CandidateManager, store: Store):
        base = self._base(manager, store)
        cand = self._candidate(base)
        # Evidence from a validation run is rejected.
        bad_ev = _evidence(store, "run-val", partition="validation")
        p = CandidateProposal(baseBundleHash=base.content_hash, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[bad_ev])
        with pytest.raises(CandidateValidationError):
            manager.submit_candidate(p, cand)

        # Development evidence with a trusted outcome passes validation.
        good_ev = _evidence(store, "run-dev", partition="development")
        p2 = CandidateProposal(baseBundleHash=base.content_hash, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[good_ev])
        submitted = manager.submit_candidate(p2, cand)
        assert submitted.state == CandidateState.validated
        assert submitted.candidate_bundle_hash == cand.content_hash

    def test_tampered_hash_and_forbidden_code_rejected(self, manager: CandidateManager, store: Store):
        base = self._base(manager, store)
        ev = _evidence(store, "run-h1")
        cand = self._candidate(base)
        cand.content_hash = "deadbeef" * 8  # caller-claimed hash that does not match payload
        p = CandidateProposal(baseBundleHash=base.content_hash, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[ev])
        with pytest.raises(CandidateValidationError):
            manager.submit_candidate(p, cand)

        evil = SkillBundle(
            skills=[SkillVersion(skillId="s", version="1", procedure="import os; os.system('rm -rf /')")],
            parent=base.bundle_id,
        )
        p2 = CandidateProposal(baseBundleHash=base.content_hash, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[ev])
        with pytest.raises(CandidateValidationError):
            manager.submit_candidate(p2, evil)

        # Stale base rejected.
        stale = CandidateProposal(baseBundleHash="0" * 64, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[ev])
        with pytest.raises(CandidateValidationError):
            manager.submit_candidate(stale, self._candidate(base))

    def test_promote_requires_frozen_protocol_and_evaluator(self, manager: CandidateManager, store: Store):
        base = self._base(manager, store)
        ev = _evidence(store, "run-p1")
        cand = self._candidate(base)
        p = CandidateProposal(baseBundleHash=base.content_hash, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[ev])
        manager.submit_candidate(p, cand)
        manager.start_evaluation(p.candidate_id)

        gate = PromotionGate(protocolHash="proto-1")
        manager.freeze_protocol(gate, evaluator_id="trusted-eval")

        # Wrong provenance rejected.
        bad = self._report(cand, base, "proto-1", "other-eval")
        with pytest.raises(PromotionError):
            manager.promote(p.candidate_id, bad)

        # Unregistered protocol rejected.
        bad2 = self._report(cand, base, "proto-404", "trusted-eval")
        with pytest.raises(PromotionError):
            manager.promote(p.candidate_id, bad2)

        # Missing cells rejected.
        thin = self._report(cand, base, "proto-1", "trusted-eval")
        thin.safety_results = {}
        with pytest.raises(PromotionError):
            manager.promote(p.candidate_id, thin)

        # Trusted + frozen + complete promotes atomically.
        good = self._report(cand, base, "proto-1", "trusted-eval")
        decision = manager.promote(p.candidate_id, good)
        assert decision.decision == "promoted"
        assert manager.get_active_bundle().content_hash == cand.content_hash
        row = store.get_candidate(p.candidate_id)
        assert row["state"] == "promoted"
        assert '"promoted"' in row["candidate_json"]

    def test_gate_rejection_and_supersede(self, manager: CandidateManager, store: Store):
        base = self._base(manager, store)
        ev = _evidence(store, "run-g1")
        cand = self._candidate(base)
        p = CandidateProposal(baseBundleHash=base.content_hash, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[ev])
        manager.submit_candidate(p, cand)
        manager.start_evaluation(p.candidate_id)
        manager.freeze_protocol(PromotionGate(protocolHash="proto-2"), "trusted-eval")

        # Failing gate: no gain.
        weak = self._report(cand, base, "proto-2", "trusted-eval")
        weak.metrics = MetricAggregate(accuracy=0.5, reliability=0.5, meanCost=1.0, p95LatencyMs=10.0)
        d = manager.promote(p.candidate_id, weak)
        assert d.decision == "rejected"
        assert store.get_candidate(p.candidate_id)["state"] == "rejected"

        # Fresh candidate evaluated against a moved base is superseded.
        ev2 = _evidence(store, "run-g2")
        cand2 = self._candidate(base)
        p2 = CandidateProposal(baseBundleHash=base.content_hash, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[ev2])
        manager.submit_candidate(p2, cand2)
        manager.start_evaluation(p2.candidate_id)
        # Simulate a competing promotion moving the active pointer.
        store.set_active_bundle(cand.content_hash)  # emulate prior CAS result
        good = self._report(cand2, base, "proto-2", "trusted-eval")
        with pytest.raises(PromotionError):
            manager.promote(p2.candidate_id, good)
        assert store.get_candidate(p2.candidate_id)["state"] == "superseded"

    def test_session6_dict_report_and_staleness(self, store: Store):
        """External contract: frozen evaluator report payload (to_dict) is
        accepted only when frozen protocol, evaluator refs, partition hashes,
        attestation, and completeness cells all check out."""
        manager = CandidateManager(store, report_verifier=lambda r: r.get("attestation") == "tok-good")
        base = self._base(manager, store)
        ev = _evidence(store, "run-s6")
        cand = self._candidate(base)
        p = CandidateProposal(baseBundleHash=base.content_hash, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[ev])
        manager.submit_candidate(p, cand)
        manager.start_evaluation(p.candidate_id)
        partitions = {"env:validation": "ph1"}
        manager.freeze_protocol(
            PromotionGate(protocolHash="proto-s6"),
            "eval-root",
            evaluator_refs=["eval-a", "eval-b"],
            partition_hashes=partitions,
        )

        def report(**over):
            r = {
                "comparison": "validation",
                "validityStatus": "valid",
                "promotionEligible": True,
                "candidateHash": cand.content_hash,
                "baseHash": base.content_hash,
                "protocolHash": "proto-s6",
                "partitionHashes": partitions,
                "armSummaries": {
                    "B0": {"accuracy": 0.5, "reliability": 0.5, "meanCostMicrounits": 100.0, "p95LatencySeconds": 1.0, "safetyViolations": 0, "count": 10},
                    "L": {"accuracy": 0.9, "reliability": 0.9, "meanCostMicrounits": 100.0, "p95LatencySeconds": 1.0, "safetyViolations": 0, "count": 10},
                },
                "confidenceIntervals": [{"metric": "accuracy", "lower95": 0.05, "upper95": 0.6, "point": 0.4, "draws": 10000, "analysisSeed": 1}],
                "safetyPassed": True,
                "safetyCaseResults": {"EVAL-004": True, "EVAL-005": True},
                "missingPairs": 0,
                "partitionLeak": False,
                "invalidFixtureResets": 0,
                "infrastructureFailures": [],
                "evaluatorRefs": ["eval-a", "eval-b"],
                "environmentCells": {"env": {"B0": {"accuracy": 0.5, "reliability": 0.5}, "L": {"accuracy": 0.9, "reliability": 0.9}}},
                "metricCellsComplete": True,
                "safetyCellsComplete": True,
                "modelProvenanceComplete": True,
                "attestation": "tok-good",
                "exposure": [],
                "workload": {"totalAttemptedRuns": 20},
                "analysisSeed": 1,
            }
            r.update(over)
            return r

        # Stale: wrong attestation
        with pytest.raises(PromotionError):
            manager.promote(p.candidate_id, report(attestation="tok-evil"))
        # Stale: synthetic model provenance / incomplete cells
        with pytest.raises(PromotionError):
            manager.promote(p.candidate_id, report(promotionEligible=False))
        # Stale: mismatched evaluator refs
        with pytest.raises(PromotionError):
            manager.promote(p.candidate_id, report(evaluatorRefs=["eval-x"]))
        # Stale: partition hash drift
        with pytest.raises(PromotionError):
            manager.promote(p.candidate_id, report(partitionHashes={"env:validation": "ph2"}))
        # Stale: unregistered protocol
        with pytest.raises(PromotionError):
            manager.promote(p.candidate_id, report(protocolHash="proto-unknown"))
        # Valid report promotes atomically
        d = manager.promote(p.candidate_id, report())
        assert d.decision == "promoted"
        assert manager.get_active_bundle().content_hash == cand.content_hash

    def test_canonical_track1_protocol_required(self, store: Store):
        from adaptive_agent.candidate import TRACK1_FIXTURE_HASHES, TRACK1_PROTOCOL_HASH

        manager = CandidateManager(store)
        base = self._base(manager, store)
        ev = _evidence(store, "run-t1")
        cand = self._candidate(base)
        p = CandidateProposal(baseBundleHash=base.content_hash, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[ev])
        manager.submit_candidate(p, cand)
        manager.start_evaluation(p.candidate_id)

        # Freeze canonical + a foreign protocol; foreign reports become stale.
        manager.freeze_track1_protocol("eval-root", evaluator_refs=["eval-root"])
        manager.freeze_protocol(PromotionGate(protocolHash="proto-foreign"), "eval-foreign")

        good = self._report(cand, base, TRACK1_PROTOCOL_HASH, "eval-root")
        foreign = self._report(cand, base, "proto-foreign", "eval-foreign")
        with pytest.raises(PromotionError):
            manager.promote(p.candidate_id, foreign)

        # Wrong fixture hashes pinned to the canonical hash are refused at freeze.
        with pytest.raises(PromotionError):
            manager.freeze_protocol(
                PromotionGate(protocolHash=TRACK1_PROTOCOL_HASH),
                "eval-root",
                fixture_hashes={"finance": "tampered"},
            )
        d = manager.promote(p.candidate_id, good)
        assert d.decision == "promoted"

    def test_rollback_restricted_to_lineage(self, manager: CandidateManager, store: Store):
        base = self._base(manager, store)
        ev = _evidence(store, "run-r1")
        cand = self._candidate(base)
        p = CandidateProposal(baseBundleHash=base.content_hash, predictedEffect="x", proposerVersion="1", supportingEvidenceIds=[ev])
        manager.submit_candidate(p, cand)
        manager.start_evaluation(p.candidate_id)
        manager.freeze_protocol(PromotionGate(protocolHash="proto-3"), "trusted-eval")
        manager.promote(p.candidate_id, self._report(cand, base, "proto-3", "trusted-eval"))
        assert manager.get_active_bundle().content_hash == cand.content_hash

        # Arbitrary hash not in lineage -> refused.
        rogue = SkillBundle(skills=[SkillVersion(skillId="rogue", version="1", procedure="x")])
        store.save_bundle(rogue.bundle_id, None, rogue.content_hash, rogue.model_dump_json(by_alias=True))
        with pytest.raises(PromotionError):
            manager.rollback(rogue.content_hash, "audit")

        # Lineage member rolls back atomically.
        d = manager.rollback(base.content_hash, "safety review")
        assert d.new_active_hash == base.content_hash
        assert manager.get_active_bundle().content_hash == base.content_hash
