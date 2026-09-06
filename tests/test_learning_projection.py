from __future__ import annotations

import json
from pathlib import Path

from adaptive_agent.learning_projection import DurableBrokerLearningProjection
from test_learning_store_integration import ENVIRONMENT, RUN, _setup_store


def _broker_event(store, *, evidence_id: str, call_id: str, run_id: str = RUN):
    payload = {
        "callId": call_id, "toolVersion": "7", "status": "error", "effect": "none",
        "output": {"status": "open", "expectedAnswer": "hidden-answer", "detail": "version mismatch", "apiKey": "sk-secret"},
        "error": {"code": "VERSION_CONFLICT", "retry": "after_refresh", "message": "password=hunter2"},
    }
    ref = store.put_artifact(payload)
    store.append_evidence(evidence_id, {"run_id": run_id, "sequence": 8, "event_type": "tool_result", "content_hash": ref.sha256, "source_ref": ref.model_dump_json(by_alias=True), "trust_class": "broker", "visibility": "operator", "redacted": 0})
    return payload


def test_projection_joins_call_and_redacts_hidden_values(tmp_path: Path):
    store, _, _ = _setup_store(tmp_path)
    call_id = "call-projection"
    store.prepare_tool_call({"call_id": call_id, "run_id": RUN, "step_id": "step-1", "environment_id": ENVIRONMENT, "tool": "read", "arguments_json": json.dumps({"invoice_id": "INV-DEV-000", "apiKey": "sk-secret"}), "idempotency_key": "projection-1"})
    result = _broker_event(store, evidence_id="ev-projection", call_id=call_id)
    store.save_tool_result(call_id, json.dumps(result, sort_keys=True, separators=(",", ":")), "none")
    records = DurableBrokerLearningProjection(store).project(environment_id=ENVIRONMENT, run_id=RUN, task_id="task-durable", outcome_passed=True)
    assert len(records) == 1
    record = records[0][1]
    assert '"tool":"read"' in record["content"]
    assert '"code":"VERSION_CONFLICT"' in record["content"]
    assert '"retry":"after_refresh"' in record["content"]
    assert "expectedAnswer" not in record["content"]
    assert "hidden-answer" not in record["content"]
    assert "sk-secret" not in record["content"]
    assert "hunter2" not in record["content"]
    assert record["sourceEvidenceId"] == "ev-projection"
    assert record["sourceCallId"] == call_id
    derived = store.get_evidence("broker:ev-projection")
    assert derived["event_type"] == "learning_evidence_projection"
    assert store.evidence_provenance("broker:ev-projection")["partition"] == "development"


def test_projection_drops_call_from_other_run(tmp_path: Path):
    store, _, _ = _setup_store(tmp_path)
    call_id = "call-wrong-run"
    store.prepare_tool_call({"call_id": call_id, "run_id": "other-run", "step_id": "step-1", "environment_id": ENVIRONMENT, "tool": "read", "arguments_json": json.dumps({"invoice_id": "INV-DEV-000"}), "idempotency_key": "projection-wrong"})
    _broker_event(store, evidence_id="ev-wrong-run", call_id=call_id)
    assert DurableBrokerLearningProjection(store).project(environment_id=ENVIRONMENT, run_id=RUN, task_id="task-durable", outcome_passed=False) == []
    assert store.get_evidence("broker:ev-wrong-run") is None


def test_projection_excludes_evaluator_visibility_and_tampered_source(tmp_path: Path):
    store, _, _ = _setup_store(tmp_path)
    call_id = "call-hidden"
    store.prepare_tool_call({"call_id": call_id, "run_id": RUN, "step_id": "step-1", "environment_id": ENVIRONMENT, "tool": "read", "arguments_json": json.dumps({"invoice_id": "INV-DEV-000"}), "idempotency_key": "projection-hidden"})
    payload = _broker_event(store, evidence_id="ev-hidden", call_id=call_id)
    hidden = store.get_evidence("ev-hidden")
    hidden.pop("evidence_id", None)
    hidden["visibility"] = "evaluator_only"
    store.append_evidence("ev-hidden", hidden)
    assert DurableBrokerLearningProjection(store).project(environment_id=ENVIRONMENT, run_id=RUN, task_id="task-durable", outcome_passed=False) == []

    tampered = store.put_artifact(payload)
    tampered_event = {"run_id": RUN, "sequence": 9, "event_type": "tool_result", "content_hash": "0" * 64, "source_ref": tampered.model_dump_json(by_alias=True), "trust_class": "broker", "visibility": "operator", "redacted": 0}
    store.append_evidence("ev-tampered", tampered_event)
    assert DurableBrokerLearningProjection(store).project(environment_id=ENVIRONMENT, run_id=RUN, task_id="task-durable", outcome_passed=False) == []


def test_persisted_projection_reused_after_raw_event_is_gone(tmp_path: Path):
    import adaptive_agent.learning_runtime as runtime_module

    store, manager, _ = _setup_store(tmp_path)
    call_id = "call-restart"
    store.prepare_tool_call({"call_id": call_id, "run_id": RUN, "step_id": "step-1", "environment_id": ENVIRONMENT, "tool": "read", "arguments_json": json.dumps({"invoice_id": "INV-DEV-000"}), "idempotency_key": "projection-restart"})
    result = _broker_event(store, evidence_id="ev-restart", call_id=call_id)
    store.save_tool_result(call_id, json.dumps(result, sort_keys=True, separators=(",", ":")), "none")
    projected = DurableBrokerLearningProjection(store).project(environment_id=ENVIRONMENT, run_id=RUN, task_id="task-durable", outcome_passed=True)
    record_id, record = projected[0]
    store.save_learning_record(record_id, ENVIRONMENT, RUN, json.dumps(record, sort_keys=True, separators=(",", ":")))
    with store._connect() as connection:
        connection.execute("DELETE FROM evidence WHERE evidence_id = ?", ("ev-restart",))
        connection.commit()
    raw = runtime_module.LearningRuntime.build(store=store, manager=manager, model_client=object())._materialize_run_records(environment_id=ENVIRONMENT, run_id=RUN)
    assert len(raw) == 1
    assert raw[0]["sourceId"] == record["sourceId"]
    persisted = store.list_learning_records(environment_id=ENVIRONMENT, run_id=RUN)
    assert [row["record_id"] for row in persisted].count(record_id) == 1
