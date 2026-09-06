"""Owned, narrow projection of durable broker history into learner records."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from .retrieval import canonical_json, content_hash


_SECRET_VALUE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|AKIA[0-9A-Z]{16}|bearer\s+\S+|"
    r"(?:api[_-]?key|token|secret|password|authorization|credential)\s*[:=]\s*\S+)"
)
_HIDDEN_KEY = re.compile(r"(?i)(?:hidden|expected|evaluator|answer[_ -]?key|secret|credential|api[_-]?key|token|password|authorization)")


def _safe(value: Any) -> Any:
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items() if not _HIDDEN_KEY.search(str(key))}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


class LearningProjectionError(ValueError):
    """Raised when durable broker history cannot be safely projected."""


class DurableBrokerLearningProjection:
    """Join trusted broker evidence to its durable call, then persist a new record.

    The adapter reads original evidence and tool-call rows but never updates them.
    Only the derived ``learning_records`` row is learner-visible.
    """

    def __init__(self, store: Any) -> None:
        self.store = store

    def _manifest(self, environment_id: str) -> Mapping[str, Any]:
        environment = self.store.get_environment(environment_id)
        if not isinstance(environment, Mapping) or not isinstance(environment.get("manifest_ref"), str):
            raise LearningProjectionError("environment manifest is unavailable")
        manifest_ref = json.loads(environment["manifest_ref"])
        manifest = self.store.get_artifact(manifest_ref["sha256"])
        if not isinstance(manifest, Mapping):
            raise LearningProjectionError("environment manifest is malformed")
        return manifest

    def _tool_schemas(self, environment_id: str) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for schema in self._manifest(environment_id).get("toolSchemas", []):
            if isinstance(schema, Mapping) and isinstance(schema.get("name"), str):
                result[schema["name"]] = schema
        return result

    def _tool_call(self, call_id: str) -> Mapping[str, Any] | None:
        getter = getattr(self.store, "get_tool_call", None)
        if callable(getter):
            row = getter(call_id)
            return row if isinstance(row, Mapping) else None
        connect = getattr(self.store, "_connect", None)
        if not callable(connect):
            return None
        with connect() as connection:
            row = connection.execute("SELECT * FROM tool_calls WHERE call_id = ?", (call_id,)).fetchone()
            return dict(row) if row else None

    def _persist_derived_evidence(self, *, evidence_id: str, run_id: str, content: str) -> None:
        getter = getattr(self.store, "get_evidence", None)
        if callable(getter) and getter(evidence_id) is not None:
            return
        put_artifact = getattr(self.store, "put_artifact", None)
        append = getattr(self.store, "append_evidence", None)
        sequence = getattr(self.store, "next_event_sequence", None)
        if not callable(put_artifact) or not callable(append) or not callable(sequence):
            raise LearningProjectionError("Store lacks derived evidence persistence seam")
        ref = put_artifact(content)
        source_ref = ref.model_dump_json(by_alias=True) if hasattr(ref, "model_dump_json") else json.dumps(dict(ref), sort_keys=True, separators=(",", ":"))
        append(evidence_id, {
            "run_id": run_id, "sequence": sequence(run_id), "event_type": "learning_evidence_projection",
            "content_hash": content_hash(content), "source_ref": source_ref,
            "trust_class": "broker", "visibility": "learner", "redacted": 1,
        })

    def project(self, *, environment_id: str, run_id: str, task_id: str, outcome_passed: bool) -> list[tuple[str, Mapping[str, Any]]]:
        run = self.store.get_run(run_id)
        task = self.store.get_task(task_id)
        if not isinstance(run, Mapping) or run.get("environment_id") != environment_id or run.get("task_id") != task_id:
            raise LearningProjectionError("broker history is not bound to the requested run")
        if not isinstance(task, Mapping) or task.get("environment_id") != environment_id or task.get("partition") != "development":
            raise LearningProjectionError("broker history requires a DEVELOPMENT task")
        schemas = self._tool_schemas(environment_id)
        records: list[tuple[str, Mapping[str, Any]]] = []
        for event in self.store.list_evidence(run_id):
            if event.get("event_type") != "tool_result" or event.get("trust_class") != "broker" or event.get("visibility") not in {"learner", "operator"}:
                continue
            source_ref = event.get("source_ref")
            if not isinstance(source_ref, str):
                continue
            ref = json.loads(source_ref)
            try:
                payload = self.store.get_artifact(ref["sha256"])
            except (KeyError, ValueError):
                continue
            if not isinstance(ref.get("sha256"), str) or hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest() != ref["sha256"] or event.get("content_hash") != ref["sha256"]:
                continue
            if not isinstance(payload, Mapping) or not isinstance(payload.get("callId"), str):
                continue
            call = self._tool_call(payload["callId"])
            if not isinstance(call, Mapping) or call.get("run_id") != run_id or call.get("environment_id") != environment_id:
                continue
            tool = call.get("tool")
            schema = schemas.get(tool) if isinstance(tool, str) else None
            if schema is None:
                continue
            try:
                arguments = json.loads(call.get("arguments_json", "{}"))
                result = json.loads(call.get("result_json", "{}"))
            except (TypeError, json.JSONDecodeError):
                continue
            properties = schema.get("inputSchema", {}).get("properties", {}) if isinstance(schema.get("inputSchema"), Mapping) else {}
            safe_input = _safe({key: arguments[key] for key in properties if isinstance(arguments, Mapping) and key in arguments})
            safe_result = _safe({key: result.get(key) for key in ("status", "effect", "toolVersion", "output") if isinstance(result, Mapping) and key in result})
            safe_error = {}
            if isinstance(result, Mapping) and isinstance(result.get("error"), Mapping):
                safe_error = _safe({key: result["error"].get(key) for key in ("code", "retry") if key in result["error"]})
            details = {
                "callId": payload["callId"], "tool": tool, "input": safe_input,
                "result": safe_result, "error": safe_error,
                "sourceEvidenceId": event["evidence_id"], "sourceContentHash": event.get("content_hash"),
                "sourceArtifactHash": ref.get("sha256"), "callArgumentsHash": hashlib.sha256(str(call.get("arguments_json", "")).encode()).hexdigest(),
                "callResultHash": hashlib.sha256(str(call.get("result_json", "")).encode()).hexdigest(),
                "runId": run_id, "taskId": task_id, "environmentId": environment_id, "partition": "development",
            }
            content = f"Broker development observation: {canonical_json(details)}"
            record = {
                "kind": "live_evidence", "sourceId": f"broker:{event['evidence_id']}", "content": content,
                "contentHash": content_hash(content), "sourceContentHash": event.get("content_hash"),
                "sourceEvidenceId": event["evidence_id"], "sourceCallId": payload["callId"],
                "environmentId": environment_id, "runId": run_id, "taskId": task_id,
                "partition": "development", "visibility": "learner", "trustClass": "broker", "trustedOutcome": True, "outcomePassed": outcome_passed,
            }
            self._persist_derived_evidence(evidence_id=record["sourceId"], run_id=run_id, content=content)
            records.append((f"learning-broker-{event['evidence_id']}", record))
        return records


__all__ = ["DurableBrokerLearningProjection", "LearningProjectionError"]
