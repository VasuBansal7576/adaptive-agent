"""Production evaluation-job orchestration.

This module owns the boundary between durable runtime execution and evaluator
scoring.  The driver is the only component allowed to select or execute a
panel; ``EvaluationRunner`` receives the driver's persisted observations only.
"""
from __future__ import annotations

import argparse
import copy
import concurrent.futures
import importlib
import json
import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from adaptive_agent.benchmark import FrozenExecutionConfig, ResumableEvaluationDriver, TrustedTaskExecutor
from adaptive_agent.controller import Controller
from adaptive_agent.evaluation import (
    AblationInput,
    Arm,
    EnvironmentPackage,
    EvaluationError,
    EvaluationProtocol,
    EvaluationReport,
    EvaluationRunner,
    Partition,
    RunObservation,
)
from adaptive_agent.evaluation_store import SQLiteRunEvidenceStore, build_durable_evaluation_runner
from adaptive_agent.store import Store


@dataclass(frozen=True)
class EvaluationJobResult:
    job_id: str
    comparison: str
    status: str
    report: EvaluationReport | dict[str, Any] | None
    decision: object | None = None
    error: str | None = None
    runtime_accounting: dict[str, Any] | None = None


@dataclass(frozen=True)
class LifecycleStage:
    """One resumable stage in the complete Track 1 experiment."""

    name: str
    cells: tuple[str, ...]
    callback: Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
    retries: int = 0
    observation_recoverer: Callable[..., Sequence[Any]] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.cells or not callable(self.callback) or self.retries < 0:
            raise EvaluationError("invalid lifecycle stage")
        if len(set(self.cells)) != len(self.cells):
            raise EvaluationError("lifecycle stage cells must be unique")


LIFECYCLE_STAGE_ORDER = ("bootstrap", "training", "learning", "transfer", "adaptation", "safety", "validation", "final")
_LIFECYCLE_OPERATIONAL_CONTEXT_KEYS = frozenset({"resume"})


class EvaluationJob:
    """Run and persist one immutable comparison job."""

    def __init__(
        self,
        store: Store,
        controller: Controller,
        protocol: EvaluationProtocol,
        packages: Mapping[str, EnvironmentPackage],
        arm_bundles: Mapping[Arm | str, object],
        execute: TrustedTaskExecutor,
        *,
        total_budget_microunits: int | None = None,
        max_total_attempts: int | None = None,
    ) -> None:
        self.store = store
        self.controller = controller
        self.protocol = protocol
        self.packages = dict(packages)
        self.arm_bundles = dict(arm_bundles)
        if not callable(execute):
            raise EvaluationError("evaluation job requires a bound trusted executor")
        self.execute = execute
        self.total_budget_microunits = total_budget_microunits
        if max_total_attempts is not None and max_total_attempts < 0:
            raise EvaluationError("max_total_attempts cannot be negative")
        self.max_total_attempts = max_total_attempts
        with store.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_jobs (job_id TEXT PRIMARY KEY, comparison TEXT NOT NULL, status TEXT NOT NULL, report_ref TEXT, error TEXT, runtime_accounting_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL)"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(evaluation_jobs)")}
            if "runtime_accounting_json" not in columns:
                conn.execute("ALTER TABLE evaluation_jobs ADD COLUMN runtime_accounting_json TEXT NOT NULL DEFAULT '{}'")
            conn.commit()

    def _ensure_lifecycle_tables(self) -> None:
        with self.store.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_lifecycle_budget (job_id TEXT PRIMARY KEY, max_attempts INTEGER NOT NULL, max_input_tokens INTEGER NOT NULL, max_output_tokens INTEGER NOT NULL, max_tool_calls INTEGER NOT NULL, max_wall_micros INTEGER NOT NULL, max_cost_microunits INTEGER NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, tool_calls INTEGER NOT NULL DEFAULT 0, wall_micros INTEGER NOT NULL DEFAULT 0, cost_microunits INTEGER NOT NULL DEFAULT 0, blocked INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_lifecycle_attempts (job_id TEXT NOT NULL, stage TEXT NOT NULL, cell_key TEXT NOT NULL, attempt INTEGER NOT NULL, status TEXT NOT NULL, result_json TEXT NOT NULL DEFAULT '{}', error TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(job_id, stage, cell_key, attempt))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_lifecycle_bindings (job_id TEXT PRIMARY KEY, binding_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_lifecycle_subcalls (admission_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, stage TEXT NOT NULL, cell_key TEXT NOT NULL, subcall_key TEXT NOT NULL, status TEXT NOT NULL, estimated_input_tokens INTEGER NOT NULL, estimated_output_tokens INTEGER NOT NULL, estimated_tool_calls INTEGER NOT NULL, estimated_wall_micros INTEGER NOT NULL, estimated_cost_microunits INTEGER NOT NULL, result_json TEXT NOT NULL DEFAULT '{}', error TEXT, updated_at TEXT NOT NULL, UNIQUE(job_id, stage, cell_key, subcall_key))"
            )
            conn.commit()

    def _lifecycle_budget(self, job_id: str, limits: Mapping[str, int]) -> None:
        self._ensure_lifecycle_tables()
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO evaluation_lifecycle_budget(job_id, max_attempts, max_input_tokens, max_output_tokens, max_tool_calls, max_wall_micros, max_cost_microunits, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now')) ON CONFLICT(job_id) DO NOTHING",
                (job_id, limits["attempts"], limits["inputTokens"], limits["outputTokens"], limits["toolCalls"], limits["wallMicros"], limits["costMicrounits"]),
            )
            row = conn.execute("SELECT max_attempts, max_input_tokens, max_output_tokens, max_tool_calls, max_wall_micros, max_cost_microunits FROM evaluation_lifecycle_budget WHERE job_id = ?", (job_id,)).fetchone()
            if row is None or tuple(row) != tuple(limits[key] for key in ("attempts", "inputTokens", "outputTokens", "toolCalls", "wallMicros", "costMicrounits")):
                raise EvaluationError("lifecycle budget is already bound to different limits")
            conn.commit()

    def _reserve_lifecycle_launch(self, job_id: str, stage: str, cell_key: str, attempt: int) -> bool:
        """Atomically claim one launch, enforcing cumulative attempt budget."""
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            done = conn.execute("SELECT 1 FROM evaluation_lifecycle_attempts WHERE job_id = ? AND stage = ? AND cell_key = ? AND status = 'complete'", (job_id, stage, cell_key)).fetchone()
            if done is not None:
                conn.commit()
                return False
            existing = conn.execute("SELECT status FROM evaluation_lifecycle_attempts WHERE job_id = ? AND stage = ? AND cell_key = ? AND attempt = ?", (job_id, stage, cell_key, attempt)).fetchone()
            if existing is not None:
                conn.commit()
                return False
            budget = conn.execute("SELECT * FROM evaluation_lifecycle_budget WHERE job_id = ?", (job_id,)).fetchone()
            if budget is None:
                conn.rollback()
                raise EvaluationError("lifecycle budget is not initialized")
            subcall_count = conn.execute("SELECT COUNT(*) AS count FROM evaluation_lifecycle_subcalls WHERE job_id = ?", (job_id,)).fetchone()["count"]
            if budget["blocked"] or budget["attempts"] + subcall_count >= budget["max_attempts"] or budget["input_tokens"] >= budget["max_input_tokens"] or budget["output_tokens"] >= budget["max_output_tokens"] or budget["tool_calls"] >= budget["max_tool_calls"] or budget["wall_micros"] >= budget["max_wall_micros"] or budget["cost_microunits"] >= budget["max_cost_microunits"]:
                conn.execute("UPDATE evaluation_lifecycle_budget SET blocked = 1, updated_at = datetime('now') WHERE job_id = ?", (job_id,))
                conn.commit()
                raise EvaluationError("lifecycle launch budget exhausted")
            conn.execute("INSERT INTO evaluation_lifecycle_attempts(job_id, stage, cell_key, attempt, status, updated_at) VALUES (?, ?, ?, ?, 'running', datetime('now'))", (job_id, stage, cell_key, attempt))
            conn.execute("UPDATE evaluation_lifecycle_budget SET attempts = attempts + 1, updated_at = datetime('now') WHERE job_id = ?", (job_id,))
            conn.commit()
            return True

    def _lifecycle_attempt(self, job_id: str, stage: str, cell_key: str, attempt: int) -> tuple[str, dict[str, Any]] | None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT status, result_json FROM evaluation_lifecycle_attempts WHERE job_id = ? AND stage = ? AND cell_key = ? AND attempt = ?", (job_id, stage, cell_key, attempt)).fetchone()
        if row is None:
            return None
        try:
            result = json.loads(row["result_json"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            result = {}
        return str(row["status"]), result if isinstance(result, dict) else {}

    @staticmethod
    def _validate_lifecycle_receipt(stage: str, cell_key: str, result: Mapping[str, Any]) -> None:
        if result.get("status") not in {"complete", "completed", "succeeded", "success"}:
            raise EvaluationError("lifecycle callback did not return a successful receipt")
        if result.get("stage") != stage or result.get("cellKey") != cell_key:
            raise EvaluationError("lifecycle receipt is not bound to its stage and cell")
        usage = result.get("usage")
        if not isinstance(usage, Mapping) or any(not isinstance(usage.get(key), int) or isinstance(usage.get(key), bool) or usage[key] < 0 for key in ("inputTokens", "outputTokens", "totalTokens")) or usage["totalTokens"] != usage["inputTokens"] + usage["outputTokens"]:
            raise EvaluationError("lifecycle receipt usage is malformed")
        tool_calls = result.get("toolCalls", 0)
        if not isinstance(tool_calls, int) or isinstance(tool_calls, bool) or tool_calls < 0:
            raise EvaluationError("lifecycle receipt tool calls are malformed")
        cost = result.get("costMicrounits")
        if cost is None:
            if result.get("economicCostStatus") != "unknown":
                raise EvaluationError("lifecycle receipt cost is missing")
        elif isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
            raise EvaluationError("lifecycle receipt cost is malformed")
        wall = result.get("wallSeconds", 0)
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
            raise EvaluationError("lifecycle receipt wall time is malformed")

    def _finish_lifecycle_launch(self, job_id: str, stage: str, cell_key: str, attempt: int, status: str, result: Mapping[str, Any] | None = None, error: str | None = None) -> None:
        payload = dict(result or {})
        current = self._lifecycle_attempt(job_id, stage, cell_key, attempt)
        if current is not None and current[0] in {"complete", "failed"}:
            return
        if status == "complete":
            self._validate_lifecycle_receipt(stage, cell_key, payload)
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        input_tokens = usage.get("inputTokens", 0) if usage else 0
        output_tokens = usage.get("outputTokens", 0) if usage else 0
        tool_calls = payload.get("toolCalls", 0)
        wall_seconds = payload.get("wallSeconds", 0)
        cost_value = payload.get("costMicrounits", 0)
        charged_ids = payload.get("chargedSubcallIds", ())
        if charged_ids:
            if not isinstance(charged_ids, (list, tuple)) or not all(isinstance(value, str) and value for value in charged_ids):
                raise EvaluationError("charged subcall IDs are malformed")
            if len(charged_ids) != len(set(charged_ids)):
                raise EvaluationError("charged subcall IDs must be unique")
            with self.store.connect() as conn:
                rows = conn.execute("SELECT * FROM evaluation_lifecycle_subcalls WHERE admission_id IN (%s)" % ",".join("?" for _ in charged_ids), tuple(charged_ids)).fetchall()
            if len(rows) != len(set(charged_ids)) or any(row["status"] not in {"complete", "failed"} for row in rows):
                raise EvaluationError("charged subcall IDs are not durably finalized")
            if any(row["job_id"] != job_id or row["stage"] != stage or row["cell_key"] != cell_key for row in rows):
                raise EvaluationError("charged subcall IDs are not bound to this lifecycle cell")
            charged_input = charged_output = charged_tools = charged_wall = charged_cost = 0
            for row in rows:
                try:
                    child = json.loads(row["result_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise EvaluationError("charged subcall receipt is malformed") from exc
                child_usage = child.get("usage")
                if not isinstance(child_usage, Mapping) or any(not isinstance(child_usage.get(key), int) or isinstance(child_usage.get(key), bool) or child_usage[key] < 0 for key in ("inputTokens", "outputTokens", "totalTokens")) or child_usage["totalTokens"] != child_usage["inputTokens"] + child_usage["outputTokens"]:
                    raise EvaluationError("charged subcall receipt usage is malformed")
                child_tools = child.get("toolCalls", 0)
                child_wall = child.get("wallSeconds", 0)
                child_cost = child.get("costMicrounits")
                if child.get("economicCostStatus") == "unknown" or child_cost is None or not isinstance(child_tools, int) or isinstance(child_tools, bool) or child_tools < 0 or isinstance(child_wall, bool) or not isinstance(child_wall, (int, float)) or not math.isfinite(child_wall) or child_wall < 0 or isinstance(child_cost, bool) or not isinstance(child_cost, (int, float)) or not math.isfinite(child_cost) or child_cost < 0:
                    raise EvaluationError("charged subcall receipt cost or accounting is unknown")
                charged_input += child_usage["inputTokens"]
                charged_output += child_usage["outputTokens"]
                charged_tools += child_tools
                charged_wall += int(float(child_wall) * 1_000_000)
                charged_cost += int(child_cost)
            residual = payload.get("residualUsage")
            if not isinstance(residual, Mapping) or any(not isinstance(residual.get(key), int) or isinstance(residual.get(key), bool) or residual[key] < 0 for key in ("inputTokens", "outputTokens", "totalTokens")) or residual["totalTokens"] != residual["inputTokens"] + residual["outputTokens"]:
                raise EvaluationError("residual lifecycle accounting is malformed")
            reported_residual = (residual["inputTokens"], residual["outputTokens"], payload.get("residualToolCalls", 0), payload.get("residualWallSeconds", 0), payload.get("residualCostMicrounits", 0))
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (reported_residual[0], reported_residual[1], reported_residual[2])) or isinstance(reported_residual[3], bool) or not isinstance(reported_residual[3], (int, float)) or not math.isfinite(reported_residual[3]) or reported_residual[3] < 0 or isinstance(reported_residual[4], bool) or not isinstance(reported_residual[4], (int, float)) or not math.isfinite(reported_residual[4]) or reported_residual[4] < 0:
                raise EvaluationError("residual lifecycle accounting is malformed")
            full_wall = int(float(wall_seconds or 0) * 1_000_000)
            full_cost = int(cost_value) if isinstance(cost_value, (int, float)) and not isinstance(cost_value, bool) else None
            expected_residual = (input_tokens - charged_input, output_tokens - charged_output, tool_calls - charged_tools, full_wall - charged_wall, None if full_cost is None else full_cost - charged_cost)
            actual_residual = (reported_residual[0], reported_residual[1], reported_residual[2], int(float(reported_residual[3]) * 1_000_000), int(reported_residual[4]))
            if expected_residual != actual_residual:
                raise EvaluationError("lifecycle receipt aggregate does not match charged subcalls and residual")
            input_tokens, output_tokens = residual["inputTokens"], residual["outputTokens"]
            tool_calls = payload.get("residualToolCalls", 0)
            wall_seconds = payload.get("residualWallSeconds", 0)
            cost_value = payload.get("residualCostMicrounits", 0)
        unknown_cost = cost_value is None and payload.get("economicCostStatus") == "unknown"
        if unknown_cost:
            cost_value = 0
        if not isinstance(tool_calls, int) or isinstance(tool_calls, bool) or tool_calls < 0 or isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float)) or not math.isfinite(wall_seconds) or wall_seconds < 0 or isinstance(cost_value, bool) or not isinstance(cost_value, (int, float)) or not math.isfinite(cost_value) or cost_value < 0:
            raise EvaluationError("lifecycle accounting is malformed")
        wall_micros = int(float(wall_seconds or 0) * 1_000_000)
        cost = int(cost_value) if isinstance(cost_value, (int, float)) and not isinstance(cost_value, bool) else 0
        with self.store.connect() as conn:
            conn.execute("UPDATE evaluation_lifecycle_attempts SET status = ?, result_json = ?, error = ?, updated_at = datetime('now') WHERE job_id = ? AND stage = ? AND cell_key = ? AND attempt = ?", (status, json.dumps(payload, sort_keys=True, default=str), error, job_id, stage, cell_key, attempt))
            conn.execute("UPDATE evaluation_lifecycle_budget SET input_tokens = input_tokens + ?, output_tokens = output_tokens + ?, tool_calls = tool_calls + ?, wall_micros = wall_micros + ?, cost_microunits = cost_microunits + ?, updated_at = datetime('now') WHERE job_id = ?", (input_tokens, output_tokens, tool_calls, wall_micros, cost, job_id))
            budget = conn.execute("SELECT * FROM evaluation_lifecycle_budget WHERE job_id = ?", (job_id,)).fetchone()
            if budget is not None and (
                budget["input_tokens"] > budget["max_input_tokens"]
                or budget["output_tokens"] > budget["max_output_tokens"]
                or budget["tool_calls"] > budget["max_tool_calls"]
                or budget["wall_micros"] > budget["max_wall_micros"]
                or budget["cost_microunits"] > budget["max_cost_microunits"]
                or payload.get("economicCostStatus") == "unknown"
            ):
                conn.execute("UPDATE evaluation_lifecycle_budget SET blocked = 1, updated_at = datetime('now') WHERE job_id = ?", (job_id,))
            conn.commit()

    def lifecycle_accounting(self, job_id: str) -> dict[str, Any]:
        self._ensure_lifecycle_tables()
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM evaluation_lifecycle_budget WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return {}
        with self.store.connect() as conn:
            subcalls = conn.execute("SELECT COUNT(*) AS count FROM evaluation_lifecycle_subcalls WHERE job_id = ?", (job_id,)).fetchone()["count"]
        return {"attempts": row["attempts"], "subcalls": subcalls, "inputTokens": row["input_tokens"], "outputTokens": row["output_tokens"], "toolCalls": row["tool_calls"], "wallSeconds": row["wall_micros"] / 1_000_000, "costMicrounits": row["cost_microunits"], "blocked": bool(row["blocked"]), "maxAttempts": row["max_attempts"], "maxInputTokens": row["max_input_tokens"], "maxOutputTokens": row["max_output_tokens"], "maxToolCalls": row["max_tool_calls"], "maxWallSeconds": row["max_wall_micros"] / 1_000_000, "maxCostMicrounits": row["max_cost_microunits"]}

    def admit_lifecycle_subcall(self, job_id: str, stage: str, cell_key: str, subcall_key: str, *, estimated_input_tokens: int = 0, estimated_output_tokens: int = 0, estimated_tool_calls: int = 1, estimated_wall_seconds: float = 0.0, estimated_cost_microunits: int = 0) -> dict[str, Any]:
        """Reserve one nested runtime call before dispatch, atomically."""
        values = (estimated_input_tokens, estimated_output_tokens, estimated_tool_calls, estimated_cost_microunits)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values) or not isinstance(estimated_wall_seconds, (int, float)) or isinstance(estimated_wall_seconds, bool) or not math.isfinite(estimated_wall_seconds) or estimated_wall_seconds < 0:
            raise EvaluationError("subcall admission estimates are malformed")
        admission_id = f"{job_id}:{stage}:{cell_key}:{subcall_key}"
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM evaluation_lifecycle_subcalls WHERE job_id = ? AND stage = ? AND cell_key = ? AND subcall_key = ?", (job_id, stage, cell_key, subcall_key)).fetchone()
            if existing is not None:
                conn.commit()
                # A complete admission is a durable checkpoint.  Returning
                # its stored receipt lets a reopened runtime resume without a
                # second dispatch.
                try:
                    persisted = json.loads(existing["result_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    persisted = {}
                return {"admissionId": existing["admission_id"], "status": existing["status"], "result": persisted if isinstance(persisted, dict) else {}, "error": existing["error"], "reused": True, "dispatchAllowed": False}
            budget = conn.execute("SELECT * FROM evaluation_lifecycle_budget WHERE job_id = ?", (job_id,)).fetchone()
            if budget is None:
                conn.rollback()
                raise EvaluationError("lifecycle budget is not initialized")
            reserved = conn.execute("SELECT COALESCE(SUM(estimated_input_tokens), 0) AS input, COALESCE(SUM(estimated_output_tokens), 0) AS output, COALESCE(SUM(estimated_tool_calls), 0) AS tools, COALESCE(SUM(estimated_wall_micros), 0) AS wall, COALESCE(SUM(estimated_cost_microunits), 0) AS cost, COUNT(*) AS count FROM evaluation_lifecycle_subcalls WHERE job_id = ? AND status = 'reserved'", (job_id,)).fetchone()
            subcall_count = conn.execute("SELECT COUNT(*) AS count FROM evaluation_lifecycle_subcalls WHERE job_id = ?", (job_id,)).fetchone()["count"]
            estimates = (estimated_input_tokens, estimated_output_tokens, estimated_tool_calls, int(estimated_wall_seconds * 1_000_000), estimated_cost_microunits)
            totals = (budget["input_tokens"] + reserved["input"] + estimates[0], budget["output_tokens"] + reserved["output"] + estimates[1], budget["tool_calls"] + reserved["tools"] + estimates[2], budget["wall_micros"] + reserved["wall"] + estimates[3], budget["cost_microunits"] + reserved["cost"] + estimates[4])
            caps = (budget["max_input_tokens"], budget["max_output_tokens"], budget["max_tool_calls"], budget["max_wall_micros"], budget["max_cost_microunits"])
            if budget["blocked"] or budget["attempts"] + subcall_count >= budget["max_attempts"] or any(total > cap for total, cap in zip(totals, caps)):
                conn.execute("UPDATE evaluation_lifecycle_budget SET blocked = 1, updated_at = datetime('now') WHERE job_id = ?", (job_id,))
                conn.commit()
                raise EvaluationError("lifecycle subcall budget exhausted")
            conn.execute("INSERT INTO evaluation_lifecycle_subcalls(admission_id, job_id, stage, cell_key, subcall_key, status, estimated_input_tokens, estimated_output_tokens, estimated_tool_calls, estimated_wall_micros, estimated_cost_microunits, updated_at) VALUES (?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?, ?, datetime('now'))", (admission_id, job_id, stage, cell_key, subcall_key, *estimates))
            conn.commit()
        return {"admissionId": admission_id, "status": "reserved", "reused": False, "dispatchAllowed": True}

    def record_lifecycle_subcall(self, admission_id: str, *, result: Mapping[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
        """Persist nested-call receipt exactly once, including failed usage."""
        payload = dict(result or {})
        with self.store.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM evaluation_lifecycle_subcalls WHERE admission_id = ?", (admission_id,)).fetchone()
            if row is None:
                conn.rollback()
                raise EvaluationError("unknown lifecycle subcall admission")
            if row["status"] in {"complete", "failed"}:
                try:
                    persisted = json.loads(row["result_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    persisted = {}
                conn.commit()
                return {"admissionId": admission_id, "status": row["status"], "result": persisted if isinstance(persisted, dict) else {}, "error": row["error"], "reused": True, "dispatchAllowed": False}
            if result is None and error:
                # A lost receipt leaves spend unknown.  The reservation cannot
                # be released based on an operator-supplied error string.
                conn.execute("UPDATE evaluation_lifecycle_subcalls SET status = 'failed', error = ?, updated_at = datetime('now') WHERE admission_id = ?", (error, admission_id))
                conn.execute("UPDATE evaluation_lifecycle_budget SET blocked = 1, updated_at = datetime('now') WHERE job_id = ?", (row["job_id"],))
                conn.commit()
                return {"admissionId": admission_id, "status": "failed", "result": {}, "error": error, "reused": False, "dispatchAllowed": False}
            usage = payload.get("usage")
            malformed = not isinstance(usage, Mapping) or any(not isinstance(usage.get(key), int) or isinstance(usage.get(key), bool) or usage[key] < 0 for key in ("inputTokens", "outputTokens", "totalTokens")) or usage["totalTokens"] != usage["inputTokens"] + usage["outputTokens"]
            cost = payload.get("costMicrounits")
            unknown_cost = cost is None and payload.get("economicCostStatus") == "unknown"
            tool_calls = payload.get("toolCalls", 0)
            wall_seconds = payload.get("wallSeconds", 0)
            malformed = malformed or not isinstance(tool_calls, int) or isinstance(tool_calls, bool) or tool_calls < 0 or isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float)) or not math.isfinite(wall_seconds) or wall_seconds < 0 or (cost is None and not unknown_cost) or (cost is not None and (isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0))
            if malformed:
                conn.execute("UPDATE evaluation_lifecycle_subcalls SET status = 'failed', error = ?, updated_at = datetime('now') WHERE admission_id = ?", (error or "malformed subcall accounting", admission_id))
                conn.execute("UPDATE evaluation_lifecycle_budget SET blocked = 1, updated_at = datetime('now') WHERE job_id = ?", (row["job_id"],))
                conn.commit()
                raise EvaluationError("lifecycle subcall accounting is malformed")
            status = "failed" if error else "complete"
            cost_value = int(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0
            conn.execute("UPDATE evaluation_lifecycle_subcalls SET status = ?, result_json = ?, error = ?, updated_at = datetime('now') WHERE admission_id = ?", (status, json.dumps(payload, sort_keys=True, default=str), error, admission_id))
            conn.execute("UPDATE evaluation_lifecycle_budget SET input_tokens = input_tokens + ?, output_tokens = output_tokens + ?, tool_calls = tool_calls + ?, wall_micros = wall_micros + ?, cost_microunits = cost_microunits + ?, blocked = CASE WHEN ? THEN 1 ELSE blocked END, updated_at = datetime('now') WHERE job_id = ?", (usage["inputTokens"], usage["outputTokens"], tool_calls, int(float(wall_seconds or 0) * 1_000_000), cost_value, unknown_cost, row["job_id"]))
            conn.commit()
        return {"admissionId": admission_id, "status": status, "reused": False, "dispatchAllowed": False}

    def run_experiment(self, job_id: str, stages: Sequence[LifecycleStage], *, limits: Mapping[str, int] | None = None, context: Mapping[str, Any] | None = None) -> EvaluationJobResult:
        """Run the complete ordered lifecycle with durable per-cell resume.

        Callbacks are runtime-owned.  This evaluator seam only claims launches,
        records all attempts, and prevents held-out stages from being skipped.
        """
        names = tuple(stage.name for stage in stages)
        if names != LIFECYCLE_STAGE_ORDER:
            raise EvaluationError(f"lifecycle stages must be ordered as {LIFECYCLE_STAGE_ORDER}")
        default_budget = self.protocol.run_budget
        bound_limits = dict(limits or {
            "attempts": sum(len(stage.cells) * (stage.retries + 1) for stage in stages),
            "inputTokens": default_budget.model_tokens * sum(len(stage.cells) for stage in stages),
            "outputTokens": default_budget.model_tokens * sum(len(stage.cells) for stage in stages),
            "toolCalls": default_budget.tool_calls * sum(len(stage.cells) for stage in stages),
            "wallMicros": default_budget.wall_time_seconds * 1_000_000 * sum(len(stage.cells) for stage in stages),
            "costMicrounits": default_budget.cost_microunits * sum(len(stage.cells) for stage in stages),
        })
        required = {"attempts", "inputTokens", "outputTokens", "toolCalls", "wallMicros", "costMicrounits"}
        if set(bound_limits) != required or any(not isinstance(value, int) or value < 0 for value in bound_limits.values()):
            raise EvaluationError("lifecycle limits must be non-negative integer totals")
        self._lifecycle_budget(job_id, bound_limits)
        state = dict(context or {})
        # Resume is an operational control, not an experiment input.  Keep it
        # available to runtime callbacks while excluding it from the immutable
        # binding so initialize and resume invocations can reopen the same job.
        immutable_context = {
            key: value
            for key, value in state.items()
            if key not in _LIFECYCLE_OPERATIONAL_CONTEXT_KEYS
        }
        binding = {"protocol": self.protocol.start_candidate_generation().to_dict(), "context": immutable_context, "stages": [{"name": stage.name, "cells": list(stage.cells), "retries": stage.retries} for stage in stages]}
        encoded_binding = json.dumps(binding, sort_keys=True, default=str)
        with self.store.connect() as conn:
            existing_binding = conn.execute("SELECT binding_json FROM evaluation_lifecycle_bindings WHERE job_id = ?", (job_id,)).fetchone()
            if existing_binding is None:
                conn.execute("INSERT INTO evaluation_lifecycle_bindings(job_id, binding_json, updated_at) VALUES (?, ?, datetime('now'))", (job_id, encoded_binding))
            elif existing_binding["binding_json"] != encoded_binding:
                raise EvaluationError("lifecycle stages or frozen inputs are already bound differently")
            conn.commit()
        with self.store.connect() as conn:
            completed_stages = []
            for stage in stages:
                rows = conn.execute("SELECT cell_key, result_json FROM evaluation_lifecycle_attempts WHERE job_id = ? AND stage = ? AND status = 'complete'", (job_id, stage.name)).fetchall()
                completed_cells = {row["cell_key"] for row in rows}
                if completed_cells == set(stage.cells) and len(rows) == len(stage.cells):
                    completed_stages.append(stage.name)
                for row in rows:
                    try:
                        value = json.loads(row["result_json"] or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        value = {}
                    if isinstance(value, dict):
                        state.setdefault("results", {}).setdefault(stage.name, {})[row["cell_key"]] = value
        state["completedStages"] = tuple(completed_stages)
        started = time.monotonic()
        self._save(job_id, "experiment", "running", runtime_accounting=self.lifecycle_accounting(job_id))
        def run_cell(stage: LifecycleStage, cell_key: str, cell_state: Mapping[str, Any]) -> Mapping[str, Any]:
            result: Mapping[str, Any] | None = None
            for attempt in range(stage.retries + 1):
                try:
                    prior = self._lifecycle_attempt(job_id, stage.name, cell_key, attempt)
                    if prior is not None:
                        if prior[0] == "complete":
                            self._validate_lifecycle_receipt(stage.name, cell_key, prior[1])
                            result = prior[1]
                            break
                        if prior[0] == "running":
                            raise EvaluationError(f"lifecycle attempt is still running: {stage.name}/{cell_key}/{attempt}")
                        if prior[0] == "failed":
                            if attempt < stage.retries:
                                continue
                            raise EvaluationError(f"lifecycle attempt failed without a declared retry: {stage.name}/{cell_key}")
                    if not self._reserve_lifecycle_launch(job_id, stage.name, cell_key, attempt):
                        prior = self._lifecycle_attempt(job_id, stage.name, cell_key, attempt)
                        if prior is None or prior[0] != "complete":
                            raise EvaluationError(f"lifecycle launch was not reusable: {stage.name}/{cell_key}/{attempt}")
                        self._validate_lifecycle_receipt(stage.name, cell_key, prior[1])
                        result = prior[1]
                        break
                    callback_context = {
                        **copy.deepcopy(dict(cell_state)),
                        "stage": stage.name,
                        "attempt": attempt,
                        "admitSubcall": lambda subcall_key, **estimates: self.admit_lifecycle_subcall(job_id, stage.name, cell_key, subcall_key, **estimates),
                        "recordSubcall": self.record_lifecycle_subcall,
                    }
                    result = stage.callback(cell_key, callback_context)
                    if not isinstance(result, Mapping):
                        raise EvaluationError("lifecycle callback must return an object")
                    self._validate_lifecycle_receipt(stage.name, cell_key, result)
                    self._finish_lifecycle_launch(job_id, stage.name, cell_key, attempt, "complete", result)
                    break
                except Exception as exc:
                    self._finish_lifecycle_launch(job_id, stage.name, cell_key, attempt, "failed", None, str(exc))
                    if attempt >= stage.retries:
                        raise
            if result is None:
                raise EvaluationError(f"lifecycle cell produced no receipt: {stage.name}/{cell_key}")
            return result

        try:
            for stage in stages:
                if stage.name in {"validation", "final"} and any(name not in state.get("completedStages", ()) for name in ("bootstrap", "training", "learning", "transfer", "adaptation", "safety") if name != stage.name):
                    raise EvaluationError("held-out stage reached before complete development lifecycle")
                stage_state = copy.deepcopy(state)
                if stage.name in {"training", "validation", "final"} and len(stage.cells) > 1 and self.protocol.concurrency_limit > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.protocol.concurrency_limit, thread_name_prefix=f"lifecycle-{stage.name}") as executor:
                        completed = list(executor.map(lambda key: (key, run_cell(stage, key, stage_state)), stage.cells))
                else:
                    completed = [(cell_key, run_cell(stage, cell_key, stage_state)) for cell_key in stage.cells]
                for cell_key, result in completed:
                    state.setdefault("results", {}).setdefault(stage.name, {})[cell_key] = dict(result)
                state["completedStages"] = tuple((*state.get("completedStages", ()), stage.name))
            accounting = self.lifecycle_accounting(job_id)
            report = self._experiment_report(stages, state)
            runtime_accounting = {**accounting, "wallDurationSeconds": time.monotonic() - started}
            self._save(job_id, "experiment", "complete", report=report, runtime_accounting=runtime_accounting)
            return EvaluationJobResult(job_id, "experiment", "complete", report, runtime_accounting=runtime_accounting)
        except Exception as exc:
            accounting = self.lifecycle_accounting(job_id)
            self._save(job_id, "experiment", "failed", error=str(exc), runtime_accounting={**accounting, "wallDurationSeconds": time.monotonic() - started})
            return EvaluationJobResult(job_id, "experiment", "failed", None, error=str(exc), runtime_accounting={**accounting, "wallDurationSeconds": time.monotonic() - started})

    def _experiment_report(self, stages: Sequence[LifecycleStage], state: Mapping[str, Any]) -> EvaluationReport | None:
        """Build a strict report from durable validation/final lifecycle receipts."""
        panel_stages = [stage for stage in stages if stage.name in {"validation", "final"} and stage.observation_recoverer is not None]
        recoverers = [stage.observation_recoverer for stage in panel_stages if stage.observation_recoverer is not None]
        if not recoverers:
            return None
        if len(set(id(recoverer) for recoverer in recoverers)) != 1:
            raise EvaluationError("lifecycle observation recovery seam is inconsistent")
        recover = recoverers[0]
        results = state.get("results")
        if not isinstance(results, Mapping):
            raise EvaluationError("lifecycle results are missing for report assembly")
        selected = next((stage for stage in reversed(panel_stages) if isinstance(results.get(stage.name), Mapping) and results[stage.name]), None)
        if selected is None:
            raise EvaluationError("lifecycle produced no persisted validation or final receipts")
        stage_results = results[selected.name]
        observations: list[Any] = []
        for cell_key in selected.cells:
            receipt = stage_results.get(cell_key)
            if not isinstance(receipt, Mapping):
                raise EvaluationError(f"missing persisted {selected.name} receipt: {cell_key}")
            recovered = recover(receipt, stage=selected.name, cell_key=cell_key)
            if not isinstance(recovered, Sequence) or isinstance(recovered, (str, bytes)):
                raise EvaluationError("lifecycle observation recovery returned a malformed collection")
            observations.extend(recovered)
        if not observations:
            raise EvaluationError(f"{selected.name} produced no durable evaluation observations")
        first_receipt = stage_results[selected.cells[0]]
        pins = first_receipt.get("pins") if isinstance(first_receipt, Mapping) else None
        if not isinstance(pins, Mapping) or not isinstance(pins.get("baseBundleHash"), str):
            raise EvaluationError("lifecycle report lacks pinned base bundle hash")
        base_hash = pins["baseBundleHash"]
        learning_results = results.get("learning")
        candidate_hash = None
        if isinstance(learning_results, Mapping):
            for value in learning_results.values():
                if isinstance(value, Mapping) and isinstance(value.get("candidateBundleHash"), str):
                    candidate_hash = value["candidateBundleHash"]
                    break
        if candidate_hash is None:
            candidate = self.arm_bundles.get(Arm.L, self.arm_bundles.get(Arm.L.value))
            candidate_hash = getattr(candidate, "content_hash", None)
        if not isinstance(candidate_hash, str) or not candidate_hash:
            raise EvaluationError("lifecycle report lacks pinned candidate bundle hash")
        evaluator = build_durable_evaluation_runner(self.protocol, self.packages, self.store, probe_executor=self.controller)
        ablation_audit = None
        if selected.name == "final":
            ablation = self.arm_bundles.get(Arm.A, self.arm_bundles.get(Arm.A.value))
            if ablation is None:
                raise EvaluationError("final lifecycle report lacks pinned ablation bundle")
            execution = getattr(ablation, "execution_config", None)
            procedures = tuple(skill.procedure for skill in getattr(ablation, "skills", ()))
            if execution is None:
                raise EvaluationError("final lifecycle report has malformed ablation bundle")
            ablation_input = AblationInput(
                getattr(ablation, "content_hash", ""),
                json.dumps(execution.model_dump(mode="json", by_alias=True), sort_keys=True),
                procedures,
            )
            from adaptive_agent.evaluation import audit_ablation

            ablation_audit = audit_ablation(ablation_input)
        frozen = self.protocol.start_candidate_generation()
        return evaluator.report_from_observations(
            comparison=selected.name,
            base_hash=base_hash,
            candidate_hash=candidate_hash,
            observations=observations,
            expected_partitions=frozen.partition_hashes,
            ablation_audit=ablation_audit,
        )

    def planned_workload(self, candidate_count: int = 1, *, training_runs: int | None = None, transfer_runs: int = 0, safety_runs: int = 0, retries: int = 0):
        return self.protocol.workload(candidate_count, training_runs=training_runs, transfer_runs=transfer_runs, safety_runs=safety_runs, retries=retries)

    def _preflight(self, comparison: str, workload) -> None:
        if comparison not in {"validation", "final"}:
            raise EvaluationError("comparison must be validation or final")
        required = (Arm.B0, Arm.L) if comparison == "validation" else (Arm.B0, Arm.L, Arm.A)
        missing = [arm.value for arm in required if arm not in self.arm_bundles and arm.value not in self.arm_bundles]
        if missing:
            raise EvaluationError(f"missing expected arm bundles before evaluation: {', '.join(missing)}")
        if workload.validation_per_candidate != self.protocol.validation_run_count or workload.final_runs != self.protocol.final_run_count:
            raise EvaluationError("workload panel counts do not match the frozen protocol")
        if self.max_total_attempts is not None and workload.total_attempted_runs > self.max_total_attempts:
            raise EvaluationError("overall evaluation workload exhausted")
        if self.total_budget_microunits is not None:
            if self.total_budget_microunits < 0:
                raise EvaluationError("total budget cannot be negative")
            required_cost = workload.total_attempted_runs * self.protocol.run_budget.cost_microunits
            if required_cost > self.total_budget_microunits:
                raise EvaluationError("evaluation budget cannot cover the immutable panel")

    def _save(self, job_id: str, comparison: str, status: str, report: EvaluationReport | None = None, error: str | None = None, runtime_accounting: Mapping[str, Any] | None = None) -> None:
        report_ref = None
        if report is not None:
            report_ref = self.store.put_artifact(report.to_dict()).sha256
        accounting_json = json.dumps(dict(runtime_accounting or {}), sort_keys=True)
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO evaluation_jobs(job_id, comparison, status, report_ref, error, runtime_accounting_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now')) ON CONFLICT(job_id) DO UPDATE SET comparison=excluded.comparison, status=excluded.status, report_ref=excluded.report_ref, error=excluded.error, runtime_accounting_json=excluded.runtime_accounting_json, updated_at=excluded.updated_at",
                (job_id, comparison, status, report_ref, error, accounting_json),
            )
            conn.commit()

    def readback(self, job_id: str) -> EvaluationJobResult | None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM evaluation_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        report = None
        if row["report_ref"]:
            payload = self.store.get_artifact(row["report_ref"])
            if isinstance(payload, dict):
                report = payload  # serialized readback is intentionally opaque
        accounting = json.loads(row["runtime_accounting_json"] or "{}") if "runtime_accounting_json" in row.keys() else {}
        return EvaluationJobResult(job_id, row["comparison"], row["status"], report, error=row["error"], runtime_accounting=accounting)

    def run(self, job_id: str, comparison: str, *, base_hash: str, candidate_hash: str, candidate_id: str | None = None, ablation: AblationInput | None = None, candidate_count: int = 1, training_runs: int | None = None, transfer_runs: int = 0, safety_runs: int = 0, retries: int = 0) -> EvaluationJobResult:
        workload = self.planned_workload(candidate_count, training_runs=training_runs, transfer_runs=transfer_runs, safety_runs=safety_runs, retries=retries)
        self._preflight(comparison, workload)
        if comparison == "final" and ablation is None:
            raise EvaluationError("final evaluation requires pinned ablation input")
        frozen = self.protocol.start_candidate_generation()
        self._save(job_id, comparison, "running")
        driver = ResumableEvaluationDriver(
            self.store,
            self.protocol,
            self.packages,
            self.execute,
            self.arm_bundles.get(Arm.B0, self.arm_bundles.get(Arm.B0.value)),
            evidence_store=SQLiteRunEvidenceStore(self.store),
            arm_bundles=self.arm_bundles,
        )
        report: EvaluationReport | None = None
        observations: tuple[RunObservation, ...] = ()
        started = time.monotonic()
        try:
            summary = driver.run(job_id, Partition.VALIDATION if comparison == "validation" else Partition.FINAL, base_hash=base_hash, candidate_hash=candidate_hash)
            observations = tuple(item.observation for item in summary.statuses if item.observation is not None)
            evaluator = build_durable_evaluation_runner(self.protocol, self.packages, self.store, probe_executor=self.controller)
            report = evaluator.report_from_observations(
                comparison=comparison,
                base_hash=base_hash,
                candidate_hash=candidate_hash,
                observations=observations,
                expected_partitions=frozen.partition_hashes,
                ablation_audit=None if comparison == "validation" else __import__("adaptive_agent.evaluation", fromlist=["audit_ablation"]).audit_ablation(ablation) if ablation is not None else None,
            )
            accounting = self._runtime_accounting(observations, time.monotonic() - started)
            report = replace(
                report,
                actual_input_tokens=accounting["inputTokens"],
                actual_output_tokens=accounting["outputTokens"],
                nominal_cost_usd=accounting["nominalCostUsd"],
                wall_duration_seconds=accounting["wallDurationSeconds"],
                billing_basis=accounting["billingBasis"],
            )
            if summary.complete and report.validity_status == "valid" and report.promotion_eligible and candidate_id is not None:
                report.require_promotion_evidence(self.protocol, self.packages)
                decision = self.controller.candidates.promote(candidate_id, report)
                self._save(job_id, comparison, "decided", report, runtime_accounting=accounting)
                return EvaluationJobResult(job_id, comparison, "decided", report, decision, runtime_accounting=accounting)
            status = "complete" if report.validity_status == "valid" else "incomplete"
            self._save(job_id, comparison, status, report, runtime_accounting=accounting)
            return EvaluationJobResult(job_id, comparison, status, report, runtime_accounting=accounting)
        except Exception as exc:
            accounting = self._runtime_accounting(observations, time.monotonic() - started)
            self._save(job_id, comparison, "failed", report=report, error=str(exc), runtime_accounting=accounting)
            return EvaluationJobResult(job_id, comparison, "failed", report, error=str(exc), runtime_accounting=accounting)

    def _runtime_accounting(self, observations: tuple[RunObservation, ...], wall_seconds: float) -> dict[str, Any]:
        input_tokens = output_tokens = 0
        nominal_cost = 0.0
        nominal_seen = False
        economic_cost_microunits = 0.0
        economic_cost_seen = False
        inference_duration_seconds = 0.0
        inference_duration_seen = False
        economic_statuses: set[str] = set()
        for observation in observations:
            if not observation.accounting_ref:
                continue
            accounting = self.store.get_artifact(observation.accounting_ref)
            usage = {}
            if isinstance(accounting, dict):
                candidate_usage = accounting.get("aggregateUsage", accounting.get("usage", {}))
                if isinstance(candidate_usage, dict):
                    usage = candidate_usage
            input_tokens += int(usage.get("inputTokens", 0) or 0)
            output_tokens += int(usage.get("outputTokens", 0) or 0)
            if isinstance(accounting, dict):
                economic = accounting.get("economicCost")
                if isinstance(economic, dict):
                    status = economic.get("status")
                    if isinstance(status, str):
                        economic_statuses.add(status)
                    value = economic.get("microunits")
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        economic_cost_microunits += float(value)
                        economic_cost_seen = True
                duration = accounting.get("inferenceDurationSeconds")
                if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0:
                    inference_duration_seconds += float(duration)
                    inference_duration_seen = True
            evidence = self.store.get_evidence(observation.evidence_ref) if observation.evidence_ref else None
            if not evidence:
                continue
            try:
                source = json.loads(evidence["source_ref"])
                response = self.store.get_artifact(source["sha256"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                response = None
            if isinstance(response, dict):
                value = response.get("nominalCostUsd")
                usage_cost = response.get("usage", {}).get("cost", {}) if isinstance(response.get("usage"), dict) else {}
                if value is None and isinstance(usage_cost, dict):
                    value = usage_cost.get("total")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    nominal_cost += float(value)
                    nominal_seen = True
        if nominal_seen:
            billing_basis = "SDK nominal usage cost; subscription billing is separate and unmeasured"
        elif economic_statuses:
            billing_basis = f"SDK economic cost status: {', '.join(sorted(economic_statuses))}; nominal USD cost unavailable"
        else:
            billing_basis = "SDK nominal usage cost unavailable; subscription billing is separate and unmeasured"
        return {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
            "nominalCostUsd": nominal_cost if nominal_seen else None,
            "economicCostMicrounits": economic_cost_microunits if economic_cost_seen else None,
            "economicCostStatuses": sorted(economic_statuses),
            "inferenceDurationSeconds": inference_duration_seconds if inference_duration_seen else None,
            "wallDurationSeconds": wall_seconds,
            "billingBasis": billing_basis,
        }

    def run_development_smoke(self, job_id: str):
        """Run the one-task trusted development receipt used by held-out gates."""
        driver = ResumableEvaluationDriver(
            self.store,
            self.protocol,
            self.packages,
            self.execute,
            self.arm_bundles.get(Arm.B0, self.arm_bundles.get(Arm.B0.value)),
            evidence_store=SQLiteRunEvidenceStore(self.store),
            arm_bundles=self.arm_bundles,
        )
        return driver.run_development_smoke(job_id)


def build_evaluation_job(store: Store, controller: Controller, protocol: EvaluationProtocol, packages: Mapping[str, EnvironmentPackage], arm_bundles: Mapping[Arm | str, object], execute: TrustedTaskExecutor, *, total_budget_microunits: int | None = None, max_total_attempts: int | None = None) -> EvaluationJob:
    """Concrete factory binding the trusted executor to the production job."""
    return EvaluationJob(store, controller, protocol, packages, arm_bundles, execute, total_budget_microunits=total_budget_microunits, max_total_attempts=max_total_attempts)


def run_evaluation_job(job: EvaluationJob, job_id: str, comparison: str, *, base_hash: str, candidate_hash: str, candidate_id: str | None = None, ablation: AblationInput | None = None, candidate_count: int = 1, training_runs: int | None = None, transfer_runs: int = 0, safety_runs: int = 0, retries: int = 0) -> EvaluationJobResult:
    return job.run(job_id, comparison, base_hash=base_hash, candidate_hash=candidate_hash, candidate_id=candidate_id, ablation=ablation, candidate_count=candidate_count, training_runs=training_runs, transfer_runs=transfer_runs, safety_runs=safety_runs, retries=retries)


def main(argv: list[str] | None = None) -> int:
    """Run a production job from an application-owned, fully bound factory.

    The factory must return an ``EvaluationJob`` with the frozen protocol,
    fixture packages, arm bundles, and trusted runtime executor already bound.
    The CLI deliberately cannot reconstruct those dependencies from a Store.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--comparison", choices=("validation", "final"))
    parser.add_argument("--base-hash", default="")
    parser.add_argument("--candidate-hash", default="")
    parser.add_argument("--candidate-id")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    module_name, separator, function_name = args.factory.partition(":")
    if not separator:
        parser.error("--factory must be module:callable")
    factory = getattr(importlib.import_module(module_name), function_name)
    try:
        job = factory(Store(args.store))
    except Exception as exc:
        print(f"evaluation factory failed closed: {exc}")
        return 2
    if not isinstance(job, EvaluationJob):
        print("evaluation factory failed closed: it must return a fully bound EvaluationJob")
        return 2
    if args.smoke:
        result = job.run_development_smoke(args.job)
        print(result)
        return 0 if result.complete else 1
    if args.comparison is None:
        parser.error("--comparison is required unless --smoke is used")
    result = run_evaluation_job(job, args.job, args.comparison, base_hash=args.base_hash, candidate_hash=args.candidate_hash, candidate_id=args.candidate_id)
    print(result.status)
    return 0 if result.status in {"complete", "decided"} else 1


__all__ = ["EvaluationJob", "EvaluationJobResult", "LifecycleStage", "LIFECYCLE_STAGE_ORDER", "build_evaluation_job", "main", "run_evaluation_job"]


if __name__ == "__main__":
    raise SystemExit(main())
