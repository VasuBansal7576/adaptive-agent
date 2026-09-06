"""Durable, bounded development diagnostics for candidate comparisons."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class DevelopmentDiagnosticManager:
    """Own six-cell B0/L development comparisons outside promotion state."""

    def __init__(self, runtime: Any, executor: Callable[[Any, Any, Any], Any] | None = None) -> None:
        self.runtime = runtime
        self.store = runtime.controller.store
        self.executor = executor
        self._events: dict[str, threading.Event] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._ensure_tables()
        with self.store.connect() as conn:
            now = _now()
            conn.execute("UPDATE diagnostics SET state = CASE WHEN cancel_requested = 1 THEN 'cancelled' ELSE 'queued' END, error = CASE WHEN cancel_requested = 1 THEN 'cancellation requested before restart' ELSE error END, updated_at = ? WHERE state = 'running'", (now,))
            conn.execute("UPDATE diagnostic_cells SET status = CASE WHEN EXISTS (SELECT 1 FROM diagnostics WHERE diagnostics.diagnostic_id = diagnostic_cells.diagnostic_id AND diagnostics.cancel_requested = 1) THEN 'cancelled' ELSE 'queued' END, updated_at = ? WHERE status = 'running'", (now,))
            conn.commit()

    def _ensure_tables(self) -> None:
        with self.store.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS diagnostics (
                    diagnostic_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    base_bundle_hash TEXT NOT NULL,
                    candidate_bundle_hash TEXT NOT NULL,
                    protocol_hash TEXT NOT NULL,
                    environment_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    limits_json TEXT NOT NULL,
                    promotion_eligible INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS diagnostic_cells (
                    diagnostic_id TEXT NOT NULL,
                    cell_key TEXT NOT NULL,
                    arm TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    receipt_ref TEXT,
                    failure_class TEXT,
                    error TEXT,
                    started_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (diagnostic_id, cell_key)
                );
                """
            )
            conn.commit()

    @staticmethod
    def _bundle(runtime: Any, content_hash: str) -> Any:
        row = runtime.controller.store.get_bundle_by_hash(content_hash)
        if row is None:
            raise ValueError("bundle is not durable")
        from adaptive_agent.models import SkillBundle

        try:
            return SkillBundle.model_validate(json.loads(row["bundle_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("bundle is malformed") from exc

    def _record(self, diagnostic_id: str) -> dict[str, Any]:
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM diagnostics WHERE diagnostic_id = ?", (diagnostic_id,)).fetchone()
        if row is None:
            raise KeyError("diagnostic not found")
        record = dict(row)
        record["armSummaries"] = self._summaries(diagnostic_id)
        return record

    @staticmethod
    def _public(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "diagnosticId": record["diagnostic_id"],
            "candidateId": record["candidate_id"],
            "baseBundleHash": record["base_bundle_hash"],
            "candidateBundleHash": record["candidate_bundle_hash"],
            "protocolHash": record["protocol_hash"],
            "state": record["state"],
            "completedCells": int(record.get("completed_cells", 0)),
            "totalCells": 6,
            "startedAt": record.get("started_at"),
            "updatedAt": record["updated_at"],
            "armSummaries": record.get("armSummaries", []),
            "error": record.get("error"),
            "promotionEligible": False,
            "purpose": "development_diagnostic",
        }

    def _summaries(self, diagnostic_id: str) -> list[dict[str, Any]]:
        with self.store.connect() as conn:
            rows = conn.execute("SELECT arm, status, result_json FROM diagnostic_cells WHERE diagnostic_id = ? ORDER BY arm, cell_key", (diagnostic_id,)).fetchall()
        summaries = []
        for arm in ("B0", "L"):
            completed = successes = total_tokens = 0
            wall = 0.0
            for row in rows:
                if row["arm"] != arm or row["status"] not in {"completed", "failed"}:
                    continue
                completed += 1
                try:
                    value = json.loads(row["result_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    value = {}
                if value.get("passed") is True:
                    successes += 1
                usage = value.get("usage") if isinstance(value, Mapping) else None
                if isinstance(usage, Mapping) and isinstance(usage.get("totalTokens"), int):
                    total_tokens += usage["totalTokens"]
                if isinstance(value.get("wallDurationSeconds"), (int, float)):
                    wall += float(value["wallDurationSeconds"])
            summaries.append({"arm": arm, "completed": completed, "successes": successes, "meanScore": successes / completed if completed else None, "totalTokens": total_tokens, "wallDurationSeconds": wall})
        return summaries

    def _record_with_counts(self, diagnostic_id: str) -> dict[str, Any]:
        record = self._record(diagnostic_id)
        with self.store.connect() as conn:
            completed = conn.execute("SELECT COUNT(*) FROM diagnostic_cells WHERE diagnostic_id = ? AND status IN ('completed', 'failed')", (diagnostic_id,)).fetchone()[0]
            record["completed_cells"] = completed
        return self._public(record)

    def create(self, candidate_id: str, base_bundle_hash: str) -> dict[str, Any]:
        protocol = getattr(self.runtime, "_evaluation_protocol", None)
        if protocol is None:
            raise ValueError("evaluation protocol is not frozen")
        frozen = protocol.start_candidate_generation()
        active = self.runtime.controller.get_active_bundle()
        candidate = self.runtime.controller.get_candidate(candidate_id)
        if candidate is None:
            raise KeyError("candidate not found")
        if candidate.get("state") not in {"validated", "evaluating"}:
            raise ValueError("candidate is not validated")
        candidate_hash = candidate.get("candidate_bundle_hash")
        if not isinstance(candidate_hash, str) or not candidate_hash:
            try:
                candidate_hash = json.loads(candidate["candidate_json"]).get("candidateBundleHash")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                candidate_hash = None
        if not isinstance(candidate_hash, str) or not candidate_hash:
            raise ValueError("candidate bundle is not durable")
        if candidate.get("base_bundle_hash") != base_bundle_hash:
            raise ValueError("candidate base does not match requested base")
        if active is None or active.content_hash != base_bundle_hash:
            raise ValueError("diagnostic base is stale")
        self._bundle(self.runtime, base_bundle_hash)
        self._bundle(self.runtime, candidate_hash)
        with self.store.connect() as conn:
            existing = conn.execute("SELECT diagnostic_id FROM diagnostics WHERE candidate_id = ? AND base_bundle_hash = ? AND candidate_bundle_hash = ? AND protocol_hash = ? ORDER BY updated_at DESC LIMIT 1", (candidate_id, base_bundle_hash, candidate_hash, frozen.protocol_hash)).fetchone()
        if existing is not None:
            self._events.setdefault(existing["diagnostic_id"], threading.Event())
            self._locks.setdefault(existing["diagnostic_id"], threading.Lock())
            return self.get(existing["diagnostic_id"])
        environment_id = tuple(protocol.known_environments)[0]
        # Select the first public development task for every known environment.
        tasks = []
        for name in tuple(protocol.known_environments):
            values = tuple(self.runtime.packages[name].tasks_for_partition("development"))
            if not values:
                raise ValueError(f"environment {name!r} has no development tasks")
            tasks.append((name, values[0]))
        diagnostic_id = f"diag_{uuid.uuid4().hex}"
        now = _now()
        limits = dict(getattr(frozen, "inputs", {}).get("runBudget", {}))
        with self.store.connect() as conn:
            conn.execute("INSERT INTO diagnostics(diagnostic_id, candidate_id, base_bundle_hash, candidate_bundle_hash, protocol_hash, environment_id, task_id, seed, state, updated_at, limits_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)", (diagnostic_id, candidate_id, base_bundle_hash, candidate_hash, frozen.protocol_hash, environment_id, tasks[0][1].task_id, int(protocol.seeds[0]), now, json.dumps(limits, sort_keys=True)))
            for name, task in tasks:
                for arm in ("B0", "L"):
                    conn.execute("INSERT INTO diagnostic_cells(diagnostic_id, cell_key, arm, status, updated_at) VALUES (?, ?, ?, 'queued', ?)", (diagnostic_id, f"{name}:{arm}", arm, now))
            conn.commit()
        self._events[diagnostic_id] = threading.Event()
        self._locks[diagnostic_id] = threading.Lock()
        return self._record_with_counts(diagnostic_id)

    def list(self) -> list[dict[str, Any]]:
        with self.store.connect() as conn:
            ids = [row["diagnostic_id"] for row in conn.execute("SELECT diagnostic_id FROM diagnostics ORDER BY updated_at DESC").fetchall()]
        return [self._record_with_counts(value) for value in ids]

    def get(self, diagnostic_id: str) -> dict[str, Any]:
        return self._record_with_counts(diagnostic_id)

    def cancel(self, diagnostic_id: str) -> dict[str, Any]:
        event = self._events.setdefault(diagnostic_id, threading.Event())
        event.set()
        now = _now()
        with self.store.connect() as conn:
            row = conn.execute("SELECT state FROM diagnostics WHERE diagnostic_id = ?", (diagnostic_id,)).fetchone()
            if row is None:
                raise KeyError("diagnostic not found")
            if row["state"] in {"completed", "failed", "cancelled"}:
                return self.get(diagnostic_id)
            conn.execute("UPDATE diagnostics SET cancel_requested = 1, state = CASE WHEN state = 'queued' THEN 'cancelled' ELSE state END, error = CASE WHEN state = 'queued' THEN 'cancelled before dispatch' ELSE 'cancellation requested; admitted cells will finish' END, updated_at = ? WHERE diagnostic_id = ?", (now, diagnostic_id))
            conn.execute("UPDATE diagnostic_cells SET status = 'cancelled', updated_at = ? WHERE diagnostic_id = ? AND status = 'queued'", (now, diagnostic_id))
            conn.commit()
        return self.get(diagnostic_id)

    def run(self, diagnostic_id: str) -> dict[str, Any]:
        lock = self._locks.setdefault(diagnostic_id, threading.Lock())
        if not lock.acquire(blocking=False):
            return self.get(diagnostic_id)
        try:
            with self.store.connect() as conn:
                row = conn.execute("SELECT * FROM diagnostics WHERE diagnostic_id = ?", (diagnostic_id,)).fetchone()
                if row is None:
                    raise KeyError("diagnostic not found")
                if row["state"] in {"completed", "failed", "cancelled"}:
                    return self.get(diagnostic_id)
                now = _now()
                conn.execute("UPDATE diagnostics SET state = 'running', started_at = COALESCE(started_at, ?), updated_at = ? WHERE diagnostic_id = ?", (now, now, diagnostic_id))
                conn.commit()
            event = self._events.setdefault(diagnostic_id, threading.Event())
            with self.store.connect() as conn:
                rows = conn.execute("SELECT * FROM diagnostic_cells WHERE diagnostic_id = ? AND status = 'queued' ORDER BY cell_key", (diagnostic_id,)).fetchall()
            protocol = self.runtime._evaluation_protocol
            base_hash = self.get(diagnostic_id)["baseBundleHash"]
            candidate_hash = self.get(diagnostic_id)["candidateBundleHash"]
            bundles = {"B0": self._bundle(self.runtime, base_hash), "L": self._bundle(self.runtime, candidate_hash)}
            task_map = {name: tuple(self.runtime.packages[name].tasks_for_partition("development"))[0] for name in protocol.known_environments}

            def execute(row: Any) -> None:
                if event.is_set():
                    return
                claimed_at = _now()
                with self.store.connect() as conn:
                    claimed = conn.execute("UPDATE diagnostic_cells SET status = 'running', started_at = ?, updated_at = ? WHERE diagnostic_id = ? AND cell_key = ? AND status = 'queued'", (claimed_at, claimed_at, diagnostic_id, row["cell_key"])).rowcount
                    conn.commit()
                if not claimed or event.is_set():
                    return
                task_name, arm = row["cell_key"].rsplit(":", 1)
                task = task_map[task_name]
                config = SimpleNamespace(protocol=protocol, arm=arm, seed=int(protocol.seeds[0]), attempt=0, bundle_hash=bundles[arm].content_hash, arm_bundles={"B0": base_hash, "L": candidate_hash})
                try:
                    executor = self.executor or self.runtime.execute_evaluation_task
                    observation = executor(task, config, bundles[arm])
                    accounting = self.store.get_artifact(observation.accounting_ref) if hasattr(observation, "accounting_ref") else {}
                    usage = accounting.get("aggregateUsage", accounting.get("usage", {})) if isinstance(accounting, Mapping) else {}
                    value = {"taskId": task.task_id, "arm": arm, "seed": int(protocol.seeds[0]), "passed": bool(getattr(observation, "passed", observation.get("passed", False) if isinstance(observation, Mapping) else False)), "usage": dict(usage) if isinstance(usage, Mapping) else {}, "wallDurationSeconds": float(getattr(observation, "latency_seconds", observation.get("wallDurationSeconds", 0) if isinstance(observation, Mapping) else 0) or 0), "runId": getattr(observation, "run_id", None), "accountingRef": getattr(observation, "accounting_ref", None)}
                    receipt_ref = self.store.put_artifact(value).sha256
                    status, failure_class, error = "completed", "task_failure" if not value["passed"] else None, None
                except Exception as exc:
                    value, receipt_ref, status, failure_class, error = {}, None, "failed", "infrastructure_error", str(exc)
                with self.store.connect() as conn:
                    conn.execute("UPDATE diagnostic_cells SET status = ?, result_json = ?, receipt_ref = ?, failure_class = ?, error = ?, updated_at = ? WHERE diagnostic_id = ? AND cell_key = ?", (status, json.dumps(value, sort_keys=True), receipt_ref, failure_class, error, _now(), diagnostic_id, row["cell_key"]))
                    conn.commit()

            with concurrent.futures.ThreadPoolExecutor(max_workers=3, thread_name_prefix="diagnostic") as pool:
                futures = [pool.submit(execute, row) for row in rows]
                for future in futures:
                    future.result()
            with self.store.connect() as conn:
                state = conn.execute("SELECT cancel_requested FROM diagnostics WHERE diagnostic_id = ?", (diagnostic_id,)).fetchone()
                failed = conn.execute("SELECT COUNT(*) FROM diagnostic_cells WHERE diagnostic_id = ? AND status = 'failed'", (diagnostic_id,)).fetchone()[0]
                pending = conn.execute("SELECT COUNT(*) FROM diagnostic_cells WHERE diagnostic_id = ? AND status = 'queued'", (diagnostic_id,)).fetchone()[0]
                terminal = "cancelled" if state and state["cancel_requested"] else "failed" if failed else "completed" if pending == 0 else "running"
                conn.execute("UPDATE diagnostics SET state = ?, updated_at = ?, error = CASE WHEN ? = 'failed' THEN 'one or more cells failed infrastructure checks' ELSE error END WHERE diagnostic_id = ?", (terminal, _now(), terminal, diagnostic_id))
                conn.commit()
            return self.get(diagnostic_id)
        finally:
            lock.release()


__all__ = ["DevelopmentDiagnosticManager"]
