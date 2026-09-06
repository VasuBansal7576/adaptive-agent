"""Content-addressed artifact storage and SQLite metadata store.

Owns durable state for the Store/Broker/Candidate scope: artifact CAS, environment
and task records, tool-call prepared/result records, single-use approvals, evidence,
candidates, frozen evaluation protocols, promotions, and the active-bundle lineage.
"""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from adaptive_agent.models import ArtifactRef, sha256_json


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


_SENSITIVE_KEY = re.compile(r"(?i)(?:hidden|expected|evaluator|answer[_ -]?key|approval|secret|credential|api[_-]?key|token|password|authorization)")

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)\s*[:=]\s*\S+"),
)


def sanitize_for_learner(value: Any) -> Any:
    if isinstance(value, str):
        for pattern in _SECRET_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
        return value
    if isinstance(value, dict):
        return {
            key: sanitize_for_learner(item)
            for key, item in value.items()
            if not _SENSITIVE_KEY.search(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_learner(item) for item in value]
    return value


class Store:
    """Local content-addressed store backed by SQLite and flat files."""

    def __init__(self, base_dir: str | Path, db_name: str = "adaptive_agent.db") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.base_dir / "artifacts"
        self.artifact_dir.mkdir(exist_ok=True)
        self.db_path = self.base_dir / db_name
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS environments (
                    id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    manifest_ref TEXT NOT NULL,
                    registered_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    environment_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    task_ref TEXT NOT NULL,
                    partition TEXT NOT NULL,
                    goal TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS skill_bundles (
                    bundle_id TEXT PRIMARY KEY,
                    parent TEXT,
                    content_hash TEXT NOT NULL UNIQUE,
                    bundle_json TEXT NOT NULL,
                    is_active INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS active_history (
                    content_hash TEXT NOT NULL,
                    activated_at TEXT NOT NULL,
                    decision_id TEXT NOT NULL,
                    PRIMARY KEY (content_hash, activated_at)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    parent_run_id TEXT,
                    task_id TEXT NOT NULL,
                    environment_id TEXT NOT NULL,
                    bundle_id TEXT NOT NULL,
                    bundle_hash TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_fingerprint TEXT,
                    last_event_sequence INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    run_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS steps (
                    step_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    step_json TEXT NOT NULL,
                    UNIQUE(run_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    trust_class TEXT NOT NULL,
                    visibility TEXT NOT NULL,
                    redacted INTEGER NOT NULL,
                    UNIQUE(run_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    base_bundle_hash TEXT NOT NULL,
                    candidate_bundle_hash TEXT,
                    candidate_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS frozen_protocols (
                    protocol_hash TEXT PRIMARY KEY,
                    gate_json TEXT NOT NULL,
                    evaluator_id TEXT NOT NULL,
                    evaluator_refs_json TEXT NOT NULL DEFAULT '[]',
                    fixture_hashes_json TEXT NOT NULL DEFAULT '{}',
                    partition_hashes_json TEXT NOT NULL DEFAULT '{}',
                    frozen_at TEXT NOT NULL,
                    active INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    report_id TEXT PRIMARY KEY,
                    candidate_hash TEXT NOT NULL,
                    base_hash TEXT,
                    protocol_hash TEXT NOT NULL,
                    partition_ref TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    validity TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluation_queue (
                    evaluation_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    candidate_hash TEXT NOT NULL,
                    base_hash TEXT NOT NULL,
                    protocol_hash TEXT NOT NULL,
                    partition_ref TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS promotions (
                    decision_id TEXT PRIMARY KEY,
                    candidate_hash TEXT NOT NULL,
                    base_hash TEXT NOT NULL,
                    prior_active_hash TEXT NOT NULL,
                    new_active_hash TEXT,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_calls (
                    call_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    environment_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    approval_token TEXT,
                    result_json TEXT,
                    effect TEXT,
                    UNIQUE(run_id, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    token TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    environment_id TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    consumed INTEGER DEFAULT 0,
                    consumed_call_id TEXT,
                    expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_runs (
                    task_run_id TEXT PRIMARY KEY,
                    environment_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    partition TEXT NOT NULL,
                    benchmark_id TEXT,
                    arm TEXT,
                    seed INTEGER,
                    owner_id TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    state_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluator_allocations (
                    scope_id TEXT NOT NULL,
                    allocation_id TEXT PRIMARY KEY,
                    panel_index INTEGER NOT NULL,
                    panel_hash TEXT NOT NULL,
                    task_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(scope_id, panel_index)
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    passed INTEGER NOT NULL,
                    score REAL,
                    metadata_json TEXT NOT NULL,
                    checked_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS learning_records (
                    record_id TEXT PRIMARY KEY,
                    environment_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(task_runs)").fetchall()}
            for name, declaration in (("benchmark_id", "TEXT"), ("arm", "TEXT"), ("seed", "INTEGER"), ("owner_id", "TEXT")):
                if name not in columns:
                    conn.execute(f"ALTER TABLE task_runs ADD COLUMN {name} {declaration}")
            # Migration: bundle_hash column for runs created before the pin.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
            if "bundle_hash" not in cols:
                conn.execute("ALTER TABLE runs ADD COLUMN bundle_hash TEXT NOT NULL DEFAULT ''")
            if "request_fingerprint" not in cols:
                conn.execute("ALTER TABLE runs ADD COLUMN request_fingerprint TEXT")
            frozen_cols = {r["name"] for r in conn.execute("PRAGMA table_info(frozen_protocols)").fetchall()}
            for name, declaration in (
                ("evaluator_refs_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("fixture_hashes_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("partition_hashes_json", "TEXT NOT NULL DEFAULT '{}'"),
            ):
                if name not in frozen_cols:
                    conn.execute(f"ALTER TABLE frozen_protocols ADD COLUMN {name} {declaration}")
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # Public transaction seam used by evaluator-owned resumable drivers.
    # Keep the implementation shared with the Store's internal callers.
    def connect(self) -> Iterator[sqlite3.Connection]:
        return self._connect()

    # ------------------------------------------------------------------ artifacts
    def put_artifact(self, data: Any) -> ArtifactRef:
        """Store a JSON-serializable object by content hash; idempotent."""

        def default(o: Any) -> Any:
            if isinstance(o, datetime):
                return o.isoformat()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        sha = sha256_json(data)
        path = self.artifact_dir / f"{sha}.json"
        if not path.exists():
            tmp = self.artifact_dir / f".{sha}.{uuid.uuid4().hex}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=default)
            tmp.replace(path)
        return ArtifactRef(id=f"art_{sha[:16]}", version="1", sha256=sha)

    def get_artifact(self, ref: ArtifactRef | str) -> Any:
        sha = ref.sha256 if isinstance(ref, ArtifactRef) else ref
        path = self.artifact_dir / f"{sha}.json"
        if not path.exists():
            raise KeyError(f"artifact {sha} not found")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def has_artifact(self, sha: str) -> bool:
        return (self.artifact_dir / f"{sha}.json").exists() or (self.artifact_dir / f"{sha}.bin").exists()

    def put_immutable_bytes(self, data: bytes) -> dict[str, Any]:
        """Persist exact bytes by SHA-256 for patches and other opaque inputs."""
        import hashlib

        value = bytes(data)
        digest = hashlib.sha256(value).hexdigest()
        path = self.artifact_dir / f"{digest}.bin"
        if path.exists() and path.read_bytes() != value:
            raise ValueError("immutable artifact digest collision")
        if not path.exists():
            tmp = self.artifact_dir / f".{digest}.{uuid.uuid4().hex}.tmp"
            tmp.write_bytes(value)
            tmp.replace(path)
        return {"sha256": digest, "size": len(value), "immutable": True}

    def get_immutable_bytes(self, digest: str) -> bytes:
        path = self.artifact_dir / f"{digest}.bin"
        if not path.exists():
            raise KeyError(f"immutable artifact {digest} not found")
        return path.read_bytes()

    # ------------------------------------------------------------------ resumable task runs (benchmark seam)
    def claim_task_run(self, task_run_id: str, environment_id: str, task_id: str, partition: str) -> tuple[bool, dict[str, Any]]:
        """Atomically claim a task run, returning an existing row on resume."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute("SELECT * FROM task_runs WHERE task_run_id = ?", (task_run_id,)).fetchone()
                if existing is not None:
                    if (existing["environment_id"], existing["task_id"], existing["partition"]) != (environment_id, task_id, partition):
                        conn.rollback()
                        raise ValueError("task run id is already bound to a different task")
                    conn.commit()
                    return False, dict(existing)
                conn.execute("INSERT INTO task_runs (task_run_id, environment_id, task_id, partition, status, state_json, updated_at) VALUES (?, ?, ?, ?, 'running', '{}', ?)", (task_run_id, environment_id, task_id, partition, _utcnow()))
                conn.commit()
                return True, dict(conn.execute("SELECT * FROM task_runs WHERE task_run_id = ?", (task_run_id,)).fetchone())
            except Exception:
                conn.rollback()
                raise

    def update_task_run_status(self, task_run_id: str, status: str, state_json: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE task_runs SET status = ?, state_json = COALESCE(?, state_json), updated_at = ? WHERE task_run_id = ?", (status, state_json, _utcnow(), task_run_id))
            conn.commit()

    def get_task_run(self, task_run_id: str) -> dict[str, Any] | None:
        return self._get_json("task_runs", "task_run_id", task_run_id)

    def claim_benchmark_task_run(
        self, task_run_id: str, *, benchmark_id: str, environment_id: str,
        task_id: str, partition: str, arm: str | None = None,
        seed: int | None = None, owner_id: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Claim a benchmark task once and refuse foreign-owner takeover."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM task_runs WHERE task_run_id = ?", (task_run_id,)).fetchone()
            if existing is not None:
                row = dict(existing)
                if (row["environment_id"], row["task_id"], row["partition"], row.get("benchmark_id"), row.get("arm"), row.get("seed")) != (environment_id, task_id, partition, benchmark_id, arm, seed):
                    conn.rollback()
                    raise ValueError("task run id is already bound to different benchmark inputs")
                conn.commit()
                return False, row
            conn.execute(
                "INSERT INTO task_runs(task_run_id, environment_id, task_id, partition, benchmark_id, arm, seed, owner_id, status, state_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', '{}', ?)",
                (task_run_id, environment_id, task_id, partition, benchmark_id, arm, seed, owner_id, _utcnow()),
            )
            conn.commit()
            return True, dict(conn.execute("SELECT * FROM task_runs WHERE task_run_id = ?", (task_run_id,)).fetchone())

    def release_task_run(self, task_run_id: str, owner_id: str, status: str, state_json: str | None = None) -> bool:
        """Update a task run only when its current owner matches."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT owner_id FROM task_runs WHERE task_run_id = ?", (task_run_id,)).fetchone()
            if row is None or row["owner_id"] != owner_id:
                conn.rollback()
                return False
            conn.execute("UPDATE task_runs SET status = ?, state_json = COALESCE(?, state_json), updated_at = ? WHERE task_run_id = ?", (status, state_json, _utcnow(), task_run_id))
            conn.commit()
            return True

    def list_task_runs(self, environment_id: str | None = None, partition: str | None = None, status: str | None = None, benchmark_id: str | None = None, arm: str | None = None, owner_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM task_runs WHERE 1=1"
        params: list[Any] = []
        for column, value in (("environment_id", environment_id), ("partition", partition), ("status", status), ("benchmark_id", benchmark_id), ("arm", arm), ("owner_id", owner_id)):
            if value is not None:
                query += f" AND {column} = ?"
                params.append(value)
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def reserve_allocation(self, scope_id: str, allocation_id: str, panels: list[list[str]], limit: int) -> int | None:
        """Atomically reserve the next free evaluator panel."""
        if not panels or len(panels) < limit:
            raise ValueError("allocation panels must cover the configured limit")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute("SELECT 1 FROM evaluator_allocations WHERE allocation_id = ?", (allocation_id,)).fetchone() is not None:
                    conn.commit()
                    return None
                used = {int(row["panel_index"]) for row in conn.execute("SELECT panel_index FROM evaluator_allocations WHERE scope_id = ?", (scope_id,)).fetchall()}
                index = next((candidate for candidate in range(limit) if candidate not in used), None)
                if index is None:
                    conn.commit()
                    return None
                task_ids = list(panels[index])
                conn.execute("INSERT INTO evaluator_allocations (scope_id, allocation_id, panel_index, panel_hash, task_ids_json, created_at) VALUES (?, ?, ?, ?, ?, ?)", (scope_id, allocation_id, index, sha256_json(task_ids), json.dumps(task_ids, sort_keys=True), _utcnow()))
                conn.commit()
                return index
            except Exception:
                conn.rollback()
                raise

    def get_allocation(self, allocation_id: str) -> dict[str, Any] | None:
        """Read a reserved evaluator panel with decoded task IDs."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluator_allocations WHERE allocation_id = ?",
                (allocation_id,),
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        try:
            task_ids = json.loads(value.get("task_ids_json", "[]"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("stored evaluator allocation has invalid task IDs") from exc
        if not isinstance(task_ids, list) or not all(isinstance(task_id, str) for task_id in task_ids):
            raise ValueError("stored evaluator allocation task IDs must be strings")
        value["task_ids"] = task_ids
        return value

    def dev_smoke_ok(self, environment_id: str) -> bool:
        """Whether a passed trusted development run exists for an environment."""
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM outcomes o JOIN runs r ON r.run_id = o.run_id JOIN tasks t ON t.id = r.task_id WHERE r.environment_id = ? AND t.partition = 'development' AND o.passed = 1 AND EXISTS (SELECT 1 FROM evidence e WHERE e.run_id = o.run_id AND e.event_type = 'trusted_outcome' AND e.trust_class = 'evaluator') LIMIT 1", (environment_id,)).fetchone()
            return row is not None

    # ------------------------------------------------------------------ generic helpers
    def _insert_json(self, table: str, id_col: str, obj_id: str, data: dict[str, Any]) -> None:
        with self._connect() as conn:
            columns = [id_col] + list(data.keys())
            values = [obj_id] + list(data.values())
            placeholders = ",".join("?" for _ in columns)
            conn.execute(
                f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                values,
            )
            conn.commit()

    def _get_json(self, table: str, id_col: str, obj_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(f"SELECT * FROM {table} WHERE {id_col} = ?", (obj_id,)).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------ environment / tasks
    def register_environment(self, env_id: str, version: str, manifest_ref: ArtifactRef) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO environments (id, version, manifest_ref, registered_at) VALUES (?, ?, ?, ?)",
                (env_id, version, manifest_ref.model_dump_json(by_alias=True), _utcnow()),
            )
            conn.commit()

    def get_environment(self, env_id: str) -> dict[str, Any] | None:
        return self._get_json("environments", "id", env_id)

    def list_environments(self) -> list[dict[str, Any]]:
        """List registered environment rows for controller diagnostics."""
        with self._connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM environments ORDER BY id").fetchall()]

    def register_task(self, task_id: str, environment_id: str, version: str, task_ref: str, partition: str, goal: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tasks (id, environment_id, version, task_ref, partition, goal) VALUES (?, ?, ?, ?, ?, ?)",
                (task_id, environment_id, version, task_ref, partition, goal),
            )
            conn.commit()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self._get_json("tasks", "id", task_id)

    def list_tasks_by_partition(self, environment_id: str, partition: str) -> list[dict[str, Any]]:
        """Strictly filter by BOTH environment and partition."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE environment_id = ? AND partition = ?",
                (environment_id, partition),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ runs
    def save_run(self, run_id: str, data: dict[str, Any]) -> None:
        self._insert_json("runs", "run_id", run_id, data)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._get_json("runs", "run_id", run_id)

    def get_run_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE idempotency_key = ?", (key,)).fetchone()
            return dict(row) if row else None

    def claim_run(self, run_id: str) -> bool | None:
        """Atomically claim a queued run for one executor.

        ``None`` means the run does not exist, ``True`` means this caller
        changed queued to running, and ``False`` means another caller already
        claimed it or it is terminal.  The serialized RunRecord is updated in
        the same transaction as the status column so a restart cannot observe
        a partially claimed run.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT status, run_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                conn.rollback()
                return None
            if row["status"] != "queued":
                conn.rollback()
                return False
            payload = json.loads(row["run_json"])
            payload["status"] = "running"
            changed = conn.execute("UPDATE runs SET status = ?, run_json = ? WHERE run_id = ? AND status = 'queued'", ("running", json.dumps(payload, sort_keys=True), run_id)).rowcount == 1
            if changed:
                conn.commit()
            else:
                conn.rollback()
            return changed

    def update_run_status(
        self,
        run_id: str,
        status: str,
        completed_at: str | None = None,
        last_event_sequence: int | None = None,
    ) -> None:
        with self._connect() as conn:
            parts = ["status = ?"]
            values: list[Any] = [status]
            if completed_at is not None:
                parts.append("completed_at = ?")
                values.append(completed_at)
            if last_event_sequence is not None:
                parts.append("last_event_sequence = ?")
                values.append(last_event_sequence)
            values.append(run_id)
            conn.execute(f"UPDATE runs SET {', '.join(parts)} WHERE run_id = ?", values)
            conn.commit()

    # ------------------------------------------------------------------ skill bundles / active pointer
    def save_bundle(
        self,
        bundle_id: str,
        parent: str | None,
        content_hash: str,
        bundle_json: str,
        is_active: bool = False,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO skill_bundles (bundle_id, parent, content_hash, bundle_json, is_active, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (bundle_id, parent, content_hash, bundle_json, 1 if is_active else 0, _utcnow()),
            )
            conn.commit()

    def get_bundle(self, bundle_id: str) -> dict[str, Any] | None:
        return self._get_json("skill_bundles", "bundle_id", bundle_id)

    def get_bundle_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM skill_bundles WHERE content_hash = ?", (content_hash,)).fetchone()
            return dict(row) if row else None

    def get_active_bundle(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM skill_bundles WHERE is_active = 1 LIMIT 1").fetchone()
            return dict(row) if row else None

    def set_active_bundle(self, content_hash: str) -> None:
        """Unconditional active-pointer update. Use only for initial seeding;
        promotions must go through cas_active_bundle for atomicity."""
        with self._connect() as conn:
            conn.execute("UPDATE skill_bundles SET is_active = 0")
            conn.execute("UPDATE skill_bundles SET is_active = 1 WHERE content_hash = ?", (content_hash,))
            conn.commit()

    def cas_active_bundle(
        self,
        expected_active_hash: str,
        new_active_hash: str,
        candidate_id: str,
        new_candidate_state: str,
        candidate_json: str,
        promotion: dict[str, Any],
    ) -> bool:
        """Atomically: verify the active pointer, swap it, update candidate state,
        and insert the promotion decision — all-or-nothing.

        Returns True when the compare-and-swap succeeded; False when the active
        pointer no longer matches expected_active_hash.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT content_hash FROM skill_bundles WHERE is_active = 1"
                ).fetchone()
                current = row["content_hash"] if row else None
                if current != expected_active_hash:
                    conn.rollback()
                    return False
                if conn.execute(
                    "SELECT 1 FROM skill_bundles WHERE content_hash = ?", (new_active_hash,)
                ).fetchone() is None:
                    conn.rollback()
                    return False
                conn.execute("UPDATE skill_bundles SET is_active = 0")
                conn.execute(
                    "UPDATE skill_bundles SET is_active = 1 WHERE content_hash = ?",
                    (new_active_hash,),
                )
                conn.execute(
                    "UPDATE candidates SET state = ?, candidate_json = ? WHERE candidate_id = ?",
                    (new_candidate_state, candidate_json, candidate_id),
                )
                conn.execute(
                    """INSERT INTO promotions (decision_id, candidate_hash, base_hash,
                        prior_active_hash, new_active_hash, decision, reason, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        promotion["decision_id"],
                        promotion["candidate_hash"],
                        promotion["base_hash"],
                        promotion["prior_active_hash"],
                        promotion["new_active_hash"],
                        promotion["decision"],
                        promotion["reason"],
                        promotion["timestamp"],
                    ),
                )
                conn.execute(
                    "INSERT INTO active_history (content_hash, activated_at, decision_id) VALUES (?, ?, ?)",
                    (new_active_hash, _utcnow(), promotion["decision_id"]),
                )
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise

    def append_active_history(self, content_hash: str, decision_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO active_history (content_hash, activated_at, decision_id) VALUES (?, ?, ?)",
                (content_hash, _utcnow(), decision_id),
            )
            conn.commit()

    def is_in_active_lineage(self, content_hash: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM active_history WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            return row is not None

    def list_active_lineage(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content_hash FROM active_history ORDER BY activated_at"
            ).fetchall()
            return [r["content_hash"] for r in rows]

    # ------------------------------------------------------------------ steps
    def save_step(self, step_id: str, data: dict[str, Any]) -> None:
        self._insert_json("steps", "step_id", step_id, data)

    def get_step(self, step_id: str) -> dict[str, Any] | None:
        return self._get_json("steps", "step_id", step_id)

    def list_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM steps WHERE run_id = ? ORDER BY sequence", (run_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def update_step_status(self, step_id: str, status: str, step_json: str | None = None) -> None:
        with self._connect() as conn:
            if step_json is not None:
                conn.execute(
                    "UPDATE steps SET status = ?, step_json = ? WHERE step_id = ?",
                    (status, step_json, step_id),
                )
            else:
                conn.execute("UPDATE steps SET status = ? WHERE step_id = ?", (status, step_id))
            conn.commit()

    def next_event_sequence(self, run_id: str) -> int:
        """Atomically increment and return the run's event sequence for SSE ordering."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET last_event_sequence = last_event_sequence + 1 WHERE run_id = ?",
                (run_id,),
            )
            row = conn.execute(
                "SELECT last_event_sequence FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            conn.commit()
            if row is None:
                raise KeyError(f"run {run_id} not found")
            return int(row["last_event_sequence"])

    # ------------------------------------------------------------------ evidence
    def append_evidence(self, evidence_id: str, data: dict[str, Any]) -> None:
        self._insert_json("evidence", "evidence_id", evidence_id, data)

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        return self._get_json("evidence", "evidence_id", evidence_id)

    def list_evidence(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence WHERE run_id = ? ORDER BY sequence", (run_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def save_learning_record(self, record_id: str, environment_id: str, run_id: str, record_json: str) -> None:
        """Persist one immutable learner projection for restart-safe learning."""
        if not all(isinstance(value, str) and value for value in (record_id, environment_id, run_id, record_json)):
            raise ValueError("learning record fields must be non-empty strings")
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO learning_records (record_id, environment_id, run_id, record_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (record_id, environment_id, run_id, record_json, _utcnow()),
            )
            conn.commit()

    def list_learning_records(self, environment_id: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        """Read the narrow, persisted projection exposed to learning."""
        with self._connect() as conn:
            query = "SELECT * FROM learning_records WHERE 1=1"
            params: list[Any] = []
            if environment_id is not None:
                query += " AND environment_id = ?"
                params.append(environment_id)
            if run_id is not None:
                query += " AND run_id = ?"
                params.append(run_id)
            query += " ORDER BY created_at, record_id"
            rows = conn.execute(query, params).fetchall()
        if rows:
            return [dict(row) for row in rows]

        # Older stores predate the materialized projection. Reconstruct it
        # only when no persisted rows exist, preserving the existing safety
        # checks and allowing those stores to migrate on the next launch.
        if environment_id is None or run_id is None:
            return []
        records: list[dict[str, Any]] = []
        environment = self.get_environment(environment_id)
        if environment:
            try:
                manifest_ref = json.loads(environment["manifest_ref"])
                manifest = self.get_artifact(manifest_ref["sha256"])
                for ref in manifest.get("docs", []):
                    if not isinstance(ref, dict):
                        continue
                    content = self.get_artifact(ref["sha256"])
                    text = content if isinstance(content, str) else json.dumps(content, sort_keys=True, separators=(",", ":"))
                    records.append({"kind": "public_doc", "sourceId": ref.get("id", ref["sha256"]), "content": text, "contentHash": hashlib.sha256(text.encode("utf-8")).hexdigest(), "environmentId": environment_id, "visibility": "public"})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass
        trusted_row = self.get_outcome_by_run_id(run_id)
        trusted = isinstance(trusted_row, Mapping)
        outcome_passed = bool(trusted_row.get("passed")) if trusted_row is not None else False
        for row in self.list_evidence(run_id):
            provenance = self.evidence_provenance(row["evidence_id"])
            is_development_model = row.get("event_type") == "model_response" and provenance and provenance.get("partition") == "development"
            if row.get("visibility") != "learner" and not is_development_model:
                continue
            try:
                source = json.loads(row["source_ref"])
                content = self.get_artifact(source["sha256"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            text = "Verified development model execution evidence is available for this run." if is_development_model else (content if isinstance(content, str) else json.dumps(content, sort_keys=True, separators=(",", ":")))
            if not provenance or provenance.get("environment_id") != environment_id:
                continue
            content_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            records.append({"kind": "live_evidence", "sourceId": row["evidence_id"], "content": text, "contentHash": content_digest, "environmentId": environment_id, "runId": run_id, "partition": provenance.get("partition"), "visibility": "learner", "trustClass": row.get("trust_class"), "trustedOutcome": trusted})
        if trusted:
            text = "A trusted evaluator outcome is stored for this development run."
            records.append({"kind": "task_state", "sourceId": f"outcome:{run_id}", "content": text, "contentHash": hashlib.sha256(text.encode("utf-8")).hexdigest(), "environmentId": environment_id, "runId": run_id, "visibility": "learner", "trustedOutcome": trusted, "outcomePassed": outcome_passed})
        return records

    def evidence_provenance(self, evidence_id: str) -> dict[str, Any] | None:
        """Join evidence -> run -> task to expose partition and terminal status."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT e.evidence_id, e.run_id, e.trust_class, e.visibility,
                          r.status AS run_status, r.task_id, t.partition, t.environment_id
                   FROM evidence e
                   JOIN runs r ON r.run_id = e.run_id
                   JOIN tasks t ON t.id = r.task_id
                   WHERE e.evidence_id = ?""",
                (evidence_id,),
            ).fetchone()
            return dict(row) if row else None

    def has_trusted_outcome(self, run_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM outcomes WHERE run_id = ?", (run_id,)).fetchone()
            return row is not None

    # ------------------------------------------------------------------ narrow learning projections
    def get_public_docs(self, environment_id: str) -> list[dict[str, Any]]:
        """Return manifest documents that are safe for learner context."""
        row = self.get_environment(environment_id)
        if not row:
            return []
        manifest_ref = ArtifactRef.model_validate_json(row["manifest_ref"])
        manifest = self.get_artifact(manifest_ref)
        output: list[dict[str, Any]] = []
        for raw in manifest.get("docs", []):
            if not isinstance(raw, dict):
                continue
            sha = raw.get("sha256")
            if not isinstance(sha, str) or not self.has_artifact(sha):
                continue
            payload = self.get_artifact(sha)
            if isinstance(payload, dict) and payload.get("classification") in {"operator", "evaluator_only"}:
                continue
            output.append({"id": raw.get("id"), "version": raw.get("version"), "sha256": sha, "content": payload})
        return output

    def list_learner_evidence(self, environment_id: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        """Return only redacted broker tool observations from DEVELOPMENT.

        The join binds each row to its run, task partition, environment, and
        trusted outcome presence. Raw evaluator/operator evidence is excluded
        at the SQL boundary rather than filtered by learner code.
        """
        query = ("SELECT e.evidence_id, e.run_id, e.sequence, e.event_type, e.content_hash, "
                 "e.trust_class, e.visibility, e.redacted, r.task_id, r.environment_id, r.status AS run_status, t.partition, "
                 "1 AS trusted_outcome, o.passed AS outcome_passed "
                 "FROM evidence e JOIN runs r ON r.run_id=e.run_id JOIN tasks t ON t.id=r.task_id "
                 "LEFT JOIN outcomes o ON o.run_id=e.run_id "
                 "WHERE e.visibility='learner' AND e.redacted=1 AND e.trust_class='broker' "
                 "AND e.event_type IN ('tool_result','learning_evidence_projection') AND t.partition='development' "
                 "AND r.status IN ('succeeded','failed','cancelled','timed_out','outcome_unknown') "
                 "AND o.run_id IS NOT NULL")
        params: list[Any] = []
        if environment_id is not None:
            query += " AND r.environment_id=?"
            params.append(environment_id)
        if run_id is not None:
            query += " AND e.run_id=?"
            params.append(run_id)
        query += " ORDER BY e.run_id, e.sequence"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def list_run_tool_calls(self, run_id: str) -> list[dict[str, Any]]:
        """Return a sanitized broker-call projection for one development run.

        The projection joins prepared tool calls with broker result evidence.
        It deliberately omits approval tokens and keeps only fields that the
        learner may use for procedural improvement.
        """
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"run {run_id!r} not found")
        task = self.get_task(run.get("task_id")) if isinstance(run.get("task_id"), str) else None
        if not task or task.get("partition") != "development":
            raise PermissionError("tool-call projection is development-partition only")

        # Restrict the projection to tools and input fields declared by the
        # environment manifest. Missing or malformed manifests fail closed.
        schema_keys: dict[str, set[str]] | None = {}
        environment = self.get_environment(run["environment_id"])
        if environment:
            try:
                manifest_ref = json.loads(environment["manifest_ref"])
                manifest = self.get_artifact(manifest_ref["sha256"])
                for schema in manifest.get("toolSchemas", []):
                    if isinstance(schema, dict) and isinstance(schema.get("name"), str):
                        properties = (schema.get("inputSchema") or {}).get("properties") or {}
                        schema_keys[schema["name"]] = set(properties)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, AttributeError):
                schema_keys = None
        if not schema_keys:
            return []

        def valid_source(event: dict[str, Any]) -> dict[str, Any] | None:
            if event["event_type"] != "tool_result" or event["trust_class"] != "broker":
                return None
            if event["visibility"] not in {"learner", "operator"}:
                return None
            try:
                source_ref = json.loads(event["source_ref"])
                payload = self.get_artifact(source_ref["sha256"])
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                return None
            if event.get("content_hash") != source_ref["sha256"]:
                return None
            return payload if isinstance(payload, dict) else None

        with self._connect() as conn:
            calls = [dict(row) for row in conn.execute("SELECT * FROM tool_calls WHERE run_id = ? ORDER BY rowid", (run_id,)).fetchall()]
        evidence_by_call: dict[str, dict[str, Any]] = {}
        for event in self.list_evidence(run_id):
            payload = valid_source(event)
            if payload is not None and isinstance(payload.get("callId"), str):
                evidence_by_call[payload["callId"]] = event

        out: list[dict[str, Any]] = []
        matched: set[str] = set()
        for call in calls:
            try:
                arguments = json.loads(call["arguments_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                arguments = {}
            result: Any = None
            status = error_code = retry = version = effect = result_hash = None
            if isinstance(call.get("result_json"), str):
                try:
                    result = json.loads(call["result_json"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    result = None
            if isinstance(result, dict):
                status = result.get("status")
                output = result.get("output")
                error = result.get("error")
                version = result.get("toolVersion")
                effect = result.get("effect") or call.get("effect")
                if isinstance(error, dict):
                    error_code, retry = error.get("code"), error.get("retry")
                result_hash = sha256_json(result)
            else:
                output = None
                effect = call.get("effect")
            allowed = schema_keys.get(call["tool"])
            if allowed is None:
                continue
            event = evidence_by_call.get(call["call_id"], {})
            if event:
                matched.add(call["call_id"])
            out.append({
                "callId": call["call_id"], "evidenceId": event.get("evidence_id"), "tool": call["tool"],
                "input": sanitize_for_learner({key: arguments[key] for key in arguments if key in allowed}), "result": sanitize_for_learner(output), "status": status,
                "errorCode": error_code, "retry": retry, "version": version, "effect": effect,
                "idempotencyKey": call["idempotency_key"], "argumentsSha256": sha256_json(arguments),
                "resultSha256": result_hash, "evidenceContentHash": event.get("content_hash"),
                "runId": run_id, "taskId": run["task_id"], "environmentId": run["environment_id"],
                "partition": "development", "visibility": "learner", "redacted": True,
            })
        # Older runtime versions persisted tool-result evidence without a
        # prepared-call row. Preserve that history as a minimal safe record.
        for event in self.list_evidence(run_id):
            # An unjoined operator row is raw broker fidelity, not a learner
            # projection.  Only a learner-visible event may use this legacy
            # fallback; prepared calls above can still derive a safe row from
            # their persisted result while retaining the operator evidence
            # solely for provenance.
            if event.get("event_type") != "tool_result" or event.get("trust_class") != "broker" or event.get("visibility") != "operator":
                continue
            payload = valid_source(event)
            if payload is None:
                # Preserve a minimal bound row only when the source artifact
                # cannot be resolved. A resolved hash mismatch is tampering.
                try:
                    source_ref = json.loads(event["source_ref"])
                    self.get_artifact(source_ref["sha256"])
                except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                    payload = {}
                else:
                    continue
            call_id = payload.get("callId")
            if isinstance(call_id, str) and call_id in matched:
                continue
            tool = payload.get("tool")
            allowed = schema_keys.get(tool) if isinstance(tool, str) else None
            if payload and not isinstance(tool, str):
                continue
            if tool is not None and allowed is None:
                continue
            raw_input = payload.get("input")
            error = payload.get("error")
            out.append({
                "callId": call_id, "evidenceId": event.get("evidence_id"), "tool": tool,
                "input": sanitize_for_learner({key: raw_input[key] for key in raw_input if allowed is not None and key in allowed}) if isinstance(raw_input, dict) else None, "result": sanitize_for_learner(payload.get("output")), "status": payload.get("status"),
                "errorCode": error.get("code") if isinstance(error, dict) else None,
                "retry": error.get("retry") if isinstance(error, dict) else None,
                "version": payload.get("toolVersion"), "effect": payload.get("effect"),
                "idempotencyKey": None, "argumentsSha256": None, "resultSha256": sha256_json(payload),
                "evidenceContentHash": event.get("content_hash"), "runId": run_id,
                "taskId": run["task_id"], "environmentId": run["environment_id"],
                "partition": "development", "visibility": "learner", "redacted": True,
            })
        return out

    def list_learning_evidence(self, environment_id: str | None = None, run_id: str | None = None, include_broker_projection: bool = True) -> list[dict[str, Any]]:
        """Return the bounded learner evidence feed joined to trusted runs."""
        evidence = [dict(row, kind="evidence") for row in self.list_learner_evidence(environment_id, run_id)]
        if not include_broker_projection:
            return evidence
        query = "SELECT r.run_id FROM runs r JOIN tasks t ON t.id = r.task_id WHERE t.partition = 'development'"
        params: list[Any] = []
        if environment_id is not None:
            query += " AND r.environment_id = ?"
            params.append(environment_id)
        if run_id is not None:
            query += " AND r.run_id = ?"
            params.append(run_id)
        with self._connect() as conn:
            run_ids = [str(row["run_id"]) for row in conn.execute(query, params).fetchall()]
        for rid in run_ids:
            evidence.extend(dict(row, kind="broker_call") for row in self.list_run_tool_calls(rid))
        return evidence

    # ------------------------------------------------------------------ tool calls / approvals
    def prepare_tool_call(self, data: dict[str, Any]) -> bool:
        """Insert a prepared (no-result) tool call. Returns False on (run_id,
        idempotency_key) or call_id uniqueness conflict."""
        with self._connect() as conn:
            try:
                conn.execute(
                    """INSERT INTO tool_calls (call_id, run_id, step_id, environment_id,
                        tool, arguments_json, idempotency_key, approval_token, result_json, effect)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        data["call_id"],
                        data["run_id"],
                        data["step_id"],
                        data["environment_id"],
                        data["tool"],
                        data["arguments_json"],
                        data["idempotency_key"],
                        data.get("approval_token"),
                        None,
                        None,
                    ),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def consume_approval_and_prepare(
        self,
        token: str,
        call_data: dict[str, Any],
        expected_run_id: str,
        expected_tool: str,
        expected_idempotency_key: str,
        expected_arguments_json: str,
    ) -> tuple[str, dict[str, Any] | None]:
        """Atomically consume a one-use approval AND insert the prepared tool call.

        Returns ("ok", approval_row) on success, or an error tag and None:
        "invalid" (token absent/expired/consumed/mismatched), "conflict"
        (idempotency key already prepared by another call), "duplicate" (same
        prepared call already exists).
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM approvals WHERE token = ?", (token,)
                ).fetchone()
                if not row:
                    conn.rollback()
                    return "invalid", None
                approval = dict(row)
                if approval["consumed"]:
                    conn.rollback()
                    return "invalid", None
                if _utcnow() > approval["expires_at"]:
                    conn.rollback()
                    return "invalid", None
                if (
                    approval["run_id"] != expected_run_id
                    or approval["tool"] != expected_tool
                    or approval["idempotency_key"] != expected_idempotency_key
                    or approval["arguments_json"] != expected_arguments_json
                ):
                    conn.rollback()
                    return "invalid", None

                existing = conn.execute(
                    "SELECT * FROM tool_calls WHERE run_id = ? AND idempotency_key = ?",
                    (expected_run_id, expected_idempotency_key),
                ).fetchone()
                if existing:
                    conn.rollback()
                    if existing["arguments_json"] == expected_arguments_json:
                        return "duplicate", dict(existing)
                    return "conflict", dict(existing)

                conn.execute(
                    "UPDATE approvals SET consumed = 1, consumed_call_id = ? WHERE token = ?",
                    (call_data["call_id"], token),
                )
                conn.execute(
                    """INSERT INTO tool_calls (call_id, run_id, step_id, environment_id,
                        tool, arguments_json, idempotency_key, approval_token, result_json, effect)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
                    (
                        call_data["call_id"],
                        call_data["run_id"],
                        call_data["step_id"],
                        call_data["environment_id"],
                        call_data["tool"],
                        call_data["arguments_json"],
                        call_data["idempotency_key"],
                        token,
                    ),
                )
                conn.commit()
                return "ok", approval
            except Exception:
                conn.rollback()
                raise

    def get_tool_call(self, call_id: str) -> dict[str, Any] | None:
        return self._get_json("tool_calls", "call_id", call_id)

    def get_tool_call_by_idempotency(self, run_id: str, idempotency_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? AND idempotency_key = ?",
                (run_id, idempotency_key),
            ).fetchone()
            return dict(row) if row else None

    def save_tool_result(self, call_id: str, result_json: str, effect: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE tool_calls SET result_json = ?, effect = ? WHERE call_id = ?",
                (result_json, effect, call_id),
            )
            conn.commit()

    def list_unreconciled_calls(self, run_id: str) -> list[dict[str, Any]]:
        """Prepared calls with no recorded result — candidates for reconciliation."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? AND result_json IS NULL",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_call_outcome_unknown(self, call_id: str, result_json: str) -> bool:
        """Idempotently mark a prepared call as OUTCOME_UNKNOWN."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE tool_calls SET result_json = ?, effect = 'unknown' WHERE call_id = ? AND result_json IS NULL",
                (result_json, call_id),
            )
            conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------ candidates
    def save_candidate(self, candidate_id: str, data: dict[str, Any]) -> None:
        self._insert_json("candidates", "candidate_id", candidate_id, data)

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return self._get_json("candidates", "candidate_id", candidate_id)

    def update_candidate(self, candidate_id: str, state: str, candidate_json: str) -> None:
        """Keep the state column and candidate_json payload synchronized."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE candidates SET state = ?, candidate_json = ? WHERE candidate_id = ?",
                (state, candidate_json, candidate_id),
            )
            conn.commit()

    # ------------------------------------------------------------------ frozen protocols / evaluations
    def save_frozen_protocol(
        self,
        protocol_hash: str,
        gate_json: str,
        evaluator_id: str,
        *,
        evaluator_refs: list[str] | None = None,
        fixture_hashes: dict[str, str] | None = None,
        partition_hashes: dict[str, str] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO frozen_protocols (protocol_hash, gate_json, evaluator_id, evaluator_refs_json, fixture_hashes_json, partition_hashes_json, frozen_at, active) VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                (protocol_hash, gate_json, evaluator_id, json.dumps(evaluator_refs or []), json.dumps(fixture_hashes or {}), json.dumps(partition_hashes or {}), _utcnow()),
            )
            conn.commit()

    def get_frozen_protocol(self, protocol_hash: str) -> dict[str, Any] | None:
        row = self._get_json("frozen_protocols", "protocol_hash", protocol_hash)
        if row and not row["active"]:
            return None
        return row

    def deactivate_protocol(self, protocol_hash: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE frozen_protocols SET active = 0 WHERE protocol_hash = ?",
                (protocol_hash,),
            )
            conn.commit()

    def save_evaluation(self, report_id: str, data: dict[str, Any]) -> None:
        self._insert_json("evaluations", "report_id", report_id, data)

    def save_evaluation_queue(self, evaluation_id: str, data: dict[str, Any]) -> None:
        """Persist queue metadata separately from trusted evaluation reports."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO evaluation_queue(
                    evaluation_id, candidate_id, candidate_hash, base_hash,
                    protocol_hash, partition_ref, state, payload_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evaluation_id) DO UPDATE SET
                    candidate_id=excluded.candidate_id,
                    candidate_hash=excluded.candidate_hash,
                    base_hash=excluded.base_hash,
                    protocol_hash=excluded.protocol_hash,
                    partition_ref=excluded.partition_ref,
                    state=excluded.state,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at""",
                (
                    evaluation_id,
                    data["candidate_id"],
                    data["candidate_hash"],
                    data["base_hash"],
                    data["protocol_hash"],
                    data["partition_ref"],
                    data["state"],
                    data["payload_json"],
                    data.get("created_at", _utcnow()),
                    data.get("updated_at", _utcnow()),
                ),
            )
            conn.commit()

    def get_evaluation_queue(self, evaluation_id: str) -> dict[str, Any] | None:
        return self._get_json("evaluation_queue", "evaluation_id", evaluation_id)

    def list_evaluation_queue(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM evaluation_queue ORDER BY created_at").fetchall()
            return [dict(row) for row in rows]

    def get_evaluation(self, report_id: str) -> dict[str, Any] | None:
        return self._get_json("evaluations", "report_id", report_id)

    def get_evaluation_by_protocol_and_candidate(self, protocol_hash: str, candidate_hash: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE protocol_hash = ? AND candidate_hash = ?",
                (protocol_hash, candidate_hash),
            ).fetchone()
            return dict(row) if row else None

    def list_evaluations(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM evaluations ORDER BY report_id").fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------------ promotions
    def save_promotion(self, decision_id: str, data: dict[str, Any]) -> None:
        self._insert_json("promotions", "decision_id", decision_id, data)

    def get_promotion(self, decision_id: str) -> dict[str, Any] | None:
        return self._get_json("promotions", "decision_id", decision_id)

    def list_promotions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM promotions ORDER BY timestamp").fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ outcomes
    def save_outcome(self, outcome_id: str, data: dict[str, Any]) -> None:
        self._insert_json("outcomes", "outcome_id", outcome_id, data)

    def get_outcome(self, outcome_id: str) -> dict[str, Any] | None:
        return self._get_json("outcomes", "outcome_id", outcome_id)

    def get_outcome_by_run_id(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM outcomes WHERE run_id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------ approvals (issue only)
    def issue_approval(self, token: str, run_id: str, environment_id: str, tool: str, idempotency_key: str, arguments_json: str, expires_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO approvals (token, run_id, environment_id, tool, idempotency_key, arguments_json, consumed, expires_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                (token, run_id, environment_id, tool, idempotency_key, arguments_json, expires_at),
            )
            conn.commit()
