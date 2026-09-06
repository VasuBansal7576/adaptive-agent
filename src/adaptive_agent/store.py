"""Content-addressed artifact storage and SQLite metadata store.

Owns durable state for the Store/Broker/Candidate scope: artifact CAS, environment
and task records, tool-call prepared/result records, single-use approvals, evidence,
candidates, frozen evaluation protocols, promotions, and the active-bundle lineage.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from adaptive_agent.models import ArtifactRef, sha256_json


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# Credential-shaped values are masked before learner-visible projection.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|bearer)\s*[:=]\s*\S+"),
)


def sanitize_for_learner(value: Any) -> Any:
    if isinstance(value, str):
        out = value
        for pat in _SECRET_PATTERNS:
            out = pat.sub("[REDACTED]", out)
        return out
    if isinstance(value, dict):
        return {k: sanitize_for_learner(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_learner(v) for v in value]
    return value


class RunIdempotencyConflict(ValueError):
    """An idempotency key was reused with a different request fingerprint."""


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
                    request_fingerprint TEXT NOT NULL DEFAULT '',
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
                CREATE TABLE IF NOT EXISTS learning_records (
                    record_id TEXT PRIMARY KEY,
                    environment_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    passed INTEGER NOT NULL,
                    score REAL,
                    metadata_json TEXT NOT NULL,
                    checked_at TEXT NOT NULL
                );
                """
            )
            # Migration: bundle_hash column for runs created before the pin.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(runs)").fetchall()}
            if "bundle_hash" not in cols:
                conn.execute("ALTER TABLE runs ADD COLUMN bundle_hash TEXT NOT NULL DEFAULT ''")
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Public transaction seam for trusted adapters (evaluator/benchmark).

        Same semantics as _connect; exposed so session-owned adapters do not
        depend on a private method. Callers must commit explicitly.
        """
        with self._connect() as conn:
            yield conn

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
            tmp = self.artifact_dir / f"{sha}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=default)
            tmp.replace(path)
        return ArtifactRef(id=f"art_{sha[:16]}", version="1", sha256=sha)

    def get_artifact(self, ref: ArtifactRef | str) -> Any:
        sha = ref.sha256 if isinstance(ref, ArtifactRef) else ref
        path = self.artifact_dir / f"{sha}.json"
        if not path.exists():
            raise KeyError(f"artifact {sha} not found")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f"artifact {sha} failed content-hash verification")
        payload = json.loads(raw.decode("utf-8"))
        # Canonical integrity: the decoded payload's declared content hash must
        # match the digest name — a file written non-canonically under a raw
        # sha is tampered, not just mis-encoded.
        if sha256_json(payload) != sha:
            raise ValueError(f"artifact {sha} payload is not canonical for its declared hash")
        return payload

    def has_artifact(self, sha: str) -> bool:
        return (self.artifact_dir / f"{sha}.json").exists()

    def put_immutable_bytes(self, data: bytes) -> ArtifactRef:
        """Content-addressed immutable blob store (arbitrary bytes)."""
        sha = hashlib.sha256(data).hexdigest()
        path = self.artifact_dir / f"{sha}.bin"
        if not path.exists():
            tmp = self.artifact_dir / f"{sha}.tmp"
            tmp.write_bytes(data)
            tmp.replace(path)
        return ArtifactRef(id=f"blob_{sha[:16]}", version="1", sha256=sha)

    def get_immutable_bytes(self, ref: ArtifactRef | str) -> bytes:
        sha = ref.sha256 if isinstance(ref, ArtifactRef) else ref
        path = self.artifact_dir / f"{sha}.bin"
        if not path.exists():
            raise KeyError(f"blob {sha} not found")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"blob {sha} failed content-hash verification")
        return data

    # ------------------------------------------------------------------ resumable task runs (benchmark seam)
    def claim_task_run(
        self,
        task_run_id: str,
        environment_id: str,
        task_id: str,
        partition: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Atomically claim a task run (pending -> running). Returns
        (True, row) for a fresh claim, (False, existing row) when resuming."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM task_runs WHERE task_run_id = ?", (task_run_id,)
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    return False, dict(existing)
                conn.execute(
                    "INSERT INTO task_runs (task_run_id, environment_id, task_id, partition, status, state_json, updated_at) VALUES (?, ?, ?, ?, 'running', '{}', ?)",
                    (task_run_id, environment_id, task_id, partition, _utcnow()),
                )
                conn.commit()
                return True, dict(conn.execute("SELECT * FROM task_runs WHERE task_run_id = ?", (task_run_id,)).fetchone())
            except Exception:
                conn.rollback()
                raise

    def update_task_run_status(self, task_run_id: str, status: str, state_json: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE task_runs SET status = ?, state_json = COALESCE(?, state_json), updated_at = ? WHERE task_run_id = ?",
                (status, state_json, _utcnow(), task_run_id),
            )
            conn.commit()

    def get_task_run(self, task_run_id: str) -> dict[str, Any] | None:
        return self._get_json("task_runs", "task_run_id", task_run_id)

    def claim_benchmark_task_run(
        self,
        task_run_id: str,
        *,
        benchmark_id: str,
        environment_id: str,
        task_id: str,
        partition: str,
        arm: str | None = None,
        seed: int | None = None,
        owner_id: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Single-owner claim for a benchmark task run.

        Fresh insert -> (True, row) with status 'running' bound to owner_id.
        Same-owner re-claim (restart/resume) -> (False, existing row).
        Different owner while the row exists -> (False, existing row); the
        caller sees the foreign owner and must not take over.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM task_runs WHERE task_run_id = ?", (task_run_id,)
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    return False, dict(existing)
                conn.execute(
                    "INSERT INTO task_runs (task_run_id, environment_id, task_id, partition, benchmark_id, arm, seed, owner_id, status, state_json, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', '{}', ?)",
                    (task_run_id, environment_id, task_id, partition, benchmark_id, arm, seed, owner_id, _utcnow()),
                )
                conn.commit()
                return True, dict(conn.execute("SELECT * FROM task_runs WHERE task_run_id = ?", (task_run_id,)).fetchone())
            except Exception:
                conn.rollback()
                raise

    def release_task_run(self, task_run_id: str, owner_id: str, status: str, state_json: str | None = None) -> bool:
        """Owner-guarded status transition; refuses writes from other owners."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT owner_id FROM task_runs WHERE task_run_id = ?", (task_run_id,)
                ).fetchone()
                if row is None or row["owner_id"] != owner_id:
                    conn.rollback()
                    return False
                conn.execute(
                    "UPDATE task_runs SET status = ?, state_json = COALESCE(?, state_json), updated_at = ? WHERE task_run_id = ?",
                    (status, state_json, _utcnow(), task_run_id),
                )
                conn.commit()
                return True
            except Exception:
                conn.rollback()
                raise

    def list_task_runs(
        self,
        environment_id: str | None = None,
        partition: str | None = None,
        status: str | None = None,
        benchmark_id: str | None = None,
        arm: str | None = None,
        owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM task_runs WHERE 1=1"
        params: list[Any] = []
        if environment_id is not None:
            query += " AND environment_id = ?"
            params.append(environment_id)
        if partition is not None:
            query += " AND partition = ?"
            params.append(partition)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if benchmark_id is not None:
            query += " AND benchmark_id = ?"
            params.append(benchmark_id)
        if arm is not None:
            query += " AND arm = ?"
            params.append(arm)
        if owner_id is not None:
            query += " AND owner_id = ?"
            params.append(owner_id)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ atomic evaluation allocation
    def reserve_allocation(
        self,
        scope_id: str,
        allocation_id: str,
        panels: list[list[str]],
        limit: int,
    ) -> int | None:
        """Atomically reserve the next free panel for a validation allocation.

        Mirrors the benchmark/evaluator contract: None when the allocation id is
        already consumed or the panel pool is exhausted; the reserved panel
        index on success. Restart-safe: persisted in SQLite.
        """
        if not panels or len(panels) < limit:
            raise ValueError("allocation panels must cover the configured limit")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT panel_index FROM evaluator_allocations WHERE allocation_id = ?",
                    (allocation_id,),
                ).fetchone()
                if existing:
                    conn.commit()
                    return None
                used = {
                    int(r["panel_index"])
                    for r in conn.execute(
                        "SELECT panel_index FROM evaluator_allocations WHERE scope_id = ?",
                        (scope_id,),
                    ).fetchall()
                }
                index = next((i for i in range(limit) if i not in used), None)
                if index is None:
                    conn.commit()
                    return None
                task_ids = list(panels[index])
                conn.execute(
                    "INSERT INTO evaluator_allocations (scope_id, allocation_id, panel_index, panel_hash, task_ids_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (scope_id, allocation_id, index, sha256_json(task_ids), json.dumps(task_ids, sort_keys=True), _utcnow()),
                )
                conn.commit()
                return index
            except Exception:
                conn.rollback()
                raise

    def get_allocation(self, allocation_id: str) -> dict[str, Any] | None:
        """Read a reserved allocation back (crash-after-allocation recovery).

        Returns the row plus a decoded `task_ids` list, or None.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluator_allocations WHERE allocation_id = ?",
                (allocation_id,),
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["task_ids"] = json.loads(row["task_ids_json"])
        return out

    # ------------------------------------------------------------------ trusted development smoke gate
    def dev_smoke_ok(self, environment_id: str) -> bool:
        """True when the environment has at least one development-partition run
        with a trusted outcome recorded (the held-out gate prerequisite)."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT 1 FROM outcomes o
                    JOIN runs r ON r.run_id = o.run_id
                    JOIN tasks t ON t.id = r.task_id
                    WHERE r.environment_id = ? AND t.partition = 'development' AND o.passed = 1
                    LIMIT 1""",
                (environment_id,),
            ).fetchone()
            return row is not None

    # ------------------------------------------------------------------ learning projection (session7 seam)
    def get_public_docs(self, environment_id: str) -> list[dict[str, Any]]:
        """Public document contents for a manifest's doc refs.

        Only artifacts without a restricted classification are returned;
        artifacts classified 'operator' or 'evaluator_only' are excluded so no
        hidden evaluator content reaches the learner.
        """
        row = self.get_environment(environment_id)
        if not row:
            return []
        manifest_ref = ArtifactRef.model_validate_json(row["manifest_ref"])
        manifest = self.get_artifact(manifest_ref)
        out: list[dict[str, Any]] = []
        for doc in manifest.get("docs", []):
            ref = doc if isinstance(doc, dict) else {}
            sha = ref.get("sha256")
            if not sha or not self.has_artifact(sha):
                continue
            payload = self.get_artifact(sha)
            if isinstance(payload, dict) and payload.get("classification") in ("operator", "evaluator_only"):
                continue
            out.append({"id": ref.get("id"), "version": ref.get("version"), "sha256": sha, "content": payload})
        return out

    def list_learner_evidence(
        self,
        environment_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Redacted learner-visible DEVELOPMENT evidence joined to run/task and
        the trusted outcome (when present). Never returns operator or
        evaluator_only rows, and never non-development partitions."""
        query = (
            "SELECT e.evidence_id, e.run_id, e.sequence, e.event_type, e.content_hash, "
            "e.trust_class, e.visibility, e.redacted, "
            "r.task_id, r.environment_id, t.partition, "
            "o.passed AS outcome_passed, o.score AS outcome_score, o.checked_at AS outcome_checked_at "
            "FROM evidence e "
            "JOIN runs r ON r.run_id = e.run_id "
            "JOIN tasks t ON t.id = r.task_id "
            "LEFT JOIN outcomes o ON o.run_id = e.run_id "
            "WHERE e.visibility = 'learner' AND e.redacted = 1 AND t.partition = 'development'"
        )
        params: list[Any] = []
        if environment_id is not None:
            query += " AND r.environment_id = ?"
            params.append(environment_id)
        if run_id is not None:
            query += " AND e.run_id = ?"
            params.append(run_id)
        query += " ORDER BY e.run_id, e.sequence"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def list_learning_evidence(
        self,
        environment_id: str | None = None,
        run_id: str | None = None,
        include_broker_projection: bool = True,
    ) -> list[dict[str, Any]]:
        """The runtime's development evidence feed.

        Returns sanitized learner-visible evidence rows (kind="evidence",
        joined to trusted outcome) plus, per development run in scope, the
        broker call/evidence join rows from list_run_tool_calls
        (kind="broker_call") — canonical safe fields only, never raw
        operator/evaluator_only rows.
        """
        evidence = [dict(r, kind="evidence") for r in self.list_learner_evidence(environment_id, run_id)]
        if not include_broker_projection:
            return evidence
        query = (
            "SELECT r.run_id FROM runs r JOIN tasks t ON t.id = r.task_id "
            "WHERE t.partition = 'development'"
        )
        params: list[Any] = []
        if environment_id is not None:
            query += " AND r.environment_id = ?"
            params.append(environment_id)
        if run_id is not None:
            query += " AND r.run_id = ?"
            params.append(run_id)
        with self._connect() as conn:
            run_ids = [r["run_id"] for r in conn.execute(query, params).fetchall()]
        calls: list[dict[str, Any]] = []
        for rid in run_ids:
            calls.extend(dict(r, kind="broker_call") for r in self.list_run_tool_calls(rid))
        return evidence + calls

    # ------------------------------------------------------------------ broker call/evidence join (session7 seam)
    def list_run_tool_calls(self, run_id: str) -> list[dict[str, Any]]:
        """Read-only sanitized projection of one DEVELOPMENT run's broker calls
        joined to their evidence rows.

        Fails closed on non-development partitions and unknown runs. Each row:
        call_id, evidence_id, tool, input, result, errorCode, retry, version,
        arguments_sha256, result_sha256, run/task/environment bindings.
        Approval tokens are never included.
        """
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"run {run_id!r} not found")
        task = self.get_task(run["task_id"])
        if not task or task.get("partition") != "development":
            raise PermissionError("tool-call projection is development-partition only")

        # Manifest tool-schema allowlist: input keys are restricted to the
        # schema's declared properties, and calls for tools absent from the
        # manifest are not projected (fail closed).
        schema_keys: dict[str, set[str]] | None = {}
        env_row = self.get_environment(run["environment_id"])
        if env_row:
            try:
                mref = json.loads(env_row["manifest_ref"])
                manifest = self.get_artifact(mref["sha256"])
                for ts in manifest.get("toolSchemas", []):
                    if isinstance(ts, dict) and isinstance(ts.get("name"), str):
                        props = (ts.get("inputSchema") or {}).get("properties") or {}
                        schema_keys[ts["name"]] = set(props)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, AttributeError):
                schema_keys = None
        if not schema_keys:
            return []  # no validated manifest -> no projection

        def _valid_source(ev: dict[str, Any]) -> dict[str, Any] | None:
            """Broker evidence with verified source artifact + content hash."""
            if ev["event_type"] != "tool_result" or ev["trust_class"] != "broker":
                return None
            if ev["visibility"] not in {"learner", "operator"}:
                return None
            try:
                ref = json.loads(ev["source_ref"])
                payload = self.get_artifact(ref["sha256"])
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                return None
            if ev.get("content_hash") != ref["sha256"]:
                return None  # content hash must match the source artifact
            return payload if isinstance(payload, dict) else None

        calls = [
            r for r in self.list_tool_calls(run_id)
        ]
        evidence_by_call: dict[str, dict[str, Any]] = {}
        for ev in self.list_evidence(run_id):
            payload = _valid_source(ev)
            if payload is not None and isinstance(payload.get("callId"), str):
                evidence_by_call[payload["callId"]] = ev

        out: list[dict[str, Any]] = []
        matched: set[str] = set()
        for call in calls:
            try:
                args = json.loads(call["arguments_json"])
            except (TypeError, json.JSONDecodeError):
                args = {}
            result: dict[str, Any] = {}
            error_code = retry = None
            if call.get("result_json"):
                try:
                    result = json.loads(call["result_json"])
                except (TypeError, json.JSONDecodeError):
                    result = {}
            if isinstance(result, dict):
                err = result.get("error")
                if isinstance(err, dict):
                    error_code = err.get("code")
                    retry = err.get("retry")
                status = result.get("status")
                output = result.get("output")
                version = result.get("toolVersion")
                effect = result.get("effect")
                result_sha = sha256_json(result)
            else:
                status = output = version = effect = result_sha = None
            allowed = schema_keys.get(call["tool"])
            if allowed is None:
                continue  # tool not in the environment manifest
            ev = evidence_by_call.get(call["call_id"], {})
            if ev:
                matched.add(call["call_id"])
            out.append({
                "callId": call["call_id"],
                "evidenceId": ev.get("evidence_id"),
                "tool": call["tool"],
                "input": sanitize_for_learner({k: args[k] for k in args if k in allowed}),
                "result": sanitize_for_learner(output) if output is not None else None,
                "status": status,
                "errorCode": error_code,
                "retry": retry,
                "version": version,
                "effect": effect or call.get("effect"),
                "idempotencyKey": call["idempotency_key"],
                "argumentsSha256": sha256_json(args),
                "resultSha256": result_sha,
                "evidenceContentHash": ev.get("content_hash"),
                "runId": run_id,
                "taskId": run["task_id"],
                "environmentId": run["environment_id"],
                "partition": "development",
                "visibility": "learner",
                "redacted": True,
            })
        # Evidence rows with no matching tool_calls row (older app paths that
        # recorded operator-only tool_result events): project the same safe
        # fields from the sanitized artifact payload instead of hiding them.
        for ev in self.list_evidence(run_id):
            if ev["event_type"] != "tool_result" or ev["trust_class"] != "broker" or ev["visibility"] != "operator":
                continue
            payload = _valid_source(ev)
            if payload is None:
                # Emit a minimal bound row only when the source artifact is
                # unresolvable; a resolved-but-hash-mismatched row is tampered
                # and must be dropped.
                try:
                    ref = json.loads(ev["source_ref"])
                    self.get_artifact(ref["sha256"])
                except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                    payload = {}
                else:
                    continue
            call_id = payload.get("callId")
            if isinstance(call_id, str) and call_id in matched:
                continue
            tool = payload.get("tool")
            allowed = schema_keys.get(tool) if isinstance(tool, str) else None
            if tool is not None and allowed is None:
                continue  # declared tool absent from the manifest
            raw_input = payload.get("input")
            err = payload.get("error")
            out.append({
                "callId": call_id,
                "evidenceId": ev["evidence_id"],
                "tool": tool,
                "input": sanitize_for_learner({k: raw_input[k] for k in raw_input if allowed is not None and k in allowed}) if isinstance(raw_input, dict) else None,
                "result": sanitize_for_learner(payload.get("output")),
                "status": payload.get("status"),
                "errorCode": err.get("code") if isinstance(err, dict) else None,
                "retry": err.get("retry") if isinstance(err, dict) else None,
                "version": payload.get("toolVersion"),
                "effect": payload.get("effect"),
                "idempotencyKey": None,
                "argumentsSha256": None,
                "resultSha256": sha256_json(payload),
                "evidenceContentHash": ev.get("content_hash"),
                "runId": run_id,
                "taskId": run["task_id"],
                "environmentId": run["environment_id"],
                "partition": "development",
                "visibility": "learner",
                "redacted": True,
            })
        return out

    # ------------------------------------------------------------------ run receipts (EvaluationJob seam)
    def get_run_receipts(self, run_id: str) -> dict[str, Any]:
        """Exact durable receipts for one run, for EvaluationJob consumption.

        Single-path: model_response evidence payloads, accounting artifacts
        (with cumulative aggregates), the trusted outcome row, and run/task/env
        binding. Read-only; does not project hidden evaluator content.
        """
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"run {run_id!r} not found")
        model_responses: list[dict[str, Any]] = []
        accountings: list[dict[str, Any]] = []
        for row in self.list_evidence(run_id):
            try:
                ref = json.loads(row["source_ref"])
                payload = self.get_artifact(ref["sha256"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if row["event_type"] == "model_response" and isinstance(payload, dict):
                model_responses.append(payload)
            elif row["event_type"] == "accounting_recorded" and isinstance(payload, dict):
                try:
                    acct = self.get_artifact(payload["accountingRef"]["sha256"])
                    if isinstance(acct, dict):
                        accountings.append(acct)
                except (KeyError, TypeError):
                    continue
        outcome = self.get_outcome_by_run_id(run_id)
        return {
            "runId": run_id,
            "taskId": run["task_id"],
            "environmentId": run["environment_id"],
            "status": run["status"],
            "modelResponses": model_responses,
            "accounting": accountings,
            "trustedOutcome": outcome,
        }

    # ------------------------------------------------------------------ learning records (session7 seam)
    def save_learning_record(self, record_id: str, environment_id: str, run_id: str, record_json: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO learning_records (record_id, environment_id, run_id, record_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (record_id, environment_id, run_id, record_json, _utcnow()),
            )
            conn.commit()

    def list_learning_records(self, environment_id: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]:
        """Read-only learning-record query; filtered by env and/or run."""
        query = "SELECT * FROM learning_records WHERE 1=1"
        params: list[Any] = []
        if environment_id is not None:
            query += " AND environment_id = ?"
            params.append(environment_id)
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        query += " ORDER BY created_at"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            row = dict(r)
            try:
                decoded = json.loads(row.get("record_json") or "{}")
                if isinstance(decoded, dict):
                    row.update(decoded)  # record fields queryable alongside record_json
            except (TypeError, json.JSONDecodeError):
                pass
            out.append(row)
        return out

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

    def create_run_idempotent(
        self,
        idempotency_key: str,
        request_fingerprint: str,
        run_data: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Atomic request-fingerprint run creation.

        Returns ("exists", row) when the key already belongs to an identical
        request, ("created", row) on success, and raises
        RunIdempotencyConflict when the key is bound to different content.
        """
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(
                    "SELECT * FROM runs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if existing is not None:
                    conn.commit()
                    row = dict(existing)
                    if row.get("request_fingerprint") != request_fingerprint:
                        raise RunIdempotencyConflict(
                            f"idempotency key {idempotency_key!r} already bound to a different request"
                        )
                    return "exists", row
                run_data = dict(run_data)
                run_id = run_data.pop("run_id", None)
                if run_id is None:
                    raise ValueError("run_data must include run_id")
                run_data["idempotency_key"] = idempotency_key
                run_data["request_fingerprint"] = request_fingerprint
                columns = ["run_id"] + list(run_data.keys())
                values = [run_id] + list(run_data.values())
                conn.execute(
                    f"INSERT INTO runs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                    values,
                )
                conn.commit()
                row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
                return "created", dict(row)
            except RunIdempotencyConflict:
                raise
            except Exception:
                conn.rollback()
                raise

    def list_environments(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM environments ORDER BY registered_at").fetchall()
            return [dict(r) for r in rows]

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
        if data.get("event_type") == "model_observation" or data.get("eventType") == "model_observation":
            raise ValueError("model_observation is not a canonical evidence event; use model_response")
        self._insert_json("evidence", "evidence_id", evidence_id, data)

    def get_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        return self._get_json("evidence", "evidence_id", evidence_id)

    def list_evidence(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence WHERE run_id = ? ORDER BY sequence", (run_id,)
            ).fetchall()
            return [dict(r) for r in rows]

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

    def list_tool_calls(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tool_calls WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

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
        evaluator_refs: list[str] | None = None,
        fixture_hashes: dict[str, str] | None = None,
        partition_hashes: dict[str, str] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO frozen_protocols (protocol_hash, gate_json, evaluator_id,
                    evaluator_refs_json, fixture_hashes_json, partition_hashes_json, frozen_at, active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    protocol_hash,
                    gate_json,
                    evaluator_id,
                    json.dumps(sorted(evaluator_refs or [])),
                    json.dumps(fixture_hashes or {}, sort_keys=True),
                    json.dumps(partition_hashes or {}, sort_keys=True),
                    _utcnow(),
                ),
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

    def get_evaluation(self, report_id: str) -> dict[str, Any] | None:
        return self._get_json("evaluations", "report_id", report_id)

    def get_evaluation_by_protocol_and_candidate(self, protocol_hash: str, candidate_hash: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE protocol_hash = ? AND candidate_hash = ?",
                (protocol_hash, candidate_hash),
            ).fetchone()
            return dict(row) if row else None

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
