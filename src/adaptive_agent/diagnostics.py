"""Durable, bounded development diagnostics for candidate comparisons."""

from __future__ import annotations
import concurrent.futures
import fcntl
import json
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Mapping
from adaptive_agent.benchmark import FrozenExecutionConfig
from adaptive_agent.evaluation import (
    Arm,
    BudgetSpec,
    FrozenProtocol,
    ModelProvenance,
    Partition,
    Provenance,
    RunObservation,
    canonical_json,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class DevelopmentDiagnosticManager:
    """Own six-cell B0/L development comparisons outside promotion state."""

    def __init__(
        self, runtime: Any, executor: Callable[[Any, Any, Any], Any] | None = None
    ) -> None:
        self.runtime, self.store, self.executor = (
            runtime,
            runtime.controller.store,
            executor,
        )
        self._events: dict[str, threading.Event] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._owner_id = f"diag-owner-{uuid.uuid4().hex}"
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self.store.connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS diagnostics (
              diagnostic_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, base_bundle_hash TEXT NOT NULL,
              candidate_bundle_hash TEXT NOT NULL, protocol_hash TEXT NOT NULL, environment_id TEXT NOT NULL,
              task_id TEXT NOT NULL, seed INTEGER NOT NULL, state TEXT NOT NULL,
              cancel_requested INTEGER NOT NULL DEFAULT 0, started_at TEXT, updated_at TEXT NOT NULL,
              error TEXT, limits_json TEXT NOT NULL, promotion_eligible INTEGER NOT NULL DEFAULT 0,
              protocol_json TEXT, task_ids_json TEXT, launch_identity TEXT, owner_id TEXT);
            CREATE TABLE IF NOT EXISTS diagnostic_cells (
              diagnostic_id TEXT NOT NULL, cell_key TEXT NOT NULL, arm TEXT NOT NULL, status TEXT NOT NULL,
              result_json TEXT NOT NULL DEFAULT '{}', receipt_ref TEXT, failure_class TEXT, error TEXT,
              started_at TEXT, updated_at TEXT NOT NULL, task_id TEXT, environment_id TEXT, seed INTEGER,
              protocol_hash TEXT, bundle_hash TEXT, observation_json TEXT, PRIMARY KEY (diagnostic_id, cell_key));
            """)
            dcols = {r["name"] for r in conn.execute("PRAGMA table_info(diagnostics)")}
            for n, t in (
                ("protocol_json", "TEXT"),
                ("task_ids_json", "TEXT"),
                ("launch_identity", "TEXT"),
                ("owner_id", "TEXT"),
            ):
                if n not in dcols:
                    conn.execute(f"ALTER TABLE diagnostics ADD COLUMN {n} {t}")
            ccols = {
                r["name"] for r in conn.execute("PRAGMA table_info(diagnostic_cells)")
            }
            for n, t in (
                ("task_id", "TEXT"),
                ("environment_id", "TEXT"),
                ("seed", "INTEGER"),
                ("protocol_hash", "TEXT"),
                ("bundle_hash", "TEXT"),
                ("observation_json", "TEXT"),
            ):
                if n not in ccols:
                    conn.execute(f"ALTER TABLE diagnostic_cells ADD COLUMN {n} {t}")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS diagnostics_launch_identity ON diagnostics(launch_identity) WHERE launch_identity IS NOT NULL"
            )
            conn.commit()

    @staticmethod
    def _bundle(runtime: Any, content_hash: str) -> Any:
        row = runtime.controller.store.get_bundle_by_hash(content_hash)
        if row is None:
            raise ValueError("bundle is not durable")
        from adaptive_agent.models import SkillBundle

        try:
            bundle = SkillBundle.model_validate(json.loads(row["bundle_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("bundle is malformed") from exc
        if bundle.content_hash != content_hash:
            raise ValueError("bundle content hash does not match persisted identity")
        return bundle

    @contextmanager
    def _owner_lock(self, diagnostic_id: str):
        lock = self._locks.setdefault(diagnostic_id, threading.Lock())
        if not lock.acquire(False):
            yield False
            return
        path = Path(f"{self.store.db_path}.diagnostic.{diagnostic_id}.lock")
        handle = path.open("a+")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            yield True
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
                lock.release()

    def _record(self, diagnostic_id: str) -> dict[str, Any]:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM diagnostics WHERE diagnostic_id = ?", (diagnostic_id,)
            ).fetchone()
        if row is None:
            raise KeyError("diagnostic not found")
        value = dict(row)
        value["armSummaries"] = self._summaries(diagnostic_id, value)
        with self.store.connect() as conn:
            completed = conn.execute(
                "SELECT * FROM diagnostic_cells WHERE diagnostic_id = ? AND status = 'completed'",
                (diagnostic_id,),
            ).fetchall()
        if not self._validate_plan(value) or any(
            not self._validate_cached(cell, value) for cell in completed
        ):
            value["error"] = (
                value.get("error")
                or "persisted diagnostic evidence is unavailable or invalid"
            )
            value["_integrity_invalid"] = True
        return value

    def _public(self, r: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "diagnosticId": r["diagnostic_id"],
            "candidateId": r["candidate_id"],
            "baseBundleHash": r["base_bundle_hash"],
            "candidateBundleHash": r["candidate_bundle_hash"],
            "protocolHash": r["protocol_hash"],
            "state": "failed" if r.get("_integrity_invalid") else r["state"],
            "completedCells": int(r.get("completed_cells", 0)),
            "totalCells": 6,
            "startedAt": r.get("started_at"),
            "updatedAt": r["updated_at"],
            "armSummaries": r.get("armSummaries", []),
            "error": r.get("error"),
            "promotionEligible": False,
            "purpose": "development_diagnostic",
            "resumable": self._resumable(r),
        }

    def _resumable(self, record: Mapping[str, Any]) -> bool:
        if record.get("state") not in {"queued", "running"}:
            return False
        lock = self._locks.setdefault(str(record["diagnostic_id"]), threading.Lock())
        if not lock.acquire(False):
            return False
        path = Path(f"{self.store.db_path}.diagnostic.{record['diagnostic_id']}.lock")
        handle = path.open("a+")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            return True
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
                lock.release()

    def _summaries(
        self, diagnostic_id: str, record: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM diagnostic_cells WHERE diagnostic_id = ? ORDER BY arm,cell_key",
                (diagnostic_id,),
            ).fetchall()
        out = []
        for arm in ("B0", "L"):
            completed = successes = total_tokens = infra = 0
            wall = 0.0
            for row in rows:
                if row["arm"] != arm:
                    continue
                if row["status"] == "failed":
                    infra += 1
                    continue
                if row["status"] != "completed":
                    continue
                if record is not None and not self._validate_cached(row, record):
                    infra += 1
                    continue
                completed += 1
                try:
                    value = json.loads(row["result_json"] or "{}")
                except (TypeError, ValueError, json.JSONDecodeError):
                    value = {}
                successes += int(value.get("passed") is True)
                tokens = value.get("totalTokens")
                if (
                    isinstance(tokens, int)
                    and not isinstance(tokens, bool)
                    and tokens >= 0
                ):
                    total_tokens += tokens
                if isinstance(value.get("wallDurationSeconds"), (int, float)):
                    wall += float(value["wallDurationSeconds"])
            out.append(
                {
                    "arm": arm,
                    "completed": completed,
                    "successes": successes,
                    "meanScore": successes / completed if completed else None,
                    "totalTokens": total_tokens,
                    "wallDurationSeconds": wall,
                    "infrastructureErrors": infra,
                }
            )
        return out

    def _record_with_counts(self, diagnostic_id: str) -> dict[str, Any]:
        r = self._record(diagnostic_id)
        with self.store.connect() as conn:
            r["completed_cells"] = conn.execute(
                "SELECT COUNT(*) FROM diagnostic_cells WHERE diagnostic_id = ? AND status IN ('completed','failed')",
                (diagnostic_id,),
            ).fetchone()[0]
        return self._public(r)

    def create(self, candidate_id: str, base_bundle_hash: str) -> dict[str, Any]:
        protocol = getattr(self.runtime, "_evaluation_protocol", None)
        if protocol is None:
            raise ValueError("evaluation protocol is not frozen")
        frozen = protocol.start_candidate_generation()
        (
            active,
            candidate,
        ) = self.runtime.controller.get_active_bundle(), self.runtime.controller.get_candidate(
            candidate_id
        )
        if candidate is None:
            raise KeyError("candidate not found")
        if candidate.get("state") not in {"validated", "evaluating"}:
            raise ValueError("candidate is not validated")
        candidate_hash = candidate.get("candidate_bundle_hash")
        if not isinstance(candidate_hash, str) or not candidate_hash:
            try:
                candidate_hash = json.loads(candidate["candidate_json"]).get(
                    "candidateBundleHash"
                )
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
        inputs = frozen.inputs if isinstance(frozen.inputs, Mapping) else {}
        seeds = inputs.get("seeds")
        envs = inputs.get("knownEnvironments", protocol.known_environments)
        if not isinstance(seeds, list) or not seeds or not isinstance(seeds[0], int):
            raise ValueError("frozen protocol has no valid diagnostic seed")
        tasks = []
        for name in tuple(envs):
            values = tuple(
                self.runtime.packages[name].tasks_for_partition("development")
            )
            if not values:
                raise ValueError(f"environment {name!r} has no development tasks")
            tasks.append((name, values[0]))
        seed = int(seeds[0])
        task_ids = {name: task.task_id for name, task in tasks}
        identity = json.dumps(
            {
                "candidateId": candidate_id,
                "baseBundleHash": base_bundle_hash,
                "candidateBundleHash": candidate_hash,
                "protocolHash": frozen.protocol_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.store.connect() as conn:
            existing = conn.execute(
                "SELECT diagnostic_id FROM diagnostics WHERE launch_identity = ?",
                (identity,),
            ).fetchone()
        if existing is not None:
            return self.get(existing["diagnostic_id"])
        did, now = f"diag_{uuid.uuid4().hex}", _now()
        try:
            with self.store.connect() as conn:
                conn.execute(
                    "INSERT INTO diagnostics(diagnostic_id,candidate_id,base_bundle_hash,candidate_bundle_hash,protocol_hash,environment_id,task_id,seed,state,updated_at,limits_json,protocol_json,task_ids_json,launch_identity) VALUES (?,?,?,?,?,?,?,?,'queued',?,?,?,?,?)",
                    (
                        did,
                        candidate_id,
                        base_bundle_hash,
                        candidate_hash,
                        frozen.protocol_hash,
                        tasks[0][0],
                        tasks[0][1].task_id,
                        seed,
                        now,
                        json.dumps(dict(inputs.get("runBudget", {})), sort_keys=True),
                        json.dumps(frozen.to_dict(), sort_keys=True),
                        json.dumps(task_ids, sort_keys=True),
                        identity,
                    ),
                )
                for name, task in tasks:
                    for arm in ("B0", "L"):
                        conn.execute(
                            "INSERT INTO diagnostic_cells(diagnostic_id,cell_key,arm,status,updated_at,task_id,environment_id,seed,protocol_hash,bundle_hash) VALUES (?,?,?,'queued',?,?,?,?,?,?)",
                            (
                                did,
                                f"{name}:{arm}",
                                arm,
                                now,
                                task.task_id,
                                name,
                                seed,
                                frozen.protocol_hash,
                                base_bundle_hash if arm == "B0" else candidate_hash,
                            ),
                        )
                conn.commit()
        except Exception:
            with self.store.connect() as conn:
                existing = conn.execute(
                    "SELECT diagnostic_id FROM diagnostics WHERE launch_identity = ?",
                    (identity,),
                ).fetchone()
            if existing is None:
                raise
            did = existing["diagnostic_id"]
        self._events.setdefault(did, threading.Event())
        self._locks.setdefault(did, threading.Lock())
        return self._record_with_counts(did)

    def list(self) -> list[dict[str, Any]]:
        with self.store.connect() as conn:
            ids = [
                r["diagnostic_id"]
                for r in conn.execute(
                    "SELECT diagnostic_id FROM diagnostics ORDER BY updated_at DESC"
                )
            ]
        return [self._record_with_counts(i) for i in ids]

    def get(self, diagnostic_id: str) -> dict[str, Any]:
        return self._record_with_counts(diagnostic_id)

    def cancel(self, diagnostic_id: str) -> dict[str, Any]:
        self._events.setdefault(diagnostic_id, threading.Event()).set()
        now = _now()
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT state FROM diagnostics WHERE diagnostic_id = ?",
                (diagnostic_id,),
            ).fetchone()
            if row is None:
                raise KeyError("diagnostic not found")
            if row["state"] in {"completed", "failed", "cancelled"}:
                return self.get(diagnostic_id)
            conn.execute(
                "UPDATE diagnostics SET cancel_requested=1,state=CASE WHEN state='queued' THEN 'cancelled' ELSE state END,error=CASE WHEN state='queued' THEN 'cancelled before dispatch' ELSE 'cancellation requested; admitted cells will finish' END,updated_at=? WHERE diagnostic_id=?",
                (now, diagnostic_id),
            )
            conn.execute(
                "UPDATE diagnostic_cells SET status='cancelled',updated_at=? WHERE diagnostic_id=? AND status='queued'",
                (now, diagnostic_id),
            )
            conn.commit()
        return self.get(diagnostic_id)

    def _cancel_requested(self, did: str) -> bool:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM diagnostics WHERE diagnostic_id = ?",
                (did,),
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def _load_frozen(self, record: Mapping[str, Any]) -> FrozenProtocol:
        try:
            p = json.loads(record.get("protocol_json") or "")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                "diagnostic frozen protocol is missing or malformed"
            ) from exc
        if not isinstance(p, dict) or p.get("protocolHash") != record.get(
            "protocol_hash"
        ):
            raise ValueError("diagnostic frozen protocol identity mismatch")
        return FrozenProtocol(
            str(p["protocolHash"]),
            MappingProxyType(dict(p.get("fixtureHashes", {}))),
            MappingProxyType(dict(p.get("partitionHashes", {}))),
            MappingProxyType(dict(p.get("inputs", {}))),
        )

    def _validate_cached(
        self, row: Mapping[str, Any], record: Mapping[str, Any]
    ) -> bool:
        row = dict(row)
        if row.get("status") == "failed":
            return bool(row.get("failure_class")) and bool(row.get("error"))
        try:
            value = json.loads(row.get("result_json") or "{}")
            receipt = self.store.get_artifact(str(row.get("receipt_ref")))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if (
            not isinstance(value, dict)
            or not isinstance(receipt, dict)
            or receipt != value
        ):
            return False
        if (
            value.get("taskId") != row.get("task_id")
            or value.get("arm") != row.get("arm")
            or value.get("seed") != row.get("seed")
            or value.get("protocolHash") != record.get("protocol_hash")
            or value.get("bundleHash") != row.get("bundle_hash")
        ):
            return False
        from adaptive_agent.models import sha256_json

        accounting_ref = value.get("accountingRef")
        if accounting_ref is not None:
            if not isinstance(accounting_ref, str):
                return False
            try:
                accounting = self.store.get_artifact(accounting_ref)
            except (KeyError, TypeError, ValueError):
                return False
            if (
                not isinstance(accounting, Mapping)
                or sha256_json(accounting) != accounting_ref
            ):
                return False
        serialized = row.get("observation_json")
        if not serialized:
            return False
        try:
            payload = json.loads(serialized)
            payload["partition"] = Partition(payload["partition"])
            payload["arm"] = Arm(payload["arm"])
            payload["provenance"] = Provenance(
                payload.get("provenance", "deterministic_simulation")
            )
            payload["model_provenance"] = ModelProvenance(
                payload.get("model_provenance", "synthetic_model")
            )
            payload["budget"] = BudgetSpec(**payload["budget"])
            observation = RunObservation(**payload)
            verifier = getattr(self.runtime, "verify_evaluation_observation", None)
            task = next(
                task
                for task in self.runtime.packages[
                    str(row["environment_id"])
                ].tasks_for_partition("development")
                if task.task_id == row["task_id"]
            )
            config = FrozenExecutionConfig(
                self._load_frozen(record),
                Arm(str(row["arm"])),
                int(row["seed"]),
                str(row["bundle_hash"]),
            )
            if not callable(verifier) or not verifier(
                observation,
                config,
                task,
            ):
                return False
            if self._result_value(observation, task, config) != value:
                return False
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, StopIteration):
            return False
        return (
            isinstance(row.get("receipt_ref"), str)
            and sha256_json(value) == row["receipt_ref"]
        )

    def _validate_plan(self, record: Mapping[str, Any]) -> bool:
        try:
            task_ids = json.loads(record.get("task_ids_json") or "")
            frozen = self._load_frozen(record)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        environments = frozen.inputs.get("knownEnvironments")
        if (
            not isinstance(task_ids, dict)
            or not isinstance(environments, list)
            or len(environments) != 3
        ):
            return False
        expected = {
            (str(environment), arm)
            for environment in environments
            for arm in ("B0", "L")
        }
        with self.store.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM diagnostic_cells WHERE diagnostic_id = ?",
                (record["diagnostic_id"],),
            ).fetchall()
        if (
            len(rows) != 6
            or {(str(row["environment_id"]), str(row["arm"])) for row in rows}
            != expected
        ):
            return False
        for row in rows:
            if (
                task_ids.get(str(row["environment_id"])) != row["task_id"]
                or row["seed"] != record["seed"]
                or row["protocol_hash"] != record["protocol_hash"]
            ):
                return False
            expected_bundle = (
                record["base_bundle_hash"]
                if row["arm"] == "B0"
                else record["candidate_bundle_hash"]
            )
            if row["bundle_hash"] != expected_bundle:
                return False
        return True

    def _result_value(
        self, observation: Any, task: Any, config: FrozenExecutionConfig
    ) -> dict[str, Any]:
        if not isinstance(observation, RunObservation):
            raise ValueError("runtime observation must be a trusted RunObservation")
        if (
            not isinstance(observation.accounting_ref, str)
            or not observation.accounting_ref
        ):
            raise ValueError("runtime observation lacks final accounting reference")
        raw = self.store.get_artifact(observation.accounting_ref)
        if not isinstance(raw, Mapping):
            raise ValueError("runtime observation accounting receipt is malformed")
        accounting: Mapping[str, Any] = raw
        aggregate = accounting.get("aggregateUsage")
        top_level = {
            k: accounting.get(k) for k in ("inputTokens", "outputTokens", "totalTokens")
        }
        usage = aggregate if isinstance(aggregate, Mapping) else top_level
        if not isinstance(aggregate, Mapping) and not all(
            isinstance(v, int) for v in top_level.values()
        ):
            usage = top_level
        canonical = {
            k: usage.get(k) for k in ("inputTokens", "outputTokens", "totalTokens")
        }
        if not all(
            isinstance(v, int) and not isinstance(v, bool) and v >= 0
            for v in canonical.values()
        ):
            raise ValueError("evaluation accounting lacks canonical token counts")
        if (
            canonical["totalTokens"]
            != canonical["inputTokens"] + canonical["outputTokens"]
        ):
            raise ValueError("evaluation accounting totalTokens is inconsistent")
        passed = bool(
            getattr(
                observation,
                "passed",
                (
                    observation.get("passed", False)
                    if isinstance(observation, Mapping)
                    else False
                ),
            )
        )
        wall = float(
            getattr(
                observation,
                "latency_seconds",
                (
                    observation.get("wallDurationSeconds", 0)
                    if isinstance(observation, Mapping)
                    else 0
                ),
            )
            or 0
        )
        return {
            "taskId": task.task_id,
            "arm": config.arm.value,
            "seed": config.seed,
            "protocolHash": config.protocol.protocol_hash,
            "bundleHash": config.bundle_hash,
            "passed": passed,
            "inputTokens": canonical["inputTokens"],
            "outputTokens": canonical["outputTokens"],
            "totalTokens": canonical["totalTokens"],
            "usage": dict(usage),
            "wallDurationSeconds": wall,
            "runId": getattr(observation, "run_id", None),
            "accountingRef": getattr(observation, "accounting_ref", None),
        }

    def _finish(
        self,
        did: str,
        cell: Any,
        status: str,
        value: Mapping[str, Any],
        ref: str | None,
        failure: str | None,
        error: str | None,
        observation_json: str | None = None,
    ) -> None:
        with self.store.connect() as conn:
            conn.execute(
                "UPDATE diagnostic_cells SET status=?,result_json=?,receipt_ref=?,failure_class=?,error=?,observation_json=?,updated_at=? WHERE diagnostic_id=? AND cell_key=?",
                (
                    status,
                    json.dumps(dict(value), sort_keys=True),
                    ref,
                    failure,
                    error,
                    observation_json,
                    _now(),
                    did,
                    cell["cell_key"],
                ),
            )
            conn.commit()

    def run(self, diagnostic_id: str) -> dict[str, Any]:
        with self._owner_lock(diagnostic_id) as acquired:
            if not acquired:
                return self.get(diagnostic_id)
            with self.store.connect() as conn:
                row = conn.execute(
                    "SELECT * FROM diagnostics WHERE diagnostic_id = ?",
                    (diagnostic_id,),
                ).fetchone()
                if row is None:
                    raise KeyError("diagnostic not found")
                record = dict(row)
                if not self._validate_plan(record):
                    conn.execute(
                        "UPDATE diagnostics SET state='failed',error=?,updated_at=? WHERE diagnostic_id=?",
                        (
                            "diagnostic plan is incomplete or identity-mismatched",
                            _now(),
                            diagnostic_id,
                        ),
                    )
                    conn.commit()
                    return self.get(diagnostic_id)
                if record["state"] in {"completed", "failed", "cancelled"}:
                    with self.store.connect() as verify_conn:
                        cached = verify_conn.execute(
                            "SELECT * FROM diagnostic_cells WHERE diagnostic_id = ? AND status IN ('completed','failed')",
                            (diagnostic_id,),
                        ).fetchall()
                    invalid = any(
                        not self._validate_cached(cell, record) for cell in cached
                    )
                    if invalid:
                        with self.store.connect() as repair_conn:
                            repair_conn.execute(
                                "UPDATE diagnostics SET state='failed', error=?, updated_at=? WHERE diagnostic_id=?",
                                (
                                    "persisted diagnostic receipt failed integrity verification",
                                    _now(),
                                    diagnostic_id,
                                ),
                            )
                            repair_conn.commit()
                    return self.get(diagnostic_id)
                frozen = self._load_frozen(record)
                current = getattr(self.runtime, "_evaluation_protocol", None)
                current_frozen = (
                    current.start_candidate_generation()
                    if current is not None
                    else None
                )
                if (
                    current_frozen is None
                    or current_frozen.to_dict() != frozen.to_dict()
                ):
                    conn.execute(
                        "UPDATE diagnostics SET state='failed',error=?,updated_at=? WHERE diagnostic_id=?",
                        (
                            "frozen protocol no longer matches runtime protocol",
                            _now(),
                            diagnostic_id,
                        ),
                    )
                    conn.commit()
                    return self.get(diagnostic_id)
                cached = conn.execute(
                    "SELECT * FROM diagnostic_cells WHERE diagnostic_id = ? AND status = 'completed'",
                    (diagnostic_id,),
                ).fetchall()
                invalid_cells = [
                    cell["cell_key"]
                    for cell in cached
                    if not self._validate_cached(cell, record)
                ]
                if invalid_cells:
                    placeholders = ",".join("?" for _ in invalid_cells)
                    conn.execute(
                        f"UPDATE diagnostic_cells SET status='failed',failure_class='evidence_integrity',error='persisted diagnostic evidence failed verification',updated_at=? WHERE diagnostic_id=? AND cell_key IN ({placeholders})",
                        (_now(), diagnostic_id, *invalid_cells),
                    )
                    conn.execute(
                        "UPDATE diagnostics SET state='failed',error=?,updated_at=? WHERE diagnostic_id=?",
                        (
                            "persisted diagnostic evidence failed verification",
                            _now(),
                            diagnostic_id,
                        ),
                    )
                    conn.commit()
                    return self.get(diagnostic_id)
                conn.execute(
                    "UPDATE diagnostics SET state='running',owner_id=?,started_at=COALESCE(started_at,?),updated_at=? WHERE diagnostic_id=?",
                    (self._owner_id, _now(), _now(), diagnostic_id),
                )
                conn.execute(
                    "UPDATE diagnostic_cells SET status='queued',updated_at=? WHERE diagnostic_id=? AND status='running'",
                    (_now(), diagnostic_id),
                )
                conn.commit()
                rows = conn.execute(
                    "SELECT * FROM diagnostic_cells WHERE diagnostic_id=? AND status='queued' ORDER BY cell_key",
                    (diagnostic_id,),
                ).fetchall()
            bundles = {
                "B0": self._bundle(self.runtime, record["base_bundle_hash"]),
                "L": self._bundle(self.runtime, record["candidate_bundle_hash"]),
            }
            event = self._events.setdefault(diagnostic_id, threading.Event())

            def execute(cell: Any) -> None:
                if event.is_set() or self._cancel_requested(diagnostic_id):
                    return
                with self.store.connect() as conn:
                    claimed = conn.execute(
                        "UPDATE diagnostic_cells SET status='running',started_at=?,updated_at=? WHERE diagnostic_id=? AND cell_key=? AND status='queued'",
                        (_now(), _now(), diagnostic_id, cell["cell_key"]),
                    ).rowcount
                    conn.commit()
                if not claimed:
                    return
                if event.is_set() or self._cancel_requested(diagnostic_id):
                    self._finish(diagnostic_id, cell, "cancelled", {}, None, None, None)
                    return
                task = next(
                    (
                        t
                        for t in self.runtime.packages[
                            str(cell["environment_id"])
                        ].tasks_for_partition("development")
                        if t.task_id == cell["task_id"]
                    ),
                    None,
                )
                if task is None:
                    self._finish(
                        diagnostic_id,
                        cell,
                        "failed",
                        {},
                        None,
                        "infrastructure_error",
                        "persisted task is no longer available",
                    )
                    return
                config = FrozenExecutionConfig(
                    frozen,
                    Arm(str(cell["arm"])),
                    int(cell["seed"]),
                    str(cell["bundle_hash"]),
                    0,
                    {
                        "B0": record["base_bundle_hash"],
                        "L": record["candidate_bundle_hash"],
                    },
                )
                try:
                    observation = (
                        self.executor or self.runtime.execute_evaluation_task
                    )(task, config, bundles[config.arm.value])
                    verifier = getattr(
                        self.runtime, "verify_evaluation_observation", None
                    )
                    if (
                        not isinstance(observation, RunObservation)
                        or not callable(verifier)
                        or not verifier(observation, config, task)
                    ):
                        raise ValueError(
                            "runtime observation failed trusted evidence verification"
                        )
                    value = self._result_value(observation, task, config)
                    ref = self.store.put_artifact(value).sha256
                    serialized = (
                        json.dumps(
                            json.loads(canonical_json(observation)), sort_keys=True
                        )
                        if isinstance(observation, RunObservation)
                        else None
                    )
                    self._finish(
                        diagnostic_id,
                        cell,
                        "completed",
                        value,
                        ref,
                        "task_failure" if not value["passed"] else None,
                        None,
                        serialized,
                    )
                except Exception as exc:
                    self._finish(
                        diagnostic_id,
                        cell,
                        "failed",
                        {
                            "taskId": task.task_id,
                            "arm": config.arm.value,
                            "seed": config.seed,
                            "protocolHash": frozen.protocol_hash,
                            "bundleHash": config.bundle_hash,
                        },
                        None,
                        "infrastructure_error",
                        str(exc),
                    )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=3, thread_name_prefix="diagnostic"
            ) as pool:
                for future in [pool.submit(execute, c) for c in rows]:
                    future.result()
            with self.store.connect() as conn:
                cancel = conn.execute(
                    "SELECT cancel_requested FROM diagnostics WHERE diagnostic_id=?",
                    (diagnostic_id,),
                ).fetchone()
                failed = conn.execute(
                    "SELECT COUNT(*) FROM diagnostic_cells WHERE diagnostic_id=? AND status='failed'",
                    (diagnostic_id,),
                ).fetchone()[0]
                pending = conn.execute(
                    "SELECT COUNT(*) FROM diagnostic_cells WHERE diagnostic_id=? AND status IN ('queued','running')",
                    (diagnostic_id,),
                ).fetchone()[0]
                state = (
                    "cancelled"
                    if cancel and cancel["cancel_requested"]
                    else (
                        "failed"
                        if failed
                        else "completed" if pending == 0 else "running"
                    )
                )
                conn.execute(
                    "UPDATE diagnostics SET state=?,updated_at=?,error=CASE WHEN ?='failed' THEN 'one or more cells failed infrastructure checks' ELSE error END WHERE diagnostic_id=?",
                    (state, _now(), state, diagnostic_id),
                )
                conn.commit()
            return self.get(diagnostic_id)


__all__ = ["DevelopmentDiagnosticManager"]
