"""Resumable protocol driver for trusted runtime evaluation.

The driver owns orchestration and durable task status only.  Model planning and
tool execution are injected through ``execute_evaluation_task`` from the trusted
runtime; there is no scripted or synthetic solver here.
"""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Callable, Mapping

from adaptive_agent.evaluation import (
    Arm,
    BudgetSpec,
    canonical_json,
    EnvironmentPackage,
    EvaluationError,
    EvaluationProtocol,
    FrozenProtocol,
    BudgetSpec,
    ModelProvenance,
    Partition,
    Provenance,
    RunObservation,
    TaskInput,
    sha256_json,
)
from adaptive_agent.evaluation_store import SQLiteAllocationStore, SQLiteRunEvidenceStore
from adaptive_agent.store import Store


@dataclass(frozen=True)
class FrozenExecutionConfig:
    protocol: FrozenProtocol
    arm: Arm
    seed: int


TrustedTaskExecutor = Callable[[TaskInput, FrozenExecutionConfig, object], RunObservation]


@dataclass(frozen=True)
class BenchmarkTaskStatus:
    task_id: str
    environment_id: str
    partition: Partition
    arm: Arm
    seed: int
    status: str
    error: str | None = None
    observation: RunObservation | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    benchmark_id: str
    partition: Partition
    statuses: tuple[BenchmarkTaskStatus, ...]
    expected_count: int = 0

    @property
    def complete(self) -> bool:
        return len(self.statuses) == self.expected_count and self.expected_count > 0 and all(item.status == "complete" for item in self.statuses)

    @property
    def failed(self) -> bool:
        return any(item.status == "failed" for item in self.statuses)


class ResumableEvaluationDriver:
    """Execute exactly the frozen task panel and resume persisted work."""

    def __init__(self, store: Store, protocol: EvaluationProtocol, packages: Mapping[str, EnvironmentPackage], execute_evaluation_task: TrustedTaskExecutor, bundle: object, allocation_store: SQLiteAllocationStore | None = None, evidence_store: SQLiteRunEvidenceStore | None = None) -> None:
        self.store = store
        self.protocol = protocol
        self.packages = dict(packages)
        self.execute_evaluation_task = execute_evaluation_task
        self.bundle = bundle
        self.allocation_store = allocation_store or SQLiteAllocationStore(store)
        self.evidence_store = evidence_store or SQLiteRunEvidenceStore(store)
        self.owner_id = secrets.token_urlsafe(12)
        with store.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS benchmark_task_runs (benchmark_id TEXT NOT NULL, task_id TEXT NOT NULL, environment_id TEXT NOT NULL, partition TEXT NOT NULL, arm TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL, error TEXT, observation_json TEXT, owner_id TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(benchmark_id, task_id, arm, seed))")
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(benchmark_task_runs)").fetchall()}
            if "owner_id" not in columns:
                conn.execute("ALTER TABLE benchmark_task_runs ADD COLUMN owner_id TEXT")
            conn.execute("CREATE TABLE IF NOT EXISTS benchmark_plans (benchmark_id TEXT NOT NULL, partition TEXT NOT NULL, fingerprint TEXT NOT NULL, panel_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(benchmark_id, partition))")
            conn.commit()

    def run(self, benchmark_id: str, partition: Partition, *, base_hash: str = "", candidate_hash: str = "") -> BenchmarkSummary:
        frozen = self.protocol.start_candidate_generation()
        partition = Partition(partition)
        if partition is not Partition.DEVELOPMENT and not self._development_smoke_complete(benchmark_id):
            raise EvaluationError("development smoke with trusted evidence is required before held-out panels")
        tasks_by_env = self._tasks_for_partition(benchmark_id, partition, base_hash, candidate_hash, frozen)
        arms = (Arm.B0, Arm.L) if partition is not Partition.FINAL else (Arm.B0, Arm.L, Arm.A)
        statuses: list[BenchmarkTaskStatus] = []
        for environment_id, tasks in tasks_by_env.items():
            package = self.packages[environment_id]
            for task in tasks:
                for seed in self.protocol.seeds:
                    for arm in arms:
                        prior = self._load(benchmark_id, task.task_id, arm, seed)
                        if prior is not None and prior.status == "complete":
                            if prior.observation is not None and self.evidence_store.verify(prior.observation, frozen, package):
                                statuses.append(prior)
                                continue
                            self._save(benchmark_id, task, arm, seed, "failed", "persisted evidence no longer verifies", None)
                        if not self._claim(benchmark_id, task, arm, seed):
                            existing = self._load(benchmark_id, task.task_id, arm, seed)
                            if existing is not None:
                                statuses.append(existing)
                            continue
                        try:
                            observation = self.execute_evaluation_task(task, FrozenExecutionConfig(frozen, arm, seed), self.bundle)
                            self._validate_observation(observation, task, environment_id, partition, seed, arm)
                            if not self.evidence_store.verify(observation, frozen, package):
                                raise EvaluationError("runtime observation lacks trusted persisted evidence")
                            self._save(benchmark_id, task, arm, seed, "complete", None, observation)
                            statuses.append(BenchmarkTaskStatus(task.task_id, environment_id, partition, arm, seed, "complete", observation=observation))
                        except Exception as exc:
                            self._save(benchmark_id, task, arm, seed, "failed", str(exc), None)
                            statuses.append(BenchmarkTaskStatus(task.task_id, environment_id, partition, arm, seed, "failed", error=str(exc)))
        expected_count = sum(len(tasks) for tasks in tasks_by_env.values()) * len(self.protocol.seeds) * len(arms)
        return BenchmarkSummary(benchmark_id, partition, tuple(statuses), expected_count)

    def _development_smoke_complete(self, benchmark_id: str) -> bool:
        with self.store.connect() as conn:
            rows = conn.execute("SELECT task_id, environment_id, arm, seed FROM benchmark_task_runs WHERE benchmark_id = ? AND partition = 'development' AND status = 'complete' AND observation_json IS NOT NULL", (benchmark_id,)).fetchall()
        for row in rows:
            status = self._load(benchmark_id, row["task_id"], Arm(row["arm"]), int(row["seed"]))
            if status and status.observation and self.evidence_store.verify(status.observation, self.protocol.start_candidate_generation(), self.packages[row["environment_id"]]):
                return True
        return False

    def _tasks_for_partition(self, benchmark_id: str, partition: Partition, base_hash: str, candidate_hash: str, frozen: FrozenProtocol) -> dict[str, tuple[TaskInput, ...]]:
        bundle_value = self.bundle.to_dict() if hasattr(self.bundle, "to_dict") else (self.bundle if isinstance(self.bundle, Mapping) else getattr(self.bundle, "bundle_id", self.bundle.__class__.__qualname__))
        plan_fingerprint = sha256_json({"protocol": frozen.protocol_hash, "partition": partition.value, "base": base_hash, "candidate": candidate_hash, "bundle": sha256_json(bundle_value)})
        with self.store.connect() as conn:
            plan = conn.execute("SELECT * FROM benchmark_plans WHERE benchmark_id = ? AND partition = ?", (benchmark_id, partition.value)).fetchone()
        if plan is not None:
            if plan["fingerprint"] != plan_fingerprint or plan["partition"] != partition.value:
                raise EvaluationError("benchmark id is already bound to different frozen inputs")
            task_ids = json.loads(plan["panel_json"])
            return {name: tuple(task for task in self.packages[name].tasks_for_partition(partition) if task.task_id in task_ids) for name in (self.protocol.known_environments if partition is not Partition.FINAL else (*self.protocol.known_environments, self.protocol.sealed_environment))}
        if partition is Partition.VALIDATION:
            if not base_hash or not candidate_hash:
                raise EvaluationError("validation requires base and candidate hashes")
            panels = [tuple(task.task_id for name in self.protocol.known_environments for task in self.packages[name].tasks_for_partition(partition)[index * self.protocol.tasks_per_environment:(index + 1) * self.protocol.tasks_per_environment]) for index in range(self.protocol.validation_candidate_limit)]
            index = self.allocation_store.reserve_next(base_hash, f"{base_hash}:{candidate_hash}", panels, self.protocol.validation_candidate_limit)
            if index is None:
                raise EvaluationError("validation panel unavailable or already reserved")
            tasks_by_env = {name: self.packages[name].tasks_for_partition(partition)[index * self.protocol.tasks_per_environment:(index + 1) * self.protocol.tasks_per_environment] for name in self.protocol.known_environments}
        else:
            names = self.protocol.known_environments if partition is not Partition.FINAL else (*self.protocol.known_environments, self.protocol.sealed_environment)
            tasks_by_env = {name: self.packages[name].tasks_for_partition(partition) for name in names}
        panel = [task.task_id for tasks in tasks_by_env.values() for task in tasks]
        with self.store.connect() as conn:
            conn.execute("INSERT INTO benchmark_plans(benchmark_id, partition, fingerprint, panel_json, created_at) VALUES (?, ?, ?, ?, datetime('now'))", (benchmark_id, partition.value, plan_fingerprint, json.dumps(panel, sort_keys=True)))
            conn.commit()
        return tasks_by_env

    def _claim(self, benchmark_id: str, task: TaskInput, arm: Arm, seed: int) -> bool:
        with self.store.connect() as conn:
            cursor = conn.execute("INSERT INTO benchmark_task_runs(benchmark_id, task_id, environment_id, partition, arm, seed, status, error, observation_json, owner_id, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'running', NULL, NULL, ?, datetime('now')) ON CONFLICT(benchmark_id, task_id, arm, seed) DO UPDATE SET status='running', error=NULL, owner_id=?, updated_at=datetime('now') WHERE benchmark_task_runs.status IN ('failed') OR (benchmark_task_runs.status = 'running' AND benchmark_task_runs.owner_id != ?)", (benchmark_id, task.task_id, task.environment_ref.id, task.partition.value, arm.value, seed, self.owner_id, self.owner_id, self.owner_id))
            conn.commit()
            return cursor.rowcount == 1

    def _load(self, benchmark_id: str, task_id: str, arm: Arm, seed: int) -> BenchmarkTaskStatus | None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM benchmark_task_runs WHERE benchmark_id = ? AND task_id = ? AND arm = ? AND seed = ?", (benchmark_id, task_id, arm.value, seed)).fetchone()
        if not row:
            return None
        observation = None
        if row["observation_json"]:
            payload = json.loads(row["observation_json"])
            payload["partition"] = Partition(payload["partition"])
            payload["arm"] = Arm(payload["arm"])
            payload["provenance"] = Provenance(payload.get("provenance", "deterministic_simulation"))
            payload["model_provenance"] = ModelProvenance(payload.get("model_provenance", "synthetic_model"))
            payload["budget"] = BudgetSpec(**payload["budget"])
            observation = RunObservation(**payload)
        return BenchmarkTaskStatus(row["task_id"], row["environment_id"], Partition(row["partition"]), Arm(row["arm"]), int(row["seed"]), row["status"], row["error"], observation)

    def _save(self, benchmark_id: str, task: TaskInput, arm: Arm, seed: int, status: str, error: str | None, observation: RunObservation | None) -> None:
        observation_json = json.dumps(json.loads(canonical_json(observation)), sort_keys=True) if observation else None
        with self.store.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO benchmark_task_runs(benchmark_id, task_id, environment_id, partition, arm, seed, status, error, observation_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))", (benchmark_id, task.task_id, task.environment_ref.id, task.partition.value, arm.value, seed, status, error, observation_json))
            conn.commit()

    @staticmethod
    def _validate_observation(observation: RunObservation, task: TaskInput, environment_id: str, partition: Partition, seed: int, arm: Arm) -> None:
        if not isinstance(observation, RunObservation):
            raise EvaluationError("trusted executor must return RunObservation")
        if (observation.task_id, observation.environment_id, observation.partition, observation.seed, observation.arm) != (task.task_id, environment_id, partition, seed, arm):
            raise EvaluationError("trusted observation identity does not match requested task")
        if observation.model_provenance is not ModelProvenance.REAL_MODEL:
            raise EvaluationError("synthetic observation is not valid runtime evidence")


__all__ = ["BenchmarkSummary", "BenchmarkTaskStatus", "FrozenExecutionConfig", "ResumableEvaluationDriver", "TrustedTaskExecutor"]
