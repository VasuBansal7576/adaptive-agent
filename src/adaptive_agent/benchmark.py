"""Resumable protocol driver for trusted runtime evaluation.

The driver owns orchestration and durable task status only.  Model planning and
tool execution are injected through ``execute_evaluation_task`` from the trusted
runtime; there is no scripted or synthetic solver here.
"""
from __future__ import annotations

import json
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
    ModelProvenance,
    Partition,
    RunObservation,
    TaskInput,
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

    @property
    def complete(self) -> bool:
        return bool(self.statuses) and all(item.status == "complete" for item in self.statuses)

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
        with store._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS benchmark_task_runs (benchmark_id TEXT NOT NULL, task_id TEXT NOT NULL, environment_id TEXT NOT NULL, partition TEXT NOT NULL, arm TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL, error TEXT, observation_json TEXT, updated_at TEXT NOT NULL, PRIMARY KEY(benchmark_id, task_id, arm, seed))")
            conn.commit()

    def run(self, benchmark_id: str, partition: Partition, *, base_hash: str = "", candidate_hash: str = "") -> BenchmarkSummary:
        frozen = self.protocol.start_candidate_generation()
        partition = Partition(partition)
        if partition is not Partition.DEVELOPMENT and not self._development_smoke_complete(benchmark_id):
            raise EvaluationError("development smoke with trusted evidence is required before held-out panels")
        tasks_by_env = self._tasks_for_partition(benchmark_id, partition, base_hash, candidate_hash)
        arms = (Arm.B0, Arm.L) if partition is not Partition.FINAL else (Arm.B0, Arm.L, Arm.A)
        statuses: list[BenchmarkTaskStatus] = []
        for environment_id, tasks in tasks_by_env.items():
            package = self.packages[environment_id]
            for task in tasks:
                for seed in self.protocol.seeds:
                    for arm in arms:
                        prior = self._load(benchmark_id, task.task_id, arm, seed)
                        if prior is not None and prior.status == "complete":
                            statuses.append(prior)
                            continue
                        self._save(benchmark_id, task, arm, seed, "running", None, None)
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
        return BenchmarkSummary(benchmark_id, partition, tuple(statuses))

    def _development_smoke_complete(self, benchmark_id: str) -> bool:
        with self.store._connect() as conn:
            rows = conn.execute("SELECT task_id, environment_id, arm, seed FROM benchmark_task_runs WHERE benchmark_id = ? AND partition = 'development' AND status = 'complete' AND observation_json IS NOT NULL", (benchmark_id,)).fetchall()
        for row in rows:
            status = self._load(benchmark_id, row["task_id"], Arm(row["arm"]), int(row["seed"]))
            if status and status.observation and self.evidence_store.verify(status.observation, self.protocol.start_candidate_generation(), self.packages[row["environment_id"]]):
                return True
        return False

    def _tasks_for_partition(self, benchmark_id: str, partition: Partition, base_hash: str, candidate_hash: str) -> dict[str, tuple[TaskInput, ...]]:
        if partition is Partition.VALIDATION:
            if not base_hash or not candidate_hash:
                raise EvaluationError("validation requires base and candidate hashes")
            with self.store._connect() as conn:
                existing = [row["task_id"] for row in conn.execute("SELECT task_id FROM benchmark_task_runs WHERE benchmark_id = ? AND partition = ?", (benchmark_id, partition.value)).fetchall()]
            if existing:
                return {name: tuple(task for task in self.packages[name].tasks_for_partition(partition) if task.task_id in existing) for name in self.protocol.known_environments}
            panels = [tuple(task.task_id for name in self.protocol.known_environments for task in self.packages[name].tasks_for_partition(partition)[index * self.protocol.tasks_per_environment:(index + 1) * self.protocol.tasks_per_environment]) for index in range(self.protocol.validation_candidate_limit)]
            index = self.allocation_store.reserve_next(base_hash, f"{base_hash}:{candidate_hash}", panels, self.protocol.validation_candidate_limit)
            if index is None:
                raise EvaluationError("validation panel unavailable or already reserved")
            return {name: self.packages[name].tasks_for_partition(partition)[index * self.protocol.tasks_per_environment:(index + 1) * self.protocol.tasks_per_environment] for name in self.protocol.known_environments}
        names = self.protocol.known_environments if partition is not Partition.FINAL else (*self.protocol.known_environments, self.protocol.sealed_environment)
        return {name: self.packages[name].tasks_for_partition(partition) for name in names}

    def _load(self, benchmark_id: str, task_id: str, arm: Arm, seed: int) -> BenchmarkTaskStatus | None:
        with self.store._connect() as conn:
            row = conn.execute("SELECT * FROM benchmark_task_runs WHERE benchmark_id = ? AND task_id = ? AND arm = ? AND seed = ?", (benchmark_id, task_id, arm.value, seed)).fetchone()
        if not row:
            return None
        observation = None
        if row["observation_json"]:
            payload = json.loads(row["observation_json"])
            payload["partition"] = Partition(payload["partition"])
            payload["arm"] = Arm(payload["arm"])
            payload["provenance"] = payload.get("provenance", "deterministic_simulation")
            payload["model_provenance"] = payload.get("model_provenance", "synthetic_model")
            payload["budget"] = BudgetSpec(**payload["budget"])
            observation = RunObservation(**payload)
        return BenchmarkTaskStatus(row["task_id"], row["environment_id"], Partition(row["partition"]), Arm(row["arm"]), int(row["seed"]), row["status"], row["error"], observation)

    def _save(self, benchmark_id: str, task: TaskInput, arm: Arm, seed: int, status: str, error: str | None, observation: RunObservation | None) -> None:
        observation_json = json.dumps(json.loads(canonical_json(observation)), sort_keys=True) if observation else None
        with self.store._connect() as conn:
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
