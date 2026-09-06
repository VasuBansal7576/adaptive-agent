"""Resumable protocol driver for trusted runtime evaluation.

The driver owns orchestration and durable task status only.  Model planning and
tool execution are injected through ``execute_evaluation_task`` from the trusted
runtime; there is no scripted or synthetic solver here.
"""
from __future__ import annotations

import json
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Mapping

import fcntl

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
    bundle_hash: str = ""
    # Distinct retry identity while preserving compatibility with legacy
    # four-field callers.
    attempt: int = 0
    arm_bundles: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.attempt, bool) or not isinstance(self.attempt, int) or self.attempt < 0:
            raise EvaluationError("evaluation attempt must be a non-negative integer")


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

    def __init__(self, store: Store, protocol: EvaluationProtocol, packages: Mapping[str, EnvironmentPackage], execute_evaluation_task: TrustedTaskExecutor, bundle: object, allocation_store: SQLiteAllocationStore | None = None, evidence_store: SQLiteRunEvidenceStore | None = None, arm_bundles: Mapping[Arm | str, object] | None = None, owner_id: str | None = None) -> None:
        self.store = store
        self.protocol = protocol
        self.packages = dict(packages)
        self.execute_evaluation_task = execute_evaluation_task
        self.bundle = bundle
        self.arm_bundles = dict(arm_bundles or {})
        self.arm_bundles.setdefault(Arm.B0, bundle)
        self.allocation_store = allocation_store or SQLiteAllocationStore(store)
        self.evidence_store = evidence_store or SQLiteRunEvidenceStore(store)
        self.owner_id = owner_id or secrets.token_urlsafe(12)
        self._benchmark_lock_held = False
        with store.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS benchmark_task_attempts (attempt_id TEXT PRIMARY KEY, benchmark_id TEXT NOT NULL, task_id TEXT NOT NULL, environment_id TEXT NOT NULL, partition TEXT NOT NULL, arm TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL, error TEXT, observation_json TEXT, owner_id TEXT, updated_at TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS benchmark_plans (benchmark_id TEXT NOT NULL, partition TEXT NOT NULL, fingerprint TEXT NOT NULL, panel_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(benchmark_id, partition))")
            conn.commit()

    @contextmanager
    def _benchmark_lock(self, benchmark_id: str):
        lock_dir = self.store.base_dir / "benchmark-locks"
        lock_dir.mkdir(exist_ok=True)
        with (lock_dir / f"{sha256_json(benchmark_id)[:32]}.lock").open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self._benchmark_lock_held = True
            try:
                yield
            finally:
                self._benchmark_lock_held = False
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def run(self, benchmark_id: str, partition: Partition, *, base_hash: str = "", candidate_hash: str = "") -> BenchmarkSummary:
        with self._benchmark_lock(benchmark_id):
            return self._run_locked(benchmark_id, partition, base_hash=base_hash, candidate_hash=candidate_hash)

    def run_development_smoke(self, benchmark_id: str) -> BenchmarkSummary:
        """Record one trusted development receipt for held-out authorization."""
        smoke_id = f"{benchmark_id}:smoke"
        with self._benchmark_lock(benchmark_id):
            return self._run_locked(smoke_id, Partition.DEVELOPMENT, smoke=True)

    def _run_locked(self, benchmark_id: str, partition: Partition, *, base_hash: str = "", candidate_hash: str = "", smoke: bool = False) -> BenchmarkSummary:
        frozen = self.protocol.start_candidate_generation()
        partition = Partition(partition)
        if partition is not Partition.DEVELOPMENT and not self._development_smoke_complete(benchmark_id):
            raise EvaluationError("development smoke with trusted evidence is required before held-out panels")
        arms = (Arm.B0,) if partition is Partition.DEVELOPMENT else ((Arm.B0, Arm.L) if partition is not Partition.FINAL else (Arm.B0, Arm.L, Arm.A))
        missing_arms = [arm.value for arm in arms if arm not in self.arm_bundles and arm.value not in self.arm_bundles]
        if missing_arms:
            raise EvaluationError(f"missing expected arm bundles before execution: {', '.join(missing_arms)}")
        tasks_by_env = self._tasks_for_partition(benchmark_id, partition, base_hash, candidate_hash, frozen, smoke=smoke)
        if smoke:
            first_environment = next(iter(tasks_by_env))
            tasks_by_env = {first_environment: tasks_by_env[first_environment][:1]}
        seeds = (self.protocol.seeds[0],) if partition is Partition.DEVELOPMENT else self.protocol.seeds
        statuses: list[BenchmarkTaskStatus] = []
        for environment_id, tasks in tasks_by_env.items():
            package = self.packages[environment_id]
            for task in tasks:
                for seed in seeds:
                    for arm in arms:
                        selected_bundle = self.arm_bundles.get(arm, self.arm_bundles.get(arm.value))
                        bundle_hash = self._bundle_hash(selected_bundle)
                        prior = self._load(benchmark_id, task.task_id, arm, seed)
                        if prior is not None and prior.status == "complete":
                            if prior.observation is not None:
                                try:
                                    self._validate_observation(prior.observation, task, environment_id, partition, seed, arm, bundle_hash)
                                except EvaluationError:
                                    pass
                                else:
                                    if self.evidence_store.verify(prior.observation, frozen, package):
                                        statuses.append(prior)
                                        continue
                            self._save(benchmark_id, task, arm, seed, "failed", "persisted evidence no longer verifies", None)
                        if not self._claim(benchmark_id, task, arm, seed):
                            existing = self._load(benchmark_id, task.task_id, arm, seed)
                            if existing is not None:
                                statuses.append(existing)
                            continue
                        try:
                            attempt = self._next_attempt(benchmark_id, task, arm, seed)
                            observation = self.execute_evaluation_task(task, FrozenExecutionConfig(frozen, arm, seed, bundle_hash, attempt), selected_bundle)
                            self._validate_observation(observation, task, environment_id, partition, seed, arm, bundle_hash)
                            if not self.evidence_store.verify(observation, frozen, package):
                                raise EvaluationError("runtime observation lacks trusted persisted evidence")
                            self._save(benchmark_id, task, arm, seed, "complete", None, observation)
                            statuses.append(BenchmarkTaskStatus(task.task_id, environment_id, partition, arm, seed, "complete", observation=observation))
                        except Exception as exc:
                            self._save(benchmark_id, task, arm, seed, "failed", str(exc), None)
                            statuses.append(BenchmarkTaskStatus(task.task_id, environment_id, partition, arm, seed, "failed", error=str(exc)))
                        except BaseException as exc:
                            # A process interruption leaves the cell outcome
                            # unknown. Persist that attempt before propagating
                            # the interruption so a same-owner retry receives a
                            # fresh attempt number and history remains auditable.
                            self._save(benchmark_id, task, arm, seed, "uncertain", str(exc), None)
                            raise
        expected_count = sum(len(tasks) for tasks in tasks_by_env.values()) * len(seeds) * len(arms)
        return BenchmarkSummary(benchmark_id, partition, tuple(statuses), expected_count)

    def _next_attempt(self, benchmark_id: str, task: TaskInput, arm: Arm, seed: int) -> int:
        """Return the next immutable attempt number for one benchmark cell."""
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM benchmark_task_attempts WHERE benchmark_id = ? AND task_id = ? AND environment_id = ? AND partition = ? AND arm = ? AND seed = ?",
                (benchmark_id, task.task_id, task.environment_ref.id, task.partition.value, arm.value, seed),
            ).fetchone()
        return int(row["count"] if row is not None else 0)

    def _development_smoke_complete(self, benchmark_id: str) -> bool:
        rows = self.store.list_task_runs(partition="development", status="complete")
        for row in rows:
            if row.get("benchmark_id") not in {benchmark_id, f"{benchmark_id}:smoke"} or not row.get("arm") or row.get("seed") is None:
                continue
            status = self._load(str(row["benchmark_id"]), row["task_id"], Arm(row["arm"]), int(row["seed"]))
            if status and status.observation and self.evidence_store.verify(status.observation, self.protocol.start_candidate_generation(), self.packages[row["environment_id"]]):
                return True
        return False

    def _tasks_for_partition(self, benchmark_id: str, partition: Partition, base_hash: str, candidate_hash: str, frozen: FrozenProtocol, *, smoke: bool = False) -> dict[str, tuple[TaskInput, ...]]:
        if hasattr(self.bundle, "model_dump"):
            bundle_value = self.bundle.model_dump(mode="json")
        elif hasattr(self.bundle, "to_dict"):
            bundle_value = self.bundle.to_dict()
        elif isinstance(self.bundle, Mapping):
            bundle_value = dict(self.bundle)
        else:
            bundle_value = {"type": f"{type(self.bundle).__module__}.{type(self.bundle).__qualname__}"}
        arm_bundle_values = {
            str(key.value if isinstance(key, Arm) else key): self._bundle_hash(value)
            for key, value in self.arm_bundles.items()
        }
        plan_fingerprint = sha256_json({"protocol": frozen.protocol_hash, "partition": partition.value, "base": base_hash, "candidate": candidate_hash, "bundle": sha256_json(bundle_value), "armBundles": arm_bundle_values, "smoke": smoke})
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
            allocation_id = f"{benchmark_id}:{partition.value}:{base_hash}:{candidate_hash}:{plan_fingerprint}"
            index = self.allocation_store.reserve_next(base_hash, allocation_id, panels, self.protocol.validation_candidate_limit)
            if index is None:
                allocation = self.allocation_store.get(allocation_id)
                if allocation is None:
                    raise EvaluationError("validation panel unavailable or already reserved")
                task_ids = set(allocation["task_ids"])
                expected = {task_id for panel in panels for task_id in panel}
                if not task_ids or not task_ids.issubset(expected):
                    raise EvaluationError("existing validation allocation does not match immutable benchmark inputs")
                tasks_by_env = {name: tuple(task for task in self.packages[name].tasks_for_partition(partition) if task.task_id in task_ids) for name in self.protocol.known_environments}
                index = -1
            else:
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
        task_run_id = self._task_run_id(benchmark_id, task.task_id, arm, seed)
        claimed, row = self.store.claim_benchmark_task_run(task_run_id, benchmark_id=benchmark_id, environment_id=task.environment_ref.id, task_id=task.task_id, partition=task.partition.value, arm=arm.value, seed=seed, owner_id=self.owner_id)
        if claimed:
            return True
        if row.get("owner_id") == self.owner_id:
            if row.get("status") in {"failed", "uncertain"}:
                return self.store.release_task_run(task_run_id, self.owner_id, "running", json.dumps({}))
            if row.get("status") == "running":
                # The prior process may have crashed after claiming the cell.
                # The owner can safely resume it; foreign owners cannot.
                return True
        return False

    @staticmethod
    def _task_run_id(benchmark_id: str, task_id: str, arm: Arm, seed: int) -> str:
        return f"{benchmark_id}:{task_id}:{arm.value}:{seed}"

    def _load(self, benchmark_id: str, task_id: str, arm: Arm, seed: int) -> BenchmarkTaskStatus | None:
        row = self.store.get_task_run(self._task_run_id(benchmark_id, task_id, arm, seed))
        if not row:
            return None
        observation = None
        state = json.loads(row.get("state_json") or "{}")
        if isinstance(state, dict) and state.get("observation"):
            payload = state["observation"]
            payload["partition"] = Partition(payload["partition"])
            payload["arm"] = Arm(payload["arm"])
            payload["provenance"] = Provenance(payload.get("provenance", "deterministic_simulation"))
            payload["model_provenance"] = ModelProvenance(payload.get("model_provenance", "synthetic_model"))
            payload["budget"] = BudgetSpec(**payload["budget"])
            observation = RunObservation(**payload)
        return BenchmarkTaskStatus(row["task_id"], row["environment_id"], Partition(row["partition"]), Arm(row["arm"]), int(row["seed"]), row["status"], state.get("error") if isinstance(state, dict) else None, observation)

    def _save(self, benchmark_id: str, task: TaskInput, arm: Arm, seed: int, status: str, error: str | None, observation: RunObservation | None) -> None:
        observation_json = json.dumps(json.loads(canonical_json(observation)), sort_keys=True) if observation else None
        state_json = json.dumps({"observation": json.loads(observation_json) if observation_json else None, "error": error}, sort_keys=True)
        with self.store.connect() as conn:
            conn.execute("INSERT INTO benchmark_task_attempts(attempt_id, benchmark_id, task_id, environment_id, partition, arm, seed, status, error, observation_json, owner_id, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))", (secrets.token_urlsafe(18), benchmark_id, task.task_id, task.environment_ref.id, task.partition.value, arm.value, seed, status, error, observation_json, self.owner_id))
            conn.commit()
        if not self.store.release_task_run(self._task_run_id(benchmark_id, task.task_id, arm, seed), self.owner_id, status, state_json):
            raise EvaluationError("benchmark task run ownership lost before persistence")

    @staticmethod
    def _bundle_hash(bundle: object) -> str:
        bundle_hash = getattr(bundle, "content_hash", None)
        if not isinstance(bundle_hash, str) or not bundle_hash:
            raise EvaluationError("selected arm bundle has no content hash")
        if hasattr(bundle, "model_dump"):
            payload = bundle.model_dump(mode="json", by_alias=True, exclude={"content_hash"})
            if sha256_json(payload) != bundle_hash:
                raise EvaluationError("selected arm bundle content hash is invalid")
        return bundle_hash

    @staticmethod
    def _validate_observation(observation: RunObservation, task: TaskInput, environment_id: str, partition: Partition, seed: int, arm: Arm, bundle_hash: str) -> None:
        if not isinstance(observation, RunObservation):
            raise EvaluationError("trusted executor must return RunObservation")
        if (observation.task_id, observation.environment_id, observation.partition, observation.seed, observation.arm) != (task.task_id, environment_id, partition, seed, arm):
            raise EvaluationError("trusted observation identity does not match requested task")
        if observation.model_provenance is not ModelProvenance.REAL_MODEL:
            raise EvaluationError("synthetic observation is not valid runtime evidence")
        if not bundle_hash or observation.bundle_hash != bundle_hash:
            raise EvaluationError("observation bundle hash does not match the requested arm bundle")


__all__ = ["BenchmarkSummary", "BenchmarkTaskStatus", "FrozenExecutionConfig", "ResumableEvaluationDriver", "TrustedTaskExecutor"]
