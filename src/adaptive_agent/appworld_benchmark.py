"""Standalone, receipt-backed benchmark runner for the official AppWorld split.

This module deliberately knows nothing about AppWorld task answers.  The
provider/package supplies opaque task identities and the trusted runtime owns
execution, outcome projection, and evidence verification.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from adaptive_agent.evaluation import Arm, BudgetSpec, ModelProvenance, RunObservation, sha256_json


APPWORLD_VERSION = "0.1.3.post1"
DEFAULT_PUBLISHED_COUNT = 20
DEFAULT_SEED = 20260906


class AppWorldPackage(Protocol):
    """Minimal worker-2 package seam, intentionally task-content opaque."""

    def dataset_content_hash(self) -> str: ...
    def task_ids(self, split: str) -> Sequence[str]: ...
    def task(self, task_id: str) -> Any: ...


class AppWorldExecutor(Protocol):
    def __call__(self, task: Any, arm: Arm, seed: int, bundle_hash: str, budget: BudgetSpec) -> RunObservation: ...


class AppWorldEvidenceVerifier(Protocol):
    def __call__(self, observation: RunObservation, task: Any, bundle_hash: str) -> bool: ...


@dataclass(frozen=True)
class AppWorldProtocol:
    source: str
    model_profile: str
    core_planner_hash: str
    dataset_content_hash: str
    seeds: tuple[int, ...] = (0,)
    published_count: int = DEFAULT_PUBLISHED_COUNT
    sampling_seed: int = DEFAULT_SEED
    budget: BudgetSpec = BudgetSpec()
    sampled_task_ids: tuple[str, ...] = ()
    split_by_task_id: tuple[tuple[str, str], ...] = ()
    official_split_counts: tuple[tuple[str, int], ...] = ()
    protocol_hash: str = ""

    @classmethod
    def freeze(cls, package: AppWorldPackage, *, model_profile: str, core_planner_hash: str, dataset_content_hash: str | None = None, published_count: int = DEFAULT_PUBLISHED_COUNT, sampling_seed: int = DEFAULT_SEED, seeds: tuple[int, ...] = (0,), budget: BudgetSpec | None = None) -> "AppWorldProtocol":
        if not model_profile or not core_planner_hash or published_count < 1 or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ValueError("AppWorld count and seeds must be valid before execution")
        content_hash = dataset_content_hash or package.dataset_content_hash()
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError("AppWorld dataset content hash is required")
        split_ids: dict[str, tuple[str, ...]] = {}
        for split in ("train", "dev", "test_normal", "test_challenge"):
            ids = tuple(str(task_id) for task_id in package.task_ids(split))
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate AppWorld task IDs in {split}")
            split_ids[split] = ids
        all_ids = tuple(task_id for split in ("test_normal", "test_challenge") for task_id in split_ids[split])
        if published_count > len(all_ids):
            raise ValueError("published subset exceeds official final task pool")
        rng = random.Random(sampling_seed)
        sampled = tuple(sorted(rng.sample(list(all_ids), published_count)))
        split_by_id = tuple((task_id, split) for split, ids in split_ids.items() for task_id in ids if task_id in sampled)
        payload = {"source": "appworld", "version": APPWORLD_VERSION, "modelProfile": model_profile, "corePlannerHash": core_planner_hash, "datasetContentHash": content_hash, "seeds": list(seeds), "publishedCount": published_count, "samplingSeed": sampling_seed, "budget": budget.to_dict() if budget else BudgetSpec().to_dict(), "officialSplitCounts": {split: len(ids) for split, ids in split_ids.items()}, "sampledTaskIds": list(sampled), "splitByTaskId": split_by_id}
        return cls("appworld", model_profile, core_planner_hash, content_hash, seeds, published_count, sampling_seed, budget or BudgetSpec(), sampled, split_by_id, tuple((split, len(ids)) for split, ids in split_ids.items()), sha256_json(payload))

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "version": APPWORLD_VERSION, "modelProfile": self.model_profile, "corePlannerHash": self.core_planner_hash, "datasetContentHash": self.dataset_content_hash, "seeds": list(self.seeds), "publishedCount": self.published_count, "samplingSeed": self.sampling_seed, "budget": self.budget.to_dict(), "officialSplitCounts": dict(self.official_split_counts), "sampledTaskIds": list(self.sampled_task_ids), "splitByTaskId": dict(self.split_by_task_id), "protocolHash": self.protocol_hash, "scope": "published_subset" if self.published_count == DEFAULT_PUBLISHED_COUNT else "configured_subset"}


@dataclass(frozen=True)
class AppWorldReport:
    protocol: AppWorldProtocol
    arm_summaries: Mapping[str, Mapping[str, float | int]]
    confidence_intervals: tuple[Mapping[str, float | int | str], ...]
    paired_task_count: int
    missing_pairs: int
    provenance_complete: bool
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"benchmark": "appworld", "protocol": self.protocol.to_dict(), "armSummaries": dict(self.arm_summaries), "confidenceIntervals": list(self.confidence_intervals), "pairedTaskCount": self.paired_task_count, "missingPairs": self.missing_pairs, "provenanceComplete": self.provenance_complete, "limitations": list(self.limitations)}


class AppWorldBenchmarkRunner:
    """Durably execute pinned B0/L/A cells without promotion or task output export."""

    def __init__(self, store_dir: str | Path, package: AppWorldPackage, protocol: AppWorldProtocol, execute: AppWorldExecutor, verify: AppWorldEvidenceVerifier) -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.package, self.protocol, self.execute, self.verify = package, protocol, execute, verify
        self.db = self.store_dir / "appworld-benchmark.sqlite3"
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS appworld_plans (benchmark_id TEXT PRIMARY KEY, protocol_hash TEXT NOT NULL, bundle_hashes_json TEXT NOT NULL, task_ids_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("CREATE TABLE IF NOT EXISTS appworld_cells (benchmark_id TEXT NOT NULL, task_id TEXT NOT NULL, split TEXT NOT NULL, arm TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL, observation_json TEXT, error TEXT, PRIMARY KEY(benchmark_id, task_id, arm, seed))")
            conn.commit()

    def _plan(self, benchmark_id: str, bundles: Mapping[Arm | str, str]) -> None:
        normalized = {Arm(arm).value: value for arm, value in bundles.items()}
        if set(normalized) != {arm.value for arm in (Arm.B0, Arm.L, Arm.A)} or any(not isinstance(value, str) or not value for value in normalized.values()):
            raise ValueError("AppWorld requires pinned B0, L, and A bundle hashes")
        payload = (self.protocol.protocol_hash, normalized, self.protocol.sampled_task_ids)
        with sqlite3.connect(self.db) as conn:
            row = conn.execute("SELECT protocol_hash, bundle_hashes_json, task_ids_json FROM appworld_plans WHERE benchmark_id = ?", (benchmark_id,)).fetchone()
            encoded_bundles, encoded_tasks = json.dumps(normalized, sort_keys=True), json.dumps(self.protocol.sampled_task_ids)
            if row is None:
                conn.execute("INSERT INTO appworld_plans(benchmark_id, protocol_hash, bundle_hashes_json, task_ids_json) VALUES (?, ?, ?, ?)", (benchmark_id, self.protocol.protocol_hash, encoded_bundles, encoded_tasks))
            elif tuple(row) != (self.protocol.protocol_hash, encoded_bundles, encoded_tasks):
                raise ValueError("AppWorld benchmark plan is already bound to different protocol, bundles, or tasks")
            conn.commit()

    @staticmethod
    def _decode_observation(payload: str) -> RunObservation:
        value = json.loads(payload)
        value["arm"] = Arm(value["arm"])
        value["model_provenance"] = ModelProvenance(value["model_provenance"])
        from adaptive_agent.evaluation import Partition
        value["partition"] = Partition(value["partition"])
        value["budget"] = BudgetSpec(**{
            "model_tokens": value["budget"].get("model_tokens", value["budget"].get("modelTokens", 0)),
            "tool_calls": value["budget"].get("tool_calls", value["budget"].get("toolCalls", 0)),
            "child_runs": value["budget"].get("child_runs", value["budget"].get("childRuns", 0)),
            "wall_time_seconds": value["budget"].get("wall_time_seconds", value["budget"].get("wallTimeSeconds", 0)),
            "cost_microunits": value["budget"].get("cost_microunits", value["budget"].get("costMicrounits", 0)),
            "currency": value["budget"].get("currency", "USD"),
            "max_child_depth": value["budget"].get("max_child_depth", value["budget"].get("childDepth", 1)),
        })
        return RunObservation(**value)

    def run(self, benchmark_id: str, bundles: Mapping[Arm | str, str]) -> AppWorldReport:
        self._plan(benchmark_id, bundles)
        rows: list[RunObservation] = []
        split_by_id = dict(self.protocol.split_by_task_id)
        with sqlite3.connect(self.db) as conn:
            for task_id in self.protocol.sampled_task_ids:
                task = self.package.task(task_id)
                for seed in self.protocol.seeds:
                    for arm in (Arm.B0, Arm.L, Arm.A):
                        bundle_hash = bundles.get(arm, bundles.get(arm.value))
                        prior = conn.execute("SELECT status, observation_json FROM appworld_cells WHERE benchmark_id=? AND task_id=? AND arm=? AND seed=?", (benchmark_id, task_id, arm.value, seed)).fetchone()
                        if prior and prior[0] == "complete":
                            if prior[1]:
                                observation = self._decode_observation(prior[1])
                                if self.verify(observation, task, bundle_hash):
                                    rows.append(observation)
                                    continue
                            conn.execute("UPDATE appworld_cells SET status='failed', error=? WHERE benchmark_id=? AND task_id=? AND arm=? AND seed=?", ("persisted receipt failed verification", benchmark_id, task_id, arm.value, seed))
                        try:
                            observation = self.execute(task, arm, seed, bundle_hash, self.protocol.budget)
                            if observation.task_id != task_id or observation.arm != arm or observation.seed != seed or observation.model_provenance != ModelProvenance.REAL_MODEL or not self.verify(observation, task, bundle_hash):
                                raise ValueError("observation failed AppWorld identity/provenance verification")
                            conn.execute("INSERT OR REPLACE INTO appworld_cells VALUES (?, ?, ?, ?, ?, 'complete', ?, NULL)", (benchmark_id, task_id, split_by_id[task_id], arm.value, seed, json.dumps(asdict(observation), default=lambda value: value.value if hasattr(value, "value") else value, sort_keys=True)))
                            rows.append(observation)
                        except Exception as exc:
                            conn.execute("INSERT OR REPLACE INTO appworld_cells VALUES (?, ?, ?, ?, ?, 'failed', NULL, ?)", (benchmark_id, task_id, split_by_id[task_id], arm.value, seed, str(exc)))
            conn.commit()
        return self._report(rows)

    def _report(self, rows: Sequence[RunObservation]) -> AppWorldReport:
        summaries: dict[str, Mapping[str, float | int]] = {}
        for arm in (Arm.B0, Arm.L, Arm.A):
            selected = [row for row in rows if row.arm == arm]
            summaries[arm.value] = {"accuracy": sum(row.passed for row in selected) / len(selected) if selected else 0.0, "reliability": sum(row.reliable for row in selected) / len(selected) if selected else 0.0, "inputTokens": 0, "outputTokens": 0, "costMicrounits": sum(row.cost_microunits for row in selected), "latencySeconds": sum(row.latency_seconds for row in selected), "count": len(selected)}
        paired = sum(1 for task_id in self.protocol.sampled_task_ids if all(any(row.task_id == task_id and row.arm.value == arm.value for row in rows) for arm in (Arm.B0, Arm.L, Arm.A)))
        expected = len(self.protocol.sampled_task_ids) * len(self.protocol.seeds) * 3
        return AppWorldReport(self.protocol, summaries, (), paired // len(self.protocol.seeds) if self.protocol.seeds else 0, expected - len(rows), bool(rows) and len(rows) == expected, ("Published AppWorld subset; not the full benchmark.", "Taskwise outcomes, traces, and ground truth are intentionally omitted."))


__all__ = ["APPWORLD_VERSION", "AppWorldBenchmarkRunner", "AppWorldExecutor", "AppWorldPackage", "AppWorldProtocol", "AppWorldReport", "DEFAULT_PUBLISHED_COUNT", "DEFAULT_SEED"]
