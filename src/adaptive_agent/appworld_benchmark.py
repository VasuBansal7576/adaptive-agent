"""Durable, aggregate-only AppWorld benchmark orchestration."""
from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from adaptive_agent.evaluation import Arm, BudgetSpec, ModelProvenance, RunObservation, canonical_json, clustered_paired_bootstrap, sha256_json

APPWORLD_VERSION = "0.1.3.post1"
DEFAULT_PUBLISHED_COUNT = 20
DEFAULT_SEED = 20260906
FINAL_SPLITS = ("test_normal", "test_challenge")


class AppWorldPackage(Protocol):
    catalog: Any
    def provider_factory(self, task: Any, run_id: str, seed: int = 0) -> Any: ...
    def evaluate_provider(self, provider: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class AppWorldCellResult:
    observation: RunObservation
    usage: Mapping[str, int] | None = None


class AppWorldRuntime(Protocol):
    def run_appworld_cell(self, *, package: AppWorldPackage, task: Any, arm: Arm, seed: int, bundle_hash: str, budget: BudgetSpec, run_id: str) -> AppWorldCellResult: ...
    def recover_appworld_cell(self, run_id: str) -> AppWorldCellResult | None: ...
    def verify_appworld_cell(self, result: AppWorldCellResult, *, package: AppWorldPackage, task: Any, arm: Arm, seed: int, bundle_hash: str) -> bool: ...


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
        if not model_profile or not core_planner_hash or published_count < 1 or not seeds or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
            raise ValueError("invalid AppWorld protocol pins")
        catalog = package.catalog
        content_hash = dataset_content_hash or catalog.dataset_hash()
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError("AppWorld dataset hash is required")
        split_ids = {split: tuple(str(x) for x in catalog.split_ids(split)) for split in ("train", "dev", *FINAL_SPLITS)}
        if any(len(ids) != len(set(ids)) for ids in split_ids.values()):
            raise ValueError("duplicate AppWorld task IDs")
        final_ids = [x for split in FINAL_SPLITS for x in split_ids[split]]
        if published_count > len(final_ids):
            raise ValueError("published subset exceeds official final task pool")
        sampled = tuple(sorted(random.Random(sampling_seed).sample(final_ids, published_count)))
        split_by_id = tuple((task_id, split) for split, ids in split_ids.items() for task_id in ids if task_id in sampled)
        selected_budget = budget or BudgetSpec()
        payload = {"source": "appworld", "version": APPWORLD_VERSION, "modelProfile": model_profile, "corePlannerHash": core_planner_hash, "datasetContentHash": content_hash, "seeds": list(seeds), "publishedCount": published_count, "samplingSeed": sampling_seed, "budget": selected_budget.to_dict(), "officialSplitCounts": {k: len(v) for k, v in split_ids.items()}, "sampledTaskIds": list(sampled), "splitByTaskId": list(split_by_id)}
        return cls("appworld", model_profile, core_planner_hash, content_hash, tuple(seeds), published_count, sampling_seed, selected_budget, sampled, split_by_id, tuple((k, len(v)) for k, v in split_ids.items()), sha256_json(payload))

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "version": APPWORLD_VERSION, "modelProfile": self.model_profile, "corePlannerHash": self.core_planner_hash, "datasetContentHash": self.dataset_content_hash, "seeds": list(self.seeds), "publishedCount": self.published_count, "samplingSeed": self.sampling_seed, "budget": self.budget.to_dict(), "officialSplitCounts": dict(self.official_split_counts), "sampledTaskIds": list(self.sampled_task_ids), "splitByTaskId": dict(self.split_by_task_id), "protocolHash": self.protocol_hash, "scope": "published_subset" if self.published_count == DEFAULT_PUBLISHED_COUNT else "configured_subset"}


@dataclass(frozen=True)
class AppWorldReport:
    protocol: AppWorldProtocol
    arm_summaries: Mapping[str, Mapping[str, Any]]
    confidence_intervals: tuple[Mapping[str, Any], ...]
    paired_task_count: int
    missing_pairs: int
    provenance_complete: bool
    limitations: tuple[str, ...]
    def to_dict(self) -> dict[str, Any]:
        return {"benchmark": "appworld", "protocol": self.protocol.to_dict(), "armSummaries": dict(self.arm_summaries), "confidenceIntervals": list(self.confidence_intervals), "pairedTaskCount": self.paired_task_count, "missingPairs": self.missing_pairs, "provenanceComplete": self.provenance_complete, "limitations": list(self.limitations)}


class AppWorldBenchmarkRunner:
    def __init__(self, store_dir: str | Path, package: AppWorldPackage, protocol: AppWorldProtocol, runtime: AppWorldRuntime) -> None:
        self.store_dir, self.package, self.protocol, self.runtime = Path(store_dir), package, protocol, runtime
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db = self.store_dir / "appworld-benchmark.sqlite3"
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS appworld_plans (benchmark_id TEXT PRIMARY KEY, binding_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS appworld_cells (benchmark_id TEXT NOT NULL, task_id TEXT NOT NULL, split TEXT NOT NULL, arm TEXT NOT NULL, seed INTEGER NOT NULL, run_id TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT, error TEXT, PRIMARY KEY(benchmark_id, task_id, arm, seed))")
            conn.commit()

    def _plan(self, benchmark_id: str, bundles: Mapping[Arm | str, str]) -> dict[str, str]:
        normalized = {Arm(k).value: v for k, v in bundles.items()}
        if set(normalized) != {a.value for a in (Arm.B0, Arm.L, Arm.A)} or any(not isinstance(v, str) or not v for v in normalized.values()):
            raise ValueError("AppWorld requires pinned B0, L, and A bundle hashes")
        binding = {"protocol": self.protocol.to_dict(), "protocolHash": self.protocol.protocol_hash, "bundles": normalized, "dataset": self.protocol.dataset_content_hash, "modelProfile": self.protocol.model_profile, "corePlannerHash": self.protocol.core_planner_hash}
        encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        with sqlite3.connect(self.db) as conn:
            row = conn.execute("SELECT binding_json FROM appworld_plans WHERE benchmark_id=?", (benchmark_id,)).fetchone()
            if row is None: conn.execute("INSERT INTO appworld_plans VALUES (?, ?)", (benchmark_id, encoded))
            elif row[0] != encoded: raise ValueError("benchmark is already bound to different protocol, dataset, or bundles")
            conn.commit()
        return normalized

    @staticmethod
    def _encode(result: AppWorldCellResult) -> str:
        return json.dumps({"observation": json.loads(canonical_json(result.observation)), "usage": dict(result.usage) if result.usage is not None else None}, sort_keys=True)

    @staticmethod
    def _validate_result(result: AppWorldCellResult, *, task: Any, arm: Arm, seed: int, bundle_hash: str) -> None:
        if not isinstance(result, AppWorldCellResult) or not isinstance(result.observation, RunObservation):
            raise ValueError("runtime returned an invalid AppWorld cell result")
        row = result.observation
        if (row.task_id, row.arm, row.seed, row.model_provenance, row.bundle_hash) != (task.task_id, arm, seed, ModelProvenance.REAL_MODEL, bundle_hash):
            raise ValueError("runtime receipt is not bound to the requested task, arm, seed, or bundle")
        if result.usage is not None:
            keys = ("inputTokens", "outputTokens", "totalTokens")
            if any(isinstance(result.usage.get(key), bool) or not isinstance(result.usage.get(key), int) or result.usage[key] < 0 for key in keys):
                raise ValueError("runtime usage receipt is malformed")
            if result.usage["totalTokens"] != result.usage["inputTokens"] + result.usage["outputTokens"]:
                raise ValueError("runtime usage receipt does not reconcile")

    def run(self, benchmark_id: str, bundles: Mapping[Arm | str, str]) -> AppWorldReport:
        bundle_hashes = self._plan(benchmark_id, bundles)
        split_by_id, results = dict(self.protocol.split_by_task_id), []
        for task_id in self.protocol.sampled_task_ids:
            task = self.package.catalog.task(task_id, split_by_id[task_id], allow_test=True)
            for seed in self.protocol.seeds:
                for arm in (Arm.B0, Arm.L, Arm.A):
                    run_id = f"appworld:{benchmark_id}:{task_id}:{arm.value}:{seed}"
                    with sqlite3.connect(self.db) as conn: prior = conn.execute("SELECT status, run_id, result_json FROM appworld_cells WHERE benchmark_id=? AND task_id=? AND arm=? AND seed=?", (benchmark_id, task_id, arm.value, seed)).fetchone()
                    result = None
                    if prior and prior[0] == "complete" and prior[2]:
                        result = self._decode(prior[2])
                        self._validate_result(result, task=task, arm=arm, seed=seed, bundle_hash=bundle_hashes[arm.value])
                        if not self.runtime.verify_appworld_cell(result, package=self.package, task=task, arm=arm, seed=seed, bundle_hash=bundle_hashes[arm.value]): raise ValueError("persisted AppWorld receipt failed verification; refusing re-execution")
                    elif prior and prior[0] == "running":
                        result = self.runtime.recover_appworld_cell(prior[1])
                        if result is None: raise RuntimeError("AppWorld cell is in-flight without a recoverable receipt")
                        self._validate_result(result, task=task, arm=arm, seed=seed, bundle_hash=bundle_hashes[arm.value])
                        if not self.runtime.verify_appworld_cell(result, package=self.package, task=task, arm=arm, seed=seed, bundle_hash=bundle_hashes[arm.value]): raise ValueError("recovered AppWorld receipt failed verification")
                    if result is None:
                        with sqlite3.connect(self.db) as conn: conn.execute("INSERT OR REPLACE INTO appworld_cells VALUES (?, ?, ?, ?, ?, ?, 'running', NULL, NULL)", (benchmark_id, task_id, split_by_id[task_id], arm.value, seed, run_id)); conn.commit()
                        result = self.runtime.run_appworld_cell(package=self.package, task=task, arm=arm, seed=seed, bundle_hash=bundle_hashes[arm.value], budget=self.protocol.budget, run_id=run_id)
                        self._validate_result(result, task=task, arm=arm, seed=seed, bundle_hash=bundle_hashes[arm.value])
                        if not self.runtime.verify_appworld_cell(result, package=self.package, task=task, arm=arm, seed=seed, bundle_hash=bundle_hashes[arm.value]): raise ValueError("runtime AppWorld receipt failed verification")
                    with sqlite3.connect(self.db) as conn: conn.execute("UPDATE appworld_cells SET status='complete', result_json=?, error=NULL WHERE benchmark_id=? AND task_id=? AND arm=? AND seed=?", (self._encode(result), benchmark_id, task_id, arm.value, seed)); conn.commit()
                    results.append(result)
        return self._report(results)

    @staticmethod
    def _decode(payload: str) -> AppWorldCellResult:
        value, raw = json.loads(payload), json.loads(payload)["observation"]
        raw["arm"], raw["model_provenance"] = Arm(raw["arm"]), ModelProvenance(raw["model_provenance"])
        from adaptive_agent.evaluation import Partition
        raw["partition"] = Partition(raw["partition"])
        budget = raw["budget"]
        names = {"model_tokens":"modelTokens", "tool_calls":"toolCalls", "child_runs":"childRuns", "wall_time_seconds":"wallTimeSeconds", "cost_microunits":"costMicrounits", "max_child_depth":"childDepth"}
        raw["budget"] = BudgetSpec(**{k: budget.get(k, budget.get(names.get(k, k), 0)) for k in (*names, "currency")})
        return AppWorldCellResult(RunObservation(**raw), value.get("usage"))

    def _report(self, cells: Sequence[AppWorldCellResult]) -> AppWorldReport:
        rows = [c.observation for c in cells]; summaries = {}
        for arm in (Arm.B0, Arm.L, Arm.A):
            selected = [c for c in cells if c.observation.arm == arm]; usages = [c.usage for c in selected if c.usage is not None]
            summaries[arm.value] = {"accuracy": sum(r.observation.passed for r in selected) / len(selected) if selected else 0.0, "reliability": sum(r.observation.reliable for r in selected) / len(selected) if selected else 0.0, "inputTokens": sum(int(u.get("inputTokens", 0)) for u in usages) if len(usages) == len(selected) else None, "outputTokens": sum(int(u.get("outputTokens", 0)) for u in usages) if len(usages) == len(selected) else None, "costMicrounits": sum(r.observation.cost_microunits for r in selected), "latencySeconds": sum(r.observation.latency_seconds for r in selected), "count": len(selected)}
        base, candidate = [r for r in rows if r.arm == Arm.B0], [r for r in rows if r.arm == Arm.L]; keys = {(r.task_id, r.seed) for r in base} & {(r.task_id, r.seed) for r in candidate}; expected = len(self.protocol.sampled_task_ids) * len(self.protocol.seeds) * 3
        intervals = clustered_paired_bootstrap(base, candidate) if len(rows) == expected else ()
        return AppWorldReport(self.protocol, summaries, tuple(x.to_dict() for x in intervals), len(keys), expected - len(rows), bool(rows) and len(rows) == expected and all(r.model_provenance is ModelProvenance.REAL_MODEL for r in rows), ("Published AppWorld subset; not the full benchmark.", "Only aggregate outcomes and measured usage are exported; task answers and traces remain private."))


__all__ = ["APPWORLD_VERSION", "AppWorldBenchmarkRunner", "AppWorldCellResult", "AppWorldPackage", "AppWorldProtocol", "AppWorldReport", "AppWorldRuntime", "DEFAULT_PUBLISHED_COUNT", "DEFAULT_SEED"]
