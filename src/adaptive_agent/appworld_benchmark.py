"""Durable, aggregate-only AppWorld benchmark orchestration."""
from __future__ import annotations

import json
import os
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Protocol, Sequence

from adaptive_agent.benchmark import FrozenExecutionConfig
from adaptive_agent.evaluation import AblationInput, Arm, BudgetSpec, FrozenProtocol, ModelProvenance, Partition, RunObservation, audit_ablation, canonical_json, clustered_paired_bootstrap, sha256_json

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


class DurableAppWorldAdapter:
    """Concrete adapter over the production DurableRuntime seam.

    ``execute_evaluation_task`` remains the sole execution path.  In
    particular, this adapter never constructs a provider, calls a model, or
    infers an outcome from AppWorld output.
    """

    def __init__(self, runtime: Any, protocol: "AppWorldProtocol", bundle_hashes: Mapping[str, str] | None = None) -> None:
        self.runtime, self.protocol = runtime, protocol
        self.bundle_hashes = dict(bundle_hashes or {})
        self._resolved_bundles: dict[str, Any] = {}
        self._a_audit: Any | None = None

    def preflight_bundles(self, bundle_hashes: Mapping[str, str]) -> None:
        self._assert_runtime_pins()
        if self.bundle_hashes and dict(self.bundle_hashes) != dict(bundle_hashes):
            raise ValueError("AppWorld arm bundle mapping is immutable")
        self.bundle_hashes = dict(bundle_hashes)
        for arm in (Arm.B0, Arm.L, Arm.A):
            self._resolved_bundles[arm.value] = self._bundle(self.bundle_hashes[arm.value])
        a_bundle = self._resolved_bundles[Arm.A.value]
        execution_config = getattr(a_bundle, "execution_config", None)
        if getattr(a_bundle, "skills", None) or execution_config is None or getattr(execution_config, "skill_refs", None) or getattr(execution_config, "instruction_variant", "default") != "default":
            raise ValueError("AppWorld arm A retains learned skills or instruction configuration")
        procedures = tuple(str(getattr(skill, "procedure", "")) for skill in getattr(a_bundle, "skills", ()))
        config_payload = execution_config.model_dump(mode="json", by_alias=True) if callable(getattr(execution_config, "model_dump", None)) else vars(execution_config)
        self._a_audit = audit_ablation(AblationInput(a_bundle.content_hash, canonical_json(config_payload), procedures))
        if not self._a_audit.passed:
            raise ValueError("AppWorld arm A memory-disabled audit failed")

    def ablation_audit(self) -> Any:
        if self._a_audit is None:
            raise ValueError("AppWorld arm A audit was not preflighted")
        return self._a_audit

    def _assert_runtime_pins(self) -> None:
        actual_image = getattr(self.runtime, "image_digest", None)
        actual_provider = getattr(self.runtime, "provider", "openai-codex")
        actual_revision = getattr(self.runtime, "source_revision", None) or os.environ.get("ADAPTIVE_AGENT_SOURCE_REVISION")
        actual_core = getattr(self.runtime, "core_planner_hash", None)
        if actual_image != self.protocol.image_digest or not isinstance(actual_image, str) or not actual_image or actual_image == "image-unpinned":
            raise ValueError("AppWorld runtime image digest does not match the frozen protocol")
        if actual_provider != self.protocol.provider or actual_provider != "openai-codex":
            raise ValueError("AppWorld runtime provider does not match the frozen protocol")
        if actual_revision != self.protocol.source_revision or not isinstance(actual_revision, str) or not actual_revision:
            raise ValueError("AppWorld runtime source revision does not match the frozen protocol")
        if actual_core != self.protocol.core_planner_hash:
            raise ValueError("AppWorld runtime core planner hash does not match the frozen protocol")

    def _frozen(self) -> FrozenProtocol:
        inputs = {"modelProfile": self.protocol.model_profile, "provider": self.protocol.provider, "corePlannerHash": self.protocol.core_planner_hash, "imageDigest": self.protocol.image_digest, "sourceRevision": self.protocol.source_revision, "runBudget": self.protocol.budget.to_dict()}
        return FrozenProtocol(self.protocol.protocol_hash, {}, {}, inputs)

    def _task(self, task: Any) -> Any:
        partition = Partition.DEVELOPMENT if self.protocol.official_split == "train" else (Partition.VALIDATION if self.protocol.official_split == "dev" else Partition.FINAL)
        return SimpleNamespace(task_id=task.task_id, environment_id="appworld", goal=task.instruction, partition=partition, environment_ref=SimpleNamespace(id="appworld", version=APPWORLD_VERSION))

    def _bundle(self, bundle_hash: str) -> Any:
        store = self.runtime.controller.store
        row = store.get_bundle_by_hash(bundle_hash)
        if not isinstance(row, Mapping) or row.get("content_hash") != bundle_hash:
            raise ValueError("pinned AppWorld bundle is missing from the durable store")
        from adaptive_agent.models import SkillBundle
        value = row.get("bundle_json")
        payload = json.loads(value) if isinstance(value, str) else value
        if not isinstance(payload, Mapping):
            raise ValueError("stored AppWorld bundle payload is malformed")
        bundle = SkillBundle.model_validate(payload)
        expected = sha256_json(bundle.model_dump(mode="json", by_alias=True, exclude={"content_hash"}))
        if expected != bundle_hash or bundle.content_hash != bundle_hash:
            raise ValueError("stored AppWorld bundle hash does not validate")
        return bundle

    def _config(self, arm: Arm, seed: int, bundle_hash: str) -> FrozenExecutionConfig:
        return FrozenExecutionConfig(self._frozen(), arm, seed, bundle_hash, 0)

    def _usage(self, observation: RunObservation) -> Mapping[str, int]:
        row = self.runtime.controller.store.get_run(observation.run_id or "")
        if not isinstance(row, Mapping):
            raise ValueError("durable AppWorld run is missing")
        try:
            run_payload = json.loads(row.get("run_json", "{}"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("durable AppWorld run identity is malformed") from exc
        ref = run_payload.get("finalAccountingRef")
        accounting = self.runtime.controller.store.get_artifact(ref) if isinstance(ref, str) else None
        usage = accounting.get("aggregateUsage") if isinstance(accounting, Mapping) else None
        if not isinstance(usage, Mapping):
            raise ValueError("verified AppWorld final accounting has no measured usage")
        values = {key: usage.get(key) for key in ("inputTokens", "outputTokens", "totalTokens")}
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()) or values["totalTokens"] != values["inputTokens"] + values["outputTokens"]:
            raise ValueError("verified AppWorld final accounting usage is malformed")
        return values

    def run_appworld_cell(self, *, package: AppWorldPackage, task: Any, arm: Arm, seed: int, bundle_hash: str, budget: BudgetSpec, run_id: str) -> AppWorldCellResult:
        selected = self._resolved_bundles.get(arm.value) or self._bundle(bundle_hash)
        if self.bundle_hashes.get(arm.value) != bundle_hash:
            raise ValueError("AppWorld arm bundle mapping changed after preflight")
        self.runtime._evaluation_arm_bundles = dict(self.bundle_hashes)
        observation = self.runtime.execute_evaluation_task(self._task(task), self._config(arm, seed, bundle_hash), selected)
        if not self.runtime.verify_evaluation_observation(observation, self._config(arm, seed, bundle_hash), self._task(task)):
            raise ValueError("DurableRuntime rejected AppWorld observation verification")
        return AppWorldCellResult(observation, self._usage(observation))

    def recover_appworld_cell(self, *, package: AppWorldPackage, task: Any, arm: Arm, seed: int, bundle_hash: str, run_id: str) -> AppWorldCellResult | None:
        key = f"benchmark:{self.protocol.protocol_hash}:{task.task_id}:{arm.value}:{seed}:{bundle_hash}:attempt:0"
        existing = self.runtime.controller.store.get_run_by_idempotency_key(key)
        if not isinstance(existing, Mapping) or existing.get("status") not in {"succeeded", "failed"}:
            return None
        return self.run_appworld_cell(package=package, task=task, arm=arm, seed=seed, bundle_hash=bundle_hash, budget=self.protocol.budget, run_id=run_id)

    def verify_appworld_cell(self, result: AppWorldCellResult, *, package: AppWorldPackage, task: Any, arm: Arm, seed: int, bundle_hash: str) -> bool:
        try:
            self._validate(result, task=task, arm=arm, seed=seed, bundle_hash=bundle_hash)
            fresh_usage = self._usage(result.observation)
            return self.runtime.verify_evaluation_observation(result.observation, self._config(arm, seed, bundle_hash), self._task(task)) and result.usage is not None and dict(result.usage) == dict(fresh_usage)
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _validate(result: AppWorldCellResult, *, task: Any, arm: Arm, seed: int, bundle_hash: str) -> None:
        if not isinstance(result, AppWorldCellResult) or result.observation.task_id != task.task_id or result.observation.arm != arm or result.observation.seed != seed or result.observation.bundle_hash != bundle_hash or result.observation.model_provenance is not ModelProvenance.REAL_MODEL:
            raise ValueError("invalid AppWorld durable observation")


class AppWorldRuntime(Protocol):
    def run_appworld_cell(self, *, package: AppWorldPackage, task: Any, arm: Arm, seed: int, bundle_hash: str, budget: BudgetSpec, run_id: str) -> AppWorldCellResult: ...
    def recover_appworld_cell(self, *, package: AppWorldPackage, task: Any, arm: Arm, seed: int, bundle_hash: str, run_id: str) -> AppWorldCellResult | None: ...
    def verify_appworld_cell(self, result: AppWorldCellResult, *, package: AppWorldPackage, task: Any, arm: Arm, seed: int, bundle_hash: str) -> bool: ...


@dataclass(frozen=True)
class AppWorldProtocol:
    source: str
    model_profile: str
    core_planner_hash: str
    dataset_content_hash: str
    image_digest: str
    provider: str
    source_revision: str
    seeds: tuple[int, ...] = (0,)
    published_count: int = DEFAULT_PUBLISHED_COUNT
    sampling_seed: int = DEFAULT_SEED
    official_split: str = ""
    budget: BudgetSpec = BudgetSpec()
    sampled_task_ids: tuple[str, ...] = ()
    split_by_task_id: tuple[tuple[str, str], ...] = ()
    official_split_counts: tuple[tuple[str, int], ...] = ()
    protocol_hash: str = ""

    @classmethod
    def freeze(cls, package: AppWorldPackage, *, model_profile: str, core_planner_hash: str, official_split: str, image_digest: str, provider: str = "openai-codex", source_revision: str, dataset_content_hash: str | None = None, published_count: int = DEFAULT_PUBLISHED_COUNT, sampling_seed: int = DEFAULT_SEED, seeds: tuple[int, ...] = (0,), budget: BudgetSpec | None = None) -> "AppWorldProtocol":
        if official_split not in ("train", "dev", *FINAL_SPLITS) or not model_profile or not core_planner_hash or not image_digest or image_digest == "image-unpinned" or provider != "openai-codex" or not source_revision or published_count < 1 or not seeds or any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
            raise ValueError("invalid AppWorld protocol pins")
        catalog = package.catalog
        content_hash = dataset_content_hash or catalog.dataset_hash()
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError("AppWorld dataset hash is required")
        split_ids = {split: tuple(str(x) for x in catalog.split_ids(split)) for split in ("train", "dev", *FINAL_SPLITS)}
        if any(len(ids) != len(set(ids)) for ids in split_ids.values()):
            raise ValueError("duplicate AppWorld task IDs")
        selected_ids = split_ids[official_split]
        if published_count > len(selected_ids):
            raise ValueError("published subset exceeds official task pool")
        sampled = tuple(sorted(random.Random(sampling_seed).sample(selected_ids, published_count)))
        split_by_id = tuple((task_id, split) for split, ids in split_ids.items() for task_id in ids if task_id in sampled)
        selected_budget = budget or BudgetSpec()
        payload = {"source": "appworld", "version": APPWORLD_VERSION, "modelProfile": model_profile, "corePlannerHash": core_planner_hash, "datasetContentHash": content_hash, "imageDigest": image_digest, "provider": provider, "sourceRevision": source_revision, "officialSplit": official_split, "seeds": list(seeds), "publishedCount": published_count, "samplingSeed": sampling_seed, "budget": selected_budget.to_dict(), "officialSplitCounts": {k: len(v) for k, v in split_ids.items()}, "sampledTaskIds": list(sampled), "splitByTaskId": dict(split_by_id)}
        return cls("appworld", model_profile, core_planner_hash, content_hash, image_digest, provider, source_revision, tuple(seeds), published_count, sampling_seed, official_split, selected_budget, sampled, split_by_id, tuple((k, len(v)) for k, v in split_ids.items()), sha256_json(payload))

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "version": APPWORLD_VERSION, "modelProfile": self.model_profile, "corePlannerHash": self.core_planner_hash, "datasetContentHash": self.dataset_content_hash, "imageDigest": self.image_digest, "provider": self.provider, "sourceRevision": self.source_revision, "officialSplit": self.official_split, "seeds": list(self.seeds), "publishedCount": self.published_count, "samplingSeed": self.sampling_seed, "budget": self.budget.to_dict(), "officialSplitCounts": dict(self.official_split_counts), "sampledTaskIds": list(self.sampled_task_ids), "splitByTaskId": dict(self.split_by_task_id), "protocolHash": self.protocol_hash, "scope": "published_subset" if self.published_count == DEFAULT_PUBLISHED_COUNT else "configured_subset"}


@dataclass(frozen=True)
class AppWorldReport:
    protocol: AppWorldProtocol
    arm_summaries: Mapping[str, Mapping[str, Any]]
    confidence_intervals: tuple[Mapping[str, Any], ...]
    paired_task_count: int
    missing_pairs: int
    provenance_complete: bool
    limitations: tuple[str, ...]
    ablation_audit: Mapping[str, Any] | None = None
    def to_dict(self) -> dict[str, Any]:
        selected_arms = [arm for arm, summary in self.arm_summaries.items() if summary.get("count", 0) > 0]
        return {"benchmark": "appworld", "protocol": self.protocol.to_dict(), "selectedArms": selected_arms, "armSummaries": dict(self.arm_summaries), "confidenceIntervals": list(self.confidence_intervals), "pairedTaskCount": self.paired_task_count, "missingPairs": self.missing_pairs, "provenanceComplete": self.provenance_complete, "limitations": list(self.limitations), "ablationAudit": self.ablation_audit}


class AppWorldBenchmarkRunner:
    def __init__(self, store_dir: str | Path, package: AppWorldPackage, protocol: AppWorldProtocol, runtime: AppWorldRuntime) -> None:
        self.store_dir, self.package, self.protocol, self.runtime = Path(store_dir), package, protocol, runtime
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.db = self.store_dir / "appworld-benchmark.sqlite3"
        with sqlite3.connect(self.db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS appworld_plans (benchmark_id TEXT PRIMARY KEY, binding_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS appworld_cells (benchmark_id TEXT NOT NULL, task_id TEXT NOT NULL, split TEXT NOT NULL, arm TEXT NOT NULL, seed INTEGER NOT NULL, run_id TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT, error TEXT, PRIMARY KEY(benchmark_id, task_id, arm, seed))")
            conn.commit()

    def _plan(self, benchmark_id: str, bundles: Mapping[Arm | str, str], arms: Sequence[Arm]) -> dict[str, str]:
        if self.package.catalog.dataset_hash() != self.protocol.dataset_content_hash:
            raise ValueError("AppWorld dataset hash changed since protocol freeze")
        protocol_payload = self.protocol.to_dict()
        protocol_payload.pop("protocolHash", None); protocol_payload.pop("scope", None)
        if sha256_json(protocol_payload) != self.protocol.protocol_hash:
            raise ValueError("AppWorld protocol hash no longer matches canonical frozen inputs")
        normalized = {Arm(k).value: v for k, v in bundles.items()}
        if set(normalized) != {a.value for a in (Arm.B0, Arm.L, Arm.A)} or any(not isinstance(v, str) or not v for v in normalized.values()):
            raise ValueError("AppWorld requires pinned B0, L, and A bundle hashes")
        if self.protocol.official_split == "train" and tuple(arms) != (Arm.B0,):
            raise ValueError("AppWorld training is restricted to the B0 arm")
        binding = {"protocol": self.protocol.to_dict(), "protocolHash": self.protocol.protocol_hash, "bundles": normalized, "arms": [arm.value for arm in arms], "dataset": self.protocol.dataset_content_hash, "modelProfile": self.protocol.model_profile, "corePlannerHash": self.protocol.core_planner_hash}
        encoded = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        with sqlite3.connect(self.db) as conn:
            row = conn.execute("SELECT binding_json FROM appworld_plans WHERE benchmark_id=?", (benchmark_id,)).fetchone()
            if row is None: conn.execute("INSERT INTO appworld_plans VALUES (?, ?)", (benchmark_id, encoded))
            elif row[0] != encoded: raise ValueError("benchmark is already bound to different protocol, dataset, or bundles")
            conn.commit()
        preflight = getattr(self.runtime, "preflight_bundles", None)
        if not callable(preflight):
            raise ValueError("AppWorld runtime must preflight all arm bundles")
        preflight(normalized)
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

    def run(self, benchmark_id: str, bundles: Mapping[Arm | str, str], *, arms: Sequence[Arm] = (Arm.B0, Arm.L, Arm.A)) -> AppWorldReport:
        selected_arms = tuple(arms)
        if not selected_arms or any(arm not in (Arm.B0, Arm.L, Arm.A) for arm in selected_arms) or len(set(selected_arms)) != len(selected_arms):
            raise ValueError("AppWorld arms must be a non-empty subset of B0, L, and A")
        bundle_hashes = self._plan(benchmark_id, bundles, selected_arms)
        self._selected_arms = selected_arms
        split_by_id, results = dict(self.protocol.split_by_task_id), []
        for task_id in self.protocol.sampled_task_ids:
            task = self.package.catalog.task(task_id, split_by_id[task_id], allow_test=split_by_id[task_id] in FINAL_SPLITS)
            for seed in self.protocol.seeds:
                for arm in selected_arms:
                    run_id = f"appworld:{benchmark_id}:{task_id}:{arm.value}:{seed}"
                    with sqlite3.connect(self.db) as conn: prior = conn.execute("SELECT status, run_id, result_json FROM appworld_cells WHERE benchmark_id=? AND task_id=? AND arm=? AND seed=?", (benchmark_id, task_id, arm.value, seed)).fetchone()
                    result = None
                    if prior and prior[0] == "complete" and prior[2]:
                        result = self._decode(prior[2])
                        self._validate_result(result, task=task, arm=arm, seed=seed, bundle_hash=bundle_hashes[arm.value])
                        if not self.runtime.verify_appworld_cell(result, package=self.package, task=task, arm=arm, seed=seed, bundle_hash=bundle_hashes[arm.value]): raise ValueError("persisted AppWorld receipt failed verification; refusing re-execution")
                    elif prior and prior[0] == "running":
                        result = self.runtime.recover_appworld_cell(package=self.package, task=task, arm=arm, seed=seed, bundle_hash=bundle_hashes[arm.value], run_id=prior[1])
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
            summaries[arm.value] = {"accuracy": sum(r.observation.passed for r in selected) / len(selected) if selected else 0.0, "reliability": sum(r.observation.reliable for r in selected) / len(selected) if selected else 0.0, "inputTokens": sum(int(u.get("inputTokens", 0)) for u in usages) if len(usages) == len(selected) else None, "outputTokens": sum(int(u.get("outputTokens", 0)) for u in usages) if len(usages) == len(selected) else None, "totalTokens": sum(int(u.get("totalTokens", 0)) for u in usages) if len(usages) == len(selected) else None, "costMicrounits": sum(r.observation.cost_microunits for r in selected), "latencySeconds": sum(r.observation.latency_seconds for r in selected), "count": len(selected)}
        base, candidate = [r for r in rows if r.arm == Arm.B0], [r for r in rows if r.arm == Arm.L]; keys = {(r.task_id, r.seed) for r in base} & {(r.task_id, r.seed) for r in candidate}; expected = len(self.protocol.sampled_task_ids) * len(self.protocol.seeds) * len(getattr(self, "_selected_arms", (Arm.B0, Arm.L, Arm.A)))
        intervals = clustered_paired_bootstrap(base, candidate) if len(rows) == expected and {Arm.B0, Arm.L}.issubset(getattr(self, "_selected_arms", ())) else ()
        audit = getattr(self.runtime, "ablation_audit", lambda: None)()
        audit_value = audit.to_dict() if audit is not None and hasattr(audit, "to_dict") else (asdict(audit) if audit is not None else None)
        return AppWorldReport(self.protocol, summaries, tuple(x.to_dict() for x in intervals), len(keys), expected - len(rows), bool(rows) and len(rows) == expected and all(r.model_provenance is ModelProvenance.REAL_MODEL for r in rows), ("Published AppWorld subset; not the full benchmark.", "Only aggregate outcomes and measured usage are exported; task answers and traces remain private."), audit_value)


def create_appworld_benchmark_runner(runtime: Any, package: AppWorldPackage, protocol: AppWorldProtocol, bundles: Mapping[Arm | str, str], store_dir: str | Path | None = None) -> AppWorldBenchmarkRunner:
    """Create the production runner from an existing DurableRuntime."""
    selected = {Arm(key).value: value for key, value in bundles.items()}
    adapter = DurableAppWorldAdapter(runtime, protocol, selected)
    root = store_dir if store_dir is not None else runtime.controller.store.base_dir
    return AppWorldBenchmarkRunner(root, package, protocol, adapter)


__all__ = ["APPWORLD_VERSION", "AppWorldBenchmarkRunner", "AppWorldCellResult", "AppWorldPackage", "AppWorldProtocol", "AppWorldReport", "AppWorldRuntime", "DurableAppWorldAdapter", "create_appworld_benchmark_runner", "DEFAULT_PUBLISHED_COUNT", "DEFAULT_SEED"]
