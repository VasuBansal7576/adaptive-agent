"""Durable evaluator adapters over the controller Store.

These adapters are deliberately independent of the learner and use the Store's
SQLite transaction boundary and content-addressed artifacts.  A process-local
fallback is never considered durable evidence by :class:`EvaluationReport`.
"""
from __future__ import annotations

import json
import sqlite3
import math
from collections.abc import Mapping
from typing import Any, Protocol, Sequence

from adaptive_agent.evaluation import (
    EnvironmentPackage,
    EvaluationRunner,
    EvaluationProtocol,
    FrozenProtocol,
    RunEvidenceStore,
    RunObservation,
    SafetyProbeResult,
    TrustedAttestationLedger,
    TrustedEvaluatorRegistry,
    sha256_json,
)
from adaptive_agent.store import Store


class ControllerProbeExecutor(Protocol):
    """Session-4-owned real Controller/ToolBroker probe boundary."""

    def execute_probe(self, case_id: str) -> Any: ...


class ControllerSafetyProbeAdapter:
    """Bind real Controller/ToolBroker probes into the trusted registry.

    The executor is supplied by session 4 and must return a SafetyProbeResult
    containing broker outputs, provenance, and obligations.  No synthetic
    fallback is accepted by the registry.
    """

    def __init__(self, executor: ControllerProbeExecutor) -> None:
        self.executor = executor

    def _run(self, case_id: str) -> SafetyProbeResult:
        """Accept only complete, transcript-backed controller probe results."""
        raw = self.executor.execute_probe(case_id)
        if isinstance(raw, SafetyProbeResult):
            payload: Mapping[str, Any] = raw.to_dict()
        elif hasattr(raw, "to_dict") and callable(raw.to_dict):
            payload = raw.to_dict()
        elif isinstance(raw, Mapping):
            payload = raw
        else:
            raise TypeError("controller probe must return an object")
        outputs = payload.get("outputs")
        provenance = payload.get("provenance")
        obligations = payload.get("obligations")
        # Controller probes return a richer per-obligation mapping and a single
        # provenance label. Normalize that shape into the evaluator's immutable
        # tuple contract while retaining each observed detail for attestation.
        if isinstance(outputs, Mapping):
            observed = payload.get("observed")
            outputs = tuple(
                {
                    "obligation": str(name),
                    "passed": bool(observed.get(name, True)) if isinstance(observed, Mapping) else True,
                    "detail": detail,
                }
                for name, detail in outputs.items()
            )
        if isinstance(provenance, str):
            provenance = (provenance,)
        if (
            not isinstance(payload.get("passed"), bool)
            or not isinstance(outputs, (list, tuple))
            or not outputs
            or not all(isinstance(value, dict) for value in outputs)
            or not isinstance(provenance, (list, tuple))
            or not provenance
            or not all(isinstance(value, str) and value for value in provenance)
            or not isinstance(obligations, (list, tuple))
            or not obligations
            or not all(isinstance(value, str) and value for value in obligations)
        ):
            raise ValueError(f"controller probe {case_id} returned incomplete evidence")
        return SafetyProbeResult(bool(payload["passed"]), tuple(outputs), tuple(provenance), tuple(obligations))

    def eval_004(self):
        return self._run("EVAL-004")

    def eval_005(self):
        return self._run("EVAL-005")

    def register(self, registry: Any) -> None:
        registry.register_safety_probe("EVAL-004", self.eval_004)
        registry.register_safety_probe("EVAL-005", self.eval_005)


class SQLiteTrustedAttestationLedger:
    durable = True

    def __init__(self, store: Store) -> None:
        self.store = store
        with store.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS evaluator_attestations (token TEXT PRIMARY KEY, digest TEXT NOT NULL, created_at TEXT NOT NULL)")
            conn.commit()

    def put(self, token: str, digest: str) -> None:
        with self.store.connect() as conn:
            try:
                conn.execute("INSERT INTO evaluator_attestations(token, digest, created_at) VALUES (?, ?, datetime('now'))", (token, digest))
                conn.commit()
            except sqlite3.IntegrityError:
                conn.rollback()
                if self.get(token) != digest:
                    raise ValueError("attestation token already has a different digest")

    def get(self, token: str) -> str | None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT digest FROM evaluator_attestations WHERE token = ?", (token,)).fetchone()
        return str(row["digest"]) if row else None


class SQLiteAllocationStore:
    durable = True

    def __init__(self, store: Store) -> None:
        self.store = store

    def reserve_next(self, scope_id: str, allocation_id: str, panels: Sequence[Sequence[str]], limit: int) -> int | None:
        if not panels or len(panels) < limit:
            raise ValueError("allocation panels must cover the configured limit")
        return self.store.reserve_allocation(scope_id, allocation_id, [list(panel) for panel in panels], limit)

    def get(self, allocation_id: str) -> dict[str, Any] | None:
        return self.store.get_allocation(allocation_id)


class SQLiteRunEvidenceStore:
    durable = True

    def __init__(self, store: Store) -> None:
        self.store = store

    def verify(self, observation: RunObservation, frozen: FrozenProtocol, package: EnvironmentPackage) -> bool:
        if observation.model_provenance.value != "real_model" or not observation.response_id or not observation.accounting_ref or not observation.evidence_ref or not observation.outcome_ref or not observation.run_id:
            return False
        evidence = self.store.get_evidence(observation.evidence_ref)
        outcome_evidence = self.store.get_evidence(observation.outcome_ref)
        if not evidence or not outcome_evidence:
            return False
        run = self.store.get_run(observation.run_id)
        if not run or run.get("task_id") != observation.task_id or run.get("environment_id") != observation.environment_id:
            return False
        try:
            source_ref = json.loads(evidence["source_ref"])
            accounting = self.store.get_artifact(observation.accounting_ref)
            response = self.store.get_artifact(source_ref["sha256"])
            outcome_source = json.loads(outcome_evidence["source_ref"])
            outcome = self.store.get_artifact(outcome_source["sha256"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if not all(isinstance(value, dict) for value in (response, accounting, outcome)):
            return False
        receipts = accounting.get("receipts")
        aggregate_usage = accounting.get("aggregateUsage")
        if receipts is not None:
            if not isinstance(receipts, list) or not isinstance(aggregate_usage, dict):
                return False
            calculated = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
            for receipt in receipts:
                if not isinstance(receipt, dict) or not isinstance(receipt.get("usage"), dict):
                    return False
                usage = receipt["usage"]
                if any(not isinstance(usage.get(key), int) or usage[key] < 0 for key in calculated):
                    return False
                for key in calculated:
                    calculated[key] += usage[key]
            if aggregate_usage != calculated:
                return False
        if sha256_json(response) != evidence.get("content_hash"):
            return False
        if response.get("responseId") != observation.response_id or evidence.get("run_id") != observation.run_id or evidence.get("event_type") != "model_response" or outcome_evidence.get("run_id") != observation.run_id or outcome_evidence.get("event_type") != "trusted_outcome":
            return False
        if evidence.get("visibility") != "operator" or outcome_evidence.get("visibility") != "operator":
            return False
        if evidence.get("eventType") not in (None, "model_response") or outcome_evidence.get("eventType") not in (None, "trusted_outcome"):
            return False
        usage = response.get("usage")
        accounting_usage = accounting.get("usage")
        if not isinstance(usage, dict) or not isinstance(accounting_usage, dict) or usage != accounting_usage:
            return False
        if accounting.get("responseId") != observation.response_id or accounting.get("runId") != observation.run_id or accounting.get("taskId") != observation.task_id or accounting.get("environmentId") != observation.environment_id:
            return False
        if outcome.get("responseId") != observation.response_id or outcome.get("runId") != observation.run_id or outcome.get("taskId") != observation.task_id or outcome.get("environmentId") != observation.environment_id:
            return False
        if not all(isinstance(usage.get(key), int) and usage[key] >= 0 for key in ("inputTokens", "outputTokens", "totalTokens")) or usage["totalTokens"] != usage["inputTokens"] + usage["outputTokens"]:
            return False
        actual_cost = accounting.get("costMicrounits")
        actual_latency = accounting.get("durationSeconds")
        if not isinstance(actual_cost, (int, float)) or isinstance(actual_cost, bool) or not math.isfinite(actual_cost) or actual_cost < 0 or not isinstance(actual_latency, (int, float)) or isinstance(actual_latency, bool) or not math.isfinite(actual_latency) or actual_latency < 0:
            return False
        if actual_cost != observation.cost_microunits or actual_latency != observation.latency_seconds:
            return False
        if outcome.get("passed") != observation.passed or outcome.get("reliable") != observation.reliable or outcome.get("safetyViolations") != observation.safety_violations:
            return False
        if not isinstance(accounting.get("versionRefs"), dict) or accounting.get("versionRefs") != response.get("versionRefs") or not accounting.get("versionRefs"):
            return False
        expected = {
            "model": sha256_json({"profile": frozen.inputs["modelProfile"], "provider": frozen.inputs["provider"]}),
            "planner": str(frozen.inputs["corePlannerHash"]),
            "budget": sha256_json(frozen.inputs["runBudget"]),
            "policy": sha256_json(package.manifest.policy_ref),
            "schema": sha256_json(package.manifest.tool_schemas),
            "image": str(frozen.inputs["imageDigest"]),
        }
        expected_image = str(frozen.inputs["imageDigest"])
        expected_budget = sha256_json(frozen.inputs["runBudget"])
        for payload in (response, accounting):
            direct_image = payload.get("imageDigest")
            if direct_image is not None and direct_image != expected_image:
                return False
            direct_budget = payload.get("budgetRef")
            if isinstance(direct_budget, dict) and direct_budget.get("sha256") != expected_budget:
                return False
        expected_version_refs = {"policy": package.manifest.policy_ref.sha256, "schema": sha256_json(package.manifest.tool_schemas), "planner": str(frozen.inputs["corePlannerHash"]), "budget": sha256_json(frozen.inputs["runBudget"]), "image": expected_image}
        if accounting["versionRefs"] != expected_version_refs:
            return False
        return dict(observation.config_hashes) == expected and response.get("provider") == frozen.inputs["provider"] and response.get("modelProfile") == frozen.inputs["modelProfile"]


def build_durable_adapters(store: Store) -> tuple[TrustedAttestationLedger, SQLiteAllocationStore, RunEvidenceStore]:
    """Construct the evaluator-owned adapters on one persistent SQLite store."""
    return SQLiteTrustedAttestationLedger(store), SQLiteAllocationStore(store), SQLiteRunEvidenceStore(store)


def build_durable_evaluation_runner(protocol: EvaluationProtocol, packages: dict[str, EnvironmentPackage], store: Store, probe_executor: ControllerProbeExecutor | None = None) -> EvaluationRunner:
    """Build the evaluator with SQLite adapters and optional real broker probes."""
    ledger, allocations, evidence = build_durable_adapters(store)
    registry = TrustedEvaluatorRegistry(ledger)
    if probe_executor is not None:
        ControllerSafetyProbeAdapter(probe_executor).register(registry)
    return EvaluationRunner(protocol, packages, evaluator_registry=registry, allocation_store=allocations, evidence_store=evidence)


__all__ = ["ControllerProbeExecutor", "ControllerSafetyProbeAdapter", "SQLiteAllocationStore", "SQLiteRunEvidenceStore", "SQLiteTrustedAttestationLedger", "build_durable_adapters", "build_durable_evaluation_runner"]
