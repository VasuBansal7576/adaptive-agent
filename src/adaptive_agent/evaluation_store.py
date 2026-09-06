"""Durable evaluator adapters over the controller Store.

These adapters are deliberately independent of the learner and use the Store's
SQLite transaction boundary and content-addressed artifacts.  A process-local
fallback is never considered durable evidence by :class:`EvaluationReport`.
"""
from __future__ import annotations

import json
from typing import Any, Protocol, Sequence

from adaptive_agent.evaluation import (
    BudgetSpec,
    EnvironmentPackage,
    EvaluationRunner,
    EvaluationProtocol,
    FrozenProtocol,
    RunEvidenceStore,
    RunObservation,
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

    def eval_004(self):
        return self.executor.execute_probe("EVAL-004")

    def eval_005(self):
        return self.executor.execute_probe("EVAL-005")

    def register(self, registry: Any) -> None:
        registry.register_safety_probe("EVAL-004", self.eval_004)
        registry.register_safety_probe("EVAL-005", self.eval_005)


class SQLiteTrustedAttestationLedger:
    durable = True

    def __init__(self, store: Store) -> None:
        self.store = store
        with store._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS evaluator_attestations (token TEXT PRIMARY KEY, digest TEXT NOT NULL, created_at TEXT NOT NULL)")
            conn.commit()

    def put(self, token: str, digest: str) -> None:
        with self.store._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO evaluator_attestations(token, digest, created_at) VALUES (?, ?, datetime('now'))", (token, digest))
            conn.commit()

    def get(self, token: str) -> str | None:
        with self.store._connect() as conn:
            row = conn.execute("SELECT digest FROM evaluator_attestations WHERE token = ?", (token,)).fetchone()
        return str(row["digest"]) if row else None


class SQLiteAllocationStore:
    durable = True

    def __init__(self, store: Store) -> None:
        self.store = store
        with store._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS evaluator_allocations (scope_id TEXT NOT NULL, allocation_id TEXT PRIMARY KEY, panel_index INTEGER NOT NULL, panel_hash TEXT NOT NULL, task_ids_json TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(scope_id, panel_index))")
            conn.commit()

    def reserve_next(self, scope_id: str, allocation_id: str, panels: Sequence[Sequence[str]], limit: int) -> int | None:
        if not panels or len(panels) < limit:
            raise ValueError("allocation panels must cover the configured limit")
        with self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT panel_index FROM evaluator_allocations WHERE allocation_id = ?", (allocation_id,)).fetchone()
            if existing:
                conn.rollback()
                return None
            used = {int(row["panel_index"]) for row in conn.execute("SELECT panel_index FROM evaluator_allocations WHERE scope_id = ?", (scope_id,)).fetchall()}
            index = next((candidate for candidate in range(limit) if candidate not in used), None)
            if index is None:
                conn.rollback()
                return None
            task_ids = list(panels[index])
            conn.execute("INSERT INTO evaluator_allocations(scope_id, allocation_id, panel_index, panel_hash, task_ids_json, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))", (scope_id, allocation_id, index, sha256_json(task_ids), json.dumps(task_ids, sort_keys=True)))
            conn.commit()
            return index


class SQLiteRunEvidenceStore:
    durable = True

    def __init__(self, store: Store) -> None:
        self.store = store

    def verify(self, observation: RunObservation, frozen: FrozenProtocol, package: EnvironmentPackage) -> bool:
        if observation.model_provenance.value != "real_model" or not observation.response_id or not observation.accounting_ref or not observation.evidence_ref:
            return False
        evidence = self.store.get_evidence(observation.evidence_ref)
        call = self.store.get_tool_call(observation.response_id)
        if not evidence or not call or call.get("environment_id") != observation.environment_id:
            return False
        if call.get("result_json") is None:
            return False
        try:
            source_ref = json.loads(evidence["source_ref"])
            response = json.loads(call["result_json"])
            accounting = self.store.get_artifact(observation.accounting_ref)
            observed = self.store.get_artifact(source_ref["sha256"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if sha256_json(observed) != evidence.get("content_hash"):
            return False
        if sha256_json(response) != source_ref.get("sha256") and response != observed:
            return False
        if not isinstance(accounting, dict) or accounting.get("responseId") != observation.response_id:
            return False
        expected = {
            "model": sha256_json({"profile": frozen.payload["modelProfile"], "provider": frozen.payload["provider"]}),
            "planner": frozen.payload["corePlannerHash"],
            "budget": sha256_json(BudgetSpec()),
            "policy": sha256_json(package.manifest.policy_ref),
            "schema": sha256_json(package.manifest.tool_schemas),
            "image": sha256_json({"modelProfile": frozen.payload["modelProfile"]}),
        }
        return dict(observation.config_hashes) == expected


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
