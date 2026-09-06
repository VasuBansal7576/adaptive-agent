"""Production evaluation-job orchestration.

This module owns the boundary between durable runtime execution and evaluator
scoring.  The driver is the only component allowed to select or execute a
panel; ``EvaluationRunner`` receives the driver's persisted observations only.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping

from adaptive_agent.benchmark import FrozenExecutionConfig, ResumableEvaluationDriver, TrustedTaskExecutor
from adaptive_agent.controller import Controller
from adaptive_agent.evaluation import (
    AblationInput,
    Arm,
    EnvironmentPackage,
    EvaluationError,
    EvaluationProtocol,
    EvaluationReport,
    EvaluationRunner,
    Partition,
    RunObservation,
)
from adaptive_agent.evaluation_store import SQLiteRunEvidenceStore, build_durable_evaluation_runner
from adaptive_agent.store import Store


@dataclass(frozen=True)
class EvaluationJobResult:
    job_id: str
    comparison: str
    status: str
    report: EvaluationReport | dict[str, Any] | None
    decision: object | None = None
    error: str | None = None
    runtime_accounting: dict[str, Any] | None = None


class EvaluationJob:
    """Run and persist one immutable comparison job."""

    def __init__(
        self,
        store: Store,
        controller: Controller,
        protocol: EvaluationProtocol,
        packages: Mapping[str, EnvironmentPackage],
        arm_bundles: Mapping[Arm | str, object],
        execute: TrustedTaskExecutor,
        *,
        total_budget_microunits: int | None = None,
        max_total_attempts: int | None = None,
    ) -> None:
        self.store = store
        self.controller = controller
        self.protocol = protocol
        self.packages = dict(packages)
        self.arm_bundles = dict(arm_bundles)
        self.execute = execute
        self.total_budget_microunits = total_budget_microunits
        if max_total_attempts is not None and max_total_attempts < 0:
            raise EvaluationError("max_total_attempts cannot be negative")
        self.max_total_attempts = max_total_attempts
        with store.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_jobs (job_id TEXT PRIMARY KEY, comparison TEXT NOT NULL, status TEXT NOT NULL, report_ref TEXT, error TEXT, runtime_accounting_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL)"
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(evaluation_jobs)")}
            if "runtime_accounting_json" not in columns:
                conn.execute("ALTER TABLE evaluation_jobs ADD COLUMN runtime_accounting_json TEXT NOT NULL DEFAULT '{}'")
            conn.commit()

    def planned_workload(self, candidate_count: int = 1, *, training_runs: int | None = None, transfer_runs: int = 0, safety_runs: int = 0, retries: int = 0):
        return self.protocol.workload(candidate_count, training_runs=training_runs, transfer_runs=transfer_runs, safety_runs=safety_runs, retries=retries)

    def _preflight(self, comparison: str, workload) -> None:
        if comparison not in {"validation", "final"}:
            raise EvaluationError("comparison must be validation or final")
        required = (Arm.B0, Arm.L) if comparison == "validation" else (Arm.B0, Arm.L, Arm.A)
        missing = [arm.value for arm in required if arm not in self.arm_bundles and arm.value not in self.arm_bundles]
        if missing:
            raise EvaluationError(f"missing expected arm bundles before evaluation: {', '.join(missing)}")
        if workload.validation_per_candidate != self.protocol.validation_run_count or workload.final_runs != self.protocol.final_run_count:
            raise EvaluationError("workload panel counts do not match the frozen protocol")
        if self.max_total_attempts is not None and workload.total_attempted_runs > self.max_total_attempts:
            raise EvaluationError("overall evaluation workload exhausted")
        if self.total_budget_microunits is not None:
            if self.total_budget_microunits < 0:
                raise EvaluationError("total budget cannot be negative")
            required_cost = workload.total_attempted_runs * self.protocol.run_budget.cost_microunits
            if required_cost > self.total_budget_microunits:
                raise EvaluationError("evaluation budget cannot cover the immutable panel")

    def _save(self, job_id: str, comparison: str, status: str, report: EvaluationReport | None = None, error: str | None = None, runtime_accounting: Mapping[str, Any] | None = None) -> None:
        report_ref = None
        if report is not None:
            report_ref = self.store.put_artifact(report.to_dict()).sha256
        accounting_json = json.dumps(dict(runtime_accounting or {}), sort_keys=True)
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO evaluation_jobs(job_id, comparison, status, report_ref, error, runtime_accounting_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now')) ON CONFLICT(job_id) DO UPDATE SET comparison=excluded.comparison, status=excluded.status, report_ref=excluded.report_ref, error=excluded.error, runtime_accounting_json=excluded.runtime_accounting_json, updated_at=excluded.updated_at",
                (job_id, comparison, status, report_ref, error, accounting_json),
            )
            conn.commit()

    def readback(self, job_id: str) -> EvaluationJobResult | None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM evaluation_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        report = None
        if row["report_ref"]:
            payload = self.store.get_artifact(row["report_ref"])
            if isinstance(payload, dict):
                report = payload  # serialized readback is intentionally opaque
        accounting = json.loads(row["runtime_accounting_json"] or "{}") if "runtime_accounting_json" in row.keys() else {}
        return EvaluationJobResult(job_id, row["comparison"], row["status"], report, error=row["error"], runtime_accounting=accounting)

    def run(self, job_id: str, comparison: str, *, base_hash: str, candidate_hash: str, candidate_id: str | None = None, ablation: AblationInput | None = None, candidate_count: int = 1, training_runs: int | None = None, transfer_runs: int = 0, safety_runs: int = 0, retries: int = 0) -> EvaluationJobResult:
        workload = self.planned_workload(candidate_count, training_runs=training_runs, transfer_runs=transfer_runs, safety_runs=safety_runs, retries=retries)
        self._preflight(comparison, workload)
        if comparison == "final" and ablation is None:
            raise EvaluationError("final evaluation requires pinned ablation input")
        frozen = self.protocol.start_candidate_generation()
        self._save(job_id, comparison, "running")
        driver = ResumableEvaluationDriver(
            self.store,
            self.protocol,
            self.packages,
            self.execute,
            self.arm_bundles.get(Arm.B0, self.arm_bundles.get(Arm.B0.value)),
            evidence_store=SQLiteRunEvidenceStore(self.store),
            arm_bundles=self.arm_bundles,
        )
        report: EvaluationReport | None = None
        observations: tuple[RunObservation, ...] = ()
        started = time.monotonic()
        try:
            summary = driver.run(job_id, Partition.VALIDATION if comparison == "validation" else Partition.FINAL, base_hash=base_hash, candidate_hash=candidate_hash)
            observations = tuple(item.observation for item in summary.statuses if item.observation is not None)
            evaluator = build_durable_evaluation_runner(self.protocol, self.packages, self.store, probe_executor=self.controller)
            report = evaluator.report_from_observations(
                comparison=comparison,
                base_hash=base_hash,
                candidate_hash=candidate_hash,
                observations=observations,
                expected_partitions=frozen.partition_hashes,
                ablation_audit=None if comparison == "validation" else __import__("adaptive_agent.evaluation", fromlist=["audit_ablation"]).audit_ablation(ablation) if ablation is not None else None,
            )
            accounting = self._runtime_accounting(observations, time.monotonic() - started)
            report = replace(
                report,
                actual_input_tokens=accounting["inputTokens"],
                actual_output_tokens=accounting["outputTokens"],
                nominal_cost_usd=accounting["nominalCostUsd"],
                wall_duration_seconds=accounting["wallDurationSeconds"],
                billing_basis=accounting["billingBasis"],
            )
            if summary.complete and report.validity_status == "valid" and report.promotion_eligible and candidate_id is not None:
                report.require_promotion_evidence(self.protocol, self.packages)
                decision = self.controller.candidates.promote(candidate_id, report)
                self._save(job_id, comparison, "decided", report, runtime_accounting=accounting)
                return EvaluationJobResult(job_id, comparison, "decided", report, decision, runtime_accounting=accounting)
            status = "complete" if report.validity_status == "valid" else "incomplete"
            self._save(job_id, comparison, status, report, runtime_accounting=accounting)
            return EvaluationJobResult(job_id, comparison, status, report, runtime_accounting=accounting)
        except Exception as exc:
            accounting = self._runtime_accounting(observations, time.monotonic() - started)
            self._save(job_id, comparison, "failed", report=report, error=str(exc), runtime_accounting=accounting)
            return EvaluationJobResult(job_id, comparison, "failed", report, error=str(exc), runtime_accounting=accounting)

    def _runtime_accounting(self, observations: tuple[RunObservation, ...], wall_seconds: float) -> dict[str, Any]:
        input_tokens = output_tokens = 0
        nominal_cost = 0.0
        nominal_seen = False
        economic_cost_microunits = 0.0
        economic_cost_seen = False
        economic_statuses: set[str] = set()
        for observation in observations:
            if not observation.accounting_ref:
                continue
            accounting = self.store.get_artifact(observation.accounting_ref)
            usage = {}
            if isinstance(accounting, dict):
                candidate_usage = accounting.get("aggregateUsage", accounting.get("usage", {}))
                if isinstance(candidate_usage, dict):
                    usage = candidate_usage
            input_tokens += int(usage.get("inputTokens", 0) or 0)
            output_tokens += int(usage.get("outputTokens", 0) or 0)
            if isinstance(accounting, dict):
                economic = accounting.get("economicCost")
                if isinstance(economic, dict):
                    status = economic.get("status")
                    if isinstance(status, str):
                        economic_statuses.add(status)
                    value = economic.get("microunits")
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        economic_cost_microunits += float(value)
                        economic_cost_seen = True
            evidence = self.store.get_evidence(observation.evidence_ref) if observation.evidence_ref else None
            if not evidence:
                continue
            try:
                source = json.loads(evidence["source_ref"])
                response = self.store.get_artifact(source["sha256"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                response = None
            if isinstance(response, dict):
                value = response.get("nominalCostUsd")
                usage_cost = response.get("usage", {}).get("cost", {}) if isinstance(response.get("usage"), dict) else {}
                if value is None and isinstance(usage_cost, dict):
                    value = usage_cost.get("total")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    nominal_cost += float(value)
                    nominal_seen = True
        if nominal_seen:
            billing_basis = "SDK nominal usage cost; subscription billing is separate and unmeasured"
        elif economic_statuses:
            billing_basis = f"SDK economic cost status: {', '.join(sorted(economic_statuses))}; nominal USD cost unavailable"
        else:
            billing_basis = "SDK nominal usage cost unavailable; subscription billing is separate and unmeasured"
        return {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
            "nominalCostUsd": nominal_cost if nominal_seen else None,
            "economicCostMicrounits": economic_cost_microunits if economic_cost_seen else None,
            "economicCostStatuses": sorted(economic_statuses),
            "wallDurationSeconds": wall_seconds,
            "billingBasis": billing_basis,
        }

    def run_development_smoke(self, job_id: str):
        """Run the one-task trusted development receipt used by held-out gates."""
        driver = ResumableEvaluationDriver(
            self.store,
            self.protocol,
            self.packages,
            self.execute,
            self.arm_bundles.get(Arm.B0, self.arm_bundles.get(Arm.B0.value)),
            evidence_store=SQLiteRunEvidenceStore(self.store),
            arm_bundles=self.arm_bundles,
        )
        return driver.run_development_smoke(job_id)


def build_evaluation_job(store: Store, controller: Controller, protocol: EvaluationProtocol, packages: Mapping[str, EnvironmentPackage], arm_bundles: Mapping[Arm | str, object], execute: TrustedTaskExecutor, *, total_budget_microunits: int | None = None, max_total_attempts: int | None = None) -> EvaluationJob:
    """Concrete factory binding the trusted executor to the production job."""
    return EvaluationJob(store, controller, protocol, packages, arm_bundles, execute, total_budget_microunits=total_budget_microunits, max_total_attempts=max_total_attempts)


def run_evaluation_job(job: EvaluationJob, job_id: str, comparison: str, *, base_hash: str, candidate_hash: str, candidate_id: str | None = None, ablation: AblationInput | None = None, candidate_count: int = 1, training_runs: int | None = None, transfer_runs: int = 0, safety_runs: int = 0, retries: int = 0) -> EvaluationJobResult:
    return job.run(job_id, comparison, base_hash=base_hash, candidate_hash=candidate_hash, candidate_id=candidate_id, ablation=ablation, candidate_count=candidate_count, training_runs=training_runs, transfer_runs=transfer_runs, safety_runs=safety_runs, retries=retries)


def main(argv: list[str] | None = None) -> int:
    """Run a production job from an application-owned factory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--comparison", choices=("validation", "final"))
    parser.add_argument("--base-hash", default="")
    parser.add_argument("--candidate-hash", default="")
    parser.add_argument("--candidate-id")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    module_name, separator, function_name = args.factory.partition(":")
    if not separator:
        parser.error("--factory must be module:callable")
    factory = getattr(importlib.import_module(module_name), function_name)
    job = factory(Store(args.store))
    if args.smoke:
        result = job.run_development_smoke(args.job)
        print(result)
        return 0 if result.complete else 1
    if args.comparison is None:
        parser.error("--comparison is required unless --smoke is used")
    result = run_evaluation_job(job, args.job, args.comparison, base_hash=args.base_hash, candidate_hash=args.candidate_hash, candidate_id=args.candidate_id)
    print(result.status)
    return 0 if result.status in {"complete", "decided"} else 1


__all__ = ["EvaluationJob", "EvaluationJobResult", "build_evaluation_job", "main", "run_evaluation_job"]


if __name__ == "__main__":
    raise SystemExit(main())
