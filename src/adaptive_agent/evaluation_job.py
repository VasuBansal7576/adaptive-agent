"""Production evaluation-job orchestration.

This module owns the boundary between durable runtime execution and evaluator
scoring.  The driver is the only component allowed to select or execute a
panel; ``EvaluationRunner`` receives the driver's persisted observations only.
"""
from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
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
    ) -> None:
        self.store = store
        self.controller = controller
        self.protocol = protocol
        self.packages = dict(packages)
        self.arm_bundles = dict(arm_bundles)
        self.execute = execute
        self.total_budget_microunits = total_budget_microunits
        with store.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS evaluation_jobs (job_id TEXT PRIMARY KEY, comparison TEXT NOT NULL, status TEXT NOT NULL, report_ref TEXT, error TEXT, updated_at TEXT NOT NULL)"
            )
            conn.commit()

    def _preflight(self, comparison: str) -> None:
        if comparison not in {"validation", "final"}:
            raise EvaluationError("comparison must be validation or final")
        required = (Arm.B0, Arm.L) if comparison == "validation" else (Arm.B0, Arm.L, Arm.A)
        missing = [arm.value for arm in required if arm not in self.arm_bundles and arm.value not in self.arm_bundles]
        if missing:
            raise EvaluationError(f"missing expected arm bundles before evaluation: {', '.join(missing)}")
        expected = self.protocol.validation_run_count if comparison == "validation" else self.protocol.final_run_count
        if self.total_budget_microunits is not None:
            if self.total_budget_microunits < 0:
                raise EvaluationError("total budget cannot be negative")
            required_cost = expected * self.protocol.run_budget.cost_microunits
            if required_cost > self.total_budget_microunits:
                raise EvaluationError("evaluation budget cannot cover the immutable panel")

    def _save(self, job_id: str, comparison: str, status: str, report: EvaluationReport | None = None, error: str | None = None) -> None:
        report_ref = None
        if report is not None:
            report_ref = self.store.put_artifact(report.to_dict()).sha256
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO evaluation_jobs(job_id, comparison, status, report_ref, error, updated_at) VALUES (?, ?, ?, ?, ?, datetime('now')) ON CONFLICT(job_id) DO UPDATE SET comparison=excluded.comparison, status=excluded.status, report_ref=excluded.report_ref, error=excluded.error, updated_at=excluded.updated_at",
                (job_id, comparison, status, report_ref, error),
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
        return EvaluationJobResult(job_id, row["comparison"], row["status"], report, error=row["error"])

    def run(self, job_id: str, comparison: str, *, base_hash: str, candidate_hash: str, candidate_id: str | None = None, ablation: AblationInput | None = None) -> EvaluationJobResult:
        self._preflight(comparison)
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
            if summary.complete and report.validity_status == "valid" and report.promotion_eligible and candidate_id is not None:
                report.require_promotion_evidence(self.protocol, self.packages)
                decision = self.controller.candidates.promote(candidate_id, report)
                self._save(job_id, comparison, "decided", report)
                return EvaluationJobResult(job_id, comparison, "decided", report, decision)
            self._save(job_id, comparison, "complete" if report.validity_status == "valid" else "incomplete", report)
            return EvaluationJobResult(job_id, comparison, "complete" if report.validity_status == "valid" else "incomplete", report)
        except Exception as exc:
            self._save(job_id, comparison, "failed", report=report, error=str(exc))
            return EvaluationJobResult(job_id, comparison, "failed", report, error=str(exc))

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


def build_evaluation_job(store: Store, controller: Controller, protocol: EvaluationProtocol, packages: Mapping[str, EnvironmentPackage], arm_bundles: Mapping[Arm | str, object], execute: TrustedTaskExecutor, *, total_budget_microunits: int | None = None) -> EvaluationJob:
    """Concrete factory binding the trusted executor to the production job."""
    return EvaluationJob(store, controller, protocol, packages, arm_bundles, execute, total_budget_microunits=total_budget_microunits)


def run_evaluation_job(job: EvaluationJob, job_id: str, comparison: str, *, base_hash: str, candidate_hash: str, candidate_id: str | None = None, ablation: AblationInput | None = None) -> EvaluationJobResult:
    return job.run(job_id, comparison, base_hash=base_hash, candidate_hash=candidate_hash, candidate_id=candidate_id, ablation=ablation)


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
