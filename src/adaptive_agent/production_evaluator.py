"""Runnable production evaluator entry point.

The integrated application runtime is imported inside ``build_job`` so this
module remains an evaluator-owned boundary.  Deployment supplies the runtime
package on ``PYTHONPATH``; this launcher never creates a synthetic executor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_TOKENS = 20_000
WALL_SECONDS = 90


def _bundle_by_hash(store: Any, content_hash: str) -> Any:
    from adaptive_agent.models import SkillBundle

    row = store.get_bundle_by_hash(content_hash)
    if row is None:
        raise RuntimeError(f"bundle is not present in durable store: {content_hash}")
    return SkillBundle.model_validate(json.loads(row["bundle_json"]))


def _candidate_bundle(store: Any, candidate_id: str) -> Any:
    row = store.get_candidate(candidate_id)
    if row is None:
        raise RuntimeError(f"candidate is not present in durable store: {candidate_id}")
    candidate_hash = row.get("candidate_bundle_hash", row.get("candidateBundleHash"))
    if not isinstance(candidate_hash, str) or not candidate_hash:
        raise RuntimeError(f"candidate has no durable bundle hash: {candidate_id}")
    return _bundle_by_hash(store, candidate_hash)


def _import_candidate(source_dir: str, target_store: Any, candidate_id: str) -> None:
    """Copy only the bound candidate/base bundle records into a clean Store."""
    from adaptive_agent.store import Store

    source = Store(source_dir)
    candidate = source.get_candidate(candidate_id)
    if candidate is None:
        raise RuntimeError(f"candidate is not present in source store: {candidate_id}")
    base_hash = candidate.get("base_bundle_hash", candidate.get("baseBundleHash"))
    candidate_hash = candidate.get("candidate_bundle_hash", candidate.get("candidateBundleHash"))
    if not isinstance(base_hash, str) or not isinstance(candidate_hash, str):
        raise RuntimeError("candidate source record lacks base/candidate bundle hashes")
    active = target_store.get_active_bundle()
    if active is None or active.get("content_hash") != base_hash:
        raise RuntimeError("clean Store base bundle does not match candidate base hash")
    bundle = source.get_bundle_by_hash(candidate_hash)
    if bundle is None:
        raise RuntimeError(f"candidate bundle is not present in source store: {candidate_hash}")
    target_store.save_bundle(
        bundle["bundle_id"], bundle.get("parent"), bundle["content_hash"], bundle["bundle_json"], False
    )
    target_store.save_candidate(candidate_id, candidate)


def _require_bound_real_receipt(source_dir: str, run_id: str) -> None:
    """Require the supplied run to have canonical trusted evidence already."""
    from adaptive_agent.store import Store

    source = Store(source_dir)
    run = source.get_run(run_id)
    if run is None or run.get("status") != "succeeded":
        raise RuntimeError(f"bound source run is not a succeeded durable run: {run_id}")
    evidence = source.list_evidence(run_id)
    model = [row for row in evidence if row.get("event_type") == "model_response" and row.get("visibility") == "operator"]
    # Older durable runs keep evaluator attestations evaluator-only so they
    # never enter the operator/SSE projection.  They are still canonical
    # evaluator receipts and may seed a fresh production Store; newly created
    # benchmark runs are verified by SQLiteRunEvidenceStore independently.
    trusted = [
        row
        for row in evidence
        if row.get("event_type") == "trusted_outcome"
        and row.get("visibility") in {"operator", "evaluator_only"}
    ]
    if not model or not trusted:
        raise RuntimeError(
            f"bound source run lacks canonical operator-visible model/trusted outcome evidence: {run_id}"
        )


def build_job(data_dir: str, source_data_dir: str, candidate_id: str, source_run_id: str, *, a_hash: str | None = None) -> tuple[Any, Any, Any]:
    """Build an evaluator job around the integrated durable runtime.

    The import is deliberately local: an evaluator process must resolve the
    deployed ``DurableRuntime`` implementation at execution time.
    """
    from adaptive_agent.app import create_runtime_app
    from adaptive_agent.evaluation import Arm, BudgetSpec, EvaluationProtocol

    target = Path(data_dir)
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"data directory must be a new empty directory: {data_dir}")
    _require_bound_real_receipt(source_data_dir, source_run_id)

    app = create_runtime_app(data_dir=data_dir)
    runtime = app.state.durable_runtime
    _import_candidate(source_data_dir, runtime.controller.store, candidate_id)
    evaluation_module = __import__("adaptive_agent.evaluation", fromlist=["__file__"])
    analysis_hash = hashlib.sha256(Path(evaluation_module.__file__).read_bytes()).hexdigest()
    protocol = EvaluationProtocol(
        core_planner_hash=runtime.core_planner_hash,
        image_digest=runtime.image_digest,
        analysis_code_hash=analysis_hash,
        run_budget=BudgetSpec(model_tokens=MODEL_TOKENS, wall_time_seconds=WALL_SECONDS),
    )
    protocol.freeze(runtime.packages)
    active = runtime.controller.get_active_bundle()
    if active is None:
        raise RuntimeError("durable runtime has no active base bundle")
    learned = _candidate_bundle(runtime.controller.store, candidate_id)
    bundles: dict[Any, Any] = {Arm.B0: active, Arm.L: learned}
    if a_hash is not None:
        bundles[Arm.A] = _bundle_by_hash(runtime.controller.store, a_hash)
    job = runtime.build_evaluation_job(protocol, bundles)
    return app, protocol, job


def _workload(protocol: Any, candidate_count: int, training: int, transfer: int, safety: int, retries: int) -> dict[str, Any]:
    plan = protocol.workload(candidate_count, training_runs=training, transfer_runs=transfer, safety_runs=safety, retries=retries)
    return plan.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the durable Luna evaluation entry point")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--source-data-dir", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--comparison", choices=("smoke", "validation", "final"), required=True)
    parser.add_argument("--base-hash", default="")
    parser.add_argument("--candidate-hash", default="")
    parser.add_argument("--a-hash")
    parser.add_argument("--candidate-count", type=int, required=True)
    parser.add_argument("--training-runs", type=int, required=True)
    parser.add_argument("--transfer-runs", type=int, required=True)
    parser.add_argument("--safety-runs", type=int, required=True)
    parser.add_argument("--retries", type=int, required=True)
    parser.add_argument("--ablation-system-instructions")
    parser.add_argument("--ablation-retrieval-input", action="append", default=[])
    args = parser.parse_args(argv)

    app, protocol, job = build_job(args.data_dir, args.source_data_dir, args.candidate_id, args.source_run_id, a_hash=args.a_hash)
    workload = _workload(protocol, args.candidate_count, args.training_runs, args.transfer_runs, args.safety_runs, args.retries)
    print(json.dumps({"job": args.job, "comparison": args.comparison, "workload": workload}, sort_keys=True))
    if workload["transferRuns"] < 1:
        parser.error("--transfer-runs must be positive; EVAL-002 requires a real transfer workload")
    if args.comparison != "smoke" and args.training_runs < 1:
        parser.error("--training-runs must be positive for the complete frozen workload")
    if args.comparison == "smoke":
        result = job.run_development_smoke(args.job)
        print(json.dumps({"complete": result.complete, "expected": result.expected_count, "statuses": [item.status for item in result.statuses]}))
        return 0 if result.complete else 1
    if not args.base_hash or not args.candidate_hash:
        parser.error("--base-hash and --candidate-hash are required for held-out panels")
    if args.comparison == "final":
        if args.a_hash is None:
            parser.error("--a-hash is required for the final B0/L/A panel")
        if not args.ablation_system_instructions or not args.ablation_retrieval_input:
            parser.error("final evaluation requires explicit ablation instructions and retrieval inputs")
        from adaptive_agent.evaluation import AblationInput, audit_ablation

        ablation = AblationInput(args.a_hash, args.ablation_system_instructions, tuple(args.ablation_retrieval_input))
        audit = audit_ablation(ablation)
        if not audit.passed:
            parser.error(f"ablation audit failed: {audit.retained_learned_artifacts}")
    else:
        ablation = None
    result = job.run(args.job, args.comparison, base_hash=args.base_hash, candidate_hash=args.candidate_hash, candidate_id=None, ablation=ablation, training_runs=args.training_runs, transfer_runs=args.transfer_runs, safety_runs=args.safety_runs, retries=args.retries)
    print(json.dumps({"status": result.status, "error": result.error, "runtimeAccounting": result.runtime_accounting}, sort_keys=True, default=str))
    return 0 if result.status in {"complete", "decided"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
