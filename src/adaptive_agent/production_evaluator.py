"""Runnable production evaluator entry point.

The integrated application runtime is imported inside ``build_job`` so this
module remains an evaluator-owned boundary.  Deployment supplies the runtime
package on ``PYTHONPATH``; this launcher never creates a synthetic executor.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any


DEFAULT_CANDIDATE = "cand_3afed3a310f64597a669fb1795e1445d"
DEFAULT_DATA_DIR = "/private/tmp/adaptive-agent-browser-api"
DEFAULT_MAX_ATTEMPTS = 20_000
DEFAULT_WALL_SECONDS = 90


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


def build_job(data_dir: str, candidate_id: str, *, a_hash: str | None = None, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> tuple[Any, Any, Any]:
    """Build an evaluator job around the integrated durable runtime.

    The import is deliberately local: an evaluator process must resolve the
    deployed ``DurableRuntime`` implementation at execution time.
    """
    from adaptive_agent.app import create_runtime_app
    from adaptive_agent.evaluation import Arm, EvaluationProtocol

    app = create_runtime_app(data_dir=data_dir)
    runtime = app.state.durable_runtime
    protocol = EvaluationProtocol()
    protocol.freeze(runtime.packages)
    active = runtime.controller.get_active_bundle()
    if active is None:
        raise RuntimeError("durable runtime has no active base bundle")
    learned = _candidate_bundle(runtime.controller.store, candidate_id)
    bundles: dict[Any, Any] = {Arm.B0: active, Arm.L: learned}
    if a_hash is not None:
        bundles[Arm.A] = _bundle_by_hash(runtime.controller.store, a_hash)
    job = runtime.build_evaluation_job(protocol, bundles)
    job.max_total_attempts = max_attempts
    return app, protocol, job


def _workload(protocol: Any, candidate_count: int, training: int, transfer: int, safety: int, retries: int) -> dict[str, Any]:
    plan = protocol.workload(candidate_count, training_runs=training, transfer_runs=transfer, safety_runs=safety, retries=retries)
    return plan.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the durable Luna evaluation entry point")
    parser.add_argument("--data-dir", default=os.environ.get("ADAPTIVE_AGENT_DATA", DEFAULT_DATA_DIR))
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    parser.add_argument("--job", required=True)
    parser.add_argument("--comparison", choices=("smoke", "validation", "final"), default="smoke")
    parser.add_argument("--base-hash", default="")
    parser.add_argument("--candidate-hash", default="")
    parser.add_argument("--a-hash")
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--training-runs", type=int, default=60)
    parser.add_argument("--transfer-runs", type=int, default=0)
    parser.add_argument("--safety-runs", type=int, default=2)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    args = parser.parse_args(argv)

    app, protocol, job = build_job(args.data_dir, args.candidate_id, a_hash=args.a_hash, max_attempts=args.max_attempts)
    workload = _workload(protocol, args.candidate_count, args.training_runs, args.transfer_runs, args.safety_runs, args.retries)
    print(json.dumps({"job": args.job, "comparison": args.comparison, "workload": workload}, sort_keys=True))
    if args.comparison == "smoke":
        result = job.run_development_smoke(args.job)
        print(json.dumps({"complete": result.complete, "expected": result.expected_count, "statuses": [item.status for item in result.statuses]}))
        return 0 if result.complete else 1
    if not args.base_hash or not args.candidate_hash:
        parser.error("--base-hash and --candidate-hash are required for held-out panels")
    if args.comparison == "final" and args.a_hash is None:
        parser.error("--a-hash is required for the final B0/L/A panel")
    result = job.run(args.job, args.comparison, base_hash=args.base_hash, candidate_hash=args.candidate_hash, candidate_id=None, training_runs=args.training_runs, transfer_runs=args.transfer_runs, safety_runs=args.safety_runs, retries=args.retries)
    print(json.dumps({"status": result.status, "error": result.error, "runtimeAccounting": result.runtime_accounting}, sort_keys=True, default=str))
    return 0 if result.status in {"complete", "decided"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
