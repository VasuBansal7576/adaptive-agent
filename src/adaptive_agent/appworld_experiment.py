"""Runnable, pinned AppWorld train/learn/evaluate experiment.

This command is intentionally a thin orchestration layer.  DurableRuntime
owns model execution, LearningRuntime owns proposal generation, and the
benchmark runner owns resumable panel state and aggregate-only reporting.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from adaptive_agent.app import create_runtime_app
from adaptive_agent.appworld_benchmark import AppWorldProtocol, create_appworld_benchmark_runner
from adaptive_agent.appworld_provider import AppWorldPackage
from adaptive_agent.evaluation import Arm, BudgetSpec
from adaptive_agent.learning_runtime import LearningRuntime
from adaptive_agent.planner import PrimeCliModelClient

MANIFEST_NAME = "appworld-experiment-manifest.json"
STATE_NAME = "appworld-experiment-state.json"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _protocol(package: AppWorldPackage, args: argparse.Namespace, split: str, count: int) -> AppWorldProtocol:
    return AppWorldProtocol.freeze(
        package,
        model_profile=args.model_profile,
        core_planner_hash=args.core_planner_hash,
        official_split=split,
        image_digest=args.image_digest,
        source_revision=args.source_revision,
        dataset_content_hash=args.dataset_content_hash or package.catalog.dataset_hash(),
        published_count=count,
        sampling_seed=args.sampling_seed,
        seeds=(args.seed,),
        budget=BudgetSpec(model_tokens=args.model_tokens, wall_time_seconds=args.wall_seconds, cost_microunits=args.cost_microunits),
    )


def _manifest(args: argparse.Namespace, protocols: Sequence[AppWorldProtocol]) -> dict[str, Any]:
    return {"experiment": "appworld-train-learn-evaluate", "protocols": {p.official_split: p.to_dict() for p in protocols}, "pins": {"sourceRevision": args.source_revision, "imageDigest": args.image_digest, "provider": "openai-codex", "modelProfile": args.model_profile, "corePlannerHash": args.core_planner_hash, "datasetContentHash": protocols[0].dataset_content_hash, "budget": protocols[0].budget.to_dict()}}


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / MANIFEST_NAME
    if args.initialize and manifest_path.exists():
        raise RuntimeError("experiment is already initialized; use --resume")
    if args.initialize and data_dir.exists() and any(data_dir.iterdir()):
        raise RuntimeError("initialization requires a new empty data directory")
    if args.resume and not manifest_path.exists():
        raise RuntimeError("cannot resume an experiment without its immutable manifest")
    data_dir.mkdir(parents=True, exist_ok=True)
    client = PrimeCliModelClient(executable=args.prime_executable, coding_agent_dir=args.coding_agent_dir)
    app = create_runtime_app(model_runner=client, learning_model_client=client, data_dir=data_dir, appworld_root=args.appworld_root, appworld_python=args.appworld_python)
    runtime = app.state.durable_runtime
    package = runtime.packages.get("appworld")
    if not isinstance(package, AppWorldPackage):
        raise RuntimeError("AppWorld package was not registered")
    protocols = (_protocol(package, args, "train", 8), _protocol(package, args, "dev", 20), _protocol(package, args, "test_normal", 20))
    frozen_manifest = _manifest(args, protocols)
    if args.resume:
        if json.loads(manifest_path.read_text()) != frozen_manifest:
            raise RuntimeError("resume pins do not match the immutable experiment manifest")
    else:
        _json(manifest_path, frozen_manifest)
    active = runtime.controller.get_active_bundle()
    if active is None:
        raise RuntimeError("durable runtime has no active base bundle")
    base_hash = active.content_hash
    state_path = data_dir / STATE_NAME
    state = json.loads(state_path.read_text()) if args.resume and state_path.exists() else None
    if isinstance(state, dict):
        if state.get("baseBundleHash") != base_hash or not isinstance(state.get("learnedBundleHash"), str):
            raise RuntimeError("resume state is not bound to the active base bundle")
        source_ids = state.get("trainingRunIds", [])
        if not isinstance(source_ids, list) or len(source_ids) != 8 or any(not isinstance(run_id, str) or not run_id for run_id in source_ids):
            raise RuntimeError("resume state has an invalid training run set")
        learned_hash = state["learnedBundleHash"]
    else:
        bundles = {Arm.B0: base_hash, Arm.L: base_hash, Arm.A: base_hash}
        train_runner = create_appworld_benchmark_runner(runtime, package, protocols[0], bundles, data_dir / "train")
        train_runner.run("train", bundles, arms=(Arm.B0,))
        import sqlite3
        with sqlite3.connect(train_runner.db) as conn:
            rows = conn.execute("SELECT result_json FROM appworld_cells WHERE benchmark_id=? AND status='complete' ORDER BY task_id", ("train",)).fetchall()
        source_ids = [json.loads(row[0])["observation"]["run_id"] for row in rows]
        if len(source_ids) != 8:
            raise RuntimeError("training panel did not produce exactly eight durable runs")
        learning = LearningRuntime.build(store=runtime.controller.store, manager=runtime.controller.candidates, model_client=client)
        proposal = learning.propose_completed_runs(source_ids, primary_run_id=source_ids[0], goal="Learn reusable AppWorld procedures", feedback={"status": "completed"})
        candidate = proposal.authoritative_candidate
        learned_hash = candidate.get("candidateBundleHash") or candidate.get("candidate_bundle_hash")
        if not isinstance(learned_hash, str) or not learned_hash:
            raise RuntimeError("learning proposal did not produce a durable candidate bundle")
    bundles = {Arm.B0: base_hash, Arm.L: learned_hash, Arm.A: base_hash}
    reports = []
    for protocol, name in zip(protocols[1:], ("dev", "test_normal")):
        runner = create_appworld_benchmark_runner(runtime, package, protocol, bundles, data_dir / name)
        reports.append(runner.run(name, bundles).to_dict())
    state = {"trainingRunIds": source_ids, "baseBundleHash": base_hash, "learnedBundleHash": learned_hash, "reports": reports}
    _json(data_dir / STATE_NAME, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pinned AppWorld train/learn/evaluate experiment")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--resume", action="store_true")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--appworld-root", required=True)
    parser.add_argument("--appworld-python", default=os.environ.get("APPWORLD_PYTHON", ""))
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--core-planner-hash", required=True)
    parser.add_argument("--model-profile", default="openai-codex/gpt-5.6-luna")
    parser.add_argument("--dataset-content-hash")
    parser.add_argument("--sampling-seed", type=int, default=20260906)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-tokens", type=int, default=4096)
    parser.add_argument("--wall-seconds", type=float, default=90.0)
    parser.add_argument("--cost-microunits", type=int, default=100000)
    parser.add_argument("--prime-executable", default="prime-agent")
    parser.add_argument("--coding-agent-dir", default=os.environ.get("PRIME_AGENT_CODING_AGENT_DIR"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run_experiment(build_parser().parse_args(argv)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
