"""Runnable, pinned AppWorld train/learn/evaluate experiment."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from adaptive_agent.app import create_runtime_app
from adaptive_agent.appworld_benchmark import AppWorldProtocol, create_appworld_benchmark_runner
from adaptive_agent.appworld_provider import AppWorldPackage
from adaptive_agent.evaluation import Arm, BudgetSpec
from adaptive_agent.learning_runtime import LearningRuntime
from adaptive_agent.planner import MODEL_NAME, MODEL_PROVIDER, PrimeCliModelClient

MANIFEST_NAME = "appworld-experiment-manifest.json"
STATE_NAME = "appworld-experiment-state.json"
MODEL_TOKENS = 20_000
TOOL_CALLS = 32
WALL_SECONDS = 90


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _protocol(package: AppWorldPackage, args: argparse.Namespace, split: str, count: int) -> AppWorldProtocol:
    if args.model_tokens != MODEL_TOKENS:
        raise ValueError(f"AppWorld experiment requires {MODEL_TOKENS} model tokens")
    return AppWorldProtocol.freeze(package, model_profile=args.model_profile, core_planner_hash=args.core_planner_hash, official_split=split, image_digest=args.image_digest, source_revision=args.source_revision, dataset_content_hash=args.dataset_content_hash or package.catalog.dataset_hash(), published_count=count, sampling_seed=args.sampling_seed, seeds=(args.seed,), budget=BudgetSpec(model_tokens=args.model_tokens, tool_calls=TOOL_CALLS, wall_time_seconds=WALL_SECONDS, cost_microunits=args.cost_microunits))


def _validate_runtime(runtime: Any, args: argparse.Namespace) -> None:
    if args.model_profile != MODEL_NAME:
        raise RuntimeError(f"unsupported AppWorld model profile: {args.model_profile}")
    source = os.environ.get("ADAPTIVE_AGENT_SOURCE_REVISION")
    if not source or source != args.source_revision:
        raise RuntimeError("ADAPTIVE_AGENT_SOURCE_REVISION must match --source-revision")
    if getattr(runtime, "image_digest", None) != args.image_digest or not args.image_digest.startswith("sha256:"):
        raise RuntimeError("runtime image digest does not match --image-digest")
    if getattr(runtime, "core_planner_hash", None) != args.core_planner_hash:
        raise RuntimeError("runtime core planner hash does not match --core-planner-hash")
    if getattr(runtime, "provider", MODEL_PROVIDER) != MODEL_PROVIDER or getattr(runtime, "model_profile", MODEL_NAME) != MODEL_NAME:
        raise RuntimeError("runtime provider or model profile is not the pinned Luna configuration")
    if MODEL_PROVIDER != "openai-codex":
        raise RuntimeError("unsupported model provider")
    runtime.source_revision = source
    runtime.provider = MODEL_PROVIDER
    runtime.model_profile = MODEL_NAME


def _manifest(args: argparse.Namespace, protocols: Sequence[AppWorldProtocol], base_hash: str) -> dict[str, Any]:
    return {"experiment": "appworld-train-learn-evaluate", "baseBundleHash": base_hash, "protocols": {p.official_split: p.to_dict() for p in protocols}, "pins": {"sourceRevision": args.source_revision, "imageDigest": args.image_digest, "provider": MODEL_PROVIDER, "modelProfile": args.model_profile, "corePlannerHash": args.core_planner_hash, "datasetContentHash": protocols[0].dataset_content_hash, "budget": protocols[0].budget.to_dict()}}


def _candidate_binding(runtime: Any, learned_hash: str, base_hash: str, source_ids: Sequence[str], expected_receipts: Sequence[str] | None = None) -> list[str]:
    store = runtime.controller.store
    bundle = store.get_bundle_by_hash(learned_hash)
    candidate = store.get_candidate_by_bundle_hash(learned_hash)
    if not isinstance(bundle, Mapping) or not isinstance(candidate, Mapping) or candidate.get("base_bundle_hash") != base_hash:
        raise RuntimeError("learned candidate is not bound to the frozen base bundle")
    from adaptive_agent.models import SkillBundle, sha256_json
    try:
        bundle_value = json.loads(bundle["bundle_json"]) if isinstance(bundle.get("bundle_json"), str) else bundle["bundle_json"]
        parsed_bundle = SkillBundle.model_validate(bundle_value)
        expected_hash = sha256_json(parsed_bundle.model_dump(mode="json", by_alias=True, exclude={"content_hash"}))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("learned candidate bundle receipt is malformed") from exc
    if parsed_bundle.content_hash != learned_hash or expected_hash != learned_hash:
        raise RuntimeError("learned candidate bundle hash does not validate")
    try:
        proposal = json.loads(candidate["candidate_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("learned candidate receipt is malformed") from exc
    evidence_ids = proposal.get("supportingEvidenceIds", proposal.get("supporting_evidence_ids", ()))
    if not isinstance(evidence_ids, list) or not evidence_ids:
        raise RuntimeError("learned candidate evidence binding is malformed")
    allowed = set(source_ids)
    for evidence_id in evidence_ids:
        provenance = store.evidence_provenance(evidence_id)
        if not isinstance(provenance, Mapping) or provenance.get("run_id") not in allowed:
            raise RuntimeError("learned candidate evidence is outside the frozen training source set")
    learner_rows = [row for run_id in source_ids for row in store.list_evidence(run_id) if row.get("event_type") == "learning_model_observation"]
    if not learner_rows:
        raise RuntimeError("learned candidate lacks an authenticated learner receipt")
    receipt_bindings: list[str] = []
    for row in learner_rows:
        try:
            source_ref = json.loads(row["source_ref"])
            artifact_hash = source_ref["sha256"]
            artifact = store.get_artifact(artifact_hash)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("learner receipt CAS binding is malformed") from exc
        expected_response = str(row.get("evidence_id", "")).removeprefix("learning-model-")
        if not isinstance(artifact, Mapping) or row.get("content_hash") != artifact_hash or artifact.get("responseId") != expected_response:
            raise RuntimeError("learner receipt CAS binding failed")
        usage = artifact.get("usage")
        if not isinstance(usage, Mapping) or any(key not in usage for key in ("inputTokens", "outputTokens", "totalTokens")):
            raise RuntimeError("learner receipt usage binding is incomplete")
        receipt_bindings.append(f"{row['evidence_id']}:{artifact_hash}")
    if expected_receipts is not None and list(expected_receipts) != receipt_bindings:
        raise RuntimeError("learner receipt binding changed after learning")
    return receipt_bindings


def _derive_ablation(runtime: Any, learned_hash: str) -> str:
    """Persist the production-style L-derived, memory-disabled A bundle."""
    from adaptive_agent.models import SkillBundle, sha256_json

    store = runtime.controller.store
    learned_row = store.get_bundle_by_hash(learned_hash)
    if not isinstance(learned_row, Mapping):
        raise RuntimeError("learned bundle is missing")
    raw = learned_row.get("bundle_json")
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, Mapping):
        raise RuntimeError("learned bundle payload is malformed")
    learned = SkillBundle.model_validate(payload)
    if learned.content_hash != learned_hash or sha256_json(learned.model_dump(mode="json", by_alias=True, exclude={"content_hash"})) != learned_hash:
        raise RuntimeError("learned bundle hash does not validate")
    ablation_payload = learned.model_dump(mode="json", by_alias=True)
    ablation_payload["bundle_id"] = f"{learned.bundle_id}:memory-disabled"
    ablation_payload["parent"] = learned.content_hash
    ablation_payload["skills"] = []
    config = dict(ablation_payload.get("executionConfig", {}))
    config["skill_refs"] = []
    config["instruction_variant"] = "default"
    ablation_payload["executionConfig"] = config
    ablation_payload["contentHash"] = ""
    ablation = SkillBundle.model_validate(ablation_payload)
    existing = store.get_bundle(ablation.bundle_id)
    if existing is None:
        store.save_bundle(ablation.bundle_id, ablation.parent, ablation.content_hash, ablation.model_dump_json(by_alias=True), False)
    elif existing.get("content_hash") != ablation.content_hash:
        raise RuntimeError("memory-disabled bundle binding conflicts with durable state")
    return ablation.content_hash


def _training_ids(runner: Any) -> list[str]:
    with sqlite3.connect(runner.db) as conn:
        rows = conn.execute("SELECT result_json FROM appworld_cells WHERE benchmark_id=? AND status='complete' ORDER BY task_id", ("train",)).fetchall()
    return [json.loads(row[0])["observation"]["run_id"] for row in rows]


def _read_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("experiment state is malformed")
    return value


def run_experiment(args: argparse.Namespace, *, progress: Any | None = None) -> dict[str, Any]:
    def emit(stage: str, status: str, **details: Any) -> None:
        if callable(progress):
            progress({"stage": stage, "status": status, **details})

    data_dir = Path(args.data_dir)
    manifest_path, state_path = data_dir / MANIFEST_NAME, data_dir / STATE_NAME
    if args.initialize and (manifest_path.exists() or (data_dir.exists() and any(data_dir.iterdir()))):
        raise RuntimeError("initialization requires a new empty data directory")
    if args.resume and not manifest_path.exists():
        raise RuntimeError("cannot resume without an immutable experiment manifest")
    data_dir.mkdir(parents=True, exist_ok=True)
    client = PrimeCliModelClient(executable=args.prime_executable, coding_agent_dir=args.coding_agent_dir)
    app = create_runtime_app(model_runner=client, learning_model_client=client, data_dir=data_dir, appworld_root=args.appworld_root, appworld_python=args.appworld_python)
    runtime = app.state.durable_runtime
    _validate_runtime(runtime, args)
    package = runtime.packages.get("appworld")
    if not isinstance(package, AppWorldPackage):
        raise RuntimeError("AppWorld package was not registered")
    active = runtime.controller.get_active_bundle()
    if active is None:
        raise RuntimeError("durable runtime has no active base bundle")
    base_hash = active.content_hash
    protocols = (_protocol(package, args, "train", 8), _protocol(package, args, "dev", 20), _protocol(package, args, "test_normal", 20))
    manifest = _manifest(args, protocols, base_hash)
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("resume pins or base bundle do not match the immutable manifest")
    else:
        _atomic_json(manifest_path, manifest)
    current = _read_state(state_path)
    if current and (current.get("baseBundleHash") != base_hash or current.get("manifestHash") != protocols[0].protocol_hash):
        raise RuntimeError("experiment state is not bound to the frozen base or manifest")
    if current and current.get("trainingStatus") == "failed":
        raise RuntimeError("training previously failed; refusing a budget-expanding retry")
    if current and current.get("learningStatus") == "failed":
        raise RuntimeError("learning previously failed; refusing a redispatch")
    if current and current.get("learningStatus") == "in_flight":
        raise RuntimeError("learning was interrupted without an authenticated recoverable receipt")
    source_ids = current.get("trainingRunIds") if current else None
    learned_hash = current.get("learnedBundleHash") if current else None
    ablation_hash = current.get("ablationBundleHash") if current else None
    receipt_bindings = current.get("learningReceiptBindings") if current else None
    if not source_ids:
        state = {"manifestHash": protocols[0].protocol_hash, "baseBundleHash": base_hash, "trainingRunIds": [], "trainingStatus": "running", "learningStatus": "pending", "devStatus": "pending", "finalStatus": "pending"}
        _atomic_json(state_path, state)
        bundles = {Arm.B0: base_hash, Arm.L: base_hash, Arm.A: base_hash}
        train_runner = create_appworld_benchmark_runner(runtime, package, protocols[0], bundles, data_dir / "train")
        try:
            emit("train", "running", expected=8, arms=[Arm.B0.value])
            train_runner.run("train", bundles, arms=(Arm.B0,))
        except Exception as exc:
            with sqlite3.connect(train_runner.db) as conn:
                result_rows = [row[0] for row in conn.execute("SELECT result_json FROM appworld_cells WHERE benchmark_id=?", ("train",))]
            run_ids = []
            for result_json in result_rows:
                try:
                    run_ids.append(json.loads(result_json)["observation"]["run_id"])
                except (TypeError, KeyError, json.JSONDecodeError):
                    continue
            terminal_failure = any(
                isinstance((run := runtime.controller.store.get_run(run_id)), Mapping)
                and run.get("status") in {"failed", "timed_out", "outcome_unknown"}
                for run_id in run_ids
            )
            if terminal_failure:
                failed_state = _read_state(state_path) or {}
                failed_state.update({"trainingStatus": "failed", "trainingError": str(exc)})
                _atomic_json(state_path, failed_state)
            raise
        with sqlite3.connect(train_runner.db) as conn:
            rows = conn.execute("SELECT result_json FROM appworld_cells WHERE benchmark_id=? AND status='complete' ORDER BY task_id", ("train",)).fetchall()
        source_ids = [json.loads(row[0])["observation"]["run_id"] for row in rows]
        if len(source_ids) != 8:
            raise RuntimeError("training panel did not produce exactly eight durable runs")
        state.update({"trainingRunIds": source_ids, "trainingStatus": "complete", "learningStatus": "in_flight"})
        _atomic_json(state_path, state)
        emit("train", "complete", count=8, arms=[Arm.B0.value])
        emit("learning", "running")
        learning = LearningRuntime.build(store=runtime.controller.store, manager=runtime.controller.candidates, model_client=client, token_budget=args.model_tokens, wall_seconds=WALL_SECONDS)
        proposal = learning.propose_completed_runs(source_ids, primary_run_id=source_ids[0], goal="Learn reusable AppWorld procedures", feedback={"status": "completed"})
        candidate = proposal.authoritative_candidate
        learned_hash = candidate.get("candidateBundleHash") or candidate.get("candidate_bundle_hash")
        if not isinstance(learned_hash, str) or not learned_hash:
            raise RuntimeError("learning proposal did not produce a durable candidate bundle")
        receipt_bindings = _candidate_binding(runtime, learned_hash, base_hash, source_ids)
        ablation_hash = _derive_ablation(runtime, learned_hash)
        state.update({"learnedBundleHash": learned_hash, "learningReceiptBindings": receipt_bindings, "ablationBundleHash": ablation_hash, "learningStatus": "complete", "devStatus": "pending"})
        _atomic_json(state_path, state)
        emit("learning", "complete", sourceCount=8)
    if not isinstance(source_ids, list) or len(source_ids) != 8 or len(set(source_ids)) != 8 or not isinstance(learned_hash, str):
        raise RuntimeError("experiment state lacks a complete authenticated learning result")
    # Re-open the train panel on every invocation.  The runner verifies every
    # persisted receipt and only dispatches missing cells, so editable state
    # cannot replace the strict eight-run source declaration.
    train_runner = create_appworld_benchmark_runner(runtime, package, protocols[0], {Arm.B0: base_hash, Arm.L: base_hash, Arm.A: base_hash}, data_dir / "train")
    train_runner.run("train", {Arm.B0: base_hash, Arm.L: base_hash, Arm.A: base_hash}, arms=(Arm.B0,))
    recovered_source_ids = _training_ids(train_runner)
    if recovered_source_ids != source_ids:
        raise RuntimeError("durable train receipts do not match the frozen eight-run source declaration")
    if not isinstance(receipt_bindings, list) or not receipt_bindings:
        raise RuntimeError("experiment state lacks authenticated learning receipt bindings")
    _candidate_binding(runtime, learned_hash, base_hash, source_ids, receipt_bindings)
    expected_ablation_hash = _derive_ablation(runtime, learned_hash)
    if isinstance(ablation_hash, str) and ablation_hash and ablation_hash != expected_ablation_hash:
        raise RuntimeError("saved ablation bundle does not match deterministic L-derived ablation")
    ablation_hash = expected_ablation_hash
    if not current or current.get("ablationBundleHash") != ablation_hash:
        saved_state = _read_state(state_path) or {}
        saved_state["ablationBundleHash"] = ablation_hash
        _atomic_json(state_path, saved_state)
    bundles = {Arm.B0: base_hash, Arm.L: learned_hash, Arm.A: ablation_hash}
    if current and current.get("learningStatus") == "complete":
        emit("learning", "complete", sourceCount=8)
    reports: list[dict[str, Any]] = []
    emit("dev", "running", expected=60, arms=[arm.value for arm in (Arm.B0, Arm.L, Arm.A)])
    dev_runner = create_appworld_benchmark_runner(runtime, package, protocols[1], bundles, data_dir / "dev")
    dev_report = dev_runner.run("dev", bundles).to_dict()
    if dev_report.get("selectedArms") != [arm.value for arm in (Arm.B0, Arm.L, Arm.A)] or not dev_report.get("provenanceComplete"):
        raise RuntimeError("dev panel is not a complete B0/L/A receipt set")
    reports.append(dev_report)
    dev_state = _read_state(state_path) or {}
    dev_state.update({"devStatus": "complete", "devReport": dev_report})
    _atomic_json(state_path, dev_state)
    emit("dev", "complete", count=60, arms=[arm.value for arm in (Arm.B0, Arm.L, Arm.A)])
    final_package = AppWorldPackage(replace(package.config, allow_test=True))
    if final_package.catalog.dataset_hash() != package.catalog.dataset_hash() or final_package.public_fixture_hash() != package.public_fixture_hash():
        raise RuntimeError("final-enabled AppWorld package does not match the frozen public manifest")
    runtime.packages["appworld"] = final_package
    emit("test_normal", "running", expected=60, arms=[arm.value for arm in (Arm.B0, Arm.L, Arm.A)])
    final_runner = create_appworld_benchmark_runner(runtime, final_package, protocols[2], bundles, data_dir / "test_normal")
    final_report = final_runner.run("test_normal", bundles).to_dict()
    if final_report.get("selectedArms") != [arm.value for arm in (Arm.B0, Arm.L, Arm.A)] or not final_report.get("provenanceComplete"):
        raise RuntimeError("test_normal panel is not a complete B0/L/A receipt set")
    reports.append(final_report)
    final_state = _read_state(state_path) or {}
    final_state.update({"finalStatus": "complete", "finalReport": final_report})
    _atomic_json(state_path, final_state)
    emit("test_normal", "complete", count=60, arms=[arm.value for arm in (Arm.B0, Arm.L, Arm.A)])
    if len(reports) < 2:
        saved = _read_state(state_path) or {}
        reports = [saved[key] for key in ("devReport", "finalReport") if isinstance(saved.get(key), dict)]
    return {"stages": {"train": "complete", "learning": "complete", "dev": "complete", "test_normal": "complete"}, "reports": reports}


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
    parser.add_argument("--model-profile", default=MODEL_NAME)
    parser.add_argument("--dataset-content-hash")
    parser.add_argument("--sampling-seed", type=int, default=20260906)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-tokens", type=int, default=MODEL_TOKENS)
    parser.add_argument("--cost-microunits", type=int, default=100000)
    parser.add_argument("--prime-executable", default="prime-agent")
    parser.add_argument("--coding-agent-dir", default=os.environ.get("PRIME_AGENT_CODING_AGENT_DIR"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_experiment(build_parser().parse_args(argv), progress=lambda event: print(json.dumps({"stageProgress": event}, sort_keys=True), flush=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
