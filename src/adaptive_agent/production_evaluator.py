"""Runnable production evaluator entry point.

The integrated application runtime is imported inside ``build_job`` so this
module remains an evaluator-owned boundary.  Deployment supplies the runtime
package on ``PYTHONPATH``; this launcher never creates a synthetic executor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


MODEL_TOKENS = 20_000
WALL_SECONDS = 90
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHARED_ROOT_QA_DATA = Path("/private/tmp/adaptive-agent-browser-api")


def _reject_shared_qa_path(*values: str | os.PathLike[str] | None) -> None:
    """Keep evaluator stores away from the shared browser/QA database."""
    forbidden = SHARED_ROOT_QA_DATA.resolve()
    resolved = [Path(value).expanduser().resolve() for value in values if value is not None]
    if any(path == forbidden or forbidden in path.parents for path in resolved):
        raise RuntimeError(f"shared root QA data directory is forbidden: {forbidden}")
    if len(resolved) == 2 and resolved[0] == resolved[1]:
        raise RuntimeError("source and target evaluator stores must be isolated")


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


def _require_bound_real_receipt(source_dir: str, run_id: str) -> None:
    """Require the supplied run to have canonical trusted evidence already."""
    from adaptive_agent.store import Store
    from adaptive_agent.evaluation import sha256_json

    source_store = Store(source_dir)
    run = source_store.get_run(run_id)
    if run is None or run.get("status") != "succeeded":
        raise RuntimeError(f"bound source run is not a succeeded durable run: {run_id}")
    evidence = source_store.list_evidence(run_id)
    model = [row for row in evidence if row.get("event_type") == "model_response" and row.get("trust_class") in {"broker", "system"} and row.get("visibility") in {"operator", "evaluator_only"}]
    trusted = [row for row in evidence if row.get("event_type") == "trusted_outcome" and row.get("trust_class") == "evaluator" and row.get("visibility") in {"operator", "evaluator_only"}]
    if not model or not trusted:
        raise RuntimeError(
            f"bound source run lacks canonical model/trusted outcome evidence: {run_id}"
        )
    try:
        run_payload = json.loads(run.get("run_json") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"bound source run payload is malformed: {run_id}") from exc
    if not isinstance(run_payload, dict):
        raise RuntimeError(f"bound source run payload is malformed: {run_id}")

    def read_canonical(row: dict[str, Any]) -> dict[str, Any]:
        artifact_ref = json.loads(row["source_ref"])
        if artifact_ref.get("sha256") != row.get("content_hash"):
            raise RuntimeError("bound source evidence reference hash mismatch")
        artifact = artifact_ref.get("sha256")
        value = source_store.get_artifact(artifact)
        if not isinstance(value, dict) or sha256_json(value) != artifact_ref.get("sha256"):
            raise RuntimeError("bound source evidence artifact is not canonical")
        return value

    try:
        response = read_canonical(model[-1])
        outcome = read_canonical(trusted[-1])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        raise RuntimeError(f"bound source run lacks canonical trusted evidence: {run_id}") from exc
    task_id = run.get("task_id")
    environment_id = run.get("environment_id")
    if not isinstance(task_id, str) or not isinstance(environment_id, str):
        raise RuntimeError(f"bound source run identity is incomplete: {run_id}")
    identity = (run_id, task_id, environment_id, run_payload.get("arm"), run_payload.get("seed"), run_payload.get("bundleHash"))
    if any(value is None for value in identity[3:]):
        raise RuntimeError(f"bound source run execution identity is incomplete: {run_id}")
    for payload in (response, outcome):
        if (payload.get("runId"), payload.get("taskId"), payload.get("environmentId"), payload.get("arm"), payload.get("seed"), payload.get("bundleHash")) != identity:
            raise RuntimeError(f"bound source evidence identity does not match run: {run_id}")
    if response.get("responseId") != outcome.get("responseId"):
        raise RuntimeError(f"bound source outcome is not bound to model response: {run_id}")


def _docker_image_digest() -> str:
    image = os.environ.get("ADAPTIVE_AGENT_DOCKER_IMAGE", "adaptive-prime-runtime:a3a6cdeee3e92853dba9")
    result = subprocess.run(["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}} {{.Id}}"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"configured Docker image is unavailable: {image}")
    try:
        payload, local_id = result.stdout.strip().split(" ", 1)
        digests = json.loads(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("Docker image digest could not be resolved") from exc
    if isinstance(digests, list) and digests:
        return str(digests[0])
    if isinstance(local_id, str) and local_id.startswith("sha256:"):
        return local_id
    raise RuntimeError("Docker image has no immutable digest")


def _require_pin(name: str, value: str, *, image: bool = False) -> str:
    if not isinstance(value, str) or not value or value in {"core-planner-unset", "image-unpinned"}:
        raise RuntimeError(f"{name} is not pinned")
    if image:
        if "@sha256:" not in value and not SHA256.fullmatch(value.removeprefix("sha256:")):
            raise RuntimeError(f"{name} is not an immutable Docker digest")
    elif not SHA256.fullmatch(value):
        raise RuntimeError(f"{name} is not a SHA-256 pin")
    return value


def _persist_or_verify_frozen(store: Any, frozen: Any, *, initialize: bool) -> None:
    payload = frozen.to_dict()
    existing = store.get_frozen_protocol(frozen.protocol_hash)
    if initialize:
        if existing is not None:
            raise RuntimeError("initialization refuses an existing frozen protocol")
        store.save_frozen_protocol(frozen.protocol_hash, json.dumps(payload, sort_keys=True), "production-evaluator")
        return
    if existing is None or json.loads(existing.get("gate_json", "{}")) != payload:
        raise RuntimeError("resume frozen protocol pins do not match")


def _persist_or_verify_workload(store: Any, job_id: str, workload: dict[str, Any], *, initialize: bool) -> None:
    """Bind the complete workload to the durable job before execution."""
    with store.connect() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS evaluation_workloads (job_id TEXT PRIMARY KEY, workload_json TEXT NOT NULL)"
        )
        row = conn.execute("SELECT workload_json FROM evaluation_workloads WHERE job_id = ?", (job_id,)).fetchone()
        encoded = json.dumps(workload, sort_keys=True)
        if row is None:
            if not initialize:
                raise RuntimeError("resume job has no persisted workload binding")
            conn.execute("INSERT INTO evaluation_workloads(job_id, workload_json) VALUES (?, ?)", (job_id, encoded))
            conn.commit()
            return
        if initialize or row["workload_json"] != encoded:
            raise RuntimeError("job workload is already frozen with different counts")


def build_job(data_dir: str, source_data_dir: str | None, candidate_id: str | None, source_run_id: str | None, *, a_hash: str | None = None, initialize: bool = False, force_job: bool = False) -> tuple[Any, Any, Any]:
    """Build an evaluator job around the integrated durable runtime.

    The import is deliberately local: an evaluator process must resolve the
    deployed ``DurableRuntime`` implementation at execution time.
    """
    from adaptive_agent.app import create_runtime_app
    from adaptive_agent.evaluation import Arm, BudgetSpec, EvaluationProtocol

    target = Path(data_dir)
    _reject_shared_qa_path(target, source_data_dir)
    if initialize and target.exists() and any(target.iterdir()):
        raise RuntimeError(f"data directory must be a new empty directory: {data_dir}")
    if source_data_dir and source_run_id:
        _require_bound_real_receipt(source_data_dir, source_run_id)
    image_digest = _docker_image_digest()
    os.environ["ADAPTIVE_AGENT_IMAGE_DIGEST"] = image_digest

    app = create_runtime_app(data_dir=data_dir)
    runtime = app.state.durable_runtime
    evaluation_module = __import__("adaptive_agent.evaluation", fromlist=["__file__"])
    analysis_hash = hashlib.sha256(Path(evaluation_module.__file__).read_bytes()).hexdigest()
    core_pin = _require_pin("core planner hash", runtime.core_planner_hash)
    image_pin = _require_pin("Docker image digest", image_digest, image=True)
    analysis_pin = _require_pin("analysis code hash", analysis_hash)
    protocol = EvaluationProtocol(
        core_planner_hash=core_pin,
        image_digest=image_pin,
        analysis_code_hash=analysis_pin,
        run_budget=BudgetSpec(model_tokens=MODEL_TOKENS, wall_time_seconds=WALL_SECONDS),
        concurrency_limit=4,
    )
    protocol.freeze(runtime.packages)
    _persist_or_verify_frozen(runtime.controller.store, protocol.start_candidate_generation(), initialize=initialize)
    active = runtime.controller.get_active_bundle()
    if active is None:
        raise RuntimeError("durable runtime has no active base bundle")
    bundles: dict[Any, Any] = {Arm.B0: active}
    if candidate_id is not None:
        learned = _candidate_bundle(runtime.controller.store, candidate_id)
        bundles[Arm.L] = learned
    if a_hash is not None:
        bundles[Arm.A] = _bundle_by_hash(runtime.controller.store, a_hash)
    if candidate_id is None and not force_job:
        return app, protocol, runtime.build_evaluation_driver(protocol, bundles)
    job = runtime.build_evaluation_job(protocol, bundles)
    return app, protocol, job


def _workload(protocol: Any, candidate_count: int, training: int, transfer: int, safety: int, retries: int) -> dict[str, Any]:
    plan = protocol.workload(candidate_count, training_runs=training, transfer_runs=transfer, safety_runs=safety, retries=retries)
    return plan.to_dict()


def _lifecycle_stages(runtime: Any, protocol: Any, declared_retries: int) -> tuple[Any, ...]:
    """Bind the evaluator lifecycle to the runtime-owned stage callback."""
    from adaptive_agent.evaluation_job import LifecycleStage

    callback = getattr(runtime, "run_experiment_stage", None)
    if not callable(callback):
        raise RuntimeError("runtime does not expose the complete experiment stage callback")
    known = tuple(protocol.known_environments)
    training = tuple(
        task.task_id
        for environment_id in known
        for task in runtime.packages[environment_id].tasks_for_partition("development")
    )
    cells = {
        "bootstrap": ("pinned-development",),
        "training": training,
        "learning": ("candidate-generation",),
        "transfer": tuple(f"leave-out:{environment_id}" for environment_id in known),
        "adaptation": tuple(f"adapt:{environment_id}" for environment_id in known),
        "safety": tuple(protocol.safety_case_ids),
        "validation": tuple(f"validation:{index}" for index in range(protocol.validation_run_count)),
        "final": tuple(f"final:{index}" for index in range(protocol.final_run_count)),
    }

    def invoke(cell_key: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
        return callback(cell_key=cell_key, context={**dict(context), "declaredRetries": declared_retries, "sealedEnvironment": protocol.sealed_environment})

    return tuple(LifecycleStage(name, cells[name], invoke) for name in ("bootstrap", "training", "learning", "transfer", "adaptation", "safety", "validation", "final"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the durable Luna evaluation entry point")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--source-data-dir")
    parser.add_argument("--candidate-id")
    parser.add_argument("--source-run-id")
    parser.add_argument("--job", required=True)
    parser.add_argument("--comparison", choices=("experiment", "smoke", "validation", "final"), required=True)
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
    parser.add_argument("--initialize", action="store_true", help="create and freeze a new clean Store")
    parser.add_argument("--resume", action="store_true", help="resume an existing experiment Store")
    args = parser.parse_args(argv)

    if args.initialize and args.resume:
        parser.error("--initialize and --resume are mutually exclusive")

    if args.comparison not in {"smoke", "experiment"} and (not args.candidate_id or not args.source_data_dir or not args.source_run_id):
        parser.error("held-out evaluation requires candidate/source store/source run bindings")
    app, protocol, job = build_job(args.data_dir, args.source_data_dir, args.candidate_id, args.source_run_id, a_hash=args.a_hash, initialize=args.initialize, force_job=args.comparison == "experiment")
    workload = _workload(protocol, args.candidate_count, args.training_runs, args.transfer_runs, args.safety_runs, args.retries)
    print(json.dumps({"job": args.job, "comparison": args.comparison, "workload": workload}, sort_keys=True))
    if workload["transferRuns"] < 1:
        parser.error("--transfer-runs must be positive; EVAL-002 requires a real transfer workload")
    if args.comparison != "smoke" and args.training_runs < 1:
        parser.error("--training-runs must be positive for the complete frozen workload")
    _persist_or_verify_workload(app.state.durable_runtime.controller.store, args.job, workload, initialize=args.initialize)
    if args.comparison == "experiment":
        stages = _lifecycle_stages(app.state.durable_runtime, protocol, args.retries)
        expected_counts = {stage.name: len(stage.cells) for stage in stages}
        if args.candidate_count != 1 or args.training_runs != expected_counts["training"] or args.transfer_runs != expected_counts["transfer"] or args.safety_runs != expected_counts["safety"]:
            parser.error(f"complete experiment requires execution counts training={expected_counts['training']}, transfer={expected_counts['transfer']}, safety={expected_counts['safety']}, candidate-count=1")
        actual_attempts = sum(expected_counts.values()) + args.retries
        limits = {
            "attempts": actual_attempts,
            "inputTokens": actual_attempts * protocol.run_budget.model_tokens,
            "outputTokens": actual_attempts * protocol.run_budget.model_tokens,
            "toolCalls": actual_attempts * protocol.run_budget.tool_calls,
            "wallMicros": int(actual_attempts * protocol.run_budget.wall_time_seconds * 1_000_000),
            "costMicrounits": actual_attempts * protocol.run_budget.cost_microunits,
        }
        result = job.run_experiment(args.job, stages, limits=limits, context={"resume": args.resume or not args.initialize})
        print(json.dumps({"status": result.status, "error": result.error, "runtimeAccounting": result.runtime_accounting}, sort_keys=True, default=str))
        return 0 if result.status == "complete" else 1
    if args.comparison != "smoke":
        job.max_total_attempts = workload["totalAttemptedRuns"]
        job.total_budget_microunits = workload["totalAttemptedRuns"] * protocol.run_budget.cost_microunits
    if args.comparison == "smoke":
        result = job.run_development_smoke(args.job)
        print(json.dumps({"complete": result.complete, "expected": result.expected_count, "statuses": [item.status for item in result.statuses]}))
        return 0 if result.complete else 1
    if not args.base_hash or not args.candidate_hash:
        parser.error("--base-hash and --candidate-hash are required for held-out panels")
    if args.comparison == "final":
        if args.a_hash is None:
            parser.error("--a-hash is required for the final B0/L/A panel")
        from adaptive_agent.evaluation import AblationInput, audit_ablation

        if args.ablation_system_instructions or args.ablation_retrieval_input:
            parser.error("ablation inputs are derived from the pinned A bundle; arbitrary CLI text is forbidden")
        a_bundle = _bundle_by_hash(app.state.durable_runtime.controller.store, args.a_hash)
        execution = a_bundle.execution_config.model_dump(mode="json", by_alias=True)
        procedures = tuple(skill.procedure for skill in a_bundle.skills)
        ablation = AblationInput(args.a_hash, json.dumps(execution, sort_keys=True), procedures)
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
