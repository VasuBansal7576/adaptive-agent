"""Executable application factory for the integrated Adaptive Agent stack.

The factory wires the durable Store/EnvironmentRegistry/ToolBroker/Controller
seam into the operator API and seeds only trusted fixture references.  A live
model runner is injected by deployment, so the default process cannot silently
claim that an unavailable provider is a simulation or a successful run.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import math
import os
import json
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from adaptive_agent.api import ControlPlane, ModelRunner, OutcomeEvaluator, create_app, _ref, IdempotencyConflict
from adaptive_agent.constants import DEFAULT_MODEL_TOKENS
from adaptive_agent.broker import Capability, ProviderExecutionOutcome, ToolBroker, ToolProvider
from adaptive_agent.controller import Controller
from adaptive_agent.environment import EnvironmentRegistry
from adaptive_agent.evaluation import EvaluationError, build_environment_packages, sha256_json, FixtureSession, Outcome as FixtureOutcome, TrustedEvaluatorRegistry
from adaptive_agent.learning import LearningService, PlannerLearningAdapter
from adaptive_agent.learning_store import DurableLearningSourceAdapter, CandidateManagerLearningAdapter, LearningStoreError
from adaptive_agent.learning_runtime import LEARNING_SOURCE_RUN_CAP, LearningRuntime, LearningRuntimeError
from adaptive_agent.planner import PrimeCliModelClient, LunaPlanner, PlannerResult, PlannerLimits
from adaptive_agent.prime_child_planner import LunaChildPlanner
from adaptive_agent.prime_runtime import Capability as PrimeCapability, CapabilityBroker, PrimeRuntimeAdapter, PrimeRuntimeConfig
from adaptive_agent.models import ArtifactRef, EnvironmentManifest as DurableManifest, TaskInput as DurableTask, ToolSchema as DurableTool, RunStatus, ToolRequest, ToolError, ToolErrorCode, Outcome as DurableOutcome, canonical_usage
from adaptive_agent.store import Store
from adaptive_agent.evaluation_store import build_durable_adapters


def _manifest_value(value: Any) -> Any:
    """Serialize manifest boundary objects independent of their implementation."""
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json", by_alias=True)
        except TypeError:
            return model_dump()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return value


def _effective_observation_cost(accounting: Mapping[str, Any]) -> int:
    """Return an observation cost only when its source is complete.

    Measured microunits are authoritative when present.  A nominal USD value
    is usable as a clearly labeled proxy only when the accounting artifact
    explicitly states complete, non-empty receipt coverage.
    """
    cost = accounting.get("costMicrounits")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(float(cost)) and cost >= 0:
        return int(round(float(cost)))
    nominal = accounting.get("nominalCostUsd")
    coverage = accounting.get("nominalCostCoverage")
    complete_coverage = (
        accounting.get("nominalCostStatus") == "complete"
        and isinstance(coverage, Mapping)
        and isinstance(coverage.get("knownReceipts"), int)
        and not isinstance(coverage.get("knownReceipts"), bool)
        and isinstance(coverage.get("totalReceipts"), int)
        and not isinstance(coverage.get("totalReceipts"), bool)
        and coverage["totalReceipts"] > 0
        and coverage["knownReceipts"] == coverage["totalReceipts"]
    )
    if complete_coverage and isinstance(nominal, (int, float)) and not isinstance(nominal, bool) and math.isfinite(float(nominal)) and nominal >= 0:
        return int(round(float(nominal) * 1_000_000))
    raise LearningRuntimeError("evaluation accounting lacks complete economic or nominal cost")


def _freeze_core_planner_hash() -> str:
    """Hash the executable planner sources once for this process.

    Skill bundles are data selected by a run and are deliberately excluded;
    this identity must remain stable when a learned bundle is compared with
    its baseline.  An explicit deployment pin wins when supplied.
    """
    configured = os.environ.get("ADAPTIVE_AGENT_CORE_PLANNER_HASH")
    if configured and configured.strip():
        return configured.strip()
    digest = hashlib.sha256()
    # Include every executable planner boundary, including the child planner
    # adapter and provider client implementation imported by planner.py.
    for name in ("planner.py", "prime_child_planner.py", "prime_runtime.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _freeze_image_digest() -> str:
    """Capture the trusted Docker image identity before candidate generation."""
    configured = os.environ.get("ADAPTIVE_AGENT_IMAGE_DIGEST") or os.environ.get("ADAPTIVE_AGENT_DOCKER_IMAGE_DIGEST")
    return configured.strip() if isinstance(configured, str) and configured.strip() else "image-unpinned"


class _FixtureProvider(ToolProvider):
    """Prime host-request provider backed by one reset, trusted fixture task."""

    def execute(self, run_id: str, tool: str, arguments: dict[str, Any]) -> ProviderExecutionOutcome:
        if run_id != self.session.task_id and run_id != self._run_id:
            raise RuntimeError("provider is bound to a different run")
        try:
            result = self.package.invoke(self.session, tool, arguments)
        except EvaluationError as exc:
            message = str(exc)
            if not message.startswith(("unknown tool arguments:", "missing tool arguments:", "invalid value for tool argument")):
                raise
            return ProviderExecutionOutcome(
                output={"ok": False, "code": ToolErrorCode.INVALID_INPUT.value},
                status="error",
                effect="none",
                error=ToolError(code=ToolErrorCode.INVALID_INPUT, message=message, retry="never"),
            )
        return ProviderExecutionOutcome(
            output=dict(result.output),
            status="ok" if result.status == "ok" else "error",
            effect=result.side_effect if result.side_effect in {"none", "confirmed", "unknown"} else "unknown",
        )

    def effect(self, tool: str) -> str:
        schema = next((item for item in self.package.manifest.tool_schemas if item.name == tool), None)
        if schema is None:
            raise RuntimeError(f"unknown tool {tool!r}")
        return schema.effect

    def version(self, tool: str) -> str:
        schema = next((item for item in self.package.manifest.tool_schemas if item.name == tool), None)
        if schema is None:
            raise RuntimeError(f"unknown tool {tool!r}")
        return schema.version

    def __init__(self, package: Any, task: Any, run_id: str, seed: int = 0) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("fixture reset seed must be an integer")
        self.package = package
        self.session = package.reset(task.task_id, seed)
        self._run_id = run_id
        self.seed = seed


class _RegisteredPackage:
    """Executable shell for an operator-registered manifest.

    Registration supplies schemas and task goals, while the deployment may
    provide the actual model/evaluator.  Unknown tools fail closed through the
    broker; direct model runs can still use the durable lifecycle.
    """

    def __init__(self, manifest: DurableManifest, tasks: list[DurableTask]) -> None:
        self.environment_id = manifest.environment_id
        self.manifest = manifest
        self._tasks = tasks
        self.evaluator_ready = False

    def learner_tasks(self) -> list[Any]:
        return list(self._tasks)

    def learner_documents(self) -> list[Any]:
        return []

    def reset(self, task_id: str, seed: int) -> FixtureSession:
        return FixtureSession(self.environment_id, task_id, seed, {})

    def invoke(self, session: FixtureSession, tool: str, arguments: Mapping[str, Any], **_: Any) -> Any:
        raise RuntimeError(f"no trusted fixture provider is registered for {self.environment_id}:{tool}")

    def evaluate(self, task_id: str, session: FixtureSession) -> FixtureOutcome:
        return FixtureOutcome(False, False, 0, "registered environment evaluator is not configured", "unconfigured")


class DurableRuntime:
    """Adapter used by the HTTP layer to make Controller the source of truth."""

    def __init__(self, controller: Controller, registry: EnvironmentRegistry, packages: Mapping[str, Any], model_runner: Any | None = None, evaluator: Any | None = None, model_ref: Mapping[str, Any] | None = None, budget_ref: Mapping[str, Any] | None = None, control_plane: Any | None = None, package_bindings: Mapping[str, Any] | None = None, learning_model_client: Any | None = None, evaluation_executor: Any | None = None, experiment_stage_runner: Any | None = None) -> None:
        self.controller, self.registry, self.packages = controller, registry, dict(packages)
        if package_bindings:
            self.packages.update(package_bindings)
        self.model_runner, self.evaluator = model_runner, evaluator
        self.model_ref, self.budget_ref = dict(model_ref or _ref("model-profile", "1")), dict(budget_ref or _ref("budget-default", "1"))
        self.control_plane = control_plane
        self.learning_model_client = learning_model_client
        self.evaluation_executor = evaluation_executor
        self.experiment_stage_runner = experiment_stage_runner
        self._evaluation_protocol: Any | None = None
        # These pins are process-scoped and are captured before any learned
        # candidate can be generated.  They never derive from the active
        # skill bundle, which is the arm-specific input under evaluation.
        self.core_planner_hash = _freeze_core_planner_hash()
        self.image_digest = _freeze_image_digest()
        self._learning_runtime: LearningRuntime | None = None
        self._cancel_events: dict[str, threading.Event] = {}
        self._run_started_at: dict[str, float] = {}
        self._run_last_receipt_at: dict[str, float] = {}
        self._run_receipts: dict[str, list[dict[str, Any]]] = {}
        self._approvals: dict[tuple[str, str], bool] = {}
        self._evaluation_arm_bundles: dict[str, str] = {}
        # The canonical Controller seam owns lifecycle persistence.  Older
        # adapters exposed an atomic claim helper; keep a process-local guard
        # only for the canonical seam, which intentionally leaves claiming to
        # its caller.
        self._launch_claim_lock = threading.Lock()
        self._tasks = {
            task.task_id: task
            for name in self.packages
            for task in registry.list_tasks_by_partition(name, "development")
        }
        self._reload_registered_environments()

    def _get_or_create_experiment_stage_runner(self) -> Any:
        """Return the runtime-owned runner, lazily restoring it after restart."""
        callback = self.experiment_stage_runner
        if callback is None and self._evaluation_protocol is not None:
            from adaptive_agent.experiment_runtime import DefaultExperimentStageRunner

            callback = DefaultExperimentStageRunner(self, self._evaluation_protocol)
            self.experiment_stage_runner = callback
        return callback

    def run_experiment_stage(self, *, cell_key: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run one lifecycle cell through the runtime-owned implementation.

        The production evaluator binds this callback when constructing the
        complete experiment lifecycle.  A missing binding fails closed instead
        of manufacturing a receipt from workload arithmetic.
        """
        callback = self._get_or_create_experiment_stage_runner()
        if not callable(callback):
            raise LearningRuntimeError("runtime does not have a trusted experiment stage runner")
        result = callback(cell_key=cell_key, context=dict(context))
        if not isinstance(result, Mapping):
            raise LearningRuntimeError("experiment stage runner returned a non-object")
        return dict(result)

    def establish_clean_experiment(self, protocol: Any) -> dict[str, Any]:
        """Verify and persist the immutable bootstrap provenance for a job."""
        frozen = protocol.start_candidate_generation()
        inputs = getattr(frozen, "inputs", None)
        active = self.controller.get_active_bundle()
        active_hash = getattr(active, "content_hash", None)
        adapters = getattr(self.controller, "evaluator_adapters", ())
        verifier_ready = isinstance(adapters, tuple) and len(adapters) >= 3 and bool(getattr(adapters[2], "durable", False))
        image = self.image_digest
        image_available = False
        if isinstance(image, str) and image and image != "image-unpinned":
            inspected = subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False)
            image_available = inspected.returncode == 0
        known = tuple(getattr(protocol, "known_environments", ()))
        partitions_clean = bool(known) and all(
            environment_id in self.packages
            and callable(getattr(self.packages[environment_id], "reset", None))
            and callable(getattr(self.packages[environment_id], "evaluate", None))
            and bool(tuple(self.packages[environment_id].tasks_for_partition("development")))
            for environment_id in known
        )
        clean = bool(
            isinstance(inputs, Mapping)
            and isinstance(active_hash, str)
            and bool(active_hash)
            and self.controller.store.get_bundle_by_hash(active_hash) is not None
            and verifier_ready
            and image_available
            and partitions_clean
            and inputs.get("corePlannerHash") == self.core_planner_hash
            and inputs.get("imageDigest") == image
            and inputs.get("provider") == "openai-codex"
            and inputs.get("modelProfile") == "openai-codex/gpt-5.6-luna"
        )
        if not clean:
            return {"clean": False, "actualDocker": image_available, "provenanceRef": ""}
        provenance = {
            "kind": "clean_experiment_bootstrap",
            "protocolHash": frozen.protocol_hash,
            "corePlannerHash": self.core_planner_hash,
            "imageDigest": image,
            "baseBundleHash": active_hash,
            "developmentEnvironments": list(known),
            "developmentTaskIds": [task.task_id for environment_id in known for task in self.packages[environment_id].tasks_for_partition("development")],
        }
        ref = self.controller.store.put_artifact(provenance)
        return {"clean": True, "actualDocker": True, "provenanceRef": ref.sha256}

    def verify_evaluation_observation(self, observation: Any, config: Any, task: Any) -> bool:
        """Verify a benchmark receipt using the durable strict evaluator adapter."""
        frozen = getattr(config, "protocol", None)
        if frozen is None or not isinstance(getattr(frozen, "inputs", None), Mapping):
            return False
        environment_id = getattr(getattr(task, "environment_ref", None), "id", None) or getattr(task, "environment_id", None)
        package = self.packages.get(environment_id) if isinstance(environment_id, str) else None
        if package is None or getattr(config, "bundle_hash", None) != getattr(observation, "bundle_hash", None):
            return False
        adapters = getattr(self.controller, "evaluator_adapters", ())
        verifier = adapters[2] if isinstance(adapters, tuple) and len(adapters) >= 3 else None
        if verifier is None or not bool(getattr(verifier, "durable", False)):
            return False
        try:
            return verifier.verify(observation, frozen, package) is True
        except (AttributeError, KeyError, TypeError, ValueError):
            return False

    def _reload_registered_environments(self) -> None:
        """Rebuild manifest/task projections from SQLite after a restart."""
        with self.controller.store.connect() as conn:
            rows = conn.execute("SELECT id FROM environments ORDER BY id").fetchall()
        for row in rows:
            env_id = str(row["id"])
            if env_id in self.packages:
                continue
            manifest = self.registry.get_manifest(env_id)
            if manifest is None:
                continue
            tasks = self.registry.list_tasks_by_partition(env_id, "development")
            if not tasks:
                continue
            self.packages[env_id] = _RegisteredPackage(manifest, tasks)
            self._tasks.update({task.task_id: task for task in tasks})

    def register_environment(self, payload: Any) -> dict[str, Any]:
        from adaptive_agent.models import ArtifactRef, EnvironmentManifest, TaskInput, ToolSchema

        docs = [ArtifactRef.model_validate(ref) for ref in payload.docs]
        manifest = EnvironmentManifest(
            schemaVersion=payload.schema_version,
            environmentId=payload.environment_id,
            version=payload.version,
            docs=docs,
            toolSchemas=[ToolSchema.model_validate(tool.model_dump(by_alias=True)) for tool in payload.tool_schemas],
            policyRef=ArtifactRef.model_validate(payload.policy_ref),
            evaluatorRef=ArtifactRef.model_validate(payload.evaluator_ref),
            resetRef=ArtifactRef.model_validate(payload.reset_ref),
            executionModes=list(payload.execution_modes),
            capabilities=list(payload.capabilities),
        )
        self.registry.register(manifest)
        tasks: list[Any] = []
        for index, goal in enumerate(payload.task_goals):
            task_id = f"{payload.environment_id}-development-{index:02d}"
            task = TaskInput(taskId=task_id, environmentRef=ArtifactRef(id=payload.environment_id, version=payload.version, sha256=sha256_json({"environmentId": payload.environment_id, "version": payload.version})), goal=goal, partition="development")
            self.registry.register_task(task)
            tasks.append(task)
        self.packages[payload.environment_id] = _RegisteredPackage(manifest, tasks)
        self._tasks.update({task.task_id: task for task in tasks})
        package = self.packages[payload.environment_id]
        return {"environmentId": payload.environment_id, "version": payload.version, "validationState": "valid", "evaluatorReady": callable(getattr(package, "evaluate", None)) and not isinstance(package, _RegisteredPackage), "toolCount": len(manifest.tool_schemas), "policyScope": manifest.policy_ref.id, "executionModes": list(manifest.execution_modes), "capabilities": list(manifest.capabilities)}

    def list_candidates(self) -> list[dict[str, Any]]:
        out = []
        with self.controller.store.connect() as conn:
            rows = conn.execute(
                "SELECT candidate_id, state, base_bundle_hash, candidate_bundle_hash, candidate_json FROM candidates ORDER BY created_at"
            ).fetchall()
        for row in rows:
            try:
                payload = json.loads(row["candidate_json"])
                if not isinstance(payload, dict):
                    continue
                def value_for(alias: str, snake: str, fallback: Any = None) -> Any:
                    value = payload.get(alias, payload.get(snake, fallback))
                    return value

                operations = value_for("editOperations", "edit_operations", [])
                if not isinstance(operations, list):
                    operations = []
                operations = [
                    item if isinstance(item, str) else json.dumps(item, sort_keys=True, separators=(",", ":"))
                    for item in operations
                ]
                changed = value_for("changedArtifactHashes", "changed_artifact_hashes", [])
                supporting = value_for("supportingEvidenceIds", "supporting_evidence_ids", [])
                out.append(
                    {
                        "candidateId": row["candidate_id"],
                        "state": row["state"],
                        "predictedEffect": value_for("predictedEffect", "predicted_effect", ""),
                        "baseBundleHash": value_for("baseBundleHash", "base_bundle_hash", row["base_bundle_hash"]),
                        "candidateBundleHash": value_for("candidateBundleHash", "candidate_bundle_hash", row["candidate_bundle_hash"]),
                        "editOperations": operations,
                        "changedArtifactHashes": changed if isinstance(changed, list) else [],
                        "supportingEvidenceIds": supporting if isinstance(supporting, list) else [],
                        "proposerVersion": value_for("proposerVersion", "proposer_version", ""),
                    }
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return out

    def create_candidate(self, payload: Any) -> dict[str, Any]:
        adapter = CandidateManagerLearningAdapter(self.controller.store, self.controller.candidates, proposal_type=__import__("adaptive_agent.models", fromlist=["CandidateProposal"]).CandidateProposal, bundle_type=__import__("adaptive_agent.models", fromlist=["SkillBundle"]).SkillBundle, skill_type=__import__("adaptive_agent.models", fromlist=["SkillVersion"]).SkillVersion)
        value = dict(adapter.create_candidate(payload.model_dump(by_alias=True)))
        if "candidate_id" in value:
            value["candidateId"] = value.pop("candidate_id")
        return value

    def queue_evaluation(self, payload: Any) -> dict[str, Any]:
        protocol = self._evaluation_protocol
        protocol_hash = getattr(payload, "protocol_hash", None)
        partition_ref = getattr(payload, "partition_ref", None)
        if protocol is not None:
            frozen = protocol.start_candidate_generation()
            if not isinstance(protocol_hash, str) or not protocol_hash:
                protocol_hash = frozen.protocol_hash
            if not isinstance(partition_ref, Mapping):
                partition_ref = self._validation_partition_ref(frozen)
            expected_hash = frozen.partition_hashes.get(f"{protocol.known_environments[0]}:validation")
            supplied_hash = partition_ref.get("sha256") if isinstance(partition_ref, Mapping) else None
            if protocol_hash != frozen.protocol_hash:
                raise ValueError("evaluation protocol does not match the frozen server protocol")
            if not isinstance(expected_hash, str) or supplied_hash != expected_hash:
                raise ValueError("evaluation partition does not match the frozen server partition")
        if not isinstance(protocol_hash, str) or not protocol_hash or not isinstance(partition_ref, Mapping):
            raise ValueError("evaluation launch requires frozen protocol and partition bindings")
        candidate = self.controller.get_candidate(payload.candidate_id)
        if candidate is None:
            raise KeyError("candidate not found")
        if payload.base_bundle_hash != candidate.get("base_bundle_hash"):
            raise ValueError("evaluation base does not match candidate")
        self.controller.start_evaluation(payload.candidate_id)
        evaluation_id = f"eval_{__import__('uuid').uuid4().hex}"
        response = {
            "evaluationId": evaluation_id,
            "candidateId": payload.candidate_id,
            "baseBundleHash": payload.base_bundle_hash,
            "protocolHash": protocol_hash,
            "partitionRef": dict(partition_ref),
            "state": "queued",
            "trusted": False,
        }
        self.controller.store.save_evaluation_queue(
            evaluation_id,
            {
                "candidate_id": payload.candidate_id,
                "candidate_hash": candidate.get("candidate_bundle_hash") or payload.candidate_id,
                "base_hash": payload.base_bundle_hash,
                "protocol_hash": protocol_hash,
                "partition_ref": json.dumps(dict(partition_ref), sort_keys=True),
                "state": "queued",
                "payload_json": json.dumps(response, sort_keys=True),
            },
        )
        return response

    @staticmethod
    def _validation_partition_ref(frozen: Any) -> dict[str, str]:
        environments = tuple(getattr(frozen, "inputs", {}).get("knownEnvironments", ()))
        hashes = getattr(frozen, "partition_hashes", {})
        if not environments:
            raise ValueError("frozen protocol has no known environments")
        key = f"{environments[0]}:validation"
        digest = hashes.get(key)
        if not isinstance(digest, str) or not digest:
            raise ValueError("frozen protocol has no validation partition")
        return {"id": "validation", "version": "1", "sha256": digest}

    def launch_evaluation(self, evaluation_id: str) -> dict[str, Any]:
        """Execute one queued evaluation through the bound durable job."""
        queued = self.controller.store.get_evaluation_queue(evaluation_id)
        if queued is None:
            raise KeyError("evaluation not found")
        if queued.get("state") == "completed":
            report = self.controller.store.get_evaluation(evaluation_id)
            if report is None:
                return {"evaluationId": evaluation_id, "state": "completed", "trusted": False, "trustReason": "missing evaluator report"}
            try:
                payload = json.loads(report["report_json"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            normalized = dict(payload) if isinstance(payload, Mapping) else {}
            normalized["evaluationId"] = evaluation_id
            normalized["candidateId"] = report.get("candidate_id")
            normalized["baseBundleHash"] = report.get("base_hash")
            normalized["protocolHash"] = report.get("protocol_hash")
            normalized["trusted"] = self._verify_evaluation_report(normalized, report)
            if not normalized["trusted"]:
                normalized["trustReason"] = "unverified evaluator report"
            return normalized
        with self.controller.store.connect() as conn:
            claimed = conn.execute(
                "UPDATE evaluation_queue SET state = 'running', updated_at = datetime('now') WHERE evaluation_id = ? AND state IN ('queued', 'failed')",
                (evaluation_id,),
            ).rowcount
            conn.commit()
        if claimed != 1:
            current = self.controller.store.get_evaluation_queue(evaluation_id) or queued
            return {"evaluationId": evaluation_id, "candidateId": current.get("candidate_id"), "state": current.get("state", "running"), "trusted": False, "trustReason": "evaluation is already running"}
        protocol = getattr(self, "_evaluation_protocol", None)
        if protocol is None:
            raise ValueError("evaluation protocol is not bound")
        candidate_id = str(queued["candidate_id"])
        candidate = self.controller.get_candidate(candidate_id)
        active = self.controller.get_active_bundle()
        if candidate is None or active is None:
            raise KeyError("candidate or base bundle not found")
        candidate_hash = candidate.get("candidate_bundle_hash")
        if not isinstance(candidate_hash, str) or not candidate_hash:
            raise ValueError("candidate has no durable bundle hash")
        bundle = self.controller.store.get_bundle_by_hash(candidate_hash)
        if bundle is None:
            raise ValueError("candidate bundle is not durable")
        job = self.build_evaluation_job(protocol, {"B0": active, "L": __import__("adaptive_agent.models", fromlist=["SkillBundle"]).SkillBundle.model_validate(json.loads(bundle["bundle_json"]))})
        result = job.run(evaluation_id, "validation", base_hash=str(queued["base_hash"]), candidate_hash=candidate_hash, candidate_id=candidate_id)
        result_payload = result.report.to_dict() if hasattr(result.report, "to_dict") else result.report
        stored_report = self.controller.store.get_evaluation(evaluation_id)
        # EvaluationJob persists its execution record separately from the
        # operator-facing evaluation table. Mirror every measured report under
        # the queue ID so replay and readback use one durable identity.
        if isinstance(result_payload, Mapping):
            self.controller.store.save_evaluation(
                evaluation_id,
                {
                    "candidate_hash": queued["candidate_hash"],
                    "base_hash": queued["base_hash"],
                    "protocol_hash": queued["protocol_hash"],
                    "partition_ref": queued["partition_ref"],
                    "report_json": json.dumps(dict(result_payload), sort_keys=True),
                    "validity": str(result_payload.get("validityStatus", result.status)),
                },
            )
            stored_report = self.controller.store.get_evaluation(evaluation_id)
        trusted = bool(isinstance(result_payload, Mapping) and stored_report is not None and self._verify_evaluation_report(result_payload, stored_report))
        terminal_state = "failed" if result.status == "failed" else "completed"
        queue_payload = dict(result_payload) if isinstance(result_payload, Mapping) else {
            "evaluationId": evaluation_id,
            "candidateId": candidate_id,
            "state": result.status,
            "trusted": trusted,
            "error": result.error,
        }
        self.controller.store.save_evaluation_queue(
            evaluation_id,
            {**queued, "state": terminal_state, "payload_json": json.dumps(queue_payload, sort_keys=True), "updated_at": datetime.now(timezone.utc).isoformat()},
        )
        return {"evaluationId": evaluation_id, "candidateId": candidate_id, "state": result.status, "trusted": trusted, "error": result.error}

    def execute_evaluation_task(self, task: Any, frozen_config: Any, bundle: Any) -> Any:
        """Execute one frozen benchmark cell through the shared run path."""
        from adaptive_agent.evaluation import Arm, BudgetSpec, ModelProvenance, Partition, Provenance, RunObservation
        from adaptive_agent.models import ArtifactRef, RunRequest, SkillBundle

        env_id = getattr(getattr(task, "environment_ref", None), "id", None) or getattr(task, "environment_id", None)
        task_id = getattr(task, "task_id", None)
        goal = getattr(task, "goal", None)
        if not isinstance(env_id, str) or not isinstance(task_id, str) or not isinstance(goal, str):
            raise LearningRuntimeError("evaluation task lacks environment, id, or goal")
        package = self.packages.get(env_id)
        if package is None or not callable(getattr(package, "reset", None)) or not callable(getattr(package, "evaluate", None)):
            raise LearningRuntimeError(f"trusted evaluator package is unavailable for {env_id}")
        protocol = getattr(frozen_config, "protocol", None)
        inputs = getattr(protocol, "inputs", None)
        if not isinstance(inputs, Mapping):
            raise LearningRuntimeError("evaluation execution config is not frozen")
        arm = getattr(frozen_config, "arm", None)
        seed = getattr(frozen_config, "seed", None)
        attempt = getattr(frozen_config, "attempt", 0)
        if arm is None or not isinstance(seed, int):
            raise LearningRuntimeError("evaluation execution config lacks arm or seed")
        arm_value = getattr(arm, "value", str(arm))
        if arm_value not in {"B0", "L", "A"}:
            raise LearningRuntimeError(f"evaluation arm is invalid: {arm_value!r}")
        if isinstance(seed, bool):
            raise LearningRuntimeError("evaluation execution seed must be an integer")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
            raise LearningRuntimeError("evaluation execution attempt must be a non-negative integer")
        if isinstance(bundle, SkillBundle):
            durable_bundle = bundle
        elif isinstance(bundle, Mapping):
            durable_bundle = SkillBundle.model_validate(bundle)
        else:
            durable_bundle = self.controller.get_active_bundle() or SkillBundle()
        expected_bundle_hash = getattr(frozen_config, "bundle_hash", None)
        bundle_hash = getattr(durable_bundle, "content_hash", None)
        if not isinstance(bundle_hash, str) or not bundle_hash:
            raise LearningRuntimeError("evaluation arm bundle has no content hash")
        if expected_bundle_hash and bundle_hash != expected_bundle_hash:
            raise LearningRuntimeError("evaluation bundle hash does not match supplied bundle")
        expected_arm_bundle = self._evaluation_arm_bundles.get(arm_value)
        if expected_arm_bundle is not None and expected_arm_bundle != bundle_hash:
            raise LearningRuntimeError("evaluation arm bundle does not match the frozen arm mapping")

        version = str(getattr(getattr(task, "environment_ref", None), "version", "1"))
        environment_ref = ArtifactRef(id=env_id, version=version, sha256=sha256_json({"environmentId": env_id, "version": version}))
        task_provenance = getattr(package, "task_provenance", None)
        provenance = task_provenance(task_id) if callable(task_provenance) else {}
        durable_task = DurableTask(taskId=task_id, environmentRef=environment_ref, goal=goal, partition=getattr(getattr(task, "partition", None), "value", getattr(task, "partition", "development")), provenance=provenance)
        self.registry.register_task(durable_task)
        task_ref = self.controller.store.put_artifact(durable_task.model_dump(mode="json", by_alias=True))
        provider_name = str(inputs.get("provider", "openai-codex"))
        model_name = str(inputs.get("modelProfile", "openai-codex/gpt-5.6-luna"))
        if provider_name != "openai-codex" or model_name != "openai-codex/gpt-5.6-luna":
            raise LearningRuntimeError("evaluation requires the pinned Luna subscription model")
        budget_value = dict(inputs.get("runBudget", {})) if isinstance(inputs.get("runBudget", {}), Mapping) else {}
        model_ref = self.controller.store.put_artifact({"provider": provider_name, "model": model_name})
        budget_ref = self.controller.store.put_artifact(budget_value)
        protocol_hash = str(getattr(protocol, "protocol_hash", "protocol-unset"))
        # Benchmark fixtures are reset per cell and may include declared write
        # actions.  Batch mode is the manifest-authorized path that issues the
        # one-use approvals required by the broker for those writes.
        request = RunRequest(taskRef=task_ref, modelProfileRef=model_ref, budgetRef=budget_ref, idempotencyKey=f"benchmark:{protocol_hash}:{task_id}:{arm_value}:{seed}:{durable_bundle.content_hash}:attempt:{attempt}", executionMode="batch")
        run = self.controller.create_run(request, durable_task, skill_bundle=durable_bundle)
        row = self.controller.store.get_run(run.run_id)
        if row:
            persisted = {key: value for key, value in row.items() if key != "run_id"}
            run_payload = json.loads(row.get("run_json", "{}"))
            arm_bundles = dict(self._evaluation_arm_bundles) if self._evaluation_arm_bundles else {arm_value: bundle_hash}
            arm_bundles.setdefault(arm_value, bundle_hash)
            run_payload.update({"arm": arm_value, "seed": seed, "bundleHash": bundle_hash, "armBundles": arm_bundles})
            persisted["run_json"] = json.dumps(run_payload, sort_keys=True)
            self.controller.store.save_run(run.run_id, persisted)

        # Prefer the explicit evaluation runner, then use the injected client
        # for benchmark-only Prime executions that have no separate runner.
        model_client = self.model_runner if self.model_runner is not None and callable(getattr(self.model_runner, "invoke", None)) else None
        if model_client is None and self.learning_model_client is not None and callable(getattr(self.learning_model_client, "invoke", None)):
            model_client = self.learning_model_client
        # launch() owns claim, reset, Prime Docker, broker budget, retries, and
        # trusted outcome persistence for both API and benchmark executions.
        core_hash = str(inputs.get("corePlannerHash", self.core_planner_hash))
        image_digest = str(inputs.get("imageDigest", self.image_digest))
        execution_started = time.monotonic()
        self.launch(run.run_id, task_override=task, package_override=package, model_client_override=model_client, seed=seed, arm=arm_value, bundle_hash=durable_bundle.content_hash, core_planner_hash=core_hash, image_digest=image_digest)
        evidence_rows = self.controller.store.list_evidence(run.run_id)
        model_rows = [row for row in evidence_rows if row.get("event_type") == "model_response"]
        outcome_rows = [row for row in evidence_rows if row.get("event_type") == "trusted_outcome"]
        if not model_rows or not outcome_rows:
            raise LearningRuntimeError("evaluation run did not produce trusted model and outcome evidence")
        model_row = model_rows[-1]
        model_payload = self.controller.store.get_artifact(json.loads(model_row["source_ref"])["sha256"])
        if not isinstance(model_payload, Mapping):
            raise LearningRuntimeError("evaluation model receipt is missing")
        run_row = self.controller.store.get_run(run.run_id)
        try:
            run_payload = json.loads(run_row.get("run_json", "{}")) if run_row is not None else {}
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LearningRuntimeError("evaluation run identity is malformed") from exc
        if not isinstance(run_payload, dict):
            raise LearningRuntimeError("evaluation run identity is malformed")
        bound_ref = run_payload.get("finalAccountingRef")
        if bound_ref is not None:
            # A terminal receipt is immutable once bound.  A missing or
            # malformed bound artifact fails closed instead of falling back to
            # a model receipt and silently losing terminal metrics.
            if not isinstance(bound_ref, str) or not bound_ref:
                raise LearningRuntimeError("evaluation terminal accounting reference is malformed")
            accounting_ref = bound_ref
            try:
                accounting = self.controller.store.get_artifact(accounting_ref)
            except KeyError as exc:
                raise LearningRuntimeError("evaluation terminal accounting receipt is missing") from exc
            if not isinstance(accounting, Mapping):
                raise LearningRuntimeError("evaluation terminal accounting receipt is missing")
        else:
            model_accounting_ref = model_payload.get("accountingRef")
            accounting_ref = model_accounting_ref.get("sha256") if isinstance(model_accounting_ref, Mapping) else None
            if not isinstance(accounting_ref, str) or not accounting_ref:
                raise LearningRuntimeError("evaluation model accounting receipt is missing")
            try:
                accounting = self.controller.store.get_artifact(accounting_ref)
            except KeyError as exc:
                raise LearningRuntimeError("evaluation model accounting receipt is missing") from exc
            # Kernel and broker work can finish after the final model response.
            # Publish one immutable accounting artifact for the completed run
            # so replays resolve the exact same terminal metrics.
            if isinstance(accounting, Mapping):
                final_accounting = dict(accounting)
                final_accounting["toolCalls"] = sum(1 for row in evidence_rows if row.get("event_type") == "tool_result")
                final_accounting["durationSeconds"] = max(float(accounting.get("durationSeconds", 0) or 0), time.monotonic() - execution_started)
                final_accounting["inferenceDurationSeconds"] = float(accounting.get("inferenceDurationSeconds", 0) or 0)
                accounting_ref = self.controller.store.put_artifact(final_accounting).sha256
                accounting = final_accounting
                # Bind the terminal artifact to the durable run projection once.
                run_row = self.controller.store.get_run(run.run_id)
                if run_row is not None:
                    try:
                        run_json = json.loads(run_row.get("run_json", "{}"))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        run_json = {}
                    if isinstance(run_json, dict):
                        run_json["finalAccountingRef"] = accounting_ref
                        run_row["run_json"] = json.dumps(run_json, sort_keys=True)
                        self.controller.store.save_run(run.run_id, run_row)
            else:
                raise LearningRuntimeError("evaluation model accounting receipt is missing")
        if (
            accounting.get("responseId") != model_payload.get("responseId")
            or accounting.get("runId") != run.run_id
            or accounting.get("taskId") != task_id
            or accounting.get("environmentId") != env_id
            or accounting.get("arm") != arm_value
            or accounting.get("seed") != seed
            or accounting.get("bundleHash") != bundle_hash
            or accounting.get("versionRefs") != model_payload.get("versionRefs")
        ):
            raise LearningRuntimeError("evaluation terminal accounting receipt is not bound to the model receipt")
        # Evaluator evidence references name the durable evidence row; the
        # row's sourceRef points at the immutable outcome artifact.
        outcome_ref = outcome_rows[-1]["evidence_id"]
        budget = BudgetSpec(model_tokens=int(budget_value.get("modelTokens", DEFAULT_MODEL_TOKENS)), tool_calls=int(budget_value.get("toolCalls", 32)), child_runs=int(budget_value.get("childRuns", 0)), wall_time_seconds=int(budget_value.get("wallTimeSeconds", 90)), cost_microunits=int(budget_value.get("costMicrounits", 100000)), currency=str(budget_value.get("currency", "USD")))
        outcome = self.controller.store.get_outcome_by_run_id(run.run_id) or {}
        passed = bool(outcome.get("passed"))
        try:
            outcome_meta = json.loads(outcome.get("metadata_json", "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            outcome_meta = {}
        reliable = bool(outcome_meta.get("reliable", passed)) if isinstance(outcome_meta, Mapping) else passed
        safety_violations = int(outcome_meta.get("safetyViolations", 0) or 0) if isinstance(outcome_meta, Mapping) else 0
        version_refs = model_payload.get("versionRefs") if isinstance(model_payload, Mapping) else None
        if not isinstance(version_refs, Mapping):
            raise LearningRuntimeError("evaluation model receipt lacks frozen config hashes")
        policy_ref = _manifest_value(package.manifest.policy_ref)
        schemas = tuple(_manifest_value(schema) for schema in package.manifest.tool_schemas)
        observation_kwargs = {"provenance": Provenance.DETERMINISTIC_SIMULATION, "model_provenance": ModelProvenance.REAL_MODEL, "model_profile": model_name, "core_planner_hash": core_hash, "budget": budget, "response_id": model_payload.get("responseId"), "accounting_ref": accounting_ref, "evidence_ref": model_row["evidence_id"], "outcome_ref": outcome_ref, "config_hashes": {"model": sha256_json({"profile": inputs.get("modelProfile", model_name), "provider": inputs.get("provider", provider_name)}), "planner": str(inputs.get("corePlannerHash", core_hash)), "budget": sha256_json(inputs.get("runBudget", budget_value)), "policy": sha256_json(policy_ref), "schema": sha256_json(schemas), "image": str(inputs.get("imageDigest", image_digest))}, "run_id": run.run_id}
        # Session-6's evaluator model includes bundle_hash; keep this worker
        # compatible with the pre-merge evaluator while exposing it whenever
        # the authoritative type is present.
        try:
            from dataclasses import fields
            if any(field.name == "bundle_hash" for field in fields(RunObservation)):
                observation_kwargs["bundle_hash"] = durable_bundle.content_hash
        except TypeError:
            pass
        duration = accounting.get("durationSeconds")
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(float(duration)) or duration < 0:
            raise LearningRuntimeError("evaluation accounting lacks complete duration")
        return RunObservation(task_id, env_id, Partition(getattr(getattr(task, "partition", None), "value", getattr(task, "partition", "development"))), seed, Arm(arm_value), passed, reliable, safety_violations, _effective_observation_cost(accounting), float(duration), **observation_kwargs)

    def run_evaluation_job(self, task: Mapping[str, Any], frozen_config: Mapping[str, Any] | None, bundle: Mapping[str, Any] | None) -> None:
        """Run one trusted evaluation task and durably record its result.

        The executor is injected by the evaluator owner.  Without that
        binding, the task is marked blocked with an honest provider/evaluator
        diagnostic rather than pretending that a queued row is an evaluation.
        """
        evaluation_id = task.get("evaluationId")
        if not isinstance(evaluation_id, str):
            return
        stored = self.controller.store.get_evaluation_queue(evaluation_id)
        if not stored:
            return
        try:
            if self.evaluation_executor is None:
                raise LearningRuntimeError("trusted evaluation provider/evaluator is not configured")
            result = self.evaluation_executor(task=task, frozen_config=frozen_config, bundle=bundle)
            if not isinstance(result, Mapping):
                raise LearningRuntimeError("trusted evaluation executor returned a non-object")
            report = dict(result)
            report.setdefault("evaluationId", evaluation_id)
            report.setdefault("state", "completed")
            validity = str(report.get("validity", report.get("state", "completed")))
        except Exception as exc:
            report = {**json.loads(stored["payload_json"]), "state": "blocked", "trusted": False, "error": str(exc)}
            validity = "blocked"
        if validity == "blocked":
            self.controller.store.save_evaluation_queue(
                evaluation_id,
                {**stored, "state": "blocked", "payload_json": json.dumps(report, sort_keys=True), "updated_at": datetime.now(timezone.utc).isoformat()},
            )
            return
        self.controller.store.save_evaluation(
            evaluation_id,
            {
                "candidate_hash": stored["candidate_hash"],
                "base_hash": stored["base_hash"],
                "protocol_hash": stored["protocol_hash"],
                "partition_ref": stored["partition_ref"],
                "report_json": json.dumps(report, sort_keys=True),
                "validity": validity,
            },
        )
        self.controller.store.save_evaluation_queue(
            evaluation_id,
            {**stored, "state": "completed", "payload_json": json.dumps(report, sort_keys=True), "updated_at": datetime.now(timezone.utc).isoformat()},
        )

    def build_evaluation_driver(self, protocol: Any, arm_bundles: Mapping[Any, Any] | None = None, *, owner_id: str | None = None) -> Any:
        """Wire the evaluator-owned resumable driver to this task executor."""
        from adaptive_agent.benchmark import ResumableEvaluationDriver

        self._evaluation_protocol = protocol
        active = self.controller.get_active_bundle()
        if active is None:
            raise LearningRuntimeError("no active bundle is available for evaluation")
        selected = dict(arm_bundles or {})
        selected.setdefault("B0", active)
        arm_hashes: dict[str, str] = {}
        for key, value in selected.items():
            arm_name = getattr(key, "value", str(key))
            content_hash = getattr(value, "content_hash", None)
            if not isinstance(content_hash, str) or not content_hash:
                raise LearningRuntimeError(f"evaluation arm bundle {arm_name!r} has no content hash")
            arm_hashes[arm_name] = content_hash
        self._evaluation_arm_bundles = arm_hashes
        self._evaluation_base_bundle_hash = arm_hashes["B0"]
        base = next(value for key, value in selected.items() if getattr(key, "value", str(key)) == "B0")
        return ResumableEvaluationDriver(
            self.controller.store,
            protocol,
            self.packages,
            self.execute_evaluation_task,
            base,
            arm_bundles=selected,
            owner_id=owner_id,
        )

    def build_evaluation_job(self, protocol: Any, arm_bundles: Mapping[Any, Any], *, total_budget_microunits: int | None = None) -> Any:
        """Build the production evaluation job around this runtime's executor.

        The returned job uses the same controller, durable store, package
        instances, and bound ``execute_evaluation_task`` callback as the HTTP
        runtime.  This keeps CLI or application factories from accidentally
        substituting a synthetic evaluator or a second model invocation path.
        """
        from adaptive_agent.evaluation import Arm
        from adaptive_agent.evaluation_job import build_evaluation_job
        from adaptive_agent.models import PromotionGate

        self._evaluation_protocol = protocol

        active = self.controller.get_active_bundle()
        if active is None:
            raise LearningRuntimeError("no active bundle is available for evaluation")
        selected = dict(arm_bundles)
        if Arm.B0 not in selected and Arm.B0.value not in selected:
            selected[Arm.B0] = active
        arm_hashes: dict[str, str] = {}
        for key, value in selected.items():
            arm_name = getattr(key, "value", str(key))
            content_hash = getattr(value, "content_hash", None)
            if not isinstance(content_hash, str) or not content_hash:
                raise LearningRuntimeError(f"evaluation arm bundle {arm_name!r} has no content hash")
            arm_hashes[arm_name] = content_hash
        self._evaluation_arm_bundles = arm_hashes
        self._evaluation_base_bundle_hash = arm_hashes[Arm.B0.value]
        frozen = protocol.start_candidate_generation()
        evaluator_refs = sorted({
            self.packages[name].manifest.evaluator_ref.id
            for name in protocol.known_environments
        })
        phase_environments = {
            "validation": tuple(protocol.known_environments),
            "final": (*protocol.known_environments, protocol.sealed_environment),
        }
        phase_evaluator_refs = {
            phase: sorted(self.packages[name].manifest.evaluator_ref.id for name in environments)
            for phase, environments in phase_environments.items()
        }
        gate_config = protocol.gate_config
        if self.controller.store.get_frozen_protocol(frozen.protocol_hash) is None:
            self.controller.candidates.freeze_protocol(
                PromotionGate(
                    protocolHash=frozen.protocol_hash,
                    minBalancedAccuracyGain=gate_config.min_accuracy_gain,
                    ciLowerBound=gate_config.ci_lower_bound,
                    maxCostRatio=gate_config.max_cost_ratio,
                    maxLatencyRatio=gate_config.max_latency_ratio,
                    maxCostMicrounits=gate_config.max_cost_microunits,
                    maxLatencySeconds=gate_config.max_latency_seconds,
                    requirePerEnvironmentNonRegression=gate_config.require_per_environment_non_regression,
                ),
                evaluator_id="|".join(evaluator_refs),
                evaluator_refs=evaluator_refs,
                fixture_hashes=dict(frozen.fixture_hashes),
                partition_hashes={
                    f"{name}:validation": frozen.partition_hashes[f"{name}:validation"]
                    for name in protocol.known_environments
                },
                protocol_inputs=dict(frozen.inputs),
                phase_evaluator_refs=phase_evaluator_refs,
            )
        # CandidateManager consumes the evaluator's serialized report contract.
        # Bind its verifier to the same durable attestation ledger used by the
        # runtime's evaluation read path before any promotion decision runs.
        self.controller.candidates.report_verifier = self._verify_promotion_report
        return build_evaluation_job(
            self.controller.store,
            self.controller,
            protocol,
            self.packages,
            selected,
            self.execute_evaluation_task,
            total_budget_microunits=total_budget_microunits,
        )

    def _verify_promotion_report(self, report: Mapping[str, Any]) -> bool:
        """Verify a candidate report against frozen inputs and the durable ledger."""
        if not isinstance(report, Mapping):
            return False
        try:
            protocol_hash = report.get("protocolHash")
            if not isinstance(protocol_hash, str) or not protocol_hash:
                return False
            frozen = self.controller.store.get_frozen_protocol(protocol_hash)
            if frozen is None:
                return False
            inputs = json.loads(frozen["protocol_inputs_json"] or "{}")
            known_value = inputs.get("knownEnvironments")
            if not isinstance(known_value, list) or not known_value or any(not isinstance(name, str) or not name for name in known_value):
                return False
            known = tuple(known_value)
            validation_key = f"{known[0]}:validation"
            partition_hashes = inputs.get("partitionHashes")
            if not isinstance(partition_hashes, Mapping):
                return False
        except (AttributeError, IndexError, KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError):
            return False
        row = {
            "protocol_hash": protocol_hash,
            "candidate_hash": report.get("candidateHash"),
            "base_hash": report.get("baseHash"),
            "partition_ref": {"id": "validation", "sha256": partition_hashes[validation_key]},
        }
        return self._verify_evaluation_report(report, row)

    def _verify_evaluation_report(self, report: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
        """Verify a serialized report against the durable evaluator ledger."""
        if not isinstance(report, Mapping):
            return False
        try:
            protocol_hash = row.get("protocol_hash")
            if not isinstance(protocol_hash, str) or not protocol_hash:
                return False
            frozen = self.controller.store.get_frozen_protocol(protocol_hash)
            if frozen is None:
                return False
            if row.get("protocol_hash") != protocol_hash or report.get("protocolHash") != protocol_hash:
                return False
            if report.get("candidateHash") != row.get("candidate_hash") or report.get("baseHash") != row.get("base_hash"):
                return False
            if report.get("validityStatus") != "valid":
                return False
            partition_ref = row.get("partition_ref")
            if isinstance(partition_ref, str):
                partition_ref = json.loads(partition_ref)
            if not isinstance(partition_ref, Mapping) or partition_ref.get("id") not in {"validation", "final"}:
                return False
            phase = str(partition_ref["id"])
            inputs = json.loads(frozen["protocol_inputs_json"] or "{}")
            known_value = inputs.get("knownEnvironments")
            sealed = inputs.get("sealedEnvironment")
            if not isinstance(known_value, list) or not known_value or any(not isinstance(name, str) or not name for name in known_value) or not isinstance(sealed, str) or not sealed:
                return False
            known = tuple(known_value)
            environments = known if phase == "validation" else (*known, sealed)
            safety_case_ids = inputs.get("safetyCaseIds")
            if not isinstance(safety_case_ids, list) or any(not isinstance(case_id, str) or not case_id for case_id in safety_case_ids) or len(set(safety_case_ids)) != len(safety_case_ids):
                return False
            if report.get("comparison") != phase:
                return False
            if report.get("expectedEnvironments") != list(environments):
                return False
            if report.get("requiredSafetyCaseIds") != safety_case_ids:
                return False
            frozen_partitions = inputs.get("partitionHashes")
            if not isinstance(frozen_partitions, Mapping):
                return False
            expected_keys = tuple(f"{name}:{phase}" for name in environments)
            expected_partitions = {key: frozen_partitions[key] for key in expected_keys}
            if not expected_partitions:
                return False
            if report.get("partitionHashes") != expected_partitions:
                return False
            first_partition = next(iter(expected_partitions))
            if partition_ref.get("sha256") != expected_partitions[first_partition]:
                return False
            phase_refs = json.loads(frozen["phase_evaluator_refs_json"] or "{}")
            phase_ref_values = phase_refs.get(phase)
            if not isinstance(phase_ref_values, list) or any(not isinstance(value, str) or not value for value in phase_ref_values):
                return False
            expected_refs = tuple(sorted(phase_ref_values))
            refs = report.get("evaluatorRefs")
            if not isinstance(refs, (list, tuple)) or tuple(sorted(str(value) for value in refs)) != expected_refs:
                return False
            attestation_payload = {
                "comparison": report.get("comparison"),
                "candidateHash": report.get("candidateHash"),
                "baseHash": report.get("baseHash"),
                "protocolHash": report.get("protocolHash"),
                "partitionHashes": report.get("partitionHashes"),
                "evaluatorRefs": report.get("evaluatorRefs"),
                "environmentCells": report.get("environmentCells"),
                "armSummaries": report.get("armSummaries"),
                "confidenceIntervals": report.get("confidenceIntervals"),
                "validityStatus": report.get("validityStatus"),
                "safetyPassed": report.get("safetyPassed"),
                "safetyCaseResults": report.get("safetyCaseResults"),
                "safetyProbeOutputs": report.get("safetyProbeOutputs"),
                "missingPairs": report.get("missingPairs"),
                "partitionLeak": report.get("partitionLeak"),
                "invalidFixtureResets": report.get("invalidFixtureResets"),
                "infrastructureFailures": report.get("infrastructureFailures"),
                "metricCellsComplete": report.get("metricCellsComplete"),
                "safetyCellsComplete": report.get("safetyCellsComplete"),
                "modelProvenanceComplete": report.get("modelProvenanceComplete"),
                "auxiliarySummaries": report.get("auxiliarySummaries"),
                "auxiliaryOverhead": report.get("auxiliaryOverhead"),
                "auxiliaryExposure": report.get("auxiliaryExposure"),
                "auxiliaryLimitations": report.get("auxiliaryLimitations"),
                "gateConfig": report.get("gateConfig"),
                "expectedEnvironments": report.get("expectedEnvironments"),
                "requiredSafetyCaseIds": report.get("requiredSafetyCaseIds"),
            }
            required = tuple(attestation_payload)
            if any(key not in report for key in required):
                return False
            ledger = self.controller.evaluator_adapters[0]
            registry = TrustedEvaluatorRegistry(ledger)
            for name in environments:
                package = self.packages.get(name)
                if package is None:
                    return False
                registry.register(package)
            token = report.get("attestation")
            return registry.verify(token if isinstance(token, str) else None, attestation_payload)
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def list_evaluations(self) -> list[dict[str, Any]]:
        out = []
        report_ids: set[str] = set()
        queue_rows = {
            str(row["evaluation_id"]): row
            for row in self.controller.store.list_evaluation_queue()
            if isinstance(row.get("evaluation_id"), str)
        }

        def public_job(payload: Mapping[str, Any], *, evaluation_id: str | None = None, row: Mapping[str, Any] | None = None) -> dict[str, Any]:
            value = dict(payload)
            if evaluation_id is not None:
                value.setdefault("evaluationId", evaluation_id)
            if row is not None:
                for output_key, row_key in (("candidateId", "candidate_id"), ("baseBundleHash", "base_hash"), ("protocolHash", "protocol_hash"), ("state", "state")):
                    if output_key not in value and row.get(row_key) is not None:
                        value[output_key] = row[row_key]
            if "candidate_id" in value:
                value["candidateId"] = value.pop("candidate_id")
            if "base_hash" in value:
                value["baseBundleHash"] = value.pop("base_hash")
            if "protocol_hash" in value:
                value["protocolHash"] = value.pop("protocol_hash")
            if "validity" in value and "validityStatus" not in value:
                value["validityStatus"] = value.pop("validity")
            canonical = bool(row is not None and self._verify_evaluation_report(value, row))
            if "candidateId" not in value:
                candidate_hash = value.get("candidateHash")
                if isinstance(candidate_hash, str):
                    candidate = self.controller.store.get_candidate_by_bundle_hash(candidate_hash)
                    if candidate is not None:
                        value["candidateId"] = candidate.get("candidate_id")
            raw_state = value.get("state")
            if raw_state in {"completed", "complete", "decided", "failed", "error", "incomplete"}:
                value["state"] = "valid" if canonical else "invalid"
            elif raw_state not in {"queued", "running", "valid", "invalid", "cancelled"}:
                validity = value.get("validityStatus")
                value["state"] = "valid" if canonical and validity == "valid" else "invalid"
            value["trusted"] = canonical
            if not canonical:
                value.setdefault("trustReason", "unverified evaluator report")
                if "reason" not in value:
                    error = value.get("error")
                    if isinstance(error, Mapping) and isinstance(error.get("message"), str):
                        value["reason"] = error["message"]
                    elif isinstance(error, str) and error:
                        value["reason"] = error
                    else:
                        value["reason"] = value["trustReason"]
            if isinstance(payload.get("comparison"), str) and payload["comparison"] in {"validation", "final"}:
                safe_keys = {
                    "comparison", "validityStatus", "promotionEligible", "candidateHash", "baseHash", "protocolHash",
                    "armSummaries", "confidenceIntervals", "safetyPassed", "missingPairs", "metricCellsComplete",
                    "safetyCellsComplete", "modelProvenanceComplete", "infrastructureFailures", "analysisSeed",
                    "nominalCostUsd", "actualInputTokens", "actualOutputTokens", "wallDurationSeconds", "billingBasis",
                }
                value["report"] = {key: value[key] for key in safe_keys if key in value}
                for key in ("attestation", "exposure", "safetyProbeOutputs", "environmentCells", "workload"):
                    value.pop(key, None)
            return value

        for row in self.controller.store.list_evaluations():
            try:
                report = json.loads(row["report_json"])
                if isinstance(report, dict):
                    report_id = str(row.get("report_id", report.get("evaluationId", "")))
                    # The report table stores evaluator output; queue metadata
                    # owns the operator-facing candidate and lifecycle fields.
                    metadata = queue_rows.get(report_id)
                    normalized = public_job(report, evaluation_id=report_id or None, row=metadata or row)
                    out.append(normalized)
                    if report_id:
                        report_ids.add(report_id)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        for row in queue_rows.values():
            try:
                queued = json.loads(row["payload_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(queued, dict) and row["evaluation_id"] not in report_ids:
                out.append(public_job(queued, evaluation_id=row["evaluation_id"], row=row))
        return out

    def active_versions(self) -> list[dict[str, Any]]:
        active = self.controller.store.get_active_bundle()
        current = active.get("content_hash") if active else None
        return [{"bundleHash": value, "state": "active" if value == current else "lineage"} for value in self.controller.store.list_active_lineage()]

    def rollback_candidate(self, candidate_id: str, reason: str) -> dict[str, Any]:
        candidate = self.controller.get_candidate(candidate_id)
        if candidate is None:
            raise KeyError("candidate not found")
        decision = self.controller.rollback(candidate["base_bundle_hash"], reason)
        return decision.model_dump(mode="json", by_alias=True)

    def launch_learning(self, payload: Any) -> dict[str, Any]:
        # Single-run callers send runId only; multi-run learning declares the
        # bounded source set via runIds.  Every listed run must be a completed
        # development run with a trusted outcome — invalid sources fail closed.
        raw_ids = getattr(payload, "run_ids", None) or []
        if not isinstance(raw_ids, (list, tuple)) or any(not isinstance(run_id, str) or not run_id for run_id in raw_ids):
            raise KeyError("learning source set is malformed")
        requested = [payload.run_id] if getattr(payload, "run_id", None) else []
        requested += list(raw_ids)
        source_ids = list(dict.fromkeys(requested))
        if not source_ids:
            raise KeyError("run not found")
        # Reject oversized source sets before any evidence is materialized.
        if len(source_ids) > LEARNING_SOURCE_RUN_CAP:
            raise KeyError("learning source set exceeds the frozen bound")
        store = self.controller.store
        bindings: list[tuple[str, str]] = []
        for run_id in source_ids:
            stored_run = store.get_run(run_id)
            if not isinstance(stored_run, Mapping) or stored_run.get("status") not in {"succeeded", "failed", "cancelled", "timed_out", "outcome_unknown"}:
                raise KeyError("learning source is not a completed run")
            source_task = store.get_task(stored_run["task_id"])
            if not isinstance(source_task, Mapping) or source_task.get("partition") != "development":
                raise KeyError("learning source is not a development run")
            if store.get_outcome_by_run_id(run_id) is None:
                raise KeyError("learning source lacks a trusted outcome")
            bindings.append((str(stored_run["environment_id"]), run_id))
        # Deterministic exposure order: (environment, run); an explicit runId
        # caller keeps that run as the proposal's primary identity.
        ordered_ids = [run_id for _env, run_id in sorted(bindings)]
        primary_id = payload.run_id if getattr(payload, "run_id", None) else ordered_ids[0]
        stored = store.get_run(primary_id)
        run = self.controller.get_run(primary_id)
        if stored is None or run is None:
            raise KeyError("run not found")
        package = self.packages.get(stored["environment_id"])
        task = self.registry.get_task(stored["task_id"])
        if package is None or task is None:
            raise KeyError("development task not found")
        if self._learning_runtime is None:
            model_client = self.learning_model_client if self.learning_model_client is not None else (self.model_runner if hasattr(self.model_runner, "invoke") else None)
            self._learning_runtime = LearningRuntime.build(store=self.controller.store, manager=self.controller.candidates, model_client=model_client)
        before = {
            row.get("evidence_id")
            for row in self.controller.store.list_evidence(primary_id)
            if row.get("event_type") == "learning_model_observation"
        }
        goal = task["goal"] if isinstance(task, Mapping) else None
        status_feedback = run.status.value
        if len(ordered_ids) > 1:
            propose_multi = getattr(self._learning_runtime, "propose_completed_runs", None)
            if not callable(propose_multi):
                raise LearningRuntimeError("runtime lacks the multi-run learning seam")
            proposal = propose_multi(ordered_ids, primary_run_id=primary_id, goal=goal, feedback={"status": status_feedback})
        else:
            proposal = self._learning_runtime.propose_completed_run(primary_id, goal=goal, feedback={"status": status_feedback})
        learning_rows = [
            row for row in self.controller.store.list_evidence(primary_id)
            if row.get("event_type") == "learning_model_observation" and row.get("evidence_id") not in before
        ]
        learning_accounting = {"wallSeconds": 0.0, "accountingComplete": bool(learning_rows)}
        nominal_total = 0.0
        nominal_seen = False
        cost_total = 0
        cost_complete = bool(learning_rows)
        for row in learning_rows:
            try:
                source = json.loads(row["source_ref"])
                observation = self.controller.store.get_artifact(source["sha256"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                learning_accounting["accountingComplete"] = False
                continue
            if not isinstance(observation, Mapping):
                learning_accounting["accountingComplete"] = False
                continue
            duration = observation.get("durationSeconds")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration >= 0:
                learning_accounting["wallSeconds"] += float(duration)
            else:
                learning_accounting["accountingComplete"] = False
            nominal = observation.get("nominalCostUsd")
            if isinstance(nominal, (int, float)) and not isinstance(nominal, bool) and nominal >= 0:
                nominal_total += float(nominal)
                nominal_seen = True
            else:
                learning_accounting["accountingComplete"] = False
            cost = observation.get("costMicrounits")
            status = observation.get("economicCostStatus")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0 and status != "unknown":
                cost_total += int(cost)
            else:
                cost_complete = False
        if nominal_seen:
            learning_accounting["nominalCostUsd"] = nominal_total
        if cost_complete:
            learning_accounting["costMicrounits"] = cost_total
        else:
            learning_accounting["economicCostStatus"] = "unknown"
        candidate = dict(proposal.authoritative_candidate)
        if "candidate_id" in candidate:
            candidate["candidateId"] = candidate.pop("candidate_id")
        result = {"actionId": f"learn_{__import__('uuid').uuid4().hex}", "runId": primary_id, "predictedEffect": proposal.candidate_payload["predictedEffect"], "evidenceIds": proposal.candidate_payload["supportingEvidenceIds"], "proposalRef": self.controller.store.put_artifact(proposal.bundle_patch).model_dump(mode="json", by_alias=True), "candidate": candidate, "status": "staged", "createdAt": __import__("adaptive_agent.api", fromlist=["_now"])._now(), **learning_accounting}
        if len(ordered_ids) > 1:
            result["sourceRunIds"] = ordered_ids
        return result

    def learning_runtime(self, *, environment_id: str | None = None, run_id: str) -> dict[str, Any]:
        """Return the narrow learner context for one durable development run.

        The Store owns the projection and its visibility rules.  This adapter
        only binds the requested run to its registered environment before
        returning the public documents and trusted development evidence.
        """
        store = self.controller.store
        run = store.get_run(run_id)
        if run is None:
            raise KeyError("run not found")
        if environment_id is None:
            environment_id = run.get("environment_id")
        if not isinstance(environment_id, str) or not environment_id:
            raise ValueError("run environment binding is missing")
        environment = store.get_environment(environment_id)
        if environment is None:
            raise KeyError("environment not found")
        if run.get("environment_id") != environment_id:
            raise ValueError("run does not belong to environment")
        task_id = run.get("task_id")
        task = store.get_task(task_id) if isinstance(task_id, str) else None
        if task is None:
            raise KeyError("run task not found")
        if task.get("environment_id") != environment_id or task.get("partition") != "development":
            raise ValueError("learning runtime requires a development task")
        public_docs = store.get_public_docs(environment_id)
        # Keep this API on the unified, sanitized Store projection.  It owns
        # the learner visibility and development-partition filters, including
        # exclusion of operator and evaluator-only evidence.
        development_evidence = store.list_learning_evidence(environment_id=environment_id, run_id=run_id)
        return {
            "environmentId": environment_id,
            "runId": run_id,
            "publicDocs": public_docs,
            "developmentEvidence": development_evidence,
        }

    def list_environments(self) -> list[dict[str, Any]]:
        registered = {task.environment_ref.id for task in self._tasks.values()}
        return [{"environmentId": name, "version": package.manifest.version, "validationState": "valid", "evaluatorReady": callable(getattr(package, "evaluate", None)) and not isinstance(package, _RegisteredPackage), "toolCount": len(package.manifest.tool_schemas), "policyScope": package.manifest.policy_ref.id, "executionModes": list(package.manifest.execution_modes), "capabilities": list(package.manifest.capabilities)} for name, package in self.packages.items() if name in registered]

    def list_tasks(self, environment_id: str) -> list[dict[str, Any]]:
        package = self.packages.get(environment_id)
        if package is None:
            return []
        return [{
            "taskId": task.task_id,
            "goal": task.goal,
            "environmentRef": task.environment_ref.model_dump(mode="json", by_alias=True),
            "allowedInputRefs": [ref.model_dump(mode="json", by_alias=True) for ref in task.allowed_input_refs],
            "executionModes": list(package.manifest.execution_modes),
        } for task in self._tasks.values() if task.environment_ref.id == environment_id]

    def _public_run(self, run: Any, task: Any) -> dict[str, Any]:
        value = run.model_dump(mode="json", by_alias=True)
        if value.get("outcomeRef") is None:
            value.pop("outcomeRef", None)
        value.update({"environmentId": task.environment_ref.id, "goal": task.goal})
        status = getattr(run.status, "value", run.status)
        partition = getattr(getattr(task, "partition", None), "value", getattr(task, "partition", None))
        value["learningEligible"] = bool(
            partition == "development"
            and status in {"succeeded", "failed", "cancelled", "timed_out", "outcome_unknown"}
            and self.controller.store.get_outcome_by_run_id(run.run_id) is not None
            and self.controller.store.has_trusted_outcome(run.run_id)
        )
        return value

    def create_run(self, payload: Any) -> dict[str, Any]:
        if payload.execution_mode not in {mode for package in self.packages.values() for mode in package.manifest.execution_modes}:
            raise ValueError("execution mode is not declared by environment")
        task = None
        if isinstance(payload.task_ref, Mapping):
            task_id = payload.task_ref.get("id")
            if isinstance(task_id, str):
                task = self._tasks.get(task_id)
        if task is None and payload.goal:
            task = next((candidate for candidate in self._tasks.values() if candidate.goal == payload.goal and candidate.environment_ref.id == payload.environment_id), None)
        if task is None:
            raise KeyError("task is not registered for this environment and goal")
        if isinstance(payload.task_ref, Mapping) and isinstance(payload.task_ref.get("id"), str):
            stored_task = self.controller.store.get_task(payload.task_ref["id"])
            if stored_task is None:
                raise KeyError("task reference is not registered")
            if payload.task_ref.get("version") is not None or payload.task_ref.get("sha256") is not None:
                registered_ref = json.loads(stored_task["task_ref"])
                if payload.task_ref.get("version") != registered_ref.get("version") or payload.task_ref.get("sha256") != registered_ref.get("sha256"):
                    raise ValueError("task reference does not match the registered task artifact")
        if payload.environment_id is not None and task.environment_ref.id != payload.environment_id:
            raise ValueError("task does not belong to the requested environment")
        if payload.goal is not None and task.goal != payload.goal:
            raise ValueError("goal does not match the registered task")
        package = self.packages[task.environment_ref.id]
        if payload.execution_mode not in package.manifest.execution_modes:
            raise ValueError("execution mode is not declared by environment")
        from adaptive_agent.models import ArtifactRef, RunRequest

        task_ref = ArtifactRef(id=task.task_id, version="1", sha256=__import__("hashlib").sha256(task.task_id.encode()).hexdigest())
        model_payload = payload.model_profile_ref or self.model_ref
        if self.control_plane is not None:
            self.control_plane._reference_key(model_payload, "model")
        model_ref = ArtifactRef.model_validate(model_payload)
        if payload.budget is not None:
            budget_ref = self.controller.store.put_artifact(payload.budget.model_dump(mode="json", by_alias=True))
        else:
            budget_payload = payload.budget_ref or self.budget_ref
            if self.control_plane is not None:
                self.control_plane._reference_key(budget_payload, "budget")
            budget_ref = ArtifactRef.model_validate(budget_payload)
        active_refs = [ArtifactRef.model_validate(ref) for ref in getattr(payload, "active_skill_refs", [])]
        existing = self.controller.store.get_run_by_idempotency_key(payload.idempotency_key)
        if existing is not None:
            prior = self.controller.get_run(existing["run_id"])
            if prior is None or any((prior.task_ref != task_ref, prior.model_profile_ref != model_ref, prior.budget_ref != budget_ref, prior.execution_mode != payload.execution_mode, prior.active_skill_refs != active_refs)):
                raise IdempotencyConflict(existing["run_id"])
            return self._public_run(prior, task)
        run = self.controller.create_run(RunRequest(taskRef=task_ref, modelProfileRef=model_ref, budgetRef=budget_ref, idempotencyKey=payload.idempotency_key, executionMode=payload.execution_mode, activeSkillRefs=active_refs), task)
        return self._public_run(run, task)

    def _active_skills(self, run_id: str) -> list[dict[str, Any]]:
        run = self.controller.get_run(run_id)
        if run is None:
            return []
        try:
            bundle = __import__("adaptive_agent.models", fromlist=["SkillBundle"]).SkillBundle.model_validate(self.controller.store.get_artifact(run.skill_bundle_ref))
        except (KeyError, ValueError, TypeError):
            return []
        selected = {ref.id for ref in run.active_skill_refs}
        return [skill.model_dump(mode="json", by_alias=True) for skill in bundle.skills if not selected or skill.skill_id in selected]

    def _planner_environment(self, package: Any, run_id: str) -> dict[str, Any]:
        run = self.controller.get_run(run_id)
        return {
            "environmentId": package.environment_id,
            "version": package.manifest.version,
            "docs": [_manifest_value(ref) for ref in package.manifest.docs],
            "publicDocs": [_manifest_value(doc) for doc in package.learner_documents()],
            "toolSchemas": [_manifest_value(tool) for tool in package.manifest.tool_schemas],
            "executionModes": list(package.manifest.execution_modes),
            "executionMode": run.execution_mode if run else "interactive",
            "capabilities": [f"{run_id}:{tool.name}" for tool in package.manifest.tool_schemas],
            "activeSkills": self._active_skills(run_id),
            "budgetRef": run.budget_ref.model_dump(mode="json", by_alias=True) if run else None,
        }

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self.controller.get_run(run_id)
        if run is None:
            return None
        row = self.controller.store.get_run(run_id)
        task = self.registry.get_task(row["task_id"]) if row else None
        return self._public_run(run, task) if task else run.model_dump(mode="json", by_alias=True)

    def list_runs(self) -> list[dict[str, Any]]:
        with self.controller.store.connect() as conn:
            rows = conn.execute("SELECT run_id FROM runs ORDER BY created_at").fetchall()
        return [run for row in rows if (run := self.get_run(row[0])) is not None]

    def _record_model_response(self, run_id: str, package: Any, evidence: Mapping[str, Any]) -> Any:
        """Persist parent-owned model evidence with frozen execution pins.

        ``receipts`` may contain successful and failed retry responses.  We
        retain every receipt and aggregate usage/cost/duration instead of
        silently dropping failed attempts or manufacturing a zero duration.
        """
        stored = self.controller.store.get_run(run_id)
        run = self.controller.get_run(run_id)
        if stored is None or run is None:
            raise KeyError("run not found")
        response_id = evidence.get("responseId") or evidence.get("response_id")
        if not isinstance(response_id, str) or not response_id.strip():
            raise ValueError("model response evidence requires responseId")
        provider = evidence.get("provider")
        model = evidence.get("modelProfile") or evidence.get("model")
        if not isinstance(provider, str) or not provider.strip() or not isinstance(model, str) or not model.strip():
            raise ValueError("model response evidence requires provider and modelProfile")
        raw_receipts = evidence.get("receipts")
        receipts_input = raw_receipts if isinstance(raw_receipts, list) and raw_receipts else [evidence]
        receipts: list[dict[str, Any]] = []
        cache_keys = ("cacheReadInputTokens", "cacheCreationInputTokens", "cachedInputTokens")
        aggregate_inference = 0.0
        now = time.monotonic()
        started = self._run_started_at.get(run_id, now)
        previous_receipt = self._run_last_receipt_at.get(run_id, started)
        # Rehydrate the immutable receipt ledger when a runtime process is
        # restarted.  A process-local cache alone would undercount failed
        # model calls and charge only the final resumed response.
        restored_duration = 0.0
        if run_id not in self._run_receipts:
            restored_by_id: dict[str, dict[str, Any]] = {}
            for row in self.controller.store.list_evidence(run_id):
                if row.get("event_type") != "model_response":
                    continue
                try:
                    source = json.loads(row.get("source_ref", "{}"))
                    artifact = self.controller.store.get_artifact(source["sha256"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                prior = artifact.get("receipts") if isinstance(artifact, Mapping) else None
                accounting_ref = artifact.get("accountingRef") if isinstance(artifact, Mapping) else None
                if isinstance(accounting_ref, Mapping) and isinstance(accounting_ref.get("sha256"), str):
                    accounting = self.controller.store.get_artifact(accounting_ref["sha256"])
                    if isinstance(accounting, Mapping) and isinstance(accounting.get("durationSeconds"), (int, float)):
                        restored_duration = max(restored_duration, float(accounting["durationSeconds"]))
                if isinstance(prior, list):
                    for item in prior:
                        if not isinstance(item, Mapping):
                            continue
                        receipt_id = item.get("responseId")
                        if isinstance(receipt_id, str) and receipt_id:
                            restored_by_id[receipt_id] = dict(item)
            self._run_receipts[run_id] = list(restored_by_id.values())
        existing_receipt_ids = {item.get("responseId") for item in self._run_receipts[run_id]}
        for index, raw in enumerate(receipts_input):
            if not isinstance(raw, Mapping):
                raise ValueError("model accounting receipt must be an object")
            raw_usage = raw.get("usage")
            receipt_usage = canonical_usage(raw_usage)
            if isinstance(raw_usage, Mapping):
                for key in cache_keys:
                    value = raw_usage.get(key)
                    if value is None:
                        value = raw_usage.get({
                            "cacheReadInputTokens": "cache_read_input_tokens",
                            "cacheCreationInputTokens": "cache_creation_input_tokens",
                            "cachedInputTokens": "cached_input_tokens",
                        }[key])
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        receipt_usage[key] = value
                if isinstance(raw_usage.get("cost"), Mapping):
                    receipt_usage["cost"] = dict(raw_usage["cost"])
                details = raw_usage.get("prompt_tokens_details") or raw_usage.get("promptTokensDetails")
                if isinstance(details, Mapping) and "cachedInputTokens" not in receipt_usage:
                    value = details.get("cached_tokens")
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        receipt_usage["cachedInputTokens"] = value
            receipt_id = raw.get("responseId") or raw.get("response_id")
            if not isinstance(receipt_id, str) or not receipt_id.strip():
                receipt_id = response_id if index == 0 else f"{response_id}:receipt:{index}"
            duration_value = raw.get("durationSeconds")
            if isinstance(duration_value, bool) or not isinstance(duration_value, (int, float)) or not math.isfinite(duration_value) or duration_value < 0:
                duration_value = max(now - previous_receipt, 1e-6) if index == 0 else 0.0
            cost_value = raw.get("costMicrounits")
            if cost_value is not None:
                if isinstance(cost_value, bool) or not isinstance(cost_value, (int, float)) or not math.isfinite(cost_value) or cost_value < 0:
                    raise ValueError("model accounting costMicrounits must be finite and non-negative")
            nominal_value = raw.get("nominalCostUsd")
            if nominal_value is None and isinstance(raw_usage, Mapping):
                sdk_cost = raw_usage.get("cost")
                if isinstance(sdk_cost, Mapping):
                    nominal_value = sdk_cost.get("total")
            if nominal_value is not None:
                if isinstance(nominal_value, bool) or not isinstance(nominal_value, (int, float)) or not math.isfinite(float(nominal_value)) or float(nominal_value) < 0:
                    raise ValueError("model accounting nominalCostUsd must be finite and non-negative")
            aggregate_inference += float(duration_value)
            receipt_economic_status = raw.get("economicCostStatus")
            if not isinstance(receipt_economic_status, str) or not receipt_economic_status:
                receipt_economic_status = "unknown" if cost_value is None else "measured"
            if receipt_id in existing_receipt_ids:
                continue
            existing_receipt_ids.add(receipt_id)
            receipts.append({"responseId": receipt_id, "usage": receipt_usage, "durationSeconds": float(duration_value), "status": str(raw.get("status", "complete")), "economicCostStatus": receipt_economic_status, **({"costMicrounits": cost_value} if cost_value is not None else {}), **({"nominalCostUsd": float(nominal_value)} if nominal_value is not None else {})})
        self._run_last_receipt_at[run_id] = now
        self._run_receipts.setdefault(run_id, []).extend(receipts)
        all_receipts = list(self._run_receipts.get(run_id, ()))
        aggregate = {"inputTokens": sum(item["usage"]["inputTokens"] for item in all_receipts), "outputTokens": sum(item["usage"]["outputTokens"] for item in all_receipts), "totalTokens": sum(item["usage"]["totalTokens"] for item in all_receipts)}
        for key in cache_keys:
            total = sum(int(item["usage"].get(key, 0)) for item in all_receipts)
            if total:
                aggregate[key] = total
        aggregate_inference = sum(float(item.get("durationSeconds", 0.0)) for item in all_receipts)
        explicit_cost = any("costMicrounits" in item for item in all_receipts)
        aggregate_cost = sum(float(item["costMicrounits"]) for item in all_receipts if "costMicrounits" in item)
        economic_statuses = {str(item.get("economicCostStatus")) for item in all_receipts if isinstance(item.get("economicCostStatus"), str)}
        aggregate_cost_unknown = any(status == "unknown" for status in economic_statuses) or any("costMicrounits" not in item for item in all_receipts)
        nominal_values = [float(item["nominalCostUsd"]) for item in all_receipts if "nominalCostUsd" in item]
        nominal_cost_usd = sum(nominal_values) if nominal_values else None
        nominal_coverage = {"knownReceipts": len(nominal_values), "totalReceipts": len(all_receipts)}
        nominal_status = "complete" if nominal_coverage["knownReceipts"] == nominal_coverage["totalReceipts"] else "partial"
        accounting_cost = aggregate_cost if explicit_cost else (
            round(nominal_cost_usd * 1_000_000) if nominal_status == "complete" and nominal_cost_usd is not None else None
        )
        nominal_proxy_fields = {"costBasis": "nominal_budget_proxy", "billingStatus": "unknown"} if not explicit_cost and nominal_status == "complete" else {}
        usage = dict(canonical_usage(evidence.get("usage")))
        tool_calls = sum(
            1 for row in self.controller.store.list_evidence(run_id)
            if row.get("event_type") == "tool_result"
        )
        if all_receipts:
            usage.update({key: value for key, value in all_receipts[-1]["usage"].items() if key in cache_keys or key == "cost"})
        frozen_core_planner = evidence.get("corePlannerHash") if isinstance(evidence.get("corePlannerHash"), str) and evidence.get("corePlannerHash") else self.core_planner_hash
        frozen_image = evidence.get("imageDigest") if isinstance(evidence.get("imageDigest"), str) and evidence.get("imageDigest") else self.image_digest
        version_refs = {
            "policy": package.manifest.policy_ref.sha256,
            "schema": sha256_json(tuple(_manifest_value(schema) for schema in package.manifest.tool_schemas)),
            "planner": frozen_core_planner,
            "budget": run.budget_ref.sha256,
            "image": frozen_image,
        }
        run_payload: Mapping[str, Any] = {}
        try:
            decoded = json.loads(stored.get("run_json", "{}"))
            if isinstance(decoded, Mapping):
                run_payload = decoded
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        arm = evidence.get("arm") or run_payload.get("arm")
        seed = evidence.get("seed", run_payload.get("seed"))
        bundle_hash = evidence.get("bundleHash") or run_payload.get("bundleHash") or run.skill_bundle_ref.sha256
        if not isinstance(bundle_hash, str) or not bundle_hash:
            raise ValueError("model response evidence requires bundleHash")
        if arm not in {"B0", "L", "A"}:
            raise ValueError("model response evidence requires a valid arm")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("model response evidence requires an integer seed")
        arm_bundles = run_payload.get("armBundles")
        if isinstance(arm_bundles, Mapping) and arm in arm_bundles and arm_bundles.get(arm) != bundle_hash:
            raise ValueError("model response bundleHash does not match the requested arm bundle")
        whole_run_duration = max(now - started, restored_duration, aggregate_inference, 1e-6)
        economic_status = evidence.get("economicCostStatus")
        if not isinstance(economic_status, str) or aggregate_cost_unknown:
            economic_status = "unknown" if aggregate_cost_unknown else ("measured" if explicit_cost else "unknown")
        payload = {
            "runId": run_id,
            "taskId": stored["task_id"],
            "environmentId": stored["environment_id"],
            "responseId": response_id,
            "provider": provider,
            "modelProfile": model,
            "usage": usage,
            "aggregateUsage": aggregate,
            "receipts": all_receipts,
            "arm": arm,
            "seed": seed,
            "bundleHash": bundle_hash,
            "budgetRef": run.budget_ref.model_dump(mode="json", by_alias=True),
            "imageDigest": frozen_image,
            "corePlannerHash": frozen_core_planner,
            "versionRefs": version_refs,
            "planner": {"responseId": response_id, "modelProfile": model, "corePlannerHash": frozen_core_planner, "versionRefs": version_refs, "bundleHash": bundle_hash, "arm": arm, "seed": seed},
            "nominalCostUsd": nominal_cost_usd,
            "nominalCostStatus": nominal_status,
            "nominalCostCoverage": nominal_coverage,
            **nominal_proxy_fields,
        }
        accounting = {
            "responseId": response_id,
            "runId": run_id,
            "taskId": stored["task_id"],
            "environmentId": stored["environment_id"],
            "usage": usage,
            "aggregateUsage": aggregate,
            "receipts": all_receipts,
            "arm": arm,
            "seed": seed,
            "bundleHash": bundle_hash,
            "versionRefs": version_refs,
            "costMicrounits": accounting_cost,
            "economicCost": {"status": economic_status, "microunits": aggregate_cost if explicit_cost and not aggregate_cost_unknown else None, "coverage": {"knownReceipts": sum(1 for item in all_receipts if "costMicrounits" in item), "totalReceipts": len(all_receipts)}},
            "nominalCostUsd": nominal_cost_usd,
            "nominalCostStatus": nominal_status,
            "nominalCostCoverage": nominal_coverage,
            "durationSeconds": whole_run_duration,
            "inferenceDurationSeconds": aggregate_inference,
            "toolCalls": tool_calls,
            **nominal_proxy_fields,
        }
        accounting_ref = self.controller.store.put_artifact(accounting)
        payload["accountingRef"] = accounting_ref.model_dump(mode="json", by_alias=True)
        return self.controller.record_model_response(run_id, payload)

    def _claim_run(self, run_id: str) -> tuple[bool, Any | None]:
        """Claim a queued run through whichever canonical seam is present.

        The integrated controller historically exposed ``claim_run``.  The
        canonical core deliberately keeps that policy at its caller boundary,
        so this adapter performs the same queued-to-running transition before
        invoking ``execute_run`` when the helper is absent.
        """
        claim = getattr(self.controller, "claim_run", None)
        if callable(claim):
            return claim(run_id)
        with self._launch_claim_lock:
            current = self.controller.get_run(run_id)
            if current is None:
                return False, None
            if getattr(current, "status", None) != RunStatus.queued:
                return False, current
            setter = getattr(self.controller, "_set_run_status", None)
            if callable(setter):
                setter(run_id, RunStatus.running)
            else:
                self.controller.store.update_run_status(run_id, RunStatus.running.value)
            self.controller.append_event(run_id, "run_started", {"runId": run_id}, "system", "operator")
            return True, self.controller.get_run(run_id)

    def _execute_run(self, run_id: str, env_id: str, provider: ToolProvider, driver: Any, evaluate: Any) -> Any:
        """Invoke the canonical lifecycle seam and bridge legacy evaluator args."""
        execute = self.controller.execute_run
        parameters = inspect.signature(execute).parameters
        kwargs: dict[str, Any] = {}
        if "evaluate" in parameters:
            kwargs["evaluate"] = evaluate
        if "claimed" in parameters:
            kwargs["claimed"] = True
        result = execute(run_id, env_id, provider, driver, **kwargs)
        if "evaluate" in parameters:
            # Controller.execute_run deliberately absorbs driver exceptions and
            # leaves the run terminally failed.  Reconcile that terminal
            # attempt through the evaluator seam when any model evidence was
            # recorded, so measured objective failure remains scoreable and
            # failed model usage stays attached to the last response.
            current = self.controller.get_run(run_id)
            has_model_evidence = any(row.get("event_type") == "model_response" for row in self.controller.store.list_evidence(run_id))
            if current is not None and current.status == RunStatus.failed and has_model_evidence:
                # Reconcile only terminal failures with a durable response.
                # Strict evidence errors must surface to the caller.
                self._record_evaluated_outcome(run_id, evaluate())
            return result

        # The canonical core marks a completed driver run succeeded.  Apply
        # the evaluator-owned outcome immediately afterward and reconcile the
        # terminal status through the controller's own persistence method.
        current = self.controller.get_run(run_id)
        if current is None or current.status != RunStatus.succeeded:
            return result
        try:
            outcome = evaluate()
            self._record_evaluated_outcome(run_id, outcome)
            status = RunStatus.succeeded if outcome.passed else RunStatus.failed
        except Exception as exc:
            self.controller.append_event(run_id, "run_failed", {"error": str(exc)}, "system", "operator")
            status = RunStatus.failed
        setter = getattr(self.controller, "_set_run_status", None)
        if callable(setter):
            setter(run_id, status)
        else:
            completed_at = datetime.now(timezone.utc).isoformat()
            self.controller.store.update_run_status(run_id, status.value, completed_at=completed_at)
        return self.controller.get_run(run_id)

    def _record_evaluated_outcome(self, run_id: str, outcome: DurableOutcome) -> Any:
        """Record evaluator output, preserving the canonical trusted-outcome seam."""
        trusted = getattr(self.controller, "record_trusted_outcome", None)
        if callable(trusted):
            stored = self.controller.store.get_run(run_id)
            if stored is not None:
                metadata = outcome.metadata if isinstance(outcome.metadata, Mapping) else {}
                fixture_reset_ok = metadata.get("fixtureResetOk", metadata.get("fixtureResetPassed", True))
                trusted_metadata = {
                    "runId": run_id,
                    "taskId": stored["task_id"],
                    "environmentId": stored["environment_id"],
                    "passed": bool(outcome.passed),
                    "reliable": bool(metadata.get("reliable", outcome.passed)),
                    "safetyViolations": int(metadata.get("safetyViolations", 0) or 0),
                    "fixtureResetOk": bool(fixture_reset_ok),
                    **({"arm": metadata["arm"]} if isinstance(metadata.get("arm"), str) else {}),
                    **({"seed": metadata["seed"]} if isinstance(metadata.get("seed"), int) and not isinstance(metadata.get("seed"), bool) else {}),
                    **({"bundleHash": metadata["bundleHash"]} if isinstance(metadata.get("bundleHash"), str) else {}),
                    **({"goal": metadata["goal"]} if isinstance(metadata.get("goal"), str) else {}),
                    **({"aggregateEvaluation": metadata["aggregateEvaluation"]} if isinstance(metadata.get("aggregateEvaluation"), Mapping) else {}),
                }
                trusted_result = trusted(run_id, trusted_metadata)
                # Keep evaluator diagnostics such as planner status and
                # timeout classification in the private outcome row.  The
                # trusted evidence event remains sanitized by the controller.
                # Canonical metrics win over private diagnostics, while any
                # evaluator-specific fields remain durably available.
                merged_metadata = {**dict(metadata), **trusted_metadata, "runId": run_id}
                self.controller.record_outcome(
                    run_id,
                    bool(outcome.passed),
                    score=outcome.score,
                    metadata=merged_metadata,
                )
                return trusted_result
        return self.controller.record_outcome(run_id, outcome.passed, score=outcome.score, metadata=outcome.metadata)

    def _prepare_launch_identity(self, run_id: str, stored: Mapping[str, Any], run_record: Any, arm: str, seed: int, bundle_hash: str | None) -> tuple[str, int, str]:
        """Bind ordinary launches to the persisted immutable run bundle."""
        try:
            artifact = self.controller.store.get_artifact(run_record.skill_bundle_ref)
            from adaptive_agent.models import SkillBundle
            pinned_bundle = SkillBundle.model_validate(artifact)
        except (KeyError, TypeError, ValueError) as exc:
            raise LearningRuntimeError("run skill bundle is missing or malformed") from exc
        pinned_hash = pinned_bundle.content_hash
        if stored.get("bundle_hash") != pinned_hash:
            raise LearningRuntimeError("run skill bundle hash does not match its persisted content")
        if bundle_hash is not None and bundle_hash != pinned_hash:
            raise LearningRuntimeError("launch bundle hash does not match the persisted run bundle")
        try:
            payload = json.loads(stored.get("run_json", "{}"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LearningRuntimeError("run identity is malformed") from exc
        if not isinstance(payload, dict):
            raise LearningRuntimeError("run identity is malformed")
        if arm == "B0" and isinstance(payload.get("arm"), str):
            arm = payload["arm"]
        if seed == 0 and isinstance(payload.get("seed"), int) and not isinstance(payload.get("seed"), bool):
            seed = payload["seed"]
        resolved_hash = bundle_hash or pinned_hash
        payload.update({"arm": arm, "seed": seed, "bundleHash": resolved_hash})
        row = dict(stored)
        row["run_json"] = json.dumps(payload, sort_keys=True)
        self.controller.store.save_run(run_id, {key: value for key, value in row.items() if key != "run_id"})
        return arm, seed, resolved_hash

    def _bind_terminal_accounting(self, run_id: str, execution_started: float) -> None:
        """Persist one terminal receipt for ordinary runtime launches."""
        stored = self.controller.store.get_run(run_id)
        if stored is None:
            return
        try:
            run_payload = json.loads(stored.get("run_json", "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if isinstance(run_payload, dict) and isinstance(run_payload.get("finalAccountingRef"), str):
            return
        rows = self.controller.store.list_evidence(run_id)
        models = [row for row in rows if row.get("event_type") == "model_response"]
        if not models:
            return
        try:
            model = self.controller.store.get_artifact(json.loads(models[-1]["source_ref"])["sha256"])
            source = model.get("accountingRef") if isinstance(model, Mapping) else None
            source_ref = source.get("sha256") if isinstance(source, Mapping) else None
            accounting = self.controller.store.get_artifact(source_ref) if isinstance(source_ref, str) else None
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(accounting, Mapping):
            return
        final = dict(accounting)
        final["toolCalls"] = sum(row.get("event_type") == "tool_result" for row in rows)
        final["durationSeconds"] = max(float(accounting.get("durationSeconds", 0) or 0), time.monotonic() - execution_started)
        final["inferenceDurationSeconds"] = float(accounting.get("inferenceDurationSeconds", 0) or 0)
        ref = self.controller.store.put_artifact(final).sha256
        run_payload["finalAccountingRef"] = ref
        row = dict(stored)
        row["run_json"] = json.dumps(run_payload, sort_keys=True)
        self.controller.store.save_run(run_id, {key: value for key, value in row.items() if key != "run_id"})

    def _durable_evaluator_evidence(self, run_id: str) -> list[dict[str, Any]]:
        """Return immutable model/kernel event payloads to the trusted evaluator."""
        events: list[dict[str, Any]] = []
        for row in self.controller.store.list_evidence(run_id):
            event_type = row.get("event_type")
            if event_type not in {"model_response", "kernel"}:
                continue
            try:
                source_ref = json.loads(row["source_ref"])
                payload = self.controller.store.get_artifact(source_ref["sha256"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(payload, Mapping):
                events.append({
                    "evidenceId": row.get("evidence_id"),
                    "sequence": row.get("sequence"),
                    "eventType": event_type,
                    "payload": dict(payload),
                })
        return events

    def _invoke_evaluator(self, *, run_id: str, goal: str, model_output: str, environment: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Call the trusted evaluator with durable execution evidence when supported."""
        if self.evaluator is None:
            return None
        durable_evidence = self._durable_evaluator_evidence(run_id)
        kwargs: dict[str, Any] = {
            "goal": goal,
            "model_output": model_output,
            "environment": dict(environment),
            "run_id": run_id,
            "evidence": durable_evidence,
            "model_responses": [item for item in durable_evidence if item.get("eventType") == "model_response"],
            "kernel_events": [item for item in durable_evidence if item.get("eventType") == "kernel"],
        }
        try:
            parameters = inspect.signature(self.evaluator).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        forwarded = kwargs if accepts_kwargs or not parameters else {key: value for key, value in kwargs.items() if key in parameters}
        result = self.evaluator(**forwarded)
        return dict(result) if isinstance(result, Mapping) else None

    def _dispatch_tool(
        self,
        env_id: str,
        request: ToolRequest,
        capability: Capability,
        provider: ToolProvider,
        *,
        budget_remaining: dict[str, Any],
        dry_run: bool,
    ) -> Any:
        """Call Controller.dispatch_tool across the canonical/extended seams."""
        dispatch = self.controller.dispatch_tool
        parameters = inspect.signature(dispatch).parameters
        if "budget_remaining" not in parameters and budget_remaining.get("tool_calls", 0) <= 0:
            raise RuntimeError("tool call budget exhausted")
        kwargs: dict[str, Any] = {}
        if "budget_remaining" in parameters:
            kwargs["budget_remaining"] = budget_remaining
        if "dry_run" in parameters:
            kwargs["dry_run"] = dry_run
        return dispatch(env_id, request, capability, provider, **kwargs)

    def _record_broker_result(self, run_id: str, payload: dict[str, Any], *, development: bool) -> Any:
        """Persist broker fidelity using the available controller authority."""
        record = getattr(self.controller, "record_broker_tool_result", None)
        if callable(record):
            parameters = inspect.signature(record).parameters
            if "development" in parameters:
                return record(run_id, payload, development=development)
            return record(run_id, payload)
        append = getattr(self.controller, "append_broker_result", None)
        if callable(append):
            return append(run_id, payload, development=development)
        # The canonical core has no broker-result convenience method.  Keep
        # the operator event durable without inventing a second projection
        # authority in the application adapter.
        return self.controller.append_event(run_id, "tool_result", payload, "broker", "operator")

    def launch(self, run_id: str, *, task_override: Any | None = None, package_override: Any | None = None, model_client_override: Any | None = None, seed: int = 0, arm: str = "B0", bundle_hash: str | None = None, core_planner_hash: str | None = None, image_digest: str | None = None) -> None:
        stored = self.controller.store.get_run(run_id)
        if not stored:
            raise KeyError("run not found")
        task = task_override or self.registry.get_task(stored["task_id"])
        package = package_override or self.packages.get(stored["environment_id"])
        if not task or not package:
            raise KeyError("development task not found")
        # Reject an unusable paid-run budget before claiming or dispatching any
        # model request.  Clamping zero to one would silently spend against an
        # explicitly zero allocation.
        run_record = self.controller.get_run(run_id)
        try:
            budget_data = self.controller.store.get_artifact(run_record.budget_ref) if run_record is not None else {}
        except KeyError:
            budget_data = {}
        if isinstance(budget_data, Mapping):
            max_tokens = int(budget_data.get("modelTokens", DEFAULT_MODEL_TOKENS))
            wall_seconds = float(budget_data.get("wallTimeSeconds", 90))
            max_cost = int(budget_data.get("costMicrounits", 100000))
            if max_tokens <= 0 or max_cost <= 0 or wall_seconds <= 0:
                raise RuntimeError("model, cost, and wall-time budgets must be positive before dispatch")
        arm, seed, bundle_hash = self._prepare_launch_identity(run_id, stored, run_record, arm, seed, bundle_hash)
        execution_started = time.monotonic()
        claimed, current = self._claim_run(run_id)
        if current is None:
            raise KeyError("run not found")
        if not claimed:
            return
        self._run_started_at[run_id] = time.monotonic()
        self._run_last_receipt_at[run_id] = self._run_started_at[run_id]
        cancel = self._cancel_events.setdefault(run_id, threading.Event())
        provider: ToolProvider
        provider_factory = getattr(package, "provider_factory", None)
        def make_provider() -> ToolProvider:
            if callable(provider_factory):
                return provider_factory(task, run_id, seed)
            return _FixtureProvider(package, task, run_id, seed=seed)
        if self.model_runner is not None and model_client_override is None:
            provider = make_provider()
            invocation: Any = None
            model_runner = self.model_runner
            runtime = self
            class DirectDriver:
                def act(self, _ctx: Any) -> None:
                    nonlocal invocation
                    invocation = model_runner(goal=task.goal, environment=runtime._planner_environment(package, run_id), emit=lambda kind, summary, detail=None: runtime.controller.append_event(run_id, kind, {"summary": summary, "detail": detail}, "system", "operator"))
                    runtime._record_model_response(run_id, package, {"provider": invocation.provider, "model": invocation.model, "responseId": invocation.response_id, "usage": dict(invocation.usage), "arm": arm, "seed": seed, "bundleHash": bundle_hash, "corePlannerHash": core_planner_hash, "imageDigest": image_digest, **({"nominalCostUsd": invocation.nominalCostUsd} if hasattr(invocation, "nominalCostUsd") else {}), **({"costMicrounits": invocation.costMicrounits} if hasattr(invocation, "costMicrounits") else ({"costMicrounits": invocation.cost_microunits} if hasattr(invocation, "cost_microunits") else {})), **({"economicCostStatus": invocation.economicCostStatus} if hasattr(invocation, "economicCostStatus") else {})})
            def evaluate() -> DurableOutcome:
                outcome = runtime._invoke_evaluator(run_id=run_id, goal=task.goal, model_output=invocation.text, environment=runtime._planner_environment(package, run_id)) if invocation is not None and runtime.evaluator is not None else None
                if outcome is None:
                    external_evaluate = getattr(package, "evaluate_provider", None)
                    if callable(external_evaluate):
                        outcome = dict(external_evaluate(provider))
                outcome = outcome or {"passed": False}
                outcome = {**outcome, "arm": arm, "seed": seed, "bundleHash": bundle_hash, "goal": task.goal}
                return DurableOutcome(runId=run_id, passed=bool(outcome.get("passed") is True), metadata=outcome)
            try:
                self._execute_run(run_id, package.environment_id, provider, DirectDriver(), evaluate)
            finally:
                close = getattr(provider, "close", None)
                if callable(close):
                    close()
                self._bind_terminal_accounting(run_id, execution_started)
            self._cancel_events.pop(run_id, None)
            self._run_started_at.pop(run_id, None)
            self._run_last_receipt_at.pop(run_id, None)
            return
        prime: PrimeRuntimeAdapter | None = None
        provider = make_provider()
        run_record = self.controller.get_run(run_id)
        try:
            budget_data = self.controller.store.get_artifact(run_record.budget_ref) if run_record is not None else {}
        except KeyError:
            # The built-in profile is a trusted reference seeded by the control
            # plane; custom UI budgets are always persisted as artifacts.
            budget_data = {}
        budget_remaining = {"tool_calls": int(budget_data.get("toolCalls", 32)) if isinstance(budget_data, Mapping) else 32}
        read_invocation = 0
        def authorize(capability: PrimeCapability, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            nonlocal read_invocation
            canonical = json.dumps(dict(arguments), sort_keys=True, separators=(",", ":"))
            argument_hash = hashlib.sha256(canonical.encode()).hexdigest()
            if capability.effect == "read":
                read_invocation += 1
                invocation_hash = hashlib.sha256(f"{argument_hash}:{read_invocation}".encode()).hexdigest()
                idempotency_key = f"prime:{capability.id}:read:{read_invocation}:{argument_hash}"
            else:
                invocation_hash = argument_hash
                idempotency_key = f"prime:{capability.id}:{argument_hash}"
            request = ToolRequest(runId=run_id, stepId=f"planner-{invocation_hash[:16]}", tool=capability.tool, arguments=dict(arguments), idempotencyKey=idempotency_key)
            expires = datetime.fromisoformat(capability.expires_at.replace("Z", "+00:00"))
            durable_capability = Capability(run_id, package.environment_id, capability.tool, capability.effect, {}, expires)
            run = self.controller.get_run(run_id)
            schema = self.registry.get_tool_schema(package.environment_id, capability.tool)
            if run is not None and run.execution_mode == "batch" and schema is not None and schema.effect == "write":
                request.approval_token = self.controller.broker.issue_approval(package.environment_id, run_id, capability.tool, dict(arguments), request.idempotency_key)
            result = self._dispatch_tool(package.environment_id, request, durable_capability, provider, budget_remaining=budget_remaining, dry_run=run is not None and run.execution_mode == "dry_run")
            partition = self.controller.store.get_task(stored["task_id"]).get("partition") if stored and self.controller.store.get_task(stored["task_id"]) else None
            self._record_broker_result(
                run_id,
                result.model_dump(mode="json", by_alias=True),
                development=partition == "development",
            )
            # Every broker attempt consumes the shared run call budget,
            # including failed/retried calls whose receipts remain auditable.
            budget_remaining["tool_calls"] = max(0, budget_remaining["tool_calls"] - 1)
            return result.model_dump(mode="json", by_alias=True)
        try:
            max_tokens = int(budget_data.get("modelTokens", DEFAULT_MODEL_TOKENS)) if isinstance(budget_data, Mapping) else DEFAULT_MODEL_TOKENS
            wall_seconds = float(budget_data.get("wallTimeSeconds", 90)) if isinstance(budget_data, Mapping) else 90.0
            child_runs = int(budget_data.get("childRuns", 0)) if isinstance(budget_data, Mapping) else 0
            pinned_image = image_digest if isinstance(image_digest, str) and image_digest not in {"", "image-unpinned"} else None
            max_cost = int(budget_data.get("costMicrounits", 100000)) if isinstance(budget_data, Mapping) else 100000
            prime = PrimeRuntimeAdapter(PrimeRuntimeConfig(task_id=run_id, model="openai-codex/gpt-5.6-luna", provider="openai-codex", max_model_tokens=max_tokens, max_model_cost_microunits=max_cost, max_total_wall_seconds=wall_seconds, child_runs=child_runs, max_child_depth=1, docker_image=pinned_image, ao_session_id=os.environ.get("AO_SESSION_ID")), broker=CapabilityBroker(run_id, authorizer=authorize))
            for tool in package.manifest.tool_schemas:
                expires = (datetime.now(timezone.utc) + timedelta(seconds=90)).isoformat()
                prime.broker.register(PrimeCapability(f"{run_id}:{tool.name}", tool.name, tool.version, tool.effect, "run", expires))
            coding_dir = os.environ.get("PRIME_AGENT_CODING_AGENT_DIR")
            if not coding_dir and model_client_override is None:
                raise RuntimeError("PRIME_AGENT_CODING_AGENT_DIR is required for the Prime CLI model client")
            client = model_client_override or PrimeCliModelClient(coding_agent_dir=coding_dir)
        except Exception as exc:
            self.controller.append_event(run_id, "run_failed", {"error": str(exc)}, "system", "operator")
            if (current := self.controller.get_run(run_id)) is not None and current.status != RunStatus.cancelled:
                self.controller._set_run_status(run_id, RunStatus.failed)
            self._cancel_events.pop(run_id, None)
            self._run_started_at.pop(run_id, None)
            self._run_last_receipt_at.pop(run_id, None)
            self._bind_terminal_accounting(run_id, execution_started)
            if prime is not None:
                prime.close(remove_workspace=True)
            return
        runtime = self

        # Parent and child model calls share the adapter's trusted ledger. The
        # child planner is attached only after the authenticated client exists,
        # and its receipts are routed through the same parent-owned evidence
        # path as the main planner.
        def persist_child(evidence: Mapping[str, Any]) -> Any:
            runtime._record_model_response(run_id, package, {**dict(evidence), "arm": arm, "seed": seed, "bundleHash": bundle_hash, "corePlannerHash": core_planner_hash, "imageDigest": image_digest})
            return prime.record_model_observation(evidence, trusted_parent=True)

        def persist_parent(evidence: Mapping[str, Any]) -> Any:
            runtime._record_model_response(run_id, package, {**dict(evidence), "arm": arm, "seed": seed, "bundleHash": bundle_hash, "corePlannerHash": core_planner_hash, "imageDigest": image_digest})
            return prime.record_model_observation(evidence, trusted_parent=True)

        child_planner = LunaChildPlanner(client, budget=prime.planner_budget, observation_sink=persist_child)
        prime.child_planner = child_planner

        parent_client = child_planner.parent_model_client(observation_sink=persist_parent)

        class Sink:
            def __init__(self, controller: Controller) -> None:
                self._controller = controller

            def record_model_observation(self, evidence: Mapping[str, Any], *, trusted_parent: bool = False) -> Any:
                return None
        class Driver:
            def __init__(self, controller: Controller) -> None:
                self._controller = controller
                self.result: PlannerResult | None = None
            def act(self, ctx: Any) -> None:
                max_tokens = int(budget_data.get("modelTokens", DEFAULT_MODEL_TOKENS)) if isinstance(budget_data, Mapping) else DEFAULT_MODEL_TOKENS
                wall_seconds = float(budget_data.get("wallTimeSeconds", 90)) if isinstance(budget_data, Mapping) else 90.0
                if max_tokens <= 0:
                    self.result = PlannerResult("budget_exhausted", None, 0, 0, 0, (), ())
                    return
                if wall_seconds <= 0:
                    self.result = PlannerResult("timed_out", None, 0, 0, 0, (), ())
                    return
                planner = LunaPlanner(parent_client, prime, Sink(self._controller), limits=PlannerLimits(max_model_tokens=max_tokens, max_wall_seconds=wall_seconds), emit=lambda event: self._controller.append_event(run_id, event.kind, {"summary": event.summary, "detail": event.detail}, "system", "operator"))
                self.result = planner.run(goal=task.goal, environment=runtime._planner_environment(package, run_id), active_skills=runtime._active_skills(run_id), cancel=cancel)
        driver = Driver(self.controller)
        def evaluate() -> DurableOutcome:
            result = driver.result
            if result is None or result.status != "succeeded":
                return DurableOutcome(runId=run_id, passed=False, metadata={"status": result.status if result else "planner_failed", "arm": arm, "seed": seed, "bundleHash": bundle_hash, "goal": task.goal})
            if runtime.evaluator is not None:
                evaluated = runtime._invoke_evaluator(
                    run_id=run_id,
                    goal=task.goal,
                    model_output=result.answer or "",
                    environment=runtime._planner_environment(package, run_id),
                ) or {"passed": False, "diagnostic": "trusted evaluator returned a non-object"}
                passed = evaluated.get("passed") is True
                try:
                    score = float(evaluated.get("score", 1.0 if passed else 0.0))
                except (TypeError, ValueError):
                    score = 1.0 if passed else 0.0
                return DurableOutcome(runId=run_id, passed=passed, score=score, metadata={**evaluated, "arm": arm, "seed": seed, "bundleHash": bundle_hash, "goal": task.goal})
            external_evaluate = getattr(package, "evaluate_provider", None)
            if callable(external_evaluate):
                evaluated = dict(external_evaluate(provider))
                return DurableOutcome(runId=run_id, passed=bool(evaluated.get("passed")), score=float(evaluated.get("score", 1.0 if evaluated.get("passed") else 0.0)), metadata={**evaluated, "plannerStatus": result.status, "arm": arm, "seed": seed, "bundleHash": bundle_hash, "goal": task.goal})
            fixture = package.evaluate(task.task_id, provider.session)
            return DurableOutcome(runId=run_id, passed=fixture.passed, score=1.0 if fixture.passed else 0.0, metadata={"reason": fixture.reason, "evaluatorVersion": fixture.evaluator_version, "plannerStatus": result.status, "arm": arm, "seed": seed, "bundleHash": bundle_hash, "goal": task.goal})
        try:
            self._execute_run(run_id, package.environment_id, provider, driver, evaluate)
        finally:
            close = getattr(provider, "close", None)
            if callable(close):
                close()
            if prime is not None:
                prime.close(remove_workspace=True)
            self._bind_terminal_accounting(run_id, execution_started)
            self._cancel_events.pop(run_id, None)
            self._run_started_at.pop(run_id, None)
            self._run_last_receipt_at.pop(run_id, None)

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        return self.controller.events(run_id, after)

    def cancel(self, run_id: str) -> dict[str, Any]:
        event = self._cancel_events.get(run_id)
        if event is not None:
            event.set()
        run = self.controller.cancel_run(run_id)
        if run is None:
            raise KeyError("run not found")
        return run.model_dump(mode="json", by_alias=True)

    def submit_approval(self, run_id: str, approval_id: str, approved: bool) -> dict[str, Any]:
        if self.get_run(run_id) is None:
            raise KeyError("run not found")
        self._approvals[(run_id, approval_id)] = approved
        self.controller.append_event(run_id, "approval", {"approvalId": approval_id, "approved": approved}, "operator", "operator")
        return {"runId": run_id, "approvalId": approval_id, "approved": approved}


def _seed_durable_stack(plane: ControlPlane, store_dir: Path, appworld_config: Any | None = None) -> tuple[Controller, EnvironmentRegistry, dict[str, Any]]:
    """Create the durable seam and register the public fixture projections."""
    store = Store(store_dir)
    registry = EnvironmentRegistry(store)
    plane.trust_reference("model", plane.default_model_ref)
    plane.trust_reference("budget", plane.default_budget_ref)
    packages = build_environment_packages()
    for package in packages.values():
        manifest = package.manifest
        public_tasks = package.learner_tasks()
        # Persist learner-facing document contents under the exact manifest
        # hashes.  This applies even to sealed packages whose development task
        # list is intentionally empty: public documentation remains a valid
        # learner projection while evaluator-owned tasks stay unregistered.
        for document in package.learner_documents():
            document_ref = store.put_artifact({"id": document.document_id, "version": document.version, "text": document.text})
            expected_ref = next((ref for ref in manifest.docs if ref.id == document.document_id and ref.version == document.version), None)
            if expected_ref is None or document_ref.sha256 != expected_ref.sha256:
                raise RuntimeError(f"public document hash mismatch for {manifest.environment_id}:{document.document_id}")
        # Validation/final and sealed fixtures stay evaluator-owned. They are
        # allocated only by the frozen evaluation protocol, never registered as
        # operator training tasks.
        if not public_tasks:
            registry.register(DurableManifest(
                schemaVersion=1,
                environmentId=manifest.environment_id,
                version=manifest.version,
                docs=[ArtifactRef.model_validate(ref.to_dict()) for ref in manifest.docs],
                toolSchemas=[DurableTool.model_validate(tool.to_dict()) for tool in manifest.tool_schemas],
                policyRef=ArtifactRef.model_validate(manifest.policy_ref.to_dict()),
                evaluatorRef=ArtifactRef.model_validate(manifest.evaluator_ref.to_dict()),
                resetRef=ArtifactRef.model_validate(manifest.reset_ref.to_dict()),
                executionModes=list(manifest.execution_modes),
                capabilities=list(manifest.capabilities),
            ))
            continue
        for kind, ref in (("policy", manifest.policy_ref), ("evaluator", manifest.evaluator_ref), ("reset", manifest.reset_ref)):
            plane.trust_reference(kind, ref.to_dict())
        for ref in manifest.docs:
            plane.trust_reference("docs", ref.to_dict())
        api_manifest: dict[str, Any] = {
            "environmentId": manifest.environment_id,
            "version": manifest.version,
            "docs": [ref.to_dict() for ref in manifest.docs],
            "taskGoals": [public_tasks[0].goal],
            "toolSchemas": [tool.to_dict() for tool in manifest.tool_schemas],
            "policyRef": manifest.policy_ref.to_dict(),
            "evaluatorRef": manifest.evaluator_ref.to_dict(),
            "resetRef": manifest.reset_ref.to_dict(),
            "executionModes": list(manifest.execution_modes),
            "capabilities": list(manifest.capabilities),
        }
        from adaptive_agent.api import EnvironmentRegistration

        plane.register_environment(EnvironmentRegistration.model_validate(api_manifest))
        durable_manifest = DurableManifest(
            schemaVersion=1,
            environmentId=manifest.environment_id,
            version=manifest.version,
            docs=[ArtifactRef.model_validate(ref.to_dict()) for ref in manifest.docs],
            toolSchemas=[DurableTool.model_validate(tool.to_dict()) for tool in manifest.tool_schemas],
            policyRef=ArtifactRef.model_validate(manifest.policy_ref.to_dict()),
            evaluatorRef=ArtifactRef.model_validate(manifest.evaluator_ref.to_dict()),
            resetRef=ArtifactRef.model_validate(manifest.reset_ref.to_dict()),
            executionModes=list(manifest.execution_modes),
            capabilities=list(manifest.capabilities),
        )
        registry.register(durable_manifest)
        for task in public_tasks:
            registry.register_task(DurableTask(
                taskId=task.task_id,
                environmentRef=ArtifactRef.model_validate(task.environment_ref.to_dict()),
                goal=task.goal,
                allowedInputRefs=[ArtifactRef(id=ref, version="1", sha256=hashlib.sha256(ref.encode("utf-8")).hexdigest()) for ref in task.allowed_input_refs],
                partition=task.partition.value,
            ))
    if appworld_config is not None:
        from adaptive_agent.appworld_provider import AppWorldPackage, register_appworld

        appworld_package = AppWorldPackage(appworld_config)
        register_appworld(registry, appworld_config, splits=("train",), runtime_partition="development")
        register_appworld(registry, appworld_config, splits=("dev",), runtime_partition="validation")
        for kind, ref in (("policy", appworld_package.manifest.policy_ref), ("evaluator", appworld_package.manifest.evaluator_ref), ("reset", appworld_package.manifest.reset_ref)):
            plane.trust_reference(kind, ref.model_dump(mode="json", by_alias=True))
        for ref in appworld_package.manifest.docs:
            plane.trust_reference("docs", ref.model_dump(mode="json", by_alias=True))
        packages[appworld_package.environment_id] = appworld_package
    controller = Controller(store, registry, ToolBroker(store, registry))
    # Construct evaluator-owned durable adapters on the same SQLite store so
    # trusted attestations, allocations, and evidence verification survive a
    # process restart.  The executable run path still evaluates only through
    # the registered fixture package.
    controller.evaluator_adapters = build_durable_adapters(store)
    if controller.get_active_bundle() is None:
        from adaptive_agent.models import SkillBundle

        controller.candidates.initialize_active_bundle(SkillBundle())
    return controller, registry, packages


def create_runtime_app(
    *,
    model_runner: ModelRunner | None = None,
    evaluator: OutcomeEvaluator | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    console_dist: str | os.PathLike[str] | None = None,
    package_bindings: Mapping[str, Any] | None = None,
    learning_model_client: Any | None = None,
    evaluation_executor: Any | None = None,
    experiment_stage_runner: Any | None = None,
    evaluation_protocol: Any | None = None,
    evaluation_arm_bundles: Mapping[Any, Any] | None = None,
    appworld_root: str | os.PathLike[str] | None = None,
    appworld_python: str | None = None,
):
    """Build the API, durable controller seam, and optional static console."""
    kwargs: dict[str, Any] = {}
    if model_runner is not None:
        kwargs["model_runner"] = model_runner
    if evaluator is not None:
        kwargs["evaluator"] = evaluator
    model_profile = {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "tier": "medium", "maxTokens": DEFAULT_MODEL_TOKENS}
    budget_profile = {"modelTokens": DEFAULT_MODEL_TOKENS, "toolCalls": 32, "childRuns": 0, "wallTimeSeconds": 90, "costMicrounits": 100000, "currency": "USD"}
    model_ref = _ref("model-profile", "1", model_profile)
    budget_ref = _ref("budget-default", "1", budget_profile)
    kwargs["seed_test_references"] = False
    kwargs["default_model_ref"] = model_ref
    kwargs["default_budget_ref"] = budget_ref
    plane = ControlPlane(**kwargs)
    appworld_config = None
    if appworld_root is not None:
        from adaptive_agent.appworld_provider import AppWorldConfig

        appworld_config = AppWorldConfig(Path(appworld_root), python=appworld_python or "")
    controller, registry, packages = _seed_durable_stack(plane, Path(data_dir or os.environ.get("ADAPTIVE_AGENT_DATA", ".adaptive-agent")), appworld_config)
    # Materialize the built-in budget profile under its trusted content hash so
    # every RunRecord budgetRef resolves to immutable bytes, just like custom
    # UI budgets.
    controller.store.put_artifact(budget_profile)
    durable_runtime = DurableRuntime(controller, registry, packages, model_runner=model_runner, evaluator=evaluator, model_ref=model_ref, budget_ref=budget_ref, control_plane=plane, package_bindings=package_bindings, learning_model_client=learning_model_client, evaluation_executor=evaluation_executor, experiment_stage_runner=experiment_stage_runner)
    app = create_app(plane, durable_runtime=durable_runtime)
    app.state.controller = controller
    if evaluation_protocol is None:
        from adaptive_agent.evaluation import EvaluationProtocol

        evaluation_protocol = EvaluationProtocol(
            image_digest=durable_runtime.image_digest,
            core_planner_hash=durable_runtime.core_planner_hash,
        )
        evaluation_protocol.freeze(durable_runtime.packages)
    app.state.evaluation_driver = durable_runtime.build_evaluation_driver(evaluation_protocol, evaluation_arm_bundles)
    resolved_console = Path(console_dist) if console_dist is not None else Path(__file__).resolve().parents[2] / "console" / "dist"
    if resolved_console.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=os.fspath(resolved_console), html=True), name="console")
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Adaptive Agent API and operator console backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default=os.environ.get("ADAPTIVE_AGENT_DATA", ".adaptive-agent"))
    parser.add_argument("--console-dist", default=os.environ.get("ADAPTIVE_AGENT_CONSOLE_DIST"))
    parser.add_argument("--appworld-root", default=os.environ.get("APPWORLD_ROOT"))
    parser.add_argument("--appworld-python", default=os.environ.get("APPWORLD_PYTHON"))
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_runtime_app(data_dir=args.data_dir, console_dist=args.console_dist, appworld_root=args.appworld_root, appworld_python=args.appworld_python), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
