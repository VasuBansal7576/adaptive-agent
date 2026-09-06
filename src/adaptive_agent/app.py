"""Executable application factory for the integrated Adaptive Agent stack.

The factory wires the durable Store/EnvironmentRegistry/ToolBroker/Controller
seam into the operator API and seeds only trusted fixture references.  A live
model runner is injected by deployment, so the default process cannot silently
claim that an unavailable provider is a simulation or a successful run.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from adaptive_agent.api import ControlPlane, ModelRunner, OutcomeEvaluator, create_app, _ref, IdempotencyConflict
from adaptive_agent.constants import DEFAULT_MODEL_TOKENS
from adaptive_agent.broker import Capability, ToolBroker, ToolProvider
from adaptive_agent.controller import Controller
from adaptive_agent.environment import EnvironmentRegistry
from adaptive_agent.evaluation import build_environment_packages, sha256_json, FixtureSession, Outcome as FixtureOutcome
from adaptive_agent.learning import LearningService, PlannerLearningAdapter
from adaptive_agent.learning_store import DurableLearningSourceAdapter, CandidateManagerLearningAdapter, LearningStoreError
from adaptive_agent.learning_runtime import LearningRuntime, LearningRuntimeError
from adaptive_agent.planner import PrimeCliModelClient, LunaPlanner, PlannerResult, PlannerLimits
from adaptive_agent.prime_runtime import Capability as PrimeCapability, CapabilityBroker, PrimeRuntimeAdapter, PrimeRuntimeConfig
from adaptive_agent.models import ArtifactRef, EnvironmentManifest as DurableManifest, TaskInput as DurableTask, ToolSchema as DurableTool, RunStatus, ToolRequest, Outcome as DurableOutcome, canonical_usage
from adaptive_agent.store import Store
from adaptive_agent.evaluation_store import build_durable_adapters


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
    for name in ("planner.py", "prime_runtime.py"):
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

    def execute(self, run_id: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if run_id != self.session.task_id and run_id != self._run_id:
            raise RuntimeError("provider is bound to a different run")
        result = self.package.invoke(self.session, tool, arguments)
        return dict(result.output)

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
        self.package = package
        self.session = package.reset(task.task_id, seed)
        self._run_id = run_id


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

    def __init__(self, controller: Controller, registry: EnvironmentRegistry, packages: Mapping[str, Any], model_runner: Any | None = None, evaluator: Any | None = None, model_ref: Mapping[str, Any] | None = None, budget_ref: Mapping[str, Any] | None = None, control_plane: Any | None = None, package_bindings: Mapping[str, Any] | None = None, learning_model_client: Any | None = None, evaluation_executor: Any | None = None) -> None:
        self.controller, self.registry, self.packages = controller, registry, dict(packages)
        if package_bindings:
            self.packages.update(package_bindings)
        self.model_runner, self.evaluator = model_runner, evaluator
        self.model_ref, self.budget_ref = dict(model_ref or _ref("model-profile", "1")), dict(budget_ref or _ref("budget-default", "1"))
        self.control_plane = control_plane
        self.learning_model_client = learning_model_client
        self.evaluation_executor = evaluation_executor
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
        self._tasks = {
            task.task_id: task
            for name in self.packages
            for task in registry.list_tasks_by_partition(name, "development")
        }
        self._reload_registered_environments()

    def _reload_registered_environments(self) -> None:
        """Rebuild manifest/task projections from SQLite after a restart."""
        with self.controller.store._connect() as conn:
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
        with self.controller.store._connect() as conn:
            rows = conn.execute("SELECT candidate_json FROM candidates ORDER BY created_at").fetchall()
        for row in rows:
            try:
                value = json.loads(row["candidate_json"])
                if "candidate_id" in value:
                    value["candidateId"] = value.pop("candidate_id")
                out.append(value)
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
        candidate = self.controller.get_candidate(payload.candidate_id)
        if candidate is None:
            raise KeyError("candidate not found")
        if payload.base_bundle_hash != candidate.get("base_bundle_hash"):
            raise ValueError("evaluation base does not match candidate")
        self.controller.start_evaluation(payload.candidate_id)
        evaluation_id = f"eval_{__import__('uuid').uuid4().hex}"
        data = {"candidate_hash": candidate.get("candidate_bundle_hash") or payload.candidate_id, "base_hash": payload.base_bundle_hash, "protocol_hash": payload.protocol_hash, "partition_ref": json.dumps(payload.partition_ref, sort_keys=True), "report_json": json.dumps({"evaluationId": evaluation_id, "candidateId": payload.candidate_id, "baseBundleHash": payload.base_bundle_hash, "protocolHash": payload.protocol_hash, "partitionRef": payload.partition_ref, "state": "queued", "trusted": False}, sort_keys=True), "validity": "queued"}
        self.controller.store.save_evaluation(evaluation_id, data)
        return json.loads(data["report_json"])

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
        if arm is None or not isinstance(seed, int):
            raise LearningRuntimeError("evaluation execution config lacks arm or seed")
        arm_value = getattr(arm, "value", str(arm))
        if isinstance(bundle, SkillBundle):
            durable_bundle = bundle
        elif isinstance(bundle, Mapping):
            durable_bundle = SkillBundle.model_validate(bundle)
        else:
            durable_bundle = self.controller.get_active_bundle() or SkillBundle()
        expected_bundle_hash = getattr(frozen_config, "bundle_hash", None)
        if expected_bundle_hash and durable_bundle.content_hash != expected_bundle_hash:
            raise LearningRuntimeError("evaluation bundle hash does not match supplied bundle")

        version = str(getattr(getattr(task, "environment_ref", None), "version", "1"))
        environment_ref = ArtifactRef(id=env_id, version=version, sha256=sha256_json({"environmentId": env_id, "version": version}))
        durable_task = DurableTask(taskId=task_id, environmentRef=environment_ref, goal=goal, partition=getattr(getattr(task, "partition", None), "value", getattr(task, "partition", "development")))
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
        request = RunRequest(taskRef=task_ref, modelProfileRef=model_ref, budgetRef=budget_ref, idempotencyKey=f"benchmark:{protocol_hash}:{task_id}:{arm_value}:{seed}:{durable_bundle.content_hash}", executionMode="replay")
        run = self.controller.create_run(request, durable_task, skill_bundle=durable_bundle)
        row = self.controller.store.get_run(run.run_id)
        if row:
            persisted = {key: value for key, value in row.items() if key != "run_id"}
            run_payload = json.loads(row.get("run_json", "{}"))
            run_payload.update({"arm": arm_value, "seed": seed, "bundleHash": durable_bundle.content_hash})
            persisted["run_json"] = json.dumps(run_payload, sort_keys=True)
            self.controller.store.save_run(run.run_id, persisted)

        model_client = self.learning_model_client
        if model_client is None and self.model_runner is not None:
            runner = self.model_runner
            if callable(getattr(runner, "invoke", None)):
                # A provider client can be injected directly.  Keeping this
                # object intact preserves its own response/accounting seam.
                model_client = runner
            elif callable(runner):
                class RunnerClient:
                    def invoke(self, *, goal: str, environment: Mapping[str, Any], **_: Any) -> Mapping[str, Any]:
                        invocation = runner(goal=goal, environment=dict(environment), emit=lambda *_args: None)
                        return {"text": invocation.text, "provider": invocation.provider, "model": invocation.model, "responseId": invocation.response_id, "usage": dict(invocation.usage)}
                model_client = RunnerClient()
        # launch() owns claim, reset, Prime Docker, broker budget, retries, and
        # trusted outcome persistence for both API and benchmark executions.
        core_hash = str(inputs.get("corePlannerHash", self.core_planner_hash))
        image_digest = str(inputs.get("imageDigest", self.image_digest))
        self.launch(run.run_id, task_override=task, package_override=package, model_client_override=model_client, seed=seed, arm=arm_value, bundle_hash=durable_bundle.content_hash, core_planner_hash=core_hash, image_digest=image_digest)
        evidence_rows = self.controller.store.list_evidence(run.run_id)
        model_rows = [row for row in evidence_rows if row.get("event_type") == "model_response"]
        outcome_rows = [row for row in evidence_rows if row.get("event_type") == "trusted_outcome"]
        if not model_rows or not outcome_rows:
            raise LearningRuntimeError("evaluation run did not produce trusted model and outcome evidence")
        model_row = model_rows[-1]
        model_payload = self.controller.store.get_artifact(json.loads(model_row["source_ref"])["sha256"])
        accounting_ref = model_payload.get("accountingRef", {}).get("sha256") if isinstance(model_payload, Mapping) else None
        accounting = self.controller.store.get_artifact(accounting_ref) if isinstance(accounting_ref, str) else {}
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
        observation_kwargs = {"provenance": Provenance.DETERMINISTIC_SIMULATION, "model_provenance": ModelProvenance.REAL_MODEL, "model_profile": model_name, "core_planner_hash": core_hash, "budget": budget, "response_id": model_payload.get("responseId"), "accounting_ref": accounting_ref, "evidence_ref": model_row["evidence_id"], "outcome_ref": outcome_ref, "config_hashes": {"model": sha256_json({"profile": model_name, "provider": provider_name}), "planner": core_hash, "budget": sha256_json(budget_value), "policy": sha256_json(package.manifest.policy_ref), "schema": sha256_json(package.manifest.tool_schemas), "image": image_digest}, "run_id": run.run_id}
        # Session-6's evaluator model includes bundle_hash; keep this worker
        # compatible with the pre-merge evaluator while exposing it whenever
        # the authoritative type is present.
        try:
            from dataclasses import fields
            if any(field.name == "bundle_hash" for field in fields(RunObservation)):
                observation_kwargs["bundle_hash"] = durable_bundle.content_hash
        except TypeError:
            pass
        return RunObservation(task_id, env_id, Partition(getattr(getattr(task, "partition", None), "value", getattr(task, "partition", "development"))), seed, Arm(arm_value), passed, reliable, safety_violations, int(accounting.get("costMicrounits") or 0), float(accounting.get("durationSeconds", 1e-6)), **observation_kwargs)

    def run_evaluation_job(self, task: Mapping[str, Any], frozen_config: Mapping[str, Any] | None, bundle: Mapping[str, Any] | None) -> None:
        """Run one trusted evaluation task and durably record its result.

        The executor is injected by the evaluator owner.  Without that
        binding, the task is marked blocked with an honest provider/evaluator
        diagnostic rather than pretending that a queued row is an evaluation.
        """
        evaluation_id = task.get("evaluationId")
        if not isinstance(evaluation_id, str):
            return
        stored = self.controller.store.get_evaluation(evaluation_id)
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
            report = {**json.loads(stored["report_json"]), "state": "blocked", "trusted": False, "error": str(exc)}
            validity = "blocked"
        updated = dict(stored)
        updated.pop("report_id", None)
        self.controller.store.save_evaluation(
            evaluation_id,
            {
                **updated,
                "report_json": json.dumps(report, sort_keys=True),
                "validity": validity,
            },
        )

    def build_evaluation_driver(self, protocol: Any, arm_bundles: Mapping[Any, Any] | None = None) -> Any:
        """Wire the evaluator-owned resumable driver to this task executor."""
        from adaptive_agent.benchmark import ResumableEvaluationDriver

        active = self.controller.get_active_bundle()
        if active is None:
            raise LearningRuntimeError("no active bundle is available for evaluation")
        selected = dict(arm_bundles or {})
        selected.setdefault("B0", active)
        return ResumableEvaluationDriver(
            self.controller.store,
            protocol,
            self.packages,
            self.execute_evaluation_task,
            active,
            arm_bundles=selected,
        )

    def list_evaluations(self) -> list[dict[str, Any]]:
        out = []
        for row in self.controller.store.list_evaluations():
            try:
                out.append(json.loads(row["report_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
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
        stored = self.controller.store.get_run(payload.run_id)
        run = self.controller.get_run(payload.run_id)
        if stored is None or run is None:
            raise KeyError("run not found")
        package = self.packages.get(stored["environment_id"])
        task = self.registry.get_task(stored["task_id"])
        if package is None or task is None:
            raise KeyError("development task not found")
        if self._learning_runtime is None:
            model_client = self.learning_model_client if self.learning_model_client is not None else (self.model_runner if hasattr(self.model_runner, "invoke") else None)
            self._learning_runtime = LearningRuntime.build(store=self.controller.store, manager=self.controller.candidates, model_client=model_client)
        proposal = self._learning_runtime.propose_completed_run(payload.run_id, goal=task["goal"] if isinstance(task, Mapping) else None, feedback={"status": run.status.value})
        candidate = dict(proposal.authoritative_candidate)
        if "candidate_id" in candidate:
            candidate["candidateId"] = candidate.pop("candidate_id")
        return {"actionId": f"learn_{__import__('uuid').uuid4().hex}", "runId": payload.run_id, "predictedEffect": proposal.candidate_payload["predictedEffect"], "evidenceIds": proposal.candidate_payload["supportingEvidenceIds"], "proposalRef": self.controller.store.put_artifact(proposal.bundle_patch).model_dump(mode="json", by_alias=True), "candidate": candidate, "status": "staged", "createdAt": __import__("adaptive_agent.api", fromlist=["_now"])._now()}

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

    @staticmethod
    def _public_run(run: Any, task: Any) -> dict[str, Any]:
        value = run.model_dump(mode="json", by_alias=True)
        if value.get("outcomeRef") is None:
            value.pop("outcomeRef", None)
        value.update({"environmentId": task.environment_ref.id, "goal": task.goal})
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
            "docs": [ref.to_dict() for ref in package.manifest.docs],
            "publicDocs": [doc.to_dict() for doc in package.learner_documents()],
            "toolSchemas": [tool.to_dict() for tool in package.manifest.tool_schemas],
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
        with self.controller.store._connect() as conn:
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
        aggregate = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        aggregate_cost = 0.0
        explicit_cost = False
        aggregate_inference = 0.0
        now = time.monotonic()
        started = self._run_started_at.get(run_id, now)
        previous_receipt = self._run_last_receipt_at.get(run_id, started)
        for index, raw in enumerate(receipts_input):
            if not isinstance(raw, Mapping):
                raise ValueError("model accounting receipt must be an object")
            receipt_usage = canonical_usage(raw.get("usage"))
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
                aggregate_cost += float(cost_value)
                explicit_cost = True
            aggregate_inference += float(duration_value)
            for key in aggregate:
                aggregate[key] += receipt_usage[key]
            receipts.append({"responseId": receipt_id, "usage": receipt_usage, "durationSeconds": float(duration_value), "status": str(raw.get("status", "complete")), **({"costMicrounits": cost_value} if cost_value is not None else {})})
        self._run_last_receipt_at[run_id] = now
        self._run_receipts.setdefault(run_id, []).extend(receipts)
        all_receipts = list(self._run_receipts.get(run_id, ()))
        aggregate = {"inputTokens": sum(item["usage"]["inputTokens"] for item in all_receipts), "outputTokens": sum(item["usage"]["outputTokens"] for item in all_receipts), "totalTokens": sum(item["usage"]["totalTokens"] for item in all_receipts)}
        aggregate_inference = sum(float(item.get("durationSeconds", 0.0)) for item in all_receipts)
        aggregate_cost = sum(float(item.get("costMicrounits", 0.0)) for item in all_receipts if isinstance(item.get("costMicrounits"), (int, float)) and not isinstance(item.get("costMicrounits"), bool))
        explicit_cost = any("costMicrounits" in item for item in all_receipts)
        usage = canonical_usage(evidence.get("usage"))
        frozen_core_planner = evidence.get("corePlannerHash") if isinstance(evidence.get("corePlannerHash"), str) and evidence.get("corePlannerHash") else self.core_planner_hash
        frozen_image = evidence.get("imageDigest") if isinstance(evidence.get("imageDigest"), str) and evidence.get("imageDigest") else self.image_digest
        version_refs = {
            "policy": run.policy_ref.sha256,
            "schema": sha256_json(package.manifest.tool_schemas),
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
        arm = evidence.get("arm") or run_payload.get("arm") or "B0"
        seed = evidence.get("seed", run_payload.get("seed", 0))
        bundle_hash = evidence.get("bundleHash") or run_payload.get("bundleHash") or run.skill_bundle_ref.sha256
        whole_run_duration = max(now - started, 1e-6)
        economic_status = evidence.get("economicCostStatus")
        if not isinstance(economic_status, str):
            economic_status = "measured" if explicit_cost else ("subscription_marginal" if provider == "openai-codex" else "unavailable")
        cost: float | None = aggregate_cost if explicit_cost else (0.0 if economic_status == "subscription_marginal" else None)
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
            "costMicrounits": cost,
            "economicCost": {"status": economic_status, "microunits": cost},
            "durationSeconds": whole_run_duration,
            "inferenceDurationSeconds": aggregate_inference,
        }
        accounting_ref = self.controller.store.put_artifact(accounting)
        payload["accountingRef"] = accounting_ref.model_dump(mode="json", by_alias=True)
        return self.controller.append_event(run_id, "model_response", payload, "system", "operator")

    def launch(self, run_id: str, *, task_override: Any | None = None, package_override: Any | None = None, model_client_override: Any | None = None, seed: int = 0, arm: str = "B0", bundle_hash: str | None = None, core_planner_hash: str | None = None, image_digest: str | None = None) -> None:
        stored = self.controller.store.get_run(run_id)
        if not stored:
            raise KeyError("run not found")
        task = task_override or self.registry.get_task(stored["task_id"])
        package = package_override or self.packages.get(stored["environment_id"])
        if not task or not package:
            raise KeyError("development task not found")
        claimed, current = self.controller.claim_run(run_id)
        if current is None:
            raise KeyError("run not found")
        if not claimed:
            return
        self._run_started_at[run_id] = time.monotonic()
        self._run_last_receipt_at[run_id] = self._run_started_at[run_id]
        cancel = self._cancel_events.setdefault(run_id, threading.Event())
        provider: _FixtureProvider
        if self.model_runner is not None and model_client_override is None:
            provider = _FixtureProvider(package, task, run_id, seed=seed)
            invocation: Any = None
            model_runner = self.model_runner
            runtime = self
            class DirectDriver:
                def act(self, _ctx: Any) -> None:
                    nonlocal invocation
                    invocation = model_runner(goal=task.goal, environment=runtime._planner_environment(package, run_id), emit=lambda kind, summary, detail=None: runtime.controller.append_event(run_id, kind, {"summary": summary, "detail": detail}, "system", "operator"))
                    runtime._record_model_response(run_id, package, {"provider": invocation.provider, "model": invocation.model, "responseId": invocation.response_id, "usage": dict(invocation.usage), "arm": arm, "seed": seed, "bundleHash": bundle_hash, "corePlannerHash": core_planner_hash, "imageDigest": image_digest})
            def evaluate() -> DurableOutcome:
                outcome = dict(self.evaluator(goal=task.goal, model_output=invocation.text, environment=self._planner_environment(package, run_id))) if self.evaluator is not None and invocation is not None else {"passed": False}
                return DurableOutcome(runId=run_id, passed=bool(outcome.get("passed") is True), metadata=outcome)
            self.controller.execute_run(run_id, package.environment_id, provider, DirectDriver(), evaluate=evaluate, claimed=True)
            self._cancel_events.pop(run_id, None)
            self._run_started_at.pop(run_id, None)
            self._run_last_receipt_at.pop(run_id, None)
            return
        prime: PrimeRuntimeAdapter | None = None
        provider = _FixtureProvider(package, task, run_id, seed=seed)
        run_record = self.controller.get_run(run_id)
        try:
            budget_data = self.controller.store.get_artifact(run_record.budget_ref) if run_record is not None else {}
        except KeyError:
            # The built-in profile is a trusted reference seeded by the control
            # plane; custom UI budgets are always persisted as artifacts.
            budget_data = {}
        budget_remaining = {"tool_calls": int(budget_data.get("toolCalls", 32)) if isinstance(budget_data, Mapping) else 32}
        def authorize(capability: PrimeCapability, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
            canonical = json.dumps(dict(arguments), sort_keys=True, separators=(",", ":"))
            request = ToolRequest(runId=run_id, stepId=f"planner-{hashlib.sha256(canonical.encode()).hexdigest()[:16]}", tool=capability.tool, arguments=dict(arguments), idempotencyKey=f"prime:{capability.id}:{hashlib.sha256(canonical.encode()).hexdigest()}")
            expires = datetime.fromisoformat(capability.expires_at.replace("Z", "+00:00"))
            durable_capability = Capability(run_id, package.environment_id, capability.tool, capability.effect, {}, expires)
            run = self.controller.get_run(run_id)
            schema = self.registry.get_tool_schema(package.environment_id, capability.tool)
            if run is not None and run.execution_mode == "batch" and schema is not None and schema.effect == "write":
                request.approval_token = self.controller.broker.issue_approval(package.environment_id, run_id, capability.tool, dict(arguments), request.idempotency_key)
            result = self.controller.dispatch_tool(package.environment_id, request, durable_capability, provider, budget_remaining=budget_remaining, dry_run=run is not None and run.execution_mode == "dry_run")
            partition = self.controller.store.get_task(stored["task_id"]).get("partition") if stored and self.controller.store.get_task(stored["task_id"]) else None
            self.controller.append_broker_result(
                run_id,
                result.model_dump(mode="json", by_alias=True),
                development=partition == "development",
            )
            # Every broker attempt consumes the shared run call budget,
            # including failed/retried calls whose receipts remain auditable.
            budget_remaining["tool_calls"] = max(0, budget_remaining["tool_calls"] - 1)
            return result.model_dump(mode="json", by_alias=True)
        try:
            pinned_image = image_digest if isinstance(image_digest, str) and image_digest not in {"", "image-unpinned"} else None
            prime = PrimeRuntimeAdapter(PrimeRuntimeConfig(task_id=run_id, model="openai-codex/gpt-5.6-luna", provider="openai-codex", max_model_tokens=int(budget_data.get("modelTokens", DEFAULT_MODEL_TOKENS)) if isinstance(budget_data, Mapping) else DEFAULT_MODEL_TOKENS, max_total_wall_seconds=float(budget_data.get("wallTimeSeconds", 90)) if isinstance(budget_data, Mapping) else 90.0, child_runs=int(budget_data.get("childRuns", 0)) if isinstance(budget_data, Mapping) else 0, max_child_depth=int(budget_data.get("childDepth", 1)) if isinstance(budget_data, Mapping) else 1, docker_image=pinned_image, ao_session_id=os.environ.get("AO_SESSION_ID")), broker=CapabilityBroker(run_id, authorizer=authorize))
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
            if prime is not None:
                prime.close(remove_workspace=True)
            return
        runtime = self
        from adaptive_agent.prime_child_planner import LunaChildPlanner

        # Parent and child model calls share the adapter's trusted ledger. The
        # child planner is attached only after the authenticated client exists,
        # and its receipts are routed through the same parent-owned evidence
        # path as the main planner.
        child_planner = LunaChildPlanner(client, budget=prime.planner_budget)
        prime.child_planner = child_planner

        def record_child_model_observation(evidence: Mapping[str, Any]) -> Any:
            result = prime.record_model_observation(evidence, trusted_parent=True)
            runtime._record_model_response(run_id, package, {**dict(evidence), "arm": arm, "seed": seed, "bundleHash": bundle_hash, "corePlannerHash": core_planner_hash, "imageDigest": image_digest})
            return result

        child_planner.observation_sink = record_child_model_observation

        class Sink:
            def __init__(self, controller: Controller) -> None:
                self._controller = controller

            def record_model_observation(self, evidence: Mapping[str, Any], *, trusted_parent: bool = False) -> Any:
                result = prime.record_model_observation(evidence, trusted_parent=trusted_parent)
                runtime._record_model_response(run_id, package, {**dict(evidence), "arm": arm, "seed": seed, "bundleHash": bundle_hash, "corePlannerHash": core_planner_hash, "imageDigest": image_digest})
                child_planner.record_parent_model_usage(evidence["usage"])
                return result
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
                planner = LunaPlanner(client, prime, Sink(self._controller), limits=PlannerLimits(max_model_tokens=max(1, max_tokens), max_wall_seconds=max(0.1, wall_seconds)), emit=lambda event: self._controller.append_event(run_id, event.kind, {"summary": event.summary, "detail": event.detail}, "system", "operator"))
                self.result = planner.run(goal=task.goal, environment=runtime._planner_environment(package, run_id), active_skills=runtime._active_skills(run_id), cancel=cancel)
        driver = Driver(self.controller)
        def evaluate() -> DurableOutcome:
            result = driver.result
            if result is None or result.status != "succeeded":
                return DurableOutcome(runId=run_id, passed=False, metadata={"status": result.status if result else "planner_failed"})
            fixture = package.evaluate(task.task_id, provider.session)
            return DurableOutcome(runId=run_id, passed=fixture.passed, score=1.0 if fixture.passed else 0.0, metadata={"reason": fixture.reason, "evaluatorVersion": fixture.evaluator_version, "plannerStatus": result.status})
        try:
            self.controller.execute_run(run_id, package.environment_id, provider, driver, evaluate=evaluate, claimed=True)
        finally:
            if prime is not None:
                prime.close(remove_workspace=True)
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


def _seed_durable_stack(plane: ControlPlane, store_dir: Path) -> tuple[Controller, EnvironmentRegistry, dict[str, Any]]:
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
    evaluation_protocol: Any | None = None,
    evaluation_arm_bundles: Mapping[Any, Any] | None = None,
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
    controller, registry, packages = _seed_durable_stack(plane, Path(data_dir or os.environ.get("ADAPTIVE_AGENT_DATA", ".adaptive-agent")))
    # Materialize the built-in budget profile under its trusted content hash so
    # every RunRecord budgetRef resolves to immutable bytes, just like custom
    # UI budgets.
    controller.store.put_artifact(budget_profile)
    durable_runtime = DurableRuntime(controller, registry, packages, model_runner=model_runner, evaluator=evaluator, model_ref=model_ref, budget_ref=budget_ref, control_plane=plane, package_bindings=package_bindings, learning_model_client=learning_model_client, evaluation_executor=evaluation_executor)
    app = create_app(plane, durable_runtime=durable_runtime)
    app.state.controller = controller
    if evaluation_protocol is not None:
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
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_runtime_app(data_dir=args.data_dir, console_dist=args.console_dist), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
