"""Executable application factory for the integrated Adaptive Agent stack.

The factory wires the durable Store/EnvironmentRegistry/ToolBroker/Controller
seam into the operator API and seeds only trusted fixture references.  A live
model runner is injected by deployment, so the default process cannot silently
claim that an unavailable provider is a simulation or a successful run.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from adaptive_agent.api import ControlPlane, ModelRunner, OutcomeEvaluator, create_app, _ref, IdempotencyConflict
from adaptive_agent.broker import Capability, ToolBroker, ToolProvider
from adaptive_agent.controller import Controller
from adaptive_agent.environment import EnvironmentRegistry
from adaptive_agent.evaluation import build_environment_packages
from adaptive_agent.planner import PrimeCliModelClient, LunaPlanner, PlannerResult, PlannerLimits
from adaptive_agent.prime_runtime import Capability as PrimeCapability, CapabilityBroker, PrimeRuntimeAdapter, PrimeRuntimeConfig
from adaptive_agent.models import ArtifactRef, EnvironmentManifest as DurableManifest, TaskInput as DurableTask, ToolSchema as DurableTool, RunStatus, ToolRequest, Outcome as DurableOutcome
from adaptive_agent.store import Store
from adaptive_agent.evaluation_store import build_durable_adapters


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

    def __init__(self, package: Any, task: Any, run_id: str) -> None:
        self.package = package
        self.session = package.reset(task.task_id, 0)
        self._run_id = run_id


class DurableRuntime:
    """Adapter used by the HTTP layer to make Controller the source of truth."""

    def __init__(self, controller: Controller, registry: EnvironmentRegistry, packages: Mapping[str, Any], model_runner: Any | None = None, evaluator: Any | None = None, model_ref: Mapping[str, Any] | None = None, budget_ref: Mapping[str, Any] | None = None, control_plane: Any | None = None) -> None:
        self.controller, self.registry, self.packages = controller, registry, dict(packages)
        self.model_runner, self.evaluator = model_runner, evaluator
        self.model_ref, self.budget_ref = dict(model_ref or _ref("model-profile", "1")), dict(budget_ref or _ref("budget-default", "1"))
        self.control_plane = control_plane
        self._cancel_events: dict[str, threading.Event] = {}
        self._approvals: dict[tuple[str, str], bool] = {}
        self._tasks = {
            task.task_id: task
            for name in self.packages
            for task in registry.list_tasks_by_partition(name, "development")
        }

    def list_environments(self) -> list[dict[str, Any]]:
        registered = {task.environment_ref.id for task in self._tasks.values()}
        return [{"environmentId": name, "version": package.manifest.version, "validationState": "valid", "evaluatorReady": True, "toolCount": len(package.manifest.tool_schemas), "policyScope": package.manifest.policy_ref.id} for name, package in self.packages.items() if name in registered]

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

    def launch(self, run_id: str) -> None:
        stored = self.controller.store.get_run(run_id)
        if not stored:
            raise KeyError("run not found")
        task = self.registry.get_task(stored["task_id"])
        package = self.packages.get(stored["environment_id"])
        if not task or not package:
            raise KeyError("development task not found")
        cancel = self._cancel_events.setdefault(run_id, threading.Event())
        provider: _FixtureProvider
        if self.model_runner is not None:
            provider = _FixtureProvider(package, task, run_id)
            invocation: Any = None
            model_runner = self.model_runner
            runtime = self
            class DirectDriver:
                def act(self, _ctx: Any) -> None:
                    nonlocal invocation
                    invocation = model_runner(goal=task.goal, environment=runtime._planner_environment(package, run_id), emit=lambda kind, summary, detail=None: runtime.controller.append_event(run_id, kind, {"summary": summary, "detail": detail}, "system", "operator"))
                    runtime.controller.append_event(run_id, "model_observation", {"provider": invocation.provider, "model": invocation.model, "responseId": invocation.response_id, "usage": dict(invocation.usage)}, "system", "operator")
            def evaluate() -> DurableOutcome:
                outcome = dict(self.evaluator(goal=task.goal, model_output=invocation.text, environment=self._planner_environment(package, run_id))) if self.evaluator is not None and invocation is not None else {"passed": False}
                return DurableOutcome(runId=run_id, passed=bool(outcome.get("passed") is True), metadata=outcome)
            self.controller.execute_run(run_id, package.environment_id, provider, DirectDriver(), evaluate=evaluate)
            self._cancel_events.pop(run_id, None)
            return
        prime: PrimeRuntimeAdapter | None = None
        provider = _FixtureProvider(package, task, run_id)
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
            if run is not None and run.execution_mode == "dry_run" and schema is not None and schema.effect == "write":
                request.approval_token = self.controller.broker.issue_approval(package.environment_id, run_id, capability.tool, dict(arguments), request.idempotency_key)
            result = self.controller.dispatch_tool(package.environment_id, request, durable_capability, provider, budget_remaining=budget_remaining)
            if result.status == "ok":
                budget_remaining["tool_calls"] = max(0, budget_remaining["tool_calls"] - 1)
            return result.model_dump(mode="json", by_alias=True)
        try:
            prime = PrimeRuntimeAdapter(PrimeRuntimeConfig(task_id=run_id), broker=CapabilityBroker(run_id, authorizer=authorize))
            for tool in package.manifest.tool_schemas:
                expires = (datetime.now(timezone.utc) + timedelta(seconds=90)).isoformat()
                prime.broker.register(PrimeCapability(f"{run_id}:{tool.name}", tool.name, tool.version, tool.effect, "run", expires))
            coding_dir = os.environ.get("PRIME_AGENT_CODING_AGENT_DIR")
            if not coding_dir:
                raise RuntimeError("PRIME_AGENT_CODING_AGENT_DIR is required for the Prime CLI model client")
            client = PrimeCliModelClient(coding_agent_dir=coding_dir)
        except Exception as exc:
            self.controller.append_event(run_id, "run_failed", {"error": str(exc)}, "system", "operator")
            if (current := self.controller.get_run(run_id)) is not None and current.status != RunStatus.cancelled:
                self.controller._set_run_status(run_id, RunStatus.failed)
            self._cancel_events.pop(run_id, None)
            if prime is not None:
                prime.close(remove_workspace=True)
            return
        runtime = self
        class Sink:
            def __init__(self, controller: Controller) -> None:
                self._controller = controller

            def record_model_observation(self, evidence: Mapping[str, Any], *, trusted_parent: bool = False) -> Any:
                result = prime.record_model_observation(evidence, trusted_parent=trusted_parent)
                self._controller.append_event(run_id, "model_observation", dict(evidence), "system", "operator")
                return result
        class Driver:
            def __init__(self, controller: Controller) -> None:
                self._controller = controller
                self.result: PlannerResult | None = None
            def act(self, ctx: Any) -> None:
                max_tokens = int(budget_data.get("modelTokens", 4000)) if isinstance(budget_data, Mapping) else 4000
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
            self.controller.execute_run(run_id, package.environment_id, provider, driver, evaluate=evaluate)
        finally:
            if prime is not None:
                prime.close(remove_workspace=True)
            self._cancel_events.pop(run_id, None)

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
):
    """Build the API, durable controller seam, and optional static console."""
    kwargs: dict[str, Any] = {}
    if model_runner is not None:
        kwargs["model_runner"] = model_runner
    if evaluator is not None:
        kwargs["evaluator"] = evaluator
    model_profile = {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "tier": "medium", "maxTokens": 4000}
    budget_profile = {"modelTokens": 4000, "toolCalls": 32, "childRuns": 0, "wallTimeSeconds": 90, "costMicrounits": 100000, "currency": "USD"}
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
    durable_runtime = DurableRuntime(controller, registry, packages, model_runner=model_runner, evaluator=evaluator, model_ref=model_ref, budget_ref=budget_ref, control_plane=plane)
    app = create_app(plane, durable_runtime=durable_runtime)
    app.state.controller = controller
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
