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
from pathlib import Path
from typing import Any, Mapping

from adaptive_agent.api import ControlPlane, ModelRunner, OutcomeEvaluator, create_app, _ref
from adaptive_agent.broker import ToolBroker
from adaptive_agent.controller import Controller
from adaptive_agent.environment import EnvironmentRegistry
from adaptive_agent.evaluation import build_environment_packages
from adaptive_agent.planner import PrimeCliModelClient, LunaPlanner
from adaptive_agent.prime_runtime import Capability as PrimeCapability, CapabilityBroker, PrimeRuntimeAdapter, PrimeRuntimeConfig
from adaptive_agent.models import ArtifactRef, EnvironmentManifest as DurableManifest, TaskInput as DurableTask, ToolSchema as DurableTool, RunStatus
from adaptive_agent.store import Store


class _FixtureProvider:
    """Prime host-request provider backed by one reset, trusted fixture task."""

    def __init__(self, package: Any, task: Any) -> None:
        self.package = package
        self.session = package.reset(task.task_id, 0)

    def call(self, tool: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        result = self.package.invoke(self.session, tool, arguments)
        return {"tool": result.tool, "status": result.status, "output": result.output, "sideEffect": result.side_effect}


class DurableRuntime:
    """Adapter used by the HTTP layer to make Controller the source of truth."""

    def __init__(self, controller: Controller, registry: EnvironmentRegistry, packages: Mapping[str, Any], model_runner: Any | None = None, evaluator: Any | None = None, model_ref: Mapping[str, Any] | None = None, budget_ref: Mapping[str, Any] | None = None) -> None:
        self.controller, self.registry, self.packages = controller, registry, dict(packages)
        self.model_runner, self.evaluator = model_runner, evaluator
        self.model_ref, self.budget_ref = dict(model_ref or _ref("model-profile", "1")), dict(budget_ref or _ref("budget-default", "1"))
        self._tasks = {name: registry.list_tasks_by_partition(name, "development")[0] for name, package in self.packages.items() if registry.list_tasks_by_partition(name, "development")}

    def list_environments(self) -> list[dict[str, Any]]:
        return [{"environmentId": name, "version": package.manifest.version, "validationState": "ready", "evaluatorReady": True, "toolCount": len(package.manifest.tool_schemas), "policyScope": package.manifest.policy_ref.id} for name, package in self.packages.items() if name in self._tasks]

    @staticmethod
    def _public_run(run: Any, task: Any) -> dict[str, Any]:
        value = run.model_dump(mode="json", by_alias=True)
        value.update({"environmentId": task.environment_ref.id, "goal": task.goal, "executionMode": "interactive"})
        return value

    def create_run(self, payload: Any) -> dict[str, Any]:
        task = self._tasks.get(payload.environment_id)
        if task is None:
            raise KeyError("environment is not registered for development runs")
        from adaptive_agent.models import ArtifactRef, RunRequest

        task_ref = ArtifactRef(id=task.task_id, version="1", sha256=__import__("hashlib").sha256(task.task_id.encode()).hexdigest())
        model_ref = ArtifactRef.model_validate(payload.model_profile_ref or self.model_ref)
        budget_ref = ArtifactRef.model_validate(payload.budget_ref or self.budget_ref)
        run = self.controller.create_run(RunRequest(taskRef=task_ref, modelProfileRef=model_ref, budgetRef=budget_ref, idempotencyKey=payload.idempotency_key), task)
        return self._public_run(run, task)

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
        if self.model_runner is not None:
            self.controller._set_run_status(run_id, RunStatus.running)
            invocation = self.model_runner(goal=task.goal, environment={"environmentId": package.environment_id}, emit=lambda kind, summary, detail=None: self.controller.append_event(run_id, kind, {"summary": summary, "detail": detail}, "system", "operator"))
            self.controller.append_event(run_id, "model_observation", {"provider": invocation.provider, "model": invocation.model, "responseId": invocation.response_id, "usage": dict(invocation.usage)}, "system", "operator")
            outcome = dict(self.evaluator(goal=task.goal, model_output=invocation.text, environment={"environmentId": package.environment_id})) if self.evaluator is not None else {"passed": False}
            self.controller.record_outcome(run_id, bool(outcome.get("passed") is True), metadata=outcome)
            self.controller._set_run_status(run_id, RunStatus.succeeded if outcome.get("passed") is True else RunStatus.failed)
            return
        prime = PrimeRuntimeAdapter(PrimeRuntimeConfig(task_id=run_id))
        provider = _FixtureProvider(package, task)
        for tool in package.manifest.tool_schemas:
            prime.broker.register(PrimeCapability(f"{run_id}:{tool.name}", tool.name, tool.version, tool.effect, "run", "never"), lambda args, name=tool.name: provider.call(name, args))
        coding_dir = os.environ.get("PRIME_AGENT_CODING_AGENT_DIR") or str(Path.home() / ".local" / "share" / "opencode")
        client = PrimeCliModelClient(coding_agent_dir=coding_dir)
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

            def act(self, ctx: Any) -> None:
                planner = LunaPlanner(client, prime, Sink(self._controller), emit=lambda event: self._controller.append_event(run_id, event.kind, {"summary": event.summary, "detail": event.detail}, "system", "operator"))
                planner.run(goal=task.goal, environment={"environmentId": package.environment_id, "version": package.manifest.version, "toolSchemas": [tool.to_dict() for tool in package.manifest.tool_schemas], "executionModes": list(package.manifest.execution_modes), "capabilities": [f"{run_id}:{tool.name}" for tool in package.manifest.tool_schemas]})
        try:
            self.controller.execute_run(run_id, package.environment_id, provider, Driver(self.controller))
            outcome = package.evaluate(task.task_id, provider.session)
            self.controller.record_outcome(run_id, outcome.passed, metadata={"reason": outcome.reason, "evaluatorVersion": outcome.evaluator_version})
        finally:
            prime.close(remove_workspace=True)

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        return self.controller.events(run_id, after)

    def cancel(self, run_id: str) -> dict[str, Any]:
        run = self.controller.cancel_run(run_id)
        if run is None:
            raise KeyError("run not found")
        return run.model_dump(mode="json", by_alias=True)


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
    durable_runtime = DurableRuntime(controller, registry, packages, model_runner=model_runner, evaluator=evaluator, model_ref=model_ref, budget_ref=budget_ref)
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
