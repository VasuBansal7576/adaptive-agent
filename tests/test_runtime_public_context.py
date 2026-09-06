import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import adaptive_agent.app as app_module
from adaptive_agent.prime_runtime import ChildPlannerBudget, SharedBudget


class ContextProvider:
    def __init__(self, package, task, run_id, seed, context):
        self._inner = app_module._FixtureProvider(package, task, run_id, seed)
        self._context = context

    def public_context(self):
        return self._context

    def execute(self, run_id, tool, arguments):
        return self._inner.execute(run_id, tool, arguments)

    def effect(self, tool):
        return self._inner.effect(tool)

    def version(self, tool):
        return self._inner.version(tool)


class DirectInvocation:
    text = "done"
    provider = "openai-codex"
    model = "openai-codex/gpt-5.6-luna"
    response_id = "public-context-direct"
    usage = {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}


class PrimeClient:
    def __init__(self):
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "provider": "openai-codex",
            "model": "openai-codex/gpt-5.6-luna",
            "responseId": "public-context-prime",
            "text": json.dumps({"action": "finish", "answer": "done"}),
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }


class FakePrime:
    def __init__(self, config, broker):
        self.config = config
        self.broker = broker
        self.child_planner = None
        self._budget = SharedBudget(
            config.max_total_wall_seconds,
            config.max_total_artifact_bytes,
            config.max_artifact_count,
            config.child_runs,
            max(config.max_model_tokens, 100),
        )

    @property
    def planner_budget(self):
        return ChildPlannerBudget(self._budget)

    def record_model_observation(self, evidence, *, trusted_parent=False):
        return evidence

    def execute(self, code, *, timeout=None, cancel=None):
        return SimpleNamespace(status="ok", result="null", stdout="", stderr="", error=None)

    def close(self, remove_workspace=True):
        return None


def _run_with_provider(tmp_path, context, *, model_runner, monkeypatch):
    app = app_module.create_runtime_app(
        data_dir=tmp_path,
        model_runner=model_runner,
        evaluator=lambda **_: {"passed": True},
    )
    runtime = app.state.durable_runtime
    package = runtime.packages["finance"]
    package.provider_factory = lambda task, run_id, seed: ContextProvider(package, task, run_id, seed, context)
    api = TestClient(app, base_url="http://127.0.0.1")
    assert api.get("/session/bootstrap").status_code == 200
    task = package.tasks_for_partition("development")[0]
    response = api.post(
        "/runs",
        json={
            "goal": task.goal,
            "environmentId": "finance",
            "idempotencyKey": "public-context",
            "budget": {
                "modelTokens": 100,
                "toolCalls": 1,
                "childRuns": 0,
                "wallTimeSeconds": 30,
                "costMicrounits": 1000,
                "currency": "USD",
            },
        },
    )
    assert response.status_code == 201, response.text
    return app, runtime, response.json()["runId"], package


def _assert_authority_is_unchanged(environment, runtime, run_id, package):
    run = runtime.controller.get_run(run_id)
    assert run is not None
    assert environment["environmentId"] == package.environment_id
    assert environment["version"] == package.manifest.version
    assert environment["toolSchemas"] == [tool.to_dict() for tool in package.manifest.tool_schemas]
    assert environment["capabilities"] == [f"{run_id}:{tool.name}" for tool in package.manifest.tool_schemas]
    assert environment["budgetRef"] == run.budget_ref.model_dump(mode="json", by_alias=True)


def test_direct_runner_receives_namespaced_provider_context_without_authority_override(tmp_path):
    context = {
        "taskId": "finance-development-00",
        "allowedApps": ["phone"],
        "capabilities": ["attacker-capability"],
        "toolSchemas": [{"name": "attacker-tool"}],
        "version": "attacker-version",
        "budgetRef": {"sha256": "attacker"},
    }
    seen = []

    def runner(**kwargs):
        seen.append(kwargs)
        return DirectInvocation()

    app, runtime, run_id, package = _run_with_provider(tmp_path, context, model_runner=runner, monkeypatch=None)
    runtime.launch(run_id)
    assert len(seen) == 1
    environment = seen[0]["environment"]
    assert environment["taskContext"] == context
    _assert_authority_is_unchanged(environment, runtime, run_id, package)


def test_prime_driver_receives_namespaced_provider_context(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "PrimeRuntimeAdapter", FakePrime)
    context = {"taskId": "finance-development-00", "instruction": "public task details", "appDescriptions": {"phone": "A public app"}}
    client = PrimeClient()
    app, runtime, run_id, package = _run_with_provider(tmp_path, context, model_runner=None, monkeypatch=monkeypatch)
    runtime.launch(run_id, model_client_override=client)
    assert len(client.calls) == 1
    environment = client.calls[0]["environment"]
    assert environment["taskContext"] == context
    _assert_authority_is_unchanged(environment, runtime, run_id, package)


@pytest.mark.parametrize("context", [None, {"oversized": "x" * (64 * 1024)}])
@pytest.mark.parametrize("prime", [False, True])
def test_invalid_provider_context_is_rejected_before_model_dispatch(tmp_path, monkeypatch, context, prime):
    if prime:
        monkeypatch.setattr(app_module, "PrimeRuntimeAdapter", FakePrime)
        model = PrimeClient()
        app, runtime, run_id, _ = _run_with_provider(tmp_path, context, model_runner=None, monkeypatch=monkeypatch)
        runtime.launch(run_id, model_client_override=model)
        assert model.calls == []
    else:
        calls = []

        def runner(**kwargs):
            calls.append(kwargs)
            return DirectInvocation()

        app, runtime, run_id, _ = _run_with_provider(tmp_path, context, model_runner=runner, monkeypatch=monkeypatch)
        runtime.launch(run_id)
        assert calls == []
    assert runtime.controller.get_run(run_id).status.value == "failed"
