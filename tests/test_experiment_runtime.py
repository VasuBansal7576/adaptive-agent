from __future__ import annotations

from types import SimpleNamespace

import pytest

from adaptive_agent.experiment_runtime import (
    DefaultExperimentStageRunner,
    ExperimentRuntimeError,
)
import adaptive_agent.experiment_runtime as experiment_runtime


class Bundle:
    def __init__(self, content_hash: str):
        self.content_hash = content_hash


class Store:
    def __init__(self):
        self.artifacts = {
            "acct-1": {
                "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
                "toolCalls": 1,
                "durationSeconds": 0.25,
                "costMicrounits": 7,
            }
        }

    def get_bundle_by_hash(self, content_hash):
        return {"bundle_json": "{}", "content_hash": content_hash}

    def get_artifact(self, ref):
        return self.artifacts[ref]

    def get_evidence(self, evidence_id):
        if evidence_id == "model-1":
            return {"event_type": "model_response"}
        if evidence_id == "outcome-1":
            return {"event_type": "trusted_outcome"}
        return None

    def get_run(self, run_id):
        return {"status": "succeeded"} if run_id == "dev-run" else None

    def get_outcome_by_run_id(self, run_id):
        return {"passed": 1} if run_id == "dev-run" else None

    def list_evidence(self, _run_id):
        return []


class Package:
    def __init__(self, environment_id):
        self.environment_id = environment_id
        self._tasks = {
            "development": [self._task("development", 0)],
            "validation": [self._task("validation", 0)],
            "final": [self._task("final", 0)],
        }

    def _task(self, partition, index):
        return SimpleNamespace(
            task_id=f"{self.environment_id}-{partition}-{index}",
            environment_ref=SimpleNamespace(id=self.environment_id),
            goal=f"{partition} goal",
        )

    def tasks_for_partition(self, partition):
        return tuple(self._tasks[partition])


class Controller:
    def __init__(self):
        self.store = Store()

    def get_active_bundle(self):
        return Bundle("base")


class Runtime:
    core_planner_hash = "core"
    image_digest = "image"

    def __init__(self):
        self.controller = Controller()
        self.packages = {name: Package(name) for name in ("known-a", "known-b", "known-c", "sealed")}
        self.calls = []
        self.configs = []

    def execute_evaluation_task(self, task, config, bundle):
        self.calls.append((task.task_id, config.arm, config.seed, bundle.content_hash))
        self.configs.append(config)
        return SimpleNamespace(
            run_id=f"run-{len(self.calls)}",
            evidence_ref="model-1",
            outcome_ref="outcome-1",
            accounting_ref="acct-1",
            response_id=f"response-{len(self.calls)}",
            model_provenance="real_model",
        )

    def verify_evaluation_observation(self, observation, config, task):
        return True

    def establish_clean_experiment(self, protocol):
        return {"clean": True, "actualDocker": True, "provenanceRef": "clean-provenance"}


class Protocol:
    known_environments = ("known-a", "known-b", "known-c")
    sealed_environment = "sealed"
    tasks_per_environment = 1
    seeds = (17, 23, 29)
    safety_case_ids = ("EVAL-004",)

    def start_candidate_generation(self):
        return SimpleNamespace(
            protocol_hash="protocol",
            inputs={
                "provider": "openai-codex",
                "modelProfile": "openai-codex/gpt-5.6-luna",
                "corePlannerHash": "core",
                "imageDigest": "image",
                "runBudget": {"modelTokens": 20},
            },
        )


def test_default_runner_executes_real_receipt_bound_training_cell():
    runtime = Runtime()
    runner = DefaultExperimentStageRunner(runtime, Protocol())

    result = runner(cell_key="known-a-development-0", context={"stage": "training", "attempt": 0})

    assert result["status"] == "complete"
    assert result["runIds"] == ["run-1"]
    assert result["usage"] == {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}
    assert runtime.calls == [("known-a-development-0", "B0", 17, "base")]
    assert runtime.configs[0].protocol.inputs["modelProfile"] == "openai-codex/gpt-5.6-luna"


def test_panel_mapping_keeps_frozen_validation_and_final_counts_without_duplication():
    runtime = Runtime()
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    runner._candidate = lambda _context: Bundle("candidate")
    context = {"stage": "validation", "attempt": 0, "results": {"learning": {"candidate-generation": {"candidateBundleHash": "candidate"}}}}

    first = runner(cell_key="validation:0", context=context)
    second = runner(cell_key="validation:1", context=context)
    assert first["arm"] == "B0" and second["arm"] == "L"
    assert first["environmentId"] == second["environmentId"] == "known-a"
    assert first["seed"] == second["seed"] == 17
    with pytest.raises(ExperimentRuntimeError):
        runner(cell_key="validation:360", context=context)

    final_context = {**context, "stage": "final", "ablationBundleHash": "base"}
    sealed = runner(cell_key="final:27", context=final_context)
    assert sealed["environmentId"] == "sealed"
    assert sealed["sealed"] is True
    with pytest.raises(ExperimentRuntimeError):
        runner(cell_key="final:720", context=final_context)


def test_sealed_environment_is_rejected_before_final():
    runtime = Runtime()
    runner = DefaultExperimentStageRunner(runtime, Protocol())

    with pytest.raises(ExperimentRuntimeError, match="sealed environment"):
        runner(cell_key="leave-out:sealed", context={"stage": "transfer", "results": {}})


def test_rotation_candidate_does_not_replace_primary_candidate(monkeypatch):
    runtime = Runtime()
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    runner._candidate_hash = "primary"
    monkeypatch.setattr(runner, "_learning_observation_usage", lambda _run_id, **_: ({"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, ["learning-ref"]))

    runner._learning_receipt(
        "leave-out:known-a",
        "dev-run",
        {"candidate": {"candidateId": "rotation", "candidateBundleHash": "rotation", "baseBundleHash": "base"}},
        bind_primary=False,
    )

    assert runner._candidate_hash == "primary"


def test_adaptation_learns_from_support_before_query(monkeypatch):
    runtime = Runtime()
    runtime.launch_learning = lambda payload: {"candidate": {"candidateId": "adapted", "candidateBundleHash": "adapted", "baseBundleHash": "base"}}
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    runtime.controller.store.get_run = lambda _run_id: {"status": "succeeded"}
    runtime.controller.store.get_outcome_by_run_id = lambda _run_id: {"passed": True}
    monkeypatch.setattr(runner, "_candidate", lambda _context: Bundle("primary"))
    monkeypatch.setattr(experiment_runtime, "_load_bundle", lambda _runtime, content_hash: Bundle(content_hash))
    monkeypatch.setattr(runner, "_learning_observation_usage", lambda _run_id, **_: ({"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, ["learning-ref"]))
    admissions = []
    checkpoints = []

    def admit(key, **_estimates):
        admissions.append(key)
        return {"admissionId": key, "status": "reserved", "reused": False}

    def record(admission_id, *, result=None, error=None):
        checkpoints.append((admission_id, result, error))

    context = {
        "results": {"learning": {"candidate-generation": {"candidateBundleHash": "primary"}}},
        "admitSubcall": admit,
        "recordSubcall": record,
    }

    runner._adaptation("adapt:known-a", context, 0)

    assert runtime.calls[-2:] == [
        ("known-a-development-0", "L", 17, "primary"),
        ("known-a-validation-0", "L", 23, "adapted"),
    ]
    assert admissions == [
        "adaptation:adapt:known-a:support",
        "learning:adapt:known-a:run-1",
        "adaptation:adapt:known-a:query",
    ]
    assert all(error is None for _, _, error in checkpoints)


def test_partial_known_cost_remains_unknown():
    runtime = Runtime()
    runtime.controller.store.artifacts["acct-unknown"] = {
        "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
        "toolCalls": 1,
        "durationSeconds": 0.25,
        "economicCost": {"status": "unknown", "microunits": 7},
    }
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    first = SimpleNamespace(run_id="one", evidence_ref="model-1", outcome_ref="outcome-1", accounting_ref="acct-1")
    second = SimpleNamespace(run_id="two", evidence_ref="model-1", outcome_ref="outcome-1", accounting_ref="acct-unknown")

    receipt = experiment_runtime._observation_receipt(runtime, "transfer", "leave-out:known-a", [first, second], runner.pins)

    assert receipt["economicCostStatus"] == "unknown"
    assert "costMicrounits" not in receipt


def test_learning_receipt_preserves_nominal_proxy_and_wall_time(monkeypatch):
    runtime = Runtime()
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    monkeypatch.setattr(runner, "_learning_observation_usage", lambda _run_id, **_: ({"inputTokens": 4, "outputTokens": 3, "totalTokens": 7}, ["learning-ref"]))

    receipt = runner._learning_receipt(
        "learning",
        "dev-run",
        {
            "candidate": {"candidateId": "candidate", "candidateBundleHash": "candidate", "baseBundleHash": "base"},
            "wallSeconds": 1.25,
            "nominalCostUsd": 0.000004,
        },
        bind_primary=True,
    )

    assert receipt["wallSeconds"] == 1.25
    assert receipt["costMicrounits"] == 4
    assert receipt["costBasis"] == "nominal_budget_proxy"
    assert receipt["billingStatus"] == "unknown"


def test_event_type_only_evidence_is_not_accepted():
    runtime = Runtime()
    runtime.verify_evaluation_observation = lambda *_args: False
    runner = DefaultExperimentStageRunner(runtime, Protocol())

    with pytest.raises(ExperimentRuntimeError, match="strict verifier"):
        runner(cell_key="known-a-development-0", context={"stage": "training", "attempt": 0})


def test_bootstrap_requires_clean_provenance_receipt():
    runtime = Runtime()
    runtime.establish_clean_experiment = lambda _protocol: {"clean": False, "actualDocker": True, "provenanceRef": "dirty"}
    runner = DefaultExperimentStageRunner(runtime, Protocol())

    with pytest.raises(ExperimentRuntimeError, match="provenance"):
        runner(cell_key="bootstrap", context={"stage": "bootstrap", "attempt": 0})


def test_nested_learning_is_admitted_and_checkpointed(monkeypatch):
    runtime = Runtime()
    runtime.launch_learning = lambda payload: {"candidate": {"candidateId": "nested", "candidateBundleHash": "nested", "baseBundleHash": "base"}}
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    monkeypatch.setattr(runner, "_learning_observation_usage", lambda _run_id, **_: ({"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, ["learning-ref"]))
    admissions = []
    checkpoints = []

    def admit(key, **estimates):
        admissions.append((key, estimates))
        return {"admissionId": "admission-1", "status": "reserved", "reused": False}

    def record(admission_id, *, result=None, error=None):
        checkpoints.append((admission_id, result, error))

    runner._candidate_from_run(
        "dev-run",
        "transfer:known-a",
        bind_primary=False,
        context={"admitSubcall": admit, "recordSubcall": record},
    )

    assert admissions[0][0] == "learning:transfer:known-a:dev-run"
    assert admissions[0][1]["estimated_cost_microunits"] == 0
    assert checkpoints[0][0] == "admission-1"
    assert checkpoints[0][1]["candidateBundleHash"] == "nested"


def test_nested_learning_reuses_durable_checkpoint_without_relaunch():
    runtime = Runtime()
    runtime.launch_learning = lambda _payload: pytest.fail("reused nested call must not relaunch")
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    recovered = {"candidateId": "nested", "candidateBundleHash": "nested", "baseBundleHash": "base", "status": "complete"}

    receipt = runner._candidate_from_run(
        "dev-run",
        "transfer:known-a",
        bind_primary=False,
        context={"admitSubcall": lambda *_args, **_kwargs: {"admissionId": "admission-1", "status": "complete", "reused": True, "result": recovered}},
    )

    assert receipt is recovered


def test_actual_durable_runtime_rejects_unfrozen_execution_before_model_dispatch():
    pytest.importorskip("fastapi")
    pytest.importorskip("adaptive_agent.constants")
    from adaptive_agent.app import DurableRuntime, LearningRuntimeError

    class Package:
        def reset(self, *_args, **_kwargs):
            return object()

        def evaluate(self, *_args, **_kwargs):
            return object()

    class ModelClient:
        calls = 0

        def invoke(self, **_kwargs):
            self.calls += 1
            return {}

    runtime = DurableRuntime.__new__(DurableRuntime)
    runtime.packages = {"known-a": Package()}
    runtime.learning_model_client = ModelClient()
    task = SimpleNamespace(
        task_id="known-a-development-0",
        environment_ref=SimpleNamespace(id="known-a", version="1"),
        goal="probe",
    )
    config = SimpleNamespace(protocol=SimpleNamespace(), arm="B0", seed=17, bundle_hash="base")

    with pytest.raises(LearningRuntimeError, match="not frozen"):
        runtime.execute_evaluation_task(task, config, Bundle("base"))
    assert runtime.learning_model_client.calls == 0


@pytest.mark.parametrize("zero_field", ["modelTokens", "costMicrounits"])
def test_actual_durable_runtime_rejects_zero_budget_before_model_dispatch(tmp_path, zero_field):
    pytest.importorskip("fastapi")
    pytest.importorskip("adaptive_agent.constants")
    from fastapi.testclient import TestClient
    from adaptive_agent.app import create_runtime_app

    calls = []

    def deterministic_runner(**_kwargs):
        calls.append(True)
        return SimpleNamespace(
            text="deterministic response",
            provider="openai-codex",
            model="openai-codex/gpt-5.6-luna",
            response_id="deterministic-response",
            usage={"inputTokens": 1, "outputTokens": 1},
        )

    app = create_runtime_app(model_runner=deterministic_runner, evaluator=lambda **_: {"passed": True}, data_dir=tmp_path)
    api = TestClient(app, base_url="http://127.0.0.1")
    assert api.get("/session/bootstrap").status_code == 200
    task = app.state.durable_runtime.packages["finance"].tasks_for_partition("development")[0]
    run = api.post(
        "/runs",
        json={
            "goal": task.goal,
            "environmentId": "finance",
            "idempotencyKey": f"zero-budget-{zero_field}",
            "budget": {
                "modelTokens": 0 if zero_field == "modelTokens" else 1,
                "toolCalls": 1,
                "childRuns": 0,
                "wallTimeSeconds": 1,
                "costMicrounits": 0 if zero_field == "costMicrounits" else 1,
                "currency": "USD",
            },
        },
    ).json()
    with pytest.raises(RuntimeError, match="budgets must be positive"):
        app.state.durable_runtime.launch(run["runId"])
    assert calls == []
