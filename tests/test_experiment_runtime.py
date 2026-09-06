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
        return {"admissionId": key, "status": "reserved", "reused": False, "dispatchAllowed": True}

    def record(admission_id, *, result=None, error=None):
        checkpoints.append((admission_id, result, error))

    context = {
        "results": {"learning": {"candidate-generation": {"candidateBundleHash": "primary"}}},
        "admitSubcall": admit,
        "recordSubcall": record,
    }

    result = runner._adaptation("adapt:known-a", context, 0)

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
    assert len(result["chargedSubcallIds"]) == 3
    assert result["residualUsage"] == {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}


def test_partial_known_cost_remains_unknown():
    runtime = Runtime()
    runtime.controller.store.artifacts["acct-unknown"] = {
        "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
        "toolCalls": 1,
        "durationSeconds": 0.25,
        "economicCost": {"status": "unknown", "microunits": 7},
    }
    runtime.controller.store.artifacts["acct-nominal"] = {
        "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
        "toolCalls": 1,
        "durationSeconds": 0.25,
        "nominalCostUsd": 0.000007,
    }
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    first = SimpleNamespace(run_id="one", evidence_ref="model-1", outcome_ref="outcome-1", accounting_ref="acct-1")
    second = SimpleNamespace(run_id="two", evidence_ref="model-1", outcome_ref="outcome-1", accounting_ref="acct-unknown")

    receipt = experiment_runtime._observation_receipt(runtime, "transfer", "leave-out:known-a", [first, second], runner.pins)

    assert receipt["economicCostStatus"] == "unknown"
    assert "costMicrounits" not in receipt

    partial = SimpleNamespace(run_id="one", evidence_ref="model-1", outcome_ref="outcome-1", accounting_ref="acct-nominal")
    partial_receipt = experiment_runtime._observation_receipt(runtime, "transfer", "leave-out:known-a", [partial, second], runner.pins)
    assert partial_receipt["economicCostStatus"] == "unknown"
    assert "costMicrounits" not in partial_receipt


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
        return {"admissionId": "admission-1", "status": "reserved", "reused": False, "dispatchAllowed": True}

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
        context={"admitSubcall": lambda *_args, **_kwargs: {"admissionId": "admission-1", "status": "complete", "reused": True, "dispatchAllowed": False, "result": recovered}},
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


def test_transfer_charged_subcalls_are_accounted_once_by_real_evaluation_job(tmp_path, monkeypatch):
    pytest.importorskip("adaptive_agent.evaluation_job")
    from adaptive_agent.evaluation import EvaluationProtocol, build_environment_packages
    from adaptive_agent.evaluation_job import EvaluationJob, LifecycleStage
    from adaptive_agent.store import Store as EvaluationStore

    runtime = Runtime()
    runtime.launch_learning = lambda _payload: {
        "candidate": {"candidateId": "rotation", "candidateBundleHash": "rotation", "baseBundleHash": "base"},
        "wallSeconds": 0.01,
        "nominalCostUsd": 0.000004,
    }
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    monkeypatch.setattr(experiment_runtime, "_load_bundle", lambda _runtime, content_hash: Bundle(content_hash))
    monkeypatch.setattr(runner, "_learning_observation_usage", lambda _run_id, **_: ({"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, ["learning-ref"]))

    def generic(cell, context):
        return {
            "status": "complete",
            "stage": context["stage"],
            "cellKey": cell,
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            "toolCalls": 1,
            "wallSeconds": 0.01,
            "costMicrounits": 1,
        }

    def transfer(cell, context):
        return runner._transfer(cell, context, int(context["attempt"]))

    stages = (
        LifecycleStage("bootstrap", ("bootstrap-0",), generic),
        LifecycleStage("training", ("known-b-development-0",), lambda cell, context: {**generic(cell, context), "environmentId": "known-b", "runIds": ["dev-run"], "taskIds": [cell]}),
        LifecycleStage("learning", ("learning-0",), generic),
        LifecycleStage("transfer", ("leave-out:known-a",), transfer),
        LifecycleStage("adaptation", ("adaptation-0",), generic),
        LifecycleStage("safety", ("safety-0",), generic),
        LifecycleStage("validation", ("validation-0",), generic),
        LifecycleStage("final", ("final-0",), generic),
    )
    protocol = EvaluationProtocol()
    packages = build_environment_packages()
    protocol.freeze(packages)
    job = EvaluationJob(EvaluationStore(tmp_path), object(), protocol, packages, {}, lambda *_args: None)

    result = job.run_experiment(
        "charged-transfer",
        stages,
        limits={"attempts": 12, "inputTokens": 100, "outputTokens": 100, "toolCalls": 100, "wallMicros": 100_000_000, "costMicrounits": 100},
    )

    assert result.status == "complete", result.error
    accounting = job.lifecycle_accounting("charged-transfer")
    assert accounting["subcalls"] == 2
    assert accounting["costMicrounits"] == 18
    assert accounting["inputTokens"] == 11
    assert accounting["outputTokens"] == 10


def test_recovery_uses_nominal_cost_proxy_and_strict_store_verifier(tmp_path):
    evaluation = pytest.importorskip("adaptive_agent.evaluation")
    evidence_store = pytest.importorskip("adaptive_agent.evaluation_store")
    from adaptive_agent.store import Store as DurableStore
    import json

    packages = evaluation.build_environment_packages()
    protocol = evaluation.EvaluationProtocol(image_digest="sha256:image", core_planner_hash="core")
    frozen = protocol.freeze(packages)
    package = packages["finance"]
    task = package.tasks_for_partition(evaluation.Partition.VALIDATION)[0]
    store = DurableStore(tmp_path)
    bundle_hash = "b" * 64
    store.save_bundle("bundle-1", None, bundle_hash, "{}", is_active=True)
    store.register_task(task.task_id, "finance", package.manifest.version, "task-ref", "validation", task.goal)
    run_id = "validation-run"
    store.save_run(run_id, {"task_id": task.task_id, "environment_id": "finance", "bundle_id": "bundle-1", "status": "succeeded", "idempotency_key": "validation-key", "request_fingerprint": "validation-fingerprint", "last_event_sequence": 0, "created_at": "now", "run_json": json.dumps({"arm": "L", "seed": 17, "bundleHash": bundle_hash})})
    versions = {"model": evaluation.sha256_json({"profile": protocol.model_profile, "provider": protocol.provider}), "planner": protocol.core_planner_hash, "budget": evaluation.sha256_json(frozen.inputs["runBudget"]), "policy": evaluation.sha256_json(package.manifest.policy_ref), "schema": evaluation.sha256_json(package.manifest.tool_schemas), "image": protocol.image_digest}
    response_id = "response-validation"
    usage = {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
    accounting = {"responseId": response_id, "runId": run_id, "taskId": task.task_id, "environmentId": "finance", "usage": usage, "aggregateUsage": usage, "costMicrounits": None, "nominalCostUsd": 0.000012, "economicCost": {"status": "unknown", "microunits": None}, "durationSeconds": 1.25, "versionRefs": versions}
    accounting_ref = store.put_artifact(accounting)
    response = {"responseId": response_id, "provider": protocol.provider, "modelProfile": protocol.model_profile, "usage": usage, "versionRefs": versions, "arm": "L", "seed": 17, "bundleHash": bundle_hash, "accountingRef": accounting_ref.model_dump(mode="json", by_alias=True)}
    response_ref = store.put_artifact(response)
    store.append_evidence("validation-model", {"run_id": run_id, "sequence": 1, "event_type": "model_response", "content_hash": response_ref.sha256, "source_ref": response_ref.model_dump_json(by_alias=True), "trust_class": "broker", "visibility": "operator", "redacted": 0})
    outcome = {"responseId": response_id, "runId": run_id, "taskId": task.task_id, "environmentId": "finance", "passed": True, "reliable": True, "safetyViolations": 0, "fixtureResetOk": True, "arm": "L", "seed": 17, "bundleHash": bundle_hash}
    outcome_ref = store.put_artifact(outcome)
    store.append_evidence("validation-outcome", {"run_id": run_id, "sequence": 2, "event_type": "trusted_outcome", "content_hash": outcome_ref.sha256, "source_ref": outcome_ref.model_dump_json(by_alias=True), "trust_class": "evaluator", "visibility": "operator", "redacted": 0})

    runtime = SimpleNamespace(controller=SimpleNamespace(store=store), core_planner_hash="core", image_digest="sha256:image", packages=packages)
    runtime.controller.get_active_bundle = lambda: Bundle(bundle_hash)
    runner = DefaultExperimentStageRunner(runtime, protocol)
    receipt = {"stage": "validation", "cellKey": "validation:0", "runIds": [run_id], "taskIds": [task.task_id], "evidenceRefs": ["validation-model"], "outcomeRefs": ["validation-outcome"]}
    observations = runner.recover_evaluation_observations(receipt)
    assert observations[0].cost_microunits == 12
    assert evidence_store.SQLiteRunEvidenceStore(store).verify(observations[0], frozen, package)
