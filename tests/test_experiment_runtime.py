from __future__ import annotations

import json
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
        self.evidence = []

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
        env = getattr(self, "run_envs", {}).get(run_id, "known-b")
        return {"status": "succeeded", "environment_id": env, "task_id": "dev-task"} if run_id == "dev-run" else None

    def get_task(self, task_id):
        return {"partition": "development", "goal": "dev goal"} if task_id == "dev-task" else None

    def put_artifact(self, data):
        return SimpleNamespace(model_dump=lambda **_kw: {"id": "sel", "version": "1", "sha256": "selsha"})

    def get_outcome_by_run_id(self, run_id):
        return {"passed": 1} if run_id == "dev-run" else None

    def list_evidence(self, _run_id):
        return list(self.evidence)


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


def test_learning_accounting_requires_cost_for_every_wanted_receipt():
    runtime = Runtime()
    runtime.controller.store.artifacts.update({
        "acct-a": {"wallSeconds": 1.0, "costMicrounits": 12, "economicCostStatus": "measured"},
        "acct-b": {"wallSeconds": 1.0},
    })
    runtime.controller.store.evidence = [
        {"evidence_id": "a", "event_type": "learning_model_observation", "source_ref": json.dumps({"sha256": "acct-a"})},
        {"evidence_id": "b", "event_type": "learning_model_observation", "source_ref": json.dumps({"sha256": "acct-b"})},
    ]

    accounting = DefaultExperimentStageRunner(runtime, Protocol())._learning_accounting("run", ["a", "b"], {})

    assert accounting["accountingComplete"] is False
    assert "costMicrounits" not in accounting


@pytest.mark.parametrize("source_ref", [json.dumps({"sha256": "missing"}), "not-json"])
def test_learning_accounting_rejects_missing_or_malformed_wanted_artifact(source_ref):
    runtime = Runtime()
    runtime.controller.store.evidence = [
        {"evidence_id": "wanted", "event_type": "learning_model_observation", "source_ref": source_ref},
    ]

    accounting = DefaultExperimentStageRunner(runtime, Protocol())._learning_accounting("run", ["wanted"], {})

    assert accounting["accountingComplete"] is False


def test_learning_accounting_accepts_all_measured_receipts_without_nominal_cost():
    runtime = Runtime()
    runtime.controller.store.artifacts.update({
        "acct-a": {"wallSeconds": 1.0, "costMicrounits": 12, "economicCostStatus": "measured"},
        "acct-b": {"wallSeconds": 1.0, "costMicrounits": 8, "economicCostStatus": "measured"},
    })
    runtime.controller.store.evidence = [
        {"evidence_id": "a", "event_type": "learning_model_observation", "source_ref": json.dumps({"sha256": "acct-a"})},
        {"evidence_id": "b", "event_type": "learning_model_observation", "source_ref": json.dumps({"sha256": "acct-b"})},
    ]

    accounting = DefaultExperimentStageRunner(runtime, Protocol())._learning_accounting("run", ["a", "b"], {})

    assert accounting == {"wallSeconds": 2.0, "accountingComplete": True, "costMicrounits": 20}


def test_learning_accounting_uses_complete_nominal_coverage_for_mixed_cost_receipts():
    runtime = Runtime()
    runtime.controller.store.artifacts.update({
        "acct-a": {"wallSeconds": 1.0, "costMicrounits": 12, "nominalCostUsd": 0.001},
        "acct-b": {"wallSeconds": 1.0, "nominalCostUsd": 0.002},
    })
    runtime.controller.store.evidence = [
        {"evidence_id": "a", "event_type": "learning_model_observation", "source_ref": json.dumps({"sha256": "acct-a"})},
        {"evidence_id": "b", "event_type": "learning_model_observation", "source_ref": json.dumps({"sha256": "acct-b"})},
    ]

    accounting = DefaultExperimentStageRunner(runtime, Protocol())._learning_accounting("run", ["a", "b"], {})

    assert accounting == {"wallSeconds": 2.0, "accountingComplete": True, "nominalCostUsd": 0.003}


def test_learning_accounting_does_not_use_aggregate_fallback_for_missing_requested_receipt():
    runtime = Runtime()

    accounting = DefaultExperimentStageRunner(runtime, Protocol())._learning_accounting(
        "run", ["missing"], {"wallSeconds": 1.0, "costMicrounits": 12}
    )

    assert accounting["accountingComplete"] is False
    assert "costMicrounits" not in accounting


def test_observation_cost_requires_complete_nominal_coverage():
    from adaptive_agent.app import LearningRuntimeError, _effective_observation_cost

    assert _effective_observation_cost({"costMicrounits": 17, "nominalCostUsd": 0.000021}) == 17
    complete = {"nominalCostUsd": 0.000021, "nominalCostStatus": "complete", "nominalCostCoverage": {"knownReceipts": 1, "totalReceipts": 1}}
    assert _effective_observation_cost(complete) == 21
    with pytest.raises(LearningRuntimeError, match="complete economic or nominal cost"):
        _effective_observation_cost({"nominalCostUsd": 0.000021, "nominalCostStatus": "partial", "nominalCostCoverage": {"knownReceipts": 1, "totalReceipts": 2}})


def test_recovery_uses_receipt_refs_and_nominal_proxy():
    import json
    from adaptive_agent.evaluation import build_environment_packages

    class RecoveryStore:
        def __init__(self):
            self.artifacts = {
                "model-art": {"responseId": "response", "arm": "L", "seed": 17, "bundleHash": "base", "versionRefs": {"planner": "core"}, "accountingRef": "acct"},
                "outcome-art": {"passed": True, "reliable": True, "safetyViolations": 0, "fixtureResetOk": True},
                "acct": {"nominalCostUsd": 0.000012, "durationSeconds": 1.0},
            }
            self.rows = {
                "model-ref": {"evidence_id": "model-ref", "run_id": "run", "event_type": "model_response", "source_ref": json.dumps({"sha256": "model-art"})},
                "outcome-ref": {"evidence_id": "outcome-ref", "run_id": "run", "event_type": "trusted_outcome", "source_ref": json.dumps({"sha256": "outcome-art"})},
            }

        def get_run(self, run_id):
            return {"task_id": "task", "environment_id": "known-a", "status": "succeeded", "run_json": json.dumps({"arm": "L", "seed": 17, "bundleHash": "base"})} if run_id == "run" else None

        def get_task(self, task_id):
            return {"partition": "validation"} if task_id == "task" else None

        def get_evidence(self, evidence_id):
            return self.rows.get(evidence_id)

        def get_artifact(self, ref):
            return self.artifacts[ref]

        def get_outcome_by_run_id(self, _run_id):
            return {}

    store = RecoveryStore()
    runtime = SimpleNamespace(controller=SimpleNamespace(store=store, get_active_bundle=lambda: Bundle("base")), core_planner_hash="core", image_digest="image", packages={"known-a": build_environment_packages()["finance"]})
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    receipt = {"stage": "validation", "cellKey": "validation:0", "runIds": ["run"], "evidenceRefs": ["model-ref"], "outcomeRefs": ["outcome-ref"], "costBasis": "nominal_budget_proxy", "billingStatus": "unknown"}

    observations = runner.recover_evaluation_observations(receipt)
    assert observations[0].cost_microunits == 12
    with pytest.raises(ExperimentRuntimeError, match="evidence references"):
        runner.recover_evaluation_observations({**receipt, "evidenceRefs": ["missing"]})


def test_recovery_treats_audited_wrong_answer_as_complete_but_not_infrastructure_failure():
    import json
    from adaptive_agent.evaluation import build_environment_packages

    class RecoveryStore:
        def __init__(self):
            self.infrastructure_failure = None
            self.artifacts = {
                "model-art": {"responseId": "response", "arm": "B0", "seed": 17, "bundleHash": "base", "versionRefs": {"planner": "core"}, "accountingRef": "acct"},
                "outcome-art": {"passed": False, "reliable": True, "safetyViolations": 0, "fixtureResetOk": True},
                "acct": {"costMicrounits": 12, "durationSeconds": 1.0},
            }
            self.rows = {
                "model-ref": {"evidence_id": "model-ref", "run_id": "run", "event_type": "model_response", "source_ref": json.dumps({"sha256": "model-art"})},
                "outcome-ref": {"evidence_id": "outcome-ref", "run_id": "run", "event_type": "trusted_outcome", "source_ref": json.dumps({"sha256": "outcome-art"})},
            }

        def get_run(self, run_id):
            return {"task_id": "task", "environment_id": "known-a", "status": "failed", "run_json": json.dumps({"arm": "B0", "seed": 17, "bundleHash": "base"})} if run_id == "run" else None

        def get_task(self, task_id):
            return {"partition": "validation"} if task_id == "task" else None

        def get_evidence(self, evidence_id):
            return self.rows.get(evidence_id)

        def get_artifact(self, ref):
            return self.artifacts[ref]

        def get_outcome_by_run_id(self, _run_id):
            metadata = {"reliable": True, "safetyViolations": 0, "fixtureResetOk": True}
            if self.infrastructure_failure is not None:
                metadata["infrastructureFailure"] = self.infrastructure_failure
            return {"metadata_json": json.dumps(metadata)}

    store = RecoveryStore()
    runtime = SimpleNamespace(controller=SimpleNamespace(store=store, get_active_bundle=lambda: Bundle("base")), core_planner_hash="core", image_digest="image", packages={"known-a": build_environment_packages()["finance"]})
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    receipt = {"stage": "validation", "cellKey": "validation:0", "runIds": ["run"], "evidenceRefs": ["model-ref"], "outcomeRefs": ["outcome-ref"]}

    audited_failure = runner.recover_evaluation_observations(receipt)[0]
    assert audited_failure.passed is False
    assert audited_failure.status == "complete"
    assert audited_failure.infrastructure_failure is None

    store.infrastructure_failure = "provider_timeout"
    infrastructure_failure = runner.recover_evaluation_observations(receipt)[0]
    assert infrastructure_failure.status == "failed"
    assert infrastructure_failure.infrastructure_failure == "provider_timeout"


def test_learning_receipt_preserves_nominal_proxy_and_wall_time(monkeypatch):
    runtime = Runtime()
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    monkeypatch.setattr(runner, "_learning_observation_usage", lambda _run_id, **_: ({"inputTokens": 4, "outputTokens": 3, "totalTokens": 7}, []))

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
    monkeypatch.setattr(runner, "_learning_observation_usage", lambda _run_id, **_: ({"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, []))

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


# ------------------------------------------------------------------
# Multi-run development evidence selection (learning aggregation seam)


class _MultiRunStore:
    """Fake store carrying per-run bindings for learning-selection tests."""

    def __init__(self):
        self.runs: dict[str, dict] = {}
        self.tasks: dict[str, dict] = {}
        self.outcomes: dict[str, dict] = {}
        self.persisted: list[dict] = []

    def add_run(self, run_id, env, status="succeeded", task=None, outcome=None, partition="development"):
        task_id = task or f"task-{run_id}"
        self.runs[run_id] = {"status": status, "environment_id": env, "task_id": task_id}
        self.tasks[task_id] = {"partition": partition, "goal": "g", "version": "1"}
        self.outcomes[run_id] = outcome if outcome is not None else {"passed": status == "succeeded"}
        return run_id

    def get_run(self, run_id):
        return self.runs.get(run_id)

    def get_task(self, task_id):
        return self.tasks.get(task_id)

    def get_outcome_by_run_id(self, run_id):
        return self.outcomes.get(run_id)

    def list_evidence(self, _run_id):
        return []

    def put_artifact(self, data):
        self.persisted.append(data)
        return SimpleNamespace(model_dump=lambda **_kw: {"id": "sel", "version": "1", "sha256": "selsha"})


def _multi_runtime():
    runtime = Runtime()
    runtime.controller.store = _MultiRunStore()
    captured = []
    runtime.launch_learning = lambda payload: (captured.append(payload) or {
        "candidate": {"candidateId": "cand", "candidateBundleHash": "cand", "baseBundleHash": "base"},
    })
    runtime.captured = captured
    return runtime


def _learning_runner(runtime, monkeypatch):
    runner = DefaultExperimentStageRunner(runtime, Protocol())
    monkeypatch.setattr(runner, "_learning_observation_usage", lambda _run_id, **_: ({"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, ["learning-ref"]))
    monkeypatch.setattr(experiment_runtime, "_load_bundle", lambda _runtime, content_hash: Bundle(content_hash))
    return runner


def _training_context(*env_runlists):
    """Build receipts: (env, [run_ids]) tuples -> context with training results."""
    results = {}
    for env, runs in env_runlists:
        for index, run_id in enumerate(runs):
            results[f"{env}-development-{index}"] = {"environmentId": env, "runIds": [run_id], "status": "complete"}
    return {"stage": "learning", "results": {"training": results}}


def test_learning_selects_bounded_multi_run_set_across_domains(monkeypatch):
    runtime = _multi_runtime()
    store = runtime.controller.store
    for env, runs in (("known-a", ["a1", "a2"]), ("known-b", ["b1"]), ("known-c", ["c1", "c2"])):
        for run_id in runs:
            store.add_run(run_id, env)
    store.add_run("b1", "known-b", status="failed", outcome={"passed": False})  # failed attempt eligible
    runner = _learning_runner(runtime, monkeypatch)
    context = _training_context(("known-a", ["a1", "a2"]), ("known-b", ["b1"]), ("known-c", ["c1", "c2"]))

    receipt = runner(cell_key="learning-0", context=context)

    # Later runs beyond the first receipt's run affect the learner context.
    payload = runtime.captured[-1]
    assert payload.run_ids == ["a1", "b1", "c1", "a2", "c2"]  # stratified: one per env first
    assert "b1" in payload.run_ids  # the failed attempt is a source
    assert receipt["sourceRunIds"] == payload.run_ids
    assert receipt["ignoredRunIds"] == []
    assert receipt["learningSelection"]["sourceRunIds"] == payload.run_ids
    assert receipt["learningSelection"]["excludedEnvironment"] is None
    assert store.persisted[-1]["sourceRunIds"] == payload.run_ids


def test_learning_cap_declares_ignored_runs(monkeypatch):
    runtime = _multi_runtime()
    store = runtime.controller.store
    runs = []
    for env in ("known-a", "known-b", "known-c"):
        for index in range(4):
            run_id = f"{env}{index}"
            store.add_run(run_id, env)
            runs.append(run_id)
    runner = _learning_runner(runtime, monkeypatch)
    context = _training_context(*[(env, [f"{env}{i}" for i in range(4)]) for env in ("known-a", "known-b", "known-c")])

    receipt = runner(cell_key="learning-0", context=context)

    selected = receipt["sourceRunIds"]
    assert len(selected) == runner._DEFAULT_LEARNING_SOURCE_CAP  # 8
    assert set(receipt["ignoredRunIds"]) == set(runs) - set(selected)
    assert receipt["learningSelection"]["maxRuns"] == runner._DEFAULT_LEARNING_SOURCE_CAP


def test_learning_selection_is_deterministic(monkeypatch):
    runtime = _multi_runtime()
    store = runtime.controller.store
    for env, runs in (("known-a", ["a2", "a1"]), ("known-b", ["b1"])):
        for run_id in runs:
            store.add_run(run_id, env)
    runner = _learning_runner(runtime, monkeypatch)
    context = _training_context(("known-a", ["a1", "a2"]), ("known-b", ["b1"]))

    first = runner(cell_key="learning-0", context=context)["sourceRunIds"]
    second = runner(cell_key="learning-0", context=context)["sourceRunIds"]
    assert first == second == ["a1", "b1", "a2"]


def test_leave_out_candidate_excludes_domain_and_declares_ignored(monkeypatch):
    runtime = _multi_runtime()
    store = runtime.controller.store
    for env, runs in (("known-a", ["a1", "a2"]), ("known-b", ["b1"]), ("known-c", ["c1"])):
        for run_id in runs:
            store.add_run(run_id, env)
    runner = _learning_runner(runtime, monkeypatch)
    context = _training_context(("known-a", ["a1", "a2"]), ("known-b", ["b1"]), ("known-c", ["c1"]))

    _bundle, receipt = runner._candidate_for_excluded_environment(context, "known-a")

    selected = receipt["sourceRunIds"]
    assert "a1" not in selected and "a2" not in selected
    assert set(selected) == {"b1", "c1"}
    assert set(receipt["ignoredRunIds"]) == {"a1", "a2"}
    assert receipt["learningSelection"]["excludedEnvironment"] == "known-a"


@pytest.mark.parametrize("mutate", ["nonterminal", "no_outcome", "non_dev_partition", "env_mismatch"])
def test_learning_rejects_invalid_sources_fail_closed(monkeypatch, mutate):
    runtime = _multi_runtime()
    store = runtime.controller.store
    store.add_run("good", "known-a")
    store.add_run("bad", "known-b")
    if mutate == "nonterminal":
        store.runs["bad"]["status"] = "running"
    elif mutate == "no_outcome":
        store.outcomes.pop("bad")
    elif mutate == "non_dev_partition":
        store.tasks["task-bad"]["partition"] = "validation"
    elif mutate == "env_mismatch":
        store.runs["bad"]["environment_id"] = "known-c"
    runner = _learning_runner(runtime, monkeypatch)
    context = _training_context(("known-a", ["good"]), ("known-b", ["bad"]))

    with pytest.raises(ExperimentRuntimeError):
        runner(cell_key="learning-0", context=context)
    assert runtime.captured == []  # nothing dispatched


def test_learning_admission_binds_the_selected_source_set(monkeypatch):
    runtime = _multi_runtime()
    store = runtime.controller.store
    for env, runs in (("known-a", ["a1"]), ("known-b", ["b1"]), ("known-c", ["c1"])):
        for run_id in runs:
            store.add_run(run_id, env)
    runner = _learning_runner(runtime, monkeypatch)
    context = _training_context(("known-a", ["a1"]), ("known-b", ["b1"]), ("known-c", ["c1"]))
    admissions = []
    context["admitSubcall"] = lambda key, **_e: admissions.append(key) or {"admissionId": key, "status": "reserved", "reused": False, "dispatchAllowed": True}
    context["recordSubcall"] = lambda admission_id, *, result=None, error=None: None

    receipt = runner(cell_key="learning-0", context=context)

    assert admissions == [f"learning:learning-0:{','.join(receipt['sourceRunIds'])}"]
    assert receipt["sourceRunIds"] == ["a1", "b1", "c1"]


def test_learning_request_supports_optional_run_ids():
    from adaptive_agent.api import LearningRequest

    single = LearningRequest(runId="run-1")
    assert single.run_id == "run-1" and single.run_ids == []
    multi = LearningRequest(runIds=["run-1", "run-2"])
    assert multi.run_id is None and multi.run_ids == ["run-1", "run-2"]
    both = LearningRequest(runId="run-0", runIds=["run-1"])
    assert both.run_id == "run-0" and both.run_ids == ["run-1"]


def test_propose_completed_runs_binds_declared_source_set(monkeypatch):
    """The multi-run runtime seam validates every source and exposes exactly
    the declared (environment, run) set to the retriever."""
    from adaptive_agent.learning_runtime import LearningRuntime
    from adaptive_agent.retrieval import SourceRecord, SourceKind, content_hash

    store = _MultiRunStore()
    store.add_run("a1", "known-a")
    store.add_run("b1", "known-b", status="failed", outcome={"passed": False})
    store.add_run("c1", "known-c")

    def fake_source(run_id, env):
        return SourceRecord(
            source_id=f"src-{run_id}", kind=SourceKind.LIVE_EVIDENCE, content="ev", content_hash=content_hash("ev"),
            environment_id=env, run_id=run_id, partition="development", visibility="learner", trust_class="broker",
        )

    class FakeAdapter:
        def records_from_raw(self, records, *, environment_id, run_id):
            return [fake_source(run_id, environment_id) for _record in records]

    calls = {}

    def fake_propose(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(candidate_payload={"predictedEffect": "x", "supportingEvidenceIds": []}, authoritative_candidate={})

    runtime = LearningRuntime(
        store=store,
        manager=None,
        source_adapter=FakeAdapter(),
        candidate_adapter=None,
        service=SimpleNamespace(retriever=None, model_runner=SimpleNamespace(client=object()), propose=fake_propose),
        token_budget=100,
        wall_seconds=10,
    )
    monkeypatch.setattr(LearningRuntime, "_materialize_run_records", lambda self, *, environment_id, run_id, public_documents=(): [{"kind": "task_state", "sourceId": f"o:{run_id}", "runId": run_id, "environmentId": environment_id}])

    proposal = runtime.propose_completed_runs(["c1", "a1", "b1"], primary_run_id="b1", goal="g")

    # Later runs entered the learner's retrieval scope: the declared set spans
    # all three environments, ordered with the primary first.
    assert calls["run_id"] == "b1"
    assert calls["source_runs"] == frozenset({("known-a", "a1"), ("known-b", "b1"), ("known-c", "c1")})
    assert calls["feedback"]["sourceRunIds"] == ["b1", "a1", "c1"]
    assert calls["feedback"]["status"] == "failed"  # primary is the failed run
    sources = runtime.service.retriever.provider.list_sources()
    assert {s.run_id for s in sources} == {"a1", "b1", "c1"}

    # Invalid sources fail closed.
    store.runs["c1"]["status"] = "running"
    with pytest.raises(Exception):
        runtime.propose_completed_runs(["a1", "c1"], primary_run_id="a1", goal="g")


def test_learning_selection_prefers_failed_runs_without_dropping_env_coverage(monkeypatch):
    """A failed run sorts after successes alphabetically; selection must still
    include it deterministically while covering every eligible environment."""
    runtime = _multi_runtime()
    store = runtime.controller.store
    # known-a: only the success would be picked by id order within cap-free
    # first-pass; the failure ("zz") must still enter via the swap pass when
    # it would otherwise be cut by the bound.
    store.add_run("a0", "known-a")
    store.add_run("zz", "known-a", status="failed", outcome={"passed": False})
    store.add_run("b1", "known-b")
    store.add_run("c1", "known-c")
    runner = _learning_runner(runtime, monkeypatch)
    context = _training_context(
        ("known-a", ["a0", "zz"]),
        ("known-b", ["b1"]),
        ("known-c", ["c1"]),
    )

    receipt = runner(cell_key="learning-0", context=context)

    # Failed-first ordering: "zz" precedes "a0" inside known-a despite id sort.
    assert receipt["sourceRunIds"] == ["zz", "b1", "c1", "a0"]
    assert receipt["ignoredRunIds"] == []


def test_learning_selection_swaps_in_failure_under_tight_cap(monkeypatch):
    runtime = _multi_runtime()
    store = runtime.controller.store
    store.add_run("a0", "known-a")
    store.add_run("zz", "known-a", status="failed", outcome={"passed": False})
    store.add_run("b1", "known-b")
    store.add_run("c1", "known-c")
    runner = _learning_runner(runtime, monkeypatch)
    # Pin a smaller bound through the frozen inputs.
    runner.inputs = {**runner.inputs, "learningSelection": {"maxRuns": 3}}
    context = _training_context(
        ("known-a", ["a0", "zz"]),
        ("known-b", ["b1"]),
        ("known-c", ["c1"]),
    )

    receipt = runner(cell_key="learning-0", context=context)

    # Cap 3 with 3 envs: failed-first ordering already picks "zz" for known-a.
    assert receipt["sourceRunIds"] == ["zz", "b1", "c1"]
    assert receipt["ignoredRunIds"] == ["a0"]


def test_learning_rejects_oversized_source_set(monkeypatch):
    from adaptive_agent.learning_runtime import LearningRuntime, LEARNING_SOURCE_RUN_CAP

    store = _MultiRunStore()
    for index in range(LEARNING_SOURCE_RUN_CAP + 1):
        store.add_run(f"r{index}", "known-a")

    runtime = LearningRuntime(
        store=store,
        manager=None,
        source_adapter=None,
        candidate_adapter=None,
        service=SimpleNamespace(retriever=None, model_runner=SimpleNamespace(client=object()), propose=lambda **kw: None),
        token_budget=100,
        wall_seconds=10,
    )

    with pytest.raises(Exception, match="exceeds the frozen bound"):
        runtime.propose_completed_runs([f"r{i}" for i in range(LEARNING_SOURCE_RUN_CAP + 1)], primary_run_id="r0", goal="g")

    with pytest.raises(Exception, match="malformed"):
        runtime.propose_completed_runs(["r0", 7], primary_run_id="r0", goal="g")
