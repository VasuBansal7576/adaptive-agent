from __future__ import annotations

import pytest

from adaptive_agent.experiment_lifecycle import (
    ExperimentLifecycleError,
    InfrastructureFailure,
    LifecyclePolicy,
    lifecycle_json,
    run_experiment_lifecycle,
)


def _policy(**overrides):
    values = {
        "development_task_ids": ("known-dev-1", "known-dev-2"),
        "reserved_transfer_environment": "reserved-env",
        "adaptation_support_task_ids": ("reserved-support-1",),
        "infrastructure_retries": 1,
    }
    values.update(overrides)
    return LifecyclePolicy(**values)


class Callbacks:
    def __init__(self):
        self.calls = []

    def bootstrap_development(self, policy, *, attempt):
        self.calls.append(("bootstrap", attempt))
        return {
            "clean": True,
            "pins": {
                "corePlannerHash": "core",
                "imageDigest": "image",
                "modelProfile": "model",
                "protocolHash": "protocol",
                "analysisCodeHash": "analysis",
                "activeBundleHash": "base",
            },
            "developmentTaskIds": list(policy.development_task_ids),
            "exposedEnvironments": [],
            "executed": True,
            "runIds": ["bootstrap-1"],
            "status": "complete",
        }

    def run_development(self, bootstrap, policy, *, attempt):
        self.calls.append(("development", attempt))
        return {
            "executed": True,
            "runIds": ["dev-1", "dev-2"],
            "taskIds": list(policy.development_task_ids),
            "exposedEnvironments": [],
            "learning": {"actual": True, "provider": "local-test"},
            "partition": "development",
            "status": "complete",
        }

    def generate_candidate(self, development, bootstrap, *, attempt):
        self.calls.append(("candidate", attempt))
        return {"actual": True, "candidateId": "candidate-1", "baseBundleHash": "base", "sourceRunIds": ["dev-1", "dev-2"]}

    def run_transfer(self, candidate, environment_id, *, reset, exposed, attempt):
        self.calls.append(("transfer", attempt, reset, exposed))
        return {"executed": True, "runIds": ["transfer-1"], "environmentId": environment_id, "resetBefore": reset, "exposedToLearning": exposed, "status": "complete"}

    def run_adaptation(self, candidate, environment_id, support_task_ids, *, reset, attempt):
        self.calls.append(("adaptation", attempt, reset))
        return {"executed": True, "runIds": ["adapt-1"], "environmentId": environment_id, "supportTaskIds": list(support_task_ids), "resetBefore": reset, "queryExposedToLearning": False, "status": "complete"}

    def run_registered_safety(self, candidate, case_ids, *, attempt):
        self.calls.append(("safety", attempt))
        return {"registeredCaseIds": list(case_ids), "passedCaseIds": list(case_ids)}

    def run_validation(self, candidate, *, attempt):
        self.calls.append(("validation", attempt))
        return {"executed": True, "runIds": ["validation-1"], "partition": "validation", "heldOut": True, "exposureLeak": False, "status": "complete"}

    def run_final(self, candidate, *, attempt):
        self.calls.append(("final", attempt))
        return {"executed": True, "runIds": ["final-1"], "partition": "final", "sealed": True, "exposureLeak": False, "selectedCandidateAfterObservation": False, "status": "complete"}


def test_complete_lifecycle_orders_real_receipts_and_discloses_provider_boundary():
    callbacks = Callbacks()
    result = run_experiment_lifecycle(callbacks, _policy())

    assert result.status == "complete"
    assert [item[0] for item in callbacks.calls] == ["bootstrap", "development", "candidate", "transfer", "adaptation", "safety", "validation", "final"]
    assert result.phases["candidate"]["candidateId"] == "candidate-1"
    assert result.retry_attempts == ()
    assert "No paid calls" in lifecycle_json(result)


def test_transfer_workload_count_is_not_execution_evidence():
    callbacks = Callbacks()
    original = callbacks.run_transfer
    callbacks.run_transfer = lambda *args, **kwargs: {"transferRuns": 100}  # type: ignore[method-assign]

    with pytest.raises(ExperimentLifecycleError, match="executed durable receipt"):
        run_experiment_lifecycle(callbacks, _policy())
    callbacks.run_transfer = original  # keep the fixture explicit for debuggers


def test_retry_is_bounded_and_only_declared_infrastructure_failures_retry():
    callbacks = Callbacks()
    original = callbacks.run_development
    attempts = []

    def flaky(bootstrap, policy, *, attempt):
        attempts.append(attempt)
        if attempt == 0:
            raise InfrastructureFailure("transient runtime unavailable")
        return original(bootstrap, policy, attempt=attempt)

    callbacks.run_development = flaky  # type: ignore[method-assign]
    result = run_experiment_lifecycle(callbacks, _policy(infrastructure_retries=1))
    assert attempts == [0, 1]
    assert [item["phase"] for item in result.retry_attempts] == ["development"]


def test_heldout_leak_fails_before_validation_or_final():
    callbacks = Callbacks()
    original = callbacks.run_development

    def leaked(bootstrap, policy, *, attempt):
        receipt = dict(original(bootstrap, policy, attempt=attempt))
        receipt["exposedEnvironments"] = ["reserved-env"]
        return receipt

    callbacks.run_development = leaked  # type: ignore[method-assign]
    with pytest.raises(ExperimentLifecycleError, match="leaked"):
        run_experiment_lifecycle(callbacks, _policy())
    assert not any(item[0] in {"validation", "final"} for item in callbacks.calls)
