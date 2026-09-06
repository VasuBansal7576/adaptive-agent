"""Bounded orchestration for the complete independent experiment lifecycle.

The helper owns ordering and admission checks only.  Runtime execution,
learning, candidate construction, fixture reset, evaluation, and accounting are
injected callbacks owned by the trusted runtime/evaluator.  In particular, an
integer workload plan is never accepted as evidence that a transfer or
adaptation experiment ran.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class ExperimentLifecycleError(ValueError):
    """Raised when a lifecycle receipt violates the frozen experiment contract."""


class InfrastructureFailure(RuntimeError):
    """A declared retryable infrastructure failure from an injected callback."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class LifecycleCallbacks(Protocol):
    """Trusted runtime callbacks coordinated by :func:`run_experiment_lifecycle`.

    Each callback returns a durable receipt mapping.  ``attempt`` is supplied so
    the callback can bind retries to distinct attempt identities while keeping
    the task/seed/bundle identity unchanged.
    """

    def bootstrap_development(self, policy: "LifecyclePolicy", *, attempt: int) -> Mapping[str, Any]: ...

    def run_development(self, bootstrap: Mapping[str, Any], policy: "LifecyclePolicy", *, attempt: int) -> Mapping[str, Any]: ...

    def generate_candidate(self, development: Mapping[str, Any], bootstrap: Mapping[str, Any], *, attempt: int) -> Mapping[str, Any]: ...

    def run_transfer(self, candidate: Mapping[str, Any], environment_id: str, *, reset: bool, exposed: bool, attempt: int) -> Mapping[str, Any]: ...

    def run_adaptation(self, candidate: Mapping[str, Any], environment_id: str, support_task_ids: Sequence[str], *, reset: bool, attempt: int) -> Mapping[str, Any]: ...

    def run_registered_safety(self, candidate: Mapping[str, Any], case_ids: Sequence[str], *, attempt: int) -> Mapping[str, Any]: ...

    def run_validation(self, candidate: Mapping[str, Any], *, attempt: int) -> Mapping[str, Any]: ...

    def run_final(self, candidate: Mapping[str, Any], *, attempt: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class LifecyclePolicy:
    """Frozen lifecycle inputs, including explicit non-panel workload terms."""

    development_task_ids: tuple[str, ...]
    reserved_transfer_environment: str
    adaptation_support_task_ids: tuple[str, ...]
    safety_case_ids: tuple[str, ...] = ("EVAL-003", "EVAL-004", "EVAL-005")
    required_pins: tuple[str, ...] = (
        "corePlannerHash",
        "imageDigest",
        "modelProfile",
        "protocolHash",
        "analysisCodeHash",
        "activeBundleHash",
    )
    infrastructure_retries: int = 0
    provider_disclosure: str = "No paid calls are made by the lifecycle helper; callbacks own provider access."

    def __post_init__(self) -> None:
        if not self.development_task_ids:
            raise ExperimentLifecycleError("development bootstrap requires at least one task")
        if not self.reserved_transfer_environment:
            raise ExperimentLifecycleError("a reserved transfer environment is required")
        if not self.adaptation_support_task_ids:
            raise ExperimentLifecycleError("adaptation requires an explicit support task set")
        if not self.safety_case_ids or len(set(self.safety_case_ids)) != len(self.safety_case_ids):
            raise ExperimentLifecycleError("safety cases must be a non-empty unique registration")
        if self.infrastructure_retries < 0:
            raise ExperimentLifecycleError("infrastructure retry count cannot be negative")
        if not self.required_pins or len(set(self.required_pins)) != len(self.required_pins):
            raise ExperimentLifecycleError("required pins must be non-empty and unique")


@dataclass(frozen=True)
class LifecycleResult:
    status: str
    phases: Mapping[str, Mapping[str, Any]]
    retry_attempts: tuple[Mapping[str, Any], ...] = ()
    provider_disclosure: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "phases": {name: dict(receipt) for name, receipt in self.phases.items()},
            "retryAttempts": [dict(item) for item in self.retry_attempts],
            "providerDisclosure": self.provider_disclosure,
        }


def _as_mapping(value: Any, phase: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentLifecycleError(f"{phase} callback did not return a receipt object")
    return value


def _strings(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item for item in value):
        raise ExperimentLifecycleError(f"{name} must be a list of non-empty strings")
    result = tuple(value)
    if not allow_empty and not result:
        raise ExperimentLifecycleError(f"{name} must not be empty")
    return result


def _receipt(value: Any, phase: str, *, partition: str | None = None, sealed: bool | None = None) -> Mapping[str, Any]:
    receipt = _as_mapping(value, phase)
    if receipt.get("executed") is not True:
        raise ExperimentLifecycleError(f"{phase} requires an executed durable receipt")
    if receipt.get("workloadOnly") is True or receipt.get("synthetic") is True:
        raise ExperimentLifecycleError(f"{phase} cannot be represented by arithmetic or synthetic evidence")
    _strings(receipt.get("runIds"), f"{phase}.runIds")
    if receipt.get("status") not in {"complete", "valid", "passed"}:
        raise ExperimentLifecycleError(f"{phase} receipt is not complete")
    if partition is not None and receipt.get("partition") != partition:
        raise ExperimentLifecycleError(f"{phase} receipt is not a {partition} partition")
    if sealed is not None and receipt.get("sealed") is not sealed:
        raise ExperimentLifecycleError(f"{phase} sealed flag is invalid")
    return receipt


def _retry(
    phase: str,
    callback: Any,
    max_retries: int,
    retry_log: list[Mapping[str, Any]],
) -> Mapping[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            return _as_mapping(callback(attempt), phase)
        except InfrastructureFailure as exc:
            retry_log.append({"phase": phase, "attempt": attempt, "retryable": exc.retryable, "error": str(exc)})
            if not exc.retryable or attempt >= max_retries:
                raise ExperimentLifecycleError(f"{phase} infrastructure failure exhausted declared retries: {exc}") from exc
    raise AssertionError("unreachable")


def run_experiment_lifecycle(callbacks: LifecycleCallbacks, policy: LifecyclePolicy) -> LifecycleResult:
    """Run the complete bounded sequence and return only durable phase receipts.

    The call order is deliberately fixed:

    ``clean bootstrap -> development learning -> candidate -> transfer ->
    adaptation -> registered safety -> validation -> final``.

    Validation and final callbacks are never invoked if an earlier phase fails,
    and the final callback receives no validation outcome or held-out payload.
    """
    retry_log: list[Mapping[str, Any]] = []
    phases: dict[str, Mapping[str, Any]] = {}
    retry_count = policy.infrastructure_retries

    bootstrap = _retry(
        "bootstrap",
        lambda attempt: callbacks.bootstrap_development(policy, attempt=attempt),
        retry_count,
        retry_log,
    )
    if bootstrap.get("clean") is not True:
        raise ExperimentLifecycleError("development bootstrap is not clean")
    pins = bootstrap.get("pins")
    if not isinstance(pins, Mapping):
        raise ExperimentLifecycleError("development bootstrap has no pinned inventory")
    for name in policy.required_pins:
        value = pins.get(name, pins.get(name[0].lower() + name[1:]))
        if not isinstance(value, str) or not value or value in {"unset", "un pinned", "image-unpinned", "core-planner-unset"}:
            raise ExperimentLifecycleError(f"development bootstrap pin {name!r} is missing or unpinned")
    development_exposure = _strings(bootstrap.get("developmentTaskIds"), "bootstrap.developmentTaskIds")
    if set(development_exposure) != set(policy.development_task_ids):
        raise ExperimentLifecycleError("development bootstrap task set does not match the frozen policy")
    if policy.reserved_transfer_environment in set(_strings(bootstrap.get("exposedEnvironments", []), "bootstrap.exposedEnvironments", allow_empty=True)):
        raise ExperimentLifecycleError("reserved transfer environment was exposed during development bootstrap")
    phases["bootstrap"] = bootstrap

    development = _receipt(
        _retry("development", lambda attempt: callbacks.run_development(bootstrap, policy, attempt=attempt), retry_count, retry_log),
        "development",
        partition="development",
    )
    learning = development.get("learning")
    if not isinstance(learning, Mapping) or learning.get("actual") is not True:
        raise ExperimentLifecycleError("development receipt lacks actual learning evidence")
    if set(_strings(development.get("taskIds"), "development.taskIds")) != set(policy.development_task_ids):
        raise ExperimentLifecycleError("development receipt task set does not match the frozen policy")
    if development.get("heldoutAccess") is True or policy.reserved_transfer_environment in set(_strings(development.get("exposedEnvironments", []), "development.exposedEnvironments", allow_empty=True)):
        raise ExperimentLifecycleError("held-out transfer environment leaked during development")
    phases["development"] = development

    candidate = _as_mapping(
        _retry("candidate", lambda attempt: callbacks.generate_candidate(development, bootstrap, attempt=attempt), retry_count, retry_log),
        "candidate",
    )
    if candidate.get("actual") is not True or not isinstance(candidate.get("candidateId"), str) or not candidate["candidateId"]:
        raise ExperimentLifecycleError("candidate generation lacks an actual durable candidate")
    if candidate.get("baseBundleHash") != pins.get("activeBundleHash"):
        raise ExperimentLifecycleError("candidate base bundle is not pinned to the development bootstrap")
    source_runs = set(_strings(candidate.get("sourceRunIds"), "candidate.sourceRunIds"))
    development_runs = set(_strings(development.get("runIds"), "development.runIds"))
    if not source_runs.issubset(development_runs):
        raise ExperimentLifecycleError("candidate cites runs outside actual development evidence")
    phases["candidate"] = candidate

    transfer = _receipt(
        _retry(
            "transfer",
            lambda attempt: callbacks.run_transfer(candidate, policy.reserved_transfer_environment, reset=True, exposed=False, attempt=attempt),
            retry_count,
            retry_log,
        ),
        "transfer",
    )
    if transfer.get("environmentId") != policy.reserved_transfer_environment or transfer.get("resetBefore") is not True or transfer.get("exposedToLearning") is not False:
        raise ExperimentLifecycleError("transfer receipt violates exposure/reset discipline")
    development_runs = set(_strings(development.get("runIds"), "development.runIds"))
    transfer_runs = set(_strings(transfer.get("runIds"), "transfer.runIds"))
    if development_runs & transfer_runs:
        raise ExperimentLifecycleError("transfer reuses a development run")
    phases["transfer"] = transfer

    adaptation = _receipt(
        _retry(
            "adaptation",
            lambda attempt: callbacks.run_adaptation(candidate, policy.reserved_transfer_environment, policy.adaptation_support_task_ids, reset=True, attempt=attempt),
            retry_count,
            retry_log,
        ),
        "adaptation",
    )
    if adaptation.get("environmentId") != policy.reserved_transfer_environment or adaptation.get("resetBefore") is not True:
        raise ExperimentLifecycleError("adaptation receipt violates reset discipline")
    if set(_strings(adaptation.get("supportTaskIds"), "adaptation.supportTaskIds")) != set(policy.adaptation_support_task_ids):
        raise ExperimentLifecycleError("adaptation support set does not match the frozen policy")
    if adaptation.get("queryExposedToLearning") is not False:
        raise ExperimentLifecycleError("adaptation query tasks were exposed to learning")
    adaptation_runs = set(_strings(adaptation.get("runIds"), "adaptation.runIds"))
    if (development_runs | transfer_runs) & adaptation_runs:
        raise ExperimentLifecycleError("adaptation reuses a prior experiment run")
    phases["adaptation"] = adaptation

    safety = _as_mapping(
        _retry("safety", lambda attempt: callbacks.run_registered_safety(candidate, policy.safety_case_ids, attempt=attempt), retry_count, retry_log),
        "safety",
    )
    registered = set(_strings(safety.get("registeredCaseIds"), "safety.registeredCaseIds"))
    passed = set(_strings(safety.get("passedCaseIds"), "safety.passedCaseIds", allow_empty=True))
    if registered != set(policy.safety_case_ids) or passed != registered:
        raise ExperimentLifecycleError("registered safety suite is incomplete or failing")
    phases["safety"] = safety

    validation = _receipt(
        _retry("validation", lambda attempt: callbacks.run_validation(candidate, attempt=attempt), retry_count, retry_log),
        "validation",
        partition="validation",
    )
    if validation.get("heldOut") is not True or validation.get("exposureLeak") is True:
        raise ExperimentLifecycleError("validation receipt is not independent held-out evidence")
    phases["validation"] = validation

    final = _receipt(
        _retry("final", lambda attempt: callbacks.run_final(candidate, attempt=attempt), retry_count, retry_log),
        "final",
        partition="final",
        sealed=True,
    )
    if final.get("exposureLeak") is True or final.get("selectedCandidateAfterObservation") is True:
        raise ExperimentLifecycleError("final receipt violates sealed evaluation discipline")
    phases["final"] = final

    return LifecycleResult("complete", phases, tuple(retry_log), policy.provider_disclosure)


def lifecycle_json(result: LifecycleResult) -> str:
    """Canonical serialization for durable operator evidence."""
    return json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":"))


__all__ = [
    "ExperimentLifecycleError",
    "InfrastructureFailure",
    "LifecycleCallbacks",
    "LifecyclePolicy",
    "LifecycleResult",
    "lifecycle_json",
    "run_experiment_lifecycle",
]
