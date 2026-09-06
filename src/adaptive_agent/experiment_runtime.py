"""Concrete default runner for the durable experiment lifecycle.

This module is intentionally an adapter around the integrated DurableRuntime
and evaluator APIs.  It does not manufacture observations or accept workload
counts as evidence.  Every task cell is executed through
``runtime.execute_evaluation_task`` and every learning cell goes through the
runtime's trusted learning entry point.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping


class ExperimentRuntimeError(RuntimeError):
    """Raised when a lifecycle cell cannot be backed by durable evidence."""


_CELL_INDEX = re.compile(r"^(?:validation|final):([0-9]+)$")
_LEAVE_OUT = re.compile(r"^leave-out:([^:]+)$")
_ADAPT = re.compile(r"^adapt:([^:]+)$")


@dataclass(frozen=True)
class _ExecutionConfig:
    protocol: Any
    arm: str
    seed: int
    bundle_hash: str
    attempt: int


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentRuntimeError(f"{label} is not an object")
    return value


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentRuntimeError(f"{label} is missing")
    return value


def _protocol_inputs(protocol: Any) -> Mapping[str, Any]:
    frozen = protocol.start_candidate_generation()
    inputs = _mapping(getattr(frozen, "inputs", None), "frozen protocol inputs")
    for key in ("provider", "modelProfile", "corePlannerHash", "imageDigest", "runBudget"):
        if key not in inputs:
            raise ExperimentRuntimeError(f"frozen protocol pin {key!r} is missing")
    return inputs


def _bundle_hash(bundle: Any) -> str:
    return _required_string(getattr(bundle, "content_hash", None), "bundle content hash")


def _load_bundle(runtime: Any, content_hash: str) -> Any:
    row = runtime.controller.store.get_bundle_by_hash(content_hash)
    if row is None:
        raise ExperimentRuntimeError(f"bundle {content_hash!r} is not durable")
    try:
        from adaptive_agent.models import SkillBundle

        return SkillBundle.model_validate(json.loads(row["bundle_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExperimentRuntimeError("durable bundle is malformed") from exc


def _package(runtime: Any, environment_id: str) -> Any:
    package = runtime.packages.get(environment_id)
    if package is None:
        raise ExperimentRuntimeError(f"environment package {environment_id!r} is unavailable")
    return package


def _tasks(package: Any, partition: str) -> tuple[Any, ...]:
    getter = getattr(package, "tasks_for_partition", None)
    if not callable(getter):
        raise ExperimentRuntimeError("environment package has no partition task accessor")
    values = tuple(getter(partition))
    if not values:
        raise ExperimentRuntimeError(f"environment package has no {partition} tasks")
    return values


def _task_environment(task: Any) -> str:
    ref = getattr(task, "environment_ref", None)
    value = getattr(ref, "id", None) or getattr(task, "environment_id", None)
    return _required_string(value, "task environment")


def _task_id(task: Any) -> str:
    return _required_string(getattr(task, "task_id", None), "task id")


def _active_bundle(runtime: Any) -> Any:
    bundle = runtime.controller.get_active_bundle()
    if bundle is None:
        raise ExperimentRuntimeError("no active base bundle is available")
    return bundle


def _pins(runtime: Any, protocol: Any, inputs: Mapping[str, Any], base_hash: str) -> dict[str, str]:
    core = _required_string(getattr(runtime, "core_planner_hash", None), "runtime core planner hash")
    image = _required_string(getattr(runtime, "image_digest", None), "runtime image digest")
    if core != inputs["corePlannerHash"] or image != inputs["imageDigest"]:
        raise ExperimentRuntimeError("runtime pins do not match the frozen protocol")
    if inputs["provider"] != "openai-codex" or inputs["modelProfile"] != "openai-codex/gpt-5.6-luna":
        raise ExperimentRuntimeError("frozen experiment is not pinned to the authenticated Prime model")
    frozen = protocol.start_candidate_generation()
    pins = {
        "corePlannerHash": core,
        "imageDigest": image,
        "modelProfile": inputs["modelProfile"],
        "protocolHash": _required_string(getattr(frozen, "protocol_hash", None), "protocol hash"),
        "baseBundleHash": base_hash,
        "activeBundleHash": base_hash,
    }
    analysis_hash = inputs.get("analysisCodeHash")
    if isinstance(analysis_hash, str) and analysis_hash:
        pins["analysisCodeHash"] = analysis_hash
    return pins


def _accounting(runtime: Any, observation: Any) -> tuple[dict[str, int], int, float, int | None, str | None]:
    ref = getattr(observation, "accounting_ref", None)
    if not isinstance(ref, str) or not ref:
        raise ExperimentRuntimeError("observation lacks a durable accounting reference")
    value = runtime.controller.store.get_artifact(ref)
    accounting = _mapping(value, "accounting artifact")
    usage = accounting.get("aggregateUsage", accounting.get("usage"))
    usage = _mapping(usage, "accounting usage")
    fields = {key: usage.get(key) for key in ("inputTokens", "outputTokens", "totalTokens")}
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in fields.values()):
        raise ExperimentRuntimeError("accounting usage is incomplete")
    if fields["totalTokens"] != fields["inputTokens"] + fields["outputTokens"]:
        raise ExperimentRuntimeError("accounting usage total is inconsistent")
    calls = accounting.get("toolCalls", accounting.get("toolCallCount", 0))
    if not isinstance(calls, int) or isinstance(calls, bool) or calls < 0:
        raise ExperimentRuntimeError("accounting tool calls are malformed")
    wall = accounting.get("durationSeconds", accounting.get("inferenceDurationSeconds", 0))
    if not isinstance(wall, (int, float)) or isinstance(wall, bool) or wall < 0:
        raise ExperimentRuntimeError("accounting duration is malformed")
    cost = accounting.get("costMicrounits")
    if cost is not None and (not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0):
        raise ExperimentRuntimeError("accounting cost is malformed")
    economic = accounting.get("economicCost")
    economic_status = economic.get("status") if isinstance(economic, Mapping) else None
    return {key: int(value) for key, value in fields.items()}, int(calls), float(wall), int(cost) if cost is not None else None, economic_status


def _observation_receipt(runtime: Any, stage: str, cell_key: str, observations: list[Any], pins: Mapping[str, str], *, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not observations:
        raise ExperimentRuntimeError(f"{stage}/{cell_key} produced no observations")
    usage = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
    tool_calls = 0
    wall = 0.0
    cost = 0
    cost_seen = False
    economic_unknown = False
    run_ids: list[str] = []
    evidence_refs: list[str] = []
    outcome_refs: list[str] = []
    for observation in observations:
        run_id = _required_string(getattr(observation, "run_id", None), "observation run id")
        evidence = _required_string(getattr(observation, "evidence_ref", None), "observation evidence ref")
        outcome = _required_string(getattr(observation, "outcome_ref", None), "observation outcome ref")
        if run_id in run_ids:
            raise ExperimentRuntimeError("duplicate observation run id")
        run_ids.append(run_id)
        evidence_refs.append(evidence)
        outcome_refs.append(outcome)
        current_usage, current_calls, current_wall, current_cost, economic_status = _accounting(runtime, observation)
        for key in usage:
            usage[key] += current_usage[key]
        tool_calls += current_calls
        wall += current_wall
        if current_cost is not None:
            cost += current_cost
            cost_seen = True
        economic_unknown = economic_unknown or economic_status == "unknown" or current_cost is None
    receipt: dict[str, Any] = {
        "stage": stage,
        "cellKey": cell_key,
        "status": "complete",
        "executed": True,
        "actual": True,
        "usage": usage,
        "toolCalls": tool_calls,
        "wallSeconds": wall,
        "runIds": run_ids,
        "evidenceRefs": evidence_refs,
        "outcomeRefs": outcome_refs,
        "pins": dict(pins),
    }
    if cost_seen and not economic_unknown:
        receipt["costMicrounits"] = cost
    else:
        receipt["economicCostStatus"] = "unknown"
    if extra:
        receipt.update(dict(extra))
    return receipt


class DefaultExperimentStageRunner:
    """Runtime-owned stage implementation used by the production evaluator."""

    def __init__(self, runtime: Any, protocol: Any) -> None:
        self.runtime = runtime
        self.protocol = protocol
        self.inputs = _protocol_inputs(protocol)
        self.base_bundle = _active_bundle(runtime)
        self.base_hash = _bundle_hash(self.base_bundle)
        self.pins = _pins(runtime, protocol, self.inputs, self.base_hash)
        self._candidate_id: str | None = None
        self._candidate_hash: str | None = None
        self._rotation_candidates: dict[str, tuple[str, str]] = {}

    def __call__(self, *, cell_key: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
        stage = _required_string(context.get("stage"), "lifecycle stage")
        attempt = context.get("attempt", 0)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
            raise ExperimentRuntimeError("lifecycle attempt is malformed")
        sealed = _required_string(getattr(self.protocol, "sealed_environment", None), "sealed environment")
        if stage != "final" and sealed in cell_key:
            raise ExperimentRuntimeError("sealed environment is not available before final")
        if stage == "bootstrap":
            return self._bootstrap(cell_key)
        if stage == "training":
            return self._training(cell_key, attempt)
        if stage == "learning":
            return self._learning(cell_key, context)
        if stage == "transfer":
            return self._transfer(cell_key, context, attempt)
        if stage == "adaptation":
            return self._adaptation(cell_key, context, attempt)
        if stage == "safety":
            return self._safety(cell_key)
        if stage in {"validation", "final"}:
            return self._panel_cell(stage, cell_key, context, attempt)
        raise ExperimentRuntimeError(f"unsupported lifecycle stage {stage!r}")

    def _bootstrap(self, cell_key: str) -> Mapping[str, Any]:
        known = tuple(getattr(self.protocol, "known_environments", ()))
        for environment_id in known:
            _package(self.runtime, environment_id)
        provenance = getattr(self.runtime, "establish_clean_experiment", None)
        if not callable(provenance):
            raise ExperimentRuntimeError("runtime lacks clean experiment provenance seam")
        clean = _mapping(provenance(self.protocol), "clean experiment provenance")
        if clean.get("clean") is not True or not isinstance(clean.get("provenanceRef"), str) or not clean["provenanceRef"]:
            raise ExperimentRuntimeError("experiment provenance is not clean")
        return {
            "stage": "bootstrap",
            "cellKey": cell_key,
            "status": "complete",
            "executed": True,
            "clean": True,
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "toolCalls": 0,
            "wallSeconds": 0,
            "costMicrounits": 0,
            "pins": dict(self.pins),
            "developmentEnvironments": list(known),
            "developmentTaskIds": [
                _task_id(task)
                for environment_id in known
                for task in _tasks(_package(self.runtime, environment_id), "development")
            ],
            "exposedEnvironments": list(known),
            "sealedEnvironment": self.protocol.sealed_environment,
            "provenanceRef": clean["provenanceRef"],
        }

    def _task_for_cell(self, cell_key: str, partition: str, environment_id: str | None = None, index: int = 0) -> Any:
        if environment_id is None:
            raise ExperimentRuntimeError("task environment is required")
        tasks = _tasks(_package(self.runtime, environment_id), partition)
        if index < 0 or index >= len(tasks):
            raise ExperimentRuntimeError(f"task index {index} is outside {environment_id}/{partition}")
        return tasks[index]

    def _execute(self, task: Any, arm: str, seed: int, bundle: Any, attempt: int) -> Any:
        config = _ExecutionConfig(self.protocol, arm, seed, _bundle_hash(bundle), attempt)
        observation = self.runtime.execute_evaluation_task(task, config, bundle)
        if getattr(observation, "run_id", None) is None or getattr(observation, "outcome_ref", None) is None:
            raise ExperimentRuntimeError("runtime execution returned an unbound observation")
        if getattr(observation, "response_id", None) is None:
            raise ExperimentRuntimeError("runtime execution lacks a trusted model response")
        verifier = getattr(self.runtime, "verify_evaluation_observation", None)
        if not callable(verifier):
            raise ExperimentRuntimeError("runtime lacks strict evaluation evidence verifier")
        if verifier(observation, config, task) is not True:
            raise ExperimentRuntimeError("evaluation evidence failed the strict verifier")
        model_provenance = getattr(observation, "model_provenance", None)
        model_provenance = getattr(model_provenance, "value", model_provenance)
        if model_provenance != "real_model":
            raise ExperimentRuntimeError("runtime execution is not backed by a real model receipt")
        get_evidence = getattr(self.runtime.controller.store, "get_evidence", None)
        if not callable(get_evidence):
            raise ExperimentRuntimeError("store lacks trusted evidence lookup")
        model_row = get_evidence(observation.evidence_ref)
        outcome_row = get_evidence(observation.outcome_ref)
        if not isinstance(model_row, Mapping) or model_row.get("event_type") != "model_response":
            raise ExperimentRuntimeError("observation model evidence is not canonical")
        if not isinstance(outcome_row, Mapping) or outcome_row.get("event_type") != "trusted_outcome":
            raise ExperimentRuntimeError("observation outcome evidence is not evaluator-owned")
        return observation

    def _training(self, cell_key: str, attempt: int) -> Mapping[str, Any]:
        task = next(
            (
                task
                for environment_id in tuple(self.protocol.known_environments)
                for task in _tasks(_package(self.runtime, environment_id), "development")
                if _task_id(task) == cell_key
            ),
            None,
        )
        if task is None:
            raise ExperimentRuntimeError(f"development task {cell_key!r} is not registered")
        observation = self._execute(task, "B0", int(self.protocol.seeds[0]), self.base_bundle, attempt)
        return _observation_receipt(
            self.runtime,
            "training",
            cell_key,
            [observation],
            self.pins,
            extra={"partition": "development", "taskIds": [_task_id(task)], "environmentId": _task_environment(task)},
        )

    def _training_results(self, context: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        results = context.get("results", {})
        training = results.get("training", {}) if isinstance(results, Mapping) else {}
        if not isinstance(training, Mapping):
            raise ExperimentRuntimeError("training receipts are unavailable")
        values = [value for value in training.values() if isinstance(value, Mapping)]
        if not values:
            raise ExperimentRuntimeError("learning requires completed development receipts")
        return values

    def _learning(self, cell_key: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
        receipts = self._training_results(context)
        candidate_source = next(iter(receipts), None)
        if candidate_source is None:
            raise ExperimentRuntimeError("learning has no development source receipt")
        run_id = _required_string(candidate_source.get("runIds", [None])[0], "development source run")
        stored = self.runtime.controller.store.get_run(run_id)
        if not isinstance(stored, Mapping) or stored.get("status") not in {"succeeded", "failed", "cancelled", "timed_out", "outcome_unknown"}:
            raise ExperimentRuntimeError("candidate generation requires a completed development run")
        if self.runtime.controller.store.get_outcome_by_run_id(run_id) is None:
            raise ExperimentRuntimeError("candidate generation requires a trusted development outcome")
        return self._candidate_from_run(run_id, cell_key, bind_primary=True, context=context)

    def _candidate_from_run(self, run_id: str, cell_key: str, *, bind_primary: bool, prior_learning_refs: set[str] | None = None, context: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        stored = self.runtime.controller.store.get_run(run_id)
        if not isinstance(stored, Mapping) or stored.get("status") not in {"succeeded", "failed", "cancelled", "timed_out", "outcome_unknown"}:
            raise ExperimentRuntimeError("candidate generation requires a completed run")
        if self.runtime.controller.store.get_outcome_by_run_id(run_id) is None:
            raise ExperimentRuntimeError("candidate generation requires a trusted outcome")
        baseline_refs = prior_learning_refs if prior_learning_refs is not None else self._learning_observation_ids(run_id)
        admission = self._admit_subcall(context, f"learning:{cell_key}:{run_id}")
        if admission is not None and admission.get("reused") is True:
            if admission.get("status") != "complete" or not isinstance(admission.get("result"), Mapping):
                raise ExperimentRuntimeError("nested subcall checkpoint requires durable result recovery")
            recovered = _mapping(admission["result"], "recovered nested learning receipt")
            candidate_id = _required_string(recovered.get("candidateId"), "recovered candidate id")
            candidate_hash = _required_string(recovered.get("candidateBundleHash"), "recovered candidate bundle hash")
            if bind_primary:
                self._candidate_id, self._candidate_hash = candidate_id, candidate_hash
            else:
                self._rotation_candidates[cell_key] = (candidate_id, candidate_hash)
            return recovered
        try:
            result = self.runtime.launch_learning(SimpleNamespace(run_id=run_id))
            receipt = self._learning_receipt(cell_key, run_id, result, bind_primary=bind_primary, prior_learning_refs=baseline_refs)
        except Exception as exc:
            self._record_subcall(admission, error=str(exc))
            raise
        self._record_subcall(admission, result=receipt)
        return receipt

    def _admit_subcall(self, context: Mapping[str, Any] | None, subcall_key: str) -> Mapping[str, Any] | None:
        if not isinstance(context, Mapping):
            return None
        admit = context.get("admitSubcall")
        if not callable(admit):
            return None
        budget = self.inputs.get("runBudget")
        budget = budget if isinstance(budget, Mapping) else {}
        admission = _mapping(admit(
            subcall_key,
            estimated_input_tokens=int(budget.get("modelTokens", 0) or 0),
            estimated_output_tokens=int(budget.get("modelTokens", 0) or 0),
            estimated_tool_calls=int(budget.get("toolCalls", 0) or 0),
            estimated_wall_seconds=float(budget.get("wallTimeSeconds", 0) or 0),
            estimated_cost_microunits=int(budget.get("costMicrounits", 0) or 0),
        ), "nested subcall admission")
        recorder = context.get("recordSubcall")
        if callable(recorder):
            return {**admission, "record": recorder}
        return admission

    @staticmethod
    def _record_subcall(admission: Mapping[str, Any] | None, *, result: Mapping[str, Any] | None = None, error: str | None = None) -> None:
        if admission is None:
            return
        if admission.get("reused") is True:
            if admission.get("status") != "complete":
                raise ExperimentRuntimeError("nested subcall admission is already in progress")
            return
        recorder = admission.get("record")
        if callable(recorder):
            recorder(admission.get("admissionId"), result=result, error=error)

    def _learning_receipt(self, cell_key: str, run_id: str, result: Any, *, bind_primary: bool, prior_learning_refs: set[str] | None = None) -> Mapping[str, Any]:
        result = _mapping(result, "learning result")
        candidate = _mapping(result.get("candidate"), "learning candidate")
        candidate_id = _required_string(candidate.get("candidateId", candidate.get("candidate_id")), "candidate id")
        candidate_hash = _required_string(candidate.get("candidateBundleHash", candidate.get("candidate_bundle_hash")), "candidate bundle hash")
        if candidate.get("baseBundleHash", candidate.get("base_bundle_hash")) != self.base_hash:
            raise ExperimentRuntimeError("candidate is not based on the pinned development bundle")
        if bind_primary:
            self._candidate_id, self._candidate_hash = candidate_id, candidate_hash
        else:
            self._rotation_candidates[cell_key] = (candidate_id, candidate_hash)
        learning_usage, learning_refs = self._learning_observation_usage(run_id, prior_refs=prior_learning_refs)
        return {
            "stage": "learning",
            "cellKey": cell_key,
            "status": "complete",
            "usage": learning_usage,
            "toolCalls": 0,
            "wallSeconds": 0,
            "economicCostStatus": "unknown",
            "pins": dict(self.pins),
            "candidateId": candidate_id,
            "candidateBundleHash": candidate_hash,
            "sourceRunIds": [run_id],
            "modelObservationRefs": learning_refs,
        }

    def _candidate_for_excluded_environment(self, context: Mapping[str, Any], excluded: str) -> tuple[Any, Mapping[str, Any]]:
        """Generate a leave-one-environment-out candidate from eligible runs."""
        receipts = self._training_results(context)
        eligible = [receipt for receipt in receipts if receipt.get("environmentId") != excluded]
        if not eligible:
            raise ExperimentRuntimeError("transfer training set does not prove leave-one-environment-out exclusion")
        source = eligible[0]
        run_ids = source.get("runIds")
        if not isinstance(run_ids, list) or not run_ids or not isinstance(run_ids[0], str):
            raise ExperimentRuntimeError("transfer source receipt lacks a durable run")
        run_id = run_ids[0]
        receipt = self._candidate_from_run(run_id, f"leave-out:{excluded}", bind_primary=False, context=context)
        return _load_bundle(self.runtime, receipt["candidateBundleHash"]), receipt

    def _learning_observation_usage(self, run_id: str, *, prior_refs: set[str] | None = None) -> tuple[dict[str, int], list[str]]:
        rows = self.runtime.controller.store.list_evidence(run_id)
        total = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}
        refs: list[str] = []
        for row in rows:
            if row.get("event_type") != "learning_model_observation":
                continue
            evidence_id = row.get("evidence_id")
            if prior_refs is not None and evidence_id in prior_refs:
                continue
            source = row.get("source_ref")
            if not isinstance(source, str):
                raise ExperimentRuntimeError("learning observation has no durable source reference")
            try:
                source_data = json.loads(source)
                payload = self.runtime.controller.store.get_artifact(source_data["sha256"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ExperimentRuntimeError("learning observation source is malformed") from exc
            usage = _mapping(payload, "learning observation").get("usage")
            usage = _mapping(usage, "learning observation usage")
            fields = {key: usage.get(key) for key in total}
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in fields.values()):
                raise ExperimentRuntimeError("learning observation usage is incomplete")
            if fields["totalTokens"] != fields["inputTokens"] + fields["outputTokens"]:
                raise ExperimentRuntimeError("learning observation usage total is inconsistent")
            for key in total:
                total[key] += fields[key]
            refs.append(_required_string(evidence_id, "learning observation evidence ref"))
        if not refs:
            raise ExperimentRuntimeError("candidate generation lacks an authenticated learning observation")
        return total, refs

    def _learning_observation_ids(self, run_id: str) -> set[str]:
        rows = self.runtime.controller.store.list_evidence(run_id)
        return {
            evidence_id
            for row in rows
            if row.get("event_type") == "learning_model_observation"
            for evidence_id in (row.get("evidence_id"),)
            if isinstance(evidence_id, str) and evidence_id
        }

    def _candidate(self, context: Mapping[str, Any]) -> Any:
        candidate_hash = self._candidate_hash
        learning = context.get("results", {}).get("learning", {}) if isinstance(context.get("results", {}), Mapping) else {}
        if candidate_hash is None and isinstance(learning, Mapping):
            for receipt in learning.values():
                if isinstance(receipt, Mapping) and isinstance(receipt.get("candidateBundleHash"), str):
                    candidate_hash = receipt["candidateBundleHash"]
                    break
        if candidate_hash is None:
            raise ExperimentRuntimeError("candidate bundle is unavailable")
        return _load_bundle(self.runtime, candidate_hash)

    def _transfer(self, cell_key: str, context: Mapping[str, Any], attempt: int) -> Mapping[str, Any]:
        match = _LEAVE_OUT.fullmatch(cell_key)
        if match is None:
            raise ExperimentRuntimeError(f"invalid transfer cell {cell_key!r}")
        environment_id = match.group(1)
        if environment_id not in tuple(self.protocol.known_environments):
            raise ExperimentRuntimeError("transfer environment is not a known environment")
        receipts = self._training_results(context)
        eligible = [receipt for receipt in receipts if receipt.get("environmentId") != environment_id]
        if not eligible:
            raise ExperimentRuntimeError("transfer training set does not prove leave-one-environment-out exclusion")
        candidate, candidate_receipt = self._candidate_for_excluded_environment(context, environment_id)
        validation_task = self._task_for_cell(cell_key, "validation", environment_id, 0)
        observation = self._execute(validation_task, "L", int(self.protocol.seeds[0]), candidate, attempt)
        return _observation_receipt(
            self.runtime,
            "transfer",
            cell_key,
            [observation],
            self.pins,
            extra={"environmentId": environment_id, "partition": "validation", "resetBefore": True, "trainingExcludedEnvironment": environment_id, "trainingSourceRunIds": list(candidate_receipt["sourceRunIds"]), "candidateId": candidate_receipt["candidateId"], "candidateBundleHash": candidate_receipt["candidateBundleHash"], "learningReceipt": dict(candidate_receipt)},
        )
        return self._merge_receipts(receipt, candidate_receipt)

    def _adaptation(self, cell_key: str, context: Mapping[str, Any], attempt: int) -> Mapping[str, Any]:
        match = _ADAPT.fullmatch(cell_key)
        if match is None:
            raise ExperimentRuntimeError(f"invalid adaptation cell {cell_key!r}")
        environment_id = match.group(1)
        candidate = self._candidate(context)
        support = self._task_for_cell(cell_key, "development", environment_id, 0)
        query = self._task_for_cell(cell_key, "validation", environment_id, 0)
        support_observation = self._execute(support, "L", int(self.protocol.seeds[0]), candidate, attempt)
        support_learning = self._candidate_from_run(
            getattr(support_observation, "run_id"),
            cell_key,
            bind_primary=False,
            context=context,
        )
        adapted_candidate = _load_bundle(self.runtime, support_learning["candidateBundleHash"])
        query_observation = self._execute(query, "L", int(self.protocol.seeds[1]), adapted_candidate, attempt)
        receipt = _observation_receipt(
            self.runtime,
            "adaptation",
            cell_key,
            [support_observation, query_observation],
            self.pins,
            extra={"environmentId": environment_id, "resetBefore": True, "supportTaskIds": [_task_id(support)], "queryTaskIds": [_task_id(query)], "queryExposedToLearning": False},
        )
        receipt["supportRunIds"] = [getattr(support_observation, "run_id")]
        receipt["queryRunIds"] = [getattr(query_observation, "run_id")]
        receipt["adaptedCandidateId"] = support_learning["candidateId"]
        receipt["adaptedCandidateBundleHash"] = support_learning["candidateBundleHash"]
        receipt["learningReceipt"] = dict(support_learning)
        return self._merge_receipts(receipt, support_learning)

    @staticmethod
    def _merge_receipts(receipt: Mapping[str, Any], nested: Mapping[str, Any]) -> dict[str, Any]:
        """Include authenticated nested-learning accounting exactly once."""
        merged = dict(receipt)
        outer_usage = _mapping(merged.get("usage"), "outer receipt usage")
        nested_usage = _mapping(nested.get("usage"), "nested receipt usage")
        merged["usage"] = {
            key: int(outer_usage[key]) + int(nested_usage[key])
            for key in ("inputTokens", "outputTokens", "totalTokens")
        }
        merged["toolCalls"] = int(merged.get("toolCalls", 0)) + int(nested.get("toolCalls", 0))
        merged["wallSeconds"] = float(merged.get("wallSeconds", 0)) + float(nested.get("wallSeconds", 0))
        outer_cost = merged.get("costMicrounits")
        nested_cost = nested.get("costMicrounits")
        if isinstance(outer_cost, (int, float)) and not isinstance(outer_cost, bool) and isinstance(nested_cost, (int, float)) and not isinstance(nested_cost, bool):
            merged["costMicrounits"] = outer_cost + nested_cost
            merged.pop("economicCostStatus", None)
        else:
            merged.pop("costMicrounits", None)
            merged["economicCostStatus"] = "unknown"
        merged["nestedRunIds"] = list(nested.get("runIds", []))
        merged["nestedEvidenceRefs"] = list(nested.get("modelObservationRefs", []))
        return merged

    def _safety(self, cell_key: str) -> Mapping[str, Any]:
        execute_probe = getattr(self.runtime.controller, "execute_probe", None)
        if not callable(execute_probe):
            raise ExperimentRuntimeError("controller has no registered safety probe executor")
        started = time.monotonic()
        result = execute_probe(cell_key)
        wall_seconds = max(time.monotonic() - started, 1e-9)
        if not isinstance(result, Mapping):
            converter = getattr(result, "model_dump", None) or getattr(result, "to_dict", None)
            if callable(converter):
                result = converter()
        result = _mapping(result, "safety probe result")
        if result.get("caseId") != cell_key or result.get("passed") is not True:
            raise ExperimentRuntimeError(f"registered safety probe {cell_key!r} did not pass")
        raw_tool_calls = result.get("toolCalls")
        tool_calls = raw_tool_calls if isinstance(raw_tool_calls, int) and not isinstance(raw_tool_calls, bool) and raw_tool_calls >= 0 else len(result.get("obligations", ()))
        return {
            "stage": "safety",
            "cellKey": cell_key,
            "status": "complete",
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "toolCalls": tool_calls,
            "wallSeconds": wall_seconds,
            "costMicrounits": 0,
            "pins": dict(self.pins),
            "safetyCaseId": cell_key,
            "safetyEvidence": result.get("evidence", result.get("detail")),
            "probeDurationSeconds": result.get("wallSeconds", wall_seconds),
        }

    def _panel_cell(self, stage: str, cell_key: str, context: Mapping[str, Any], attempt: int) -> Mapping[str, Any]:
        match = _CELL_INDEX.fullmatch(cell_key)
        if match is None:
            raise ExperimentRuntimeError(f"invalid {stage} cell {cell_key!r}")
        index = int(match.group(1))
        known = tuple(self.protocol.known_environments)
        panel_environments = known if stage == "validation" else (*known, self.protocol.sealed_environment)
        tasks_per_environment = int(self.protocol.tasks_per_environment)
        seeds = tuple(self.protocol.seeds)
        arms = ("B0", "L") if stage == "validation" else ("B0", "L", "A")
        expected = len(panel_environments) * tasks_per_environment * len(seeds) * len(arms)
        if index >= expected:
            raise ExperimentRuntimeError(f"{stage} index is outside the frozen panel")
        arm = arms[index % len(arms)]
        seed = int(seeds[(index // len(arms)) % len(seeds)])
        task_index = (index // (len(arms) * len(seeds))) % tasks_per_environment
        environment_index = index // (tasks_per_environment * len(seeds) * len(arms))
        environment_id = panel_environments[environment_index]
        if stage == "final" and environment_id == self.protocol.sealed_environment:
            # The package remains evaluator-owned until this exact final cell.
            pass
        if stage != "final" and environment_id == self.protocol.sealed_environment:
            raise ExperimentRuntimeError("sealed environment reached before primary final")
        bundle = self.base_bundle if arm == "B0" else None
        if arm == "A":
            ablation_hash = context.get("ablationBundleHash")
            if not isinstance(ablation_hash, str):
                bundles = getattr(self.runtime, "_evaluation_arm_bundles", None)
                if isinstance(bundles, Mapping):
                    ablation = bundles.get("A")
                    if ablation is None:
                        ablation = bundles.get("ablation")
                    if ablation is not None:
                        if isinstance(ablation, str):
                            ablation_hash = ablation
                        else:
                            bundle = ablation
                            ablation_hash = _bundle_hash(ablation)
            if not isinstance(ablation_hash, str):
                raise ExperimentRuntimeError("final ablation bundle hash is not pinned")
            if bundle is None or _bundle_hash(bundle) != ablation_hash:
                bundle = _load_bundle(self.runtime, ablation_hash)
        elif arm == "L":
            bundle = self._candidate(context)
        if bundle is None:
            raise ExperimentRuntimeError(f"evaluation arm {arm!r} has no bundle")
        task = self._task_for_cell(cell_key, "validation" if stage == "validation" else "final", environment_id, task_index)
        observation = self._execute(task, arm, seed, bundle, attempt)
        return _observation_receipt(
            self.runtime,
            stage,
            cell_key,
            [observation],
            self.pins,
            extra={"partition": stage, "environmentId": environment_id, "taskIds": [_task_id(task)], "arm": arm, "seed": seed, "sealed": stage == "final"},
        )


def build_default_experiment_stage_runner(runtime: Any, protocol: Any) -> DefaultExperimentStageRunner:
    """Construct the authenticated-runtime-backed default stage runner."""
    return DefaultExperimentStageRunner(runtime, protocol)


__all__ = ["DefaultExperimentStageRunner", "ExperimentRuntimeError", "build_default_experiment_stage_runner"]
