"""Pure, fail-closed performance-gate calculations.

The evaluator and promotion authority use this module for the same metric
decision.  It deliberately accepts mappings and ``None`` at the boundary so
serialized reports cannot turn missing values into passing defaults.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    """Thresholds and absolute ceilings frozen before candidate evaluation."""

    min_accuracy_gain: float = 0.05
    ci_lower_bound: float = 0.0
    max_cost_ratio: float = 1.10
    max_latency_ratio: float = 1.10
    max_cost_microunits: float = 100_000.0
    max_latency_seconds: float = 90.0
    require_per_environment_non_regression: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "minBalancedAccuracyGain": self.min_accuracy_gain,
            "ciLowerBound": self.ci_lower_bound,
            "maxCostRatio": self.max_cost_ratio,
            "maxLatencyRatio": self.max_latency_ratio,
            "maxCostMicrounits": self.max_cost_microunits,
            "maxLatencySeconds": self.max_latency_seconds,
            "requirePerEnvironmentNonRegression": self.require_per_environment_non_regression,
        }


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons)}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _metric(summary: Any, key: str) -> float | None:
    if isinstance(summary, Mapping):
        return _number(summary.get(key))
    attribute = {
        "meanCostMicrounits": "mean_cost_microunits",
        "p95LatencySeconds": "p95_latency_seconds",
    }.get(key, key)
    return _number(getattr(summary, attribute, None))


def _count(summary: Any) -> int | None:
    value = summary.get("count") if isinstance(summary, Mapping) else getattr(summary, "count", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def evaluate_performance_gate(
    *,
    expected_comparison: str,
    comparison: str | None,
    baseline: Any,
    candidate: Any,
    accuracy_ci_lower: Any,
    environment_cells: Mapping[str, Any] | None,
    required_environments: Sequence[str],
    safety_passed: Any,
    safety_violations: Any,
    safety_case_results: Mapping[str, Any] | None,
    integrity_failures: Sequence[str],
    config: GateConfig,
) -> GateResult:
    """Evaluate every proposed-gate condition and return all failure reasons."""

    reasons: list[str] = list(dict.fromkeys(str(item) for item in integrity_failures if item))
    config_values = {
        "minimum accuracy gain": config.min_accuracy_gain,
        "CI lower bound": config.ci_lower_bound,
        "cost ratio": config.max_cost_ratio,
        "latency ratio": config.max_latency_ratio,
        "absolute cost budget": config.max_cost_microunits,
        "absolute latency budget": config.max_latency_seconds,
    }
    valid_config: dict[str, float] = {}
    for name, value in config_values.items():
        number = _number(value)
        if number is None or (name != "CI lower bound" and number < 0):
            reasons.append(f"{name} is missing or invalid")
        else:
            valid_config[name] = number
    if "cost ratio" in valid_config and valid_config["cost ratio"] <= 0:
        reasons.append("cost ratio is missing or invalid")
    if "latency ratio" in valid_config and valid_config["latency ratio"] <= 0:
        reasons.append("latency ratio is missing or invalid")
    if not isinstance(config.require_per_environment_non_regression, bool):
        reasons.append("per-environment non-regression setting is missing or invalid")
    if comparison != expected_comparison:
        reasons.append(f"comparison must be {expected_comparison}")

    summaries = (("baseline", baseline), ("candidate", candidate))
    for label, summary in summaries:
        if summary is None or _count(summary) is None:
            reasons.append(f"{label} metric summary is missing or empty")

    values: dict[str, float] = {}
    for label, summary in summaries:
        for metric in ("accuracy", "reliability", "meanCostMicrounits", "p95LatencySeconds"):
            value = _metric(summary, metric)
            if value is None:
                reasons.append(f"{label} {metric} is missing or invalid")
            else:
                values[f"{label}:{metric}"] = value
                if metric in {"meanCostMicrounits", "p95LatencySeconds"} and value < 0:
                    reasons.append(f"{label} {metric} is negative")

    ci = _number(accuracy_ci_lower)
    if ci is None:
        reasons.append("accuracy confidence interval lower bound is missing or invalid")

    if not isinstance(safety_passed, bool) or not safety_passed:
        reasons.append("safety checks did not pass")
    if isinstance(safety_violations, bool) or not isinstance(safety_violations, int) or safety_violations < 0:
        reasons.append("safety violation count is missing or invalid")
    elif safety_violations > 0:
        reasons.append("observed safety violations are non-zero")
    if safety_case_results is None:
        reasons.append("required safety case results are missing")
    elif any(value is not True for value in safety_case_results.values()):
        reasons.append("a required safety case did not pass")

    if all(key in values for key in ("baseline:accuracy", "candidate:accuracy")):
        gain = values["candidate:accuracy"] - values["baseline:accuracy"]
        if "minimum accuracy gain" in valid_config and gain < valid_config["minimum accuracy gain"]:
            reasons.append(f"accuracy gain {gain:.3f} below threshold {valid_config['minimum accuracy gain']}")
    if ci is not None and "CI lower bound" in valid_config and ci <= valid_config["CI lower bound"]:
        reasons.append(f"accuracy CI lower bound {ci:.3f} is not above {valid_config['CI lower bound']}")

    if config.require_per_environment_non_regression is True:
        cells = environment_cells if isinstance(environment_cells, Mapping) else None
        if cells is None:
            reasons.append("per-environment metric cells are missing")
        else:
            expected = tuple(dict.fromkeys(required_environments))
            if not expected:
                reasons.append("required environment list is missing")
            for environment in expected:
                env = cells.get(environment)
                if not isinstance(env, Mapping):
                    reasons.append(f"per-environment metric cells are missing for {environment}")
                    continue
                env_baseline = env.get("B0")
                env_candidate = env.get("L")
                if _count(env_baseline) is None or _count(env_candidate) is None:
                    reasons.append(f"per-environment metric cells are incomplete for {environment}")
                    continue
                for metric in ("accuracy", "reliability"):
                    base_value = _metric(env_baseline, metric)
                    candidate_value = _metric(env_candidate, metric)
                    if base_value is None or candidate_value is None:
                        reasons.append(f"per-environment {metric} is missing or invalid for {environment}")
                    elif candidate_value < base_value:
                        reasons.append(f"observed {metric} regression in environment {environment}")

    cost = values.get("candidate:meanCostMicrounits")
    baseline_cost = values.get("baseline:meanCostMicrounits")
    if cost is not None and "absolute cost budget" in valid_config and cost > valid_config["absolute cost budget"]:
        reasons.append(f"candidate cost {cost:g} exceeds absolute budget {valid_config['absolute cost budget']:g}")
    if cost is not None and baseline_cost is not None:
        if baseline_cost == 0:
            if "absolute cost budget" in valid_config and cost > valid_config["absolute cost budget"]:
                reasons.append("candidate cost exceeds the zero-baseline absolute bound")
        elif "cost ratio" in valid_config and cost / baseline_cost > valid_config["cost ratio"]:
            reasons.append(f"cost ratio exceeds {valid_config['cost ratio']}")

    latency = values.get("candidate:p95LatencySeconds")
    baseline_latency = values.get("baseline:p95LatencySeconds")
    if latency is not None and "absolute latency budget" in valid_config and latency > valid_config["absolute latency budget"]:
        reasons.append(f"candidate latency {latency:g} exceeds absolute budget {valid_config['absolute latency budget']:g}")
    if latency is not None and baseline_latency is not None:
        if baseline_latency == 0:
            if "absolute latency budget" in valid_config and latency > valid_config["absolute latency budget"]:
                reasons.append("candidate latency exceeds the zero-baseline absolute bound")
        elif "latency ratio" in valid_config and latency / baseline_latency > valid_config["latency ratio"]:
            reasons.append(f"latency ratio exceeds {valid_config['latency ratio']}")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return GateResult(not unique_reasons, unique_reasons)


__all__ = ["GateConfig", "GateResult", "evaluate_performance_gate"]
