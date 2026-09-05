"""Adaptive Agent bounded execution and control-plane package."""

from .planner import LunaPlanner, PlannerError, PlannerLimits, PlannerResult
from .prime_runtime import (
    AdapterError,
    ArtifactRef,
    Capability,
    CapabilityBroker,
    CapabilitySet,
    ExecutionMode,
    ExecutionResult,
    ModelObservation,
    PrimeRuntimeAdapter,
    PrimeRuntimeConfig,
    SecurityViolation,
)

_API_EXPORTS = {"ControlPlane", "app", "create_app", "make_authenticated_model_runner"}


def __getattr__(name: str):
    if name in _API_EXPORTS:
        from .api import ControlPlane, app, create_app, make_authenticated_model_runner
        return {"ControlPlane": ControlPlane, "app": app, "create_app": create_app, "make_authenticated_model_runner": make_authenticated_model_runner}[name]
    raise AttributeError(name)


__all__ = [
    "AdapterError", "ArtifactRef", "Capability", "CapabilityBroker", "CapabilitySet",
    "ControlPlane", "ExecutionMode", "ExecutionResult", "LunaPlanner", "ModelObservation",
    "PlannerError", "PlannerLimits", "PlannerResult", "PrimeRuntimeAdapter",
    "PrimeRuntimeConfig", "SecurityViolation", "app", "create_app", "make_authenticated_model_runner",
]
