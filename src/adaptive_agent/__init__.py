"""Adaptive Agent bounded execution and durable control-plane package."""

from .planner import LunaInvocation, LunaPlanner, PlannerError, PlannerLimits, PlannerResult, PrimeCliModelClient, make_luna_model_runner
from .prime_runtime import (
    AdapterError, ArtifactRef as PrimeArtifactRef, Capability as PrimeCapability,
    CapabilityBroker, CapabilitySet, ExecutionMode as PrimeExecutionMode,
    ExecutionResult, ModelObservation, PrimeRuntimeAdapter, PrimeRuntimeConfig,
    SecurityViolation,
)
from .models import (
    ArtifactRef, Budget, CandidateProposal, EnvironmentManifest, EvaluationReport,
    EvidenceRecord, PromotionDecision, RunRequest, TaskInput, ToolRequest, ToolResult,
)

Capability = PrimeCapability

_API_EXPORTS = {"ControlPlane", "app", "create_app", "make_authenticated_model_runner"}


def __getattr__(name: str):
    if name in _API_EXPORTS:
        from .api import ControlPlane, app, create_app, make_authenticated_model_runner
        return {"ControlPlane": ControlPlane, "app": app, "create_app": create_app, "make_authenticated_model_runner": make_authenticated_model_runner}[name]
    raise AttributeError(name)


__all__ = [
    "AdapterError", "ArtifactRef", "PrimeArtifactRef", "PrimeCapability", "CapabilityBroker", "CapabilitySet",
    "Capability",
    "ControlPlane", "PrimeExecutionMode", "ExecutionResult", "LunaPlanner", "PrimeCliModelClient", "ModelObservation",
    "PlannerError", "PlannerLimits", "PlannerResult", "PrimeRuntimeAdapter", "LunaInvocation", "make_luna_model_runner",
    "PrimeRuntimeConfig", "SecurityViolation", "app", "create_app", "make_authenticated_model_runner",
    "Budget", "CandidateProposal", "EnvironmentManifest", "EvaluationReport", "EvidenceRecord", "PromotionDecision",
    "RunRequest", "TaskInput", "ToolRequest", "ToolResult",
]
