"""Bounded, evidence-linked learning proposals.

The learner may suggest a procedure/configuration patch, but this module never
activates it and never treats learner text or predicted effects as outcomes.
The injected candidate sink is the narrow seam to session 2's authoritative
candidate/store implementation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .retrieval import AccessFilteredRetriever, Citation, RetrievalError, RetrievalResult, canonical_json


class LearningError(ValueError):
    pass


class ModelInvocation(Protocol):
    text: str
    provider: str
    model: str
    response_id: str
    usage: Mapping[str, Any]


class AuthenticatedModelRunner(Protocol):
    def __call__(self, *, goal: str, environment: dict[str, Any], emit: Callable[[str, str, str | None], None]) -> ModelInvocation: ...


class CandidateSink(Protocol):
    def create_candidate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ActiveBundleReader(Protocol):
    def __call__(self) -> str: ...


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise LearningError(f"{label} must be a SHA-256 digest")
    return value


def sanitize_feedback(feedback: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only coarse, non-authoritative diagnostics for the model prompt.

    In particular, ``passed``, scores, expected answers, evaluator internals,
    and learner assertions are never forwarded or used as candidate evidence.
    """
    if feedback is None:
        return {}
    if not isinstance(feedback, Mapping):
        raise LearningError("feedback must be an object")
    clean: dict[str, Any] = {}
    if isinstance(feedback.get("status"), str) and feedback["status"] in {"failed", "succeeded", "outcome_unknown", "cancelled"}:
        clean["status"] = feedback["status"]
    if isinstance(feedback.get("failureClass"), str) and len(feedback["failureClass"]) <= 128:
        clean["failureClass"] = feedback["failureClass"]
    diagnostic = feedback.get("diagnostic")
    if isinstance(diagnostic, str) and len(diagnostic) <= 1000:
        lowered = diagnostic.casefold()
        if not any(term in lowered for term in ("expected answer", "hidden answer", "evaluator code", "secret", "credential")):
            clean["diagnostic"] = diagnostic
    observations = feedback.get("observations")
    if isinstance(observations, list):
        clean["observations"] = [item for item in observations if isinstance(item, str) and len(item) <= 300][:8]
    return clean


@dataclass(frozen=True)
class LearningProposal:
    base_bundle_hash: str
    candidate_payload: dict[str, Any]
    bundle_patch: dict[str, Any]
    citations: tuple[Citation, ...]
    model_provenance: dict[str, Any]
    authoritative_candidate: Mapping[str, Any]


_TOP_LEVEL = {"predictedEffect", "editOperations", "changedArtifactHashes", "supportingEvidenceIds", "proposerVersion", "skill", "executionConfigPatch"}
_SKILL_FIELDS = {"procedure", "applicability", "preconditions", "failureHandling"}
_PATCH_FIELDS = {"path", "operation", "value"}
_PRIVILEGED = {"broker", "evaluator", "policy", "store", "credential", "secret", "expectedanswer", "actionsequence", "planner"}
_FIXTURE_ID = re.compile(r"\b(?:INV|TKT|PAY|SMP|SLOT|DEV|FIN|SUP|IT)-[A-Z0-9_-]+\b", re.IGNORECASE)


def _parse_model_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) < 3 or not lines[-1].strip().startswith("```"):
            raise LearningError("model proposal code fence is incomplete")
        value = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LearningError("model proposal must be JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) - _TOP_LEVEL:
        raise LearningError("model proposal contains unknown or privileged fields")
    required = {"predictedEffect", "editOperations", "supportingEvidenceIds", "proposerVersion", "skill"}
    if not required.issubset(parsed):
        raise LearningError("model proposal is missing required fields")
    return parsed


def _canonical_operations(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise LearningError("at least one edit operation is required")
    if len(value) > 3:
        raise LearningError("candidate exceeds the three-artifact bound")
    operations: list[str] = []
    for item in value:
        if isinstance(item, str):
            raise LearningError("edit operations must identify a bounded path and operation")
        if not isinstance(item, Mapping) or set(item) - _PATCH_FIELDS or not _PATCH_FIELDS.issubset(item):
            raise LearningError("malformed edit operation")
        path, operation = item["path"], item["operation"]
        if not isinstance(path, str) or not isinstance(operation, str) or operation not in {"add", "replace", "remove"}:
            raise LearningError("malformed edit operation")
        if not (path.startswith("skills/") or path == "executionConfig/instructionVariant"):
            raise LearningError("candidate path is outside the learner bundle")
        lowered = path.casefold()
        if any(term in lowered for term in _PRIVILEGED) or path.startswith("skills/../"):
            raise LearningError("candidate attempts to edit a trusted control")
        serialized = json.dumps({"operation": operation, "path": path, "value": item.get("value")}, sort_keys=True, separators=(",", ":"))
        operations.append(serialized)
    if sum(len(item.splitlines()) for item in operations) > 200:
        raise LearningError("candidate exceeds the 200-line bound")
    return operations


def _validate_skill(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) - _SKILL_FIELDS or "procedure" not in value:
        raise LearningError("skill patch must contain only a generic procedure and bounds")
    procedure = value["procedure"]
    if not isinstance(procedure, str) or not procedure.strip() or len(procedure.splitlines()) > 120:
        raise LearningError("skill procedure is empty or too large")
    lowered = procedure.casefold()
    if any(term in lowered for term in _PRIVILEGED) or _FIXTURE_ID.search(procedure):
        raise LearningError("skill contains a trusted-control reference or fixture identifier")
    if any(marker in procedure for marker in ("call_tool(", "tool_sequence", "action_sequence", "expectedAnswer")):
        raise LearningError("skill contains a literal workflow or hidden-answer field")
    result = {"procedure": procedure}
    for key in ("applicability", "preconditions", "failureHandling"):
        if key in value:
            if key == "applicability" and not isinstance(value[key], Mapping):
                raise LearningError("skill applicability must be an object")
            if key != "applicability" and (not isinstance(value[key], list) or not all(isinstance(item, str) for item in value[key])):
                raise LearningError(f"skill {key} must be a string array")
            result[key] = value[key]
    return result


class LearningService:
    def __init__(self, retriever: AccessFilteredRetriever, model_runner: AuthenticatedModelRunner, candidate_sink: CandidateSink, active_bundle_hash: ActiveBundleReader) -> None:
        self.retriever = retriever
        self.model_runner = model_runner
        self.candidate_sink = candidate_sink
        self.active_bundle_hash = active_bundle_hash

    def propose(self, *, run_id: str, environment_id: str, goal: str, base_bundle_hash: str | None = None, environment: Mapping[str, Any] | None = None, feedback: Mapping[str, Any] | None = None, allowed_skill_ids: set[str] | None = None, emit: Callable[[str, str, str | None], None] | None = None) -> LearningProposal:
        active = _require_hash(self.active_bundle_hash(), "active bundle hash")
        if base_bundle_hash is not None and base_bundle_hash != active:
            raise LearningError("candidate base is not the pinned active bundle")
        result = self.retriever.search(goal, environment_id=environment_id, run_id=run_id, allowed_skill_ids=allowed_skill_ids)
        if not result.evidence:
            raise LearningError("learning requires verified development evidence")
        safe_environment = {key: environment[key] for key in ("environmentId", "version", "toolSchemas", "executionModes", "capabilities") if environment and key in environment}
        safe_environment["learningContext"] = result.prompt_payload()
        safe_environment["sanitizedFeedback"] = sanitize_feedback(feedback)
        safe_environment["baseBundleHash"] = active
        invocation = self.model_runner(goal=goal, environment=safe_environment, emit=emit or (lambda *_: None))
        if not all(isinstance(getattr(invocation, attr, None), str) and getattr(invocation, attr).strip() for attr in ("provider", "model", "response_id", "text")):
            raise LearningError("authenticated model invocation provenance is incomplete")
        usage = getattr(invocation, "usage", None)
        if not isinstance(usage, Mapping) or not usage:
            raise LearningError("authenticated model usage is missing")
        parsed = _parse_model_json(invocation.text)
        operations = _canonical_operations(parsed["editOperations"])
        skill = _validate_skill(parsed["skill"])
        predicted = parsed["predictedEffect"]
        proposer_version = parsed["proposerVersion"]
        if not isinstance(predicted, str) or not predicted.strip() or len(predicted) > 500 or not isinstance(proposer_version, str) or not proposer_version.strip():
            raise LearningError("predicted effect and proposer version are required")
        evidence_ids = parsed["supportingEvidenceIds"]
        if not isinstance(evidence_ids, list) or not evidence_ids or not all(isinstance(item, str) and item for item in evidence_ids):
            raise LearningError("supporting evidence must be a non-empty id array")
        try:
            evidence = self.retriever.require_development_evidence(evidence_ids, environment_id=environment_id, run_id=run_id)
        except RetrievalError as exc:
            raise LearningError(str(exc)) from exc
        config_patch = parsed.get("executionConfigPatch", {})
        if config_patch and (not isinstance(config_patch, Mapping) or set(config_patch) - {"instructionVariant"}):
            raise LearningError("execution configuration patch is outside the bounded learner surface")
        bundle_patch = {"skills": skill, "editOperations": operations, "executionConfigPatch": dict(config_patch)}
        patch_hash = _sha256(bundle_patch)
        declared_hashes = parsed.get("changedArtifactHashes")
        if declared_hashes is not None and declared_hashes != [patch_hash]:
            raise LearningError("changed artifact hash does not match the proposed patch")
        payload = {"baseBundleHash": active, "editOperations": operations, "changedArtifactHashes": [patch_hash], "supportingEvidenceIds": [item.source_id for item in evidence], "predictedEffect": predicted, "proposerVersion": proposer_version}
        authoritative = self.candidate_sink.create_candidate(payload)
        if not isinstance(authoritative, Mapping):
            raise LearningError("authoritative candidate store returned a non-object")
        state = authoritative.get("state")
        if state is not None and state not in {"draft", "validated", "staged"}:
            raise LearningError("learning sink attempted to activate or decide a candidate")
        if emit:
            emit("learning", "Evidence-linked bounded candidate staged for independent evaluation.", None)
        return LearningProposal(active, payload, bundle_patch, tuple(Citation(item.source_id, item.content_hash, item.kind) for item in evidence), {"provider": invocation.provider, "model": invocation.model, "responseId": invocation.response_id, "usage": dict(usage)}, dict(authoritative))


class Session2CandidateAdapter:
    """Adapter contract for session 2's ``ControlPlane.create_candidate``.

    ``request_factory`` is normally ``CandidateProposalRequest``.  Keeping it
    injected avoids importing or mutating session 2's API module.
    """

    def __init__(self, control_plane: Any, request_factory: Callable[..., Any] | None = None) -> None:
        self.control_plane = control_plane
        self.request_factory = request_factory

    def create_candidate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = self.request_factory(**dict(payload)) if self.request_factory else payload
        result = self.control_plane.create_candidate(request)
        if not isinstance(result, Mapping):
            raise LearningError("session 2 candidate API returned a non-object")
        return result
