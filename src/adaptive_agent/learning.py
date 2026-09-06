"""Bounded, evidence-linked learning proposals.

The learner may suggest a procedure/configuration patch, but this module never
activates it and never treats learner text or predicted effects as outcomes.
The injected candidate sink is the narrow seam to session 2's authoritative
candidate/store implementation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from threading import Event
from typing import Any, Callable, Mapping, Protocol, Sequence

from .retrieval import AccessFilteredRetriever, Citation, RetrievalError, RetrievalResult, SourceRecord, canonical_json


class LearningError(ValueError):
    pass


class ModelInvocation(Protocol):
    text: str
    provider: str
    model: str
    response_id: str
    usage: Mapping[str, Any]


def _optional_nonnegative_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        raise LearningError(f"{label} must be finite and non-negative")
    return float(value)


def _model_accounting(raw: Mapping[str, Any], usage: Mapping[str, Any], duration_seconds: float) -> dict[str, Any]:
    """Normalize provider usage while keeping billing and nominal cost distinct."""
    nominal = raw.get("nominalCostUsd")
    if nominal is None:
        provider_cost = usage.get("cost")
        if isinstance(provider_cost, Mapping):
            nominal = provider_cost.get("total")
    try:
        nominal = _optional_nonnegative_number(nominal, "nominalCostUsd")
    except LearningError:
        nominal = None
    measured = raw.get("costMicrounits")
    try:
        measured = _optional_nonnegative_number(measured, "costMicrounits")
    except LearningError:
        measured = None
    status = raw.get("economicCostStatus")
    if status is not None and (not isinstance(status, str) or not status):
        status = "unknown"
    # Prime subscription billing is not observable from the invocation.  A
    # provider SDK cost is nominal usage, never proof of billed spend.
    status = status or ("measured" if measured is not None else "unknown")
    return {
        "durationSeconds": duration_seconds,
        "economicCostStatus": status,
        **({"costMicrounits": int(round(measured))} if measured is not None else {}),
        **({"nominalCostUsd": nominal} if nominal is not None else {}),
    }


class AuthenticatedModelRunner(Protocol):
    def __call__(self, *, goal: str, environment: dict[str, Any], emit: Callable[[str, str, str | None], None]) -> ModelInvocation: ...


class PlannerModelClient(Protocol):
    """Structural copy of session 2's planner client seam."""

    def invoke(self, *, goal: str, environment: Mapping[str, Any], messages: Sequence[Mapping[str, str]]) -> Mapping[str, Any]: ...


class PlannerEvidenceSink(Protocol):
    """Parent-owned sink for trusted model observations."""

    def record_model_observation(self, evidence: Mapping[str, Any], *, trusted_parent: bool = False) -> Any: ...


class CandidateSink(Protocol):
    def persist_candidate_patch(self, patch_bytes: bytes, content_hash: str) -> Mapping[str, Any]: ...

    def create_candidate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ActiveBundleReader(Protocol):
    def __call__(self) -> str: ...


PROPOSAL_CONTRACT: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["predictedEffect", "editOperations", "supportingEvidenceIds", "proposerVersion", "skill"],
    "properties": {
        "predictedEffect": {"type": "string", "minLength": 1, "maxLength": 500, "description": "A hypothesis only, never an outcome claim."},
        "editOperations": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["path", "operation", "value"],
                "properties": {
                    "path": {"type": "string", "description": "Only skills/<generic-id>/{procedure,applicability,preconditions,failureHandling} or executionConfig/instructionVariant; no empty, '.', '..', absolute, or trusted-control paths."},
                    "operation": {"enum": ["add", "replace", "remove"]},
                    "value": {"description": "The exact value applied at path; remove requires null."},
                },
            },
        },
        "changedArtifactHashes": {"type": "array", "maxItems": 1, "description": "Optional caller-independent assertion; the service recomputes the hash from exact persisted patch bytes."},
        "supportingEvidenceIds": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1, "description": "Verified DEVELOPMENT evidence IDs only."}},
        "proposerVersion": {"type": "string", "minLength": 1},
        "skill": {
            "type": "object", "additionalProperties": False, "required": ["procedure"],
            "properties": {
                "procedure": {"type": "string", "minLength": 1, "description": "Generic learned procedure; may include legitimate broker/API orchestration."},
                "applicability": {"type": "object"},
                "preconditions": {"type": "array", "items": {"type": "string"}},
                "failureHandling": {"type": "array", "items": {"type": "string"}},
            },
        },
        "executionConfigPatch": {"type": "object", "additionalProperties": False, "properties": {"instructionVariant": {"type": "string"}}},
    },
    "description": "Every supplied skill/config field must equal the value of exactly one edit operation. The service applies and hashes this single canonical patch; predictedEffect and feedback never establish truth.",
}


def proposal_contract(allowed_supporting_evidence_ids: Sequence[str] | None = None) -> dict[str, Any]:
    contract = deepcopy(PROPOSAL_CONTRACT)
    if allowed_supporting_evidence_ids is not None:
        contract["properties"]["supportingEvidenceIds"]["items"] = {"enum": list(allowed_supporting_evidence_ids)}
    return contract


def proposal_contract_json(allowed_supporting_evidence_ids: Sequence[str] | None = None) -> str:
    return json.dumps(proposal_contract(allowed_supporting_evidence_ids), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise LearningError(f"{label} must be a SHA-256 digest")
    return value


def _usage_tokens(usage: Mapping[str, Any]) -> int | None:
    for key in ("totalTokens", "total_tokens", "outputTokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


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
        if not any(term in lowered for term in ("expected answer", "hidden answer", "hidden evaluator", "evaluator trace", "evaluator code", "secret", "credential")):
            clean["diagnostic"] = diagnostic
    observations = feedback.get("observations")
    if isinstance(observations, list):
        clean["observations"] = [item for item in observations if isinstance(item, str) and len(item) <= 300][:8]
    return clean


sanitize_execution_feedback = sanitize_feedback


class PlannerLearningAdapter:
    """Adapt session 2's PlannerModelClient to proposal generation.

    The adapter deliberately records provenance through the parent sink and
    passes proposal instructions as messages.  It does not import ``planner``
    or execute a planner/kernel turn, which keeps learning independent from
    task execution and prevents learner text from becoming authority.
    """

    def __init__(self, client: PlannerModelClient, evidence_sink: PlannerEvidenceSink) -> None:
        self.client = client
        self.evidence_sink = evidence_sink

    @staticmethod
    def messages_for(goal: str) -> list[dict[str, str]]:
        # The environment is already a first-class field in the planner
        # request.  Repeating it inside a message doubled large learning
        # contexts and made the parent token cap unenforceable.
        return [
            {
                "role": "system",
                "content": (
                    "Propose one generic bounded learning patch from verified development evidence. "
                    "Return exactly one JSON object matching the proposalContract in the authoritative "
                    "environment payload. Predicted effects are hypotheses, not outcomes. Only patch the "
                    "bounded skill/config paths; do not emit fixture IDs, hidden evaluator material, or "
                    "authority-bearing fields."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Use the authoritative environment payload. Read its learningContext, proposalContract, "
                    "sourceRuns, baseBundleHash, proposalLimits, and sanitizedFeedback. Cite only evidence "
                    "IDs present in learningContext.developmentEvidence and return the required JSON object."
                ),
            },
        ]

    def __call__(self, *, goal: str, environment: dict[str, Any], emit: Callable[[str, str, str | None], None], remaining_deadline: float | None = None, cancel: Event | None = None, token_cap: int | None = None) -> ModelInvocation:
        messages = self.messages_for(goal)
        prompt_budget = environment.get("promptBudgetBytes")
        if isinstance(prompt_budget, int) and not isinstance(prompt_budget, bool) and prompt_budget > 0:
            serialized_size = _serialized_planner_request_bytes(goal, environment, token_cap=token_cap)
            if serialized_size > prompt_budget:
                raise LearningError("learning request exceeds the deterministic serialized prompt budget")
        invoke = self.client.invoke
        parameters = inspect.signature(invoke).parameters
        accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        invoke_kwargs: dict[str, Any] = {"goal": goal, "environment": environment, "messages": messages}
        for name, value in (("remaining_deadline", remaining_deadline), ("cancel", cancel), ("token_cap", token_cap)):
            if accepts_kwargs or name in parameters:
                invoke_kwargs[name] = value
        started = time.monotonic()
        raw = invoke(**invoke_kwargs)
        duration_seconds = max(time.monotonic() - started, 0.0)
        if not isinstance(raw, Mapping):
            raise LearningError("planner client returned a non-object response")
        provider, model = raw.get("provider"), raw.get("model")
        response_id = raw.get("responseId", raw.get("response_id"))
        text, usage = raw.get("text"), raw.get("usage")
        if not all(isinstance(value, str) and value.strip() for value in (provider, model, response_id)) or not isinstance(usage, Mapping) or not usage:
            raise LearningError("planner response provenance is incomplete")
        accounting = _model_accounting(raw, usage, duration_seconds)
        self.evidence_sink.record_model_observation({"provider": provider, "model": model, "responseId": response_id, "usage": dict(usage), **accounting}, trusted_parent=True)
        if not isinstance(text, str) or not text.strip():
            raise LearningError("planner response text is missing")
        emit("model", "Authenticated model proposal received.", response_id)
        return _Invocation(text=text, provider=provider, model=model, response_id=response_id, usage=dict(usage))


@dataclass(frozen=True)
class _Invocation:
    text: str
    provider: str
    model: str
    response_id: str
    usage: Mapping[str, Any]


@dataclass(frozen=True)
class LearningProposal:
    base_bundle_hash: str
    candidate_payload: dict[str, Any]
    bundle_patch: dict[str, Any]
    citations: tuple[Citation, ...]
    model_provenance: dict[str, Any]
    authoritative_candidate: Mapping[str, Any]
    patch_bytes: bytes


_TOP_LEVEL = {"predictedEffect", "editOperations", "changedArtifactHashes", "supportingEvidenceIds", "proposerVersion", "skill", "executionConfigPatch"}
_PATCH_FIELDS = {"path", "operation", "value"}
_FIXTURE_ID = re.compile(r"\b(?:INV|TKT|PAY|SMP|SLOT|DEV)-[A-Z0-9_-]+\b", re.IGNORECASE)
_HIDDEN_LITERAL = re.compile(r"(?i)(expected\s*answer|hidden\s*answer|answer\s*key|evaluator[_ -]?only)")
_SKILL_FIELDS = {"procedure", "applicability", "preconditions", "failureHandling"}
_MAX_PATCH_BYTES = 32_768
_MAX_CHANGED_LINES = 200

# The provider reports token usage, while the request boundary only exposes a
# serialized string.  This is a byte ceiling, not a tokenization guarantee:
# one UTF-8 byte is allowed per available token, with explicit space reserved
# for output and provider/request framing.  Measured provider usage remains
# authoritative after dispatch.
_PROMPT_OUTPUT_HEADROOM_TOKENS = 1_024
_PROMPT_PROVIDER_HEADROOM_BYTES = 1_024
_PROMPT_FRAMING_HEADROOM_BYTES = 1_024
_MIN_USEFUL_EXCERPT_BYTES = 32


def _prompt_budget_bytes(token_cap: int | None) -> int | None:
    if token_cap is None:
        return None
    reserved = _PROMPT_OUTPUT_HEADROOM_TOKENS + _PROMPT_PROVIDER_HEADROOM_BYTES + _PROMPT_FRAMING_HEADROOM_BYTES
    budget = token_cap - reserved
    if budget <= 0:
        raise LearningError("learning proposal token cap is below the minimum serialized prompt budget")
    return budget


def _serialized_planner_request_bytes(goal: str, environment: Mapping[str, Any], *, token_cap: int | None = None) -> int:
    messages = PlannerLearningAdapter.messages_for(goal)
    request = json.dumps(
        {"goal": goal, "environment": environment, "messages": messages},
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    if token_cap is not None:
        request = f"A hard parent token cap of {max(0, token_cap)} applies to this call.\n" + request
    return len(request.encode("utf-8"))


def _pack_learning_context(
    result: RetrievalResult,
    *,
    goal: str,
    base_environment: Mapping[str, Any],
    feedback: Mapping[str, Any],
    active_bundle_hash: str,
    source_runs: frozenset[tuple[str, str]] | None,
    token_cap: int | None,
) -> tuple[RetrievalResult, dict[str, Any]]:
    """Select a deterministic, serialized-size-bounded learner context.

    Evidence is admitted per selected source run before optional documents,
    task state, or skills.  Long excerpts may be shortened, but their
    citations always retain the original source hash.  This keeps provenance
    verifiable while preventing a single durable artifact from consuming the
    whole model request.
    """
    budget = _prompt_budget_bytes(token_cap)
    source_keys = frozenset(source_runs or ())
    available_keys = frozenset((item.environment_id or "", item.run_id or "") for item in result.evidence)
    ordered_evidence = sorted(result.evidence, key=lambda item: (not item.failure, item.environment_id or "", item.run_id or "", -item.score, item.source_id))
    required_keys = list(dict.fromkeys((item.environment_id or "", item.run_id or "") for item in ordered_evidence))
    if source_keys:
        required_keys = [key for key in required_keys if key in source_keys]
    if source_keys and not source_keys.issubset(available_keys):
        missing = sorted(source_keys - available_keys)[0]
        raise LearningError(f"selected learning source run lacks verified development evidence: {missing[1]}")
    if not result.evidence:
        raise LearningError("learning requires verified development evidence")

    evidence = ordered_evidence
    mandatory: list[Any] = []
    for key in required_keys:
        item = next((candidate for candidate in evidence if (candidate.environment_id or "", candidate.run_id or "") == key), None)
        if item is not None and item not in mandatory:
            mandatory.append(item)
    if not mandatory:
        mandatory = [evidence[0]]
    unique_mandatory: list[Any] = []
    mandatory_ids: dict[str, tuple[str, str]] = {}
    for item in mandatory:
        key = (item.environment_id or "", item.run_id or "")
        previous_key = mandatory_ids.get(item.source_id)
        if previous_key is not None:
            if previous_key != key:
                raise LearningError("selected learning source runs do not have unique evidence IDs")
            continue
        mandatory_ids[item.source_id] = key
        unique_mandatory.append(item)
    mandatory = unique_mandatory
    if budget is not None and mandatory:
        # Reserve room for every selected source before filling optional
        # context.  A greedy full-size first item would otherwise crowd out a
        # later run and make coverage depend on source ordering.
        excerpt_reserve = max(_MIN_USEFUL_EXCERPT_BYTES, budget // (2 * len(mandatory)))
        mandatory = [item.with_excerpt(item.excerpt[:excerpt_reserve]) for item in mandatory]
    candidates = [(item, True) for item in mandatory]
    candidate_source_ids = {item.source_id for item in mandatory}
    for item in (*evidence, *result.docs, *result.task_state, *result.skills):
        # A mandatory item may be an excerpted copy of the source record, so
        # dataclass equality cannot be used for deduplication here.
        if item.source_id in candidate_source_ids:
            continue
        candidate_source_ids.add(item.source_id)
        candidates.append((item, False))

    selected: dict[str, list[Any]] = {"publicDocs": [], "developmentEvidence": [], "taskState": [], "activeSkills": []}
    key_for_kind = {
        "public_doc": "publicDocs",
        "live_evidence": "developmentEvidence",
        "task_state": "taskState",
        "skill": "activeSkills",
    }

    def current_result() -> RetrievalResult:
        return result.with_items(
            docs=selected["publicDocs"],
            evidence=selected["developmentEvidence"],
            task_state=selected["taskState"],
            skills=selected["activeSkills"],
        )

    def environment_for(candidate: RetrievalResult) -> dict[str, Any]:
        environment = dict(base_environment)
        environment["learningContext"] = candidate.prompt_payload()
        environment["sanitizedFeedback"] = dict(feedback)
        environment["baseBundleHash"] = active_bundle_hash
        ids = [item.source_id for item in candidate.evidence]
        environment["proposalContract"] = proposal_contract(ids)
        environment["proposalLimits"] = {"maxChangedArtifacts": 3, "maxChangedLogicalLines": _MAX_CHANGED_LINES, "maxPatchBytes": _MAX_PATCH_BYTES}
        if source_keys:
            environment["sourceRuns"] = [{"environmentId": env, "runId": run} for env, run in sorted(source_keys)]
        if budget is not None:
            environment["promptBudgetBytes"] = budget
        return environment

    def fits(candidate: RetrievalResult) -> bool:
        environment = environment_for(candidate)
        return budget is None or _serialized_planner_request_bytes(goal, environment, token_cap=token_cap) <= budget

    for item, required in candidates:
        bucket = selected[key_for_kind[item.kind.value]]
        candidate_bucket = [*bucket, item]
        selected[key_for_kind[item.kind.value]] = candidate_bucket
        if fits(current_result()):
            continue
        selected[key_for_kind[item.kind.value]] = bucket
        # Preserve a useful prefix of a required source even when its durable
        # record is larger than the remaining request budget.
        low, high = 0, len(item.excerpt)
        best: Any | None = None
        while low <= high:
            middle = (low + high) // 2
            shortened = item.with_excerpt(item.excerpt[:middle])
            selected[key_for_kind[item.kind.value]] = [*bucket, shortened]
            if fits(current_result()):
                best = shortened
                low = middle + 1
            else:
                high = middle - 1
        selected[key_for_kind[item.kind.value]] = bucket
        if best is not None and len(best.excerpt.encode("utf-8")) >= min(_MIN_USEFUL_EXCERPT_BYTES, len(item.excerpt.encode("utf-8"))):
            selected[key_for_kind[item.kind.value]] = [*bucket, best]
        elif required:
            raise LearningError("selected learning source evidence cannot fit the serialized model prompt")

    packed = current_result()
    environment = environment_for(packed)
    if budget is not None and _serialized_planner_request_bytes(goal, environment, token_cap=token_cap) > budget:
        raise LearningError("learning request exceeds the deterministic serialized prompt budget")
    return packed, environment


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


def _logical_lines(value: Any) -> int:
    if isinstance(value, str):
        return len(value.splitlines()) if value else 0
    if isinstance(value, Mapping):
        return sum(_logical_lines(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_logical_lines(item) for item in value)
    return 0


def _canonical_operations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise LearningError("at least one edit operation is required")
    if len(value) > 3:
        raise LearningError("candidate exceeds the three-artifact bound")
    operations: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    total_bytes = 0
    total_lines = 0
    for item in value:
        if isinstance(item, str):
            raise LearningError("edit operations must identify a bounded path and operation")
        if not isinstance(item, Mapping) or set(item) - _PATCH_FIELDS or not _PATCH_FIELDS.issubset(item):
            raise LearningError("malformed edit operation")
        path, operation = item["path"], item["operation"]
        if not isinstance(path, str) or not isinstance(operation, str) or operation not in {"add", "replace", "remove"}:
            raise LearningError("malformed edit operation")
        segments = path.split("/")
        if path.startswith("/") or any(segment in {"", ".", ".."} for segment in segments):
            raise LearningError("candidate path contains an unsafe segment")
        skill_path = len(segments) == 3 and segments[0] == "skills" and segments[2] in _SKILL_FIELDS
        config_path = segments == ["executionConfig", "instructionVariant"]
        if not (skill_path or config_path):
            raise LearningError("candidate path is outside the learner bundle")
        if skill_path and (not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", segments[1]) or _FIXTURE_ID.search(segments[1])):
            raise LearningError("candidate skill path is not a stable generic identifier")
        if path in seen_paths:
            raise LearningError("candidate contains duplicate applied paths")
        seen_paths.add(path)
        applied_value = item["value"]
        if operation == "remove" and applied_value is not None:
            raise LearningError("remove operations must have a null applied value")
        if operation != "remove" and applied_value is None:
            raise LearningError("add and replace operations require an applied value")
        expected_type = str if segments[-1] in {"procedure", "instructionVariant"} else (dict if segments[-1] == "applicability" else list)
        if operation != "remove" and not isinstance(applied_value, expected_type):
            raise LearningError(f"applied value for {path} has the wrong type")
        if operation != "remove" and segments[-1] in {"preconditions", "failureHandling"} and not all(isinstance(entry, str) for entry in applied_value):
            raise LearningError(f"applied value for {path} must contain only strings")
        canonical_operation = {"operation": operation, "path": path, "value": applied_value}
        operations.append(canonical_operation)
        total_bytes += len(canonical_json(canonical_operation).encode("utf-8"))
        total_lines += _logical_lines(applied_value)
    if total_lines > _MAX_CHANGED_LINES or total_bytes > _MAX_PATCH_BYTES:
        raise LearningError("candidate exceeds the 200-line bound")
    return operations


def _validate_skill(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) - _SKILL_FIELDS or "procedure" not in value:
        raise LearningError("skill patch must contain only a generic procedure and bounds")
    procedure = value["procedure"]
    if not isinstance(procedure, str) or not procedure.strip() or len(procedure.splitlines()) > 120:
        raise LearningError("skill procedure is empty or too large")
    if _FIXTURE_ID.search(procedure) or _HIDDEN_LITERAL.search(procedure):
        raise LearningError("skill contains a fixture identifier or hidden-answer literal")
    result = {"procedure": procedure}
    for key in ("applicability", "preconditions", "failureHandling"):
        if key in value:
            if key == "applicability" and not isinstance(value[key], Mapping):
                raise LearningError("skill applicability must be an object")
            if key != "applicability" and (not isinstance(value[key], list) or not all(isinstance(item, str) for item in value[key])):
                raise LearningError(f"skill {key} must be a string array")
            result[key] = value[key]
    if _HIDDEN_LITERAL.search(canonical_json(result)) or _FIXTURE_ID.search(canonical_json(result)):
        raise LearningError("skill contains a fixture identifier or hidden-answer literal")
    return result


def _validate_proposal_payload(
    parsed: Mapping[str, Any],
    *,
    retriever: AccessFilteredRetriever,
    environment_id: str,
    run_id: str,
    allowed_source_runs: frozenset[tuple[str, str]] | None = None,
    exposed_evidence_ids: frozenset[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, str, tuple[SourceRecord, ...], dict[str, Any], dict[str, Any]]:
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
        evidence = retriever.require_development_evidence(evidence_ids, environment_id=environment_id, run_id=run_id, allowed_source_runs=allowed_source_runs)
    except RetrievalError as exc:
        raise LearningError(str(exc)) from exc
    if exposed_evidence_ids is not None and not set(evidence_ids).issubset(exposed_evidence_ids):
        raise LearningError("supporting evidence must be selected from the exposed learning context")
    config_patch = parsed.get("executionConfigPatch", {})
    if config_patch and (not isinstance(config_patch, Mapping) or set(config_patch) - {"instructionVariant"}):
        raise LearningError("execution configuration patch is outside the bounded learner surface")
    applied_skill: dict[str, Any] = {}
    applied_config: dict[str, Any] = {}
    for operation in operations:
        path = operation["path"]
        destination = applied_config if path == "executionConfig/instructionVariant" else applied_skill
        field = "instructionVariant" if path == "executionConfig/instructionVariant" else path.split("/")[-1]
        destination[field] = operation["value"]
    if applied_skill != skill:
        raise LearningError("skill fields must equal the exact values in the applied operations")
    if applied_config != dict(config_patch):
        raise LearningError("execution configuration must equal the exact values in the applied operations")
    bundle_patch = {"operations": operations, "skill": applied_skill, "executionConfigPatch": applied_config}
    return operations, skill, predicted, proposer_version, evidence, dict(config_patch), bundle_patch


class LearningService:
    def __init__(self, retriever: AccessFilteredRetriever, model_runner: AuthenticatedModelRunner, candidate_sink: CandidateSink, active_bundle_hash: ActiveBundleReader) -> None:
        self.retriever = retriever
        self.model_runner = model_runner
        self.candidate_sink = candidate_sink
        self.active_bundle_hash = active_bundle_hash

    def propose(self, *, run_id: str, environment_id: str, goal: str, base_bundle_hash: str | None = None, environment: Mapping[str, Any] | None = None, feedback: Mapping[str, Any] | None = None, allowed_skill_ids: set[str] | None = None, emit: Callable[[str, str, str | None], None] | None = None, remaining_deadline: float | None = None, cancel: Event | None = None, token_cap: int | None = None, max_repair_attempts: int = 0, source_runs: frozenset[tuple[str, str]] | None = None) -> LearningProposal:
        if cancel is not None and cancel.is_set():
            raise LearningError("learning proposal cancelled before model invocation")
        if remaining_deadline is not None and remaining_deadline <= 0:
            raise LearningError("learning proposal deadline expired before model invocation")
        if token_cap is not None and token_cap <= 0:
            raise LearningError("learning proposal token cap must be positive")
        if max_repair_attempts < 0 or max_repair_attempts > 1:
            raise LearningError("learning proposal repair attempts must be 0 or 1")
        active = _require_hash(self.active_bundle_hash(), "active bundle hash")
        if base_bundle_hash is not None and base_bundle_hash != active:
            raise LearningError("candidate base is not the pinned active bundle")
        if source_runs is not None:
            result = self.retriever.search(goal, environment_id=environment_id, run_id=run_id, allowed_skill_ids=allowed_skill_ids, allowed_source_runs=source_runs)
        else:
            result = self.retriever.search(goal, environment_id=environment_id, run_id=run_id, allowed_skill_ids=allowed_skill_ids)
        if not result.evidence:
            raise LearningError("learning requires verified development evidence")
        base_environment = {key: environment[key] for key in ("environmentId", "version", "toolSchemas", "executionModes", "capabilities") if environment and key in environment}
        runner = self.model_runner
        parameters = inspect.signature(runner).parameters
        accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
        total_used_tokens = 0
        attempt_feedback = sanitize_feedback(feedback)
        deadline_started = time.monotonic()
        for attempt in range(max_repair_attempts + 1):
            if cancel is not None and cancel.is_set():
                raise LearningError("learning proposal cancelled before model invocation")
            remaining_tokens = None if token_cap is None else token_cap - total_used_tokens
            if remaining_tokens is not None and remaining_tokens <= 0:
                raise LearningError("learning proposal token cap exhausted before repair")
            attempt_result, attempt_environment = _pack_learning_context(
                result,
                goal=goal,
                base_environment=base_environment,
                feedback=attempt_feedback,
                active_bundle_hash=active,
                source_runs=source_runs,
                token_cap=remaining_tokens,
            )
            attempt_deadline = remaining_deadline
            if remaining_deadline is not None:
                attempt_deadline = remaining_deadline - (time.monotonic() - deadline_started)
                if attempt_deadline <= 0:
                    raise LearningError("learning proposal deadline expired before repair")
            runner_kwargs: dict[str, Any] = {"goal": goal, "environment": attempt_environment, "emit": emit or (lambda *_: None)}
            for name, value in (("remaining_deadline", attempt_deadline), ("cancel", cancel), ("token_cap", remaining_tokens)):
                if accepts_kwargs or name in parameters:
                    runner_kwargs[name] = value
            invocation = runner(**runner_kwargs)
            if cancel is not None and cancel.is_set():
                raise LearningError("learning proposal cancelled after model invocation")
            if not all(isinstance(getattr(invocation, attr, None), str) and getattr(invocation, attr).strip() for attr in ("provider", "model", "response_id", "text")):
                error = LearningError("authenticated model invocation provenance is incomplete")
            else:
                usage = getattr(invocation, "usage", None)
                if not isinstance(usage, Mapping) or not usage:
                    error = LearningError("authenticated model usage is missing")
                else:
                    used_tokens = _usage_tokens(usage)
                    if remaining_tokens is not None and used_tokens is not None and used_tokens > remaining_tokens:
                        raise LearningError("learning proposal exceeded the model token cap")
                    if used_tokens is not None:
                        total_used_tokens += used_tokens
                    try:
                        parsed = _parse_model_json(invocation.text)
                        operations, skill, predicted, proposer_version, evidence, config_patch, bundle_patch = _validate_proposal_payload(
                            parsed,
                            retriever=self.retriever,
                            environment_id=environment_id,
                            run_id=run_id,
                            allowed_source_runs=source_runs,
                            exposed_evidence_ids=frozenset(item.source_id for item in attempt_result.evidence),
                        )
                        break
                    except LearningError as exc:
                        error = exc
            if attempt >= max_repair_attempts:
                raise error
            if token_cap is not None and _usage_tokens(getattr(invocation, "usage", {})) is None:
                raise LearningError("cannot repair proposal without token usage") from error
            attempt_feedback = {"status": "failed", "failureClass": "malformed_proposal", "diagnostic": str(error)[:1000]}
        else:
            raise LearningError("learning proposal repair failed")
        patch_bytes = canonical_json(bundle_patch).encode("utf-8")
        if len(patch_bytes) > _MAX_PATCH_BYTES:
            raise LearningError("candidate patch exceeds the byte bound")
        patch_hash = hashlib.sha256(patch_bytes).hexdigest()
        declared_hashes = parsed.get("changedArtifactHashes")
        if declared_hashes not in (None, []) and declared_hashes != [patch_hash]:
            raise LearningError("changed artifact hash does not match the proposed patch")
        persist = getattr(self.candidate_sink, "persist_candidate_patch", None)
        if not callable(persist):
            raise LearningError("candidate sink must persist exact immutable patch bytes before staging")
        persisted = persist(patch_bytes, patch_hash)
        if not isinstance(persisted, Mapping) or persisted.get("sha256", persisted.get("contentHash")) != patch_hash or persisted.get("stored") is not True or persisted.get("immutable") is not True:
            raise LearningError("candidate sink did not attest the exact persisted patch bytes")
        if "size" in persisted and persisted["size"] != len(patch_bytes):
            raise LearningError("candidate sink persisted a different patch size")
        payload = {"baseBundleHash": active, "editOperations": operations, "changedArtifactHashes": [patch_hash], "supportingEvidenceIds": [item.source_id for item in evidence], "predictedEffect": predicted, "proposerVersion": proposer_version}
        payload["editOperations"] = [canonical_json(operation) for operation in operations]
        authoritative = self.candidate_sink.create_candidate(payload)
        if not isinstance(authoritative, Mapping):
            raise LearningError("authoritative candidate store returned a non-object")
        state = authoritative.get("state")
        if state is not None and state not in {"draft", "validated", "staged"}:
            raise LearningError("learning sink attempted to activate or decide a candidate")
        if emit:
            emit("learning", "Evidence-linked bounded candidate staged for independent evaluation.", None)
        return LearningProposal(active, payload, bundle_patch, tuple(Citation(item.source_id, item.content_hash, item.kind) for item in evidence), {"provider": invocation.provider, "model": invocation.model, "responseId": invocation.response_id, "usage": dict(usage)}, dict(authoritative), patch_bytes)


class Session2CandidateAdapter:
    """Adapter contract for session 2's ``ControlPlane.create_candidate``.

    ``request_factory`` is normally ``CandidateProposalRequest``.  Keeping it
    injected avoids importing or mutating session 2's API module.
    """

    def __init__(self, control_plane: Any, request_factory: Callable[..., Any] | None = None, patch_store: Callable[[bytes, str], Mapping[str, Any]] | None = None) -> None:
        self.control_plane = control_plane
        self.request_factory = request_factory
        self.patch_store = patch_store

    def persist_candidate_patch(self, patch_bytes: bytes, content_hash: str) -> Mapping[str, Any]:
        if self.patch_store is None:
            raise LearningError("session 2 adapter requires an authoritative immutable patch store")
        result = self.patch_store(patch_bytes, content_hash)
        if not isinstance(result, Mapping):
            raise LearningError("authoritative patch store returned a non-object")
        return result

    def create_candidate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = self.request_factory(**dict(payload)) if self.request_factory else payload
        result = self.control_plane.create_candidate(request)
        if not isinstance(result, Mapping):
            raise LearningError("session 2 candidate API returned a non-object")
        return result
