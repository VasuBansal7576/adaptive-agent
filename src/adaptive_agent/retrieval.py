"""Access-filtered, provenance-preserving retrieval for bounded learning.

This module has no knowledge of a domain fixture.  A source is usable only when
its content hash, visibility, environment scope, and evaluation partition have
already been checked.  Ranking is intentionally boring and deterministic; the
trust boundary is the filter, not the ranker.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Protocol


class RetrievalError(ValueError):
    """Raised when untrusted or inconsistent retrieval input is supplied."""


class SourceKind(StrEnum):
    PUBLIC_DOC = "public_doc"
    LIVE_EVIDENCE = "live_evidence"
    TASK_STATE = "task_state"
    SKILL = "skill"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]{2,}", value.casefold()) if token not in {"the", "and", "for", "with", "from"}}


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    kind: SourceKind
    content: str
    content_hash: str
    environment_id: str | None = None
    run_id: str | None = None
    partition: str | None = None
    visibility: str = "learner"
    trust_class: str = "operator"
    verified: bool = True
    active: bool = True
    citations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.content:
            raise RetrievalError("source id and content are required")
        if self.content_hash != content_hash(self.content):
            raise RetrievalError(f"content hash mismatch for source {self.source_id}")
        if not isinstance(self.verified, bool) or not isinstance(self.active, bool):
            raise RetrievalError("verified and active must be booleans")
        if not isinstance(self.visibility, str) or not isinstance(self.trust_class, str):
            raise RetrievalError("visibility and trust_class must be strings")
        if "authority" in self.metadata and not isinstance(self.metadata["authority"], bool):
            raise RetrievalError("skill authority metadata must be boolean")
        if self.visibility not in {"learner", "operator", "evaluator_only"}:
            raise RetrievalError("unknown source visibility")
        if self.kind is SourceKind.LIVE_EVIDENCE and self.partition not in {"training", "development", "validation", "final"}:
            raise RetrievalError("live evidence must have an explicit evaluation partition")
        if self.kind is SourceKind.SKILL and self.metadata.get("authority", False):
            raise RetrievalError("learned skills cannot carry authority")
        if self.kind is SourceKind.PUBLIC_DOC and self.visibility == "evaluator_only":
            raise RetrievalError("public documentation cannot be evaluator-only")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceRecord":
        """Construct a record from a store adapter without trusting its hash."""
        if not isinstance(value, Mapping):
            raise RetrievalError("source record must be an object")
        raw_kind = value.get("kind")
        try:
            kind = raw_kind if isinstance(raw_kind, SourceKind) else SourceKind(str(raw_kind))
        except ValueError as exc:
            raise RetrievalError("unknown source kind") from exc
        content = value.get("content")
        if not isinstance(content, str):
            raise RetrievalError("source content must be text")
        supplied_hash = value.get("contentHash", value.get("content_hash"))
        if not isinstance(supplied_hash, str):
            raise RetrievalError("source content hash is required")
        verified = value.get("verified", False)
        active = value.get("active", True)
        if not isinstance(verified, bool) or not isinstance(active, bool):
            raise RetrievalError("verified and active must be booleans")
        citations = value.get("citations", ())
        if not isinstance(citations, (list, tuple)) or not all(isinstance(item, str) for item in citations):
            raise RetrievalError("citations must be a string array")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise RetrievalError("metadata must be an object")
        source_id = value.get("sourceId", value.get("source_id", ""))
        visibility = value.get("visibility", "learner")
        trust_class = value.get("trustClass", value.get("trust_class", "operator"))
        environment_id = value.get("environmentId", value.get("environment_id"))
        run_id = value.get("runId", value.get("run_id"))
        if not isinstance(source_id, str) or not isinstance(visibility, str) or not isinstance(trust_class, str):
            raise RetrievalError("source id, visibility, and trust class must be strings")
        if environment_id is not None and not isinstance(environment_id, str):
            raise RetrievalError("environment id must be a string")
        if run_id is not None and not isinstance(run_id, str):
            raise RetrievalError("run id must be a string")
        return cls(
            source_id=source_id,
            kind=kind,
            content=content,
            content_hash=supplied_hash,
            environment_id=environment_id,
            run_id=run_id,
            partition=value.get("partition"),
            visibility=visibility,
            trust_class=trust_class,
            verified=verified,
            active=active,
            citations=tuple(citations),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class Citation:
    source_id: str
    content_hash: str
    kind: SourceKind

    def to_dict(self) -> dict[str, str]:
        return {"sourceId": self.source_id, "contentHash": self.content_hash, "kind": self.kind.value}


@dataclass(frozen=True)
class ContextItem:
    source_id: str
    kind: SourceKind
    excerpt: str
    citation: Citation
    score: float
    environment_id: str | None = None
    run_id: str | None = None
    failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = {"sourceId": self.source_id, "kind": self.kind.value, "excerpt": self.excerpt, "citation": self.citation.to_dict(), "score": self.score}
        if self.environment_id is not None:
            result["environmentId"] = self.environment_id
        if self.run_id is not None:
            result["runId"] = self.run_id
        return result

    def with_excerpt(self, excerpt: str) -> "ContextItem":
        """Return a bounded view without changing the cited source hash."""
        return ContextItem(self.source_id, self.kind, excerpt, self.citation, self.score, self.environment_id, self.run_id, self.failure)


@dataclass(frozen=True)
class RetrievalResult:
    docs: tuple[ContextItem, ...] = ()
    evidence: tuple[ContextItem, ...] = ()
    task_state: tuple[ContextItem, ...] = ()
    skills: tuple[ContextItem, ...] = ()

    @property
    def all_items(self) -> tuple[ContextItem, ...]:
        return self.docs + self.evidence + self.task_state + self.skills

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(item.source_id for item in self.all_items)

    def citations(self) -> tuple[Citation, ...]:
        return tuple(item.citation for item in self.all_items)

    def prompt_payload(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "publicDocs": [item.to_dict() for item in self.docs],
            "developmentEvidence": [item.to_dict() for item in self.evidence],
            "taskState": [item.to_dict() for item in self.task_state],
            "activeSkills": [item.to_dict() for item in self.skills],
        }

    def with_items(self, *, docs: Iterable[ContextItem] = (), evidence: Iterable[ContextItem] = (), task_state: Iterable[ContextItem] = (), skills: Iterable[ContextItem] = ()) -> "RetrievalResult":
        return RetrievalResult(tuple(docs), tuple(evidence), tuple(task_state), tuple(skills))


class SourceProvider(Protocol):
    def list_sources(self) -> Iterable[SourceRecord | Mapping[str, Any]]: ...


class InMemorySourceProvider:
    """Small adapter used by tests and by the session-2 integration seam."""

    def __init__(self, sources: Iterable[SourceRecord | Mapping[str, Any]] = ()) -> None:
        self._sources = list(sources)

    def list_sources(self) -> tuple[SourceRecord, ...]:
        return tuple(source if isinstance(source, SourceRecord) else SourceRecord.from_mapping(source) for source in self._sources)


class AccessFilteredRetriever:
    """Filters source authority and scope before calculating a relevance score."""

    def __init__(self, provider: SourceProvider, *, max_items_per_kind: int = 8) -> None:
        if max_items_per_kind <= 0:
            raise ValueError("max_items_per_kind must be positive")
        self.provider = provider
        self.max_items_per_kind = max_items_per_kind

    def _allowed(self, source: SourceRecord, *, environment_id: str, run_id: str, allowed_skill_ids: set[str] | None, allowed_source_runs: frozenset[tuple[str, str]] | None = None) -> bool:
        if not source.active or not source.verified or source.visibility != "learner":
            return False
        if allowed_source_runs is not None:
            # Declared multi-run exposure set: only sources whose
            # (environment, run) pair was explicitly selected may enter the
            # learner context — anything outside the set stays fail-closed.
            allowed_envs = {env for env, _run in allowed_source_runs}
            if source.environment_id not in {None, *allowed_envs}:
                return False
            if source.kind is SourceKind.PUBLIC_DOC:
                return source.trust_class in {"operator", "system"}
            bound = (source.environment_id, source.run_id) in allowed_source_runs
            if source.kind is SourceKind.LIVE_EVIDENCE:
                return bound and source.partition == "development" and source.trust_class in {"broker", "evaluator", "system"}
            if source.kind is SourceKind.TASK_STATE:
                return bound and source.trust_class in {"system", "broker"}
            if source.kind is SourceKind.SKILL:
                return source.metadata.get("bundleState") == "active" and (allowed_skill_ids is None or source.source_id in allowed_skill_ids)
            return False
        if source.environment_id not in {None, environment_id}:
            return False
        if source.kind is SourceKind.PUBLIC_DOC:
            return source.trust_class in {"operator", "system"}
        if source.kind is SourceKind.LIVE_EVIDENCE:
            return source.run_id == run_id and source.partition == "development" and source.trust_class in {"broker", "evaluator", "system"}
        if source.kind is SourceKind.TASK_STATE:
            return source.run_id == run_id and source.trust_class in {"system", "broker"}
        if source.kind is SourceKind.SKILL:
            return source.metadata.get("bundleState") == "active" and (allowed_skill_ids is None or source.source_id in allowed_skill_ids)
        return False

    def search(self, query: str, *, environment_id: str, run_id: str, allowed_skill_ids: set[str] | None = None, allowed_source_runs: frozenset[tuple[str, str]] | None = None) -> RetrievalResult:
        query_tokens = _tokens(query)
        evidence_run_of: dict[str, str] = {}
        buckets: dict[SourceKind, list[ContextItem]] = {kind: [] for kind in SourceKind}
        # Do not rank first and filter later.  This loop is intentionally the
        # only place where provider records become learner-visible context.
        for raw in self.provider.list_sources():
            source = raw if isinstance(raw, SourceRecord) else SourceRecord.from_mapping(raw)
            if not self._allowed(source, environment_id=environment_id, run_id=run_id, allowed_skill_ids=allowed_skill_ids, allowed_source_runs=allowed_source_runs):
                continue
            overlap = len(query_tokens & _tokens(source.content))
            score = overlap / max(len(query_tokens), 1)
            # Development evidence is retained after the access filter even
            # when lexical relevance is zero.  The learner must be able to
            # inspect a complete verified failure set, while ranking still
            # orders the relevant records first.  Other sources stay query
            # bounded.
            if score == 0 and query_tokens and source.kind is not SourceKind.LIVE_EVIDENCE:
                continue
            buckets[source.kind].append(ContextItem(
                source.source_id,
                source.kind,
                source.content,
                Citation(source.source_id, source.content_hash, source.kind),
                score,
                source.environment_id,
                source.run_id,
                source.kind is SourceKind.LIVE_EVIDENCE and source.metadata.get("outcomePassed") is False,
            ))
            if allowed_source_runs is not None and source.kind is SourceKind.LIVE_EVIDENCE:
                evidence_run_of[source.source_id] = source.run_id or ""
        result: dict[SourceKind, tuple[ContextItem, ...]] = {}
        for kind, items in buckets.items():
            ordered = sorted(items, key=lambda item: (-item.score, item.source_id))
            if kind is SourceKind.LIVE_EVIDENCE and allowed_source_runs is not None:
                # Bound evidence per selected run first so one chatty run
                # cannot consume the whole window and erase environment or
                # failure coverage; rank within each run deterministically.
                per_run: dict[str, list[ContextItem]] = {}
                for item in ordered:
                    per_run.setdefault(evidence_run_of.get(item.source_id, ""), []).append(item)
                result[kind] = tuple(item for run_key in sorted(per_run) for item in sorted(per_run[run_key], key=lambda item: (not item.failure, -item.score, item.source_id))[: self.max_items_per_kind])
            else:
                result[kind] = tuple(ordered[: self.max_items_per_kind])
        return RetrievalResult(docs=result[SourceKind.PUBLIC_DOC], evidence=result[SourceKind.LIVE_EVIDENCE], task_state=result[SourceKind.TASK_STATE], skills=result[SourceKind.SKILL])

    def require_development_evidence(self, evidence_ids: Iterable[str], *, environment_id: str, run_id: str, allowed_source_runs: frozenset[tuple[str, str]] | None = None) -> tuple[SourceRecord, ...]:
        requested = tuple(dict.fromkeys(str(item) for item in evidence_ids))
        if not requested:
            raise RetrievalError("at least one development evidence citation is required")
        available: dict[str, SourceRecord] = {}
        for raw in self.provider.list_sources():
            source = raw if isinstance(raw, SourceRecord) else SourceRecord.from_mapping(raw)
            if source.kind is SourceKind.LIVE_EVIDENCE and self._allowed(source, environment_id=environment_id, run_id=run_id, allowed_skill_ids=None, allowed_source_runs=allowed_source_runs):
                available[source.source_id] = source
        missing = [source_id for source_id in requested if source_id not in available]
        if missing:
            raise RetrievalError(f"development evidence does not exist in this run: {missing[0]}")
        return tuple(available[source_id] for source_id in requested)
