"""Shared test fixtures for the Store/Broker/Candidate scope.

The driver/provider here is a SYNTHETIC TEST-ONLY stub. It proves broker and
promotion behavior; it makes no learning claim.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from adaptive_agent.broker import ToolBroker, ToolProvider
from adaptive_agent.candidate import CandidateManager
from adaptive_agent.environment import EnvironmentRegistry
from adaptive_agent.models import ArtifactRef, EnvironmentManifest, ToolSchema
from adaptive_agent.store import Store


class FakeProvider(ToolProvider):
    """In-memory test provider with a versioned record and an optional crash."""

    def __init__(self, interference: bool = False, crash_on_write: bool = False) -> None:
        self.interference = interference
        self.crash_on_write = crash_on_write
        self.state: dict[str, dict[str, Any]] = {}
        self.effects_applied: set[str] = set()  # idempotency keys that landed

    def reset(self, run_id: str) -> None:
        self.state[run_id] = {
            "records": {"record-1": {"version": 1, "value": "initial"}},
            "reads": {},
        }

    def effect(self, tool: str) -> str:
        return "write" if tool == "update_record" else "read"

    def version(self, tool: str) -> str:
        return "1"

    def execute(self, run_id: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        st = self.state[run_id]
        if tool == "list_records":
            return {"records": [{"id": k, **v} for k, v in st["records"].items()]}
        if tool == "read_record":
            rid = arguments["record_id"]
            rec = st["records"][rid]
            st["reads"][rid] = rec["version"]
            return {"id": rid, "version": rec["version"], "value": rec["value"]}
        if tool == "update_record":
            if self.crash_on_write:
                raise RuntimeError("provider crash before commit")
            rid = arguments["record_id"]
            rec = st["records"][rid]
            if self.interference and rid not in st["reads"]:
                rec["version"] += 1
            if rec["version"] != arguments["version"]:
                return {"ok": False, "current_version": rec["version"], "current_value": rec["value"]}
            rec["version"] += 1
            rec["value"] = arguments["value"]
            return {"ok": True, "current_version": rec["version"], "current_value": rec["value"]}
        raise RuntimeError("unknown tool")

    def reconcile(self, run_id: str, tool: str, arguments: dict[str, Any], idempotency_key: str) -> str:
        if self.crash_on_write:
            return "unknown"
        return "confirmed" if idempotency_key in self.effects_applied else "no_effect"


NEUTRAL_MANIFEST = EnvironmentManifest(
    environmentId="neutral-test-env",
    version="1.0.0",
    docs=[ArtifactRef(id="doc", version="1", sha256="0" * 64)],
    toolSchemas=[
        ToolSchema(
            name="list_records",
            version="1",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
            outputSchema={"type": "object"},
            effect="read",
        ),
        ToolSchema(
            name="read_record",
            version="1",
            inputSchema={
                "type": "object",
                "required": ["record_id"],
                "properties": {"record_id": {"type": "string"}},
                "additionalProperties": False,
            },
            outputSchema={"type": "object"},
            effect="read",
        ),
        ToolSchema(
            name="update_record",
            version="1",
            inputSchema={
                "type": "object",
                "required": ["record_id", "version", "value"],
                "properties": {
                    "record_id": {"type": "string"},
                    "version": {"type": "integer", "minimum": 1},
                    "value": {"type": "string", "maxLength": 64},
                },
                "additionalProperties": False,
            },
            outputSchema={"type": "object"},
            effect="write",
        ),
    ],
    policyRef=ArtifactRef(id="pol", version="1", sha256="0" * 64),
    evaluatorRef=ArtifactRef(id="eval", version="1", sha256="0" * 64),
    resetRef=ArtifactRef(id="rst", version="1", sha256="0" * 64),
    capabilities=["stateful", "versioned"],
)


@pytest.fixture
def workspace() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def store(workspace: Path) -> Store:
    return Store(workspace)


@pytest.fixture
def registry(store: Store) -> EnvironmentRegistry:
    return EnvironmentRegistry(store)


@pytest.fixture
def registered_env(registry: EnvironmentRegistry) -> EnvironmentManifest:
    registry.register(NEUTRAL_MANIFEST)
    return NEUTRAL_MANIFEST


@pytest.fixture
def broker(store: Store, registry: EnvironmentRegistry) -> ToolBroker:
    return ToolBroker(store, registry)


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider(interference=True)


@pytest.fixture
def manager(store: Store) -> CandidateManager:
    return CandidateManager(store)
