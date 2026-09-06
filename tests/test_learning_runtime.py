"""Factory/wiring tests for LearningRuntime; real-provider smoke is separate."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
core_src = os.environ.get("ADAPTIVE_AGENT_CORE_SRC")
if core_src:
    import adaptive_agent

    adaptive_agent.__path__.append(str(Path(core_src) / "adaptive_agent"))
    adaptive_agent.__path__.append(str(ROOT / "src" / "adaptive_agent"))

try:
    from adaptive_agent.candidate import CandidateManager
    from adaptive_agent.learning_runtime import LearningRuntime
    from adaptive_agent.models import CandidateProposal, SkillBundle, SkillVersion
except ImportError as exc:
    pytest.skip(f"durable core is unavailable in this isolated worker: {exc}", allow_module_level=True)

from test_learning_store_integration import ENVIRONMENT, RUN, _setup_store  # noqa: E402


class FakeClient:
    def invoke(self, *, goal, environment, messages, remaining_deadline=None, cancel=None, token_cap=None):
        evidence_id = environment["learningContext"]["developmentEvidence"][0]["sourceId"]
        procedure = "Read the current version before retrying a bounded reconciliation."
        payload = {
            "predictedEffect": "reduce version conflicts",
            "editOperations": [{"path": "skills/runtime-reconciliation/procedure", "operation": "add", "value": procedure}],
            "supportingEvidenceIds": [evidence_id],
            "proposerVersion": "runtime-test",
            "skill": {"procedure": procedure},
        }
        import json

        return {"provider": "openai-codex", "model": "test-luna", "responseId": "resp-runtime-test", "text": json.dumps(payload), "usage": {"totalTokens": 12}}


def test_runtime_composes_durable_learning_and_restart_readback(tmp_path: Path):
    store, manager, evidence_id = _setup_store(tmp_path)
    runtime = LearningRuntime.build(store=store, manager=manager, model_client=FakeClient(), token_budget=1000, wall_seconds=20)
    result = runtime.propose_completed_run(RUN)
    assert result.authoritative_candidate["state"] == "validated"
    assert result.candidate_payload["supportingEvidenceIds"] == [evidence_id]
    restarted_store = type(store)(tmp_path)
    restarted_manager = CandidateManager(restarted_store)
    restarted = LearningRuntime.build(store=restarted_store, manager=restarted_manager, model_client=FakeClient(), token_budget=1000, wall_seconds=20)
    assert restarted.reload_candidate(result.authoritative_candidate["candidate_id"])["state"] == "validated"
