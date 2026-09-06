from pathlib import Path

import pytest

from adaptive_agent.production_evaluator import _reject_shared_qa_path


def test_evaluator_rejects_shared_root_qa_store_and_children():
    with pytest.raises(RuntimeError, match="shared root QA"):
        _reject_shared_qa_path("/private/tmp/adaptive-agent-browser-api")
    with pytest.raises(RuntimeError, match="shared root QA"):
        _reject_shared_qa_path("/private/tmp/adaptive-agent-browser-api/child")


def test_evaluator_requires_distinct_source_and_target_stores(tmp_path: Path):
    with pytest.raises(RuntimeError, match="isolated"):
        _reject_shared_qa_path(tmp_path, tmp_path)
