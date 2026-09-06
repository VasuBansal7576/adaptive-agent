from __future__ import annotations

import os
from pathlib import Path


from adaptive_agent.prime_safety_probe import run_eval_005_prime_runtime_probe


def test_eval005_prime_probe_uses_real_docker_and_isolated_store(tmp_path: Path):
    if not os.environ.get("AO_SESSION_ID"):
        os.environ["AO_SESSION_ID"] = "eval005-test"
    result = run_eval_005_prime_runtime_probe(tmp_path / "eval005")
    assert result["caseId"] == "EVAL-005"
    assert result["passed"] is True
    assert result["provider"] == {"name": "deterministic-test-provider", "simulated": True, "paidCall": False}
    assert result["provenance"]["actualDocker"] is True
    assert len(result["receipts"]) == 2
    assert all(receipt["provenance"]["actualDocker"] for receipt in result["receipts"])
    assert all(receipt["simulatedModel"] for receipt in result["receipts"])


def test_eval005_receipts_show_recovery_and_retained_over_cap_usage(tmp_path: Path):
    result = run_eval_005_prime_runtime_probe(tmp_path / "eval005")
    failure, cost = result["cases"]
    assert failure["outcome"]["plannerFailureVisibleToParent"]["kind"] == "host_error"
    assert failure["outcome"]["childFailureVisibleToParent"]["status"] == "error"
    assert failure["outcome"]["recovery"]["status"] == "ok"
    assert failure["outcome"]["capDenial"]["kind"] == "host_error"
    assert failure["outcome"]["budget"] == {
        "childRunsUsed": 3,
        "childRunsCap": 3,
        "modelTokensUsed": 4,
        "modelTokensCap": 10,
        "modelCostMicrounitsUsed": 2,
        "modelCostMicrounitsCap": 10,
        "costAccountingBlocked": False,
    }
    assert cost["outcome"]["withinCap"]["status"] == "ok"
    assert cost["outcome"]["overCap"]["kind"] == "host_error"
    assert cost["outcome"]["postCapDenial"]["kind"] == "host_error"
    assert cost["outcome"]["budget"]["modelTokensUsed"] == 4
    assert cost["outcome"]["budget"]["modelCostMicrounitsUsed"] == 4
    assert cost["modelCalls"] == 2
    assert [item["responseId"] for item in cost["modelReceipts"]] == ["eval005-simulated-1", "eval005-simulated-2"]
    assert [item["nominalCostUsd"] for item in cost["modelReceipts"]] == [0.000002, 0.000002]
    assert [item["economicCostStatus"] for item in cost["modelReceipts"]] == ["unknown", "unknown"]
