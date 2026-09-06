from __future__ import annotations

import json

from adaptive_agent.controller import Controller
from adaptive_agent.environment import EnvironmentRegistry
from adaptive_agent.models import ArtifactRef
from adaptive_agent.safety_probe import PROBE_CASES, run_eval_003, run_prime_runtime_safety_probe
from adaptive_agent.store import Store


def test_eval_003_runs_all_cases_in_an_explicit_isolated_store(tmp_path):
    probe_dir = tmp_path / "probe"
    result = run_eval_003(probe_dir, require_runtime=False)

    assert result["caseId"] == "EVAL-003"
    assert result["passed"] is True
    assert result["provider"]["simulated"] is True
    assert result["provider"]["name"] == "simulated-test-provider"
    cases = result["detail"]["cases"]
    assert [case["case"] for case in cases] == list(PROBE_CASES)
    assert all(case["passed"] for case in cases)
    assert len(result["evidence"]) == len(PROBE_CASES)
    assert len(result["evidenceReceipts"]) == len(PROBE_CASES)
    assert all("contentHash" in receipt and "payload" in receipt for receipt in result["evidenceReceipts"])

    store = Store(probe_dir)
    evidence = store.list_evidence(result["detail"]["runId"])
    assert len(evidence) == len(PROBE_CASES) + 1  # run_created plus one receipt per case
    encoded = json.dumps(
        [store.get_artifact(ArtifactRef.model_validate_json(row["source_ref"])) for row in evidence]
    )
    assert "probe-hidden-answer" not in encoded


def test_controller_eval_003_does_not_write_controller_store(tmp_path):
    store = Store(tmp_path / "controller")
    controller = Controller(store, EnvironmentRegistry(store))

    result = controller.execute_probe("EVAL-003")

    assert result["passed"] is False
    assert result["provider"]["runtimeRequired"] is True
    assert result["provider"]["runtime"]["executed"] is False
    assert store.list_environments() == []
    assert list((tmp_path / "controller").glob("**/*"))


class _NonDockerRuntime:
    def execute(self, code):
        class Result:
            status = "error"
            provenance = {"isolation": "test double"}

        return Result()


def test_runtime_boundary_requires_actual_docker_provenance():
    result = run_prime_runtime_safety_probe(_NonDockerRuntime(), "public injection")

    assert result["executed"] is True
    assert result["actualDocker"] is False
    assert all(case["classification"] == "missing_security_violation" for case in result["cases"].values())
    assert not all(case["passed"] for case in result["cases"].values())


class _DockerSecurityRuntime:
    def execute(self, code):
        class Result:
            status = "error"
            error = {"ename": "SecurityViolation", "evalue": "redacted"}
            provenance = {"isolation": "per-run Docker container"}

        return Result()


def test_runtime_boundary_classifies_expected_security_violations():
    result = run_prime_runtime_safety_probe(_DockerSecurityRuntime(), "public injection")

    assert result["actualDocker"] is True
    assert all(case["classification"] == "expected_security_violation" for case in result["cases"].values())
    assert all(case["passed"] for case in result["cases"].values())


class SecurityViolation(Exception):
    pass


class _DockerSourcePolicyRuntime:
    def provenance(self):
        return {"isolation": "per-run Docker container"}

    def execute(self, code):
        if code == "open('probe-secret')":
            raise SecurityViolation("filesystem access denied by source policy")

        class Result:
            status = "error"
            error = {"ename": "SecurityViolation", "evalue": "redacted"}
            provenance = {"isolation": "per-run Docker container"}

        return Result()


def test_filesystem_source_policy_security_violation_is_expected():
    result = run_prime_runtime_safety_probe(_DockerSourcePolicyRuntime(), "public injection")

    filesystem = result["cases"]["filesystem_access"]
    assert filesystem["passed"] is True
    assert filesystem["classification"] == "expected_security_violation"
    assert filesystem["denialStage"] == "source_policy"
    assert result["actualDocker"] is True
