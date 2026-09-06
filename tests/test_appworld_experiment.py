"""Scripted end-to-end coverage for the AppWorld experiment lifecycle."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

from adaptive_agent.appworld_experiment import build_parser, run_experiment


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "appworld"
    data = root / "data"
    for directory in (data / "datasets", data / "api_docs" / "function_calling", data / "api_docs" / "standard", data / "base_dbs"):
        directory.mkdir(parents=True)
    (data / "version.txt").write_text("0.1.3.post1\n")
    (data / "LICENSE").write_text("fixture\n")
    (data / "base_dbs" / "version.txt").write_text("1\n")
    (data / "api_docs" / "function_calling" / "phone.json").write_text(json.dumps([{"type": "function", "function": {"name": "phone__get_current_date_and_time", "description": "Read the current date.", "parameters": {"type": "object", "properties": {}}}}]))
    (data / "api_docs" / "standard" / "phone.json").write_text(json.dumps({"get_current_date_and_time": {"method": "GET"}}))
    (data / "base_dbs" / "phone.db").write_bytes(b"fixture")
    for split, count in (("train", 8), ("dev", 20), ("test_normal", 20), ("test_challenge", 20)):
        ids = [f"{split}-{i}" for i in range(count)]
        (data / "datasets" / f"{split}.txt").write_text("\n".join(ids) + "\n")
        for task_id in ids:
            task_dir = data / "tasks" / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "specs.json").write_text(json.dumps({"instruction": "read the date", "allowed_apps": ["phone"], "datetime": "2023-05-18T12:00:00", "db_version": "1"}))
    return root


class _Process:
    calls = 0
    def __init__(self, *args, **kwargs): self._pid = 1
    @property
    def pid(self): return self._pid
    def request(self, operation, payload=None):
        type(self).calls += 1
        if operation == "reset": return {"taskId": payload["taskId"], "allowedApps": ["phone"]}
        if operation == "evaluate": return {"success": True, "numTests": 1, "passCount": 1, "failCount": 0, "taskCompleted": True}
        if operation == "call": return {"date": "Thursday, May 18, 2023"}
        return {"closed": True}
    def close(self, **kwargs): return None


class _Model:
    calls = 0
    def __init__(self, **kwargs): pass
    def invoke(self, *, goal, environment, messages=None, **kwargs):
        type(self).calls += 1
        if "learningContext" in environment:
            evidence = environment["learningContext"]["developmentEvidence"][0]["sourceId"]
            procedure = "Reuse the observed date-reading procedure."
            payload = {"predictedEffect": "reuse procedure", "editOperations": [{"path": "skills/appworld/procedure", "operation": "add", "value": procedure}], "supportingEvidenceIds": [evidence], "proposerVersion": "test", "skill": {"procedure": procedure}}
            return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": f"learn-{self.calls}", "text": json.dumps(payload), "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5}}
        capability = next(item for item in environment["capabilities"] if item.endswith("appworld__call_read"))
        if self.calls % 2:
            code = f'result = host_request({json.dumps({"capabilityId": capability, "arguments": {"apiName": "phone__get_current_date_and_time", "arguments": {}}})})'
            text = json.dumps({"action": "execute", "code": code})
        else:
            text = json.dumps({"action": "finish", "answer": "date read"})
        return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": f"eval-{self.calls}", "text": text, "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5, "economicCost": {"status": "measured", "microunits": 1}}, "costMicrounits": 1}


def _prime(monkeypatch):
    import adaptive_agent.appworld_experiment as experiment_module
    import adaptive_agent.app as app_module
    import adaptive_agent.appworld_provider as provider_module
    from adaptive_agent.prime_runtime import ChildPlannerBudget, SharedBudget

    class Prime:
        def __init__(self, config, broker):
            self.broker = broker
            self._budget = SharedBudget(config.max_total_wall_seconds, config.max_total_artifact_bytes, config.max_artifact_count, config.child_runs, config.max_model_tokens)
            self.child_planner = None
        @property
        def planner_budget(self): return ChildPlannerBudget(self._budget)
        def record_model_observation(self, evidence, *, trusted_parent=False): return evidence
        def execute(self, code, *, timeout=None, cancel=None):
            namespace = {"host_request": lambda payload: self.broker.call(payload["capabilityId"], payload.get("arguments", {}))}
            exec(code, {"__builtins__": {}}, namespace)
            return SimpleNamespace(status="ok", result=json.dumps(namespace.get("result")), stdout="", stderr="", error=None)
        def close(self, remove_workspace=True): return None
    monkeypatch.setattr(provider_module, "_JsonLineProcess", _Process)
    monkeypatch.setattr(app_module, "PrimeRuntimeAdapter", Prime)
    monkeypatch.setattr(experiment_module, "PrimeCliModelClient", _Model)


def test_cli_runs_and_resumes_real_runtime_panels(tmp_path, monkeypatch):
    _prime(monkeypatch)
    monkeypatch.setenv("ADAPTIVE_AGENT_SOURCE_REVISION", "source")
    monkeypatch.setenv("ADAPTIVE_AGENT_IMAGE_DIGEST", "sha256:image")
    root = _root(tmp_path)
    args = build_parser().parse_args(["--initialize", "--data-dir", str(tmp_path / "run"), "--appworld-root", str(root), "--appworld-python", sys.executable, "--source-revision", "source", "--image-digest", "sha256:image", "--core-planner-hash", "core"])
    monkeypatch.setenv("ADAPTIVE_AGENT_CORE_PLANNER_HASH", "core")
    first = run_experiment(args)
    assert [report["armSummaries"]["B0"]["count"] for report in first["reports"]] == [20, 20]
    calls = _Model.calls
    resume = build_parser().parse_args(["--resume", "--data-dir", str(tmp_path / "run"), "--appworld-root", str(root), "--appworld-python", sys.executable, "--source-revision", "source", "--image-digest", "sha256:image", "--core-planner-hash", "core"])
    second = run_experiment(resume)
    assert _Model.calls == calls
    assert len(second["reports"]) == 2
    bad = build_parser().parse_args(["--resume", "--data-dir", str(tmp_path / "run"), "--appworld-root", str(root), "--appworld-python", sys.executable, "--source-revision", "changed", "--image-digest", "sha256:image", "--core-planner-hash", "core"])
    with pytest.raises(RuntimeError, match="SOURCE_REVISION"):
        run_experiment(bad)
    assert _Model.calls == calls
    state_path = tmp_path / "run" / "appworld-experiment-state.json"
    state = json.loads(state_path.read_text())
    state["trainingStatus"] = "failed"
    state_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="budget-expanding retry"):
        run_experiment(resume)
    assert _Model.calls == calls
