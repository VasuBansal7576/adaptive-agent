"""Scripted end-to-end coverage for the AppWorld experiment lifecycle."""
from __future__ import annotations

import json
import sqlite3
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
    evaluation_calls = 0
    turns: dict[str, int] = {}
    def __init__(self, **kwargs): pass
    def invoke(self, *, goal, environment, messages=None, **kwargs):
        type(self).calls += 1
        if "learningContext" in environment:
            evidence = environment["learningContext"]["developmentEvidence"][0]["sourceId"]
            procedure = "Reuse the observed date-reading procedure."
            payload = {"predictedEffect": "reuse procedure", "editOperations": [{"path": "skills/appworld/procedure", "operation": "add", "value": procedure}], "supportingEvidenceIds": [evidence], "proposerVersion": "test", "skill": {"procedure": procedure}}
            return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": f"learn-{self.calls}", "text": json.dumps(payload), "usage": {"input": 2, "output": 3, "totalTokens": 5, "cacheRead": 0, "cacheWrite": 0, "cost": {"total": 0.001}}}
        type(self).evaluation_calls += 1
        run_key = environment["capabilities"][0]
        turn = type(self).turns.get(run_key, 0) + 1
        type(self).turns[run_key] = turn
        capability = next(item for item in environment["capabilities"] if item.endswith("appworld__call_read"))
        if turn == 1:
            code = f'result = host_request({json.dumps({"capabilityId": capability, "arguments": {"apiName": "phone__get_current_date_and_time", "arguments": {}}})})'
            text = json.dumps({"action": "execute", "code": code})
        else:
            text = json.dumps({"action": "finish", "answer": "date read"})
        return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": f"eval-{self.calls}", "text": text, "usage": {"input": 2, "output": 3, "totalTokens": 5, "cacheRead": 0, "cacheWrite": 0, "cost": {"total": 0.001}}}


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


def _args(data_dir: Path, root: Path, mode: str):
    return build_parser().parse_args([f"--{mode}", "--data-dir", str(data_dir), "--appworld-root", str(root), "--appworld-python", sys.executable, "--source-revision", "source", "--image-digest", "sha256:image", "--core-planner-hash", "core"])


class _Interrupted(RuntimeError):
    pass


def _reset_model():
    _Model.calls = 0
    _Model.evaluation_calls = 0
    _Model.turns = {}


def test_cli_runs_and_resumes_real_runtime_panels(tmp_path, monkeypatch):
    _reset_model()
    _prime(monkeypatch)
    monkeypatch.setenv("ADAPTIVE_AGENT_SOURCE_REVISION", "source")
    monkeypatch.setenv("ADAPTIVE_AGENT_IMAGE_DIGEST", "sha256:image")
    root = _root(tmp_path)
    from adaptive_agent.appworld_provider import AppWorldConfig, AppWorldError, AppWorldPackage
    with pytest.raises(AppWorldError, match="sealed"):
        AppWorldPackage(AppWorldConfig(root, python=sys.executable)).catalog.task("test_normal-0", "test_normal")
    args = build_parser().parse_args(["--initialize", "--data-dir", str(tmp_path / "run"), "--appworld-root", str(root), "--appworld-python", sys.executable, "--source-revision", "source", "--image-digest", "sha256:image", "--core-planner-hash", "core"])
    monkeypatch.setenv("ADAPTIVE_AGENT_CORE_PLANNER_HASH", "core")
    first = run_experiment(args)
    assert [report["selectedArms"] for report in first["reports"]] == [["B0", "L", "A"], ["B0", "L", "A"]]
    assert all(all(report["armSummaries"][arm]["count"] == 20 for arm in ("B0", "L", "A")) for report in first["reports"])
    for report in first["reports"]:
        for arm in ("B0", "L", "A"):
            summary = report["armSummaries"][arm]
            assert summary["totalTokens"] == summary["inputTokens"] + summary["outputTokens"]
    assert _Model.evaluation_calls == 2 * (8 + 20 * 3 + 20 * 3)
    from adaptive_agent.store import Store
    durable_store = Store(tmp_path / "run")
    first_run = durable_store.get_run(json.loads((tmp_path / "run" / "appworld-experiment-state.json").read_text())["trainingRunIds"][0])
    accounting = durable_store.get_artifact(json.loads(first_run["run_json"])["finalAccountingRef"])
    assert accounting["nominalCostStatus"] == "complete"
    assert accounting.get("economicCostStatus", "unknown") == "unknown"
    experiment_state = json.loads((tmp_path / "run" / "appworld-experiment-state.json").read_text())
    ablation = durable_store.get_bundle_by_hash(experiment_state["ablationBundleHash"])
    ablation_payload = json.loads(ablation["bundle_json"])
    assert ablation["parent"] == experiment_state["learnedBundleHash"]
    assert ablation_payload["skills"] == []
    assert ablation_payload["executionConfig"]["skill_refs"] == []
    assert ablation_payload["executionConfig"]["instruction_variant"] == "default"
    calls = _Model.calls
    state_path = tmp_path / "run" / "appworld-experiment-state.json"
    dev_db = tmp_path / "run" / "dev" / "appworld-benchmark.sqlite3"
    with sqlite3.connect(dev_db) as conn:
        rowid, cached = conn.execute("SELECT rowid,result_json FROM appworld_cells LIMIT 1").fetchone()
        tampered = json.loads(cached)
        tampered["usage"]["inputTokens"] += 1
        conn.execute("UPDATE appworld_cells SET result_json=? WHERE rowid=?", (json.dumps(tampered), rowid))
        conn.commit()
    resume_args = ["--resume", "--data-dir", str(tmp_path / "run"), "--appworld-root", str(root), "--appworld-python", sys.executable, "--source-revision", "source", "--image-digest", "sha256:image", "--core-planner-hash", "core"]
    with pytest.raises(ValueError, match="runtime usage receipt does not reconcile"):
        run_experiment(build_parser().parse_args(resume_args))
    assert _Model.calls == calls
    with sqlite3.connect(dev_db) as conn:
        conn.execute("UPDATE appworld_cells SET result_json=? WHERE rowid=?", (cached, rowid))
        conn.commit()
    resume = build_parser().parse_args(resume_args)
    second = run_experiment(resume)
    assert _Model.calls == calls
    assert len(second["reports"]) == 2
    checkpoint = json.loads(state_path.read_text())
    checkpoint["ablationBundleHash"] = "tampered"
    state_path.write_text(json.dumps(checkpoint))
    with pytest.raises(RuntimeError, match="deterministic L-derived ablation"):
        run_experiment(resume)
    assert _Model.calls == calls
    checkpoint["ablationBundleHash"] = experiment_state["ablationBundleHash"]
    state_path.write_text(json.dumps(checkpoint))
    checkpoint = json.loads(state_path.read_text())
    checkpoint["learningStatus"] = "in_flight"
    state_path.write_text(json.dumps(checkpoint))
    with pytest.raises(RuntimeError, match="authenticated recoverable receipt"):
        run_experiment(resume)
    assert _Model.calls == calls
    state_path.write_text(json.dumps({**checkpoint, "learningStatus": "complete"}))
    bad = build_parser().parse_args(["--resume", "--data-dir", str(tmp_path / "run"), "--appworld-root", str(root), "--appworld-python", sys.executable, "--source-revision", "changed", "--image-digest", "sha256:image", "--core-planner-hash", "core"])
    with pytest.raises(RuntimeError, match="SOURCE_REVISION"):
        run_experiment(bad)
    assert _Model.calls == calls
    state = json.loads(state_path.read_text())
    state["trainingStatus"] = "failed"
    state_path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="budget-expanding retry"):
        run_experiment(resume)
    assert _Model.calls == calls


def test_cli_recovers_after_learning_checkpoint_without_redispatch(tmp_path, monkeypatch):
    _reset_model()
    _prime(monkeypatch)
    monkeypatch.setenv("ADAPTIVE_AGENT_SOURCE_REVISION", "source")
    monkeypatch.setenv("ADAPTIVE_AGENT_IMAGE_DIGEST", "sha256:image")
    monkeypatch.setenv("ADAPTIVE_AGENT_CORE_PLANNER_HASH", "core")
    root = _root(tmp_path)
    data_dir = tmp_path / "learning-interrupted"

    def stop_after_learning(event):
        if event["stage"] == "learning" and event["status"] == "complete":
            raise _Interrupted("after learning checkpoint")

    with pytest.raises(_Interrupted):
        run_experiment(_args(data_dir, root, "initialize"), progress=stop_after_learning)
    calls = _Model.calls
    assert json.loads((data_dir / "appworld-experiment-state.json").read_text())["learningStatus"] == "complete"
    result = run_experiment(_args(data_dir, root, "resume"))
    assert len(result["reports"]) == 2
    assert _Model.calls == 257
    assert _Model.calls > calls


def test_cli_recovers_after_one_new_dev_cell_without_repeating_finished_work(tmp_path, monkeypatch):
    _reset_model()
    _prime(monkeypatch)
    monkeypatch.setenv("ADAPTIVE_AGENT_SOURCE_REVISION", "source")
    monkeypatch.setenv("ADAPTIVE_AGENT_IMAGE_DIGEST", "sha256:image")
    monkeypatch.setenv("ADAPTIVE_AGENT_CORE_PLANNER_HASH", "core")
    root = _root(tmp_path)
    data_dir = tmp_path / "dev-interrupted"
    import adaptive_agent.appworld_experiment as experiment_module
    original_factory = experiment_module.create_appworld_benchmark_runner
    interrupted = {"done": False}

    def interrupting_factory(*args, **kwargs):
        runner = original_factory(*args, **kwargs)
        if runner.protocol.official_split == "dev":
            original_cell = runner.runtime.run_appworld_cell

            def one_cell_then_stop(**cell_kwargs):
                result = original_cell(**cell_kwargs)
                if not interrupted["done"]:
                    interrupted["done"] = True
                    raise _Interrupted("after one new dev cell")
                return result

            runner.runtime.run_appworld_cell = one_cell_then_stop
        return runner

    monkeypatch.setattr(experiment_module, "create_appworld_benchmark_runner", interrupting_factory)
    with pytest.raises(_Interrupted):
        run_experiment(_args(data_dir, root, "initialize"))
    monkeypatch.setattr(experiment_module, "create_appworld_benchmark_runner", original_factory)
    run_experiment(_args(data_dir, root, "resume"))
    assert _Model.calls == 257
