from types import SimpleNamespace
import json
import sqlite3
import sys
import pytest

from adaptive_agent.appworld_benchmark import AppWorldBenchmarkRunner, AppWorldCellResult, AppWorldProtocol, DurableAppWorldAdapter
from adaptive_agent.evaluation import Arm, ModelProvenance, Partition, RunObservation, sha256_json


class Catalog:
    def split_ids(self, split): return tuple(f"{split}-{index}" for index in range(30))
    def dataset_hash(self): return "dataset-hash"
    def task(self, task_id, split, allow_test=False): return SimpleNamespace(task_id=task_id, split=split, instruction="do it")


class Package:
    catalog = Catalog()
    def provider_factory(self, task, run_id, seed=0): raise AssertionError("runtime owns provider construction")
    def evaluate_provider(self, provider): raise AssertionError("runtime owns evaluation")


class Runtime:
    image_digest = "sha256:test"
    provider = "openai-codex"
    source_revision = "test-revision"
    def __init__(self): self.calls, self.results, self.mapping = [], {}, {}
    def preflight_bundles(self, mapping): self.mapping = dict(mapping)
    def ablation_audit(self): return None
    def run_appworld_cell(self, *, package, task, arm, seed, bundle_hash, budget, run_id):
        self.calls.append(run_id)
        result = AppWorldCellResult(RunObservation(task.task_id, "appworld", Partition.FINAL, seed, arm, arm is not Arm.B0, True, 0, 3, 1.0, model_provenance=ModelProvenance.REAL_MODEL, response_id=run_id, run_id=run_id, bundle_hash=bundle_hash), {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5})
        self.results[run_id] = result
        return result
    def recover_appworld_cell(self, *, run_id, **kwargs): return self.results.get(run_id)
    def verify_appworld_cell(self, result, **kwargs):
        row = result.observation
        return row.task_id == kwargs["task"].task_id and row.arm == kwargs["arm"] and row.seed == kwargs["seed"] and row.bundle_hash == kwargs["bundle_hash"] and row.model_provenance is ModelProvenance.REAL_MODEL


def test_freeze_selects_official_final_subset_without_inspecting_answers():
    protocol = AppWorldProtocol.freeze(Package(), model_profile="model", core_planner_hash="core", official_split="test_normal", image_digest="sha256:test", source_revision="test-revision")
    assert len(protocol.sampled_task_ids) == 20
    assert set(protocol.split_by_task_id) <= {(task, split) for split in ("test_normal", "test_challenge") for task in Package().catalog.split_ids(split)}


def test_runner_is_durable_and_reports_measured_usage(tmp_path):
    protocol = AppWorldProtocol.freeze(Package(), model_profile="model", core_planner_hash="core", official_split="test_normal", image_digest="sha256:test", source_revision="test-revision", published_count=2)
    runtime = Runtime()
    runner = AppWorldBenchmarkRunner(tmp_path, Package(), protocol, runtime)
    report = runner.run("job", {Arm.B0: "b0", Arm.L: "l", Arm.A: "a"})
    assert len(runtime.calls) == 6
    assert report.paired_task_count == 2
    assert report.arm_summaries["B0"]["inputTokens"] == 4
    runner.run("job", {Arm.B0: "b0", Arm.L: "l", Arm.A: "a"})
    assert len(runtime.calls) == 6


def test_production_durable_runtime_adapter_reopens_without_dispatch(tmp_path, monkeypatch):
    import adaptive_agent.app as app_module
    provider_module = pytest.importorskip("adaptive_agent.appworld_provider")
    monkeypatch.setenv("ADAPTIVE_AGENT_IMAGE_DIGEST", "sha256:test-image")
    monkeypatch.setenv("ADAPTIVE_AGENT_SOURCE_REVISION", "test-source")

    def public_root():
        root = tmp_path / "appworld"
        data = root / "data"
        for path in (data / "datasets", data / "api_docs" / "function_calling", data / "api_docs" / "standard", data / "tasks" / "dev-1", data / "base_dbs"):
            path.mkdir(parents=True, exist_ok=True)
        (data / "version.txt").write_text("0.1.0\n"); (data / "LICENSE").write_text("public test fixture\n"); (data / "base_dbs" / "version.txt").write_text("0.1.0\n")
        for split, task_id in (("train", "train-1"), ("dev", "dev-1"), ("test_normal", "test-1"), ("test_challenge", "challenge-1")):
            (data / "datasets" / f"{split}.txt").write_text(task_id + "\n")
        (data / "tasks" / "dev-1" / "specs.json").write_text(json.dumps({"instruction": "read the clock in dev", "allowed_apps": ["phone"], "datetime": "2023-05-18T12:00:00", "db_version": "0.1.0"}))
        (data / "api_docs" / "function_calling" / "phone.json").write_text(json.dumps([{"type": "function", "function": {"name": "phone__get_current_date_and_time", "description": "Read the current date and time.", "parameters": {"type": "object", "properties": {}}}}]))
        (data / "api_docs" / "standard" / "phone.json").write_text(json.dumps({"get_current_date_and_time": {"method": "GET"}})); (data / "base_dbs" / "phone.db").write_bytes(b"fixture")
        return root

    class FakeProcess:
        def __init__(self, *args, **kwargs): self._pid = id(self)
        @property
        def pid(self): return self._pid
        def request(self, operation, payload=None):
            if operation == "reset": return {"taskId": payload["taskId"], "allowedApps": ["phone"]}
            if operation == "evaluate": return {"success": True, "numTests": 1, "passCount": 1, "failCount": 0, "taskCompleted": True}
            if operation == "call": return {"date": "Thursday, May 18, 2023"}
            return {"closed": True}
        def close(self, **kwargs): return None

    monkeypatch.setattr(provider_module, "_JsonLineProcess", FakeProcess)
    config = provider_module.AppWorldConfig(public_root(), python=sys.executable, allow_test=False)
    package = provider_module.AppWorldPackage(config)

    class Model:
        def __init__(self): self.calls = 0
        def invoke(self, *, goal, environment, messages=None, emit=None, **kwargs):
            self.calls += 1
            return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": f"response-{self.calls}", "text": "finish", "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5, "economicCost": {"status": "measured", "microunits": 7}}, "costMicrounits": 7}

    model = Model()
    from adaptive_agent.prime_runtime import ChildPlannerBudget, SharedBudget
    class FakePrime:
        def __init__(self, config, broker):
            self.config, self.broker = config, broker
            self.child_planner = None
            self._budget = SharedBudget(config.max_total_wall_seconds, config.max_total_artifact_bytes, config.max_artifact_count, config.child_runs, config.max_model_tokens)
        @property
        def planner_budget(self): return ChildPlannerBudget(self._budget)
        def record_model_observation(self, evidence, *, trusted_parent=False): return evidence
        def execute(self, code, *, timeout=None, cancel=None):
            namespace = {"host_request": lambda payload: self.broker.call(payload["capabilityId"], payload.get("arguments", {}))}
            exec(code, {"__builtins__": {}}, namespace)
            return SimpleNamespace(status="ok", result=json.dumps(namespace.get("result")), stdout="", stderr="", error=None)
        def close(self, remove_workspace=True): return None
    monkeypatch.setattr(app_module, "PrimeRuntimeAdapter", FakePrime)
    app = app_module.create_runtime_app(model_runner=model, evaluator=lambda **_: {"passed": True, "reliable": True}, data_dir=tmp_path)
    runtime = app.state.durable_runtime
    runtime.source_revision = "test-source"
    runtime.packages["appworld"] = package
    runtime.registry.register(package.manifest)
    active = runtime.controller.get_active_bundle()
    assert active is not None
    protocol = AppWorldProtocol.freeze(package, model_profile="openai-codex/gpt-5.6-luna", core_planner_hash=runtime.core_planner_hash, official_split="dev", image_digest=runtime.image_digest, source_revision="test-source", dataset_content_hash=package.catalog.dataset_hash(), published_count=1, seeds=(0,))
    bundles = {Arm.B0: active.content_hash, Arm.L: active.content_hash, Arm.A: active.content_hash}
    adapter = DurableAppWorldAdapter(runtime, protocol, {key.value: value for key, value in bundles.items()})
    runner = AppWorldBenchmarkRunner(tmp_path / "appworld", package, protocol, adapter)
    first = runner.run("production", bundles)
    assert first.provenance_complete and first.ablation_audit["passed"]
    assert model.calls == 3
    fresh_app = app_module.create_runtime_app(model_runner=model, evaluator=lambda **_: {"passed": True, "reliable": True}, data_dir=tmp_path)
    fresh_runtime = fresh_app.state.durable_runtime
    fresh_runtime.source_revision = "test-source"
    fresh_runtime.packages["appworld"] = package
    fresh_runtime.registry.register(package.manifest)
    fresh_adapter = DurableAppWorldAdapter(fresh_runtime, protocol, {key.value: value for key, value in bundles.items()})
    second = AppWorldBenchmarkRunner(tmp_path / "appworld", package, protocol, fresh_adapter).run("production", bundles)
    assert second.provenance_complete and model.calls == 3

    # A cached usage edit is rejected against the fresh verified accounting,
    # before the shared runtime can dispatch another model request.
    with sqlite3.connect(tmp_path / "appworld" / "appworld-benchmark.sqlite3") as conn:
        row = conn.execute("SELECT result_json FROM appworld_cells LIMIT 1").fetchone()
        tampered = json.loads(row[0]); tampered["usage"]["inputTokens"] = 7; tampered["usage"]["totalTokens"] = 10
        conn.execute("UPDATE appworld_cells SET result_json=?", (json.dumps(tampered),)); conn.commit()
    with pytest.raises(ValueError, match="usage receipt does not reconcile"):
        AppWorldBenchmarkRunner(tmp_path / "appworld", package, protocol, fresh_adapter).run("production", bundles)
    assert model.calls == 3

    from adaptive_agent.models import SkillBundle
    learned_payload = {**active.model_dump(mode="json", by_alias=True, exclude={"content_hash"}), "skills": [{"skillId": "learned", "version": "1", "procedure": "arbitrary learned procedure"}]}
    learned = SkillBundle.model_validate(learned_payload)
    learned_hash = learned.content_hash
    fresh_runtime.controller.store.save_bundle(learned.bundle_id, learned.parent, learned_hash, json.dumps(learned.model_dump(mode="json", by_alias=True)))
    audit_adapter = DurableAppWorldAdapter(fresh_runtime, protocol)
    retained = dict(bundles); retained[Arm.A] = learned_hash
    with pytest.raises(ValueError, match="retains learned skills"):
        audit_adapter.preflight_bundles({key.value: value for key, value in retained.items()})

    # Planner identity is checked by the production adapter before any task
    # is loaded from the catalog.
    fresh_runtime.core_planner_hash = "mismatched-core"
    with pytest.raises(ValueError, match="core planner hash"):
        DurableAppWorldAdapter(fresh_runtime, protocol, {key.value: value for key, value in bundles.items()}).preflight_bundles({key.value: value for key, value in bundles.items()})
