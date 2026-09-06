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
    from tests import test_appworld_provider as provider_tests
    provider_module = __import__("adaptive_agent.appworld_provider", fromlist=["AppWorldPackage"])
    monkeypatch.setenv("ADAPTIVE_AGENT_IMAGE_DIGEST", "sha256:test-image")
    monkeypatch.setenv("ADAPTIVE_AGENT_SOURCE_REVISION", "test-source")

    public_root = provider_tests._public_root(tmp_path)
    worker_operations = []

    class FakeProcess:
        def __init__(self, *args, **kwargs): self._pid = id(self)
        @property
        def pid(self): return self._pid
        def request(self, operation, payload=None):
            worker_operations.append(operation)
            if operation == "reset": return {"taskId": payload["taskId"], "allowedApps": ["phone"]}
            if operation == "evaluate": return {"success": True, "numTests": 1, "passCount": 1, "failCount": 0, "taskCompleted": True}
            if operation == "call": return {"date": "Thursday, May 18, 2023"}
            return {"closed": True}
        def close(self, **kwargs): return None

    monkeypatch.setattr(provider_module, "_JsonLineProcess", FakeProcess)
    config = provider_module.AppWorldConfig(public_root, python=sys.executable, allow_test=False)
    package = provider_module.AppWorldPackage(config)
    assert package.task_provenance("dev-1")["officialSplit"] == "dev"

    class Model:
        def __init__(self): self.calls = 0
        def invoke(self, *, goal, environment, messages=None, emit=None, **kwargs):
            self.calls += 1
            run_capability = next(capability for capability in environment["capabilities"] if capability.endswith("appworld__call_read"))
            if self.calls % 2:
                code = f'result = host_request({json.dumps({"capabilityId": run_capability, "arguments": {"apiName": "phone__get_current_date_and_time", "arguments": {}}})})'
                text = json.dumps({"action": "execute", "code": code})
            else:
                text = json.dumps({"action": "finish", "answer": "clock read"})
            return {"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": f"response-{self.calls}", "text": text, "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5, "economicCost": {"status": "measured", "microunits": 7}}, "costMicrounits": 7}

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
    app = app_module.create_runtime_app(model_runner=model, evaluator=None, data_dir=tmp_path)
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
    assert model.calls == 6 and worker_operations.count("reset") == 3 and worker_operations.count("call") == 3 and worker_operations.count("evaluate") == 3
    with runtime.controller.store.connect() as conn:
        assert any("aggregateEvaluation" in row[0] for row in conn.execute("SELECT metadata_json FROM outcomes"))
    fresh_app = app_module.create_runtime_app(model_runner=model, evaluator=None, data_dir=tmp_path)
    fresh_runtime = fresh_app.state.durable_runtime
    fresh_runtime.source_revision = "test-source"
    fresh_runtime.packages["appworld"] = package
    fresh_runtime.registry.register(package.manifest)
    fresh_adapter = DurableAppWorldAdapter(fresh_runtime, protocol, {key.value: value for key, value in bundles.items()})
    second = AppWorldBenchmarkRunner(tmp_path / "appworld", package, protocol, fresh_adapter).run("production", bundles)
    assert second.provenance_complete and model.calls == 6 and worker_operations.count("call") == 3

    # A cached usage edit is rejected against the fresh verified accounting,
    # before the shared runtime can dispatch another model request.
    with sqlite3.connect(tmp_path / "appworld" / "appworld-benchmark.sqlite3") as conn:
        row = conn.execute("SELECT result_json FROM appworld_cells LIMIT 1").fetchone()
        tampered = json.loads(row[0]); tampered["usage"]["inputTokens"] = 7; tampered["usage"]["totalTokens"] = 10
        conn.execute("UPDATE appworld_cells SET result_json=?", (json.dumps(tampered),)); conn.commit()
    with pytest.raises(ValueError, match="persisted AppWorld receipt failed verification"):
        AppWorldBenchmarkRunner(tmp_path / "appworld", package, protocol, fresh_adapter).run("production", bundles)
    assert model.calls == 6

    from adaptive_agent.models import SkillBundle
    learned_payload = {**active.model_dump(mode="json", by_alias=True, exclude={"content_hash"}), "skills": [{"skillId": "learned", "version": "1", "procedure": "arbitrary learned procedure"}]}
    learned = SkillBundle.model_validate(learned_payload)
    learned_hash = learned.content_hash
    fresh_runtime.controller.store.save_bundle(learned.bundle_id, learned.parent, learned_hash, json.dumps(learned.model_dump(mode="json", by_alias=True)))
    assert fresh_runtime.controller.store.get_bundle_by_hash(learned_hash)["content_hash"] == learned_hash
    audit_adapter = DurableAppWorldAdapter(fresh_runtime, protocol)
    assert audit_adapter._bundle(learned_hash).content_hash == learned_hash
    retained = {"B0": learned_hash, "L": learned_hash, "A": learned_hash}
    with pytest.raises(ValueError, match="retains learned skills"):
        audit_adapter.preflight_bundles(retained)

    # Planner identity is checked by the production adapter before any task
    # is loaded from the catalog.
    fresh_runtime.core_planner_hash = "mismatched-core"
    with pytest.raises(ValueError, match="core planner hash"):
        DurableAppWorldAdapter(fresh_runtime, protocol, {key.value: value for key, value in bundles.items()}).preflight_bundles({key.value: value for key, value in bundles.items()})
