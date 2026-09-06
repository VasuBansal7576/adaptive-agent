from types import SimpleNamespace
import json
import sqlite3
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
    monkeypatch.setenv("ADAPTIVE_AGENT_IMAGE_DIGEST", "sha256:test-image")
    monkeypatch.setenv("ADAPTIVE_AGENT_SOURCE_REVISION", "test-source")

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
    active = runtime.controller.get_active_bundle()
    assert active is not None
    base = runtime.packages["finance"]
    tasks = base.tasks_for_partition(Partition.VALIDATION)
    source_task = tasks[0]
    from adaptive_agent.models import ArtifactRef as DurableRef, EnvironmentManifest as DurableManifest, ToolSchema as DurableSchema
    durable_manifest = DurableManifest(environmentId="appworld", version="1", docs=[DurableRef(id="docs", version="1", sha256="d")], toolSchemas=[DurableSchema(name=t.name, version=t.version, inputSchema=t.input_schema, outputSchema=t.output_schema, effect=t.effect) for t in base.manifest.tool_schemas], policyRef=DurableRef(id="policy", version="1", sha256="p"), evaluatorRef=DurableRef(id="eval", version="1", sha256="e"), resetRef=DurableRef(id="reset", version="1", sha256="r"), executionModes=["batch"])

    class Catalog:
        def split_ids(self, split): return (source_task.task_id,) if split == "test_normal" else ()
        def dataset_hash(self): return "production-dataset"
        def task(self, task_id, split, allow_test=False): return SimpleNamespace(task_id=task_id, instruction=source_task.goal, split=split)

    class Package:
        catalog = Catalog()
        manifest = durable_manifest
        environment_id = "appworld"
        reset = base.reset
        evaluate = base.evaluate

    runtime.packages["appworld"] = base
    runtime.registry.register(Package.manifest)
    protocol = AppWorldProtocol.freeze(Package(), model_profile="openai-codex/gpt-5.6-luna", core_planner_hash=runtime.core_planner_hash, official_split="test_normal", image_digest=runtime.image_digest, source_revision="test-source", dataset_content_hash="production-dataset", published_count=1, seeds=(0,))
    bundles = {Arm.B0: active.content_hash, Arm.L: active.content_hash, Arm.A: active.content_hash}
    adapter = DurableAppWorldAdapter(runtime, protocol, {key.value: value for key, value in bundles.items()})
    runner = AppWorldBenchmarkRunner(tmp_path / "appworld", Package(), protocol, adapter)
    first = runner.run("production", bundles)
    assert first.provenance_complete and first.ablation_audit["passed"]
    assert model.calls == 3
    fresh_app = app_module.create_runtime_app(model_runner=model, evaluator=lambda **_: {"passed": True, "reliable": True}, data_dir=tmp_path)
    fresh_runtime = fresh_app.state.durable_runtime
    fresh_runtime.source_revision = "test-source"
    fresh_runtime.packages["appworld"] = fresh_runtime.packages["finance"]
    fresh_adapter = DurableAppWorldAdapter(fresh_runtime, protocol, {key.value: value for key, value in bundles.items()})
    second = AppWorldBenchmarkRunner(tmp_path / "appworld", Package(), protocol, fresh_adapter).run("production", bundles)
    assert second.provenance_complete and model.calls == 3

    # A cached usage edit is rejected against the fresh verified accounting,
    # before the shared runtime can dispatch another model request.
    with sqlite3.connect(tmp_path / "appworld" / "appworld-benchmark.sqlite3") as conn:
        row = conn.execute("SELECT result_json FROM appworld_cells LIMIT 1").fetchone()
        tampered = json.loads(row[0]); tampered["usage"]["inputTokens"] = 999
        conn.execute("UPDATE appworld_cells SET result_json=?", (json.dumps(tampered),)); conn.commit()
    with pytest.raises(ValueError, match="usage receipt does not reconcile"):
        AppWorldBenchmarkRunner(tmp_path / "appworld", Package(), protocol, fresh_adapter).run("production", bundles)
    assert model.calls == 3

    from adaptive_agent.models import SkillBundle
    learned_payload = {**active.model_dump(mode="json", by_alias=True, exclude={"contentHash"}), "skills": [{"skillId": "learned", "version": "1", "procedure": "arbitrary learned procedure"}]}
    learned_hash = sha256_json(learned_payload)
    learned_payload["contentHash"] = learned_hash
    learned = SkillBundle.model_validate(learned_payload)
    audit_adapter = DurableAppWorldAdapter(fresh_runtime, protocol)
    original_bundle = audit_adapter._bundle
    audit_adapter._bundle = lambda bundle_hash: learned if bundle_hash == learned_hash else original_bundle(bundle_hash)
    retained = dict(bundles); retained[Arm.A] = learned_hash
    with pytest.raises(ValueError, match="retains learned skills"):
        audit_adapter.preflight_bundles({key.value: value for key, value in retained.items()})

    # Planner identity is checked by the production adapter before any task
    # is loaded from the catalog.
    fresh_runtime.core_planner_hash = "mismatched-core"
    with pytest.raises(ValueError, match="core planner hash"):
        DurableAppWorldAdapter(fresh_runtime, protocol, {key.value: value for key, value in bundles.items()}).preflight_bundles({key.value: value for key, value in bundles.items()})
