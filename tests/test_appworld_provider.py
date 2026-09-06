from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from adaptive_agent.appworld_provider import (
    AppWorldCatalog,
    AppWorldConfig,
    AppWorldError,
    AppWorldPackage,
    AppWorldProtocolError,
    AppWorldUnavailable,
    AppWorldProvider,
    _JsonLineProcess,
    build_manifest,
    public_tool_schemas,
    register_appworld,
)
from adaptive_agent.broker import Capability, ToolBroker
from adaptive_agent.evaluation import Partition
from adaptive_agent.environment import EnvironmentRegistry
from adaptive_agent.models import ToolRequest
from adaptive_agent.store import Store


def _public_root(tmp_path: Path) -> Path:
    root = tmp_path / "appworld"
    data = root / "data"
    (data / "datasets").mkdir(parents=True)
    (data / "api_docs" / "function_calling").mkdir(parents=True)
    (data / "api_docs" / "standard").mkdir(parents=True)
    (data / "tasks" / "train-1").mkdir(parents=True)
    (data / "base_dbs").mkdir(parents=True)
    (data / "version.txt").write_text("0.1.0\n")
    (data / "LICENSE").write_text("public test fixture\n")
    (data / "base_dbs" / "version.txt").write_text("0.1.0\n")
    for split in ("train", "dev", "test_normal", "test_challenge"):
        (data / "datasets" / f"{split}.txt").write_text({"train": "train-1\n", "dev": "dev-1\n", "test_normal": "test-1\n", "test_challenge": "challenge-1\n"}[split])
    (data / "tasks" / "train-1" / "specs.json").write_text(json.dumps({"instruction": "read the clock", "allowed_apps": ["phone"], "datetime": "2023-05-18T12:00:00", "db_version": "0.1.0"}))
    (data / "tasks" / "dev-1").mkdir(parents=True)
    (data / "tasks" / "dev-1" / "specs.json").write_text(json.dumps({"instruction": "read the clock in dev", "allowed_apps": ["phone"], "datetime": "2023-05-18T12:00:00", "db_version": "0.1.0"}))
    (data / "api_docs" / "function_calling" / "phone.json").write_text(json.dumps([{"type": "function", "function": {"name": "phone__get_current_date_and_time", "description": "Read the current date and time.", "parameters": {"type": "object", "properties": {}}}}]))
    (data / "api_docs" / "standard" / "phone.json").write_text(json.dumps({"get_current_date_and_time": {"method": "GET"}}))
    (data / "base_dbs" / "phone.db").write_bytes(b"fixture")
    return root


def test_catalog_reads_exact_public_split_ids_and_seals_test(tmp_path: Path):
    root = _public_root(tmp_path)
    catalog = AppWorldCatalog(AppWorldConfig(root, python=sys.executable))
    assert catalog.split_ids("train") == ("train-1",)
    assert catalog.runtime_manifest().split_counts == {"train": 1, "dev": 1, "test_normal": 1, "test_challenge": 1}
    assert catalog.task("train-1", "train").instruction == "read the clock"
    with pytest.raises(AppWorldError, match="sealed"):
        catalog.task("test-1", "test_normal")


def test_package_maps_official_splits_to_runtime_partitions(tmp_path: Path):
    package = AppWorldPackage(AppWorldConfig(_public_root(tmp_path), python=sys.executable))
    assert {task.split for task in package.learner_tasks()} == {"train"}
    assert {task.family for task in package.tasks_for_partition(Partition.DEVELOPMENT)} == {"appworld:train"}
    assert {task.family for task in package.tasks_for_partition(Partition.VALIDATION)} == {"appworld:dev"}
    assert package.tasks_for_partition(Partition.FINAL) == ()


def test_public_hash_excludes_ground_truth(tmp_path: Path):
    root = _public_root(tmp_path)
    catalog = AppWorldCatalog(AppWorldConfig(root, python=sys.executable))
    ground_truth = root / "data" / "tasks" / "train-1" / "ground_truth"
    ground_truth.mkdir()
    (ground_truth / "private.json").write_text("one")
    first = catalog.public_data_hash()
    (ground_truth / "private.json").write_text("two")
    assert catalog.public_data_hash() == first


def test_manifest_contains_public_tool_schemas(tmp_path: Path):
    root = _public_root(tmp_path)
    manifest = build_manifest(AppWorldConfig(root, python=sys.executable))
    assert manifest.environment_id == "appworld"
    assert [schema.name for schema in manifest.tool_schemas] == [
        "appworld__search_api_docs", "appworld__get_api_doc", "appworld__call_read", "appworld__call_write"
    ]
    assert manifest.tool_schemas[2].effect == "read"


def test_missing_authoritative_method_fails_closed(tmp_path: Path):
    root = _public_root(tmp_path)
    (root / "data" / "api_docs" / "standard" / "phone.json").write_text(json.dumps({"get_current_date_and_time": {}}))
    with pytest.raises(AppWorldUnavailable, match="authoritative HTTP method"):
        public_tool_schemas(AppWorldConfig(root, python=sys.executable))


def test_register_appworld_defaults_to_train_and_dev(tmp_path: Path):
    root = _public_root(tmp_path)
    store = Store(tmp_path / "store")
    registry = EnvironmentRegistry(store)
    manifest_ref, task_refs = register_appworld(registry, AppWorldConfig(root, python=sys.executable))
    assert manifest_ref.sha256
    assert len(task_refs) == 2
    assert registry.list_tasks_by_partition("appworld", "development")[0].task_id == "train-1"
    assert registry.list_tasks_by_partition("appworld", "validation")[0].task_id == "dev-1"


def test_provider_uses_one_worker_process_and_brokerable_result(tmp_path: Path):
    root = _public_root(tmp_path)
    worker = tmp_path / "worker.py"
    worker.write_text(
        '''import json, sys
for line in sys.stdin:
    f = json.loads(line)
    op = f["operation"]
    p = f.get("payload", {})
    if op == "reset":
        r = {"taskId": p["taskId"]}
    elif op == "call":
        r = {"date": "Thursday, May 18, 2023"}
    elif op == "evaluate":
        r = {"success": True, "numTests": 1, "passCount": 1, "failCount": 0, "taskCompleted": True}
    elif op == "close":
        print(json.dumps({"id": f["id"], "ok": True, "result": {"closed": True}}), flush=True)
        break
    else:
        raise RuntimeError(op)
    print(json.dumps({"id": f["id"], "ok": True, "result": r}), flush=True)
'''
    )
    config = AppWorldConfig(root, python=sys.executable)
    task = AppWorldCatalog(config).task("train-1", "train")
    command = [sys.executable, "-u", str(worker)]
    with AppWorldProvider(config, task, "run-1", worker_command=command) as provider:
        first_pid = provider.process_id
        result = provider.execute("run-1", "appworld__call_read", {"apiName": "phone__get_current_date_and_time", "arguments": {}})
        assert result.output == {"date": "Thursday, May 18, 2023"}
        assert provider.effect("appworld__call_read") == "read"
        assert provider.evaluate_aggregate()["success"] is True
    with AppWorldProvider(config, task, "run-2", worker_command=command) as second:
        assert second.process_id != first_pid


def test_hung_worker_timeout_is_bounded_and_cleaned_up(tmp_path: Path):
    worker = tmp_path / "hung_worker.py"
    worker.write_text("import time\ntime.sleep(10)\n")
    process = _JsonLineProcess([sys.executable, "-u", str(worker)], os.environ, 0.05)
    with pytest.raises(AppWorldProtocolError, match="timed out"):
        process.request("reset", {"taskId": "train-1"})
    assert process._proc.poll() is not None
    process.close(force=True)


def test_worker_closed_stdout_then_hung_is_cleaned_up(tmp_path: Path):
    worker = tmp_path / "closed_stdout_worker.py"
    worker.write_text("import os, time\nos.close(1)\ntime.sleep(10)\n")
    process = _JsonLineProcess([sys.executable, "-u", str(worker)], os.environ, 1.0)
    with pytest.raises(AppWorldProtocolError, match="closed"):
        process.request("reset", {"taskId": "train-1"})
    assert process._proc.poll() is not None
    process.close(force=True)


def test_provider_rejects_wrong_run(tmp_path: Path):
    root = _public_root(tmp_path)
    worker = tmp_path / "worker.py"
    worker.write_text("import json,sys\nfor line in sys.stdin:\n f=json.loads(line); print(json.dumps({'id':f['id'],'ok':True,'result':{}}),flush=True)\n")
    task = AppWorldCatalog(AppWorldConfig(root, python=sys.executable)).task("train-1", "train")
    with AppWorldProvider(AppWorldConfig(root, python=sys.executable), task, "run-1", worker_command=[sys.executable, "-u", str(worker)]) as provider:
        with pytest.raises(AppWorldError, match="different run"):
            provider.execute("run-2", "appworld__call_read", {"apiName": "phone__get_current_date_and_time", "arguments": {}})


def test_appworld_call_crosses_authoritative_broker(tmp_path: Path):
    root = _public_root(tmp_path)
    config = AppWorldConfig(root, python=sys.executable)
    task = AppWorldCatalog(config).task("train-1", "train")
    worker = tmp_path / "worker.py"
    worker.write_text(
        'import json,sys\n'
        'for line in sys.stdin:\n'
        ' f=json.loads(line); op=f["operation"]\n'
        ' r={"taskId":"train-1"} if op=="reset" else ({"date":"ok"} if op=="call" else ({"closed":True} if op=="close" else {}))\n'
        ' print(json.dumps({"id":f["id"],"ok":True,"result":r}),flush=True)\n'
        ' if op=="close": break\n'
    )
    store = Store(tmp_path / "store")
    registry = EnvironmentRegistry(store)
    registry.register(build_manifest(config))
    broker = ToolBroker(store, registry)
    with AppWorldProvider(config, task, "run-1", worker_command=[sys.executable, "-u", str(worker)]) as provider:
        result = broker.request_tool_call(
            "appworld",
            ToolRequest(runId="run-1", stepId="step-1", tool="appworld__call_read", arguments={"apiName": "phone__get_current_date_and_time", "arguments": {}}, idempotencyKey="read-1"),
            Capability("run-1", "appworld", "appworld__call_read", "read"),
            provider,
        )
        assert result.status == "ok"
        assert result.output == {"date": "ok"}
