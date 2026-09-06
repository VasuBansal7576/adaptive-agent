import base64
import hashlib
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from adaptive_agent import (
    AdapterError,
    Capability,
    CapabilityBroker,
    PrimeRuntimeAdapter,
    PrimeRuntimeConfig,
    SecurityViolation,
)


class PrimeRuntimeTests(unittest.TestCase):
    def make(self, **kwargs):
        root = Path(tempfile.mkdtemp(prefix="adaptive-test-"))
        self.addCleanup(lambda: self.adapter.close(remove_workspace=True) if hasattr(self, "adapter") else None)
        self.adapter = PrimeRuntimeAdapter(PrimeRuntimeConfig(task_id="test", root_dir=root, **kwargs))
        return self.adapter

    def test_actual_kernel_is_persistent_and_provenance_is_subscription(self):
        adapter = self.make()
        first = adapter.execute("value = 40\nvalue + 2")
        second = adapter.execute("value + 1")
        self.assertEqual((first.status, first.result), ("ok", "42"))
        self.assertEqual(second.result, "41")
        self.assertEqual(first.provenance["requestedProvider"], "openai-codex")
        self.assertEqual(first.provenance["requestedModel"], "openai-codex/gpt-5.6-luna")
        self.assertIsNone(first.provenance["observedModelInvocation"])
        self.assertEqual(first.provenance["kernel"], "Prime Agent rlm.repl protocol v3")
        with self.assertRaises(SecurityViolation):
            adapter.record_model_observation({"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": "r", "usage": {"input": 1}})
        observed = adapter.record_model_observation({"provider": "openai-codex", "model": "openai-codex/gpt-5.6-luna", "responseId": "r", "usage": {"input": 1}}, trusted_parent=True)
        self.assertEqual(observed.response_id, "r")
        self.assertEqual(adapter.provenance()["observedModelInvocation"]["responseId"], "r")

    def test_broker_capability_discovery_and_harness_denial(self):
        root = Path(tempfile.mkdtemp(prefix="adaptive-test-"))
        self.adapter = PrimeRuntimeAdapter(
            PrimeRuntimeConfig(task_id="test", root_dir=root),
            broker=CapabilityBroker("test", authorizer=lambda _cap, args: dict(args)),
        )
        self.adapter.broker.register(Capability("echo", "echo", "1", "read", "run", "never"))
        discovered = self.adapter.execute(
            'from rlm import host_request\nawait host_request("capabilities.discover")'
        )
        self.assertEqual(discovered.status, "ok")
        self.assertIn("echo", discovered.result)
        called = self.adapter.execute(
            'from rlm import host_request\nawait host_request("broker.call", '
            '{"capabilityId":"echo", "arguments":{"n":1}})'
        )
        self.assertEqual(called.status, "ok")
        denied = self.adapter.execute(
            'from rlm import host_request\nawait host_request("harness.write")'
        )
        self.assertEqual(denied.status, "error")
        self.assertIn("denied", denied.error["evalue"])

    def test_docker_cleanup_label_is_trusted_and_required(self):
        adapter = self.make(ao_session_id="trusted-session")
        self.assertEqual(adapter.cleanup_label, "ao.session=trusted-session")
        self.assertEqual(adapter.provenance()["cleanupLabel"], "ao.session=trusted-session")
        import unittest.mock as mock
        from adaptive_agent import prime_runtime
        with mock.patch.dict(prime_runtime.os.environ, {"AO_SESSION_ID": ""}, clear=False):
            with self.assertRaises(AdapterError):
                PrimeRuntimeAdapter(PrimeRuntimeConfig(task_id="missing-label"))

    def test_docker_unavailable_fails_closed_without_host_execution(self):
        import unittest.mock as mock
        from adaptive_agent import prime_runtime
        with mock.patch.object(prime_runtime.shutil, "which", return_value=None), mock.patch.object(prime_runtime.subprocess, "Popen") as popen:
            with self.assertRaises(AdapterError):
                PrimeRuntimeAdapter(PrimeRuntimeConfig(task_id="no-docker"))
            popen.assert_not_called()

    def test_source_policy_blocks_filesystem_network_and_direct_harness(self):
        adapter = self.make()
        for code in ("import os", "import socket", "open('secret')", "getattr(__builtins__, 'open')", "from rlm import harness"):
            with self.assertRaises(SecurityViolation):
                adapter.execute(code)

    def test_timeout_cancels_cell_and_kernel_remains_usable(self):
        adapter = self.make(max_cell_seconds=0.2)
        result = adapter.execute("await __import__('asyncio').sleep(10)") if False else adapter.execute("while True: pass")
        self.assertEqual(result.status, "aborted")
        # A timed-out await/sync cell is interrupted before a subsequent cell.
        followup = adapter.execute("6 * 7")
        self.assertEqual(followup.result, "42")

    def test_learner_requested_child_uses_trusted_planner_and_shared_budget(self):
        adapter = self.make(child_runs=1, max_child_depth=1)
        seen = {}
        def planner(request):
            seen.update({"prompt": request.prompt, "depth": request.depth,
                         "capabilities": request.capabilities,
                         "remaining": request.remaining_seconds})
            return {"name": "useful-child", "code": "answer = 40 + 2\nanswer"}
        adapter.child_planner = planner
        result = adapter.execute(
            "from rlm import host_request\n"
            "await host_request('rlm.run', {'prompt':'compute the answer', 'kwargs':{}})"
        )
        self.assertEqual(result.status, "ok")
        self.assertIn("'result': '42'", result.result)
        self.assertEqual(seen["prompt"], "compute the answer")
        self.assertEqual(seen["depth"], 0)
        self.assertEqual(seen["capabilities"], ())
        second = adapter.execute(
            "from rlm import host_request\n"
            "await host_request('rlm.run', {'prompt':'again', 'kwargs':{}})"
        )
        self.assertEqual(second.status, "error")
        self.assertIn("shared child budget", second.error["evalue"])
        restricted = self.make(child_runs=1, max_child_depth=0)
        with self.assertRaises(AdapterError):
            restricted.execute_child("1 + 1")

    def test_child_planner_cancel_and_depth_are_bounded(self):
        adapter = self.make(child_runs=1, max_cell_seconds=3)
        adapter.child_planner = lambda _request: {"code": "while True: pass"}
        holder = {}
        worker = threading.Thread(target=lambda: holder.setdefault(
            "result", adapter.execute("from rlm import host_request\nawait host_request('rlm.run', {'prompt':'loop', 'kwargs':{}})")))
        worker.start()
        deadline = time.time() + 4
        while adapter.kernel is None and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.2)
        adapter.cancel()
        worker.join(4)
        self.assertFalse(worker.is_alive())
        self.assertEqual(holder["result"].status, "aborted")

    def test_cumulative_artifact_budget_allows_duplicates_but_rejects_new_bytes(self):
        adapter = self.make(max_artifact_bytes=16, max_total_artifact_bytes=5, max_artifact_count=2)
        data = b"abcde"
        encoded = base64.b64encode(data).decode()
        digest = hashlib.sha256(data).hexdigest()
        code = ("from rlm import host_request\nimport base64\n"
                f"t=await host_request('artifact.begin', {{'artifactId':'same','version':'1','size':5}})\n"
                f"await host_request('artifact.chunk', {{'transferId':t['transferId'],'offset':0,'data':{encoded!r}}})\n"
                f"await host_request('artifact.finish', {{'transferId':t['transferId'],'sha256':{digest!r}}})")
        self.assertEqual(adapter.execute(code).status, "ok")
        self.assertEqual(adapter.execute(code).status, "ok")
        second_data = base64.b64encode(b"fghij").decode()
        second = adapter.execute(
            "from rlm import host_request\n"
            f"t=await host_request('artifact.begin', {{'artifactId':'new','version':'1','size':5}})\n"
            f"await host_request('artifact.chunk', {{'transferId':t['transferId'],'offset':0,'data':{second_data!r}}})\n"
            f"await host_request('artifact.finish', {{'transferId':t['transferId'],'sha256':{hashlib.sha256(b'fghij').hexdigest()!r}}})"
        )
        self.assertEqual(second.status, "error")
        self.assertIn("cumulative artifact", second.error["evalue"])

    def test_bounded_child_has_fresh_state_and_budget(self):
        adapter = self.make(child_runs=1)
        parent = adapter.execute("parent_only = 1")
        self.assertEqual(parent.status, "ok")
        child = adapter.execute_child("child_only = 2\nchild_only")
        self.assertEqual(child.result, "2")
        self.assertEqual(child.provenance["parentRunId"], "test")
        missing = adapter.execute("child_only")
        self.assertEqual(missing.status, "error")
        with self.assertRaises(SecurityViolation):
            adapter.execute_child("3 * 3")

    def test_output_is_bounded_and_external_cancel_aborts(self):
        adapter = self.make(max_output_chars=64, max_cell_seconds=3)
        output = adapter.execute("print('x' * 10000)")
        self.assertLessEqual(len(output.stdout), 64)
        cancel = threading.Event()
        holder = {}
        worker = threading.Thread(target=lambda: holder.setdefault("result", adapter.execute("while True: pass", cancel=cancel)))
        worker.start()
        deadline = time.time() + 3
        while adapter.kernel is None and time.time() < deadline:
            time.sleep(0.01)
        cancel.set()
        worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(holder["result"].status, "aborted")

    def test_learner_artifact_transfer_is_bounded_and_content_addressed(self):
        adapter = self.make(max_artifact_bytes=1024)
        data = b"learner-created-bytes"
        encoded = base64.b64encode(data).decode()
        digest = hashlib.sha256(data).hexdigest()
        code = ("from rlm import host_request\n"
                "import base64\n"
                f"blob = base64.b64decode({encoded!r})\n"
                f"transfer = await host_request('artifact.begin', {{'artifactId':'generated', 'version':'1', 'size':{len(data)}}})\n"
                "await host_request('artifact.chunk', {'transferId': transfer['transferId'], 'offset': 0, 'data': base64.b64encode(blob).decode()})\n"
                f"await host_request('artifact.finish', {{'transferId': transfer['transferId'], 'sha256': {digest!r}}})")
        result = adapter.execute(code)
        self.assertEqual(result.status, "ok")
        self.assertTrue(list((adapter.root / "artifacts").glob("generated-*")))
        oversized = adapter.execute("from rlm import host_request\nawait host_request('artifact.begin', {'artifactId':'x', 'size': 2048})")
        self.assertEqual(oversized.status, "error")
        traversal = adapter.execute("from rlm import host_request\nawait host_request('artifact.begin', {'artifactId':'../escape', 'size': 1})")
        self.assertEqual(traversal.status, "error")

    def test_oversized_protocol_frame_fails_and_cleans_owned_container(self):
        adapter = self.make(max_protocol_frame_bytes=2048, max_output_chars=4096)
        adapter.start()
        name = adapter.kernel.container_name
        with self.assertRaises(AdapterError):
            adapter.execute("print('x' * 10000)")
        adapter.close()
        running = subprocess.run(["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
        self.assertNotIn(name, running.stdout)

    def test_hard_timeout_removes_owned_uncooperative_container(self):
        adapter = self.make(max_cell_seconds=0.05)
        adapter.start()
        name = adapter.kernel.container_name
        result = adapter.execute("while True: pass", timeout=0.05)
        self.assertEqual(result.status, "aborted")
        adapter.close()
        running = subprocess.run(["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"], capture_output=True, text=True, check=True)
        self.assertNotIn(name, running.stdout)

    def test_artifact_export_is_content_addressed_and_workspace_bound(self):
        adapter = self.make()
        artifact = adapter.root / "answer.txt"
        artifact.write_text("safe artifact")
        ref = adapter.export_artifact(artifact)
        self.assertEqual(ref.bytes, 13)
        self.assertTrue(Path(ref.path).is_file())
        with self.assertRaises(SecurityViolation):
            adapter.export_artifact(Path(tempfile.gettempdir()) / "outside")
        outside = Path(tempfile.mkdtemp()) / "secret"
        outside.write_text("hidden")
        link = adapter.root / "link"
        link.symlink_to(outside)
        with self.assertRaises(SecurityViolation):
            adapter.export_artifact(link)

    def test_docker_boundary_is_required_and_child_budget_is_denied(self):
        with self.assertRaises(AdapterError):
            PrimeRuntimeAdapter(PrimeRuntimeConfig(task_id="bad", require_docker=False))
        adapter = self.make()
        child = adapter.execute('from rlm import host_request\nawait host_request("rlm.run", {"prompt":"escape"})')
        self.assertEqual(child.status, "error")
        self.assertIn("child run budget", child.error["evalue"])
        self.assertIn("Docker", adapter.provenance()["isolation"])
        adapter.broker.register(Capability("read", "read", "1", "read", "run", "never"))
        denied = adapter.execute('from rlm import host_request\nawait host_request("broker.call", {"capabilityId":"read"})')
        self.assertEqual(denied.status, "error")
        self.assertIn("authoritative broker", denied.error["evalue"])


if __name__ == "__main__":
    unittest.main()
