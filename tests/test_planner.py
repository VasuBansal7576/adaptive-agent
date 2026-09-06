from dataclasses import dataclass
import stat
import time
from threading import Event

import pytest

from adaptive_agent.planner import LunaPlanner, PlannerError, PlannerLimits, PlannerTimedOut, PrimeCliModelClient, make_luna_model_runner


@dataclass
class KernelResult:
    status: str = "ok"
    result: str = "broker-result"
    stdout: str = ""
    stderr: str = ""
    error: dict | None = None


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def invoke(self, *, goal, environment, messages):
        self.messages.append((goal, environment, list(messages)))
        return next(self.responses)


class Kernel:
    def __init__(self):
        self.code = []

    def execute(self, code, *, timeout=None, cancel=None):
        self.code.append(code)
        return KernelResult(stdout="token=secret-value")


class Sink:
    def __init__(self):
        self.observations = []

    def record_model_observation(self, evidence, *, trusted_parent=False):
        self.observations.append((evidence, trusted_parent))


def response(response_id, text):
    return {
        "provider": "openai-codex",
        "model": "openai-codex/gpt-5.6-luna",
        "responseId": response_id,
        "text": text,
        "usage": {"outputTokens": 4},
    }


def test_generic_model_python_loop_round_trips_sanitized_feedback_and_evidence():
    client = Client([
        response("r1", '{"action":"execute","code":"from rlm import host_request\\nresult = host_request({\\"type\\": \\"broker.call\\"})"}'),
        response("r2", '{"action":"finish","answer":"complete"}'),
    ])
    kernel, sink = Kernel(), Sink()
    result = LunaPlanner(client, kernel, sink).run(
        goal="complete the supplied task",
        environment={"toolSchemas": [{"name": "counter.read"}], "credentials": {"token": "hidden"}},
        active_skills=({"id": "skill-1", "instructions": "read first"},),
    )
    assert result.status == "succeeded"
    assert result.answer == "complete"
    assert result.kernel_steps == 1
    assert kernel.code and "host_request" in kernel.code[0]
    assert len(sink.observations) == 2
    assert all(trusted for _, trusted in sink.observations)
    feedback = client.messages[1][2][-1]["content"]
    assert "broker-result" in feedback
    assert "token=[REDACTED]" in feedback
    assert "secret-value" not in feedback


def test_model_identity_and_action_shape_are_strict():
    client = Client([{**response("r1", '{"action":"finish","answer":"done"}'), "model": "fixture"}])
    with pytest.raises(PlannerError, match="authenticated model response"):
        LunaPlanner(client, Kernel(), Sink()).run(goal="goal", environment={})

    malformed = Client([response("r1", "```python\nprint(1)\n```")])
    with pytest.raises(PlannerError, match="JSON action"):
        LunaPlanner(malformed, Kernel(), Sink()).run(goal="goal", environment={})


def test_turn_and_cancellation_boundaries_stop_before_another_model_call():
    client = Client([response("r1", '{"action":"execute","code":"1 + 1"}')])
    limits = PlannerLimits(max_turns=1)
    result = LunaPlanner(client, Kernel(), Sink(), limits=limits).run(goal="goal", environment={})
    assert result.status == "budget_exhausted"
    assert len(client.messages) == 1

    client = Client([response("r1", '{"action":"finish","answer":"done"}')])
    cancel = Event()
    cancel.set()
    result = LunaPlanner(client, Kernel(), Sink()).run(goal="goal", environment={}, cancel=cancel)
    assert result.status == "cancelled"
    assert not client.messages


def test_control_plane_runner_adapter_returns_authenticated_final_invocation():
    client = Client([response("r1", '{"action":"finish","answer":"done"}')])
    events = []
    invocation = make_luna_model_runner(client, Kernel(), Sink())(
        goal="goal", environment={"toolSchemas": []}, emit=lambda *event: events.append(event)
    )
    assert invocation.text == "done"
    assert invocation.provider == "openai-codex"
    assert invocation.response_id == "r1"
    assert events[-1][0] == "status"


def test_prime_cli_client_parses_json_lines_and_normalizes_bare_model(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOULD_NOT_COPY", "secret")
    executable = tmp_path / "prime-agent"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os\n"
        "if 'SHOULD_NOT_COPY' in os.environ: raise SystemExit(17)\n"
        "print(json.dumps({'type': 'message_end', 'message': {\n"
        "  'role': 'assistant', 'provider': 'openai-codex', 'model': 'gpt-5.6-luna',\n"
        "  'responseId': 'resp-cli', 'content': [{'type': 'text', 'text': '{\\\"action\\\":\\\"finish\\\",\\\"answer\\\":\\\"ok\\\"}'}],\n"
        "  'usage': {'input': 7, 'output': 3, 'totalTokens': 10}}}))\n"
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    client = PrimeCliModelClient(executable=str(executable), coding_agent_dir=tmp_path)
    result = client.invoke(goal="goal", environment={}, messages=[], remaining_deadline=5, token_cap=20)
    assert result["provider"] == "openai-codex"
    assert result["model"] == "openai-codex/gpt-5.6-luna"
    assert result["responseId"] == "resp-cli"
    assert result["usage"]["totalTokens"] == 10


def test_model_usage_is_recorded_before_token_cap_rejection():
    client = Client([response("r1", '{"action":"finish","answer":"done"}')])
    sink = Sink()
    result = LunaPlanner(client, Kernel(), sink, limits=PlannerLimits(max_model_tokens=3)).run(goal="goal", environment={})
    assert result.status == "budget_exhausted"
    assert result.model_tokens == 4
    assert len(sink.observations) == 1


def test_prime_cli_client_terminates_when_deadline_expires(tmp_path):
    executable = tmp_path / "prime-agent-slow"
    executable.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(2)\n")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    client = PrimeCliModelClient(executable=str(executable), coding_agent_dir=tmp_path)
    started = time.monotonic()
    with pytest.raises(PlannerTimedOut):
        client.invoke(goal="goal", environment={}, messages=[], remaining_deadline=0.05)
    assert time.monotonic() - started < 1


def test_cancel_after_model_turn_prevents_finish_or_kernel_execution():
    cancel = Event()

    class CancellingClient(Client):
        def invoke(self, **kwargs):
            cancel.set()
            return response("r1", '{"action":"finish","answer":"done"}')

    sink = Sink()
    result = LunaPlanner(CancellingClient([]), Kernel(), sink).run(goal="goal", environment={}, cancel=cancel)
    assert result.status == "cancelled"
    assert len(sink.observations) == 1
