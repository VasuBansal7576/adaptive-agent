from dataclasses import dataclass
from threading import Event

import pytest

from adaptive_agent.planner import LunaPlanner, PlannerError, PlannerLimits, make_luna_model_runner


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
