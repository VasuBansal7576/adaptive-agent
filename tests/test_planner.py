from dataclasses import dataclass
import json
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


def test_prime_cli_serialization_keeps_full_environment_once_with_unicode(tmp_path):
    captured = tmp_path / "request.bin"
    executable = tmp_path / "prime-agent-capture"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(captured)!r}).write_bytes(sys.argv[-1].encode('utf-8'))\n"
        "print(json.dumps({'type':'message_end','message':{'role':'assistant','provider':'openai-codex','model':'gpt-5.6-luna','responseId':'captured','content':[{'type':'text','text':'{\\\"action\\\":\\\"finish\\\",\\\"answer\\\":\\\"ok\\\"}'}],'usage':{'totalTokens':3}}}))\n"
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    environment = {
        "taskContext": {"instruction": "完成任务 🌍", "marker": "ENVIRONMENT-唯一-雪"},
        "docs": [{"content": "DOCS-RAW-唯一"}],
        "publicDocs": [{"content": "DOCS-唯一"}],
        "capabilities": ["CAPABILITY-唯一"],
        "toolSchemas": [{"name": "TOOL-唯一"}],
        "schema": {"name": "SCHEMA-唯一"},
        "budgetRef": {"id": "BUDGET-唯一"},
        "activeSkills": [{"id": "SKILL-唯一", "procedure": "Use the tool."}],
    }
    messages = [{"role": "system", "content": LunaPlanner(None, None, None)._system_prompt(environment, environment["activeSkills"])}]
    messages.append({"role": "user", "content": "USER-唯一"})
    goal = "GOAL-唯一"
    PrimeCliModelClient(executable=str(executable), coding_agent_dir=tmp_path).invoke(
        goal=goal, environment=environment, messages=messages, remaining_deadline=5
    )
    raw = captured.read_bytes()
    expected = json.dumps({"goal": goal, "environment": environment, "messages": messages}, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    assert raw == expected
    decoded = json.loads(raw)
    assert decoded["environment"] == environment
    assert decoded["messages"] == messages
    for marker in ("ENVIRONMENT-唯一-雪", "DOCS-RAW-唯一", "DOCS-唯一", "CAPABILITY-唯一", "TOOL-唯一", "SCHEMA-唯一", "BUDGET-唯一", "SKILL-唯一"):
        assert raw.count(marker.encode("utf-8")) == 1


def test_system_prompt_uses_first_class_environment_and_preserves_distinct_skills():
    environment = {"activeSkills": [{"id": "environment-skill"}], "taskContext": {"marker": "environment-only"}}
    explicit = ({"id": "caller-skill", "procedure": "caller-only"},)
    prompt = LunaPlanner(None, None, None)._system_prompt(environment, explicit)
    assert "environment-only" not in prompt
    assert "caller-skill" in prompt
    assert "environment-skill" not in prompt

    standalone = LunaPlanner(None, None, None)._system_prompt({}, explicit)
    assert "caller-skill" in standalone


def test_system_prompt_deduplicates_overlapping_skills_by_full_value():
    shared = {"id": "shared", "procedure": "same"}
    environment = {"activeSkills": [shared]}
    explicit = (shared, {"id": "shared", "procedure": "different"}, {"id": "new"})
    prompt = LunaPlanner(None, None, None)._system_prompt(environment, explicit)
    assert prompt.count('"id":"shared"') == 1
    assert '"procedure":"different"' in prompt
    assert '"id":"new"' in prompt


def test_system_prompt_rejects_supplemental_skill_json_that_would_be_truncated():
    planner = LunaPlanner(None, None, None, limits=PlannerLimits(max_context_chars=256))
    long_skill = {"id": "long", "procedure": "雪" * 300}
    with pytest.raises(PlannerError, match="supplemental active skill context exceeds"):
        planner.run(goal="goal", environment={}, active_skills=(long_skill,))


def test_prime_cli_client_discards_unbounded_intermediate_jsonl_events(tmp_path):
    executable = tmp_path / "prime-agent-many-events"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "for _ in range(3000):\n"
        " print(json.dumps({'type':'message_update','message':{'role':'assistant','content':[{'type':'text','text':'x'*100}]}}))\n"
        "print(json.dumps({'type':'message_end','message':{'role':'assistant','provider':'openai-codex','model':'gpt-5.6-luna','responseId':'bounded','content':[{'type':'text','text':'{\\\"action\\\":\\\"finish\\\",\\\"answer\\\":\\\"ok\\\"}'}],'usage':{'totalTokens':17}}}))\n"
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    result = PrimeCliModelClient(executable=str(executable), coding_agent_dir=tmp_path, max_output_chars=64).invoke(
        goal="goal", environment={}, messages=[], remaining_deadline=5
    )
    assert result["text"] == '{"action":"finish","answer":"ok"}'
    assert result["usage"] == {"totalTokens": 17}


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
