import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from adaptive_agent import (  # noqa: E402
    AdapterError,
    ChildPlanRequest,
    ChildPlannerBudget,
    SecurityViolation,
    SharedBudget,
)
from adaptive_agent.prime_child_planner import LunaChildPlanner  # noqa: E402


class FakeClient:
    def __init__(self, usage=3, text=None):
        self.calls = 0
        self.usage = usage
        self.text = text

    def invoke(self, **kwargs):
        self.calls += 1
        self.last = kwargs
        child_input = kwargs["environment"]["childInput"]
        text = self.text if self.text is not None else '{"name":"child","code":"%d * %d"}' % (
            child_input["kwargs"]["left"], child_input["kwargs"]["right"]
        )
        return {
            "provider": "openai-codex",
            "model": "gpt-5.6-luna",
            "responseId": f"child-response-{self.calls}",
            "text": text,
            "usage": {"totalTokens": self.usage},
        }


def request(ledger, prompt="compute", kwargs=None):
    budget = ChildPlannerBudget(ledger)
    return ChildPlanRequest(prompt, kwargs or {"left": 21, "right": 2}, "run", 0,
                            budget.remaining_seconds, (), budget.cancel_event, budget)


class ChildPlannerTests(unittest.TestCase):
    def test_planner_returns_bounded_code_and_forwards_shared_cancel(self):
        ledger = SharedBudget(30, 1000, 4, 1, 100)
        client = FakeClient()
        plan = LunaChildPlanner(client)(request(ledger))
        self.assertTrue(plan.code.endswith("21 * 2"))
        self.assertIn('kwargs = json.loads', plan.code)
        self.assertEqual(client.last["environment"]["childInput"]["kwargs"], {"left": 21, "right": 2})
        self.assertEqual(ledger.model_tokens_used, 3)
        self.assertIs(client.last["cancel"], ledger.cancel_event)
        self.assertIsNone(client.last["token_cap"])

    def test_completed_over_cap_usage_is_retained_and_blocks_next_dispatch(self):
        ledger = SharedBudget(30, 1000, 4, 1, 5)
        client = FakeClient(usage=7)
        planner = LunaChildPlanner(client)
        with self.assertRaises(SecurityViolation):
            planner(request(ledger))
        self.assertEqual(ledger.model_tokens_used, 7)
        self.assertEqual(request(ledger).budget.remaining_model_tokens, 0)
        with self.assertRaises(SecurityViolation):
            planner(request(ledger))
        self.assertEqual(client.calls, 1)

    def test_only_structured_kwargs_change_generated_computation(self):
        ledger = SharedBudget(30, 1000, 4, 2, 100)
        client = FakeClient()
        planner = LunaChildPlanner(client)
        first = planner(request(ledger, prompt="same", kwargs={"left": 6, "right": 7}))
        second = planner(request(ledger, prompt="same", kwargs={"left": 8, "right": 9}))
        self.assertTrue(first.code.endswith("6 * 7"))
        self.assertTrue(second.code.endswith("8 * 9"))
        self.assertIn('"left":6', first.code)
        self.assertIn('"left":8', second.code)
        self.assertEqual(client.calls, 2)

    def test_observation_is_persisted_before_malformed_plan_or_exhaustion(self):
        observations = []
        malformed = FakeClient(text="not-json")
        ledger = SharedBudget(30, 1000, 4, 1, 100)
        with self.assertRaises(AdapterError):
            LunaChildPlanner(malformed, observation_sink=observations.append)(request(ledger))
        self.assertEqual(observations[0]["responseId"], "child-response-1")
        self.assertEqual(observations[0]["provider"], "openai-codex")
        self.assertEqual(observations[0]["model"], "openai-codex/gpt-5.6-luna")
        self.assertEqual(observations[0]["usage"]["totalTokens"], 3)
        empty = FakeClient(text="")
        observations = []
        ledger = SharedBudget(30, 1000, 4, 1, 100)
        with self.assertRaises(AdapterError):
            LunaChildPlanner(empty, observation_sink=observations.append)(request(ledger))
        self.assertEqual(observations[0]["responseId"], "child-response-1")
        self.assertEqual(ledger.model_tokens_used, 3)
        over = FakeClient(usage=7)
        observations = []
        ledger = SharedBudget(30, 1000, 4, 1, 5)
        with self.assertRaises(SecurityViolation):
            LunaChildPlanner(over, observation_sink=observations.append)(request(ledger))
        self.assertEqual(observations[0]["responseId"], "child-response-1")
        self.assertEqual(ledger.model_tokens_used, 7)

    def test_parent_receipt_uses_same_ledger_and_overage_is_retained(self):
        ledger = SharedBudget(30, 1000, 4, 1, 5)
        budget = ChildPlannerBudget(ledger)
        planner = LunaChildPlanner(FakeClient(), budget=budget)
        with self.assertRaises(SecurityViolation):
            planner.record_parent_model_usage({"totalTokens": 7})
        self.assertEqual(ledger.model_tokens_used, 7)
        self.assertEqual(budget.remaining_model_tokens, 0)

    def test_cancelled_request_does_not_dispatch_model(self):
        ledger = SharedBudget(30, 1000, 4, 1, 100)
        ledger.cancel()
        client = FakeClient()
        with self.assertRaises(SecurityViolation):
            LunaChildPlanner(client)(request(ledger))
        self.assertEqual(client.calls, 0)


if __name__ == "__main__":
    unittest.main()
