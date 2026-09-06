import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from adaptive_agent import (  # noqa: E402
    ChildPlanRequest,
    ChildPlannerBudget,
    SecurityViolation,
    SharedBudget,
)
from adaptive_agent.prime_child_planner import LunaChildPlanner  # noqa: E402


class FakeClient:
    def __init__(self, usage=3):
        self.calls = 0
        self.usage = usage

    def invoke(self, **kwargs):
        self.calls += 1
        self.last = kwargs
        return {
            "provider": "openai-codex",
            "model": "gpt-5.6-luna",
            "responseId": f"child-response-{self.calls}",
            "text": '{"name":"child","code":"21 * 2"}',
            "usage": {"totalTokens": self.usage},
        }


def request(ledger):
    budget = ChildPlannerBudget(ledger)
    return ChildPlanRequest("compute", {}, "run", 0, budget.remaining_seconds, (), budget.cancel_event, budget)


class ChildPlannerTests(unittest.TestCase):
    def test_planner_returns_bounded_code_and_forwards_shared_cancel(self):
        ledger = SharedBudget(30, 1000, 4, 1, 100)
        client = FakeClient()
        plan = LunaChildPlanner(client)(request(ledger))
        self.assertEqual(plan.code, "21 * 2")
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
