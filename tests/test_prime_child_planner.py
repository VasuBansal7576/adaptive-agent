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
    parse_model_usage,
)
from adaptive_agent.prime_child_planner import LunaChildPlanner, _usage_tokens  # noqa: E402


class FakeClient:
    def __init__(self, usage=3, text=None):
        self.calls = 0
        self.usage = usage
        self.text = text

    def invoke(self, **kwargs):
        self.calls += 1
        self.last = kwargs
        child_input = kwargs["environment"].get("childInput", {"kwargs": {"left": 21, "right": 2}})
        text = self.text if self.text is not None else '{"name":"child","code":"%d * %d"}' % (
            child_input["kwargs"]["left"], child_input["kwargs"]["right"]
        )
        return {
            "provider": "openai-codex",
            "model": "gpt-5.6-luna",
            "responseId": f"child-response-{self.calls}",
            "text": text,
            "usage": self.usage if isinstance(self.usage, dict) else {"totalTokens": self.usage},
        }


def request(ledger, prompt="compute", kwargs=None):
    budget = ChildPlannerBudget(ledger)
    return ChildPlanRequest(prompt, kwargs or {"left": 21, "right": 2}, "run", 0,
                            budget.remaining_seconds, (), budget.cancel_event, budget)


class UsageNormalizationTests(unittest.TestCase):
    def test_complete_input_output_shapes_are_summed(self):
        for usage in (
            {"inputTokens": 9000, "outputTokens": 1},
            {"input_tokens": 9000, "output_tokens": 1},
            {"input": 9000, "output": 1},
            {"totalTokens": 9001, "inputTokens": 9000, "outputTokens": 1},
        ):
            self.assertEqual(_usage_tokens(usage), 9001)

    def test_contradictory_or_invalid_usage_is_rejected(self):
        for usage in (
            {"totalTokens": 9000, "inputTokens": 9000, "outputTokens": 1},
            {"totalTokens": 1, "total_tokens": 2},
            {"inputTokens": -1, "outputTokens": 1},
            {"inputTokens": 9000, "outputTokens": "1"},
            {"inputTokens": 9000},
        ):
            with self.assertRaises(AdapterError):
                _usage_tokens(usage)


class CostGuardTests(unittest.TestCase):
    def test_sdk_nominal_cost_is_canonical_microunits(self):
        receipt = parse_model_usage({"inputTokens": 9, "outputTokens": 1, "cost": {"total": 0.00001}}, require_cost=True)
        self.assertEqual((receipt.tokens, receipt.cost_microunits, receipt.currency), (10, 10, "USD"))
        with self.assertRaises(AdapterError):
            parse_model_usage({"totalTokens": 10, "inputTokens": 9, "outputTokens": 1, "cost": {"total": 0.00001}, "costMicrounits": 11}, require_cost=True)
        with self.assertRaises(AdapterError):
            parse_model_usage({"inputTokens": 9, "outputTokens": 1, "cost": {"total": 0.00001, "currency": "EUR"}}, require_cost=True)

    def test_unknown_cost_is_rejected_when_cost_guard_is_configured(self):
        ledger = SharedBudget(30, 1000, 4, 1, 100, max_model_cost_microunits=10)
        observations = []
        with self.assertRaises(AdapterError):
            LunaChildPlanner(FakeClient(usage={"inputTokens": 1, "outputTokens": 1}), observation_sink=observations.append)(request(ledger))
        self.assertEqual(observations, [])
        self.assertEqual((ledger.model_tokens_used, ledger.model_cost_microunits_used), (0, 0))

    def test_zero_cost_budget_blocks_parent_and_child_preflight(self):
        ledger = SharedBudget(30, 1000, 4, 1, 100, max_model_cost_microunits=0)
        client = FakeClient(usage={"inputTokens": 1, "outputTokens": 1, "cost": {"total": 0.000001}})
        planner = LunaChildPlanner(client, budget=ChildPlannerBudget(ledger))
        with self.assertRaises(SecurityViolation):
            planner.parent_model_client().invoke(goal="blocked", environment={}, messages=[], remaining_deadline=5)
        with self.assertRaises(SecurityViolation):
            planner(request(ledger))
        self.assertEqual(client.calls, 0)

    def test_shared_cost_ledger_is_atomic_under_concurrent_child_usage(self):
        from concurrent.futures import ThreadPoolExecutor
        ledger = SharedBudget(30, 1000, 4, 4, 100, max_model_cost_microunits=10)
        def charge(_index):
            try:
                ledger.record_model_usage(1, 4, "USD")
                return "ok"
            except SecurityViolation:
                return "exhausted"
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(charge, range(4)))
        self.assertEqual(ledger.model_cost_microunits_used, 16)
        self.assertEqual(ledger.model_tokens_used, 4)
        self.assertIn("exhausted", results)
        self.assertEqual(ledger.max_model_cost_microunits - ledger.model_cost_microunits_used, -6)

    def test_parent_child_cumulative_nominal_cost_retains_overage_and_blocks(self):
        ledger = SharedBudget(30, 1000, 4, 1, 100, max_model_cost_microunits=10)
        observations = []
        client = FakeClient(usage={"inputTokens": 1, "outputTokens": 1, "cost": {"total": 0.000007}})
        planner = LunaChildPlanner(client, budget=ChildPlannerBudget(ledger), observation_sink=observations.append)
        parent = planner.parent_model_client(observation_sink=observations.append)
        parent.invoke(goal="parent", environment={}, messages=[], remaining_deadline=5)
        client.usage = {"input_tokens": 1, "output_tokens": 1, "cost": {"total": 0.000004}}
        with self.assertRaises(SecurityViolation):
            planner(request(ledger))
        self.assertEqual(ledger.model_cost_microunits_used, 11)
        self.assertEqual([item["costMicrounits"] for item in observations], [7, 4])
        self.assertEqual(len(observations), 2)
        with self.assertRaises(SecurityViolation):
            parent.invoke(goal="blocked", environment={}, messages=[], remaining_deadline=5)
        with self.assertRaises(SecurityViolation):
            planner(request(ledger))
        self.assertEqual(client.calls, 2)


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
        invalid = FakeClient(usage={"inputTokens": "9000", "outputTokens": 1})
        observations = []
        ledger = SharedBudget(30, 1000, 4, 1, 100)
        with self.assertRaises(AdapterError):
            LunaChildPlanner(invalid, observation_sink=observations.append)(request(ledger))
        self.assertEqual(observations, [])
        self.assertEqual(ledger.model_tokens_used, 0)
        over = FakeClient(usage=7)
        observations = []
        ledger = SharedBudget(30, 1000, 4, 1, 5)
        with self.assertRaises(SecurityViolation):
            LunaChildPlanner(over, observation_sink=observations.append)(request(ledger))
        self.assertEqual(observations[0]["responseId"], "child-response-1")
        self.assertEqual(ledger.model_tokens_used, 7)

    def test_parent_proxy_charges_same_ledger_and_blocks_next_dispatch(self):
        ledger = SharedBudget(30, 1000, 4, 1, 5)
        budget = ChildPlannerBudget(ledger)
        client = FakeClient(usage=7)
        observations = []
        proxy = LunaChildPlanner(client, budget=budget).parent_model_client(observations.append)
        with self.assertRaises(SecurityViolation):
            proxy.invoke(goal="parent", environment={}, messages=[], remaining_deadline=5)
        self.assertEqual(client.calls, 1)
        self.assertEqual(observations[0]["responseId"], "child-response-1")
        self.assertEqual(ledger.model_tokens_used, 7)
        with self.assertRaises(SecurityViolation):
            proxy.invoke(goal="parent", environment={}, messages=[], remaining_deadline=5)
        self.assertEqual(client.calls, 1)

    def test_parent_and_child_cumulative_usage_blocks_next_dispatch(self):
        ledger = SharedBudget(30, 1000, 4, 1, 10)
        budget = ChildPlannerBudget(ledger)
        client = FakeClient(usage={"inputTokens": 6, "outputTokens": 1})
        observations = []
        planner = LunaChildPlanner(client, budget=budget, observation_sink=observations.append)
        parent = planner.parent_model_client(observation_sink=observations.append)
        parent.invoke(goal="parent", environment={}, messages=[], remaining_deadline=5)
        client.usage = {"input_tokens": 3, "output_tokens": 1}
        with self.assertRaises(SecurityViolation):
            planner(request(ledger))
        self.assertEqual(ledger.model_tokens_used, 11)
        self.assertEqual(len(observations), 2)
        with self.assertRaises(SecurityViolation):
            parent.invoke(goal="blocked", environment={}, messages=[], remaining_deadline=5)
        with self.assertRaises(SecurityViolation):
            planner(request(ledger))
        self.assertEqual(client.calls, 2)

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
