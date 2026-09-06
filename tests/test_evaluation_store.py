import tempfile
import unittest
from pathlib import Path

from adaptive_agent.evaluation_store import (
    SQLiteAllocationStore,
    SQLiteTrustedAttestationLedger,
)
from adaptive_agent.store import Store


class DurableEvaluatorStoreTests(unittest.TestCase):
    def test_attestation_survives_store_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first = SQLiteTrustedAttestationLedger(Store(path))
            first.put("token-1", "digest-1")
            reopened = SQLiteTrustedAttestationLedger(Store(path))
            self.assertTrue(reopened.durable)
            self.assertEqual(reopened.get("token-1"), "digest-1")

    def test_next_panel_is_atomic_disjoint_and_survives_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            store = Store(path)
            first = SQLiteAllocationStore(store)
            panels = (("task-0",), ("task-1",), ("task-2",))
            self.assertEqual(first.reserve_next("base", "candidate-0", panels, 3), 0)
            self.assertEqual(first.reserve_next("base", "candidate-1", panels, 3), 1)
            self.assertIsNone(first.reserve_next("base", "candidate-0", panels, 3))
            reopened = SQLiteAllocationStore(Store(path))
            self.assertEqual(reopened.reserve_next("base", "candidate-2", panels, 3), 2)
            self.assertIsNone(reopened.reserve_next("base", "candidate-3", panels, 3))


if __name__ == "__main__":
    unittest.main()
