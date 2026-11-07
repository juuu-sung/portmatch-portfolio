import unittest

import numpy as np
import pandas as pd

from portmatch.memory import TrainingMemory
from portmatch.validation import make_folds, sample_validation, summarize


class MemoryValidationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame({"vessel_id": ["A", "B", "C"]})
        self.train = pd.DataFrame({"call_sign": ["CS_A", "CS_A", "CS_B"],
                                   "vessel_name": ["BLUE-1", "BLUE 1", "BLUE 1"],
                                   "target_vessel_id": ["A", "A", "B"]})

    def test_lookup_is_training_only_and_preserves_conflicts(self):
        memory = TrainingMemory().fit(self.train, self.catalog)
        self.assertTrue(memory.lookup({"call_sign": "CS_C", "vessel_name": "UNKNOWN"}).empty)
        matches = memory.lookup({"vessel_name": "blue.1"})
        self.assertEqual(matches.catalog_index.tolist(), [0, 1])
        self.assertAlmostEqual(matches.lookup_purity.iloc[0], 2 / 3)
        self.assertEqual(memory.training_vessels, frozenset(["A", "B"]))
        self.assertNotIn("UNKNOWN", memory.aliases.vessel_name.tolist())

    def test_group_split_never_shares_target(self):
        train = pd.DataFrame({"target_vessel_id": np.repeat(list("ABCDEFGHIJKL"), 3),
                              "vessel_name": ["배", "SHIP"] * 18})
        folds = make_folds(train, "vessel", 3, 42)
        for fold in range(3):
            self.assertFalse(set(train[folds == fold].target_vessel_id) &
                             set(train[folds != fold].target_vessel_id))
        np.testing.assert_array_equal(folds, make_folds(train, "vessel", 3, 42))

    def test_small_sample_has_both_languages(self):
        frame = pd.DataFrame({"vessel_name": ["SHIP"] * 100 + ["배"]})
        sample = sample_validation(frame, 5, 42)
        self.assertEqual(len(sample), 5)
        self.assertIn("배", sample.vessel_name.tolist())

    def test_absent_language_does_not_fabricate_weighted_score(self):
        rows = pd.DataFrame({"has_hangul": [False], "recall_union": [True], "candidate_count": [10]})
        self.assertIsNone(summarize(rows)["weighted_candidate_recall"])


if __name__ == "__main__":
    unittest.main()
