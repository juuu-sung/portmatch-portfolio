import unittest

import numpy as np
import pandas as pd

from portmatch.sampling import SamplingConfig, sample_training_candidates


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.candidates = pd.DataFrame({"catalog_index": np.arange(300),
                                        "call_match": 0, "alias_match": 0})
        self.features = pd.DataFrame({
            "name_edit_similarity": np.linspace(0, 1, 300),
            "romanized_ratio": np.linspace(1, 0, 300),
            "phonetic_ratio": np.sin(np.arange(300)) ** 2,
            "international_relative_gap": np.abs(np.arange(300) - 150) / 300,
            "flag_lift_log": np.arange(300) % 7,
            "type_lift_log": np.arange(300) % 13,
        })

    def select_ids(self, candidates=None, features=None, query="Q", config=None):
        candidates = self.candidates if candidates is None else candidates
        features = self.features if features is None else features
        positions = sample_training_candidates(candidates, features, 100, query, config)
        return set(candidates.iloc[positions].catalog_index), positions

    def test_positive_once_unique_positions_and_budget(self):
        ids, positions = self.select_ids()
        self.assertEqual(len(ids), 129)
        self.assertIn(100, ids)
        self.assertEqual(len(np.unique(positions)), len(positions))
        np.testing.assert_array_equal(positions, np.sort(positions))

    def test_hard_views_keep_different_distractors_and_false_lookup(self):
        candidates = self.candidates.copy()
        candidates.loc[250, "call_match"] = 1
        ids, _ = self.select_ids(candidates=candidates, config=SamplingConfig(8, 0))
        self.assertTrue({100, 250, 299, 0, 150}.issubset(ids))

    def test_random_selection_is_query_local_and_permutation_invariant(self):
        config = SamplingConfig(15, 20, seed=43)
        expected, _ = self.select_ids(config=config)
        self.select_ids(query="another query", config=config)
        again, _ = self.select_ids(config=config)
        self.assertEqual(expected, again)
        permutation = np.random.default_rng(7).permutation(300)
        candidates = self.candidates.iloc[permutation].reset_index(drop=True)
        features = self.features.iloc[permutation].reset_index(drop=True)
        reordered, _ = self.select_ids(candidates, features, config=config)
        self.assertEqual(expected, reordered)
        another, _ = self.select_ids(query="different query", config=config)
        self.assertNotEqual(expected, another)

    def test_missing_positive_is_not_injected(self):
        positions = sample_training_candidates(self.candidates, self.features, 999, "Q")
        self.assertEqual(positions.size, 0)

    def test_small_groups_keep_all_candidates_and_positive_only_is_valid(self):
        c = self.candidates.iloc[98:103]
        f = self.features.iloc[98:103]
        np.testing.assert_array_equal(sample_training_candidates(c, f, 100, "Q"), np.arange(5))
        np.testing.assert_array_equal(
            sample_training_candidates(c.iloc[[2]], f.iloc[[2]], 100, "Q"), [0])

    def test_nan_scores_do_not_crash_or_change_inputs(self):
        c = self.candidates.copy()
        f = self.features.copy()
        f[:] = np.nan
        f.loc[299, "name_edit_similarity"] = 1
        before_c, before_f = c.copy(deep=True), f.copy(deep=True)
        ids, _ = self.select_ids(c, f, config=SamplingConfig(5, 5))
        self.assertEqual(len(ids), 11)
        self.assertIn(299, ids)
        pd.testing.assert_frame_equal(c, before_c)
        pd.testing.assert_frame_equal(f, before_f)

    def test_pure_random_sampling_and_hard_only_sampling(self):
        for config in (SamplingConfig(0, 9), SamplingConfig(9, 0)):
            ids, _ = self.select_ids(config=config)
            self.assertEqual(len(ids), 10)
            self.assertIn(100, ids)

    def test_reject_invalid_config_alignment_and_duplicate_candidates(self):
        for config in ((0, 0), (-1, 2), (1.5, 2), (True, 2)):
            with self.assertRaises(ValueError):
                SamplingConfig(*config)
        with self.assertRaises(ValueError):
            self.select_ids(features=self.features.iloc[::-1])
        c = self.candidates.copy()
        c.loc[5, "catalog_index"] = 4
        with self.assertRaises(ValueError):
            self.select_ids(candidates=c)


if __name__ == "__main__":
    unittest.main()
