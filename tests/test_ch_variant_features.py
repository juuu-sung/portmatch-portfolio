"""Behavioral checks for the additive, coherent CH pronunciation experiment."""
import unittest

import numpy as np

from portmatch.ch_variant_features import FEATURE_NAMES, ChVariantFeatureComputer
from portmatch.name_features import FEATURE_NAMES as V1_NAMES, NameFeatureComputer


class ChVariantFeaturesTests(unittest.TestCase):
    def values(self, query, names):
        return ChVariantFeatureComputer(names).compute(query, np.arange(len(names)))

    def feature(self, query, names, suffix):
        return self.values(query, names)[:, FEATURE_NAMES.index("name_ch_token_" + suffix)]

    def test_hard_and_affricate_ch_are_both_available_through_residual_vowels(self):
        for query in ("OCEAN KEM", "OCEAN JEM"):
            with self.subTest(query=query):
                values = self.values(query, ["OCEAN CHEM"])[0]
                np.testing.assert_array_equal(values, np.ones(7))
        legacy = NameFeatureComputer(["OCEAN CHEM"]).compute("OCEAN KEM", [0])[0]
        self.assertLess(legacy[V1_NAMES.index("name_token_residual_vowel_ratio")], 1)

    def test_one_pronunciation_is_used_for_first_last_and_alignment(self):
        # Choosing CH->K for the first token and CH->J for the last token
        # independently would invent a perfect match unavailable to either view.
        values = self.values("KEM JEM", ["CHEM CHEM"])[0]
        self.assertEqual(values[0], .75)
        self.assertEqual(values[1], .5)
        self.assertEqual(values[2] + values[3], 1.5)
        self.assertEqual(values[4], .5)
        self.assertLess(values[5], 1)
        self.assertEqual(values[6], 1)

    def test_residual_boundaries_are_recomputed_for_the_selected_pronunciation(self):
        values = self.values("KEM RUBY", ["CHEM PEARL"])[0]
        legacy = NameFeatureComputer(["KEM PEARL"]).compute("KEM RUBY", [0])[0]
        for new_suffix, old_suffix in (
                ("residual_context_ratio", "residual_context_ratio"),
                ("residual_vowel_ratio", "residual_vowel_ratio"),
                ("residual_present", "residual_present")):
            self.assertEqual(values[FEATURE_NAMES.index("name_ch_token_" + new_suffix)],
                             legacy[V1_NAMES.index("name_token_" + old_suffix)])
        self.assertEqual(values[6], 1)
        self.assertLess(values[4], 1)
        self.assertLess(values[5], 1)

    def test_no_ch_pairs_reproduce_the_seven_v1_values(self):
        names = ["OCEAN RUBY", "OCEAN PEARL", "OCEAN RUBY 11", "SEA", "ALPHA ALPHA", ""]
        legacy = NameFeatureComputer(names)
        new = ChVariantFeatureComputer(names)
        original_columns = [V1_NAMES.index(name.replace("name_ch_token_", "name_token_"))
                            for name in FEATURE_NAMES]
        for query in ("OCEAN RUBY", "오션 루비", "SHIP 1", "ALPHA", None):
            with self.subTest(query=query):
                np.testing.assert_array_equal(new.compute(query, range(len(names))),
                                              legacy.compute(query, range(len(names)))[:, original_columns])

    def test_numbers_stay_exact_tokens_and_duplicate_tokens_cannot_be_reused(self):
        values = self.values("KEM 1", ["CHEM 1", "CHEM 10"])
        self.assertEqual(values[0, 0], 1)
        self.assertEqual(values[1, 0], .5)
        self.assertEqual(values[1, 1], 0)
        self.assertEqual(values[1, 3], 0)
        full = self.feature("1", ["1", "10"], "context_alignment")
        np.testing.assert_array_equal(full, [1, 0])
        scores = self.feature("KEM KEM", ["CHEM CHEM", "CHEM"], "context_alignment")
        np.testing.assert_array_equal(scores, [1, .5])
        ordered = self.feature("KEM RUBY", ["CHEM RUBY", "RUBY CHEM"], "context_alignment")
        self.assertGreater(ordered[0], ordered[1])

    def test_empty_names_and_absent_residuals_are_not_matches(self):
        computer = ChVariantFeatureComputer([None, "", "---", "CHEM"])
        np.testing.assert_array_equal(computer.compute(None, [0, 1, 2, 3]), np.zeros((4, 7)))
        np.testing.assert_array_equal(computer.compute("KEM", [0, 1, 2]), np.zeros((3, 7)))
        np.testing.assert_array_equal(computer.compute("KEM", [3])[0, 4:], np.zeros(3))
        self.assertEqual(computer.compute("KEM", []).shape, (0, 7))

    def test_pairs_are_independent_of_catalog_order_extension_and_other_queries(self):
        names = ["OCEAN CHEM", "OCEAN PEARL", "CHEM RUBY"]
        computer = ChVariantFeatureComputer(names)
        original = computer.compute("OCEAN KEM", [0, 1, 2])
        computer.compute("unrelated 99", [2])
        np.testing.assert_array_equal(computer.compute("OCEAN KEM", [2, 0, 2]), original[[2, 0, 2]])
        extended = ChVariantFeatureComputer(names + ["OCEAN CHEM", "OTHER"])
        np.testing.assert_array_equal(extended.compute("OCEAN KEM", [0, 1, 2]), original)

    def test_tie_breaking_is_symmetric_and_deterministic(self):
        for a, b in (("KEM JEM", "CHEM CHEM"), ("CHEM RUBY", "JEM KEM"),
                     ("CHEM KEM", "JEM CHEM"), ("OCEAN KEM", "OCEAN CHEM")):
            with self.subTest(a=a, b=b):
                np.testing.assert_array_equal(self.values(a, [b]), self.values(b, [a]))

    def test_unicode_normalization_and_finite_feature_bounds(self):
        values = self.values("ＯＣＥＡＮ—ＫＥＭ １", ["Ocean-Chem 1", "OCEAN CHEM １"])
        np.testing.assert_array_equal(values[0], values[1])
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values >= 0) & (values <= 1)).all())
        self.assertEqual(len(set(FEATURE_NAMES)), 7)

    def test_invalid_indices_are_rejected(self):
        computer = ChVariantFeatureComputer(["CHEM", "RUBY"])
        for indices in ([.5], [-1], [2], [np.nan], [np.inf], [[0]], [True], ["0"]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                computer.compute("KEM", indices)


if __name__ == "__main__":
    unittest.main()
