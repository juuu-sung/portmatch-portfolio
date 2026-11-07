"""Behavioral checks for additive name views without learning from labels."""
import unittest

import numpy as np

from portmatch.name_features import FEATURE_NAMES, NameFeatureComputer, contextual_key


class NameFeaturesTests(unittest.TestCase):
    def feature(self, query, names, feature):
        return NameFeatureComputer(names).compute(query, np.arange(len(names)))[:, FEATURE_NAMES.index(feature)]

    def test_missing_names_are_not_perfect_matches(self):
        computer = NameFeatureComputer([None, "", "STAR"])
        np.testing.assert_array_equal(computer.compute(None, [0, 1, 2]), np.zeros((3, len(FEATURE_NAMES))))
        np.testing.assert_array_equal(computer.compute("STAR", [0, 1]), np.zeros((2, len(FEATURE_NAMES))))

    def test_shapes_dtype_range_and_empty_candidates(self):
        computer = NameFeatureComputer(["ALPHA SEA", "베타 11", "007", "---"])
        values = computer.compute("ALPHA SEA 11", [0, 1, 2, 3])
        self.assertEqual(values.shape, (4, 17))
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(((values >= 0) & (values <= 1)).all())
        self.assertEqual(computer.compute("A", []).shape, (0, 17))
        self.assertEqual(len(set(FEATURE_NAMES)), 17)

    def test_order_subset_duplicate_candidates_and_other_queries_are_independent(self):
        computer = NameFeatureComputer(["ALPHA MOON", "BETA MOON", "ALPHA SUN"])
        before = computer.compute("알파 문", [0, 1, 2])
        computer.compute("unrelated 99", [2])
        after = computer.compute("알파 문", [2, 0, 2])
        np.testing.assert_array_equal(after, before[[2, 0, 2]])

    def test_catalog_extension_does_not_change_existing_pairs(self):
        left = NameFeatureComputer(["ALPHA MOON"])
        right = NameFeatureComputer(["ALPHA MOON", "ALPHA MOON", "BETA SUN"])
        np.testing.assert_array_equal(left.compute("알파 문", [0]), right.compute("알파 문", [0]))

    def test_unicode_and_punctuation_boundaries_are_normalized(self):
        computer = NameFeatureComputer(["Alpha-Star １", "ALPHA STAR 1"])
        a = computer.compute("ＡＬＰＨＡ—ＳＴＡＲ 1", [0, 1])
        np.testing.assert_array_equal(a[0], a[1])
        self.assertEqual(a[0, FEATURE_NAMES.index("name_token_roman_alignment")], 1)

    def test_numbers_are_not_partial_matches_or_collapsed(self):
        self.assertIn("11", contextual_key("SHIP 11"))
        self.assertNotEqual(contextual_key("SHIP 1"), contextual_key("SHIP 11"))
        scores = self.feature("SHIP 1", ["SHIP 1", "SHIP 10"], "name_token_context_weakest")
        self.assertEqual(scores.tolist(), [1, 0])
        full = self.feature("1", ["1", "10"], "name_token_context_alignment")
        self.assertEqual(full.tolist(), [1, 0])

    def test_alignment_accounts_for_duplicates_order_and_extra_words(self):
        scores = self.feature("SEA SEA", ["SEA SEA", "SEA", "SEA LAND"], "name_token_roman_alignment")
        self.assertEqual(scores[0], 1)
        self.assertEqual(scores[1], .5)
        self.assertLess(scores[2], 1)
        ordered = self.feature("ALPHA BETA", ["ALPHA BETA", "BETA ALPHA"], "name_token_roman_alignment")
        self.assertGreater(ordered[0], ordered[1])

    def test_residual_words_distinguish_a_shared_boundary(self):
        scores = self.feature("OCEAN RUBY", ["OCEAN RUBY", "OCEAN PEARL"], "name_token_residual_context_ratio")
        self.assertEqual(scores[0], 1)
        self.assertLess(scores[1], scores[0])
        flags = self.feature("RUBY", ["RUBY", "PEARL"], "name_token_residual_present")
        self.assertEqual(flags.tolist(), [0, 0])

    def test_context_soft_c_and_hangul_coda_are_not_legacy_all_c_to_k(self):
        self.assertEqual(contextual_key("CELL"), contextual_key("SELL"))
        self.assertEqual(contextual_key("CAT"), contextual_key("KAT"))
        self.assertEqual(contextual_key("릿"), contextual_key("RIT"))
        self.assertNotEqual(contextual_key("릿"), contextual_key("RIS"))
        # Repeated consonants across distinct token boundaries are retained.
        self.assertEqual(contextual_key("SEA SAIL"), "SSL")

    def test_ch_variants_are_additive(self):
        self.assertNotEqual(contextual_key("CHEM"), contextual_key("KEM"))
        self.assertEqual(contextual_key("CHEM", hard_ch=True), contextual_key("KEM"))
        basic = self.feature("CHEM", ["KEM"], "name_context_ratio")[0]
        variant = self.feature("CHEM", ["KEM"], "name_context_variant_ratio")[0]
        self.assertGreater(variant, basic)

    def test_suffix_variant_keeps_original_and_does_not_strip_short_words(self):
        computer = NameFeatureComputer(["USEOK", "USEOKHO", "SAN"])
        values = computer.compute("우석호", [0, 1])
        score = FEATURE_NAMES.index("name_suffix_variant_context_ratio")
        base = FEATURE_NAMES.index("name_context_ratio")
        self.assertTrue((values[:, score] >= values[:, base]).all())
        self.assertGreater(values[0, score], values[0, base])
        self.assertEqual(values[1, score], 1)
        self.assertEqual(self.feature("산호", ["SAN"], "name_suffix_variant_present")[0], 0)
        self.assertEqual(self.feature("호", ["HO"], "name_suffix_variant_present")[0], 0)

    def test_vowel_view_can_distinguish_a_consonant_collision(self):
        context = self.feature("MILE", ["MILE", "MOLE"], "name_context_ratio")
        vowel = self.feature("MILE", ["MILE", "MOLE"], "name_vowel_fold_ratio")
        self.assertEqual(context.tolist(), [1, 1])
        self.assertGreater(vowel[0], vowel[1])

    def test_invalid_indices_are_rejected_without_silent_truncation(self):
        computer = NameFeatureComputer(["A", "B"])
        for indices in ([.5], [-1], [2], [np.nan], [np.inf], [[0]], [True], ["0"]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                computer.compute("A", indices)


if __name__ == "__main__":
    unittest.main()
