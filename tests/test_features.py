"""Tests for meaningful missing-data, leakage and row-independence behavior."""

import json
import unittest

import numpy as np
import pandas as pd

from portmatch.features import CategoryCompatibility, PairFeatureBuilder


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame([
            dict(vessel_id="A", vessel_name="MORNING STAR 1", gross_tonnage=1000,
                 flag_name="Korea", vessel_type="Tanker"),
            dict(vessel_id="B", vessel_name="MORNING STAR 2", gross_tonnage=1000,
                 flag_name="Japan", vessel_type="Bulk"),
            dict(vessel_id="C", vessel_name="OTHER", gross_tonnage=np.nan,
                 flag_name="Japan", vessel_type="Tanker"),
        ])
        self.train = pd.DataFrame([
            dict(target_vessel_id="A", flag_name="한국", vessel_type="유조선"),
            dict(target_vessel_id="A", flag_name="한국", vessel_type="유조선"),
        ])
        self.category = CategoryCompatibility().fit(self.train, self.catalog)
        self.builder = PairFeatureBuilder(self.catalog, self.category)

    def test_same_tonnage_distinguished_by_name_digits_and_category(self):
        row = dict(vessel_name="morning-star 1", gross_tonnage=1000,
                   international_gross_tonnage=1000, flag_name="한국", vessel_type="유조선")
        f = self.builder.transform(row, pd.DataFrame({"catalog_index": [0, 1]}))
        self.assertEqual(f.gross_exact.tolist(), [1, 1])
        self.assertEqual(f.numbers_conflict.tolist(), [0, 1])
        self.assertGreater(f.flag_probability.iloc[0], f.flag_probability.iloc[1])
        self.assertGreater(f.type_probability.iloc[0], f.type_probability.iloc[1])

    def test_missing_tonnage_is_never_perfect_match(self):
        f = self.builder.transform(dict(vessel_name="OTHER"), pd.DataFrame({"catalog_index": [2]}))
        self.assertEqual(float(f.gross_exact.iloc[0]), 0)
        self.assertEqual(float(f.gross_pair_valid.iloc[0]), 0)
        self.assertEqual(float(f.catalog_tonnage_missing.iloc[0]), 1)
        self.assertTrue(np.isnan(f.best_tonnage_gap.iloc[0]))

    def test_hangul_transliteration_is_a_separate_comparison(self):
        f = self.builder.transform(dict(vessel_name="모닝 스타 1"),
                                   pd.DataFrame({"catalog_index": [0]}))
        self.assertEqual(float(f.query_has_hangul.iloc[0]), 1)
        self.assertGreater(f.romanized_ratio.iloc[0], f.name_ratio.iloc[0])

    def test_deadweight_is_not_compared_to_gross_tonnage(self):
        candidates = pd.DataFrame({"catalog_index": [0, 1]})
        absent = self.builder.transform({}, candidates)
        supplied = self.builder.transform(dict(deadweight_tonnage=1000), candidates)
        pd.testing.assert_frame_equal(absent, supplied)

    def test_empty_candidates_preserve_the_feature_schema(self):
        nonempty = self.builder.transform({}, pd.DataFrame({"catalog_index": [0]}))
        empty = self.builder.transform({}, pd.DataFrame({"catalog_index": []}))
        self.assertEqual(list(empty.columns), list(nonempty.columns))
        self.assertEqual(len(empty), 0)

    def test_unknown_category_is_neutral_and_transform_does_not_learn(self):
        before = json.dumps(self.category.to_dict(), sort_keys=True)
        row = dict(vessel_name="UNKNOWN", flag_name="미관측", vessel_type="미관측")
        f = self.builder.transform(row, pd.DataFrame({"catalog_index": [0, 1]}))
        np.testing.assert_array_equal(f.flag_lift_log, [0, 0])
        np.testing.assert_array_equal(f.type_source_seen, [0, 0])
        self.assertEqual(before, json.dumps(self.category.to_dict(), sort_keys=True))

    def test_candidate_order_subset_and_intervening_queries_do_not_change_features(self):
        row = dict(vessel_name="MORNING STAR 1", gross_tonnage=1000, flag_name="한국")
        full = self.builder.transform(row, pd.DataFrame({"catalog_index": [0, 1, 2]}))
        self.builder.transform(dict(vessel_name="무관한 다른 기록", gross_tonnage=5000),
                               pd.DataFrame({"catalog_index": [1]}))
        reordered = self.builder.transform(row, pd.DataFrame({"catalog_index": [2, 0]}))
        np.testing.assert_allclose(full.iloc[[2, 0]].to_numpy(), reordered.to_numpy(), equal_nan=True)
        self.assertTrue(all(dtype == np.dtype("float32") for dtype in full.dtypes))
        self.assertNotIn("vessel_id", full)
        self.assertNotIn("catalog_index", full)

    def test_validation_labels_are_not_available_to_category_table(self):
        # B's category correspondence was not in training and must stay unseen.
        f = self.category.transform(dict(flag_name="일본", vessel_type="벌크선"), self.catalog)
        self.assertEqual(float(f.flag_source_seen.sum()), 0)
        self.assertEqual(float(f.type_pair_seen.sum()), 0)

    def test_category_json_round_trip(self):
        restored = CategoryCompatibility.from_dict(json.loads(json.dumps(self.category.to_dict())))
        row = dict(flag_name="한국", vessel_type="유조선")
        pd.testing.assert_frame_equal(self.category.transform(row, self.catalog),
                                      restored.transform(row, self.catalog))

    def test_invalid_catalog_index_and_missing_training_target_rejected(self):
        with self.assertRaises(ValueError):
            self.builder.transform({}, pd.DataFrame({"catalog_index": [0.5]}))
        with self.assertRaises(ValueError):
            self.builder.transform({}, pd.DataFrame({"catalog_index": [-1]}))
        bad = self.train.copy()
        bad.loc[0, "target_vessel_id"] = "NOT_IN_CATALOG"
        with self.assertRaises(ValueError):
            CategoryCompatibility().fit(bad, self.catalog)


if __name__ == "__main__":
    unittest.main()
