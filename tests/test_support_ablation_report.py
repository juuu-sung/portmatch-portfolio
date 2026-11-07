import copy
import json
import unittest

import pandas as pd

from portmatch.support_ablation_report import (
    CONTRAST, EXPERIMENT, REFERENCE, RUNNERS, build_support_ablation_report,
)


def records(variant="original"):
    cases = [("h1", "H", True, False, True, True),
             ("h2", "H", True, False, True, True),
             ("n1", "N", False, True, False, True),
             ("n2", "M", False, False, False, False)]
    rows = []
    for sample, target, hangul, base_correct, experiment_correct, found in cases:
        for i, (runner, correct) in enumerate(zip(RUNNERS, (base_correct, experiment_correct))):
            rows.append({"variant": variant, "model": runner, "sample_id": sample,
                "target_vessel_id": target, "vessel_name": "한글" if hangul else "LATIN",
                "has_hangul": hangul, "predicted_vessel_id": target if correct else "WRONG" + str(i),
                "candidate_found": found, "target_rank": 1 if correct else (2 if found else 0),
                "candidate_count": 8, "candidate_ids_sha256": "a" * 64,
                "top_score": 3. + i, "score_margin": .2,
                "retrieval_seconds": 1., "feature_seconds": 2. + i,
                "rank_seconds": 1., "total_seconds": 4. + i,
                "top_type_probability": .8 - i * .3})
    return rows


class SupportAblationReportTests(unittest.TestCase):
    def report(self, rows=None, variants=("original",), coverage=None):
        return build_support_ablation_report(pd.DataFrame(records() if rows is None else rows),
            coverage={"queries": 4, "fold": 1} if coverage is None else coverage,
            proof={"heldout_excluded": True}, startup={}, parity={}, variants=variants)

    def test_direction_language_weights_ship_counts_and_retraining_semantics(self):
        summary, changes, vessels, markdown = self.report()
        self.assertIs(summary["complete"], False)
        self.assertFalse(summary["ranker_weights_fixed"])
        self.assertEqual(summary["fixed_search_fraction"], 1.)
        self.assertEqual(summary["fixed_category_fraction"], 1.)
        self.assertEqual(summary["feature_counts"], {"baseline86": 86, "no_support82": 82})
        metric = summary["metrics"]["original"]
        self.assertAlmostEqual(metric[REFERENCE]["weighted_top1"], .15)
        self.assertAlmostEqual(metric[EXPERIMENT]["weighted_top1"], .7)
        self.assertAlmostEqual(summary["metric_deltas"]["original"]["weighted_top1_pp"], 55.)
        transition = summary["transitions"]["original"]
        self.assertEqual((transition["corrected"], transition["corrected_vessels"]), (2, 1))
        self.assertEqual((transition["regressed"], transition["regressed_vessels"]), (1, 1))
        self.assertEqual(transition["hangul"]["corrected"], 2)
        self.assertEqual(transition["non_hangul"]["regressed"], 1)
        self.assertEqual(transition["candidate_gained"], 0)
        self.assertEqual(transition["candidate_lost"], 0)
        self.assertEqual(transition["still_wrong_prediction_changed"], 1)
        self.assertEqual(transition["errors_reference"]["candidate_missing"], 1)
        self.assertEqual(transition["errors_experiment"]["candidate_missing"], 1)
        self.assertTrue(changes.reference_runner.eq(REFERENCE).all())
        self.assertTrue(changes.experiment_runner.eq(EXPERIMENT).all())
        self.assertTrue(changes.contrast.eq(CONTRAST).all())
        self.assertIn("predicted_vessel_id_reference", changes)
        self.assertIn("predicted_vessel_id_experiment", changes)
        self.assertNotIn("model", changes)
        self.assertEqual(vessels.loc[vessels.target_vessel_id.eq("H"), "corrected"].iloc[0], 2)
        self.assertIn("82개−86개", markdown)
        self.assertIn("가중치는 서로 다르므로", markdown)
        self.assertIn("계산된 NaN이 아니라", markdown)
        self.assertIn("캐시", markdown)
        json.dumps(summary, ensure_ascii=False, allow_nan=False)

    def test_variants_remain_separate_and_order_is_irrelevant(self):
        rows = records() + records("both_gt_missing")
        for row in rows:
            if row["variant"] == "both_gt_missing" and row["sample_id"] == "h1" and row["model"] == EXPERIMENT:
                row.update(predicted_vessel_id="WRONG", target_rank=2)
        variants = ("original", "both_gt_missing")
        left, right = self.report(rows, variants), self.report(rows[::-1], variants)
        pd.testing.assert_frame_equal(left[1], right[1])
        pd.testing.assert_frame_equal(left[2], right[2])
        self.assertAlmostEqual(left[0]["metric_deltas"]["both_gt_missing"]["weighted_top1_pp"], 20.)
        self.assertAlmostEqual(left[0]["metric_deltas"]["original"]["weighted_top1_pp"], 55.)

    def test_same_candidates_count_and_target_presence_are_mandatory(self):
        for values in ({"candidate_ids_sha256": "b" * 64}, {"candidate_count": 9},
                       {"candidate_found": False, "target_rank": 0}):
            rows = records()
            rows[0].update(values)
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "must preserve"):
                self.report(rows)
        rows = records()
        for row in rows:
            row["candidate_ids_sha256"] = "invalid"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.report(rows)

    def test_unpaired_data_identity_and_coverage_fail_closed(self):
        for rows in (records()[:-1], records() + [copy.deepcopy(records()[0])],
                     [dict(row, target_vessel_id="OTHER") if i == 0 else row for i, row in enumerate(records())],
                     [dict(row, model="OTHER") if i == 0 else row for i, row in enumerate(records())]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.report(rows)
        rows = records() + records("both_gt_missing")
        rows[-1]["vessel_name"] = "DIFFERENT"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.report(rows, ("original", "both_gt_missing"))
        with self.assertRaisesRegex(ValueError, "Coverage"):
            self.report(coverage={"queries": 1})

    def test_single_language_has_null_weighted_results(self):
        rows = [row for row in records() if row["has_hangul"]]
        summary, _, _, markdown = self.report(rows, coverage={"queries": 2})
        self.assertIsNone(summary["metrics"]["original"][REFERENCE]["weighted_top1"])
        self.assertIsNone(summary["metric_deltas"]["original"]["weighted_top1_pp"])
        self.assertIn("—", markdown)
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
