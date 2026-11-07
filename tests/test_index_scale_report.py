import copy
import io
import json
import unittest

import pandas as pd

from portmatch.index_scale_report import build_index_scale_report


def record(sample, target, language, runner, *, correct, found=True, predicted=None,
           variant="original"):
    large = runner == "index_100"
    return {"sample_id": sample, "target_vessel_id": target, "has_hangul": language,
        "vessel_name": "한글" if language else "LATIN", "model": runner, "variant": variant,
        "predicted_vessel_id": target if correct else (predicted or "WRONG"),
        "candidate_found": found, "target_rank": 1 if correct else (2 if found else 0),
        "candidate_count": 7 if large else 5, "candidate_ids_sha256": "b" * 64 if large else "a" * 64,
        "top_score": 3. if large else 2., "score_margin": .3 if large else .2,
        "retrieval_seconds": 2. if large else 1., "feature_seconds": 3. if large else 2.,
        "rank_seconds": 4. if large else 3., "total_seconds": 9. if large else 6.,
        "top_retrieval_phonetic_score": .7 if large else .6,
        "target_retrieval_phonetic_score": (.8 if large else .5) if found else None,
        "top_type_probability": .8 if large else .5,
        "target_catalog_name": target if found else None,
        "top_catalog_name": target if correct else "Other"}


def paired(variant="original"):
    cases = [
        ("a1", "A", True, {"correct": False, "found": False}, {"correct": True}),
        ("a2", "A", True, {"correct": False}, {"correct": True}),
        ("b1", "B", False, {"correct": True}, {"correct": False, "found": False}),
        ("c1", "C", False, {"correct": False}, {"correct": False, "predicted": "NEW_WRONG"}),
        ("d1", "D", True, {"correct": True}, {"correct": True}),
    ]
    return [record(sample, target, language, runner, variant=variant, **options)
            for sample, target, language, small, large in cases
            for runner, options in (("index_80", small), ("index_100", large))]


class IndexScaleReportTests(unittest.TestCase):
    def report(self, rows=None, *, variants=("original",), coverage=None):
        return build_index_scale_report(pd.DataFrame(paired() if rows is None else rows),
            coverage={"queries": 5, "fold": 0, "folds": 5, "heldout_queries": 5,
                      "diagnostic_sample": False} if coverage is None else coverage,
            proof={"heldout_excluded": True}, startup={"build_seconds": 1.},
            parity={"saved_original_match": True}, variants=variants)

    def test_paired_metrics_direction_and_error_reasons(self):
        summary, changes, vessels, report = self.report()
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["reference_runner"], "index_80")
        self.assertEqual(summary["experiment_runner"], "index_100")
        metrics = summary["metrics"]["original"]
        self.assertAlmostEqual(metrics["index_80"]["weighted_top1"], .7 / 3 + .3 / 2)
        self.assertAlmostEqual(metrics["index_100"]["weighted_top1"], .7)
        delta = summary["metric_deltas"]["original"]
        self.assertAlmostEqual(delta["weighted_top1_pp"], 100 * (.7 - (.7 / 3 + .3 / 2)))
        self.assertAlmostEqual(delta["weighted_candidate_recall_pp"], 100 * (.85 - (.7 * 2 / 3 + .3)))
        self.assertEqual(delta["all"]["candidate_count_delta"]["mean"], 2)
        self.assertEqual(delta["all"]["timing"]["total_seconds"]["sum_seconds_delta"], 15)
        transitions = summary["transitions"]["original"]
        self.assertEqual((transitions["corrected"], transitions["corrected_vessels"]), (2, 1))
        self.assertEqual((transitions["regressed"], transitions["regressed_vessels"]), (1, 1))
        self.assertEqual(transitions["candidate_gained"], 1)
        self.assertEqual(transitions["candidate_lost"], 1)
        self.assertEqual(transitions["still_wrong_prediction_changed"], 1)
        self.assertEqual(transitions["hangul"]["corrected"], 2)
        self.assertEqual(transitions["non_hangul"]["regressed"], 1)
        rows = changes.set_index("sample_id")
        self.assertEqual(rows.loc["a1", "error_transition"], "candidate_missing->correct")
        self.assertEqual(rows.loc["a2", "error_transition"], "ranking_error->correct")
        self.assertEqual(rows.loc["b1", "error_transition"], "correct->candidate_missing")
        self.assertTrue(pd.isna(rows.loc["a1", "target_retrieval_phonetic_score_delta"]))
        self.assertAlmostEqual(rows.loc["a2", "target_retrieval_phonetic_score_delta"], .3)
        self.assertTrue(changes.top_score_delta.eq(1).all())
        self.assertIn("predicted_vessel_id_index_100", changes)
        self.assertNotIn("predicted_vessel_id_experiment", changes)
        self.assertIn("100% − 80%", report)
        self.assertIn("원인을 확정", report)
        self.assertEqual(vessels.loc[vessels.target_vessel_id.eq("A"), "net_corrected"].iloc[0], 2)
        json.dumps(summary, ensure_ascii=False, allow_nan=False)

    def test_reordering_and_same_variant_pairing_preserve_changes(self):
        rows = paired() + paired("both_gt_missing")
        for row in rows:
            if row["variant"] == "both_gt_missing" and row["sample_id"] == "d1" and row["model"] == "index_100":
                row.update(predicted_vessel_id="WRONG", target_rank=2)
        left = self.report(rows, variants=("original", "both_gt_missing"))
        right = self.report(rows[::-1], variants=("original", "both_gt_missing"))
        pd.testing.assert_frame_equal(left[1], right[1])
        self.assertEqual(left[0]["transitions"]["original"]["regressed"], 1)
        self.assertEqual(left[0]["transitions"]["both_gt_missing"]["regressed"], 2)

    def test_vessel_aggregation_preserves_language_and_csv_diagnostics(self):
        rows = paired()
        rows += [record("a3", "A", False, runner, correct=True) for runner in ("index_80", "index_100")]
        _, changes, vessels, _ = self.report(rows, coverage={"queries": 6})
        self.assertEqual(len(vessels.loc[vessels.target_vessel_id.eq("A")]), 2)
        restored = pd.read_csv(io.StringIO(changes.to_csv(index=False)))
        self.assertIn("target_retrieval_phonetic_score_index_80", restored)
        self.assertIn("top_type_probability_delta", restored)
        self.assertEqual(restored.error_reason_index_100.eq("candidate_missing").sum(), 1)

    def test_single_language_reports_no_weighted_score(self):
        rows = [row for row in paired() if row["has_hangul"]]
        summary, _, _, report = self.report(rows, coverage={"queries": 3})
        self.assertIsNone(summary["metrics"]["original"]["index_80"]["weighted_top1"])
        self.assertIsNone(summary["metric_deltas"]["original"]["weighted_candidate_recall_pp"])
        self.assertIsNone(summary["metrics"]["original"]["index_100"]["non_hangul"]["candidate_count"]["max"])
        self.assertIn("—", report)
        json.dumps(summary, allow_nan=False)

    def test_invalid_coverage_pairing_identity_and_variants_are_rejected(self):
        mutations = [
            lambda rows: rows + [copy.deepcopy(rows[0])],
            lambda rows: rows[:-1],
            lambda rows: [dict(row, target_vessel_id="OTHER") if i == 0 else row for i, row in enumerate(rows)],
            lambda rows: [dict(row, model="unknown") if i == 0 else row for i, row in enumerate(rows)],
            lambda rows: [dict(row, sample_id=None) if i == 0 else row for i, row in enumerate(rows)],
            lambda rows: [dict(row, has_hangul="False") if i == 0 else row for i, row in enumerate(rows)],
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.report(mutate(paired()))
        with self.assertRaisesRegex(ValueError, "Coverage"):
            self.report(coverage={"queries": 99})
        with self.assertRaises(ValueError):
            self.report(variants=("original", "original"))
        with self.assertRaises(ValueError):
            self.report(variants=("invented",))

    def test_cross_variant_identity_and_uniform_query_coverage_required(self):
        rows = paired() + paired("both_gt_missing")
        for row in rows:
            if row["variant"] == "both_gt_missing" and row["sample_id"] == "a1":
                row["vessel_name"] = "DIFFERENT"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.report(rows, variants=("original", "both_gt_missing"))
        rows = paired() + [row for row in paired("both_gt_missing") if row["sample_id"] != "a1"]
        with self.assertRaisesRegex(ValueError, "identical sample IDs"):
            self.report(rows, variants=("original", "both_gt_missing"))

    def test_inconsistent_metrics_and_invalid_timing_rejected(self):
        for changes in ({"target_rank": 1}, {"candidate_count": 0}, {"total_seconds": -1},
                        {"total_seconds": 7}, {"top_score": float("inf")},
                        {"target_retrieval_phonetic_score": float("inf")}):
            rows = paired()
            rows[0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.report(rows)


if __name__ == "__main__":
    unittest.main()
