import copy
import io
import json
import unittest

import pandas as pd

from portmatch.index_factorial_report import CONTRASTS, RUNNERS, build_index_factorial_report


A, B, C, D = RUNNERS


def records(variant="original"):
    outcomes = {
        "h1": ("H1", True, (False, True, False, True)),
        "h2": ("H2", True, (True, False, True, True)),
        "n1": ("N1", False, (False, False, True, False)),
        "n2": ("N2", False, (True, True, False, True)),
    }
    rows = []
    for sample, (target, language, corrects) in outcomes.items():
        for position, (runner, correct) in enumerate(zip(RUNNERS, corrects)):
            large_search = runner in (C, D)
            found = not (sample == "n1" and not large_search)
            retrieval = 2. if large_search else 1.
            feature = (2., 3., 4., 6.)[position]
            rows.append({"sample_id": sample, "target_vessel_id": target,
                "has_hangul": language, "vessel_name": "한글" if language else "LATIN",
                "model": runner, "variant": variant,
                "predicted_vessel_id": target if correct else "WRONG",
                "candidate_found": found, "target_rank": 1 if correct else (2 if found else 0),
                "candidate_count": 7 if large_search else 5,
                "candidate_ids_sha256": ("b" if large_search else "a") * 64,
                "top_score": 3. + position, "score_margin": .1 + position / 10.,
                "retrieval_seconds": retrieval, "feature_seconds": feature,
                "rank_seconds": 1., "total_seconds": retrieval + feature + 1.,
                "top_type_probability": .2 + position / 20.,
                "target_retrieval_name_score": .5 if found else None})
    return rows


class IndexFactorialReportTests(unittest.TestCase):
    def report(self, rows=None, *, coverage=None, variants=("original",)):
        return build_index_factorial_report(pd.DataFrame(records() if rows is None else rows),
            coverage={"queries": 4, "fold": 0, "folds": 5, "heldout_queries": 4,
                      "diagnostic_sample": False} if coverage is None else coverage,
            proof={"heldout_excluded": True}, startup={"build_seconds": 1.},
            parity={"diagonal_match": True}, variants=variants)

    def test_factorial_direction_interaction_and_five_paired_contrasts(self):
        summary, changes, vessels, markdown = self.report()
        self.assertIs(summary["complete"], False)
        self.assertEqual(summary["runners"], list(RUNNERS))
        self.assertEqual(summary["contrasts"], CONTRASTS)
        self.assertEqual(len(changes), 4 * 5)
        self.assertEqual(set(changes.contrast), set(CONTRASTS))
        self.assertEqual(len(vessels), 4 * 5)
        metrics = summary["metrics"]["original"]
        for runner in (A, B, C):
            self.assertAlmostEqual(metrics[runner]["weighted_top1"], .5)
        self.assertAlmostEqual(metrics[D]["weighted_top1"], .85)
        deltas = summary["metric_deltas"]["original"]
        self.assertAlmostEqual(deltas["category_at_search80"]["weighted_top1_pp"], 0.)
        self.assertAlmostEqual(deltas["category_at_search100"]["weighted_top1_pp"], 35.)
        self.assertAlmostEqual(deltas["search_at_category80"]["weighted_top1_pp"], 0.)
        self.assertAlmostEqual(deltas["search_at_category100"]["weighted_top1_pp"], 35.)
        self.assertAlmostEqual(deltas["diagonal"]["weighted_top1_pp"], 35.)
        interaction = summary["factorial_interactions"]["original"]
        self.assertAlmostEqual(interaction["weighted_top1_pp"], 35.)
        self.assertAlmostEqual(interaction["weighted_candidate_recall_pp"], 0.)
        self.assertEqual(interaction["all"]["candidate_count_interaction"]["mean"], 0.)
        self.assertEqual(interaction["all"]["timing"]["total_seconds"]["sum_seconds_interaction"], 4.)
        self.assertEqual(interaction["all"]["timing"]["total_seconds"]["mean_seconds_interaction"], 1.)
        self.assertNotIn("sum_ratio", interaction["all"]["timing"]["total_seconds"])
        self.assertIn("(D−C)−(B−A)", markdown)
        self.assertIn("서로 다른 다섯 입력 조건의 점수를 평균내지", markdown)
        self.assertIn("통계적 유의성", markdown)
        json.dumps(summary, ensure_ascii=False, allow_nan=False)

    def test_error_rows_are_unambiguous_and_preserve_candidate_vs_ranking_errors(self):
        summary, changes, vessels, _ = self.report()
        self.assertIn("predicted_vessel_id_reference", changes)
        self.assertIn("predicted_vessel_id_experiment", changes)
        self.assertNotIn("predicted_vessel_id_index_80", changes)
        self.assertNotIn("model", changes)
        contrast = changes.loc[changes.contrast.eq("search_at_category80")].set_index("sample_id")
        self.assertTrue(contrast.reference_runner.eq(A).all())
        self.assertTrue(contrast.experiment_runner.eq(C).all())
        self.assertEqual(contrast.loc["n1", "error_transition"], "candidate_missing->correct")
        self.assertEqual(contrast.loc["n2", "error_transition"], "correct->ranking_error")
        self.assertEqual(contrast.loc["n1", "candidate_count_delta"], 2)
        self.assertTrue(pd.isna(contrast.loc["n1", "target_retrieval_name_score_delta"]))
        transitions = summary["transitions"]["original"]["search_at_category80"]
        self.assertEqual(transitions["candidate_gained"], 1)
        self.assertEqual(transitions["non_hangul"]["candidate_gained"], 1)
        self.assertEqual(transitions["errors_reference"]["candidate_missing"], 1)
        self.assertEqual(transitions["errors_experiment"]["candidate_missing"], 0)
        restored = pd.read_csv(io.StringIO(changes.to_csv(index=False)))
        self.assertIn("top_type_probability_delta", restored)
        self.assertIn("reference_runner", vessels)
        self.assertIn("experiment_runner", vessels)

    def test_single_language_interactions_are_null_not_misleading_weighted_scores(self):
        rows = [row for row in records() if row["has_hangul"]]
        summary, _, _, markdown = self.report(rows, coverage={"queries": 2})
        self.assertIsNone(summary["metrics"]["original"][D]["weighted_top1"])
        self.assertIsNone(summary["metric_deltas"]["original"]["diagonal"]["weighted_top1_pp"])
        interaction = summary["factorial_interactions"]["original"]
        self.assertIsNone(interaction["weighted_top1_pp"])
        self.assertIsNone(interaction["non_hangul"]["candidate_count_interaction"]["mean"])
        self.assertIsNone(interaction["non_hangul"]["timing"]["total_seconds"]["mean_seconds_interaction"])
        self.assertEqual(interaction["non_hangul"]["timing"]["total_seconds"]["sum_seconds_interaction"], 0.)
        self.assertAlmostEqual(interaction["hangul"]["top1_accuracy_pp"], 50.)
        self.assertIn("—", markdown)
        json.dumps(summary, allow_nan=False)

    def test_same_search_candidate_membership_count_and_presence_must_match(self):
        for runner in (B, D):
            for updates in ({"candidate_ids_sha256": "c" * 64}, {"candidate_count": 8},
                            {"candidate_found": False, "target_rank": 0,
                             "predicted_vessel_id": "WRONG"}):
                rows = records()
                for row in rows:
                    if row["sample_id"] == "h1" and row["model"] == runner:
                        row.update(updates)
                with self.subTest(runner=runner, updates=updates), self.assertRaisesRegex(ValueError, "category scope"):
                    self.report(rows)
        rows = records()
        for row in rows:
            row["candidate_ids_sha256"] = "not-a-valid-hash"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.report(rows)

    def test_row_order_does_not_change_reports_and_variants_are_kept_separate(self):
        rows = records() + records("both_gt_missing")
        for row in rows:
            if row["variant"] == "both_gt_missing" and row["sample_id"] == "h2" and row["model"] == D:
                row.update(predicted_vessel_id="WRONG", target_rank=2)
        left = self.report(rows, variants=("original", "both_gt_missing"))
        right = self.report(rows[::-1], variants=("original", "both_gt_missing"))
        pd.testing.assert_frame_equal(left[1], right[1])
        pd.testing.assert_frame_equal(left[2], right[2])
        self.assertAlmostEqual(left[0]["factorial_interactions"]["original"]["weighted_top1_pp"], 35.)
        self.assertAlmostEqual(left[0]["factorial_interactions"]["both_gt_missing"]["weighted_top1_pp"], 0.)

    def test_missing_duplicate_mismatched_queries_and_invalid_variants_rejected(self):
        mutations = [
            lambda rows: rows + [copy.deepcopy(rows[0])],
            lambda rows: rows[:-1],
            lambda rows: [dict(row, target_vessel_id="OTHER") if i == 0 else row for i, row in enumerate(rows)],
            lambda rows: [dict(row, model="unknown") if i == 0 else row for i, row in enumerate(rows)],
            lambda rows: [dict(row, has_hangul="False") if i == 0 else row for i, row in enumerate(rows)],
            lambda rows: [dict(row, total_seconds=-1.) if i == 0 else row for i, row in enumerate(rows)],
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.report(mutate(records()))
        with self.assertRaisesRegex(ValueError, "Coverage"):
            self.report(coverage={"queries": 1})
        for variants in (("original", "original"), ("invented",), ()):
            with self.subTest(variants=variants), self.assertRaises(ValueError):
                self.report(variants=variants)

    def test_identity_is_stable_across_variants_and_all_four_runners(self):
        rows = records() + records("both_gt_missing")
        for row in rows:
            if row["variant"] == "both_gt_missing" and row["sample_id"] == "h1":
                row["vessel_name"] = "DIFFERENT"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.report(rows, variants=("original", "both_gt_missing"))
        rows = records() + [row for row in records("both_gt_missing") if row["sample_id"] != "h1"]
        with self.assertRaisesRegex(ValueError, "identical sample IDs"):
            self.report(rows, variants=("original", "both_gt_missing"))


if __name__ == "__main__":
    unittest.main()
