import argparse
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from portmatch.ranker_comparison import add_comparison_parser, compare_rankers


FEATURES = ["name_ratio", "retrieval_tonnage_rank", "flag_probability"]


def prediction(sample, vessel, language, correct, *, found=True):
    return {"sample_id": sample, "target_vessel_id": vessel, "has_hangul": language,
            "predicted_vessel_id": vessel if correct else "WRONG",
            "candidate_found": found, "target_rank": 1 if correct else (2 if found else 0),
            "candidate_count": 10, "top_score": .1, "score_margin": .05}


class RankerComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base_rows = [
            [prediction("a1", "A", True, False), prediction("a2", "A", True, True),
             prediction("b1", "B", False, True)],
            [prediction("c1", "C", True, False), prediction("c2", "C", True, False),
             prediction("c3", "C", True, True), prediction("d1", "D", False, False)],
        ]
        self.exp_rows = copy.deepcopy(self.base_rows)
        for fold, row, correct in ((0, 0, True), (0, 1, False), (1, 0, True), (1, 3, True)):
            record = self.exp_rows[fold][row]
            record["predicted_vessel_id"] = record["target_vessel_id"] if correct else "WRONG"
            record["target_rank"] = 1 if correct else 2
        self.baselines = [self.write_run(fold, rows, False) for fold, rows in enumerate(self.base_rows)]
        self.experiments = [self.write_run(fold, rows, True) for fold, rows in enumerate(self.exp_rows)]

    def write_run(self, fold, rows, experiment):
        directory = self.root / ("new" if experiment else "old") / ("arbitrary_name_" + str(9 - fold))
        directory.mkdir(parents=True)
        dataset = {
            "schema_version": 1, "complete": True, "outer_split": "vessel", "outer_folds": 2,
            "outer_fold": fold, "inner_split": "mixed", "inner_folds": 3, "seed": 20261022,
            "feature_names": FEATURES, "retrieval": {"max_total": None},
            "sampling": {"hard_negatives": 96, "random_negatives": 32},
            "counts": {"valid_groups": len(rows), "valid_pairs": len(rows) * 10,
                       "outer_validation_rows": len(rows), "outer_training_rows": 7 - len(rows)},
            "provenance": {"train_sha256": "same train", "catalog_sha256": "same catalog",
                           "feature_code_sha256": "same features"},
            "sampled": False, "max_train_queries": None, "max_valid_queries": None,
            "validation_sampling": "none; every retrieved candidate retained",
        }
        selected = [name for name in FEATURES if not experiment or name != "retrieval_tonnage_rank"]
        summary = {"complete": True, "schema_version": 1, "dataset": dataset,
                   "ranker": {"n_estimators": 300, "n_jobs": 6, "seed": 20261022}}
        if experiment:
            summary["feature_selection"] = {"schema_version": 1, "source_feature_names": FEATURES,
                                            "selected_features": selected,
                                            "excluded_features": ["retrieval_tonnage_rank"],
                                            "include_patterns": [], "exclude_patterns": ["retrieval_tonnage_rank"]}
            summary["trained_feature_names"] = selected
            summary["cache_directory"] = str(self.root / "cache" / str(fold))
        (directory / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (directory / "ranker.json").write_text(json.dumps({"feature_names": selected}), encoding="utf-8")
        pd.DataFrame(rows).to_csv(directory / "validation_predictions.csv", index=False)
        return directory

    def mutate_summary(self, directory, change):
        path = directory / "training_summary.json"
        summary = json.loads(path.read_text("utf-8"))
        change(summary)
        path.write_text(json.dumps(summary), encoding="utf-8")

    def mutate_predictions(self, directory, change):
        path = directory / "validation_predictions.csv"
        frame = pd.read_csv(path)
        change(frame).to_csv(path, index=False)

    def compare(self):
        return compare_rankers(self.baselines, self.experiments, self.root / "report")

    def test_aligned_full_oof_pooled_metrics_and_vessel_changes(self):
        # Reverse both path and CSV order; identities and fold metadata must govern alignment.
        self.mutate_predictions(self.experiments[0], lambda frame: frame.iloc[::-1])
        result = compare_rankers(self.baselines[::-1], self.experiments, self.root / "report")
        self.assertTrue(result["coverage"]["full_oof"])
        self.assertEqual(result["coverage"]["queries"], 7)
        self.assertAlmostEqual(result["pooled"]["baseline"]["weighted_top1"], .43)
        self.assertAlmostEqual(result["pooled"]["experiment"]["weighted_top1"], .72)
        self.assertAlmostEqual(result["pooled"]["delta_percentage_points"]["weighted_top1"], 29.)
        mean_fold = sum(item["baseline"]["weighted_top1"] for item in result["per_fold"]) / 2
        self.assertNotAlmostEqual(result["pooled"]["baseline"]["weighted_top1"], mean_fold)
        changes = result["pooled"]["transitions"]["all"]
        self.assertEqual((changes["corrected"], changes["regressed"], changes["still_wrong"],
                          changes["unchanged_correct"]), (3, 1, 1, 2))
        self.assertEqual(changes["vessels"], {"improved": 2, "regressed": 0, "mixed": 1,
                                            "unchanged": 1, "total": 4, "net_improved": 2,
                                            "net_regressed": 0, "net_unchanged": 2})
        self.assertEqual(result["features"]["removed"], ["retrieval_tonnage_rank"])
        for filename in ("summary.json", "report.md", "row_changes.csv", "vessel_changes.csv", "per_fold_metrics.csv"):
            self.assertTrue((self.root / "report" / filename).is_file())
        rows = pd.read_csv(self.root / "report" / "row_changes.csv")
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows.loc[rows.sample_id.eq("a1"), "status"].iloc[0], "corrected")
        report = (self.root / "report" / "report.md").read_text("utf-8")
        self.assertIn("개발용 검증", report)
        self.assertIn("전체 OOF", report)
        self.assertEqual(json.loads((self.root / "report" / "summary.json").read_text("utf-8")), result)

    def test_partial_folds_are_labeled_and_output_is_not_overwritten(self):
        result = compare_rankers(self.baselines[:1], self.experiments[:1], self.root / "report")
        self.assertFalse(result["coverage"]["full_oof"])
        self.assertEqual(result["coverage"]["included_folds"], [0])
        self.assertEqual(result["coverage"]["missing_folds"], [1])
        self.assertIn("부분 검증", (self.root / "report" / "report.md").read_text("utf-8"))
        self.assertIn("포함하지 않은 fold: 1", (self.root / "report" / "report.md").read_text("utf-8"))
        before = (self.root / "report" / "summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.compare()
        self.assertEqual(before, (self.root / "report" / "summary.json").read_bytes())

    def test_missing_fold_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "same outer folds"):
            compare_rankers(self.baselines, self.experiments[:1], self.root / "report")
        self.assertFalse((self.root / "report").exists())

    def test_duplicate_fold_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate baseline outer_fold"):
            compare_rankers(self.baselines * 2, self.experiments, self.root / "report")

    def test_sample_id_sets_and_duplicates_rejected(self):
        original = (self.experiments[0] / "validation_predictions.csv").read_bytes()
        for new_value, message in (("new_id", "sample_id sets differ"), ("a2", "Duplicate sample_id")):
            with self.subTest(new_value=new_value):
                (self.experiments[0] / "validation_predictions.csv").write_bytes(original)
                self.mutate_predictions(self.experiments[0], lambda frame: frame.assign(
                    sample_id=[new_value, *frame.sample_id.iloc[1:]]))
                with self.assertRaisesRegex(ValueError, message):
                    self.compare()

    def test_language_mismatch_rejected(self):
        self.mutate_predictions(self.experiments[0], lambda frame: frame.assign(has_hangul=[False, True, False]))
        with self.assertRaisesRegex(ValueError, "has_hangul differ"):
            self.compare()

    def test_truth_mismatch_rejected_even_if_correctness_remains_consistent(self):
        def change(frame):
            frame.loc[0, ["target_vessel_id", "predicted_vessel_id"]] = "NEW_TARGET"
            return frame
        self.mutate_predictions(self.experiments[0], change)
        with self.assertRaisesRegex(ValueError, "target_vessel_id differ"):
            self.compare()

    def test_rank_and_identity_mismatch_rejected(self):
        self.mutate_predictions(self.experiments[0], lambda frame: frame.assign(target_rank=[2, 2, 1]))
        with self.assertRaisesRegex(ValueError, "identity and target rank disagree"):
            self.compare()

    def test_changed_candidate_counts_rejected_even_if_totals_same(self):
        self.mutate_predictions(self.experiments[0], lambda frame: frame.assign(candidate_count=[11, 9, 10]))
        with self.assertRaisesRegex(ValueError, "candidate_count differ"):
            self.compare()

    def test_changed_candidate_recall_rejected(self):
        def change(frame):
            frame.loc[1, "candidate_found"] = False
            frame.loc[1, "target_rank"] = 0
            return frame
        self.mutate_predictions(self.experiments[0], change)
        with self.assertRaisesRegex(ValueError, "candidate_found differ"):
            self.compare()

    def test_changed_training_configuration_and_dataset_rejected(self):
        path = self.experiments[0] / "training_summary.json"
        original = path.read_bytes()
        edits = [lambda s: s["ranker"].update(n_estimators=1000),
                 lambda s: s["dataset"]["sampling"].update(hard_negatives=48),
                 lambda s: s["dataset"]["provenance"].update(train_sha256="other"),
                 lambda s: s["dataset"].update(seed=42),
                 lambda s: s.update(complete=False)]
        for edit in edits:
            with self.subTest(edit=edit):
                path.write_bytes(original)
                self.mutate_summary(self.experiments[0], edit)
                with self.assertRaises(ValueError):
                    compare_rankers(self.baselines[:1], self.experiments[:1], self.root / "report")

    def test_saved_schema_disagreement_rejected(self):
        self.mutate_summary(self.experiments[0], lambda s: s.update(trained_feature_names=FEATURES))
        with self.assertRaisesRegex(ValueError, "Conflicting trained feature schemas"):
            self.compare()

    def test_inconsistent_feature_selection_state_rejected(self):
        self.mutate_summary(self.experiments[0], lambda s: s["feature_selection"].update(excluded_features=[]))
        with self.assertRaisesRegex(ValueError, "Saved feature selection"):
            self.compare()

    def test_cross_fold_sample_and_vessel_duplicates_rejected(self):
        original = (self.baselines[1] / "validation_predictions.csv").read_bytes()
        for field, value, message in (("sample_id", "a1", "sample_id across folds"),
                                      ("target_vessel_id", "A", "multiple vessel folds")):
            with self.subTest(field=field):
                (self.baselines[1] / "validation_predictions.csv").write_bytes(original)
                def change(frame):
                    frame.loc[0, field] = value
                    return frame
                self.mutate_predictions(self.baselines[1], change)
                with self.assertRaisesRegex(ValueError, message):
                    self.compare()

    def test_single_language_does_not_fabricate_weighted_score(self):
        for directory in (self.baselines[0], self.experiments[0]):
            self.mutate_predictions(directory, lambda frame: frame.assign(has_hangul=True))
        result = compare_rankers(self.baselines[:1], self.experiments[:1], self.root / "report")
        self.assertIsNone(result["pooled"]["baseline"]["weighted_top1"])
        self.assertIsNone(result["pooled"]["delta_percentage_points"]["weighted_top1"])
        self.assertEqual(result["pooled"]["transitions"]["non_hangul"]["vessels"]["total"], 0)
        self.assertIn("N/A", (self.root / "report" / "report.md").read_text("utf-8"))

    def test_parser_accepts_multiple_directories(self):
        parser = argparse.ArgumentParser()
        add_comparison_parser(parser.add_subparsers(dest="command"))
        args = parser.parse_args(["compare-rankers", "--baseline", "b0", "b1", "--experiment", "e1", "e0", "--output", "report"])
        self.assertEqual(args.baseline, ["b0", "b1"])
        self.assertEqual(args.experiment, ["e1", "e0"])

    def derived_pair(self):
        """Model manifests only; real conversion/integrity are covered end-to-end."""
        from portmatch.tonnage_rank import SOURCE_FEATURE, TARGET_FEATURE, transform_spec
        from portmatch.feature_selection import FeatureSelection

        bdir, edir = self.baselines[0], self.experiments[0]
        base = json.loads((bdir / "training_summary.json").read_text("utf-8"))
        base["feature_selection"] = FeatureSelection.resolve(FEATURES, exclude=[SOURCE_FEATURE]).to_dict()
        base["trained_feature_names"] = base["feature_selection"]["selected_features"]
        base["cache_metadata_sha256"] = "parent-cache-hash"
        (bdir / "training_summary.json").write_text(json.dumps(base), encoding="utf-8")
        (bdir / "ranker.json").write_text(json.dumps({"feature_names": base["trained_feature_names"]}), encoding="utf-8")

        exp = json.loads((edir / "training_summary.json").read_text("utf-8"))
        names = [TARGET_FEATURE if n == SOURCE_FEATURE else n for n in FEATURES]
        exp["dataset"]["feature_names"] = names
        exp["dataset"]["feature_transform"] = transform_spec()
        exp["dataset"]["derivation"] = {"manifest_path": "derivation.json",
                                          "manifest_sha256": "manifest-hash",
                                          "parent_metadata_sha256": "parent-cache-hash"}
        # Omit selection deliberately: fallback must use experiment's new schema.
        exp.pop("feature_selection")
        exp.pop("trained_feature_names")
        cache = self.root / "derived_cache"
        cache.mkdir()
        cache_metadata = cache / "metadata.json"
        cache_metadata.write_text(json.dumps(exp["dataset"]), encoding="utf-8")
        exp["cache_directory"] = str(cache)
        exp["cache_metadata_sha256"] = hashlib.sha256(cache_metadata.read_bytes()).hexdigest()
        (edir / "training_summary.json").write_text(json.dumps(exp), encoding="utf-8")
        (edir / "ranker.json").write_text(json.dumps({"feature_names": names}), encoding="utf-8")
        return bdir, edir, cache, base, exp

    def test_derived_experiment_requires_verified_lineage_and_uses_new_schema(self):
        bdir, edir, cache, base, exp = self.derived_pair()
        proof = {"verified": True, "files_verified": 7}
        with patch("portmatch.tonnage_rank.verify_derivation", return_value=proof) as verify:
            result = compare_rankers([bdir], [edir], self.root / "report")
        verify.assert_called_once_with(str(cache), "parent-cache-hash", base["dataset"])
        self.assertEqual(result["comparison_type"], "verified_derived_feature_experiment")
        self.assertEqual(result["features"]["added"], ["retrieval_tonnage_min_rank"])
        self.assertEqual(result["features"]["removed"], [])
        self.assertEqual(result["feature_selection"]["experiment"]["source_feature_names"],
                         exp["dataset"]["feature_names"])
        self.assertEqual(result["sources"][0]["cache_verification"]["proof"], proof)
        self.assertIn("파일 해시를 다시 검사", (self.root / "report/report.md").read_text("utf-8"))

    def test_derived_cache_integrity_failure_is_not_bypassed(self):
        bdir, edir, _, _, _ = self.derived_pair()
        with patch("portmatch.tonnage_rank.verify_derivation", side_effect=ValueError("source/destination mismatch")):
            with self.assertRaisesRegex(ValueError, "source/destination mismatch"):
                compare_rankers([bdir], [edir], self.root / "report")
        self.assertFalse((self.root / "report").exists())

    def test_derived_comparison_rejects_missing_parent_binding(self):
        bdir, edir, _, _, _ = self.derived_pair()
        self.mutate_summary(bdir, lambda s: s.pop("cache_metadata_sha256"))
        with self.assertRaisesRegex(ValueError, "cache paths and metadata hashes"):
            compare_rankers([bdir], [edir], self.root / "report")

    def test_derived_comparison_rejects_metadata_changed_after_training(self):
        bdir, edir, cache, _, _ = self.derived_pair()
        with (cache / "metadata.json").open("a", encoding="utf-8") as stream:
            stream.write(" ")
        with self.assertRaisesRegex(ValueError, "differs from the trained model"):
            compare_rankers([bdir], [edir], self.root / "report")

    def test_derived_comparison_rejects_unrelated_feature_selection_changes(self):
        from portmatch.feature_selection import FeatureSelection
        bdir, edir, _, _, exp = self.derived_pair()
        names = exp["dataset"]["feature_names"]
        self.mutate_summary(edir, lambda s: s.update(
            feature_selection=FeatureSelection.resolve(names, exclude=["flag_probability"]).to_dict()))
        (edir / "ranker.json").write_text(json.dumps({"feature_names": names[:-1]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same shared selected features"):
            compare_rankers([bdir], [edir], self.root / "report")

    def test_derived_comparison_rejects_unknown_transform_and_reverse_direction(self):
        bdir, edir, _, _, _ = self.derived_pair()
        with self.assertRaisesRegex(ValueError, "source-cache baseline"):
            compare_rankers([edir], [bdir], self.root / "report")
        self.mutate_summary(edir, lambda s: s["dataset"]["feature_transform"].update(name="other_transform"))
        with self.assertRaisesRegex(ValueError, "Unknown or changed feature transform"):
            compare_rankers([bdir], [edir], self.root / "report")

    def test_new_feature_without_semantic_metadata_cannot_use_same_cache_path(self):
        _, edir, _, _, _ = self.derived_pair()
        def remove_transform(summary):
            summary["dataset"].pop("feature_transform")
            summary["dataset"].pop("derivation")
        self.mutate_summary(edir, remove_transform)
        with self.assertRaisesRegex(ValueError, "missing its transform metadata"):
            compare_rankers([edir], [edir], self.root / "report")


if __name__ == "__main__":
    unittest.main()
