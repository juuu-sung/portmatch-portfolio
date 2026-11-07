"""Synthetic native-ranker integration for search100/category80 versus100."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.category_scope_assets import EXPERIMENT, REFERENCE, RUNNERS
from portmatch.category_scope_comparison import compare_category_scope
from portmatch.data import QUERY_COLUMNS
from portmatch.index_factorial_comparison import full_feature_prediction
from portmatch.name_feature_cache import transform_cache
from portmatch.ranker_commands import RankedMatcher, add_training_parsers, run_training_command
from portmatch.ranking import VesselRanker
from portmatch.retrieval_comparison import TIMING_COLUMNS, VARIANTS
from portmatch.robust_retrieval import RobustMatchingPipeline


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    result = argparse.ArgumentParser()
    add_training_parsers(result.add_subparsers(dest="command", required=True))
    return result


class CategoryScopeCommandTests(unittest.TestCase):
    def test_new_folds_route_without_a_previous_reference_run(self):
        for fold in (1, 3, 4):
            with self.subTest(fold=fold):
                arguments = parser().parse_args(["compare-category-scope", "--model", "model",
                    "--fold", str(fold), "--output", "output"])
                with patch("portmatch.category_scope_comparison.compare_category_scope") as compare:
                    run_training_command(arguments)
                compare.assert_called_once_with("model", "output", train_path="work/data/train.csv",
                    catalog_path="work/data/vessel_catalog.csv", subset_seed=20261027,
                    expected_fold=fold, max_queries=None, progress_every=100)

    def test_fixed_profile_and_invalid_options_are_rejected(self):
        base = ["compare-category-scope", "--model", "m", "--fold", "1", "--output", "o"]
        for extra in (["--fold", "5"], ["--subset-seed", "20261028"], ["--rounds", "20"],
                      ["--threads", "2"], ["--small-fraction", ".75"], ["--original-mass", ".75"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args([*base, *extra])
        with TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            for settings in ({"max_queries": 0}, {"max_queries": True}, {"progress_every": 0},
                             {"progress_every": True}, {"variants": ("both_gt_missing",)},
                             {"variants": ("original", "original")},
                             {"variants": ("original", "unknown")},
                             {"subset_seed": 20261028}, {"expected_fold": 5}):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    compare_category_scope("missing", output, **settings)
                self.assertFalse(output.exists())


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM unavailable")
class CategoryScopeSyntheticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        spec = importlib.util.spec_from_file_location("_category_integration_fixture",
            Path(__file__).with_name("test_robust_training.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        train, catalog = module.fixture()
        catalog.loc[catalog.index % 3 == 0, "flag_name"] = "Panama"
        catalog.loc[catalog.index % 4 == 0, "vessel_type"] = "Tanker"
        train.loc[train.index % 6 < 2, "flag_name"] = "파나마"
        train.loc[train.index % 8 < 2, "vessel_type"] = "유조선"
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        train.to_csv(cls.train_path, index=False, encoding="utf-8-sig")
        catalog.to_csv(cls.catalog_path, index=False, encoding="utf-8-sig")
        cls.model = cls.root / "model"
        source, transformed = cls.root / "source", cls.root / "transformed"
        cli = parser()
        with redirect_stdout(io.StringIO()):
            run_training_command(cli.parse_args(["prepare-ranker", "--train", str(cls.train_path),
                "--catalog", str(cls.catalog_path), "--output", str(source), "--fold", "1",
                "--retrieval-profile", "robust", "--gt-augmentation", "missing", "--original-mass", "0.5"]))
            transform_cache(source, transformed, train_path=cls.train_path, catalog_path=cls.catalog_path)
            run_training_command(cli.parse_args(["train-ranker", "--data", str(transformed),
                "--output", str(cls.model), "--rounds", "300", "--threads", "6",
                "--exclude-features", "retrieval_tonnage_rank"]))

    def compare(self, output, **kwargs):
        with redirect_stdout(io.StringIO()), patch.object(VesselRanker, "fit",
                side_effect=AssertionError("Diagnostic must never train model weights")):
            return compare_category_scope(self.model, output, train_path=self.train_path,
                catalog_path=self.catalog_path, expected_fold=1, progress_every=100, **kwargs)

    def assert_incomplete(self, output):
        if (output / "summary.json").exists():
            self.assertIs(json.loads((output / "summary.json").read_text("utf-8"))["complete"], False)
        self.assertFalse((output / "report.md").exists())

    def test_full_comparison_preserves_assets_exports_exact_inputs_and_checks_saved_predictions(self):
        output = self.root / "full_category"
        protected = [self.train_path, self.catalog_path, *[p for p in self.model.rglob("*") if p.is_file()]]
        before = {path: digest(path) for path in protected}
        forwarded, original_retrieve = [], RobustMatchingPipeline.retrieve

        def retrieve(pipeline, query):
            forwarded.append(dict(query))
            return original_retrieve(pipeline, query)

        with patch.object(RobustMatchingPipeline, "retrieve", new=retrieve):
            summary = self.compare(output)
        self.assertEqual(before, {path: digest(path) for path in protected})
        for field in ("complete", "weights_unchanged", "training_assets_unchanged",
                      "protected_files_unchanged", "ranker_shared_by_both_runners", "diagnostic_code_unchanged"):
            self.assertIs(summary[field], True)
        self.assertIs(summary["training_performed"], False)
        self.assertIs(summary["submission_created"], False)
        self.assertFalse(summary["coverage"]["diagnostic_sample"])
        self.assertEqual(summary, json.loads((output / "summary.json").read_text("utf-8")))
        expected_files = {"summary.json", "predictions.csv", "row_changes.csv", "vessel_changes.csv",
            "partitions.csv", "report.md", "candidate_features.csv", "candidate_comparisons.csv", "feature_changes.csv"}
        self.assertEqual({p.name for p in output.iterdir()}, expected_files)
        self.assertEqual(len(summary["output_files_sha256"]), 8)
        for filename, expected in summary["output_files_sha256"].items():
            self.assertEqual(digest(output / filename), expected)
        for query in forwarded:
            self.assertEqual(set(query), set(QUERY_COLUMNS) - {"sample_id"})

        predictions = pd.read_csv(output / "predictions.csv")
        self.assertEqual(set(predictions.model), set(RUNNERS))
        self.assertEqual(set(predictions.variant), set(VARIANTS))
        self.assertEqual(set(predictions.has_hangul), {True, False})
        n = summary["coverage"]["queries"]
        self.assertEqual(len(predictions), n * 10)
        self.assertEqual(len(forwarded), len(predictions) + 2)
        self.assertFalse(predictions.duplicated(["sample_id", "variant", "model"]).any())
        self.assertTrue((predictions[list(TIMING_COLUMNS)].to_numpy() >= 0).all())
        np.testing.assert_allclose(predictions.total_seconds, predictions[list(TIMING_COLUMNS[:3])].sum(axis=1))
        for runner in RUNNERS:
            self.assertEqual(set(predictions.loc[predictions.model.eq(runner), "runner_position"]), {0, 1})
        for _, group in predictions.groupby(["variant", "sample_id"]):
            group = group.set_index("model")
            self.assertEqual(set(group.runner_position), {0, 1})
            for field in ("candidate_ids_sha256", "candidate_count", "candidate_found"):
                self.assertEqual(group.loc[REFERENCE, field], group.loc[EXPERIMENT, field])
        self.assertEqual(summary["feature_isolation"]["same_search_pairs"], n * len(VARIANTS))

        saved = pd.read_csv(self.model / "validation_predictions.csv").set_index("sample_id")
        reference = predictions.loc[predictions.variant.eq("original") & predictions.model.eq(REFERENCE)]
        self.assertEqual(summary["parity"]["saved_original"]["matched_queries"], n)
        for row in reference.itertuples(index=False):
            for field in ("target_vessel_id", "predicted_vessel_id", "candidate_count", "target_rank", "has_hangul"):
                self.assertEqual(getattr(row, field), saved.loc[row.sample_id, field])
        membership = pd.read_csv(output / "partitions.csv")
        small, full, valid = (membership.loc[mask] for mask in
            (membership.selected_small, membership.selected_full, membership.role.eq("valid")))
        self.assertFalse(set(full.target_vessel_id) & set(valid.target_vessel_id))
        self.assertFalse(set(small.sample_id) & set(valid.sample_id))
        self.assertLess(set(small.sample_id), set(full.sample_id))
        self.assertEqual(small.target_vessel_id.nunique(), math.floor(.8 * full.target_vessel_id.nunique()))
        self.assertTrue(membership.groupby("target_vessel_id").selected_small.nunique().eq(1).all())
        self.assertEqual(set(predictions.sample_id), set(valid.sample_id))

        names = summary["feature_schema"]["selected_features"]
        categories = summary["feature_schema"]["category_features"]
        other = summary["feature_schema"]["non_category_features"]
        self.assertEqual((len(names), len(categories), len(other)), (86, 16, 70))
        self.assertNotIn("retrieval_tonnage_rank", names)
        features = pd.read_csv(output / "candidate_features.csv")
        self.assertFalse(features.duplicated(["sample_id", "variant", "candidate_vessel_id", "model"]).any())
        for key, group in predictions.groupby(["sample_id", "variant"]):
            expected = set(group.predicted_vessel_id) | set(group.target_vessel_id)
            detail = features.loc[features.sample_id.eq(key[0]) & features.variant.eq(key[1])]
            self.assertEqual(set(detail.candidate_vessel_id), expected)
            self.assertEqual(len(detail), len(expected) * 2)
        present = features.loc[features.present]
        matcher = RankedMatcher(self.model)
        scores = matcher.ranker.predict(present[names].to_numpy(dtype=np.float32), names)
        np.testing.assert_allclose(scores, present.score.to_numpy(), rtol=0, atol=1e-10)
        for _, group in features.groupby(["sample_id", "variant", "candidate_vessel_id"]):
            group = group.set_index("model")
            self.assertEqual(set(group.index), set(RUNNERS))
            self.assertEqual(group.loc[REFERENCE, "present"], group.loc[EXPERIMENT, "present"])
            if group.loc[REFERENCE, "present"]:
                np.testing.assert_equal(group.loc[REFERENCE, other].to_numpy(dtype=float),
                                        group.loc[EXPERIMENT, other].to_numpy(dtype=float))
        self.assertTrue(features.loc[~features.present, ["score", *names]].isna().all().all())
        changes = pd.read_csv(output / "row_changes.csv")
        self.assertEqual(len(changes), n * len(VARIANTS))
        self.assertEqual(set(changes.reference_runner), {REFERENCE})
        self.assertEqual(set(changes.experiment_runner), {EXPERIMENT})
        feature_changes = pd.read_csv(output / "feature_changes.csv")
        self.assertGreater(len(feature_changes), 0)
        self.assertTrue(feature_changes.is_category_feature.all())

    def test_saved_original_disagreement_leaves_output_incomplete(self):
        def changed_result(*args, **kwargs):
            record, details = full_feature_prediction(*args, **kwargs)
            return {**record, "candidate_count": record["candidate_count"] + 1}, details

        output = self.root / "parity_failure"
        with patch("portmatch.category_scope_comparison.full_feature_prediction", new=changed_result):
            with self.assertRaisesRegex(ValueError, "differs from saved predictions"):
                self.compare(output, max_queries=2)
        self.assertTrue((output / "summary.json").exists())
        self.assert_incomplete(output)

    def test_noncategory_isolation_failure_leaves_output_incomplete(self):
        output = self.root / "isolation_failure"
        with patch("portmatch.category_scope_evidence.validate_category_feature_isolation",
                   side_effect=ValueError("Non-category mismatch")):
            with self.assertRaisesRegex(ValueError, "Non-category mismatch"):
                self.compare(output, max_queries=2)
        self.assertTrue((output / "summary.json").exists())
        self.assert_incomplete(output)

    def test_existing_output_is_preserved(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "summary.json"
        marker.write_text("Keep previous experiment", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.compare(output)
        self.assertEqual(marker.read_text("utf-8"), "Keep previous experiment")


if __name__ == "__main__":
    unittest.main()
