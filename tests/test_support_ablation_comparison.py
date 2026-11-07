"""Synthetic native-ranker integration for shared-cache 86-to-82 ablation."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.data import QUERY_COLUMNS
from portmatch.name_feature_cache import transform_cache
from portmatch.ranker_commands import RankedMatcher, add_training_parsers, run_training_command
from portmatch.ranking import VesselRanker
from portmatch.retrieval_comparison import TIMING_COLUMNS, VARIANTS
from portmatch.robust_retrieval import RobustMatchingPipeline
from portmatch.support_ablation_comparison import compare_support_ablation
from portmatch.support_ablation_report import EXPERIMENT, REFERENCE, RUNNERS, SUPPORT_FEATURES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    result = argparse.ArgumentParser()
    add_training_parsers(result.add_subparsers(dest="command", required=True))
    return result


class SupportAblationCommandTests(unittest.TestCase):
    def test_command_routes_the_two_models_and_expected_fold(self):
        for fold in range(5):
            with self.subTest(fold=fold):
                args = parser().parse_args(["compare-support-ablation", "--baseline", "baseline",
                    "--experiment", "experiment", "--fold", str(fold), "--output", "output"])
                with patch("portmatch.support_ablation_comparison.compare_support_ablation") as compare:
                    run_training_command(args)
                compare.assert_called_once_with("baseline", "experiment", "output",
                    train_path="work/data/train.csv", catalog_path="work/data/vessel_catalog.csv",
                    expected_fold=fold, max_queries=None, progress_every=100)

    def test_profile_overrides_and_invalid_limits_fail_before_output_creation(self):
        base = ["compare-support-ablation", "--baseline", "b", "--experiment", "e",
                "--fold", "1", "--output", "o"]
        for extra in (["--fold", "5"], ["--rounds", "20"], ["--threads", "2"],
                      ["--subset-seed", "20261027"], ["--original-mass", ".75"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args([*base, *extra])
        with TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            for settings in ({"max_queries": 0}, {"max_queries": True}, {"progress_every": 0},
                             {"progress_every": True}, {"variants": ("both_gt_missing",)},
                             {"variants": ("original", "original")},
                             {"variants": ("original", "unknown")}, {"expected_fold": 5}):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    compare_support_ablation("missing", "other-missing", output, **settings)
                self.assertFalse(output.exists())


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM unavailable")
class SupportAblationSyntheticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        spec = importlib.util.spec_from_file_location("_support_ablation_integration_fixture",
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
        cls.models = {REFERENCE: cls.root / "baseline", EXPERIMENT: cls.root / "experiment"}
        cls.source, cls.cache = cls.root / "source", cls.root / "transformed"
        cli = parser()
        with redirect_stdout(io.StringIO()):
            run_training_command(cli.parse_args(["prepare-ranker", "--train", str(cls.train_path),
                "--catalog", str(cls.catalog_path), "--output", str(cls.source), "--fold", "1",
                "--retrieval-profile", "robust", "--gt-augmentation", "missing", "--original-mass", "0.5"]))
            transform_cache(cls.source, cls.cache, train_path=cls.train_path, catalog_path=cls.catalog_path)
            before = {p: digest(p) for p in cls.cache.rglob("*") if p.is_file()}
            for name in RUNNERS:
                excluded = ["retrieval_tonnage_rank", *(SUPPORT_FEATURES if name == EXPERIMENT else ())]
                run_training_command(cli.parse_args(["train-ranker", "--data", str(cls.cache),
                    "--output", str(cls.models[name]), "--rounds", "300", "--threads", "6",
                    "--exclude-features", *excluded]))
            after = {p: digest(p) for p in cls.cache.rglob("*") if p.is_file()}
            if before != after:
                raise AssertionError("Feature-selected training changed the shared source cache")

    def compare(self, output, **kwargs):
        with redirect_stdout(io.StringIO()), patch.object(VesselRanker, "fit",
                side_effect=AssertionError("Comparison must never train model weights")):
            return compare_support_ablation(self.models[REFERENCE], self.models[EXPERIMENT], output,
                train_path=self.train_path, catalog_path=self.catalog_path,
                expected_fold=1, progress_every=100, **kwargs)

    def assert_incomplete(self, output):
        self.assertTrue((output / "summary.json").exists())
        self.assertIs(json.loads((output / "summary.json").read_text("utf-8"))["complete"], False)
        self.assertFalse((output / "report.md").exists())

    def test_full_comparison_preserves_cache_and_exports_exact_selected_inputs(self):
        output = self.root / "full"
        protected = [self.train_path, self.catalog_path,
            *[p for folder in (*self.models.values(), self.cache, self.source)
              for p in folder.rglob("*") if p.is_file()]]
        before = {p: digest(p) for p in protected}
        forwarded, original_retrieve = [], RobustMatchingPipeline.retrieve

        def retrieve(pipeline, query):
            forwarded.append(dict(query))
            return original_retrieve(pipeline, query)

        with patch.object(RobustMatchingPipeline, "retrieve", new=retrieve):
            summary = self.compare(output)
        self.assertEqual(before, {p: digest(p) for p in protected})
        for field in ("complete", "weights_unchanged_during_comparison", "training_assets_unchanged",
                      "protected_files_unchanged", "diagnostic_code_unchanged"):
            self.assertIs(summary[field], True)
        self.assertIs(summary["training_performed"], False)
        self.assertIs(summary["submission_created"], False)
        self.assertIs(summary["preservation"]["shared_cache_unchanged"], True)
        self.assertIs(summary["preservation"]["models_and_full_T_assets_unchanged"], True)
        self.assertFalse(summary["coverage"]["diagnostic_sample"])
        self.assertEqual(summary, json.loads((output / "summary.json").read_text("utf-8")))
        for filename, expected in summary["output_files_sha256"].items():
            self.assertEqual(digest(output / filename), expected)
        self.assertTrue(forwarded)
        for query in forwarded:
            self.assertEqual(set(query), set(QUERY_COLUMNS) - {"sample_id"})

        predictions = pd.read_csv(output / "predictions.csv")
        self.assertEqual(set(predictions.model), set(RUNNERS))
        self.assertEqual(set(predictions.variant), set(VARIANTS))
        self.assertEqual(set(predictions.has_hangul), {True, False})
        n = summary["coverage"]["queries"]
        self.assertEqual(len(predictions), n * len(VARIANTS) * len(RUNNERS))
        self.assertFalse(predictions.duplicated(["sample_id", "variant", "model"]).any())
        self.assertTrue((predictions[list(TIMING_COLUMNS)].to_numpy() >= 0).all())
        np.testing.assert_allclose(predictions.total_seconds, predictions[list(TIMING_COLUMNS[:3])].sum(axis=1))
        for _, group in predictions.groupby(["variant", "sample_id"]):
            group = group.set_index("model")
            self.assertEqual(set(group.runner_position), {0, 1})
            for field in ("candidate_ids_sha256", "candidate_count", "candidate_found"):
                self.assertEqual(group.loc[REFERENCE, field], group.loc[EXPERIMENT, field])
        self.assertEqual(summary["feature_isolation"]["same_search_pairs"], n * len(VARIANTS))
        self.assertEqual(summary["feature_isolation"]["shared_feature_candidate_pairs"],
            int(predictions.loc[predictions.model.eq(REFERENCE), "candidate_count"].sum()))
        self.assertIs(summary["feature_isolation"]["verified"], True)

        membership = pd.read_csv(output / "partitions.csv")
        fitting, valid = (membership.loc[membership.role.eq(role)] for role in ("train", "valid"))
        self.assertFalse(set(fitting.target_vessel_id) & set(valid.target_vessel_id))
        self.assertFalse(set(fitting.sample_id) & set(valid.sample_id))
        self.assertEqual(set(predictions.sample_id), set(valid.sample_id))
        self.assertTrue(fitting[["search100", "category100"]].all().all())
        self.assertFalse(valid[["search100", "category100"]].any().any())

        features = pd.read_csv(output / "candidate_features.csv")
        self.assertFalse(features.duplicated(["sample_id", "variant", "candidate_vessel_id", "model"]).any())
        matchers = {name: RankedMatcher(path) for name, path in self.models.items()}
        names = {name: list(matcher.selection.selected_features) for name, matcher in matchers.items()}
        self.assertEqual((len(names[REFERENCE]), len(names[EXPERIMENT])), (86, 82))
        self.assertEqual(names[EXPERIMENT], [v for v in names[REFERENCE] if v not in SUPPORT_FEATURES])
        for runner in RUNNERS:
            self.assertEqual(summary["parity"]["saved_original"][runner]["matched_queries"], n)
            saved = pd.read_csv(self.models[runner] / "validation_predictions.csv").set_index("sample_id")
            actual = predictions.loc[predictions.variant.eq("original") & predictions.model.eq(runner)]
            for row in actual.itertuples(index=False):
                for field in ("target_vessel_id", "predicted_vessel_id", "candidate_count", "target_rank", "has_hangul"):
                    self.assertEqual(getattr(row, field), saved.loc[row.sample_id, field])
            present = features.loc[features.present & features.model.eq(runner)]
            scores = matchers[runner].ranker.predict(present[names[runner]].to_numpy(dtype=np.float32), names[runner])
            np.testing.assert_allclose(scores, present.score.to_numpy(), rtol=0, atol=1e-10)
            self.assertTrue(features.loc[features.model.eq(runner), "feature_count"].eq(len(names[runner])).all())
        experiment = features.loc[features.model.eq(EXPERIMENT)]
        self.assertTrue(experiment[list(SUPPORT_FEATURES)].isna().all().all())
        self.assertTrue(experiment.omitted_features.map(lambda value: set(json.loads(value)) == set(SUPPORT_FEATURES)).all())
        for key, group in predictions.groupby(["sample_id", "variant"]):
            expected = set(group.predicted_vessel_id) | set(group.target_vessel_id)
            detail = features.loc[features.sample_id.eq(key[0]) & features.variant.eq(key[1])]
            self.assertEqual(set(detail.candidate_vessel_id), expected)
            self.assertEqual(len(detail), len(expected) * 2)
        for _, group in features.groupby(["sample_id", "variant", "candidate_vessel_id"]):
            group = group.set_index("model")
            self.assertEqual(set(group.index), set(RUNNERS))
            self.assertEqual(group.loc[REFERENCE, "present"], group.loc[EXPERIMENT, "present"])
            if group.loc[REFERENCE, "present"]:
                np.testing.assert_equal(group.loc[REFERENCE, names[EXPERIMENT]].to_numpy(dtype=float),
                                        group.loc[EXPERIMENT, names[EXPERIMENT]].to_numpy(dtype=float))
        self.assertTrue(features.loc[~features.present, ["score", *names[REFERENCE]]].isna().all().all())
        changes = pd.read_csv(output / "row_changes.csv")
        self.assertEqual(len(changes), n * len(VARIANTS))
        self.assertEqual(set(changes.reference_runner), {REFERENCE})
        self.assertEqual(set(changes.experiment_runner), {EXPERIMENT})

    def test_either_saved_original_disagreement_prevents_completion(self):
        from portmatch.support_ablation_comparison import full_feature_prediction

        for runner, count in ((REFERENCE, 86), (EXPERIMENT, 82)):
            def changed_result(matcher, *args, **kwargs):
                record, details = full_feature_prediction(matcher, *args, **kwargs)
                if len(matcher.selection.selected_features) == count:
                    record = {**record, "predicted_vessel_id": "deliberate-saved-parity-mismatch"}
                return record, details

            output = self.root / ("parity_" + runner)
            with self.subTest(runner=runner), patch(
                    "portmatch.support_ablation_comparison.full_feature_prediction", new=changed_result):
                with self.assertRaisesRegex(ValueError, "differs from saved predictions"):
                    self.compare(output, max_queries=2)
            self.assert_incomplete(output)

    def test_shared_feature_mismatch_prevents_completion(self):
        from portmatch.support_ablation_comparison import full_feature_prediction

        def changed_result(matcher, *args, **kwargs):
            record, details = full_feature_prediction(matcher, *args, **kwargs)
            if len(matcher.selection.selected_features) == 82 and len(details):
                details = details.copy()
                details.iloc[0, details.columns.get_loc("name_ratio")] = 123.0
            return record, details

        output = self.root / "features_mismatch"
        with patch("portmatch.support_ablation_comparison.full_feature_prediction", new=changed_result):
            with self.assertRaisesRegex(ValueError, "altered shared 82"):
                self.compare(output, max_queries=2)
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
