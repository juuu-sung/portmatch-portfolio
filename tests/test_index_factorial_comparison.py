"""Synthetic end-to-end proof for the frozen four-way asset diagnostic."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.data import QUERY_COLUMNS
from portmatch.index_factorial_comparison import compare_index_factorial, full_feature_prediction
from portmatch.index_scale_comparison import compare_index_scale
from portmatch.name_feature_cache import transform_cache
from portmatch.ranker_commands import RankedMatcher, add_training_parsers, run_training_command
from portmatch.ranking import VesselRanker
from portmatch.retrieval_comparison import TIMING_COLUMNS, VARIANTS
from portmatch.robust_retrieval import RobustMatchingPipeline


RUNNERS = ("search80_category80", "search80_category100",
           "search100_category80", "search100_category100")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser():
    result = argparse.ArgumentParser()
    add_training_parsers(result.add_subparsers(dest="command", required=True))
    return result


class IndexFactorialCommandTests(unittest.TestCase):
    def test_command_requires_reference_and_routes_frozen_profile(self):
        arguments = parser().parse_args(["compare-index-factorial", "--model", "model",
            "--reference", "reference", "--fold", "2", "--output", "output"])
        with patch("portmatch.index_factorial_comparison.compare_index_factorial") as compare:
            run_training_command(arguments)
        compare.assert_called_once_with("model", "reference", "output",
            train_path="work/data/train.csv", catalog_path="work/data/vessel_catalog.csv",
            subset_seed=20261027, expected_fold=2, max_queries=None, progress_every=100)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(["compare-index-factorial", "--model", "model",
                                "--fold", "0", "--output", "output"])

    def test_profile_overrides_and_invalid_limits_are_rejected(self):
        base = ["compare-index-factorial", "--model", "m", "--reference", "r",
                "--fold", "0", "--output", "o"]
        for extra in (["--fold", "1"], ["--rounds", "20"], ["--threads", "2"],
                      ["--small-fraction", ".75"], ["--original-mass", ".75"]):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser().parse_args([*base, *extra])
        with TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            for settings in ({"max_queries": 0}, {"max_queries": True},
                             {"progress_every": 0}, {"progress_every": True},
                             {"variants": ("both_gt_missing",)},
                             {"variants": ("original", "original")},
                             {"variants": ("original", "unknown")}):
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    compare_index_factorial("missing", "missing-reference", output, **settings)
                self.assertFalse(output.exists())


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM unavailable")
class IndexFactorialSyntheticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        spec = importlib.util.spec_from_file_location("_factorial_fixture",
            Path(__file__).with_name("test_robust_training.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        train, catalog = module.fixture()
        # Several source/target categories exercise complete probability/support
        # tables rather than only a one-category constant fixture.
        catalog.loc[catalog.index % 3 == 0, "flag_name"] = "Panama"
        catalog.loc[catalog.index % 4 == 0, "vessel_type"] = "Tanker"
        train.loc[train.index % 6 < 2, "flag_name"] = "파나마"
        train.loc[train.index % 8 < 2, "vessel_type"] = "유조선"
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        train.to_csv(cls.train_path, index=False, encoding="utf-8-sig")
        catalog.to_csv(cls.catalog_path, index=False, encoding="utf-8-sig")
        cls.model = cls.root / "model"
        cls.reference = cls.root / "reference"
        source, transformed = cls.root / "source", cls.root / "transformed"
        cli = parser()
        with redirect_stdout(io.StringIO()):
            run_training_command(cli.parse_args(["prepare-ranker", "--train", str(cls.train_path),
                "--catalog", str(cls.catalog_path), "--output", str(source), "--fold", "0",
                "--retrieval-profile", "robust", "--gt-augmentation", "missing",
                "--original-mass", "0.5"]))
            transform_cache(source, transformed, train_path=cls.train_path, catalog_path=cls.catalog_path)
            run_training_command(cli.parse_args(["train-ranker", "--data", str(transformed),
                "--output", str(cls.model), "--rounds", "300", "--threads", "6",
                "--exclude-features", "retrieval_tonnage_rank"]))
            compare_index_scale(cls.model, cls.reference, train_path=cls.train_path,
                catalog_path=cls.catalog_path, expected_fold=0)

    def compare(self, output, reference=None, **kwargs):
        with redirect_stdout(io.StringIO()), patch.object(VesselRanker, "fit",
                side_effect=AssertionError("Diagnostic must never train model weights")):
            return compare_index_factorial(self.model, reference or self.reference, output,
                train_path=self.train_path, catalog_path=self.catalog_path,
                expected_fold=0, max_queries=4, progress_every=100, **kwargs)

    def assert_not_completed(self, output):
        if (output / "summary.json").exists():
            self.assertIs(json.loads((output / "summary.json").read_text("utf-8"))["complete"], False)
        self.assertFalse((output / "report.md").exists())

    def test_real_four_way_inference_preserves_inputs_and_exports_exact_model_features(self):
        output = self.root / "factorial"
        protected = [self.train_path, self.catalog_path,
                     *[p for d in (self.model, self.reference) for p in d.rglob("*") if p.is_file()]]
        before = {path: digest(path) for path in protected}
        forwarded_queries = []
        original = RobustMatchingPipeline.retrieve

        def retrieve(pipeline, query):
            forwarded_queries.append(dict(query))
            return original(pipeline, query)

        with patch.object(RobustMatchingPipeline, "retrieve", new=retrieve):
            summary = self.compare(output)
        self.assertEqual(before, {path: digest(path) for path in protected})
        for field in ("complete", "weights_unchanged", "training_assets_unchanged",
                      "protected_files_unchanged", "ranker_shared_by_all_runners"):
            self.assertIs(summary[field], True)
        self.assertIs(summary["training_performed"], False)
        self.assertIs(summary["submission_created"], False)
        self.assertEqual(summary, json.loads((output / "summary.json").read_text("utf-8")))
        files = {"summary.json", "predictions.csv", "row_changes.csv", "vessel_changes.csv",
                 "partitions.csv", "report.md", "candidate_features.csv",
                 "candidate_comparisons.csv", "feature_changes.csv"}
        self.assertEqual({p.name for p in output.iterdir()}, files)
        for path in output.iterdir():
            if path.name != "summary.json":
                self.assertEqual(digest(path), summary["output_files_sha256"][path.name])
        for query in forwarded_queries:
            self.assertEqual(set(query), set(QUERY_COLUMNS) - {"sample_id"})

        predictions = pd.read_csv(output / "predictions.csv")
        self.assertEqual(set(predictions.model), set(RUNNERS))
        self.assertEqual(set(predictions.variant), set(VARIANTS))
        self.assertEqual(set(predictions.has_hangul), {True, False})
        n = summary["coverage"]["queries"]
        self.assertEqual(len(predictions), n * 5 * 4)
        self.assertEqual(len(forwarded_queries), len(predictions) + 4)
        self.assertFalse(predictions.duplicated(["sample_id", "variant", "model"]).any())
        self.assertTrue((predictions[list(TIMING_COLUMNS)].to_numpy() >= 0).all())
        np.testing.assert_allclose(predictions.total_seconds,
                                   predictions[list(TIMING_COLUMNS[:3])].sum(axis=1))
        for _, group in predictions.groupby(["variant", "sample_id"]):
            group = group.set_index("model")
            for left, right in ((RUNNERS[0], RUNNERS[1]), (RUNNERS[2], RUNNERS[3])):
                for field in ("candidate_ids_sha256", "candidate_count", "candidate_found"):
                    self.assertEqual(group.loc[left, field], group.loc[right, field])

        membership = pd.read_csv(output / "partitions.csv")
        small = membership.loc[membership.selected_small]
        full = membership.loc[membership.selected_full]
        valid = membership.loc[membership.role.eq("valid")]
        self.assertFalse(set(full.target_vessel_id) & set(valid.target_vessel_id))
        self.assertLess(set(small.sample_id), set(full.sample_id))
        self.assertEqual(small.target_vessel_id.nunique(), math.floor(.8 * full.target_vessel_id.nunique()))
        self.assertTrue(membership.groupby("target_vessel_id").selected_small.nunique().eq(1).all())
        self.assertTrue(set(predictions.sample_id).issubset(set(valid.sample_id)))

        old = pd.read_csv(self.reference / "predictions.csv").set_index(["variant", "sample_id", "model"])
        for row in predictions.itertuples(index=False):
            legacy_name = {RUNNERS[0]: "index_80", RUNNERS[3]: "index_100"}.get(row.model)
            if legacy_name:
                legacy = old.loc[(row.variant, row.sample_id, legacy_name)]
                for name in ("target_vessel_id", "predicted_vessel_id", "candidate_count",
                             "target_rank", "candidate_found", "candidate_ids_sha256"):
                    self.assertEqual(getattr(row, name), legacy[name])
                self.assertAlmostEqual(row.top_score, legacy.top_score, places=10)

        names = summary["feature_schema"]["selected_features"]
        categories = summary["feature_schema"]["category_features"]
        other = summary["feature_schema"]["non_category_features"]
        self.assertEqual((len(names), len(categories), len(other)), (86, 16, 70))
        self.assertEqual(set(names), set(categories) | set(other))
        self.assertNotIn("retrieval_tonnage_rank", names)
        features = pd.read_csv(output / "candidate_features.csv")
        self.assertTrue(set(names).issubset(features.columns))
        self.assertFalse(features.duplicated(["sample_id", "variant", "candidate_vessel_id", "model"]).any())
        for key, group in predictions.groupby(["sample_id", "variant"]):
            expected = set(group.predicted_vessel_id) | set(group.target_vessel_id)
            detail = features.loc[features.sample_id.eq(key[0]) & features.variant.eq(key[1])]
            self.assertEqual(set(detail.candidate_vessel_id), expected)
            self.assertEqual(len(detail), len(expected) * 4)
            for _, candidate in detail.groupby("candidate_vessel_id"):
                self.assertEqual(set(candidate.model), set(RUNNERS))
        present = features.loc[features.present]
        matcher = RankedMatcher(self.model)
        scores = matcher.ranker.predict(present[names].to_numpy(dtype=np.float32), names)
        # The emitted values are exactly the selected float32 inputs to LightGBM.
        np.testing.assert_allclose(scores, present.score.to_numpy(), rtol=0, atol=1e-10)
        for _, group in features.groupby(["sample_id", "variant", "candidate_vessel_id"]):
            group = group.set_index("model")
            for left, right in ((RUNNERS[0], RUNNERS[1]), (RUNNERS[2], RUNNERS[3])):
                self.assertEqual(group.loc[left, "present"], group.loc[right, "present"])
                if group.loc[left, "present"]:
                    np.testing.assert_equal(group.loc[left, other].to_numpy(dtype=float),
                                            group.loc[right, other].to_numpy(dtype=float))
            for left, right in ((RUNNERS[0], RUNNERS[2]), (RUNNERS[1], RUNNERS[3])):
                if group.loc[left, "present"] and group.loc[right, "present"]:
                    np.testing.assert_equal(group.loc[left, categories].to_numpy(dtype=float),
                                            group.loc[right, categories].to_numpy(dtype=float))
        absent = features.loc[~features.present]
        self.assertTrue(absent[["score", *names]].isna().all().all())
        changes = pd.read_csv(output / "row_changes.csv")
        self.assertEqual(len(changes), n * len(VARIANTS) * 5)
        self.assertEqual(changes.contrast.nunique(), 5)

    def test_reference_hash_corruption_is_rejected_without_completed_report(self):
        copied = self.root / "tampered_reference"
        shutil.copytree(self.reference, copied)
        with (copied / "predictions.csv").open("a", encoding="utf-8") as stream:
            stream.write("tampered\n")
        output = self.root / "tampered_output"
        with self.assertRaises(ValueError):
            self.compare(output, reference=copied)
        self.assert_not_completed(output)

    def test_live_diagonal_parity_failure_keeps_partial_output_incomplete(self):
        # Retain valid reference provenance and model inputs, but simulate a
        # scoring/export regression that must fail the final diagonal audit.
        def changed_score(*args, **kwargs):
            record, details = full_feature_prediction(*args, **kwargs)
            record = {**record, "top_score": record["top_score"] + .01}
            return record, details

        output = self.root / "live_parity_failure"
        with patch("portmatch.index_factorial_comparison.full_feature_prediction", new=changed_score):
            with self.assertRaisesRegex(ValueError, "parity"):
                self.compare(output)
        self.assertTrue((output / "summary.json").exists())
        self.assert_not_completed(output)

    def test_incomplete_reference_and_existing_output_are_rejected(self):
        copied = self.root / "incomplete_reference"
        shutil.copytree(self.reference, copied)
        path = copied / "summary.json"
        state = json.loads(path.read_text("utf-8"))
        state["complete"] = False
        path.write_text(json.dumps(state), encoding="utf-8")
        output = self.root / "incomplete_output"
        with self.assertRaises(ValueError):
            self.compare(output, reference=copied)
        self.assert_not_completed(output)
        existing = self.root / "existing_output"
        existing.mkdir()
        marker = existing / "summary.json"
        marker.write_text("Keep previous experiment", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.compare(existing)
        self.assertEqual(marker.read_text("utf-8"), "Keep previous experiment")

    def test_full_feature_prediction_uses_no_label_and_marks_absent_target_after_scoring(self):
        matcher = RankedMatcher(self.model)
        # Use one held-out row and a name-independent impossible target to prove
        # the target is only inspected after retrieval/features/model scoring.
        membership = pd.read_csv(self.reference / "partitions.csv")
        identifier = membership.loc[membership.role.eq("valid"), "sample_id"].iloc[0]
        row = pd.read_csv(self.train_path).set_index("sample_id").loc[identifier].to_dict()
        row.update(sample_id=identifier, injected_label="must not be forwarded")
        target = row["target_vessel_id"]
        forwarded = []
        original = matcher.pipeline.retrieve

        def retrieve(query):
            forwarded.append(dict(query))
            return original(query)

        with patch.object(matcher.pipeline, "retrieve", new=retrieve):
            found, details = full_feature_prediction(matcher, row, target)
            absent, other = full_feature_prediction(matcher, row, "IMPOSSIBLE_TARGET")
        self.assertEqual(found["predicted_vessel_id"], absent["predicted_vessel_id"])
        self.assertEqual(found["top_score"], absent["top_score"])
        self.assertFalse(absent["candidate_found"])
        self.assertEqual(absent["target_rank"], 0)
        pd.testing.assert_frame_equal(details, other)
        for query in forwarded:
            self.assertEqual(set(query), set(QUERY_COLUMNS) - {"sample_id"})
        names = list(matcher.selection.selected_features)
        self.assertTrue(set(names).issubset(details.columns))
        self.assertEqual(len(names), 86)
        self.assertTrue(all(details[name].dtype == np.float32 for name in names))
        np.testing.assert_allclose(matcher.ranker.predict(details[names].to_numpy(), names),
                                   details.score.to_numpy(), rtol=0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
