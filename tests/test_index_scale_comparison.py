"""Fixed-weight index diagnostics on real synthetic training artifacts."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.data import QUERY_COLUMNS
from portmatch.index_scale_comparison import (
    DETAIL_COLUMNS, DIAGNOSTIC_FEATURES, RUNNERS, _changed_details,
    compare_index_scale, detailed_prediction,
)
from portmatch.name_feature_cache import transform_cache
from portmatch.ranker_commands import add_training_parsers, run_training_command
from portmatch.ranking import VesselRanker
from portmatch.retrieval_comparison import TIMING_COLUMNS, VARIANTS
from portmatch.robust_retrieval import RobustMatchingPipeline


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DetailedIndexPredictionTests(unittest.TestCase):
    def matcher(self, indices=(1, 0)):
        calls = {"retrieve": [], "features": [], "predict": []}
        candidates = pd.DataFrame({"catalog_index": indices})
        values = pd.DataFrame({name: [.2, .8] for name in DIAGNOSTIC_FEATURES})

        def retrieve(query):
            calls["retrieve"].append(dict(query))
            return candidates.copy()

        def features(query, retrieved):
            calls["features"].append((dict(query), retrieved.copy()))
            return values.copy()

        def predict(matrix, names):
            calls["predict"].append((matrix.copy(), list(names)))
            return np.array([.1, .9])

        pipeline = SimpleNamespace(
            catalog=pd.DataFrame({"vessel_id": ["A", "B", "C"],
                                  "vessel_name": ["Alpha", "Beta", "Gamma"]}),
            retrieve=retrieve, features=features,
        )
        matcher = SimpleNamespace(pipeline=pipeline, catalog_ids=np.array(["A", "B", "C"]),
            feature_transform=None,
            selection=SimpleNamespace(selected_features=DIAGNOSTIC_FEATURES,
                                      apply=lambda matrix, names: matrix),
            ranker=SimpleNamespace(predict=predict))
        return matcher, calls

    def test_target_inspection_is_post_score_aligned_and_does_not_retrieve_twice(self):
        matcher, calls = self.matcher()
        query = {"vessel_name": "알파", "sample_id": "private-id",
                 "target_vessel_id": "forbidden", "injected_label": "forbidden"}
        with patch("portmatch.feature_transforms.transform_features", side_effect=lambda *args: args[-1]):
            found, details = detailed_prediction(matcher, query, "B")
            absent, _ = detailed_prediction(matcher, query, "C")
        self.assertEqual(found["predicted_vessel_id"], "A")
        self.assertEqual(absent["predicted_vessel_id"], "A")
        self.assertEqual(found["target_rank"], 2)
        self.assertTrue(found["candidate_found"])
        self.assertEqual(absent["target_rank"], 0)
        self.assertFalse(absent["candidate_found"])
        self.assertEqual(details.index.tolist(), ["B", "A"])
        self.assertEqual(details.loc["A", "candidate_catalog_name"], "Alpha")
        self.assertEqual(details.loc["B", "score"], .1)
        for feature in DIAGNOSTIC_FEATURES:
            self.assertEqual(found["top_" + feature], .8)
            self.assertEqual(found["target_" + feature], .2)
            self.assertIsNone(absent["target_" + feature])
        self.assertEqual(found["candidate_ids_sha256"], absent["candidate_ids_sha256"])
        self.assertEqual([len(calls[name]) for name in calls], [2, 2, 2])
        for forwarded in calls["retrieve"] + [value[0] for value in calls["features"]]:
            self.assertEqual(set(forwarded), set(QUERY_COLUMNS) - {"sample_id"})
            self.assertEqual(forwarded["vessel_name"], query["vessel_name"])
        np.testing.assert_array_equal(calls["predict"][0][0], calls["predict"][1][0])
        for record in (found, absent):
            self.assertGreaterEqual(min(record[key] for key in TIMING_COLUMNS), 0)
            self.assertAlmostEqual(record["total_seconds"],
                                   sum(record[key] for key in TIMING_COLUMNS[:3]))

    def test_invalid_retrieval_is_rejected_before_feature_or_model_calls(self):
        for indices in ((0, 0), (-1, 0), (0, 3), (.5, 0), (np.nan, 0), (np.inf, 0), ()):
            with self.subTest(indices=indices):
                matcher, calls = self.matcher(indices)
                with self.assertRaisesRegex(ValueError, "catalog positions"):
                    detailed_prediction(matcher, {}, "A")
                self.assertFalse(calls["features"])
                self.assertFalse(calls["predict"])

    def test_changed_details_align_same_candidates_and_keep_absence_explicit(self):
        def details(ids, value):
            frame = pd.DataFrame({key: [value] * len(ids) for key in DIAGNOSTIC_FEATURES}, index=ids)
            frame["score"] = value
            frame["candidate_catalog_name"] = ["Name " + vessel for vessel in ids]
            return frame

        pair = {"index_80": ({"predicted_vessel_id": "A"}, details(["A"], .2)),
                "index_100": ({"predicted_vessel_id": "B"}, details(["B", "A"], .8))}
        row = {"sample_id": "S", "target_vessel_id": "C", "vessel_name": "한글"}
        result = pd.DataFrame(_changed_details(pair, row, "original"))
        self.assertEqual(len(result), 6)
        self.assertEqual(set(result.candidate_vessel_id), {"A", "B", "C"})
        self.assertEqual(set(result.model), set(RUNNERS))
        indexed = result.set_index(["candidate_vessel_id", "model"])
        self.assertEqual(indexed.loc[("A", "index_80"), "score"], .2)
        self.assertEqual(indexed.loc[("A", "index_100"), "score"], .8)
        self.assertFalse(indexed.loc[("B", "index_80"), "present"])
        self.assertTrue(indexed.loc[("B", "index_100"), "present"])
        self.assertTrue(indexed.loc[("C", "index_100"), "is_target"])
        self.assertTrue(pd.isna(indexed.loc[("C", "index_100"), "score"]))
        pair["index_100"][0]["predicted_vessel_id"] = "A"
        self.assertEqual(_changed_details(pair, row, "original"), [])

    def test_invalid_runner_requests_fail_before_loading_or_creating_output(self):
        cases = ({"variants": ("both_gt_missing",)}, {"variants": ("original", "original")},
                 {"variants": ("original", "unknown")}, {"progress_every": 0},
                 {"progress_every": True}, {"max_queries": 0}, {"max_queries": True})
        with TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    compare_index_scale("nonexistent-model", output, **kwargs)
                self.assertFalse(output.exists())


class IndexScaleSyntheticIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        specification = importlib.util.spec_from_file_location(
            "_index_scale_training_fixture", Path(__file__).with_name("test_robust_training.py"))
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        train, catalog = module.fixture()
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        train.to_csv(cls.train_path, index=False, encoding="utf-8-sig")
        catalog.to_csv(cls.catalog_path, index=False, encoding="utf-8-sig")
        cls.model = cls.root / "model"
        source, transformed = cls.root / "source", cls.root / "transformed"
        parser = argparse.ArgumentParser()
        add_training_parsers(parser.add_subparsers(dest="command", required=True))
        with redirect_stdout(io.StringIO()):
            run_training_command(parser.parse_args([
                "prepare-ranker", "--train", str(cls.train_path), "--catalog", str(cls.catalog_path),
                "--output", str(source), "--fold", "0", "--retrieval-profile", "robust",
                "--gt-augmentation", "missing", "--original-mass", "0.5",
            ]))
            transform_cache(source, transformed, train_path=cls.train_path, catalog_path=cls.catalog_path)
            run_training_command(parser.parse_args([
                "train-ranker", "--data", str(transformed), "--output", str(cls.model),
                "--rounds", "300", "--threads", "6", "--exclude-features", "retrieval_tonnage_rank",
            ]))

    def compare(self, output, **kwargs):
        with redirect_stdout(io.StringIO()), patch.object(
                VesselRanker, "fit", side_effect=AssertionError("Diagnostic must never train")):
            return compare_index_scale(self.model, output, train_path=self.train_path,
                catalog_path=self.catalog_path, expected_fold=0, max_queries=4,
                progress_every=100, **kwargs)

    def test_real_model_all_variants_are_paired_disjoint_immutable_and_match_saved_original(self):
        output = self.root / "comparison"
        protected = [self.train_path, self.catalog_path, *[p for p in self.model.rglob("*") if p.is_file()]]
        before = {path: _digest(path) for path in protected}
        queries_seen = []
        original_retrieve = RobustMatchingPipeline.retrieve

        def retrieve(pipeline, query):
            queries_seen.append(dict(query))
            return original_retrieve(pipeline, query)

        with patch.object(RobustMatchingPipeline, "retrieve", new=retrieve):
            summary = self.compare(output)
        self.assertEqual(before, {path: _digest(path) for path in protected})
        for flag in ("complete", "weights_unchanged", "training_assets_unchanged",
                     "protected_files_unchanged", "ranker_shared_by_both_indexes"):
            self.assertIs(summary[flag], True)
        self.assertIs(summary["training_performed"], False)
        self.assertIs(summary["submission_created"], False)
        self.assertTrue(summary["parity"]["all_selected_original_predictions_match"])
        expected_files = {"summary.json", "predictions.csv", "row_changes.csv", "vessel_changes.csv",
                          "changed_candidate_features.csv", "partitions.csv", "report.md"}
        self.assertEqual({path.name for path in output.iterdir()}, expected_files)
        self.assertEqual(json.loads((output / "summary.json").read_text("utf-8")), summary)
        predictions = pd.read_csv(output / "predictions.csv")
        n_queries = summary["coverage"]["queries"]
        self.assertEqual(n_queries, min(4, summary["coverage"]["heldout_queries"]))
        self.assertEqual(len(predictions), n_queries * len(VARIANTS) * len(RUNNERS))
        self.assertEqual(len(queries_seen), len(predictions) + 2)  # One untimed warmup per index.
        for query in queries_seen:
            self.assertEqual(set(query), set(QUERY_COLUMNS) - {"sample_id"})
        self.assertEqual(set(predictions.has_hangul), {True, False})
        self.assertTrue((predictions[list(TIMING_COLUMNS)].to_numpy() >= 0).all())
        np.testing.assert_allclose(predictions.total_seconds,
                                   predictions[list(TIMING_COLUMNS[:3])].sum(axis=1))
        for variant in VARIANTS:
            self.assertEqual(set(summary["metrics"][variant]), set(RUNNERS))
            rows = predictions.loc[predictions.variant.eq(variant)]
            by_runner = {runner: rows.loc[rows.model.eq(runner)] for runner in RUNNERS}
            self.assertEqual(set(by_runner["index_80"].sample_id), set(by_runner["index_100"].sample_id))
            self.assertEqual(summary["transitions"][variant]["queries"], n_queries)
            for runner, part in by_runner.items():
                self.assertEqual(summary["metrics"][variant][runner]["all"]["queries"], len(part))

        partitions = pd.read_csv(output / "partitions.csv")
        small = partitions.loc[partitions.selected_small]
        full = partitions.loc[partitions.selected_full]
        heldout = partitions.loc[partitions.role.eq("valid")]
        self.assertLess(set(small.sample_id), set(full.sample_id))
        self.assertFalse(set(full.sample_id) & set(heldout.sample_id))
        self.assertFalse(set(full.target_vessel_id) & set(heldout.target_vessel_id))
        self.assertEqual(small.target_vessel_id.nunique(), math.floor(.8 * full.target_vessel_id.nunique()))
        self.assertTrue(partitions.groupby("target_vessel_id").selected_small.nunique().eq(1).all())
        self.assertTrue(set(predictions.sample_id).issubset(set(heldout.sample_id)))
        self.assertEqual(summary["coverage"]["small"]["rows"], len(small))
        self.assertEqual(summary["coverage"]["full"]["rows"], len(full))

        saved = pd.read_csv(self.model / "validation_predictions.csv").set_index("sample_id")
        original = predictions.loc[predictions.variant.eq("original") & predictions.model.eq("index_100")]
        self.assertEqual(set(saved.index), set(heldout.sample_id))
        for row in original.itertuples(index=False):
            reference = saved.loc[row.sample_id]
            for field in ("target_vessel_id", "predicted_vessel_id", "candidate_count", "target_rank", "has_hangul"):
                self.assertEqual(getattr(row, field), reference[field])
        details = pd.read_csv(output / "changed_candidate_features.csv")
        self.assertEqual(details.columns.tolist(), DETAIL_COLUMNS)
        self.assertEqual(len(details), summary["paired_candidate_detail_rows"])
        changes = pd.read_csv(output / "row_changes.csv")
        self.assertEqual(len(changes), n_queries * len(VARIANTS))
        for path in output.iterdir():
            if path.name != "summary.json":
                self.assertEqual(_digest(path), summary["output_files_sha256"][path.name])

    def test_original_parity_failure_does_not_publish_completed_report(self):
        copied_model = self.root / "bad_saved_original"
        shutil.copytree(self.model, copied_model)
        saved_path = copied_model / "validation_predictions.csv"
        saved = pd.read_csv(saved_path)
        saved["predicted_vessel_id"] = "INVALID"
        saved.to_csv(saved_path, index=False, encoding="utf-8-sig")
        output = self.root / "failed_parity"
        with redirect_stdout(io.StringIO()), patch.object(
                VesselRanker, "fit", side_effect=AssertionError("Diagnostic must never train")):
            with self.assertRaisesRegex(ValueError, "saved|Saved|predictions|parity"):
                compare_index_scale(copied_model, output, train_path=self.train_path,
                    catalog_path=self.catalog_path, expected_fold=0, max_queries=2,
                    variants=("original",), progress_every=100)
        summary = json.loads((output / "summary.json").read_text("utf-8"))
        self.assertIs(summary["complete"], False)
        self.assertFalse((output / "report.md").exists())

    def test_existing_results_are_not_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "summary.json"
        marker.write_text("preserve previous result", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.compare(output)
        self.assertEqual(marker.read_text("utf-8"), "preserve previous result")


if __name__ == "__main__":
    unittest.main()
