"""Tiny complete workflows for additive names; synthetic fits are not an experiment."""
import argparse
import builtins
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.name_features import FEATURE_NAMES
from portmatch.ranker_commands import RankedMatcher, add_training_parsers, run_training_command
from portmatch.ranker_comparison import compare_rankers
from portmatch.training_data import iter_shards


ORDINAL_FEATURE = "retrieval_tonnage_rank"


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class NameFeatureIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.parser = argparse.ArgumentParser()
        add_training_parsers(cls.parser.add_subparsers(dest="command", required=True))
        names = [
            ("Ocean Cedar", "오션 시더"), ("Ocean Meadow", "오션 메도"),
            ("Coastal Light", "코스탈 라이트"), ("Coastal River", "코스탈 리버"),
            ("Blue Pearl", "블루 펄"), ("Blue Peak", "블루 피크"),
            ("Silver Knight", "실버 나이트"), ("Silver Kite", "실버 카이트"),
            ("Cedar No 7", "시더 7"), ("Cedar No 8", "시더 8"),
            ("Jeong Won", "정원호"), ("Jeong Wol", "정월호"),
        ]
        cls.catalog = pd.DataFrame([
            dict(vessel_id=f"V{i:02d}", vessel_name=name[0],
                 gross_tonnage=1000 + i // 3, flag_name="Korea", vessel_type="Cargo",
                 size=10., unit="m", deadweight_tonnage=1200.)
            for i, name in enumerate(names)])
        cls.queries = pd.DataFrame([
            dict(sample_id=f"S{i:02d}_{arrival}", target_vessel_id=f"V{i:02d}",
                 vessel_name=name[arrival], call_sign=f"C{i:02d}",
                 gross_tonnage=1000 + i // 3, international_gross_tonnage=1000 + i // 3,
                 flag_name="한국", vessel_type="화물선", arrival_year=2020 + arrival)
            for i, name in enumerate(names) for arrival in range(2)])
        cls.catalog.to_csv(cls.root / "catalog.csv", index=False, encoding="utf-8-sig")
        cls.queries.to_csv(cls.root / "train.csv", index=False, encoding="utf-8-sig")
        cls.source, cls.derived = cls.root / "source", cls.root / "derived"
        cls.run_cli(["prepare-ranker", "--train", str(cls.root / "train.csv"),
                     "--catalog", str(cls.root / "catalog.csv"), "--output", str(cls.source),
                     "--folds", "3", "--inner-folds", "2", "--seed", "42",
                     "--max-train-queries", "4", "--max-valid-queries", "4", "--shard-queries", "2"])
        cls.source_before = cls.hashes(cls.source)
        cls.run_cli(["transform-ranker-cache", "--transform", "token-phonetic",
                     "--data", str(cls.source), "--output", str(cls.derived),
                     "--train", str(cls.root / "train.csv"), "--catalog", str(cls.root / "catalog.csv")])
        cls.baseline, cls.experiment, cls.control = [cls.root / name for name in
                                                   ("baseline", "experiment", "control")]
        for cache, output, exclude in (
                (cls.source, cls.baseline, [ORDINAL_FEATURE]),
                (cls.derived, cls.experiment, [ORDINAL_FEATURE]),
                (cls.derived, cls.control, [ORDINAL_FEATURE, *FEATURE_NAMES])):
            cls.run_cli(["train-ranker", "--data", str(cache), "--output", str(output),
                         "--rounds", "4", "--threads", "1", "--num-leaves", "4",
                         "--min-child-samples", "1", "--seed", "42",
                         "--exclude-features", *exclude])

    @classmethod
    def run_cli(cls, arguments):
        run_training_command(cls.parser.parse_args(arguments))

    @staticmethod
    def hashes(directory):
        return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob("*") if p.is_file()}

    def test_append_preserves_every_source_array_and_the_source_files(self):
        self.assertEqual(self.source_before, self.hashes(self.source))
        metadata = json.loads((self.source / "metadata.json").read_text("utf-8"))
        appended = json.loads((self.derived / "metadata.json").read_text("utf-8"))
        source_width = len(metadata["feature_names"])
        self.assertEqual(source_width, 70)
        self.assertEqual(appended["feature_names"], [*metadata["feature_names"], *FEATURE_NAMES])
        for kind in ("train", "valid"):
            for left, right in zip(iter_shards(self.source, kind), iter_shards(self.derived, kind)):
                self.assertEqual(set(left), set(right))
                for name in left:
                    value = right[name][:, :source_width] if name == "X" else right[name]
                    self.assertEqual(left[name].dtype, value.dtype)
                    self.assertEqual(left[name].tobytes(), value.tobytes())
                self.assertTrue(np.isfinite(right["X"][:, source_width:]).all())

    def test_restored_model_and_feature_selection_match_every_cached_score(self):
        queries = self.queries.set_index("sample_id")
        for model in (self.baseline, self.experiment, self.control):
            matcher = RankedMatcher(model)
            saved = pd.read_csv(model / "validation_predictions.csv")
            for record in saved.to_dict("records"):
                row = queries.loc[record["sample_id"]].to_dict()
                row["sample_id"] = record["sample_id"]
                ranked = matcher.rank(row)
                self.assertEqual(len(ranked), record["candidate_count"])
                self.assertEqual(ranked.vessel_id.iloc[0], record["predicted_vessel_id"])
                self.assertAlmostEqual(float(ranked.score.iloc[0]), record["top_score"], places=12)
                target_rank = int(np.flatnonzero(ranked.vessel_id.to_numpy() == row["target_vessel_id"])[0]) + 1
                self.assertEqual(target_rank, record["target_rank"])
                row["target_vessel_id"] = "IGNORED_LABEL"
                row["unrecognized_column"] = "IGNORED"
                pd.testing.assert_frame_equal(matcher.rank(row), ranked, check_exact=True)
        names = RankedMatcher(self.experiment).ranker.feature_names
        self.assertEqual(len(names), 69 + len(FEATURE_NAMES))
        self.assertTrue(set(FEATURE_NAMES).issubset(names))
        self.assertNotIn(ORDINAL_FEATURE, names)

    def test_derived_69_control_reproduces_baseline_bit_exactly(self):
        baseline = pd.read_csv(self.baseline / "validation_predictions.csv")
        control = pd.read_csv(self.control / "validation_predictions.csv")
        pd.testing.assert_frame_equal(baseline, control, check_exact=True)
        summary = compare_rankers([self.baseline], [self.control], self.root / "control_comparison")
        self.assertEqual(summary["features"]["added"], [])
        self.assertEqual(summary["features"]["removed"], [])
        self.assertEqual(summary["pooled"]["transitions"]["all"]["prediction_changed"], 0)

    def test_comparison_recognizes_additive_names_and_retained_69_features(self):
        summary = compare_rankers([self.baseline], [self.experiment], self.root / "comparison")
        self.assertEqual(summary["comparison_type"], "verified_derived_feature_experiment")
        self.assertEqual(len(summary["features"]["baseline"]), 69)
        self.assertEqual(len(summary["features"]["experiment"]), 69 + len(FEATURE_NAMES))
        self.assertEqual(summary["features"]["added"], list(FEATURE_NAMES))
        self.assertEqual(summary["features"]["removed"], [])
        self.assertFalse(summary["coverage"]["full_oof"])
        self.assertEqual(summary["coverage"]["queries"], 4)

    def test_prediction_cli_is_independent_of_batch_size_and_row_order(self):
        ids = pd.read_csv(self.experiment / "validation_predictions.csv").sample_id.tolist()
        heldout = self.queries.set_index("sample_id").loc[ids].reset_index()
        inputs = {"forward": heldout, "reverse": heldout.iloc[::-1], "one": heldout.iloc[[0]]}
        predictions = {}
        for name, rows in inputs.items():
            path, output = self.root / f"query_{name}.csv", self.root / f"predictions_{name}.csv"
            rows.to_csv(path, index=False, encoding="utf-8-sig")
            self.run_cli(["predict-ranker", "--model", str(self.experiment),
                          "--queries", str(path), "--output", str(output)])
            frame = pd.read_csv(output)
            self.assertEqual(frame.sample_id.tolist(), rows.sample_id.tolist())
            predictions[name] = frame.set_index("sample_id").target_vessel_id.to_dict()
        self.assertEqual(predictions["forward"], predictions["reverse"])
        self.assertEqual(predictions["one"][ids[0]], predictions["forward"][ids[0]])

    def test_bundle_inference_does_not_require_either_training_cache(self):
        row = self.queries.iloc[0].to_dict()
        expected = RankedMatcher(self.experiment).rank(row)
        standalone = self.root / "standalone_bundle"
        shutil.copytree(self.experiment, standalone)
        cache_roots = (self.source.resolve(), self.derived.resolve())

        def disallow_cache_reads(original_open):
            def guarded(file, *args, **kwargs):
                if isinstance(file, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(file)).resolve()
                    if any(path.is_relative_to(root) for root in cache_roots):
                        raise AssertionError(f"Inference attempted to read a training cache: {path}")
                return original_open(file, *args, **kwargs)
            return guarded

        # Block both pathlib's io.open and joblib's builtin open. This makes the
        # original cache paths inaccessible without directory-renaming problems
        # from Windows sandbox handles, while retaining the unmodified bundle.
        with patch("builtins.open", disallow_cache_reads(builtins.open)), \
                patch("io.open", disallow_cache_reads(io.open)):
            restored = RankedMatcher(standalone)
            pd.testing.assert_frame_equal(restored.rank(row), expected, check_exact=True)

    def test_unknown_transform_and_inconsistent_source_schema_fail_closed(self):
        original = json.loads((self.experiment / "training_summary.json").read_text("utf-8"))
        for name in ("unknown_transform", "missing_transform", "missing_column", "reordered_columns"):
            with self.subTest(case=name):
                destination = self.root / name
                shutil.copytree(self.experiment, destination)
                manifest = json.loads(json.dumps(original))
                dataset = manifest["dataset"]
                if name == "unknown_transform":
                    dataset["feature_transform"]["name"] = "unrecognized_transform_v999"
                elif name == "missing_transform":
                    dataset.pop("feature_transform")
                elif name == "missing_column":
                    dataset["feature_names"].remove(FEATURE_NAMES[0])
                else:
                    dataset["feature_names"][-2:] = dataset["feature_names"][-2:][::-1]
                (destination / "training_summary.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    RankedMatcher(destination)


if __name__ == "__main__":
    unittest.main()
