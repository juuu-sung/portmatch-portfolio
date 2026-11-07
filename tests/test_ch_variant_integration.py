"""End-to-end CH experiment controls on tiny synthetic data, not real training."""
import argparse
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.ch_variant_features import FEATURE_NAMES
from portmatch.feature_transforms import check_parent_transform
from portmatch.name_feature_cache import transform_spec as parent_spec
from portmatch.ranker_commands import RankedMatcher, add_training_parsers, run_training_command
from portmatch.ranker_comparison import compare_rankers


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "LightGBM is not installed")
class ChVariantIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.parser = argparse.ArgumentParser()
        add_training_parsers(cls.parser.add_subparsers(dest="command", required=True))
        names = [("North Chorus", "노스 코러스"), ("North Corona", "노스 코로나"),
                 ("Blue Chime", "블루 차임"), ("Blue Kite", "블루 카이트"),
                 ("Ocean Echo", "오션 에코"), ("Ocean Pearl", "오션 펄"),
                 ("Chorus 11", "코러스11"), ("Chorus 1", "코러스1"),
                 ("Silver Knight", "실버 나이트"), ("Silver Kite", "실버 카이트"),
                 ("Coastal Meadow", "코스탈 메도"), ("Coastal River", "코스탈 리버")]
        catalog = pd.DataFrame([
            dict(vessel_id=f"V{i:02d}", vessel_name=names[i][0],
                 gross_tonnage=1000 + i // 3, flag_name="Korea", vessel_type="Cargo",
                 size=10., unit="m", deadweight_tonnage=1200.) for i in range(len(names))])
        cls.queries = pd.DataFrame([
            dict(sample_id=f"S{i:02d}_{arrival}", target_vessel_id=f"V{i:02d}",
                 vessel_name=name[arrival], call_sign=f"C{i:02d}",
                 gross_tonnage=1000 + i // 3, international_gross_tonnage=1000 + i // 3,
                 flag_name="한국", vessel_type="화물선", arrival_year=2020 + arrival)
            for i, name in enumerate(names) for arrival in range(2)])
        catalog.to_csv(cls.root / "catalog.csv", index=False, encoding="utf-8-sig")
        cls.queries.to_csv(cls.root / "train.csv", index=False, encoding="utf-8-sig")
        cls.source, cls.parent, cls.child = [cls.root / n for n in ("original", "names", "ch")]
        cls.run_cli(["prepare-ranker", "--train", str(cls.root / "train.csv"),
                     "--catalog", str(cls.root / "catalog.csv"), "--output", str(cls.source),
                     "--folds", "3", "--inner-folds", "2", "--seed", "42",
                     "--max-train-queries", "5", "--max-valid-queries", "4", "--shard-queries", "2"])
        for transform, source, destination in (("token-phonetic", cls.source, cls.parent),
                                                ("ch-variants", cls.parent, cls.child)):
            cls.run_cli(["transform-ranker-cache", "--transform", transform,
                         "--data", str(source), "--output", str(destination),
                         "--train", str(cls.root / "train.csv"),
                         "--catalog", str(cls.root / "catalog.csv")])
        cls.baseline, cls.experiment, cls.control = [cls.root / n for n in ("b86", "e93", "c86")]
        for cache, output, extra_exclusions in ((cls.parent, cls.baseline, []),
                                                (cls.child, cls.experiment, []),
                                                (cls.child, cls.control, list(FEATURE_NAMES))):
            cls.run_cli(["train-ranker", "--data", str(cache), "--output", str(output),
                         "--rounds", "4", "--threads", "1", "--num-leaves", "4",
                         "--min-child-samples", "1", "--seed", "42",
                         "--exclude-features", "retrieval_tonnage_rank", *extra_exclusions])

    @classmethod
    def run_cli(cls, arguments):
        run_training_command(cls.parser.parse_args(arguments))

    def test_parent_and_child_comparison_preserves_all_86_selected_features(self):
        result = compare_rankers([self.baseline], [self.experiment], self.root / "comparison")
        self.assertEqual(result["comparison_type"], "verified_derived_feature_experiment")
        self.assertEqual(len(result["features"]["baseline"]), 86)
        self.assertEqual(len(result["features"]["experiment"]), 93)
        self.assertEqual(result["features"]["added"], list(FEATURE_NAMES))
        self.assertEqual(result["features"]["removed"], [])
        self.assertEqual(result["coverage"]["queries"], 4)
        self.assertFalse(result["coverage"]["full_oof"])

    def test_excluding_ch_features_reproduces_parent_model_scores_exactly(self):
        baseline = pd.read_csv(self.baseline / "validation_predictions.csv")
        control = pd.read_csv(self.control / "validation_predictions.csv")
        pd.testing.assert_frame_equal(baseline, control, check_exact=True)
        result = compare_rankers([self.baseline], [self.control], self.root / "control_comparison")
        self.assertEqual(result["features"]["added"], [])
        self.assertEqual(result["pooled"]["transitions"]["all"]["prediction_changed"], 0)
        same_child = compare_rankers([self.control], [self.experiment], self.root / "same_child")
        self.assertEqual(same_child["features"]["added"], list(FEATURE_NAMES))

    def test_restored_models_match_cached_scores_and_ignore_labels(self):
        queries = self.queries.set_index("sample_id")
        for model, count in ((self.baseline, 86), (self.experiment, 93), (self.control, 86)):
            matcher = RankedMatcher(model)
            self.assertEqual(len(matcher.ranker.feature_names), count)
            for record in pd.read_csv(model / "validation_predictions.csv").to_dict("records"):
                row = queries.loc[record["sample_id"]].to_dict()
                ranked = matcher.rank(row)
                self.assertEqual(len(ranked), record["candidate_count"])
                self.assertEqual(ranked.vessel_id.iloc[0], record["predicted_vessel_id"])
                self.assertAlmostEqual(float(ranked.score.iloc[0]), record["top_score"], places=12)
                self.assertEqual(int(np.flatnonzero(ranked.vessel_id.to_numpy() == row["target_vessel_id"])[0]) + 1,
                                 record["target_rank"])
                row["target_vessel_id"] = "NOT_A_FEATURE"
                row["irrelevant_metadata"] = "IGNORED"
                pd.testing.assert_frame_equal(matcher.rank(row), ranked, check_exact=True)

    def test_batch_order_independence_and_cache_free_inference(self):
        ids = pd.read_csv(self.experiment / "validation_predictions.csv").sample_id.tolist()
        heldout = self.queries.set_index("sample_id").loc[ids].reset_index()
        outputs = []
        for number, rows in enumerate((heldout, heldout.iloc[::-1], heldout.iloc[[0]])):
            source, output = self.root / f"input{number}.csv", self.root / f"output{number}.csv"
            rows.to_csv(source, index=False, encoding="utf-8-sig")
            self.run_cli(["predict-ranker", "--model", str(self.experiment),
                         "--queries", str(source), "--output", str(output)])
            frame = pd.read_csv(output)
            self.assertEqual(frame.sample_id.tolist(), rows.sample_id.tolist())
            outputs.append(frame.set_index("sample_id").target_vessel_id.to_dict())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0][ids[0]], outputs[2][ids[0]])
        row = heldout.iloc[0].to_dict()
        expected = RankedMatcher(self.experiment).rank(row)
        roots = (self.source.resolve(), self.parent.resolve(), self.child.resolve())

        def guard(original):
            def opened(file, *args, **kwargs):
                if isinstance(file, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(file)).resolve()
                    if any(path.is_relative_to(root) for root in roots):
                        raise AssertionError(f"Inference read a training cache: {path}")
                return original(file, *args, **kwargs)
            return opened

        with patch("builtins.open", guard(builtins.open)), patch("io.open", guard(io.open)):
            pd.testing.assert_frame_equal(RankedMatcher(self.experiment).rank(row), expected, check_exact=True)

    def test_only_the_recognized_immediate_parent_can_be_compared(self):
        from portmatch.ch_variant_cache import transform_spec as child_spec
        from portmatch.tonnage_rank import transform_spec as tonnage_spec

        check_parent_transform(parent_spec(), child_spec())
        for source, derived in ((None, child_spec()), (child_spec(), parent_spec()),
                                (tonnage_spec(), child_spec()), (parent_spec(), None)):
            with self.subTest(source=source, derived=derived), self.assertRaises(ValueError):
                check_parent_transform(source, derived)
        with self.assertRaises(ValueError):
            compare_rankers([self.experiment], [self.baseline], self.root / "reverse")


if __name__ == "__main__":
    unittest.main()
