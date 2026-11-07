"""Small end-to-end checks; synthetic fits are not competition experiments."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from portmatch.ranker_commands import RankedMatcher, add_training_parsers, run_training_command
from portmatch.ranker_comparison import compare_rankers
from portmatch.tonnage_rank import SOURCE_FEATURE, TARGET_FEATURE
from portmatch.training_data import iter_shards


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class TonnageRankIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.parser = argparse.ArgumentParser()
        add_training_parsers(cls.parser.add_subparsers(dest="command", required=True))
        cls.catalog = pd.DataFrame([
            dict(vessel_id=f"V{i:02d}", vessel_name=f"SHIP {i}",
                 gross_tonnage=1000 + i // 3, flag_name="Korea", vessel_type="Cargo",
                 size=10., unit="m", deadweight_tonnage=1200.) for i in range(12)])
        cls.queries = pd.DataFrame([
            dict(sample_id=f"S{i:02d}_{arrival}", target_vessel_id=f"V{i:02d}",
                 vessel_name=("선박 " if i % 2 else "SHIP ") + str(i), call_sign=f"C{i:02d}",
                 gross_tonnage=1000 + i // 3, international_gross_tonnage=1000 + i // 3,
                 flag_name="한국", vessel_type="화물선", arrival_year=2020 + arrival)
            for i in range(12) for arrival in range(2)])
        cls.catalog.to_csv(cls.root / "catalog.csv", index=False, encoding="utf-8-sig")
        cls.queries.to_csv(cls.root / "train.csv", index=False, encoding="utf-8-sig")
        cls.source = cls.root / "source"
        cls.derived = cls.root / "derived"
        cls.run_cli(["prepare-ranker", "--train", str(cls.root / "train.csv"),
                     "--catalog", str(cls.root / "catalog.csv"), "--output", str(cls.source),
                     "--folds", "3", "--inner-folds", "2", "--seed", "42",
                     "--max-train-queries", "4", "--max-valid-queries", "4", "--shard-queries", "2"])
        cls.source_before = cls.hashes(cls.source)
        cls.run_cli(["transform-ranker-cache", "--data", str(cls.source), "--output", str(cls.derived),
                     "--train", str(cls.root / "train.csv"), "--catalog", str(cls.root / "catalog.csv")])
        cls.baseline, cls.experiment, cls.control = [cls.root / name for name in ("baseline", "experiment", "control")]
        for cache, output, exclude in ((cls.source, cls.baseline, SOURCE_FEATURE),
                                       (cls.derived, cls.experiment, None),
                                       (cls.derived, cls.control, TARGET_FEATURE)):
            command = ["train-ranker", "--data", str(cache), "--output", str(output),
                       "--rounds", "4", "--threads", "1", "--num-leaves", "4",
                       "--min-child-samples", "1", "--seed", "42"]
            if exclude:
                command += ["--exclude-features", exclude]
            cls.run_cli(command)

    @classmethod
    def run_cli(cls, arguments):
        run_training_command(cls.parser.parse_args(arguments))

    @staticmethod
    def hashes(directory):
        return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in directory.rglob("*") if p.is_file()}

    def test_only_rank_column_changes_and_source_is_untouched(self):
        self.assertEqual(self.source_before, self.hashes(self.source))
        metadata = json.loads((self.source / "metadata.json").read_text("utf-8"))
        column = metadata["feature_names"].index(SOURCE_FEATURE)
        shared = [i for i in range(70) if i != column]
        changed = 0
        for kind in ("train", "valid"):
            for left, right in zip(iter_shards(self.source, kind), iter_shards(self.derived, kind)):
                for name in left:
                    if name != "X":
                        np.testing.assert_array_equal(left[name], right[name])
                np.testing.assert_array_equal(left["X"][:, shared], right["X"][:, shared])
                changed += int(np.count_nonzero(left["X"][:, column] != right["X"][:, column]))
        self.assertGreater(changed, 0)

    def test_restored_models_match_cache_scores_and_ignore_labels(self):
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
                row["target_vessel_id"] = "IGNORED_LABEL"
                pd.testing.assert_frame_equal(matcher.rank(row), ranked)
        self.assertIn(TARGET_FEATURE, RankedMatcher(self.experiment).ranker.feature_names)
        self.assertNotIn(SOURCE_FEATURE, RankedMatcher(self.experiment).ranker.feature_names)

    def test_removing_new_column_reproduces_69_feature_control(self):
        baseline = pd.read_csv(self.baseline / "validation_predictions.csv")
        control = pd.read_csv(self.control / "validation_predictions.csv")
        pd.testing.assert_frame_equal(baseline, control, check_exact=True)

    def test_current_69_model_compares_to_transformed_70_model(self):
        summary = compare_rankers([self.baseline], [self.experiment], self.root / "comparison")
        self.assertEqual(len(summary["features"]["baseline"]), 69)
        self.assertEqual(len(summary["features"]["experiment"]), 70)
        self.assertEqual(summary["features"]["added"], [TARGET_FEATURE])
        self.assertFalse(summary["coverage"]["full_oof"])
        self.assertEqual(summary["coverage"]["queries"], 4)


if __name__ == "__main__":
    unittest.main()
