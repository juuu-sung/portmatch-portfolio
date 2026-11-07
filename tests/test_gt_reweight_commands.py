"""CLI and restored-model boundaries for mass-only augmented cache training."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from portmatch.ranker_commands import add_training_parsers, run_training_command


def parser():
    result = argparse.ArgumentParser()
    add_training_parsers(result.add_subparsers(dest="command", required=True))
    return result


class ReweightParserTests(unittest.TestCase):
    def test_required_explicit_mass_and_dispatch(self):
        command = parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            command.parse_args(["reweight-ranker-cache", "--data", "source", "--output", "child"])
        args = command.parse_args(["reweight-ranker-cache", "--data", "source", "--output", "child",
                                   "--original-mass", ".75"])
        with patch("portmatch.gt_reweight_cache.reweight_cache", return_value={
                "training_augmentation_diagnostics": {}}) as operation, \
                patch("portmatch.training_data.read_metadata", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            run_training_command(args)
        operation.assert_called_once_with("source", "child", original_mass=.75)

    def test_invalid_mass_rejected_without_input_or_output(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "must_not_exist"
            for mass in ("0", "1", "-1", "nan", "inf"):
                args = parser().parse_args(["reweight-ranker-cache", "--data", "missing_source",
                                           "--output", str(destination), "--original-mass", mass])
                with self.subTest(mass=mass), self.assertRaises(ValueError):
                    run_training_command(args)
                self.assertFalse(destination.exists())


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "LightGBM is not installed")
class ReweightTrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "reweight_command_fixture", Path(__file__).with_name("test_robust_training.py"))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.parser = parser()
        train, catalog = fixture.fixture()
        train.to_csv(cls.root / "train.csv", index=False)
        catalog.to_csv(cls.root / "catalog.csv", index=False)
        cls.raw, cls.source, cls.child = [cls.root / name for name in ("raw", "source", "mass75")]
        cls.run_cli(["prepare-ranker", "--train", str(cls.root / "train.csv"),
                     "--catalog", str(cls.root / "catalog.csv"), "--output", str(cls.raw),
                     "--retrieval-profile", "robust", "--gt-augmentation", "missing",
                     "--folds", "3", "--inner-folds", "2", "--seed", "42"])
        cls.run_cli(["transform-ranker-cache", "--transform", "token-phonetic", "--data", str(cls.raw),
                     "--output", str(cls.source), "--train", str(cls.root / "train.csv"),
                     "--catalog", str(cls.root / "catalog.csv")])
        cls.run_cli(["reweight-ranker-cache", "--data", str(cls.source), "--output", str(cls.child),
                     "--original-mass", ".75"])

    @classmethod
    def run_cli(cls, arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            run_training_command(cls.parser.parse_args(arguments))

    def test_mass75_reaches_fit_and_bundle_loads_without_training_cache(self):
        from portmatch.data import read_queries
        from portmatch.ranker_commands import RankedMatcher
        from portmatch.ranking import VesselRanker
        from portmatch.training_data import load_training_arrays

        model = self.root / "model75"
        observed = {}
        original_fit = VesselRanker.fit

        def fit(instance, *args, **kwargs):
            observed["mass"] = np.array(kwargs["query_mass"], copy=True)
            return original_fit(instance, *args, **kwargs)

        with patch.object(VesselRanker, "fit", new=fit):
            self.run_cli(["train-ranker", "--data", str(self.child), "--output", str(model),
                          "--exclude-features", "retrieval_tonnage_rank", "--rounds", "3",
                          "--threads", "1", "--num-leaves", "4", "--min-child-samples", "1", "--seed", "42"])
        arrays = load_training_arrays(self.child, return_query_mass=True)
        original_arrays = load_training_arrays(self.source, return_query_mass=True)
        np.testing.assert_array_equal(observed["mass"], arrays[-1])
        self.assertFalse(np.array_equal(observed["mass"], original_arrays[-1]))
        for before, after in zip(original_arrays[:-1], arrays[:-1]):
            np.testing.assert_array_equal(before, after)
        summary_path = model / "training_summary.json"
        summary = json.loads(summary_path.read_text("utf-8"))
        self.assertEqual(summary["dataset"]["training_augmentation"]["config"]["original_mass"], .75)
        self.assertIn("training_reweighting", summary["dataset"])
        self.assertEqual(len(summary["trained_feature_names"]), 86)
        self.assertAlmostEqual(summary["training_weighting"]["source_mass_total"], arrays[-1].sum())
        row = read_queries(self.root / "train.csv", labeled=True).iloc[0].to_dict()
        expected = RankedMatcher(model).predict(row)
        # A standalone model must not read its training-cache paths at inference.
        with patch("portmatch.gt_reweight_cache.verify_reweighting", side_effect=AssertionError("cache read")):
            self.assertEqual(RankedMatcher(model).predict(row), expected)
        changed = copy.deepcopy(summary)
        changed["dataset"]["training_augmentation"]["config"]["original_mass"] = .6
        summary_path.write_text(json.dumps(changed), encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                RankedMatcher(model)
        finally:
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

    def test_lineage_failure_prevents_training_and_output_creation(self):
        destination = self.root / "rejected"
        with patch("portmatch.gt_reweight_cache.verify_reweighting", side_effect=ValueError("changed lineage")), \
                patch("portmatch.ranking.VesselRanker.fit") as fitting:
            with self.assertRaisesRegex(ValueError, "lineage"):
                self.run_cli(["train-ranker", "--data", str(self.child), "--output", str(destination),
                              "--exclude-features", "retrieval_tonnage_rank", "--rounds", "3"])
            fitting.assert_not_called()
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
