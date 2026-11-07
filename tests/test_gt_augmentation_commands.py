"""CLI boundaries for training-only GT masks and the fixed 86-feature model."""
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
import pandas as pd

from portmatch.name_feature_cache import transform_cache
from portmatch.ranker_commands import add_training_parsers, run_training_command


def _parser():
    parser = argparse.ArgumentParser()
    add_training_parsers(parser.add_subparsers(dest="command", required=True))
    return parser


class GTAugmentationPrepareCommandTests(unittest.TestCase):
    def test_legacy_defaults_leave_augmentation_off(self):
        args = _parser().parse_args(["prepare-ranker", "--output", "unused"])
        self.assertEqual(args.gt_augmentation, "none")
        self.assertIsNone(args.original_mass)
        self.assertEqual(args.retrieval_profile, "legacy")

    def test_invalid_options_fail_before_input_reads_or_output_creation(self):
        cases = [
            ["--gt-augmentation", "missing"],
            ["--gt-augmentation", "missing", "--retrieval-profile", "hangul"],
            ["--gt-augmentation", "missing", "--retrieval-profile", "dual-gt"],
            ["--original-mass", "0.5"],
        ]
        cases.extend(["--gt-augmentation", "missing", "--retrieval-profile", "robust",
                      "--original-mass", value] for value in ("0", "1", "-0.1", "nan", "inf"))
        with tempfile.TemporaryDirectory() as root:
            for index, options in enumerate(cases):
                destination = Path(root) / str(index)
                args = _parser().parse_args(["prepare-ranker", "--output", str(destination), *options])
                with self.subTest(options=options), \
                     patch("portmatch.ranker_commands._hash_file") as hashing, \
                     patch("portmatch.ranker_commands.read_queries") as reading, \
                     patch("portmatch.training_data.prepare_training_data") as preparing:
                    with self.assertRaises(ValueError):
                        run_training_command(args)
                    hashing.assert_not_called()
                    reading.assert_not_called()
                    preparing.assert_not_called()
                    self.assertFalse(destination.exists())

    def test_valid_options_pass_explicit_configuration_and_keep_legacy_optional(self):
        cases = (([], None), (["--retrieval-profile", "robust"], None),
                 (["--retrieval-profile", "robust", "--gt-augmentation", "missing"], .5),
                 (["--retrieval-profile", "robust", "--gt-augmentation", "missing",
                   "--original-mass", "0.75"], .75))
        for options, mass in cases:
            args = _parser().parse_args(["prepare-ranker", "--output", "unused", *options])
            with self.subTest(options=options), contextlib.redirect_stdout(io.StringIO()), \
                 patch("portmatch.ranker_commands._hash_file", return_value="a" * 64), \
                 patch("portmatch.ranker_commands.read_queries", return_value=pd.DataFrame()), \
                 patch("portmatch.ranker_commands.read_catalog", return_value=pd.DataFrame()), \
                 patch("portmatch.training_data.prepare_training_data", return_value={
                     "counts": {}, "training_augmentation_diagnostics": {}}) as preparing:
                run_training_command(args)
                config = preparing.call_args.kwargs.get("augmentation_config")
                if mass is None:
                    self.assertIsNone(config)
                else:
                    self.assertEqual(config.original_mass, mass)
                    self.assertEqual(preparing.call_args.kwargs["retrieval_config"].profile, "robust")


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class GTAugmentationTrainingCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_spec = importlib.util.spec_from_file_location(
            "gt_commands_fixture", Path(__file__).with_name("test_robust_training.py"))
        fixture_module = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture_module)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.parser = _parser()
        train, catalog = fixture_module.fixture()
        train.to_csv(cls.root / "train.csv", index=False, encoding="utf-8-sig")
        catalog.to_csv(cls.root / "catalog.csv", index=False, encoding="utf-8-sig")
        cls.source, cls.derived = cls.root / "source", cls.root / "derived"
        cls.run_cli(["prepare-ranker", "--train", str(cls.root / "train.csv"),
                     "--catalog", str(cls.root / "catalog.csv"), "--output", str(cls.source),
                     "--retrieval-profile", "robust", "--gt-augmentation", "missing",
                     "--folds", "3", "--inner-folds", "2", "--seed", "42",
                     "--max-train-queries", "4", "--max-valid-queries", "4"])
        with contextlib.redirect_stdout(io.StringIO()):
            transform_cache(cls.source, cls.derived, cls.root / "train.csv", cls.root / "catalog.csv")

    @classmethod
    def run_cli(cls, arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            run_training_command(cls.parser.parse_args(arguments))

    def test_augmented_training_requires_exact_token_phonetic_86_before_creating_output(self):
        cases = ((self.source, []), (self.derived, []),
                 (self.derived, ["--exclude-features", "retrieval_tonnage_rank", "name_context_ratio"]),
                 (self.derived, ["--include-features", "name_*"]))
        for index, (data, selection) in enumerate(cases):
            output = self.root / ("rejected_" + str(index))
            with self.subTest(data=data.name, selection=selection), \
                 patch("portmatch.training_data.load_training_arrays") as loading:
                with self.assertRaises(ValueError):
                    self.run_cli(["train-ranker", "--data", str(data), "--output", str(output), *selection])
                loading.assert_not_called()
                self.assertFalse(output.exists())

    def test_augmented_transform_rejects_other_feature_experiments_before_output_creation(self):
        for transform in ("tonnage-min-rank", "ch-variants"):
            output = self.root / ("rejected_transform_" + transform)
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                self.run_cli(["transform-ranker-cache", "--data", str(self.derived),
                              "--output", str(output), "--transform", transform])
            self.assertFalse(output.exists())

    def test_augmented_training_passes_query_masses_and_records_weighting_audit(self):
        from portmatch.data import read_queries
        from portmatch.ranker_commands import RankedMatcher
        from portmatch.ranking import VesselRanker
        from portmatch.training_data import load_training_arrays

        output = self.root / "weighted_model"
        observed = {}
        original_fit = VesselRanker.fit

        def fit(instance, *args, **kwargs):
            observed["mass"] = np.array(kwargs.get("query_mass"), copy=True)
            return original_fit(instance, *args, **kwargs)

        with patch.object(VesselRanker, "fit", new=fit):
            self.run_cli(["train-ranker", "--data", str(self.derived), "--output", str(output),
                          "--exclude-features", "retrieval_tonnage_rank", "--rounds", "3",
                          "--threads", "1", "--num-leaves", "4", "--min-child-samples", "1", "--seed", "42"])
        arrays = load_training_arrays(self.derived, return_query_mass=True)
        np.testing.assert_array_equal(observed["mass"], arrays[-1])
        self.assertTrue(np.isfinite(observed["mass"]).all())
        self.assertTrue((observed["mass"] > 0).all())
        summary = json.loads((output / "training_summary.json").read_text("utf-8"))
        self.assertTrue(summary["complete"])
        self.assertEqual(len(summary["trained_feature_names"]), 86)
        self.assertNotIn("retrieval_tonnage_rank", summary["trained_feature_names"])
        audit = summary["training_weighting"]
        self.assertEqual(audit["groups"], len(arrays[2]))
        self.assertEqual(audit["pairs"], int(arrays[2].sum()))
        self.assertEqual(audit["source_mass_total"], float(arrays[-1].sum()))
        self.assertEqual(audit["query_mass_min"], float(arrays[-1].min()))
        self.assertEqual(audit["query_mass_max"], float(arrays[-1].max()))
        self.assertAlmostEqual(audit["languages"]["hangul"]["normalized_weight_share"], .7)
        self.assertAlmostEqual(audit["languages"]["non_hangul"]["normalized_weight_share"], .3)
        matcher = RankedMatcher(output)
        predictions = pd.read_csv(output / "validation_predictions.csv")
        queries = read_queries(self.root / "train.csv", labeled=True).set_index("sample_id")
        row = predictions.iloc[0]
        self.assertEqual(matcher.predict(queries.loc[row.sample_id].to_dict()), row.predicted_vessel_id)
        summary_path = output / "training_summary.json"
        saved_summary = summary_path.read_text("utf-8")
        for change in ("missing", "code", "mass", "share", "groups"):
            modified = copy.deepcopy(summary)
            if change == "missing":
                modified.pop("training_weighting")
            elif change == "code":
                modified["training_weighting"]["ranking_code_sha256"] = "0" * 64
            elif change == "mass":
                modified["training_weighting"]["source_mass_total"] += 1
            elif change == "share":
                modified["training_weighting"]["languages"]["hangul"]["normalized_weight_share"] = .5
            else:
                modified["training_weighting"]["languages"]["hangul"]["groups"] += 1
            try:
                summary_path.write_text(json.dumps(modified), encoding="utf-8")
                with self.subTest(proof_change=change), self.assertRaisesRegex(ValueError, "weighting"):
                    RankedMatcher(output)
            finally:
                summary_path.write_text(saved_summary, encoding="utf-8")

    def test_weighting_audit_is_optional_and_single_language_share_is_one(self):
        from portmatch.ranker_commands import _training_weighting_summary

        groups, language, masses = np.array([2, 3]), np.array([True, True]), np.array([.25, .75])
        self.assertIsNone(_training_weighting_summary(groups, language, None, .7))
        audit = _training_weighting_summary(groups, language, masses, .7)
        self.assertEqual(audit["languages"]["hangul"]["normalized_weight_share"], 1.)
        self.assertEqual(audit["languages"]["non_hangul"]["normalized_weight_share"], 0.)
        self.assertEqual(audit["source_mass_total"], 1.)


if __name__ == "__main__":
    unittest.main()
