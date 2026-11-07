"""Full-training CLI boundaries: fixed settings and no held-out evaluation."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from portmatch.ranker_commands import (
    COMMANDS, _check_feature_code, add_training_parsers, evaluate_cache,
    run_training_command,
)


def parser():
    result = argparse.ArgumentParser()
    add_training_parsers(result.add_subparsers(dest="command", required=True))
    return result


class FullTrainingCommandTests(unittest.TestCase):
    def test_explicit_commands_freeze_profile_and_omit_outer_fold_options(self):
        self.assertTrue({"prepare-full-ranker", "train-full-ranker"}.issubset(COMMANDS))
        command = parser()
        args = command.parse_args(["prepare-full-ranker", "--output", "new"])
        self.assertIsNone(args.max_train_queries)
        self.assertFalse(hasattr(args, "fold"))
        for argv in [
            ["prepare-full-ranker", "--output", "new", "--fold", "0"],
            ["prepare-full-ranker", "--output", "new", "--original-mass", ".75"],
            ["train-full-ranker", "--data", "cache", "--output", "new", "--rounds", "100"],
            ["train-full-ranker", "--data", "cache", "--output", "new", "--threads", "1"],
            ["train-full-ranker", "--data", "cache", "--output", "new", "--exclude-features", "name_*"],
        ]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    command.parse_args(argv)

    def test_prepare_rejects_invalid_limits_before_reading_inputs(self):
        for option in ("--shard-queries", "--max-train-queries"):
            args = parser().parse_args(["prepare-full-ranker", "--output", "unused", option, "0"])
            with self.subTest(option=option), patch("portmatch.data.read_queries") as reading:
                with self.assertRaises(ValueError):
                    run_training_command(args)
                reading.assert_not_called()

    def test_prepare_passes_original_input_provenance_and_diagnostic_limit(self):
        args = parser().parse_args(["prepare-full-ranker", "--output", "new", "--max-train-queries", "4"])
        with patch("portmatch.data.read_queries", return_value=pd.DataFrame()) as reading, \
             patch("portmatch.data.read_catalog", return_value=pd.DataFrame()), \
             patch("portmatch.ranker_commands._hash_file", return_value="a" * 64), \
             patch("portmatch.full_training_data.prepare_full_training_data", return_value={}) as operation, \
             contextlib.redirect_stdout(io.StringIO()):
            run_training_command(args)
        reading.assert_called_once_with("work/data/train.csv", labeled=True)
        self.assertEqual(operation.call_args.kwargs["max_train_queries"], 4)
        self.assertEqual(operation.call_args.kwargs["provenance"]["train_sha256"], "a" * 64)
        self.assertNotIn("outer_fold", operation.call_args.kwargs)

    def test_legacy_training_and_reweighting_reject_full_cache_before_mutation(self):
        for command, extra in (("train-ranker", []), ("reweight-ranker-cache", ["--original-mass", ".75"])):
            args = parser().parse_args([command, "--data", "cache", "--output", "unused", *extra])
            with self.subTest(command=command), \
                 patch("portmatch.training_data.read_metadata", return_value={"training_scope": "full_public_train"}), \
                 patch("portmatch.ranker_commands._new_output") as output, \
                 patch("portmatch.gt_reweight_cache.reweight_cache") as reweight:
                with self.assertRaisesRegex(ValueError, "[Ff]ull"):
                    run_training_command(args)
                output.assert_not_called()
                reweight.assert_not_called()

    def test_full_cache_cannot_report_validation_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metadata.json").write_text(json.dumps({
                "training_scope": "full_public_train", "feature_names": ["a"]}), encoding="utf-8")
            with patch("portmatch.training_data.iter_shards") as reading:
                with self.assertRaisesRegex(ValueError, "no held-out"):
                    evaluate_cache(None, path)
                reading.assert_not_called()

    def test_transform_dispatches_full_lineage_and_rejects_other_transforms(self):
        args = parser().parse_args(["transform-ranker-cache", "--transform", "token-phonetic",
                                   "--data", "cache", "--output", "new"])
        with patch("portmatch.training_data.read_metadata", return_value={"training_scope": "full_public_train"}), \
             patch("portmatch.full_ranking.transform_full_cache", return_value={}) as operation, \
             contextlib.redirect_stdout(io.StringIO()):
            run_training_command(args)
        operation.assert_called_once_with("cache", "new", "work/data/train.csv", "work/data/vessel_catalog.csv")
        args.transform = "ch-variants"
        with patch("portmatch.training_data.read_metadata", return_value={"training_scope": "full_public_train"}), \
             patch("portmatch.full_ranking.transform_full_cache") as operation:
            with self.assertRaisesRegex(ValueError, "Full training fixes"):
                run_training_command(args)
            operation.assert_not_called()

    def test_train_dispatch_uses_fixed_config_and_has_no_evaluation(self):
        args = parser().parse_args(["train-full-ranker", "--data", "cache", "--output", "model"])
        with patch("portmatch.full_ranking.train_full_ranker", return_value={}) as operation, \
             patch("portmatch.ranker_commands.evaluate_cache") as evaluate, \
             contextlib.redirect_stdout(io.StringIO()):
            run_training_command(args)
        self.assertEqual(operation.call_args.kwargs["config"].n_estimators, 300)
        self.assertEqual(operation.call_args.kwargs["config"].n_jobs, 6)
        evaluate.assert_not_called()

    def test_partial_full_marker_cannot_load_as_legacy(self):
        for marker in ({"training_scope": "unknown"}, {"full_training": {}}, {"outer_split": "none"}):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                _check_feature_code(marker)


if __name__ == "__main__":
    unittest.main()
