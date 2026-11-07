import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from portmatch.ranker_commands import COMMANDS, add_training_parsers, run_training_command
from portmatch.index_scale_comparison import compare_index_scale


def parser():
    result = argparse.ArgumentParser()
    add_training_parsers(result.add_subparsers(dest="command", required=True))
    return result


class IndexScaleCommandTests(unittest.TestCase):
    def test_explicit_command_routes_frozen_fold_and_default_inputs(self):
        self.assertIn("compare-index-scale", COMMANDS)
        args = parser().parse_args(["compare-index-scale", "--model", "m", "--fold", "2", "--output", "o"])
        with patch("portmatch.index_scale_comparison.compare_index_scale") as compare:
            run_training_command(args)
        compare.assert_called_once_with("m", "o", train_path="work/data/train.csv",
            catalog_path="work/data/vessel_catalog.csv", subset_seed=20261027,
            expected_fold=2, max_queries=None, progress_every=100)

    def test_cli_rejects_unsupported_training_and_profile_changes(self):
        for option in (["--fold", "1"], ["--threads", "3"], ["--rounds", "100"],
                       ["--small-fraction", ".75"], ["--original-mass", ".75"]):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser().parse_args(["compare-index-scale", "--model", "m", "--fold", "0",
                                         "--output", "o", *option])

    def test_invalid_diagnostics_and_output_fail_before_index_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            for settings in ({"max_queries": 0}, {"progress_every": 0},
                             {"variants": ("both_gt_missing",)},
                             {"variants": ("original", "original")}):
                with self.subTest(settings=settings), patch("portmatch.index_scale_assets.prepare_index_scale") as load:
                    with self.assertRaises(ValueError):
                        compare_index_scale("missing", output, **settings)
                    load.assert_not_called()
                    self.assertFalse(output.exists())
            output.mkdir()
            (output / "existing.txt").write_text("preserve")
            with patch("portmatch.index_scale_assets.prepare_index_scale") as load:
                with self.assertRaises(FileExistsError):
                    compare_index_scale("missing", output)
                load.assert_not_called()
            self.assertEqual((output / "existing.txt").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
