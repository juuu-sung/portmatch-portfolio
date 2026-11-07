"""Opt-in robust/GT-mask packages using tiny synthetic training data only."""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

import pandas as pd

from portmatch.data import QUERY_COLUMNS, read_queries
from portmatch.name_feature_cache import transform_cache
from portmatch.ranker_commands import add_training_parsers, run_training_command
from portmatch.submission_package import build_submission


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class RobustSubmissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture_spec = importlib.util.spec_from_file_location(
            "robust_submission_fixture", Path(__file__).with_name("test_robust_training.py"))
        fixture_module = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture_module)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.parser = argparse.ArgumentParser()
        add_training_parsers(cls.parser.add_subparsers(dest="command", required=True))
        train, cls.catalog = fixture_module.fixture()
        train.to_csv(cls.root / "train.csv", index=False, encoding="utf-8-sig")
        cls.catalog.to_csv(cls.root / "catalog.csv", index=False, encoding="utf-8-sig")
        cls.source, cls.cache = cls.root / "source", cls.root / "cache"
        cls.bundle = cls.root / "bundle"
        cls.run_cli(["prepare-ranker", "--train", str(cls.root / "train.csv"),
                     "--catalog", str(cls.root / "catalog.csv"), "--output", str(cls.source),
                     "--retrieval-profile", "robust", "--gt-augmentation", "missing",
                     "--folds", "3", "--inner-folds", "2", "--seed", "42",
                     "--max-train-queries", "4", "--max-valid-queries", "4"])
        with contextlib.redirect_stdout(io.StringIO()):
            transform_cache(cls.source, cls.cache, cls.root / "train.csv", cls.root / "catalog.csv")
        cls.run_cli(["train-ranker", "--data", str(cls.cache), "--output", str(cls.bundle),
                     "--exclude-features", "retrieval_tonnage_rank", "--rounds", "3",
                     "--threads", "6", "--num-leaves", "4", "--min-child-samples", "1", "--seed", "42"])
        cls.summary = json.loads((cls.bundle / "training_summary.json").read_text("utf-8"))
        cls.predictions = pd.read_csv(cls.bundle / "validation_predictions.csv", dtype=str)
        queries = read_queries(cls.root / "train.csv", labeled=True).set_index("sample_id", drop=False)
        cls.queries = queries.loc[cls.predictions.sample_id, QUERY_COLUMNS].reset_index(drop=True)
        (cls.bundle / "private.env").write_text("DO_NOT_PACKAGE", encoding="utf-8")

    @classmethod
    def run_cli(cls, arguments):
        with contextlib.redirect_stdout(io.StringIO()):
            run_training_command(cls.parser.parse_args(arguments))

    def test_default_stays_closed_and_opt_in_is_deterministic_and_minimal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Retrieval experiments"):
                build_submission(self.bundle, root / "default.zip")
            self.assertFalse((root / "default.zip").exists())
            first = build_submission(self.bundle, root / "one.zip", allow_robust=True)
            second = build_submission(self.bundle, root / "two.zip", allow_robust=True)
            self.assertEqual(first["sha256"], second["sha256"])
            with zipfile.ZipFile(root / "one.zip") as archive:
                self.assertEqual({name.split("/")[0] for name in archive.namelist()},
                                 {"script.py", "requirements.txt", "model"})
                self.assertFalse(any(name.endswith((".csv", ".npz", ".env")) for name in archive.namelist()))
                manifest = json.loads(archive.read("model/package_manifest.json"))
                self.assertEqual(manifest["baseline"], "robust_gt_missing_v1_86")
                minimal = json.loads(archive.read("model/ranker/training_summary.json"))
                self.assertEqual(minimal["training_weighting"], self.summary["training_weighting"])
                for name in ("retrieval", "retrieval_extension", "retrieval_index_sha256",
                             "training_augmentation", "training_augmentation_diagnostics",
                             "training_augmentation_assets"):
                    self.assertEqual(minimal["dataset"][name], self.summary["dataset"][name])
                text = json.dumps(minimal)
                for excluded in ("cache_directory", "train_path", "catalog_path", "validation_ids",
                                 "train_shards", "metrics", "source_directory"):
                    self.assertNotIn(excluded, text)
                for name in ("model.txt", "ranker.json", "index.joblib"):
                    self.assertEqual(archive.read("model/ranker/" + name), (self.bundle / name).read_bytes())

    def test_real_runtime_isolated_process_output_and_row_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_submission(self.bundle, root / "submit.zip", allow_robust=True)
            expected = self.predictions.set_index("sample_id").predicted_vessel_id.to_dict()
            cases = {"full": self.queries, "reversed": self.queries.iloc[::-1],
                     "subset": self.queries.iloc[::2], "single": self.queries.iloc[[1]]}
            for name, rows in cases.items():
                with self.subTest(case=name):
                    case = root / name
                    with zipfile.ZipFile(root / "submit.zip") as archive:
                        archive.extractall(case)
                    (case / "data").mkdir()
                    rows.to_csv(case / "data" / "test.csv", index=False)
                    self.catalog.to_csv(case / "data" / "vessel_catalog.csv", index=False)
                    completed = subprocess.run([sys.executable, "-I", str(case / "script.py")],
                                               cwd=root, text=True, capture_output=True, timeout=60)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    result = pd.read_csv(case / "output" / "submission.csv", dtype=str)
                    self.assertEqual(result.columns.tolist(), ["sample_id", "target_vessel_id"])
                    self.assertEqual(result.sample_id.tolist(), rows.sample_id.tolist())
                    self.assertEqual(result.target_vessel_id.tolist(),
                                     [expected[sample] for sample in rows.sample_id])
            # Hash verification happens before importing copied model code.
            script = root / "single" / "model" / "portmatch" / "robust_retrieval.py"
            with script.open("a", encoding="utf-8") as stream:
                stream.write("\n# modified after packaging\n")
            completed = subprocess.run([sys.executable, "-I", str(root / "single" / "script.py")],
                                       cwd=root, text=True, capture_output=True, timeout=60)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("changed size", completed.stderr)

    def test_metadata_and_weighting_tampering_and_reweighted_profile_fail_closed(self):
        for change in ("retrieval_code", "augmentation_code", "weighting", "weighting_code",
                       "language_weight", "augmentation_assets", "reweighting"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle"
                shutil.copytree(self.bundle, bundle)
                summary = copy.deepcopy(self.summary)
                if change == "retrieval_code":
                    summary["dataset"]["retrieval_extension"]["code_sha256"] = "0" * 64
                elif change == "augmentation_code":
                    summary["dataset"]["training_augmentation"]["generation_code_sha256"] = "0" * 64
                elif change == "weighting":
                    summary.pop("training_weighting")
                elif change == "weighting_code":
                    summary["training_weighting"]["ranking_code_sha256"] = "0" * 64
                elif change == "language_weight":
                    summary["training_weighting"]["languages"]["hangul"]["normalized_weight_share"] = .4
                elif change == "augmentation_assets":
                    summary["dataset"]["training_augmentation_assets"].pop("training_groups_sha256")
                else:
                    summary["dataset"]["training_reweighting"] = {"name": "gt_original_mass_v1"}
                _write_json(bundle / "training_summary.json", summary)
                output = root / "absent" / "submit.zip"
                with self.assertRaises(ValueError):
                    build_submission(bundle, output, allow_robust=True)
                self.assertFalse(output.parent.exists())

    def test_cache_asset_and_index_proofs_are_checked_without_loading_training_arrays(self):
        for change in ("querylog", "partitions", "retrieval_index"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle, cache = root / "bundle", root / "cache"
                shutil.copytree(self.bundle, bundle)
                cache.mkdir()
                for name in ("metadata.json", "derivation.json", "querylog.csv", "partitions.csv"):
                    shutil.copyfile(self.cache / name, cache / name)
                summary = copy.deepcopy(self.summary)
                if change == "retrieval_index":
                    summary["dataset"]["retrieval_index_sha256"] = "0" * 64
                    _write_json(cache / "metadata.json", summary["dataset"])
                    summary["cache_metadata_sha256"] = _digest(cache / "metadata.json")
                    _write_json(bundle / "training_summary.json", summary)
                else:
                    with (cache / (change + ".csv")).open("ab") as stream:
                        stream.write(b"changed")
                output = root / "absent" / "submit.zip"
                with self.assertRaisesRegex(ValueError, "asset proof|Robust index hash"):
                    build_submission(bundle, output, cache_directory=cache, allow_robust=True)
                self.assertFalse(output.parent.exists())

    def test_opt_in_does_not_accept_unaugmented_or_other_mass_or_thread_settings(self):
        from portmatch.gt_augmentation import GTMissingAugmentationConfig, augmentation_spec

        for change in ("unaugmented", "other_mass", "threads"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle"
                shutil.copytree(self.bundle, bundle)
                summary = copy.deepcopy(self.summary)
                if change == "unaugmented":
                    for key in ("training_augmentation", "training_augmentation_diagnostics",
                                "training_augmentation_assets"):
                        summary["dataset"].pop(key)
                    summary.pop("training_weighting")
                elif change == "other_mass":
                    summary["dataset"]["training_augmentation"] = augmentation_spec(
                        GTMissingAugmentationConfig(original_mass=.75))
                else:
                    state_path = bundle / "ranker.json"
                    state = json.loads(state_path.read_text("utf-8"))
                    state["config"]["n_jobs"] = 1
                    _write_json(state_path, state)
                _write_json(bundle / "training_summary.json", summary)
                output = root / "absent" / "submit.zip"
                with self.assertRaisesRegex(ValueError, "original-mass|six threads"):
                    build_submission(bundle, output, allow_robust=True)
                self.assertFalse(output.parent.exists())


if __name__ == "__main__":
    unittest.main()
