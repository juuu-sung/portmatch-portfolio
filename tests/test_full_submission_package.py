"""Actual full-profile ZIP tests using only a tiny synthetic public population."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from portmatch.data import read_catalog, read_queries
from portmatch.full_ranking import train_full_ranker, transform_full_cache
from portmatch.full_submission_package import build_full_submission, verify_full_copied_code
from portmatch.full_training_data import prepare_full_training_data
from portmatch.ranker_commands import feature_code_hash
from portmatch.submission_package import build_submission
from scripts.verify_full_submission import verify_full_submission


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@unittest.skipUnless(importlib.util.find_spec("lightgbm"), "LightGBM is not installed")
class FullSubmissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "full_package_fixture", Path(__file__).with_name("test_robust_training.py"))
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        train, catalog = fixture.fixture()
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        train.to_csv(cls.train_path, index=False, encoding="utf-8-sig")
        catalog.to_csv(cls.catalog_path, index=False, encoding="utf-8-sig")
        cls.train = read_queries(cls.train_path, labeled=True)
        cls.catalog = read_catalog(cls.catalog_path)
        cls.source, cls.cache, cls.bundle = (cls.root / name for name in ("source", "cache", "bundle"))
        provenance = {"train_path": str(cls.train_path), "catalog_path": str(cls.catalog_path),
                      "train_sha256": _hash(cls.train_path), "catalog_sha256": _hash(cls.catalog_path),
                      "feature_code_sha256": feature_code_hash()}
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_full_training_data(cls.train, cls.catalog, cls.source, provenance=provenance)
            transform_full_cache(cls.source, cls.cache, cls.train_path, cls.catalog_path)
            cls.summary = train_full_ranker(cls.cache, cls.bundle)
        (cls.bundle / "private.env").write_text("NEVER_INCLUDE", encoding="utf-8")

    def test_explicit_full_profile_minimal_runtime_and_deterministic_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "allow-full"):
                build_submission(self.bundle, root / "blocked.zip", allow_robust=True)
            self.assertFalse((root / "blocked.zip").exists())
            first = build_full_submission(self.bundle, root / "a.zip")
            second = build_full_submission(self.bundle, root / "b.zip")
            self.assertEqual(first["sha256"], second["sha256"])
            with zipfile.ZipFile(root / "a.zip") as zipped:
                names = zipped.namelist()
                self.assertEqual({name.split("/")[0] for name in names},
                                 {"script.py", "requirements.txt", "model"})
                self.assertFalse(any(name.endswith((".csv", ".npz", ".env")) for name in names))
                manifest = json.loads(zipped.read("model/package_manifest.json"))
                self.assertEqual(manifest["baseline"], "robust_gt_missing_v1_86_full")
                self.assertEqual(manifest["training_scope"], "full_public_train")
                minimal = json.loads(zipped.read("model/ranker/training_summary.json"))
                self.assertEqual(minimal["validation"], self.summary["validation"])
                self.assertFalse(minimal["validation"]["available"])
                self.assertEqual(minimal["dataset"]["full_training_population"]["rows"], len(self.train))
                self.assertEqual(minimal["dataset"]["full_training_pair_proof"],
                                 self.summary["dataset"]["full_training_pair_proof"])
                text = json.dumps(minimal)
                for forbidden in ("train_path", "catalog_path", "source_directory", "cache_directory",
                                  "train_shards", "metrics", "full_training_proof", "full_derivation_proof"):
                    self.assertNotIn(forbidden, text)
                for name in ("model.txt", "ranker.json", "index.joblib"):
                    self.assertEqual(zipped.read("model/ranker/" + name), (self.bundle / name).read_bytes())

    def test_actual_zip_six_probes_without_training_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_full_submission(self.bundle, root / "submit.zip")
            with contextlib.redirect_stdout(io.StringIO()):
                report = verify_full_submission(root / "submit.zip", root / "verify",
                    reference_model=self.bundle, train=self.train_path, catalog=self.catalog_path,
                    timeout=60, max_rows=6)
            self.assertTrue(report["complete"])
            self.assertTrue(report["row_independence_verified"])
            self.assertTrue(report["all_predictions_match_saved_model"])
            self.assertTrue(report["probe_rows_overlap_training"])
            self.assertFalse(report["accuracy_evaluated"])
            self.assertEqual(report["held_out_queries"], 0)
            self.assertEqual(report["rows"], 6)
            self.assertEqual(len(report["cases"]), 6)
            self.assertTrue(all(case["matches_reference"] and case["wall_seconds"] > 0
                                for case in report["cases"]))
            self.assertEqual([case["case"] for case in report["cases"]],
                             ["full", "reversed", "subset", "single",
                              "both_gt_missing_subset", "international_gt_x1_25_subset"])
            self.assertTrue(all(case["matches_full_by_id"] for case in report["cases"][:4]))
            self.assertTrue(all(case["matches_full_by_id"] is None for case in report["cases"][4:]))
            with self.assertRaises(FileExistsError):
                verify_full_submission(root / "submit.zip", root / "verify",
                    reference_model=self.bundle, train=self.train_path, catalog=self.catalog_path)

    def test_full_proofs_and_diagnostic_cache_fail_closed(self):
        for mutation in ("no_population_proof", "no_exact_population", "wrong_population",
                         "wrong_derivation", "wrong_pair_proof", "metrics", "limited"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle"
                shutil.copytree(self.bundle, bundle)
                summary = copy.deepcopy(self.summary)
                if mutation == "no_population_proof":
                    summary.pop("full_training_proof")
                elif mutation == "no_exact_population":
                    summary["full_training_proof"]["original_population_verified"] = False
                elif mutation == "wrong_population":
                    summary["full_training_proof"]["final_index_original_rows"] -= 1
                elif mutation == "wrong_derivation":
                    summary["full_derivation_proof"]["manifest_sha256"] = "0" * 64
                elif mutation == "wrong_pair_proof":
                    summary["dataset"]["full_training_pair_proof"]["sha256"] = "0" * 64
                elif mutation == "metrics":
                    summary["metrics"] = {"weighted_top1": 1.0}
                else:
                    summary["dataset"]["sampled"] = True
                    summary["dataset"]["max_train_queries"] = len(self.train)
                _write(bundle / "training_summary.json", summary)
                with self.assertRaises(ValueError):
                    build_full_submission(bundle, root / "not-created" / "submit.zip")
                self.assertFalse((root / "not-created").exists())

    def test_removed_or_null_full_marker_never_falls_back_to_fold_profile(self):
        for mutation in ("removed", "null"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "bundle"
                shutil.copytree(self.bundle, bundle)
                summary = copy.deepcopy(self.summary)
                for mapping in (summary, summary["dataset"]):
                    for name in list(mapping):
                        if name == "training_scope" or name.startswith("full_training"):
                            del mapping[name]
                if mutation == "null":
                    summary["dataset"]["training_scope"] = None
                _write(bundle / "training_summary.json", summary)
                with self.assertRaisesRegex(ValueError, "allow-full"):
                    build_submission(bundle, root / "bad.zip", allow_robust=True)
                self.assertFalse((root / "bad.zip").exists())

    def test_shared_cache_or_copied_full_source_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            for name in ("metadata.json", "derivation.json", "querylog.csv", "partitions.csv", "catalog_ids.npy", "index.joblib"):
                shutil.copyfile(self.cache / name, cache / name)
            with (cache / "catalog_ids.npy").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "asset changed"):
                build_full_submission(self.bundle, root / "bad.zip", cache_directory=cache)
            build_full_submission(self.bundle, root / "good.zip")
            with zipfile.ZipFile(root / "good.zip") as zipped:
                payload = {name: zipped.read(name) for name in zipped.namelist()}
                minimal = json.loads(payload["model/ranker/training_summary.json"])
            payload["model/portmatch/full_ranking.py"] += b"\n# changed\n"
            with self.assertRaisesRegex(ValueError, "source changed"):
                verify_full_copied_code(payload, minimal)

    def test_probe_rejects_wrong_source_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_full_submission(self.bundle, root / "submit.zip")
            bad = root / "train.csv"
            bad.write_bytes(self.train_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "provenance"):
                verify_full_submission(root / "submit.zip", root / "absent",
                    reference_model=self.bundle, train=bad, catalog=self.catalog_path)
            self.assertFalse((root / "absent").exists())


if __name__ == "__main__":
    unittest.main()
