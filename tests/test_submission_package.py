"""Submission packaging checks using a tiny synthetic model and index only."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import zipfile

import joblib
import numpy as np
import pandas as pd

from portmatch.feature_selection import FeatureSelection
from portmatch.name_feature_cache import _legacy_feature_names, transform_spec
from portmatch.name_features import FEATURE_NAMES
from portmatch.pipeline import MatchingPipeline
from portmatch.ranker_commands import RankedMatcher, feature_code_hash
from portmatch.ranking import RankerConfig, VesselRanker
from portmatch.submission_package import build_submission


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class SubmissionPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.bundle = cls.root / "bundle"
        cls.bundle.mkdir()
        cls.cache = cls.root / "cache"
        cls.cache.mkdir()
        cls.runtime = cls.root / "runtime"
        cls.runtime.mkdir()
        cls.query = dict(sample_id="S1", vessel_name="STAR ALPHA", call_sign="C1",
                         gross_tonnage=1000., international_gross_tonnage=1000.,
                         flag_name="Korea", vessel_type="Cargo", arrival_year=2024)
        catalog = pd.DataFrame([
            dict(vessel_id="V1", vessel_name="STAR ALPHA", gross_tonnage=1000., flag_name="Korea",
                 vessel_type="Cargo", size=10., unit="m", deadweight_tonnage=1200.),
            dict(vessel_id="V2", vessel_name="STAR BETA", gross_tonnage=1100., flag_name="Japan",
                 vessel_type="Cargo", size=11., unit="m", deadweight_tonnage=1300.),
        ])
        train = pd.DataFrame([{**cls.query, "target_vessel_id": "V1"}])
        pipeline = MatchingPipeline(train, catalog)
        joblib.dump(pipeline, cls.bundle / "index.joblib")
        cls.names = list(_legacy_feature_names()) + list(FEATURE_NAMES)
        selection = FeatureSelection.resolve(cls.names, exclude=["retrieval_tonnage_rank"])
        # Eight synthetic two-candidate groups; no competition cache or labels.
        X = np.zeros((16, 86), dtype=np.float32)
        X[::2, 0] = 1.
        config = RankerConfig(n_estimators=4, num_leaves=2, min_child_samples=1, n_jobs=1)
        ranker = VesselRanker(config).fit(X, np.tile([1, 0], 8), np.full(8, 2),
                                         np.array([True, False] * 4), list(selection.selected_features))
        ranker.save(cls.bundle)
        index_hash = digest((cls.bundle / "index.joblib").read_bytes())
        proof = {"schema_version": 1, "complete": True, "transform": transform_spec(),
                 "parent_metadata_sha256": "a" * 64,
                 "source_file_sha256": {"index.joblib": index_hash},
                 "derived_file_sha256": {"index.joblib": index_hash}}
        write_json(cls.cache / "derivation.json", proof)
        metadata = {
            "complete": True, "schema_version": 1,
            "feature_names": cls.names, "feature_transform": transform_spec(),
            "derivation": {"manifest_path": "derivation.json",
                           "manifest_sha256": digest((cls.cache / "derivation.json").read_bytes()),
                           "parent_metadata_sha256": "a" * 64},
            "provenance": {"feature_code_sha256": feature_code_hash(), "train_sha256": "b" * 64,
                           "catalog_sha256": "c" * 64,
                           "train_path": "C:/private_do_not_package/train.csv"},
            "outer_split": "vessel", "outer_folds": 5, "outer_fold": 0,
            "index_path": "index.joblib", "validation_ids": ["private_validation_id"],
        }
        write_json(cls.cache / "metadata.json", metadata)
        summary = {"complete": True, "schema_version": 1, "dataset": metadata,
                   "cache_directory": str(cls.cache),
                   "cache_metadata_sha256": digest((cls.cache / "metadata.json").read_bytes()),
                   "feature_selection": selection.to_dict(),
                   "trained_feature_names": list(selection.selected_features),
                   "ranker": asdict(config), "metrics": {"private_metric": 0.99}}
        write_json(cls.bundle / "training_summary.json", summary)
        # Only these allowlisted model assets should be copied.
        (cls.bundle / "validation_predictions.csv").write_text("DO_NOT_PACKAGE", encoding="utf-8")
        (cls.bundle / "secrets.env").write_text("DO_NOT_PACKAGE", encoding="utf-8")
        cls.expected_prediction = RankedMatcher(cls.bundle).predict(cls.query)
        (cls.runtime / "requirements.txt").write_text("# Uses the preinstalled environment.\n", encoding="utf-8")
        (cls.runtime / "script.py").write_text(
            "from pathlib import Path\nimport sys\n"
            "sys.path.insert(0, str(Path(__file__).parent / 'model'))\n"
            "from portmatch.ranker_commands import RankedMatcher\n"
            f"print(RankedMatcher(Path(__file__).parent / 'model' / 'ranker').predict({cls.query!r}))\n",
            encoding="utf-8")

    def build(self, output, bundle=None, cache=None):
        return build_submission(bundle or self.bundle, output, runtime_directory=self.runtime,
                                cache_directory=cache)

    def test_archive_layout_hashes_source_bytes_and_independent_roundtrip(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.build(root / "submit.zip")
            self.assertEqual(result["feature_count"], 86)
            self.assertEqual(result["sha256"], digest((root / "submit.zip").read_bytes()))
            with zipfile.ZipFile(root / "submit.zip") as archive:
                names = archive.namelist()
                self.assertEqual(set(name.split("/")[0] for name in names),
                                 {"script.py", "requirements.txt", "model"})
                self.assertEqual(names, sorted(names))
                self.assertFalse(any(".." in name or "\\" in name for name in names))
                self.assertFalse(any(name.endswith((".csv", ".npz", ".env")) for name in names))
                manifest = json.loads(archive.read("model/package_manifest.json"))
                self.assertEqual(set(manifest["files"]), set(names) - {"model/package_manifest.json"})
                for name, entry in manifest["files"].items():
                    content = archive.read(name)
                    self.assertEqual(digest(content), entry["sha256"])
                    self.assertEqual(len(content), entry["size_bytes"])
                for name in ("model.txt", "ranker.json", "index.joblib"):
                    self.assertEqual(archive.read("model/ranker/" + name), (self.bundle / name).read_bytes())
                for source in Path(__file__).resolve().parents[1].joinpath("portmatch").glob("*.py"):
                    self.assertEqual(archive.read("model/portmatch/" + source.name), source.read_bytes())
                summary_text = archive.read("model/ranker/training_summary.json").decode("utf-8")
                for omitted in ("cache_directory", "train_path", "validation_ids", "private_", "metrics"):
                    self.assertNotIn(omitted, summary_text)
                archive.extractall(root / "extracted")
            completed = subprocess.run([sys.executable, str(root / "extracted" / "script.py")],
                                       cwd=root, text=True, capture_output=True, check=True)
            self.assertEqual(completed.stdout.strip(), self.expected_prediction)

    def test_identical_inputs_produce_identical_zip_bytes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.build(root / "one.zip")
            self.build(root / "two.zip")
            self.assertEqual((root / "one.zip").read_bytes(), (root / "two.zip").read_bytes())

    def test_existing_output_is_preserved(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "submit.zip"
            target.write_bytes(b"existing submission")
            with self.assertRaises(FileExistsError):
                self.build(target)
            self.assertEqual(target.read_bytes(), b"existing submission")
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_malformed_model_summary_and_native_hash_rejected_before_output(self):
        for alteration in ("incomplete", "code", "features", "native", "retrieval_extension"):
            with self.subTest(alteration=alteration), TemporaryDirectory() as directory:
                root = Path(directory)
                bundle = root / "model"
                shutil.copytree(self.bundle, bundle)
                summary = json.loads((bundle / "training_summary.json").read_text("utf-8"))
                if alteration == "incomplete":
                    summary["complete"] = False
                elif alteration == "code":
                    summary["dataset"]["provenance"]["feature_code_sha256"] = "0" * 64
                elif alteration == "features":
                    summary["feature_selection"] = FeatureSelection.from_names(self.names).to_dict()
                elif alteration == "retrieval_extension":
                    summary["dataset"]["retrieval_extension"] = {"name": "robust_retrieval_v1"}
                else:
                    with (bundle / "model.txt").open("ab") as stream:
                        stream.write(b"changed")
                write_json(bundle / "training_summary.json", summary)
                output = root / "output" / "submit.zip"
                with self.assertRaises(ValueError):
                    self.build(output, bundle=bundle)
                self.assertFalse(output.parent.exists())

    def test_changed_index_and_tampered_derivation_rejected(self):
        for alteration in ("index", "proof", "metadata"):
            with self.subTest(alteration=alteration), TemporaryDirectory() as directory:
                root = Path(directory)
                bundle, cache = root / "model", root / "cache"
                shutil.copytree(self.bundle, bundle)
                shutil.copytree(self.cache, cache)
                path = {"index": bundle / "index.joblib", "proof": cache / "derivation.json",
                        "metadata": cache / "metadata.json"}[alteration]
                with path.open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaises(ValueError):
                    self.build(root / "submit.zip", bundle=bundle, cache=cache)
                self.assertFalse((root / "submit.zip").exists())


if __name__ == "__main__":
    unittest.main()
