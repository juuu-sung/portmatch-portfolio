"""Synthetic final-training integration without scoring fitted rows as validation."""
from dataclasses import asdict
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from portmatch.data import read_catalog, read_queries
from portmatch.full_ranking import (
    check_full_model_summary, full_ranker_spec, train_full_ranker, transform_full_cache,
)
from portmatch.full_training_data import prepare_full_training_data, verify_full_training_cache
from portmatch.ranker_commands import RankedMatcher, _hash_file, feature_code_hash
from portmatch.ranking import RankerConfig, VesselRanker
from portmatch.training_data import iter_shards, load_training_arrays


def fixture():
    spec = importlib.util.spec_from_file_location(
        "full_ranking_fixture", Path(__file__).with_name("test_robust_training.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture()


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class FullRankingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        train, catalog = fixture()
        train.loc[0, "gross_tonnage"] = np.nan
        train.loc[1, ["gross_tonnage", "international_gross_tonnage"]] = np.nan
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        train.to_csv(cls.train_path, index=False, encoding="utf-8-sig")
        catalog.to_csv(cls.catalog_path, index=False, encoding="utf-8-sig")
        cls.train = read_queries(cls.train_path, labeled=True)
        cls.catalog = read_catalog(cls.catalog_path)
        cls.source, cls.derived, cls.model = [cls.root / name for name in ("source", "derived", "model")]
        cls.provenance = {
            "train_path": str(cls.train_path), "catalog_path": str(cls.catalog_path),
            "train_sha256": _hash_file(cls.train_path), "catalog_sha256": _hash_file(cls.catalog_path),
            "feature_code_sha256": feature_code_hash(),
        }
        cls.parent = prepare_full_training_data(
            cls.train, cls.catalog, cls.source, provenance=cls.provenance, shard_queries=13)
        cls.source_files = {str(path.relative_to(cls.source)): path.read_bytes()
                            for path in cls.source.rglob("*") if path.is_file()}
        cls.metadata = transform_full_cache(cls.source, cls.derived, cls.train_path, cls.catalog_path)
        cls.fit_masses = None
        original_fit = VesselRanker.fit

        def capture_fit(instance, X, y, group, language, names, *, query_mass=None):
            cls.fit_masses = query_mass.copy()
            cls.fit_names = list(names)
            cls.fit_groups = group.copy()
            return original_fit(instance, X, y, group, language, names, query_mass=query_mass)

        with patch("portmatch.ranker_commands.evaluate_cache", side_effect=AssertionError("scored validation")):
            with patch.object(VesselRanker, "fit", new=capture_fit):
                cls.summary = train_full_ranker(cls.derived, cls.model)

    def test_transform_preserves_all_base_arrays_and_source_bytes(self):
        for relative, expected in self.source_files.items():
            self.assertEqual((self.source / relative).read_bytes(), expected)
        for old, new in zip(iter_shards(self.source), iter_shards(self.derived)):
            self.assertEqual(set(old), set(new))
            for key in old:
                values = new[key][:, :70] if key == "X" else new[key]
                self.assertEqual(old[key].dtype, values.dtype)
                self.assertEqual(old[key].shape, values.shape)
                self.assertEqual(old[key].tobytes(), values.tobytes())
            self.assertEqual(new["X"].shape[1], 87)
        self.assertEqual(self.metadata["valid_shards"], [])
        self.assertEqual(self.metadata["full_training_assets"], self.parent["full_training_assets"])
        self.assertTrue(verify_full_training_cache(self.derived)["verified"])

    def test_fit_receives_exact_source_masses_and_canonical_features_without_validation(self):
        arrays = load_training_arrays(self.derived, feature_names=self.fit_names, return_query_mass=True)
        np.testing.assert_array_equal(self.fit_masses, arrays[-1])
        np.testing.assert_array_equal(self.fit_groups, arrays[2])
        self.assertEqual(len(self.fit_names), 86)
        self.assertNotIn("retrieval_tonnage_rank", self.fit_names)
        self.assertEqual(self.summary["ranker"], asdict(RankerConfig()))
        self.assertEqual(self.summary["training_scope"], "full_public_train")
        self.assertEqual(self.summary["full_training_trainer"], full_ranker_spec())
        self.assertTrue(self.summary["full_training_proof"]["original_population_verified"])
        self.assertEqual(self.summary["validation"], {
            "available": False, "held_out_queries": 0, "reason": "no_external_holdout_full_training_scope"})
        self.assertNotIn("metrics", self.summary)
        self.assertFalse((self.model / "metrics.json").exists())
        self.assertFalse((self.model / "validation_predictions.csv").exists())
        self.assertEqual((self.model / "index.joblib").read_bytes(),
                         (self.source / "index.joblib").read_bytes())

    def test_saved_model_predicts_independently_of_order_other_rows_and_labels(self):
        matcher = RankedMatcher(self.model)
        rows = self.train.to_dict("records")[:4]
        expected = {row["sample_id"]: matcher.predict(row) for row in rows}
        reverse = {row["sample_id"]: matcher.predict(row) for row in rows[::-1]}
        self.assertEqual(expected, reverse)
        fresh = RankedMatcher(self.model)
        query = {**rows[1], "target_vessel_id": "not-a-label", "unrelated_other_rows": rows}
        self.assertEqual(fresh.predict(query), expected[query["sample_id"]])
        self.assertEqual(fresh.predict(rows[1]), expected[query["sample_id"]])

    def test_summary_fails_closed_for_scope_features_profile_metrics_and_trainer_changes(self):
        state = json.loads((self.model / "ranker.json").read_text(encoding="utf-8"))
        check_full_model_summary(self.summary, model_state=state)
        variants = []
        for key, value in (("training_scope", "outer_fold"), ("validation", {"available": True}),
                           ("metrics", {}), ("validation_metrics", {}),
                           ("full_training_trainer", {"name": "unknown"}),
                           ("ranker", asdict(RankerConfig(n_estimators=301)))):
            changed = copy.deepcopy(self.summary)
            changed[key] = value
            variants.append(changed)
        changed = copy.deepcopy(self.summary)
        changed["trained_feature_names"] = changed["trained_feature_names"][::-1]
        variants.append(changed)
        changed = copy.deepcopy(self.summary)
        changed["dataset"]["inner_folds"] = 2
        variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_full_model_summary(changed, model_state=state)
        changed_state = copy.deepcopy(state)
        changed_state["config"]["n_jobs"] = 1
        with self.assertRaisesRegex(ValueError, "Native full model"):
            check_full_model_summary(self.summary, model_state=changed_state)
        with self.assertRaisesRegex(ValueError, "missing its full-training dataset"):
            check_full_model_summary({"dataset": {}, "training_scope": "unknown"})

    def test_train_rejects_nonfixed_config_output_reuse_and_nested_cache(self):
        for config in (RankerConfig(n_estimators=2), RankerConfig(n_jobs=1),
                       RankerConfig(learning_rate=.1), RankerConfig(seed=42)):
            output = self.root / "not_created"
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "fixes RankerConfig"):
                train_full_ranker(self.derived, output, config=config)
            self.assertFalse(output.exists())
        with self.assertRaises(FileExistsError):
            train_full_ranker(self.derived, self.model)
        with self.assertRaisesRegex(ValueError, "separate"):
            train_full_ranker(self.derived, self.derived / "model")

    def test_tampered_derived_features_rejected_before_fit(self):
        data = self.root / "tampered"
        shutil.copytree(self.derived, data)
        relative = self.metadata["train_shards"][0]
        with np.load(data / relative, allow_pickle=False) as archive:
            shard = {key: archive[key] for key in archive.files}
        shard["X"][0, 0] += np.float32(.125)
        np.savez_compressed(data / relative, **shard)
        with patch.object(VesselRanker, "fit") as fit:
            with self.assertRaisesRegex(ValueError, "hash mismatch|invariant|proof|array"):
                train_full_ranker(data, self.root / "bad_model")
            fit.assert_not_called()
        self.assertFalse((self.root / "bad_model").exists())

    def test_transform_rejects_nonfull_and_changed_input_before_output(self):
        data = self.root / "not_full"
        shutil.copytree(self.source, data)
        metadata = copy.deepcopy(self.parent)
        metadata.pop("training_scope")
        (data / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            transform_full_cache(data, self.root / "bad_transform", self.train_path, self.catalog_path)
        self.assertFalse((self.root / "bad_transform").exists())
        changed = self.root / "different.csv"
        changed.write_bytes(self.train_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hash|provenance|differs"):
            transform_full_cache(self.source, self.root / "bad_input", changed, self.catalog_path)
        self.assertFalse((self.root / "bad_input").exists())

    def test_training_checks_current_original_files_before_fit(self):
        changed = self.root / "changed_original.csv"
        changed.write_bytes(self.train_path.read_bytes() + b"\n")
        output = self.root / "changed_source_model"
        with patch.object(VesselRanker, "fit") as fit:
            with self.assertRaisesRegex(ValueError, "provenance"):
                train_full_ranker(self.derived, output, train_path=changed,
                                  catalog_path=self.catalog_path)
            fit.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
