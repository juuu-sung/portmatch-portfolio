"""Full-data OOF isolation, source weighting and fixed inference population."""
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from portmatch.data import read_catalog, read_queries
from portmatch.full_training_data import (
    check_full_training_metadata, full_training_spec, prepare_full_training_data,
    verify_full_training_cache,
)
from portmatch.gt_augmentation import GTMissingAugmentationConfig
from portmatch.name_feature_cache import transform_cache, verify_derivation
from portmatch.ranker_commands import _hash_file, feature_code_hash
from portmatch.robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig
from portmatch.sampling import SamplingConfig
from portmatch.training_data import iter_shards, load_training_arrays, read_metadata


def fixture():
    spec = importlib.util.spec_from_file_location(
        "full_training_fixture", Path(__file__).with_name("test_robust_training.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture()


class FullTrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        train, catalog = fixture()
        train.loc[0, "gross_tonnage"] = np.nan
        train.loc[1, ["gross_tonnage", "international_gross_tonnage"]] = np.nan
        train.loc[2, "sample_id"] = "S::source"
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        train.to_csv(cls.train_path, index=False, encoding="utf-8-sig")
        catalog.to_csv(cls.catalog_path, index=False, encoding="utf-8-sig")
        cls.train, cls.catalog = read_queries(cls.train_path, labeled=True), read_catalog(cls.catalog_path)
        cls.provenance = {"train_sha256": _hash_file(cls.train_path),
                          "catalog_sha256": _hash_file(cls.catalog_path),
                          "feature_code_sha256": feature_code_hash()}
        cls.kwargs = dict(inner_folds=2, seed=42, shard_queries=9,
                          sampling_config=SamplingConfig(hard_negatives=2, random_negatives=1, seed=42),
                          provenance=cls.provenance)
        cls.fits, cls.retrieved = [], []
        init, retrieve = RobustMatchingPipeline.__init__, RobustMatchingPipeline.retrieve

        def build(instance, fitting, catalog, config):
            cls.fits.append(fitting.copy())
            instance._test_fitting_ids = set(fitting.sample_id)
            instance._test_fitting_targets = set(fitting.target_vessel_id)
            init(instance, fitting, catalog, config)

        def queried(instance, row):
            if "target_vessel_id" in row:
                raise AssertionError("Label passed into retrieval")
            if row["sample_id"] in instance._test_fitting_ids:
                raise AssertionError("Own query fitted its OOF index")
            cls.retrieved.append((dict(row), set(instance._test_fitting_ids),
                                  set(instance._test_fitting_targets)))
            return retrieve(instance, row)

        cls.source = cls.root / "full"
        with patch.object(RobustMatchingPipeline, "__init__", new=build), \
                patch.object(RobustMatchingPipeline, "retrieve", new=queried):
            cls.metadata = prepare_full_training_data(cls.train, cls.catalog, cls.source, **cls.kwargs)
        cls.derived = cls.root / "derived"
        cls.transformed = transform_cache(cls.source, cls.derived, cls.train_path, cls.catalog_path)

    def test_internal_queries_exclude_self_and_vessel_view_excludes_whole_vessel(self):
        self.assertEqual(len(self.fits), 5)
        partitions = pd.read_csv(self.source / "partitions.csv")
        for number, view in enumerate(("vessel", "row")):
            for fold in range(2):
                fitting = self.fits[number * 2 + fold]
                held = partitions[partitions["inner_" + view] == fold]
                self.assertFalse(set(fitting.sample_id) & set(held.sample_id))
                if view == "vessel":
                    self.assertFalse(set(fitting.target_vessel_id) & set(held.target_vessel_id))
                original = self.train[self.train.sample_id.isin(fitting.sample_id)]
                pd.testing.assert_frame_equal(original.reset_index(drop=True), fitting.reset_index(drop=True))
        for row, fitting_ids, _ in self.retrieved:
            self.assertNotIn(row["sample_id"], fitting_ids)
        # The last fit serves inference only and never generated an OOF group.
        pd.testing.assert_frame_equal(self.fits[-1], self.train)

    def test_no_outer_validation_or_score_and_all_originals_in_final_index(self):
        metadata = self.metadata
        self.assertEqual(metadata["training_scope"], "full_public_train")
        self.assertEqual(metadata["outer_split"], "none")
        self.assertIsNone(metadata["outer_fold"])
        self.assertEqual(metadata["outer_folds"], 0)
        self.assertEqual(metadata["valid_shards"], [])
        self.assertEqual(metadata["counts"]["valid_groups"], 0)
        self.assertEqual(metadata["counts"]["outer_training_rows"], len(self.train))
        self.assertFalse((self.source / "valid").exists())
        self.assertEqual(list(iter_shards(self.source, "valid")), [])
        self.assertNotIn("validation_metrics", metadata)
        pipeline = joblib.load(self.source / "index.joblib")
        self.assertEqual(pipeline.memory.training_rows, len(self.train))
        self.assertEqual(pipeline.memory.training_vessels, frozenset(self.train.target_vessel_id))
        proof = verify_full_training_cache(self.source, train_path=self.train_path, catalog_path=self.catalog_path)
        self.assertTrue(proof["verified"])
        self.assertTrue(proof["original_population_verified"])
        self.assertEqual(proof["held_out_validation_rows"], 0)
        self.assertEqual(proof["selected_sources"], len(self.train))

    def test_each_retained_original_has_unit_mass_with_original_half_policy(self):
        masses, identities = {}, []
        for shard in iter_shards(self.source):
            for source, variant, query, mass in zip(shard["sample_id"], shard["augmentation_variant"],
                                                   shard["query_id"], shard["query_mass"]):
                masses[str(source)] = masses.get(str(source), 0) + float(mass)
                identities.append(str(query))
                self.assertIn(str(variant), ("original", "both_gt_missing", "international_gt_missing", "gross_gt_missing"))
        self.assertEqual(len(identities), len(set(identities)))
        self.assertTrue(all(abs(mass - 1) < 1e-12 for mass in masses.values()))
        self.assertEqual(self.metadata["training_augmentation"]["config"]["original_mass"], .5)
        arrays = load_training_arrays(self.source, return_query_mass=True)
        self.assertAlmostEqual(float(arrays[-1].sum()), len(masses))

    def test_name_transform_preserves_full_scope_proof_and_all_invariants(self):
        proof = verify_full_training_cache(self.derived, train=self.train, catalog=self.catalog)
        self.assertTrue(proof["original_population_verified"])
        self.assertEqual(self.transformed["full_training"], self.metadata["full_training"])
        self.assertEqual(self.transformed["full_training_assets"], self.metadata["full_training_assets"])
        self.assertEqual(self.transformed["full_training_population"], self.metadata["full_training_population"])
        self.assertEqual(self.transformed["full_training_pair_proof"], self.metadata["full_training_pair_proof"])
        self.assertEqual(len(self.transformed["feature_names"]), 87)
        verify_derivation(self.derived, _hash_file(self.source / "metadata.json"))

    def test_diagnostic_limit_still_fits_all_originals_but_marks_partial_feature_generation(self):
        destination = self.root / "sampled"
        metadata = prepare_full_training_data(self.train, self.catalog, destination,
                                              max_train_queries=2, **self.kwargs)
        proof = verify_full_training_cache(destination, train=self.train, catalog=self.catalog)
        self.assertTrue(metadata["sampled"])
        self.assertEqual(proof["selected_sources"], 2)
        self.assertEqual(proof["final_index_original_rows"], len(self.train))
        self.assertEqual(metadata["full_training_population"]["rows"], len(self.train))

    def test_missing_target_not_injected_dropped_source_still_in_final_index(self):
        original = RobustMatchingPipeline.retrieve
        source = self.train.iloc[0]
        source_id = source.sample_id
        target = int(self.catalog.index[self.catalog.vessel_id.eq(source.target_vessel_id)][0])

        def omit(instance, row):
            candidates = original(instance, row)
            if row["sample_id"] == source_id:
                candidates = candidates[~candidates.catalog_index.eq(target)]
            return candidates

        destination = self.root / "missing"
        with patch.object(RobustMatchingPipeline, "retrieve", new=omit):
            metadata = prepare_full_training_data(self.train, self.catalog, destination, **self.kwargs)
        stored = {str(source) for shard in iter_shards(destination) for source in shard["sample_id"]}
        self.assertNotIn(source_id, stored)
        self.assertGreaterEqual(metadata["training_augmentation_diagnostics"]["dropped_sources"], 1)
        self.assertEqual(metadata["counts"]["selected_training_queries"], len(self.train))
        self.assertEqual(joblib.load(destination / "index.joblib").memory.training_rows, len(self.train))
        verify_full_training_cache(destination, train=self.train, catalog=self.catalog)

    def test_invalid_mode_mass_and_nonempty_output_rejected(self):
        cases = [dict(inner_split="vessel"), dict(inner_folds=1),
                 dict(augmentation_config=GTMissingAugmentationConfig(.75)),
                 dict(retrieval_config=RobustRetrievalConfig(profile="hangul")),
                 dict(retrieval_config=RobustRetrievalConfig(max_total=20))]
        for number, values in enumerate(cases):
            destination = self.root / f"invalid_{number}"
            with self.subTest(values=values), self.assertRaises(ValueError):
                prepare_full_training_data(self.train, self.catalog, destination, **(self.kwargs | values))
            self.assertFalse(destination.exists())
        with self.assertRaises(FileExistsError):
            prepare_full_training_data(self.train, self.catalog, self.source, **self.kwargs)
        self.assertIsNone(check_full_training_metadata({"outer_split": "vessel"}))
        with self.assertRaisesRegex(ValueError, "scope"):
            check_full_training_metadata({"outer_split": "vessel", "training_scope": None})
        with self.assertRaises(ValueError):
            check_full_training_metadata({"outer_split": "none"})
        invalid = dict(self.metadata)
        invalid.pop("training_scope")
        with self.assertRaises(ValueError):
            check_full_training_metadata(invalid)

    def test_changed_scope_count_index_or_input_population_rejected(self):
        changed = json.loads(json.dumps(self.metadata))
        changed["counts"]["outer_validation_rows"] = 1
        with self.assertRaisesRegex(ValueError, "counts"):
            check_full_training_metadata(changed)
        wrong_train = self.train.copy()
        wrong_train.loc[0, "vessel_name"] = "altered name"
        with self.assertRaisesRegex(ValueError, "population"):
            verify_full_training_cache(self.source, train=wrong_train, catalog=self.catalog)
        destination = self.root / "tampered"
        shutil.copytree(self.source, destination)
        with (destination / "querylog.csv").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "asset changed|audit asset changed"):
            verify_full_training_cache(destination)

    def test_raw_and_transformed_original_feature_mutation_rejected(self):
        for name, source in (("raw", self.source), ("derived", self.derived)):
            destination = self.root / ("mutated_X_" + name)
            shutil.copytree(source, destination)
            metadata = read_metadata(destination)
            path = destination / metadata["train_shards"][0]
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["X"][0, 0] = np.float32(17.5)
            np.savez_compressed(path, **arrays)
            with self.subTest(cache=name), self.assertRaisesRegex(ValueError, "feature or pair arrays changed"):
                verify_full_training_cache(destination)


if __name__ == "__main__":
    unittest.main()
