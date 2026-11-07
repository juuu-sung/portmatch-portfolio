"""Training masks preserve OOF boundaries, source mass and original validation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.gt_augmentation import (
    GTMissingAugmentationConfig, VARIANTS, augmentation_spec,
    expected_query_masses, query_identity, query_variants, update_group_digest,
)
from portmatch.robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig
from portmatch.sampling import SamplingConfig
from portmatch.training_data import iter_shards, load_training_arrays, prepare_training_data, read_metadata


def fixture():
    spec = importlib.util.spec_from_file_location(
        "robust_augmentation_fixture", Path(__file__).with_name("test_robust_training.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture()


def groups(directory, kind="train"):
    result = {}
    for shard in iter_shards(directory, kind):
        offset = 0
        for index, size in enumerate(shard["group"]):
            result[str(shard["query_id"][index])] = {
                "X": shard["X"][offset:offset + size],
                "y": shard["y"][offset:offset + size],
                "catalog_index": shard["catalog_index"][offset:offset + size],
                **{name: shard[name][index] for name in
                   ("sample_id", "has_hangul", "target_catalog_index")},
                **({name: shard[name][index] for name in ("augmentation_variant", "query_mass")}
                   if "augmentation_variant" in shard else {}),
            }
            offset += size
    return result


class GTMaskTests(unittest.TestCase):
    def test_mask_order_original_preservation_noop_and_duplicate(self):
        row = {"sample_id": "S::source", "vessel_name": "한글", "gross_tonnage": 100,
               "international_gross_tonnage": 200, "unrelated": 3}
        output = list(query_variants(row))
        self.assertEqual([name for name, _, _ in output], list(VARIANTS))
        self.assertTrue(all(status == "ready" for _, _, status in output))
        self.assertEqual(row["gross_tonnage"], 100)
        self.assertEqual(output[0][1], row)
        self.assertIsNot(output[0][1], row)
        self.assertTrue(np.isnan(output[1][1]["gross_tonnage"]))
        self.assertTrue(np.isnan(output[1][1]["international_gross_tonnage"]))
        self.assertEqual(output[2][1]["gross_tonnage"], 100)
        self.assertEqual(output[3][1]["international_gross_tonnage"], 200)
        row["gross_tonnage"] = np.nan
        self.assertEqual([status for _, _, status in query_variants(row)],
                         ["ready", "ready", "skipped_duplicate", "skipped_noop"])
        row["international_gross_tonnage"] = None
        self.assertEqual([status for _, _, status in query_variants(row)],
                         ["ready", "skipped_noop", "skipped_noop", "skipped_noop"])
        self.assertEqual(query_identity("S::source", "S::source::row::both_gt_missing",
                                        "both_gt_missing", ["row", "vessel"]),
                         ("S::source", "row"))

    def test_mass_allocation_retained_views_and_skip_fallbacks(self):
        records = [("A", view, variant) for view in ("row", "vessel") for variant in VARIANTS]
        records += [("B", "row", "original"), ("B", "vessel", "both_gt_missing"),
                    ("C", "row", "international_gt_missing"), ("C", "row", "gross_gt_missing")]
        masses = expected_query_masses(records, GTMissingAugmentationConfig())
        self.assertEqual(masses["A", "row", "original"], .25)
        self.assertAlmostEqual(masses["A", "row", "both_gt_missing"], 1 / 12)
        for identity in records[-4:]:
            self.assertEqual(masses[identity], .5)
        for source in ("A", "B", "C"):
            self.assertAlmostEqual(sum(value for key, value in masses.items() if key[0] == source), 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            expected_query_masses(records + [records[0]], GTMissingAugmentationConfig())
        for invalid in (0, 1, -1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                GTMissingAugmentationConfig(invalid)


class GTAugmentationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.train, cls.catalog = fixture()
        # Include already-missing values and delimiter-bearing source IDs.
        cls.train.loc[0, "gross_tonnage"] = np.nan
        cls.train.loc[1, ["gross_tonnage", "international_gross_tonnage"]] = np.nan
        cls.train.loc[2, "sample_id"] = "S::special"
        cls.kwargs = dict(outer_folds=3, inner_folds=2, seed=42, shard_queries=7,
                          retrieval_config=RobustRetrievalConfig(),
                          sampling_config=SamplingConfig(hard_negatives=2, random_negatives=1, seed=42))
        cls.baseline, cls.augmented = cls.root / "baseline", cls.root / "augmented"
        prepare_training_data(cls.train, cls.catalog, cls.baseline, **cls.kwargs)
        cls.fits, cls.retrieved = [], []
        original_init, original_retrieve = RobustMatchingPipeline.__init__, RobustMatchingPipeline.retrieve

        def build(instance, fitting, catalog, config):
            cls.fits.append(fitting.copy())
            original_init(instance, fitting, catalog, config)

        def retrieve(instance, query):
            cls.retrieved.append(dict(query))
            return original_retrieve(instance, query)

        with patch.object(RobustMatchingPipeline, "__init__", new=build), \
                patch.object(RobustMatchingPipeline, "retrieve", new=retrieve):
            cls.metadata = prepare_training_data(
                cls.train, cls.catalog, cls.augmented,
                augmentation_config=GTMissingAugmentationConfig(), **cls.kwargs)

    def test_original_groups_validation_and_fitting_assets_are_unchanged(self):
        legacy, augmented = groups(self.baseline), groups(self.augmented)
        originals = {key: value for key, value in augmented.items() if value["augmentation_variant"] == "original"}
        self.assertEqual(legacy.keys(), originals.keys())
        for key, expected in legacy.items():
            for name, value in expected.items():
                np.testing.assert_array_equal(value, originals[key][name])
        legacy_valid, augmented_valid = groups(self.baseline, "valid"), groups(self.augmented, "valid")
        self.assertEqual(legacy_valid.keys(), augmented_valid.keys())
        for key, expected in legacy_valid.items():
            self.assertNotIn("query_mass", augmented_valid[key])
            for name, value in expected.items():
                np.testing.assert_array_equal(value, augmented_valid[key][name])
        pd.testing.assert_frame_equal(pd.read_csv(self.baseline / "partitions.csv"),
                                      pd.read_csv(self.augmented / "partitions.csv"))
        partitions = pd.read_csv(self.augmented / "partitions.csv")
        valid = partitions[partitions.role == "valid"]
        training = partitions[partitions.role == "train"]
        for fitting in self.fits:
            expected = self.train[self.train.sample_id.isin(fitting.sample_id)]
            pd.testing.assert_frame_equal(expected.reset_index(drop=True), fitting.reset_index(drop=True))
            self.assertFalse(set(fitting.sample_id) & set(valid.sample_id))
            self.assertFalse(set(fitting.target_vessel_id) & set(valid.target_vessel_id))
        for position, view in enumerate(("vessel", "row")):
            for fold in range(2):
                fitting = self.fits[position * 2 + fold]
                held = training[training["inner_" + view] == fold]
                self.assertFalse(set(fitting.sample_id) & set(held.sample_id))
                if view == "vessel":
                    self.assertFalse(set(fitting.target_vessel_id) & set(held.target_vessel_id))

    def test_every_usable_source_has_unit_mass_and_expected_masks(self):
        augmented = groups(self.augmented)
        masses = {}
        for group in augmented.values():
            masses[group["sample_id"]] = masses.get(group["sample_id"], 0) + group["query_mass"]
            self.assertEqual(int(group["y"].sum()), 1)
        self.assertTrue(all(abs(mass - 1) < 1e-12 for mass in masses.values()))
        self.assertEqual(self.metadata["training_augmentation"], augmentation_spec(GTMissingAugmentationConfig()))
        diagnostics = self.metadata["training_augmentation_diagnostics"]
        self.assertEqual(len(masses), diagnostics["retained_sources"])
        self.assertEqual(len(augmented), diagnostics["retained_groups"])
        self.assertAlmostEqual(sum(diagnostics["variant_query_mass"].values()), len(masses))
        for variant in VARIANTS:
            self.assertAlmostEqual(diagnostics["variant_query_mass"][variant], sum(
                group["query_mass"] for group in augmented.values() if group["augmentation_variant"] == variant))
        log = pd.read_csv(self.augmented / "querylog.csv")
        ready = log[log.status.isin(["stored", "skipped_missing_target", "skipped_without_negative"])]
        self.assertEqual(len(self.retrieved), len(ready))
        arrays = load_training_arrays(self.augmented, return_query_mass=True)
        self.assertEqual(len(arrays), 6)
        self.assertEqual(arrays[-1].dtype, np.float64)
        self.assertAlmostEqual(float(arrays[-1].sum()), len(masses))
        self.assertEqual(len(load_training_arrays(self.augmented)), 5)
        self.assertIsNone(load_training_arrays(self.baseline, return_query_mass=True)[-1])

    def test_non_robust_profile_rejected_before_output_creation(self):
        for profile in ("hangul", "dual-gt"):
            destination = self.root / ("invalid_" + profile)
            with self.assertRaisesRegex(ValueError, "profile robust"):
                prepare_training_data(self.train, self.catalog, destination,
                                      retrieval_config=RobustRetrievalConfig(profile=profile),
                                      augmentation_config=GTMissingAugmentationConfig())
            self.assertFalse(destination.exists())

    def test_missing_targets_are_never_injected_and_mass_is_redistributed(self):
        partitions = pd.read_csv(self.augmented / "partitions.csv")
        selected = partitions[partitions.role == "train"].sample_id.tolist()
        source_rows = self.train.set_index("sample_id")
        sources = [source for source in selected if not source_rows.loc[source,
                   ["gross_tonnage", "international_gross_tonnage"]].isna().any()][:4]
        self.assertEqual(len(sources), 4)
        ids = dict(zip(self.catalog.vessel_id, self.catalog.index))
        targets = {source: ids[source_rows.loc[source, "target_vessel_id"]] for source in sources}
        original_retrieve = RobustMatchingPipeline.retrieve

        def retrieve(instance, row):
            result = original_retrieve(instance, row)
            source = row["sample_id"]
            if source not in targets:
                return result
            masked = pd.isna(row["gross_tonnage"]) or pd.isna(row["international_gross_tonnage"])
            # A: original unavailable; B: masks unavailable; C: entire source unavailable.
            if source == sources[2] or (source == sources[0] and not masked) or (source == sources[1] and masked):
                return result[result.catalog_index != targets[source]].copy()
            # D: changed inputs retrieve a positive without any negative.
            if source == sources[3] and masked:
                return result[result.catalog_index == targets[source]].copy()
            return result

        destination = self.root / "skipped_targets"
        with patch.object(RobustMatchingPipeline, "retrieve", new=retrieve):
            metadata = prepare_training_data(self.train, self.catalog, destination,
                                             augmentation_config=GTMissingAugmentationConfig(), **self.kwargs)
        cached = groups(destination)
        retained = {source: [value for value in cached.values() if value["sample_id"] == source]
                    for source in sources}
        self.assertTrue(all(value["augmentation_variant"] != "original" for value in retained[sources[0]]))
        self.assertTrue(all(value["augmentation_variant"] == "original" for value in retained[sources[1]]))
        self.assertFalse(retained[sources[2]])
        self.assertTrue(all(value["augmentation_variant"] == "original" for value in retained[sources[3]]))
        for source in (sources[0], sources[1], sources[3]):
            self.assertAlmostEqual(sum(value["query_mass"] for value in retained[source]), 1)
        self.assertEqual(metadata["training_augmentation_diagnostics"]["dropped_sources"], 1)
        self.assertGreater(metadata["counts"]["train_missing_target"], 0)
        self.assertGreater(metadata["counts"]["train_without_negative"], 0)
        log = pd.read_csv(destination / "querylog.csv")
        skipped = log[log.status.isin(["skipped_missing_target", "skipped_without_negative"])]
        self.assertTrue(set(skipped.query_id).isdisjoint(cached))

    def test_metadata_and_mass_tampering_are_rejected(self):
        mutations = ("policy", "missing_spec", "audit_asset", "negative_mass", "mass_policy", "diagnostics",
                     "variant", "source", "unexpected_valid_array", "unmanifested")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                target = self.root / ("tampered_" + mutation)
                shutil.copytree(self.augmented, target)
                metadata = copy.deepcopy(self.metadata)
                kind = "valid" if mutation == "unexpected_valid_array" else "train"
                if mutation == "policy":
                    metadata["training_augmentation"]["mass_policy"] = "ignore sources"
                elif mutation == "missing_spec":
                    metadata.pop("training_augmentation")
                elif mutation == "audit_asset":
                    with (target / "querylog.csv").open("ab") as stream:
                        stream.write(b"modified")
                elif mutation == "diagnostics":
                    metadata["training_augmentation_diagnostics"]["retained_views"] += 1
                else:
                    shard_path = target / metadata[kind + "_shards"][0]
                    with np.load(shard_path, allow_pickle=False) as archive:
                        shard = {key: archive[key] for key in archive.files}
                    if mutation == "negative_mass":
                        shard["query_mass"][0] = -1
                    elif mutation == "mass_policy":
                        shard["query_mass"][0] *= 2
                    elif mutation == "variant":
                        shard["augmentation_variant"][0] = "gross_gt_missing"
                    elif mutation == "source":
                        shard["sample_id"][0] = "WRONG"
                    elif mutation == "unexpected_valid_array":
                        shard["query_mass"] = np.ones(len(shard["group"]), dtype=np.float64)
                    else:
                        for key in ("training_augmentation", "training_augmentation_diagnostics", "training_augmentation_assets"):
                            metadata.pop(key)
                    np.savez_compressed(shard_path, **shard)
                    if mutation == "mass_policy":
                        digest = hashlib.sha256()
                        for path in metadata["train_shards"]:
                            with np.load(target / path, allow_pickle=False) as archive:
                                update_group_digest(digest, archive)
                        metadata["training_augmentation_assets"]["training_groups_sha256"] = digest.hexdigest()
                (target / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises(ValueError):
                    list(iter_shards(target, kind))


if __name__ == "__main__":
    unittest.main()
