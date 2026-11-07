"""Small cache-only reweighting checks, including the preserved feature lineage."""
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
from unittest.mock import patch

import numpy as np

from portmatch.data import read_catalog, read_queries
from portmatch.gt_augmentation import GTMissingAugmentationConfig, expected_query_masses
from portmatch.gt_reweight_cache import (
    MANIFEST_NAME, _invariant_digest, _mass_diagnostics, _records,
    check_reweighting_metadata, reweight_cache, validate_reweight_pair, verify_reweighting,
)
from portmatch.name_feature_cache import transform_cache
from portmatch.ranker_commands import feature_code_hash
from portmatch.robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig
from portmatch.sampling import SamplingConfig
from portmatch.training_data import iter_shards, load_training_arrays, prepare_training_data, read_metadata


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*") if p.is_file()}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class GTReweightCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        spec = importlib.util.spec_from_file_location(
            "reweight_fixture", Path(__file__).with_name("test_robust_training.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        queries, catalog = module.fixture()
        # Exercise missing/no-op masks and source IDs with the view delimiter.
        queries.loc[0, "gross_tonnage"] = np.nan
        queries.loc[1, ["gross_tonnage", "international_gross_tonnage"]] = np.nan
        queries.loc[2, "sample_id"] = "S::special"
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        queries.to_csv(cls.train_path, index=False, encoding="utf-8-sig")
        catalog.to_csv(cls.catalog_path, index=False, encoding="utf-8-sig")
        queries, catalog = read_queries(cls.train_path, labeled=True), read_catalog(cls.catalog_path)
        cls.raw, cls.source, cls.derived = [cls.root / name for name in ("raw", "source", "derived")]
        with contextlib.redirect_stdout(io.StringIO()):
            parent = prepare_training_data(
                queries, catalog, cls.raw, outer_folds=3, inner_folds=2, seed=42,
                shard_queries=11, retrieval_config=RobustRetrievalConfig(),
                augmentation_config=GTMissingAugmentationConfig(),
                sampling_config=SamplingConfig(hard_negatives=2, random_negatives=1, seed=42),
                provenance={"feature_code_sha256": feature_code_hash(),
                            "train_sha256": file_hash(cls.train_path),
                            "catalog_sha256": file_hash(cls.catalog_path)})
            # Bit-preservation includes noncanonical NaN payloads and signed zero.
            first = cls.raw / parent["train_shards"][0]
            with np.load(first, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            arrays["X"].view(np.uint32)[0, 0] = 0x7FC00042
            arrays["X"].view(np.uint32)[0, 1] = 0x80000000
            np.savez_compressed(first, **arrays)
            cls.control = transform_cache(cls.raw, cls.source, cls.train_path, cls.catalog_path)
            cls.before = inventory(cls.source)
            # Reweighting must never rerun retrieval, fit a new index or calculate features.
            with patch.object(RobustMatchingPipeline, "__init__", side_effect=AssertionError("fit")), \
                    patch.object(RobustMatchingPipeline, "retrieve", side_effect=AssertionError("retrieve")), \
                    patch.object(RobustMatchingPipeline, "features", side_effect=AssertionError("features")), \
                    patch("portmatch.name_feature_cache.transform_features", side_effect=AssertionError("features")):
                cls.metadata = reweight_cache(cls.source, cls.derived, original_mass=.75)

    def copied(self, name):
        path = self.root / name
        shutil.copytree(self.derived, path)
        return path

    def refresh_asset_proof(self, root, relative):
        """Simulate a foreign same-shaped asset whose superficial hashes were refreshed."""
        manifest = json.loads((root / MANIFEST_NAME).read_text("utf-8"))
        manifest["derived_file_sha256"][relative] = file_hash(root / relative)
        write_json(root / MANIFEST_NAME, manifest)
        metadata = json.loads((root / "metadata.json").read_text("utf-8"))
        metadata["training_reweighting"]["manifest_sha256"] = file_hash(root / MANIFEST_NAME)
        write_json(root / "metadata.json", metadata)

    def test_preserves_all_nonmass_array_bits_and_source_files(self):
        self.assertEqual(self.before, inventory(self.source))
        self.assertEqual(self.control["feature_transform"], self.metadata["feature_transform"])
        self.assertEqual(self.control["derivation"], self.metadata["derivation"])
        self.assertEqual(len(self.metadata["feature_names"]), 87)
        mass_changed = False
        for kind in ("train", "valid"):
            old_shards, new_shards = list(iter_shards(self.source, kind)), list(iter_shards(self.derived, kind))
            for old, new in zip(old_shards, new_shards):
                self.assertEqual(set(old), set(new))
                for key in old:
                    self.assertEqual(old[key].dtype, new[key].dtype)
                    self.assertEqual(old[key].shape, new[key].shape)
                    if key == "query_mass":
                        mass_changed |= not np.array_equal(old[key], new[key])
                    else:
                        self.assertEqual(np.ascontiguousarray(old[key]).tobytes(),
                                         np.ascontiguousarray(new[key]).tobytes())
        self.assertTrue(mass_changed)
        for relative, expected in self.before.items():
            if relative not in self.metadata["train_shards"] and relative != "metadata.json":
                self.assertEqual(file_hash(self.derived / relative), expected)
                self.assertFalse((self.source / relative).samefile(self.derived / relative))
        proof = verify_reweighting(self.derived, self.before["metadata.json"], self.control)
        self.assertTrue(proof["verified"])
        self.assertEqual((proof["source_original_mass"], proof["target_original_mass"]), (.5, .75))

    def test_expected_masses_diagnostics_and_training_array_loading(self):
        records, observed, languages = [], [], {}
        for shard in iter_shards(self.derived):
            records.extend(_records(self.metadata, shard))
            observed.extend(shard["query_mass"])
            languages.update({str(source): bool(hangul) for source, hangul in
                              zip(shard["sample_id"], shard["has_hangul"])})
        expected = expected_query_masses(records, GTMissingAugmentationConfig(.75))
        np.testing.assert_array_equal(observed, [expected[identity] for identity in records])
        diagnostic = _mass_diagnostics(records, languages, expected)
        for key, value in diagnostic.items():
            self.assertEqual(value, self.metadata["training_augmentation_diagnostics"][key])
        for source in languages:
            self.assertAlmostEqual(sum(mass for record, mass in expected.items() if record[0] == source), 1)
        np.testing.assert_array_equal(load_training_arrays(self.derived, return_query_mass=True)[-1], observed)
        self.assertNotEqual(self.control["training_augmentation_assets"]["training_groups_sha256"],
                            self.metadata["training_augmentation_assets"]["training_groups_sha256"])

    def test_retained_view_fallback_policy_for_missing_originals_and_masks(self):
        # An absent original or absent masks keeps its view mass; unavailable views receive none.
        records = [("A::id", "vessel", "original"), ("A::id", "vessel", "both_gt_missing"),
                   ("A::id", "vessel", "gross_gt_missing"), ("A::id", "row", "original"),
                   ("B", "row", "both_gt_missing"), ("B", "row", "gross_gt_missing"),
                   ("C", "vessel", "original")]
        masses = expected_query_masses(records, GTMissingAugmentationConfig(.75))
        self.assertEqual(masses["A::id", "vessel", "original"], .375)
        self.assertEqual(masses["A::id", "vessel", "both_gt_missing"], .0625)
        self.assertEqual(masses["A::id", "row", "original"], .5)
        self.assertEqual(masses["B", "row", "both_gt_missing"], .5)
        self.assertEqual(masses["C", "vessel", "original"], 1)
        stats = _mass_diagnostics(records, {"A::id": True, "B": False, "C": False}, masses)
        self.assertEqual(stats["total_query_mass"], 3)
        self.assertEqual(stats["hangul_query_mass"], 1)
        self.assertEqual(stats["variant_query_mass"]["original"], 1.875)

    def test_invalid_mass_overlap_reuse_raw_source_and_chaining_rejected(self):
        for value in (0, 1, True, "0.75", float("nan"), float("inf"), .5):
            destination = self.root / "invalid_mass"
            with self.subTest(value=value), self.assertRaises(ValueError):
                reweight_cache(self.source, destination, value)
            self.assertFalse(destination.exists())
        for destination in (self.source, self.source / "nested", self.root):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "separate"):
                reweight_cache(self.source, destination, .75)
        with self.assertRaises(FileExistsError):
            reweight_cache(self.source, self.derived, .75)
        for source in (self.raw, self.derived):
            destination = self.root / "invalid_source"
            with self.subTest(source=source), self.assertRaises(ValueError):
                reweight_cache(source, destination, .75)
            self.assertFalse(destination.exists())
        self.assertEqual(self.before, inventory(self.source))

    def test_metadata_binds_exact_control_and_rejects_changed_experiment(self):
        self.assertIsNone(check_reweighting_metadata(self.control))
        validate_reweight_pair(self.control, self.metadata, self.before["metadata.json"])
        with self.assertRaisesRegex(ValueError, "this control cache"):
            validate_reweight_pair(self.control, self.metadata, "a" * 64)
        with self.assertRaisesRegex(ValueError, "expected control cache"):
            verify_reweighting(self.derived, "a" * 64)
        foreign = copy.deepcopy(self.control)
        foreign["seed"] += 1
        with self.assertRaisesRegex(ValueError, "expected control dataset"):
            verify_reweighting(self.derived, expected_source_dataset=foreign)
        for key, value in (("seed", 99), ("feature_names", self.metadata["feature_names"][::-1])):
            changed = copy.deepcopy(self.metadata)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_reweight_pair(self.control, changed)
        for field, value in (("code_sha256", "0" * 64), ("target_original_mass", .6),
                             ("source_original_mass", .75), ("manifest_path", "../reweighting.json")):
            changed = copy.deepcopy(self.metadata)
            changed["training_reweighting"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_reweighting_metadata(changed)

    def test_changed_same_shaped_feature_array_rejected_even_with_refreshed_file_hash(self):
        root = self.copied("changed_features")
        relative = self.metadata["train_shards"][0]
        with np.load(root / relative, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["X"][0, 3] += 1
        np.savez_compressed(root / relative, **arrays)
        self.refresh_asset_proof(root, relative)
        with self.assertRaisesRegex(ValueError, "invariant arrays"):
            verify_reweighting(root)

    def test_changed_validation_or_audit_asset_rejected_with_refreshed_file_hash(self):
        for relative in (self.metadata["valid_shards"][0], "index.joblib"):
            root = self.copied("changed_" + Path(relative).name)
            with (root / relative).open("ab") as stream:
                stream.write(b"changed")
            self.refresh_asset_proof(root, relative)
            with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "validation or audit asset"):
                verify_reweighting(root)

    def test_invalid_mass_policy_and_extra_assets_rejected(self):
        root = self.copied("changed_mass")
        relative = self.metadata["train_shards"][0]
        with np.load(root / relative, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["query_mass"][0] *= .5
        # The invariant proof deliberately excludes mass, so the full loader must catch this.
        before = _invariant_digest(arrays)
        np.savez_compressed(root / relative, **arrays)
        self.assertEqual(before, _invariant_digest(arrays))
        self.refresh_asset_proof(root, relative)
        with self.assertRaisesRegex(ValueError, "normalization policy"):
            verify_reweighting(root)
        root = self.copied("extra_asset")
        (root / "unexpected.txt").write_text("foreign")
        with self.assertRaisesRegex(ValueError, "unexpected or missing assets"):
            verify_reweighting(root)

    def test_manifest_asset_list_or_hash_tampering_rejected(self):
        for mode in ("list", "hash"):
            root = self.copied("manifest_" + mode)
            manifest = json.loads((root / MANIFEST_NAME).read_text("utf-8"))
            if mode == "list":
                manifest["source_file_sha256"].pop("index.joblib")
            else:
                manifest["derived_file_sha256"]["index.joblib"] = "a" * 64
            write_json(root / MANIFEST_NAME, manifest)
            metadata = json.loads((root / "metadata.json").read_text("utf-8"))
            metadata["training_reweighting"]["manifest_sha256"] = file_hash(root / MANIFEST_NAME)
            write_json(root / "metadata.json", metadata)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                verify_reweighting(root)


if __name__ == "__main__":
    unittest.main()
