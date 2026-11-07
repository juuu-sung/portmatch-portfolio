"""Synthetic two-generation cache proofs; no competition cache or model runs."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from portmatch import name_feature_cache as parent_transform
from portmatch.ch_variant_cache import (
    check_transform_metadata, transform_cache, transform_features, transform_spec,
    verify_derivation,
)
from portmatch.ch_variant_features import FEATURE_NAMES
from portmatch.data import read_catalog, read_queries
from portmatch.ranker_commands import feature_code_hash
from portmatch.sampling import SamplingConfig
from portmatch.training_data import iter_shards, prepare_training_data


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(root):
    return {str(path.relative_to(root)): file_hash(path)
            for path in root.rglob("*") if path.is_file()}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class ChVariantCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        catalog = pd.DataFrame([dict(vessel_id=f"V{i:02d}", vessel_name=f"CHORUS STAR {i}",
                                    gross_tonnage=1000 + i // 2, flag_name="Korea", vessel_type="Cargo",
                                    size=10., unit="m", deadweight_tonnage=1100.) for i in range(12)])
        queries = pd.DataFrame([
            dict(sample_id=f"S{i:02d}_{arrival}", target_vessel_id=f"V{i:02d}",
                 vessel_name=("코러스 스타 " if i % 2 else "CHORUS STAR ") + str(i), call_sign=f"C{i:02d}",
                 gross_tonnage=1000 + i // 2, international_gross_tonnage=1000 + i // 2,
                 flag_name="한국", vessel_type="화물선", arrival_year=2020 + arrival)
            for i in range(12) for arrival in range(2)])
        catalog.to_csv(cls.catalog_path, index=False, encoding="utf-8-sig")
        queries.to_csv(cls.train_path, index=False, encoding="utf-8-sig")
        cls.catalog, cls.queries = read_catalog(cls.catalog_path), read_queries(cls.train_path, labeled=True)
        cls.original, cls.parent, cls.derived = cls.root / "original", cls.root / "parent", cls.root / "derived"
        cls.original_metadata = prepare_training_data(
            cls.queries, cls.catalog, cls.original, outer_folds=3, inner_folds=2, seed=42,
            shard_queries=2, max_train_queries=4, max_valid_queries=4,
            sampling_config=SamplingConfig(hard_negatives=2, random_negatives=1, seed=42),
            provenance={"feature_code_sha256": feature_code_hash(), "train_sha256": file_hash(cls.train_path),
                        "catalog_sha256": file_hash(cls.catalog_path)})
        cls.parent_metadata = parent_transform.transform_cache(
            cls.original, cls.parent, cls.train_path, cls.catalog_path)
        cls.before_original, cls.before_parent = tree_hash(cls.original), tree_hash(cls.parent)
        cls.metadata = transform_cache(cls.parent, cls.derived, cls.train_path, cls.catalog_path)
        cls.parent_hash = file_hash(cls.parent / "metadata.json")

    def test_preserves_87_columns_all_arrays_both_sources_and_recursive_proof(self):
        self.assertEqual(len(self.parent_metadata["feature_names"]), 87)
        self.assertEqual(len(self.metadata["feature_names"]), 87 + len(FEATURE_NAMES))
        for kind in ("train", "valid"):
            for old, new in zip(iter_shards(self.parent, kind), iter_shards(self.derived, kind)):
                for key in old:
                    actual = new[key][:, :87] if key == "X" else new[key]
                    self.assertEqual(old[key].dtype, actual.dtype)
                    self.assertEqual(np.ascontiguousarray(old[key]).tobytes(),
                                     np.ascontiguousarray(actual).tobytes())
        self.assertEqual(self.before_original, tree_hash(self.original))
        self.assertEqual(self.before_parent, tree_hash(self.parent))
        proof = verify_derivation(self.derived, self.parent_hash, self.parent_metadata)
        self.assertTrue(proof["verified"])
        self.assertEqual(proof["parent_derivation"]["transform"], "token_phonetic_v1")
        self.assertEqual(proof["parent_derivation"]["parent_metadata_sha256"],
                         file_hash(self.original / "metadata.json"))
        self.assertEqual(transform_spec()["parent_transform"], parent_transform.transform_spec())

    def test_inference_matches_94_column_cache_without_parent_cache_access(self):
        pipeline = joblib.load(self.original / "index.joblib")
        queries = self.queries.set_index("sample_id")
        for shard in iter_shards(self.derived, "valid"):
            offset = 0
            for sample_id, group in zip(shard["sample_id"], shard["group"]):
                row = queries.loc[sample_id].to_dict()
                candidates = pipeline.retrieve(row)
                base = pipeline.features(row, candidates)
                with patch("portmatch.training_data.read_metadata", side_effect=AssertionError("cache access")):
                    expanded = transform_features(pipeline, row, candidates, base)
                np.testing.assert_array_equal(expanded.to_numpy(), shard["X"][offset:offset + group])
                altered = {**row, "target_vessel_id": "FAKE", "flag_name": "FAKE", "call_sign": "FAKE",
                           "gross_tonnage": 987654321, "sample_id": "FAKE"}
                pd.testing.assert_frame_equal(
                    expanded, transform_features(pipeline, altered, candidates, base), check_exact=True)
                backwards = transform_features(pipeline, row, candidates.iloc[::-1], base.iloc[::-1])
                pd.testing.assert_frame_equal(expanded.iloc[::-1], backwards, check_exact=True)
                offset += group

    def test_inference_reuses_computer_and_rejects_transformed_or_misaligned_input(self):
        pipeline = joblib.load(self.original / "index.joblib")
        row = self.queries.iloc[0].to_dict()
        candidates = pipeline.retrieve(row)
        base = pipeline.features(row, candidates)
        expanded = transform_features(pipeline, row, candidates, base)
        computer = pipeline._token_phonetic_ch_computer_v1
        with patch("portmatch.ch_variant_cache.ChVariantFeatureComputer", side_effect=AssertionError("rebuilt")):
            transform_features(pipeline, row, candidates, base)
        self.assertIs(pipeline._token_phonetic_ch_computer_v1, computer)
        with self.assertRaisesRegex(ValueError, "original 70"):
            transform_features(pipeline, row, candidates, expanded)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            transform_features(pipeline, row, candidates, base.iloc[::-1])

    def test_schema_binds_parent_and_new_code_and_feature_order(self):
        check_transform_metadata(self.metadata)
        broken_parent_spec = copy.deepcopy(transform_spec())
        broken_parent_spec["parent_transform"]["code_sha256"] = "changed"
        variants = [self.original_metadata, self.parent_metadata,
                    {**self.metadata, "feature_transform": broken_parent_spec},
                    {**self.metadata, "feature_transform": {**transform_spec(), "code_sha256": "changed"}},
                    {**self.metadata, "feature_names": self.metadata["feature_names"][::-1]},
                    {key: value for key, value in self.metadata.items() if key != "derivation"}]
        for metadata in variants:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                check_transform_metadata(metadata)

    def test_only_immediate_v1_parent_and_matching_inputs_allowed(self):
        for source in (self.original, self.derived):
            destination = self.root / (source.name + "_invalid")
            with self.assertRaisesRegex(ValueError, "immediate parent"):
                transform_cache(source, destination, self.train_path, self.catalog_path)
            self.assertFalse(destination.exists())
        for number, invalid_spec in enumerate((None, [], "token_phonetic_v1")):
            source = self.root / f"invalid_spec_{number}"
            source.mkdir()
            write_json(source / "metadata.json", {**self.parent_metadata, "feature_transform": invalid_spec})
            destination = self.root / f"invalid_spec_output_{number}"
            with self.assertRaisesRegex(ValueError, "immediate parent"):
                transform_cache(source, destination, self.train_path, self.catalog_path)
            self.assertFalse(destination.exists())
        changed = self.root / "changed_train.csv"
        changed.write_bytes(self.train_path.read_bytes() + b"\n")
        destination = self.root / "bad_input"
        with self.assertRaisesRegex(ValueError, "train_sha256"):
            transform_cache(self.parent, destination, changed, self.catalog_path)
        self.assertFalse(destination.exists())

    def test_empty_output_allowed_existing_or_nested_output_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        metadata = transform_cache(self.parent, empty, self.train_path, self.catalog_path)
        self.assertEqual(metadata["feature_names"], self.metadata["feature_names"])
        with self.assertRaises(FileExistsError):
            transform_cache(self.parent, empty, self.train_path, self.catalog_path)
        for output in (self.parent, self.parent / "nested", self.root):
            with self.assertRaisesRegex(ValueError, "separate"):
                transform_cache(self.parent, output, self.train_path, self.catalog_path)
        self.assertFalse((self.parent / "nested").exists())

    def test_wrong_baseline_parent_and_changed_child_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected baseline parent"):
            verify_derivation(self.derived, "wrong", self.parent_metadata)
        with self.assertRaisesRegex(ValueError, "differs from baseline model"):
            verify_derivation(self.derived, self.parent_hash, self.original_metadata)
        changed = self.root / "changed_metadata"
        shutil.copytree(self.derived, changed)
        metadata = json.loads((changed / "metadata.json").read_text("utf-8"))
        metadata["seed"] += 1
        write_json(changed / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "metadata beyond"):
            verify_derivation(changed, self.parent_hash, self.parent_metadata)

    def test_child_assets_and_87_column_prefix_proofs_cannot_be_silently_changed(self):
        changed = self.root / "changed_asset"
        shutil.copytree(self.derived, changed)
        relative = self.metadata["valid_shards"][0]
        asset = changed / relative
        with np.load(asset, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        # The new experiment must preserve even the parent's appended 17 columns.
        arrays["X"][0, 86] += .25
        np.savez_compressed(asset, **arrays)
        with self.assertRaisesRegex(ValueError, "asset hash mismatch"):
            verify_derivation(changed, self.parent_hash, self.parent_metadata)
        manifest = json.loads((changed / "derivation.json").read_text("utf-8"))
        manifest["derived_file_sha256"][relative] = file_hash(asset)
        write_json(changed / "derivation.json", manifest)
        metadata = json.loads((changed / "metadata.json").read_text("utf-8"))
        metadata["derivation"]["manifest_sha256"] = file_hash(changed / "derivation.json")
        write_json(changed / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "invariant arrays"):
            verify_derivation(changed, self.parent_hash, self.parent_metadata)

    def test_recursive_verification_detects_original_asset_tampering(self):
        asset = self.original / self.original_metadata["valid_shards"][0]
        original = asset.read_bytes()
        try:
            asset.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "asset hash mismatch"):
                verify_derivation(self.derived, self.parent_hash, self.parent_metadata)
            output = self.root / "invalid_original"
            with self.assertRaisesRegex(ValueError, "asset hash mismatch"):
                transform_cache(self.parent, output, self.train_path, self.catalog_path)
            self.assertFalse(output.exists())
        finally:
            asset.write_bytes(original)

    def test_parent_proof_and_manifest_path_cannot_be_rebound(self):
        changed = self.root / "changed_proof"
        shutil.copytree(self.derived, changed)
        manifest = json.loads((changed / "derivation.json").read_text("utf-8"))
        manifest["parent_derivation"]["verified_assets"] += 1
        write_json(changed / "derivation.json", manifest)
        metadata = json.loads((changed / "metadata.json").read_text("utf-8"))
        metadata["derivation"]["manifest_sha256"] = file_hash(changed / "derivation.json")
        write_json(changed / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "verified parent derivation"):
            verify_derivation(changed, self.parent_hash, self.parent_metadata)
        metadata["derivation"]["manifest_path"] = "../derivation.json"
        write_json(changed / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "escapes"):
            verify_derivation(changed, self.parent_hash, self.parent_metadata)


if __name__ == "__main__":
    unittest.main()
