"""Small cache integrity tests; no competition training or full cache generation."""
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

from portmatch.data import read_catalog, read_queries
from portmatch.name_features import FEATURE_NAMES
from portmatch.name_feature_cache import (
    _legacy_feature_names, check_transform_metadata, transform_cache,
    transform_features, transform_spec, verify_derivation,
)
from portmatch.ranker_commands import feature_code_hash
from portmatch.sampling import SamplingConfig
from portmatch.training_data import iter_shards, prepare_training_data


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NameFeatureCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        catalog = pd.DataFrame([dict(vessel_id=f"V{i:02d}", vessel_name=f"STAR {i}",
                                    gross_tonnage=1000 + i // 2, flag_name="Korea", vessel_type="Cargo",
                                    size=10., unit="m", deadweight_tonnage=1100.) for i in range(12)])
        queries = pd.DataFrame([
            dict(sample_id=f"S{i:02d}_{arrival}", target_vessel_id=f"V{i:02d}",
                 vessel_name=("스타 " if i % 2 else "STAR ") + str(i), call_sign=f"C{i:02d}",
                 gross_tonnage=1000 + i // 2, international_gross_tonnage=1000 + i // 2,
                 flag_name="한국", vessel_type="화물선", arrival_year=2020 + arrival)
            for i in range(12) for arrival in range(2)])
        catalog.to_csv(cls.catalog_path, index=False, encoding="utf-8-sig")
        queries.to_csv(cls.train_path, index=False, encoding="utf-8-sig")
        cls.catalog, cls.queries = read_catalog(cls.catalog_path), read_queries(cls.train_path, labeled=True)
        cls.source, cls.derived = cls.root / "source", cls.root / "derived"
        cls.parent = prepare_training_data(
            cls.queries, cls.catalog, cls.source, outer_folds=3, inner_folds=2, seed=42,
            shard_queries=2, max_train_queries=4, max_valid_queries=4,
            sampling_config=SamplingConfig(hard_negatives=2, random_negatives=1, seed=42),
            provenance={"feature_code_sha256": feature_code_hash(), "train_sha256": file_hash(cls.train_path),
                        "catalog_sha256": file_hash(cls.catalog_path)})
        cls.before = {str(path.relative_to(cls.source)): file_hash(path)
                      for path in cls.source.rglob("*") if path.is_file()}
        cls.metadata = transform_cache(cls.source, cls.derived, cls.train_path, cls.catalog_path)

    def test_all_original_arrays_and_source_bytes_are_preserved(self):
        for kind in ("train", "valid"):
            for old, new in zip(iter_shards(self.source, kind), iter_shards(self.derived, kind)):
                self.assertEqual(new["X"].shape[1], 70 + len(FEATURE_NAMES))
                for key in old:
                    actual = new[key][:, :70] if key == "X" else new[key]
                    self.assertEqual(old[key].dtype, actual.dtype)
                    self.assertEqual(np.ascontiguousarray(old[key]).tobytes(),
                                     np.ascontiguousarray(actual).tobytes())
        after = {str(path.relative_to(self.source)): file_hash(path)
                 for path in self.source.rglob("*") if path.is_file()}
        self.assertEqual(self.before, after)
        self.assertEqual(self.metadata["feature_names"], self.parent["feature_names"] + list(FEATURE_NAMES))
        proof = verify_derivation(self.derived, file_hash(self.source / "metadata.json"), self.parent)
        self.assertTrue(proof["verified"])

    def test_inference_matches_cache_and_ignores_labels_categories_and_other_rows(self):
        pipeline = joblib.load(self.source / "index.joblib")
        queries = self.queries.set_index("sample_id")
        for shard in iter_shards(self.derived, "valid"):
            offset = 0
            for sample_id, group in zip(shard["sample_id"], shard["group"]):
                row = queries.loc[sample_id].to_dict()
                candidates = pipeline.retrieve(row)
                base = pipeline.features(row, candidates)
                expanded = transform_features(pipeline, row, candidates, base)
                np.testing.assert_array_equal(expanded.to_numpy(), shard["X"][offset:offset + group])
                altered = {**row, "target_vessel_id": "FAKE", "flag_name": "FAKE", "call_sign": "FAKE",
                           "gross_tonnage": 987654321, "sample_id": "FAKE"}
                # Only the raw name is read; the unchanged base matrix is supplied.
                again = transform_features(pipeline, altered, candidates, base)
                pd.testing.assert_frame_equal(expanded, again, check_exact=True)
                backwards = transform_features(pipeline, row, candidates.iloc[::-1], base.iloc[::-1])
                pd.testing.assert_frame_equal(expanded.iloc[::-1], backwards, check_exact=True)
                offset += group

    def test_inference_reuses_catalog_preprocessing_and_checks_alignment(self):
        pipeline = joblib.load(self.source / "index.joblib")
        row = self.queries.iloc[0].to_dict()
        candidates = pipeline.retrieve(row)
        base = pipeline.features(row, candidates)
        transform_features(pipeline, row, candidates, base)
        computer = pipeline._token_phonetic_computer_v1
        with patch("portmatch.name_feature_cache.NameFeatureComputer", side_effect=AssertionError("rebuilt")):
            transform_features(pipeline, row, candidates, base)
        self.assertIs(computer, pipeline._token_phonetic_computer_v1)
        with self.assertRaisesRegex(ValueError, "misaligned"):
            transform_features(pipeline, row, candidates, base.iloc[::-1])
        with self.assertRaisesRegex(ValueError, "original 70"):
            transform_features(pipeline, row, candidates, base.iloc[:, ::-1])
        invalid = candidates.copy()
        invalid["catalog_index"] = invalid.catalog_index.astype(float) + .5
        with self.assertRaisesRegex(ValueError, "catalog indices"):
            transform_features(pipeline, row, invalid, base)

    def test_schema_fails_closed_on_missing_changed_or_reordered_transform(self):
        check_transform_metadata(self.parent)
        check_transform_metadata(self.metadata)
        self.assertEqual(tuple(self.parent["feature_names"]), _legacy_feature_names())
        variants = [
            {"feature_names": list(FEATURE_NAMES)},
            {**self.parent, "derivation": {}},
            {**self.metadata, "feature_transform": {**transform_spec(), "code_sha256": "wrong"}},
            {**self.metadata, "feature_names": self.metadata["feature_names"][::-1]},
            {**self.metadata, "feature_names": self.parent["feature_names"]},
        ]
        for metadata in variants:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                check_transform_metadata(metadata)

    def test_input_mismatch_output_reuse_double_transform_and_nested_output_rejected(self):
        with self.assertRaises(FileExistsError):
            transform_cache(self.source, self.derived, self.train_path, self.catalog_path)
        changed = self.root / "different_train.csv"
        changed.write_bytes(self.train_path.read_bytes() + b"\n")
        output = self.root / "bad_input"
        with self.assertRaisesRegex(ValueError, "train_sha256"):
            transform_cache(self.source, output, changed, self.catalog_path)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(ValueError, "original cache"):
            transform_cache(self.derived, self.root / "twice", self.train_path, self.catalog_path)
        with self.assertRaisesRegex(ValueError, "separate"):
            transform_cache(self.source, self.source / "nested", self.train_path, self.catalog_path)
        self.assertFalse((self.source / "nested").exists())

    def test_wrong_parent_modified_assets_and_changed_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected baseline parent"):
            verify_derivation(self.derived, "wrong", self.parent)
        changed = self.root / "tampered"
        shutil.copytree(self.derived, changed)
        asset = changed / self.metadata["valid_shards"][0]
        asset.write_bytes(asset.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "asset hash mismatch"):
            verify_derivation(changed, file_hash(self.source / "metadata.json"), self.parent)
        metadata = json.loads((changed / "metadata.json").read_text("utf-8"))
        metadata["seed"] += 1
        (changed / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata beyond"):
            verify_derivation(changed, file_hash(self.source / "metadata.json"), self.parent)

    def test_shared_array_proofs_recomputed_even_after_refreshing_asset_hashes(self):
        changed = self.root / "changed_shared"
        shutil.copytree(self.derived, changed)
        relative = self.metadata["valid_shards"][0]
        asset = changed / relative
        with np.load(asset, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["X"][0, 0] += .25
        np.savez_compressed(asset, **arrays)
        manifest = json.loads((changed / "derivation.json").read_text("utf-8"))
        manifest["derived_file_sha256"][relative] = file_hash(asset)
        (changed / "derivation.json").write_text(json.dumps(manifest), encoding="utf-8")
        metadata = json.loads((changed / "metadata.json").read_text("utf-8"))
        metadata["derivation"]["manifest_sha256"] = file_hash(changed / "derivation.json")
        (changed / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invariant arrays"):
            verify_derivation(changed, file_hash(self.source / "metadata.json"), self.parent)

    def test_metadata_path_traversal_rejected_before_asset_access(self):
        changed = self.root / "path_escape"
        shutil.copytree(self.derived, changed)
        metadata = json.loads((changed / "metadata.json").read_text("utf-8"))
        metadata["derivation"]["manifest_path"] = "../derivation.json"
        (changed / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "escapes"):
            verify_derivation(changed, file_hash(self.source / "metadata.json"), self.parent)


if __name__ == "__main__":
    unittest.main()
