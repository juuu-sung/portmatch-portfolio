import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import joblib
import numpy as np
import pandas as pd

from portmatch.data import read_catalog, read_queries
from portmatch.memory import TrainingMemory
from portmatch.ranker_commands import feature_code_hash
from portmatch.retrieval import CandidateRetriever, RetrievalConfig
from portmatch.sampling import SamplingConfig
from portmatch.tonnage_rank import (
    SOURCE_FEATURE, TARGET_FEATURE, check_transform_metadata, numeric_ranks,
    transform_cache, transform_features, transform_spec, verify_derivation,
)
from portmatch.training_data import iter_shards, prepare_training_data


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retriever(tonnages, ids=None, **config):
    catalog = pd.DataFrame({"vessel_id": ids or [f"V{i}" for i in range(len(tonnages))],
                            "vessel_name": [f"SHIP {i}" for i in range(len(tonnages))],
                            "gross_tonnage": tonnages})
    memory = TrainingMemory().fit(pd.DataFrame(columns=["vessel_name", "call_sign", "target_vessel_id"]), catalog)
    return CandidateRetriever(catalog, memory, RetrievalConfig(**config))


class TonnageRankTests(unittest.TestCase):
    def test_equal_and_symmetric_gaps_minrank_before_sampling(self):
        index = retriever([1000, 1000, 990, 1010, 1020], ["Z", "A", "D", "B", "C"])
        row = {"international_gross_tonnage": 1000}
        old, new = numeric_ranks(index, row, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(old, [2, 1, 4, 3, 5])
        np.testing.assert_array_equal(new, [1, 1, 3, 3, 5])
        # A sampled group omitting all closer candidates must not restart at 1.
        np.testing.assert_array_equal(numeric_ranks(index, row, [4, 2])[1], [5, 3])
        permuted = retriever([1010, 1000, 1020, 990, 1000], ["B", "A", "C", "D", "Z"])
        np.testing.assert_array_equal(numeric_ranks(permuted, row, [4, 1, 3, 0, 2])[1], new)

    def test_anchor_gross_fallback_only_when_window_exists(self):
        index = retriever([990, 1000, 1010, 2000], numeric_fallback_k=3)
        fallback = {"international_gross_tonnage": 100000, "gross_tonnage": 1000}
        np.testing.assert_array_equal(numeric_ranks(index, fallback, [0, 1, 2, 3])[1], [2, 1, 2, 0])
        # Neither window exists: use original IGT anchor, not gross.
        missing_windows = {"international_gross_tonnage": 100000, "gross_tonnage": 10}
        np.testing.assert_array_equal(numeric_ranks(index, missing_windows, [0, 1, 2, 3])[1], [0, 3, 2, 1])

    def test_fallback_tie_boundary_and_missing_route(self):
        index = retriever([900, 900, 900, np.nan, 0], ["C", "A", "B", "D", "E"],
                          numeric_fallback_k=2, gt_relative_window=0)
        old, new = numeric_ranks(index, {"gross_tonnage": 1000}, np.arange(5))
        np.testing.assert_array_equal(old, [0, 1, 2, 0, 0])
        np.testing.assert_array_equal(new, [0, 1, 1, 0, 0])
        for query in ({}, {"gross_tonnage": -1}, {"international_gross_tonnage": np.inf}):
            np.testing.assert_array_equal(numeric_ranks(index, query, np.arange(5))[1], np.zeros(5))

    def test_inference_transform_preserves_shared_columns_and_rejects_stale_rank(self):
        index = retriever([1000, 1000, 1010])
        row = {"gross_tonnage": 1000, "target_vessel_id": "IGNORE"}
        candidates = index.retrieve(row)
        features = pd.DataFrame({"shared": [np.nan, .5, 1.], SOURCE_FEATURE: candidates.tonnage_rank,
                                 "retrieval_tonnage_hit": candidates.tonnage_hit}).astype(np.float32)
        result = transform_features(SimpleNamespace(retriever=index), row, candidates, features)
        np.testing.assert_array_equal(result[TARGET_FEATURE], [1, 1, 3])
        np.testing.assert_array_equal(result.shared, features.shared)
        self.assertIn(SOURCE_FEATURE, features)
        self.assertEqual(result.dtypes.tolist(), features.dtypes.tolist())
        with self.assertRaisesRegex(ValueError, "misaligned"):
            transform_features(SimpleNamespace(retriever=index), row, candidates, features.iloc[::-1])
        features.loc[0, SOURCE_FEATURE] = 10
        with self.assertRaisesRegex(ValueError, "Original tonnage ranks"):
            transform_features(SimpleNamespace(retriever=index), row, candidates, features)

    def test_transform_schema_fails_closed(self):
        check_transform_metadata({"feature_names": [SOURCE_FEATURE]})
        valid = {"feature_names": [TARGET_FEATURE], "feature_transform": transform_spec(), "derivation": {}}
        check_transform_metadata(valid)
        for metadata in ({"feature_names": [TARGET_FEATURE]},
                         {"feature_names": [SOURCE_FEATURE], "derivation": {}},
                         {**valid, "feature_names": [SOURCE_FEATURE]},
                         {**valid, "feature_transform": {**transform_spec(), "code_sha256": "wrong"}}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                check_transform_metadata(metadata)


class CacheTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        catalog = pd.DataFrame([dict(vessel_id=f"V{i:02d}", vessel_name=f"SHIP {i}",
                                    gross_tonnage=1000 + i // 2, flag_name="Korea", vessel_type="Cargo",
                                    size=10, unit="m", deadweight_tonnage=1100) for i in range(12)])
        queries = pd.DataFrame([
            dict(sample_id=f"S{i:02d}_{arrival}", target_vessel_id=f"V{i:02d}",
                 vessel_name=("선박 " if i % 2 else "SHIP ") + str(i), call_sign=f"C{i:02d}",
                 gross_tonnage=1000 + i // 2, international_gross_tonnage=1000 + i // 2,
                 flag_name="한국", vessel_type="화물선", arrival_year=2020 + arrival)
            for i in range(12) for arrival in range(2)])
        catalog.to_csv(cls.catalog_path, index=False)
        queries.to_csv(cls.train_path, index=False)
        cls.catalog, cls.queries = read_catalog(cls.catalog_path), read_queries(cls.train_path, labeled=True)
        cls.source, cls.derived = cls.root / "source", cls.root / "derived"
        cls.parent = prepare_training_data(
            cls.queries, cls.catalog, cls.source, outer_folds=3, inner_folds=2, seed=42,
            shard_queries=2, max_train_queries=4, max_valid_queries=4,
            sampling_config=SamplingConfig(hard_negatives=2, random_negatives=1, seed=42),
            provenance={"feature_code_sha256": feature_code_hash(), "train_sha256": file_hash(cls.train_path),
                        "catalog_sha256": file_hash(cls.catalog_path)})
        cls.before = {str(path.relative_to(cls.source)): file_hash(path) for path in cls.source.rglob("*") if path.is_file()}
        cls.metadata = transform_cache(cls.source, cls.derived, cls.train_path, cls.catalog_path)

    def test_all_shared_arrays_identical_and_source_untouched(self):
        column = self.parent["feature_names"].index(SOURCE_FEATURE)
        changed = 0
        for kind in ("train", "valid"):
            for old, new in zip(iter_shards(self.source, kind), iter_shards(self.derived, kind)):
                for key in old:
                    if key == "X":
                        np.testing.assert_array_equal(np.delete(old[key], column, 1), np.delete(new[key], column, 1))
                        changed += np.count_nonzero(old[key][:, column] != new[key][:, column])
                    else:
                        np.testing.assert_array_equal(old[key], new[key])
        self.assertGreater(changed, 0)
        after = {str(path.relative_to(self.source)): file_hash(path) for path in self.source.rglob("*") if path.is_file()}
        self.assertEqual(self.before, after)
        self.assertEqual(len(self.metadata["feature_names"]), 70)
        self.assertIn(TARGET_FEATURE, self.metadata["feature_names"])
        proof = verify_derivation(self.derived, file_hash(self.source / "metadata.json"), self.parent)
        self.assertTrue(proof["verified"])

    def test_inference_transform_matches_every_cached_valid_feature(self):
        pipeline = joblib.load(self.source / "index.joblib")
        queries = self.queries.set_index("sample_id")
        for shard in iter_shards(self.derived, "valid"):
            offset = 0
            for sample_id, group in zip(shard["sample_id"], shard["group"]):
                row = queries.loc[sample_id].to_dict()
                candidates = pipeline.retrieve(row)
                features = transform_features(pipeline, row, candidates, pipeline.features(row, candidates))
                np.testing.assert_array_equal(features.to_numpy(), shard["X"][offset:offset + group])
                offset += group

    def test_input_mismatch_output_reuse_and_double_transform_rejected(self):
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

    def test_derivation_rejects_wrong_parent_and_modified_assets(self):
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

    def test_invariants_are_recomputed_even_if_asset_hashes_are_updated(self):
        changed = self.root / "changed_shared_feature"
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


if __name__ == "__main__":
    unittest.main()
