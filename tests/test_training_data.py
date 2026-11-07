import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.sampling import SamplingConfig
from portmatch.training_data import prepare_training_data, iter_shards, load_training_arrays
from portmatch.validation import make_folds


class RecordingPipeline:
    """A tiny index exposing exactly which labels fit each queried pipeline."""
    fits = []
    queries = []
    omit_ids = set()

    def __init__(self, fitting, catalog, config):
        self.fit_ids = set(fitting.sample_id)
        self.fit_targets = set(fitting.target_vessel_id)
        self.catalog = catalog
        self.fits.append((self.fit_ids, self.fit_targets))

    def retrieve(self, row):
        assert "target_vessel_id" not in row
        self.queries.append((row["sample_id"], self.fit_ids, self.fit_targets))
        target = int(row["sample_id"].split("_")[0][1:])
        indices = np.arange(len(self.catalog))
        if row["sample_id"] in self.omit_ids:
            indices = indices[indices != target]
        return pd.DataFrame({"catalog_index": indices})

    def features(self, row, candidates):
        assert "target_vessel_id" not in row
        indices = candidates.catalog_index.to_numpy()
        return pd.DataFrame({"romanized_ratio": (indices % 3).astype(float) / 3,
                             "phonetic_ratio": (indices % 5).astype(float) / 5,
                             "international_relative_gap": indices / 100}, index=candidates.index).astype(np.float32)


def fixture():
    catalog = pd.DataFrame({"vessel_id": [f"V{i:02d}" for i in range(12)],
                            "vessel_name": [f"Ship {i}" for i in range(12)]})
    rows = []
    for vessel in range(12):
        for arrival in range(4):
            rows.append({"sample_id": f"S{vessel:02d}_{arrival}",
                         "target_vessel_id": f"V{vessel:02d}",
                         "vessel_name": ("선박" if vessel % 2 else "SHIP") + str(vessel),
                         "call_sign": f"CS{vessel:02d}", "gross_tonnage": 1000,
                         "international_gross_tonnage": 1000,
                         "flag_name": "KR", "vessel_type": "Cargo", "arrival_year": 2020 + arrival})
    return pd.DataFrame(rows), catalog


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.train, self.catalog = fixture()
        RecordingPipeline.fits, RecordingPipeline.queries = [], []
        RecordingPipeline.omit_ids = set()

    def prepare(self, output, **kwargs):
        with patch("portmatch.training_data.MatchingPipeline", RecordingPipeline):
            return prepare_training_data(self.train, self.catalog, output,
                outer_folds=3, inner_folds=2, seed=42, shard_queries=3,
                sampling_config=SamplingConfig(hard_negatives=2, random_negatives=1, seed=42), **kwargs)

    def test_nested_fits_exclude_outer_labels_and_own_query(self):
        with TemporaryDirectory() as directory:
            metadata = self.prepare(Path(directory) / "cache")
            partitions = pd.read_csv(Path(directory) / "cache/partitions.csv")
            valid = partitions[partitions.role == "valid"]
            valid_ids, valid_targets = set(valid.sample_id), set(valid.target_vessel_id)
            self.assertGreater(len(RecordingPipeline.fits), 2)
            for ids, targets in RecordingPipeline.fits:
                self.assertFalse(ids & valid_ids)
                self.assertFalse(targets & valid_targets)
            for sample_id, fitting_ids, _ in RecordingPipeline.queries:
                self.assertNotIn(sample_id, fitting_ids)
            training = partitions[partitions.role == "train"]
            vessel_calls = RecordingPipeline.queries[:len(training)]
            target_map = self.train.set_index("sample_id").target_vessel_id.to_dict()
            for sample_id, _, targets in vessel_calls:
                self.assertNotIn(target_map[sample_id], targets)
            row_calls = RecordingPipeline.queries[len(training):2 * len(training)]
            self.assertTrue(any(target_map[sample_id] in targets for sample_id, _, targets in row_calls))
            self.assertEqual(metadata["counts"]["training_attempts"], 2 * len(training))

    def test_training_sampled_validation_complete_and_arrays_align(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "cache"
            metadata = self.prepare(output)
            training = list(iter_shards(output, "train"))
            validation = list(iter_shards(output, "valid"))
            self.assertTrue(all(np.all(shard["group"] == 4) for shard in training))
            self.assertTrue(all(np.all(shard["group"] == len(self.catalog)) for shard in validation))
            self.assertTrue(all(len(shard["group"]) <= 3 for shard in training + validation))
            query_ids = np.concatenate([shard["query_id"] for shard in training]).tolist()
            self.assertEqual(len(query_ids), len(set(query_ids)))
            self.assertTrue(any(value.endswith("::vessel") for value in query_ids))
            self.assertTrue(any(value.endswith("::row") for value in query_ids))
            x, y, groups, language, names = load_training_arrays(output)
            self.assertEqual(x.shape, (int(groups.sum()), len(names)))
            self.assertEqual(len(y), len(x))
            self.assertEqual(len(language), len(groups))
            self.assertEqual(int(y.sum()), len(groups))
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(metadata["counts"]["train_pairs"], len(x))
            self.assertTrue((output / "index.joblib").exists())
            np.testing.assert_array_equal(np.load(output / "catalog_ids.npy", allow_pickle=False), self.catalog.vessel_id)

    def test_missing_positive_skipped_only_for_training(self):
        assignments = make_folds(self.train, "vessel", 3, 42)
        train_id = self.train[assignments != 0].sample_id.iloc[0]
        valid_id = self.train[assignments == 0].sample_id.iloc[0]
        RecordingPipeline.omit_ids = {train_id, valid_id}
        with TemporaryDirectory() as directory:
            output = Path(directory) / "cache"
            metadata = self.prepare(output)
            self.assertEqual(metadata["counts"]["train_missing_target"], 2)
            self.assertEqual(metadata["counts"]["valid_missing_target"], 1)
            for shard in iter_shards(output, "valid"):
                offsets = np.r_[0, np.cumsum(shard["group"])]
                for number, sample_id in enumerate(shard["sample_id"]):
                    if sample_id == valid_id:
                        labels = shard["y"][offsets[number]:offsets[number + 1]]
                        self.assertEqual(int(labels.sum()), 0)
                        self.assertEqual(len(labels), len(self.catalog) - 1)

    def test_smoke_limit_is_source_query_limit_per_view(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "cache"
            metadata = self.prepare(output, max_train_queries=4, max_valid_queries=3)
            self.assertEqual(metadata["counts"]["train_groups"], 8)
            self.assertEqual(metadata["counts"]["valid_groups"], 3)
            self.assertTrue(metadata["sampled"])
            with self.assertRaises(FileExistsError):
                self.prepare(output)

    def test_incomplete_or_misaligned_cache_rejected(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "cache"
            metadata = self.prepare(output, max_train_queries=2, max_valid_queries=2)
            metadata["counts"]["train_pairs"] += 1
            (output / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_training_arrays(output)
            metadata["complete"] = False
            (output / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                list(iter_shards(output))


if __name__ == "__main__":
    unittest.main()
