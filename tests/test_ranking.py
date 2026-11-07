import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from portmatch.ranking import RankerConfig, VesselRanker, query_sample_weights, ranking_metrics


class RankingMetricsTests(unittest.TestCase):
    def test_query_weight_mass_is_equal_within_language(self):
        group = np.array([2, 6, 3, 9])
        language = np.array([True, True, False, False])
        weights = query_sample_weights(group, language)
        masses = np.add.reduceat(weights, np.r_[0, np.cumsum(group)[:-1]])
        self.assertAlmostEqual(float(masses[0]), float(masses[1]), places=5)
        self.assertAlmostEqual(float(masses[2]), float(masses[3]), places=5)
        self.assertAlmostEqual(float(masses[:2].sum() / masses.sum()), .7, places=6)
        self.assertAlmostEqual(float(weights.mean()), 1., places=6)

    def test_single_language_weights_and_invalid_groups(self):
        weights = query_sample_weights(np.array([2, 8]), np.array([True, True]))
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2:].sum()), places=5)
        for group in ([2, 0], [2, 1.5], [2, np.nan]):
            with self.assertRaises(ValueError):
                query_sample_weights(np.array(group), np.array([True, False]))

    def test_retrieval_failure_is_counted_as_wrong_at_every_k(self):
        records = pd.DataFrame({"has_hangul": [True, True, False, False],
                                "candidate_found": [True, False, True, True],
                                "target_rank": [1, 0, 3, 1],
                                "candidate_count": [10, 10, 10, 10]})
        result = ranking_metrics(records)
        self.assertEqual(result["hangul"]["top1_accuracy"], .5)
        self.assertEqual(result["hangul"]["recall_at_200"], .5)
        self.assertEqual(result["non_hangul"]["recall_at_5"], 1.)
        self.assertEqual(result["weighted_top1"], .5)
        self.assertAlmostEqual(result["weighted_candidate_recall"], .65)

    def test_missing_language_does_not_create_a_weighted_score(self):
        records = pd.DataFrame({"has_hangul": [True], "candidate_found": [True],
                                "target_rank": [1], "candidate_count": [10]})
        self.assertIsNone(ranking_metrics(records)["weighted_top1"])
        records.loc[0, "candidate_found"] = False
        with self.assertRaises(ValueError):
            ranking_metrics(records)

    def test_ranker_config_rejects_invalid_parameters(self):
        for kwargs in ({"n_estimators": 0}, {"n_jobs": 0}, {"hangul_weight": 1.},
                       {"learning_rate": np.nan}, {"num_leaves": 1}):
            with self.assertRaises(ValueError):
                RankerConfig(**kwargs)


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class NativeRankerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.group = np.full(12, 4, dtype=np.int32)
        cls.languages = np.array([True, False] * 6)
        cls.X = np.tile(np.array([[1., .1], [.1, .8], [.2, .7], [.0, np.nan]], dtype=np.float32), (12, 1))
        cls.y = np.tile(np.array([1, 0, 0, 0], dtype=np.uint8), 12)
        cls.names = ["name_ratio", "international_relative_gap"]
        cls.model = VesselRanker(RankerConfig(n_estimators=8, min_child_samples=1,
                                             num_leaves=4, n_jobs=1)).fit(
            cls.X, cls.y, cls.group, cls.languages, cls.names)

    def test_tiny_fit_ranks_positive_and_preserves_query_order(self):
        scores = self.model.predict(self.X, self.names).reshape(-1, 4)
        self.assertTrue((scores.argmax(axis=1) == 0).all())
        permutation = np.array([3, 1, 0, 2])
        direct = self.model.predict(self.X[:4], self.names)
        reordered = self.model.predict(self.X[permutation], self.names)
        np.testing.assert_array_equal(reordered, direct[permutation])
        self.assertEqual(self.model.predict(self.X[:0], self.names).shape, (0,))

    def test_native_save_load_and_integrity_check(self):
        with TemporaryDirectory() as root:
            self.model.save(root)
            restored = VesselRanker.load(root)
            np.testing.assert_array_equal(self.model.predict(self.X, self.names),
                                          restored.predict(self.X, self.names))
            self.assertEqual(restored.feature_importance().shape, (2, 3))
            with self.assertRaises(FileExistsError):
                self.model.save(root)
            metadata = json.loads((Path(root) / "ranker.json").read_text("utf-8"))
            metadata["model_sha256"] = "wrong"
            (Path(root) / "ranker.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(ValueError):
                VesselRanker.load(root)

    def test_training_rejects_bad_labels_groups_and_identifiers(self):
        params = dict(X=self.X, y=self.y, group=self.group,
                      query_has_hangul=self.languages, feature_names=self.names)
        cases = [dict(group=np.array([2])), dict(y=np.zeros_like(self.y)),
                 dict(feature_names=["sample_id", "name_ratio"]),
                 dict(query_has_hangul=np.array([True])),
                 dict(X=np.full_like(self.X, np.inf))]
        for change in cases:
            with self.assertRaises(ValueError):
                VesselRanker().fit(**{**params, **change})

    def test_predict_rejects_schema_reordering(self):
        with self.assertRaises(ValueError):
            self.model.predict(self.X, list(reversed(self.names)))
        with self.assertRaises(ValueError):
            self.model.predict(self.X[:, :1], self.names)


if __name__ == "__main__":
    unittest.main()
