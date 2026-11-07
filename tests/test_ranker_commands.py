import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from portmatch.ranker_commands import evaluate_cache, score_order


class RankerCommandTests(unittest.TestCase):
    def test_ties_follow_vessel_identity_under_candidate_permutation(self):
        ids = np.array(["C", "A", "B"])
        candidates = np.array([0, 1, 2])
        scores = np.array([.5, .5, .9])
        order = score_order(scores, candidates, ids)
        self.assertEqual(ids[candidates[order]].tolist(), ["B", "A", "C"])
        permutation = np.array([2, 0, 1])
        order2 = score_order(scores[permutation], candidates[permutation], ids)
        self.assertEqual(ids[candidates[permutation][order2]].tolist(), ["B", "A", "C"])
        with self.assertRaises(ValueError):
            score_order([np.nan], [0], ids)

    def test_cache_evaluation_keeps_retrieval_failure_in_denominator(self):
        class DummyRanker:
            def predict(self, X, names):
                self.assert_names = names
                return X[:, 0]

        # First query's target is absent. Second is present but tied for score;
        # vessel-ID tie breaking places its true candidate second.
        shard = {
            "X": np.array([[1.], [0.], [.5], [.5]], dtype=np.float32),
            "group": np.array([2, 2]), "catalog_index": np.array([0, 1, 0, 1]),
            "target_catalog_index": np.array([2, 0]),
            "sample_id": np.array(["K", "E"]), "has_hangul": np.array([True, False]),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps({"feature_names": ["score"]}), encoding="utf-8")
            np.save(root / "catalog_ids.npy", np.array(["B", "A", "C"]))
            with patch("portmatch.training_data.iter_shards", return_value=iter([shard])):
                predictions, metrics = evaluate_cache(DummyRanker(), root)
        self.assertEqual(predictions.target_rank.tolist(), [0, 2])
        self.assertEqual(metrics["hangul"]["candidate_recall"], 0)
        self.assertEqual(metrics["non_hangul"]["recall_at_5"], 1)
        self.assertEqual(metrics["weighted_top1"], 0)


if __name__ == "__main__":
    unittest.main()
