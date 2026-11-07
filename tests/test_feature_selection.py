import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from portmatch.feature_selection import FeatureSelection
from portmatch.ranker_commands import RankedMatcher, add_training_parsers, feature_code_hash, run_train
from portmatch.training_data import load_training_arrays, prepare_training_data


class FeatureSelectionTests(unittest.TestCase):
    names = ["name_ratio", "retrieval_tonnage_rank", "flag_probability", "flag_lift_log"]

    def test_exact_and_pattern_selection_keep_source_order_and_nan(self):
        selection = FeatureSelection.resolve(self.names, include=["flag_*", "name_ratio"])
        self.assertEqual(selection.selected_features, ("name_ratio", "flag_probability", "flag_lift_log"))
        matrix = np.array([[.9, 2., np.nan, .1], [.5, 1., .8, .7]], dtype=np.float32)
        before = matrix.copy()
        np.testing.assert_array_equal(selection.apply(matrix, self.names), matrix[:, [0, 2, 3]])
        np.testing.assert_array_equal(matrix, before)
        self.assertEqual(FeatureSelection.from_dict(selection.to_dict()), selection)
        excluded = FeatureSelection.resolve(self.names, exclude=["retrieval_tonnage_rank"])
        self.assertEqual(excluded.selected_features, selection.selected_features)
        self.assertEqual(FeatureSelection.from_dict(excluded.to_dict()), excluded)

    def test_invalid_selection_is_rejected(self):
        for kwargs in ({"exclude": ["typo"]}, {"exclude": ["*"]}, {"include": []},
                       {"include": ["name_ratio"], "exclude": ["flag_*"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FeatureSelection.resolve(self.names, **kwargs)
        for names in ([], ["name_ratio", "name_ratio"], ["unknown"], list(reversed(self.names))):
            with self.subTest(names=names), self.assertRaises(ValueError):
                FeatureSelection.from_names(self.names, names)
        with self.assertRaises(ValueError):
            FeatureSelection.from_names(self.names).apply(np.ones((1, 4)), list(reversed(self.names)))

    def test_tampered_manifest_is_rejected(self):
        state = FeatureSelection.resolve(self.names, exclude=["flag_*"]).to_dict()
        state["selected_features"] = ["name_ratio"]
        with self.assertRaises(ValueError):
            FeatureSelection.from_dict(state)

    def test_parser_selection_and_comparison(self):
        parser = argparse.ArgumentParser()
        add_training_parsers(parser.add_subparsers(dest="command", required=True))
        args = parser.parse_args(["train-ranker", "--data", "cache", "--output", "model",
                                  "--exclude-features", "retrieval_tonnage_rank", "flag_*"])
        self.assertEqual(args.exclude_features, ["retrieval_tonnage_rank", "flag_*"])
        args = parser.parse_args(["compare-rankers", "--baseline", "b0", "b1",
                                  "--experiment", "e0", "e1", "--output", "report"])
        self.assertEqual(args.baseline, ["b0", "b1"])


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class CachedSelectionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.cache = cls.root / "cache"
        cls.model = cls.root / "model"
        catalog = pd.DataFrame([dict(vessel_id=f"V{i:02d}", vessel_name=f"SHIP {i}",
                                    gross_tonnage=1000 + i, flag_name="Korea", vessel_type="Cargo")
                                for i in range(12)])
        cls.queries = pd.DataFrame([
            dict(sample_id=f"S{i:02d}_{arrival}", target_vessel_id=f"V{i:02d}",
                 vessel_name=("선박 " if i % 2 else "SHIP ") + str(i), call_sign=f"C{i:02d}",
                 gross_tonnage=1000 + i, international_gross_tonnage=1000 + i,
                 flag_name="한국", vessel_type="화물선", arrival_year=2020 + arrival)
            for i in range(12) for arrival in range(2)])
        cls.metadata = prepare_training_data(
            cls.queries, catalog, cls.cache, outer_folds=3, inner_folds=2,
            seed=42, shard_queries=2, max_train_queries=4, max_valid_queries=4,
            provenance={"feature_code_sha256": feature_code_hash()})
        cls.before = cls.cache_hashes()
        cls.args = SimpleNamespace(data=str(cls.cache), output=str(cls.model), rounds=4,
                                   learning_rate=.05, num_leaves=4, min_child_samples=1,
                                   reg_lambda=1., threads=1, seed=42, hangul_weight=.7,
                                   exclude_features=["retrieval_tonnage_rank"], include_features=None)
        run_train(cls.args)

    @classmethod
    def cache_hashes(cls):
        return {str(path.relative_to(cls.cache)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in cls.cache.rglob("*") if path.is_file()}

    def test_cache_unchanged_and_only_selected_columns_loaded(self):
        full = load_training_arrays(self.cache)
        names = [name for name in full[4] if name != "retrieval_tonnage_rank"]
        selected = load_training_arrays(self.cache, feature_names=names)
        indices = [full[4].index(name) for name in names]
        np.testing.assert_array_equal(selected[0], full[0][:, indices])
        for old, new in zip(full[1:4], selected[1:4]):
            np.testing.assert_array_equal(old, new)
        self.assertEqual(selected[4], names)
        self.assertEqual(self.before, self.cache_hashes())
        summary = json.loads((self.model / "training_summary.json").read_text("utf-8"))
        state = json.loads((self.model / "ranker.json").read_text("utf-8"))
        self.assertEqual(len(summary["dataset"]["feature_names"]), 70)
        self.assertEqual(len(state["feature_names"]), 69)
        self.assertEqual(summary["trained_feature_names"], names)
        self.assertEqual(summary["feature_selection"]["excluded_features"], ["retrieval_tonnage_rank"])

    def test_restored_inference_matches_cached_validation_and_ignores_labels(self):
        matcher = RankedMatcher(self.model)
        records = pd.read_csv(self.model / "validation_predictions.csv")
        queries = self.queries.set_index("sample_id")
        for record in records.to_dict("records"):
            row = queries.loc[record["sample_id"]].to_dict()
            row["sample_id"] = record["sample_id"]
            ranked = matcher.rank(row)
            self.assertEqual(ranked.vessel_id.iloc[0], record["predicted_vessel_id"])
            self.assertEqual(len(ranked), record["candidate_count"])
            self.assertAlmostEqual(float(ranked.score.iloc[0]), record["top_score"], places=12)
            row["target_vessel_id"] = "UNRELATED_LABEL"
            pd.testing.assert_frame_equal(matcher.rank(row), ranked)

    def test_invalid_feature_request_fails_before_creating_output(self):
        output = self.root / "invalid"
        args = SimpleNamespace(**{**vars(self.args), "output": str(output), "exclude_features": ["typo"]})
        with self.assertRaisesRegex(ValueError, "matched no cached columns"):
            run_train(args)
        self.assertFalse(output.exists())
        with self.assertRaises(FileExistsError):
            run_train(self.args)

    def test_legacy_bundle_without_selection_still_loads(self):
        from portmatch.ranking import RankerConfig, VesselRanker
        import shutil

        output = self.root / "legacy"
        X, y, group, language, names = load_training_arrays(self.cache)
        ranker = VesselRanker(RankerConfig(n_estimators=2, num_leaves=4, min_child_samples=1, n_jobs=1))
        ranker.fit(X, y, group, language, names).save(output)
        shutil.copy2(self.cache / "index.joblib", output / "index.joblib")
        (output / "training_summary.json").write_text(
            json.dumps({"complete": True, "dataset": self.metadata}), encoding="utf-8")
        matcher = RankedMatcher(output)
        self.assertEqual(matcher.selection.source_feature_names, matcher.selection.selected_features)
        self.assertIn(matcher.predict(self.queries.iloc[0].to_dict()), matcher.catalog_ids)


if __name__ == "__main__":
    unittest.main()
