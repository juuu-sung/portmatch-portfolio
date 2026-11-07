"""Synthetic selection, fixed-model scope and training-asset leakage guards."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.data import read_catalog, read_queries
from portmatch.gt_augmentation import GTMissingAugmentationConfig, augmentation_spec
from portmatch.index_scale_assets import (
    _validate_scope, assert_unchanged, learned_assets_fingerprint,
    prepare_index_scale, select_training_vessels,
)
from portmatch.memory import TrainingMemory
from portmatch.ranking import RankerConfig, VesselRanker
from portmatch.robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig, retrieval_spec
from portmatch.sampling import SamplingConfig
from portmatch.validation import make_folds


def fixture():
    spec = importlib.util.spec_from_file_location(
        "index_scale_fixture", Path(__file__).with_name("test_robust_training.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture()


class IndexScaleAssetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        train, catalog = fixture()
        self.train_path, self.catalog_path = self.root / "train.csv", self.root / "catalog.csv"
        train.to_csv(self.train_path, index=False, encoding="utf-8-sig")
        catalog.to_csv(self.catalog_path, index=False, encoding="utf-8-sig")
        self.train, self.catalog = read_queries(self.train_path, labeled=True), read_catalog(self.catalog_path)
        self.fold = make_folds(self.train, "vessel", 5, 20261022)
        self.fitting = self.train.loc[self.fold != 0].copy()
        self.heldout = self.train.loc[self.fold == 0].copy()
        self.matcher = SimpleNamespace(
            pipeline=RobustMatchingPipeline(self.fitting, self.catalog, RobustRetrievalConfig()),
            catalog_ids=self.catalog.vessel_id.to_numpy(),
            ranker=SimpleNamespace(config=RankerConfig()))
        self.summary = {"complete": True, "ranker": asdict(RankerConfig()), "dataset": {
            "complete": True, "outer_split": "vessel", "outer_folds": 5, "outer_fold": 0,
            "inner_split": "mixed", "inner_folds": 3, "seed": 20261022,
            "sampled": False, "max_train_queries": None, "max_valid_queries": None,
            "retrieval": asdict(RobustRetrievalConfig()),
            "retrieval_extension": retrieval_spec(RobustRetrievalConfig()),
            "training_augmentation": augmentation_spec(GTMissingAugmentationConfig()),
            "sampling": asdict(SamplingConfig()),
            "counts": {"outer_training_rows": len(self.fitting),
                "outer_validation_rows": len(self.heldout),
                "selected_training_queries": len(self.fitting)}}}
        self.model = self.root / "model"
        self.model.mkdir()
        for name in ("model.txt", "ranker.json", "index.joblib", "validation_predictions.csv"):
            (self.model / name).write_text("protected fixture", encoding="utf-8")
        self.save_summary()

    def save_summary(self):
        (self.model / "training_summary.json").write_text(json.dumps(self.summary), encoding="utf-8")

    def prepare(self, **kwargs):
        with patch("portmatch.index_scale_assets._load_model", return_value=(
                self.matcher, copy.deepcopy(self.summary), {"load_and_integrity_seconds": 0.001})), \
                patch.object(VesselRanker, "fit", side_effect=AssertionError("ranker fitted")):
            return prepare_index_scale(self.model, self.train_path, self.catalog_path, **kwargs)

    def test_selection_uses_whole_vessels_and_is_row_order_independent(self):
        small, vessels = select_training_vessels(self.fitting)
        shuffled, repeated = select_training_vessels(self.fitting.iloc[::-1])
        self.assertEqual(vessels, repeated)
        self.assertEqual(len(vessels), int(.8 * self.fitting.target_vessel_id.nunique()))
        self.assertEqual(set(small.sample_id), set(shuffled.sample_id))
        expected = self.fitting[self.fitting.target_vessel_id.isin(vessels)]
        pd.testing.assert_frame_equal(expected, small)
        self.assertTrue(all(len(small[small.target_vessel_id.eq(vessel)]) ==
                            len(self.fitting[self.fitting.target_vessel_id.eq(vessel)]) for vessel in vessels))
        different_scores = self.fitting.assign(fake_validation_accuracy=np.linspace(0, 1, len(self.fitting)))
        self.assertEqual(select_training_vessels(different_scores)[1], vessels)

    def test_invalid_fraction_seed_and_degenerate_selection_rejected(self):
        for fraction in (0, 1, -1, 1.1, float("nan"), float("inf"), True, "0.8"):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                select_training_vessels(self.fitting, fraction=fraction)
        for seed in (-1, True, "3"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                select_training_vessels(self.fitting, seed=seed)
        with self.assertRaisesRegex(ValueError, "at least one"):
            select_training_vessels(self.fitting.iloc[:1])

    def test_prepare_reuses_full_index_and_builds_small_assets_from_same_population(self):
        result = self.prepare(expected_fold=0)
        self.assertIs(result["matcher"], self.matcher)
        small_pipeline = result["small_pipeline"]
        selected = result["membership"].query("selected_small").sample_id
        source = self.train[self.train.sample_id.isin(selected)]
        self.assertEqual(small_pipeline.memory.training_rows, len(source))
        self.assertEqual(small_pipeline.memory.training_vessels, frozenset(source.target_vessel_id))
        expected = RobustMatchingPipeline(source, self.catalog, RobustRetrievalConfig())
        self.assertEqual(learned_assets_fingerprint(expected), learned_assets_fingerprint(small_pipeline))
        self.assertIs(small_pipeline.memory, small_pipeline.retriever.memory)
        self.assertIs(small_pipeline.categories, small_pipeline.feature_builder.category_model)
        self.assertFalse(set(source.sample_id) & set(result["heldout"].sample_id))
        self.assertFalse(set(source.target_vessel_id) & set(result["heldout"].target_vessel_id))
        self.assertEqual(len(result["membership"]), len(self.train))
        self.assertTrue(result["proof"]["integrity"]["full_learned_assets_recomputed_and_equal"])
        self.assertEqual(result["proof"]["coverage"]["fold"], 0)
        self.assertEqual(len(result["protected_files"]), 7)
        json.dumps(result["proof"], allow_nan=False)
        self.assertEqual(assert_unchanged(result["protected_files"]), {"verified": True, "files": 7})

    def test_full_row_sampled_reweighted_or_changed_profiles_fail_before_native_load(self):
        variants = []
        for key, value in (("outer_split", "row"), ("training_scope", None),
                           ("sampled", True), ("max_valid_queries", 1),
                           ("training_reweighting", None), ("inner_folds", 2)):
            changed = copy.deepcopy(self.summary)
            changed["dataset"][key] = value
            variants.append(changed)
        for field, value in (("n_estimators", 301), ("n_jobs", 1)):
            changed = copy.deepcopy(self.summary)
            changed["ranker"][field] = value
            variants.append(changed)
        changed = copy.deepcopy(self.summary)
        changed["dataset"]["training_augmentation"] = augmentation_spec(GTMissingAugmentationConfig(.75))
        variants.append(changed)
        changed = copy.deepcopy(self.summary)
        changed["dataset"]["retrieval"]["context_top_k"] = 20
        variants.append(changed)
        for changed in variants:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                _validate_scope(changed, 0)
        self.summary["dataset"]["training_scope"] = "full_public_train"
        self.save_summary()
        with patch("portmatch.index_scale_assets._load_model") as loader:
            with self.assertRaisesRegex(ValueError, "outer-fold"):
                prepare_index_scale(self.model, self.train_path, self.catalog_path)
            loader.assert_not_called()

    def test_expected_fold_and_unselected_training_population_fail(self):
        with self.assertRaisesRegex(ValueError, "expected_fold"):
            self.prepare(expected_fold=2)
        self.summary["dataset"]["counts"]["selected_training_queries"] -= 1
        self.save_summary()
        with self.assertRaisesRegex(ValueError, "complete outer training"):
            self.prepare()

    def test_hidden_heldout_membership_and_learned_value_corruption_rejected(self):
        original = self.matcher.pipeline.memory
        self.matcher.pipeline.memory = TrainingMemory().fit(self.train, self.catalog)
        with self.assertRaisesRegex(ValueError, "training partition"):
            self.prepare()
        self.matcher.pipeline.memory = original
        first = next(iter(original.names.values()))
        first[next(iter(first))] += 1
        with self.assertRaisesRegex(ValueError, "memory differs"):
            self.prepare()

    def test_category_corruption_rejected(self):
        self.matcher.pipeline.categories.tables["flag_name"]["total"] += 1
        with self.assertRaisesRegex(ValueError, "categories differs"):
            self.prepare()

    def test_fingerprint_tracks_order_dtype_catalog_and_training_state(self):
        pipeline = self.matcher.pipeline
        before = learned_assets_fingerprint(pipeline)
        pipeline._query_feature_cache = {"cache": "allowed"}
        self.assertEqual(before, learned_assets_fingerprint(pipeline))
        pipeline.memory.aliases = pipeline.memory.aliases.iloc[::-1].reset_index(drop=True)
        self.assertNotEqual(before, learned_assets_fingerprint(pipeline))
        pipeline.memory.aliases = pipeline.memory.aliases.iloc[::-1].reset_index(drop=True)
        self.assertEqual(before, learned_assets_fingerprint(pipeline))
        pipeline.memory.aliases["vessel_name"] = pipeline.memory.aliases.vessel_name.astype("string")
        self.assertNotEqual(before, learned_assets_fingerprint(pipeline))
        pipeline.memory.aliases["vessel_name"] = pipeline.memory.aliases.vessel_name.astype("object")
        self.assertEqual(before, learned_assets_fingerprint(pipeline))
        pipeline.catalog.loc[0, "vessel_name"] = "Changed"
        self.assertNotEqual(before, learned_assets_fingerprint(pipeline))

    def test_file_changes_or_deletion_rejected(self):
        result = self.prepare()
        (self.model / "model.txt").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Protected input/model"):
            assert_unchanged(result["protected_files"])
        (self.model / "model.txt").write_text("protected fixture", encoding="utf-8")
        (self.model / "ranker.json").unlink()
        with self.assertRaisesRegex(ValueError, "Protected input/model"):
            assert_unchanged(result["protected_files"])
        with self.assertRaises(ValueError):
            assert_unchanged({})


if __name__ == "__main__":
    unittest.main()
