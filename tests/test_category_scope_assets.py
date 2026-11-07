"""Frozen full-search bindings and training-only category populations."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from portmatch.category_scope_assets import (
    EXPERIMENT, FACTOR_LEVELS, REFERENCE, RUNNERS, category_assets_fingerprint,
    prepare_category_scope, validate_category_wiring,
)
from portmatch.data import read_catalog, read_queries
from portmatch.gt_augmentation import GTMissingAugmentationConfig, augmentation_spec
from portmatch.features import CategoryCompatibility
from portmatch.index_scale_assets import learned_assets_fingerprint, select_training_vessels
from portmatch.ranking import RankerConfig, VesselRanker
from portmatch.robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig, retrieval_spec
from portmatch.sampling import SamplingConfig
from portmatch.validation import make_folds


class CategoryScopeAssetTests(unittest.TestCase):
    def setUp(self):
        specification = importlib.util.spec_from_file_location(
            "_category_scope_fixture", Path(__file__).with_name("test_robust_training.py"))
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        train, catalog = module.fixture()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train_path, self.catalog_path = self.root / "train.csv", self.root / "catalog.csv"
        train.to_csv(self.train_path, index=False, encoding="utf-8-sig")
        catalog.to_csv(self.catalog_path, index=False, encoding="utf-8-sig")
        self.train, self.catalog = read_queries(self.train_path, labeled=True), read_catalog(self.catalog_path)
        self.assignments = make_folds(self.train, "vessel", 5, 20261022)
        self.model = self.root / "model"
        self.model.mkdir()
        for name in ("model.txt", "ranker.json", "index.joblib", "validation_predictions.csv"):
            (self.model / name).write_text("protected fixture", encoding="utf-8")
        self.configure_fold(1)

    def configure_fold(self, fold):
        self.fitting = self.train.loc[self.assignments != fold].copy()
        heldout = self.train.loc[self.assignments == fold].copy()
        self.base = SimpleNamespace(
            pipeline=RobustMatchingPipeline(self.fitting, self.catalog, RobustRetrievalConfig()),
            catalog_ids=self.catalog.vessel_id.to_numpy(), ranker=SimpleNamespace(config=RankerConfig()),
            selection=object(), feature_transform={"fixture": "frozen"})
        self.summary = {"complete": True, "ranker": asdict(RankerConfig()), "dataset": {
            "complete": True, "outer_split": "vessel", "outer_folds": 5, "outer_fold": fold,
            "inner_split": "mixed", "inner_folds": 3, "seed": 20261022,
            "sampled": False, "max_train_queries": None, "max_valid_queries": None,
            "retrieval": asdict(RobustRetrievalConfig()),
            "retrieval_extension": retrieval_spec(RobustRetrievalConfig()),
            "training_augmentation": augmentation_spec(GTMissingAugmentationConfig()),
            "sampling": asdict(SamplingConfig()), "counts": {"outer_training_rows": len(self.fitting),
                "outer_validation_rows": len(heldout), "selected_training_queries": len(self.fitting)}}}
        (self.model / "training_summary.json").write_text(json.dumps(self.summary), encoding="utf-8")

    def prepare(self, **kwargs):
        with patch("portmatch.index_scale_assets._load_model", return_value=(
                self.base, copy.deepcopy(self.summary), {"load_and_integrity_seconds": .001})), \
                patch.object(VesselRanker, "fit", side_effect=AssertionError("diagnostic trained weights")):
            return prepare_category_scope(self.model, self.train_path, self.catalog_path, **kwargs)

    def test_new_folds_use_exact_selected_training_categories_without_validation(self):
        for fold in (1, 3, 4):
            with self.subTest(fold=fold):
                self.configure_fold(fold)
                result = self.prepare(expected_fold=fold)
                selected, vessels = select_training_vessels(self.fitting, seed=20261027)
                self.assertEqual(result["category_models"][80].to_dict(),
                    CategoryCompatibility().fit(selected, self.catalog).to_dict())
                self.assertEqual(result["category_models"][100].to_dict(),
                    CategoryCompatibility().fit(self.fitting, self.catalog).to_dict())
                heldout = result["heldout"]
                self.assertFalse(set(selected.sample_id) & set(heldout.sample_id))
                self.assertFalse(vessels & set(heldout.target_vessel_id))
                self.assertEqual(set(result["membership"].query("selected_small").sample_id), set(selected.sample_id))
                self.assertEqual(result["proof"]["selection"]["seed"], 20261027)
                self.assertEqual(result["proof"]["coverage"]["fold"], fold)
                proof = result["category_proof"]
                self.assertEqual(proof["training_scope_proof"], result["proof"])
                self.assertEqual(proof["factor_levels"], FACTOR_LEVELS)
                json.dumps(proof, allow_nan=False)

    def test_both_runners_share_full_search_fixed_model_and_independent_builders(self):
        before = learned_assets_fingerprint(self.base.pipeline)
        result = self.prepare(expected_fold=1)
        self.assertEqual(tuple(result["runners"]), RUNNERS)
        self.assertEqual(RUNNERS, (REFERENCE, EXPERIMENT))
        self.assertEqual(set(result["search_pipelines"]), {100})
        for name, runner in result["runners"].items():
            self.assertIs(runner.ranker, self.base.ranker)
            self.assertIs(runner.selection, self.base.selection)
            self.assertIs(runner.feature_transform, self.base.feature_transform)
            self.assertIs(runner.pipeline.memory, self.base.pipeline.memory)
            self.assertIs(runner.pipeline.retriever, self.base.pipeline.retriever)
            self.assertIsNot(runner.pipeline, self.base.pipeline)
            self.assertIsNot(runner.pipeline.feature_builder, self.base.pipeline.feature_builder)
            self.assertIs(runner.pipeline.feature_builder.category_model,
                          result["category_models"][FACTOR_LEVELS[name]["category"]])
        self.assertEqual(before, learned_assets_fingerprint(self.base.pipeline))
        self.assertTrue(validate_category_wiring(result)["search_fixed_at_100"])

    def test_identical_candidates_noncategory_features_and_preserved_state(self):
        result = self.prepare()
        before = category_assets_fingerprint(result)
        query = result["heldout"].iloc[0].to_dict()
        reference, experiment = (result["runners"][name].pipeline for name in RUNNERS)
        left, right = reference.retrieve(query), experiment.retrieve(query)
        pd.testing.assert_frame_equal(left, right)
        full_features, small_features = reference.features(query, left), experiment.features(query, right)
        category_columns = [name for name in full_features if name.startswith(("flag_", "type_"))]
        self.assertEqual(len(category_columns), 16)
        pd.testing.assert_frame_equal(full_features.drop(columns=category_columns),
                                      small_features.drop(columns=category_columns))
        self.assertFalse(full_features[category_columns].equals(small_features[category_columns]))
        self.assertEqual(before, category_assets_fingerprint(result))

    def test_changed_seed_or_invalid_fold_rejected_before_loading(self):
        for kwargs in ({"subset_seed": 20261028}, {"subset_seed": True},
                       {"expected_fold": True}, {"expected_fold": 5}, {"expected_fold": -1}):
            with self.subTest(kwargs=kwargs), patch("portmatch.category_scope_assets.prepare_index_scale") as prepare:
                with self.assertRaises(ValueError):
                    prepare_category_scope("model", "train", "catalog", **kwargs)
                prepare.assert_not_called()
        with self.assertRaisesRegex(ValueError, "expected_fold"):
            self.prepare(expected_fold=3)

    def test_incorrect_search_or_category_binding_rejected(self):
        result = self.prepare()
        runner = result["runners"][EXPERIMENT]
        for owner, attr, value in (
            (runner, "ranker", object()),
            (runner.pipeline, "retriever", result["small_pipeline"].retriever),
            (runner.pipeline, "memory", result["small_pipeline"].memory),
            (runner.pipeline, "categories", self.base.pipeline.categories),
            (runner.pipeline.feature_builder, "category_model", self.base.pipeline.categories),
            (runner.pipeline.feature_builder, "names", list(self.base.pipeline.feature_builder.names)),
        ):
            previous = getattr(owner, attr)
            setattr(owner, attr, value)
            with self.subTest(attr=attr), self.assertRaises(ValueError):
                validate_category_wiring(result)
            setattr(owner, attr, previous)
        self.base.pipeline.feature_builder.category_model = result["category_models"][80]
        with self.assertRaisesRegex(ValueError, "Source components"):
            validate_category_wiring(result)

    def test_fingerprints_detect_category_and_search_changes_but_ignore_query_cache(self):
        result = self.prepare()
        before = category_assets_fingerprint(result)
        result["category_models"][80].tables["flag_name"]["total"] += 1
        self.assertNotEqual(before, category_assets_fingerprint(result))
        result["category_models"][80].tables["flag_name"]["total"] -= 1
        matrix = self.base.pipeline.retriever._name_index.matrix
        previous = matrix.data[0]
        matrix.data[0] += .1
        self.assertNotEqual(before, category_assets_fingerprint(result))
        matrix.data[0] = previous
        result["runners"][EXPERIMENT].pipeline._query_cache = {"transient": "ignored"}
        self.assertEqual(before, category_assets_fingerprint(result))

    def test_saved_full_category_corruption_is_rejected(self):
        self.base.pipeline.categories.tables["flag_name"]["total"] += 1
        with self.assertRaisesRegex(ValueError, "categories differs"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
