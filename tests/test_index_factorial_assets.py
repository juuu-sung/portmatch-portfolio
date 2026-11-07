"""Synthetic guards for independent search/category scopes and immutable bases."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from portmatch.index_factorial_assets import (
    FACTOR_LEVELS, RUNNERS, factorial_assets_fingerprint,
    prepare_index_factorial, validate_factorial_wiring,
)
from portmatch.index_scale_assets import learned_assets_fingerprint, select_training_vessels
from portmatch.ranking import VesselRanker
from portmatch.robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig
from portmatch.validation import make_folds


class FactorialAssetTests(unittest.TestCase):
    def setUp(self):
        specification = importlib.util.spec_from_file_location(
            "_factorial_asset_fixture", Path(__file__).with_name("test_robust_training.py"))
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        self.train, self.catalog = module.fixture()
        assignments = make_folds(self.train, "vessel", 5, 20261022)
        fitting = self.train.loc[assignments != 0].copy()
        heldout = self.train.loc[assignments == 0].copy()
        selected, _ = select_training_vessels(fitting)
        self.small = RobustMatchingPipeline(selected, self.catalog, RobustRetrievalConfig())
        self.full = RobustMatchingPipeline(fitting, self.catalog, RobustRetrievalConfig())
        self.base = SimpleNamespace(pipeline=self.full, ranker=object(), selection=object(),
            feature_transform={"fixture": "fixed"}, catalog_ids=self.catalog.vessel_id.to_numpy())
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.protected = Path(self.temporary.name) / "protected-model.txt"
        self.protected.write_text("unchanged synthetic weights", encoding="utf-8")
        self.context = {"matcher": self.base, "small_pipeline": self.small,
            "summary": {"complete": True}, "fitting": fitting, "heldout": heldout,
            "membership": self.train[["sample_id", "target_vessel_id"]].copy(),
            "proof": {"coverage": {"fold": 0}, "integrity": {
                "heldout_rows_excluded": True, "heldout_vessels_excluded": True}},
            "protected_files": {str(self.protected): hashlib.sha256(self.protected.read_bytes()).hexdigest()}}

    def prepare(self, **kwargs):
        # The upstream scale module has separate real-bundle partition tests.
        # This boundary verifies the factorial wrapper requests that exact path.
        with patch("portmatch.index_factorial_assets.prepare_index_scale",
                   return_value=self.context) as upstream, patch.object(
                VesselRanker, "fit", side_effect=AssertionError("diagnostic trained weights")):
            result = prepare_index_factorial("model", "train.csv", "catalog.csv", **kwargs)
            upstream.assert_called_once_with("model", "train.csv", "catalog.csv",
                fraction=.8, subset_seed=kwargs.get("subset_seed", 20261027),
                expected_fold=kwargs.get("expected_fold"))
        return result

    def test_four_cells_copy_wrappers_share_frozen_model_and_preserve_sources(self):
        before = {80: learned_assets_fingerprint(self.small), 100: learned_assets_fingerprint(self.full)}
        source_builders = (self.small.feature_builder, self.full.feature_builder)
        result = self.prepare(expected_fold=0)
        self.assertEqual(tuple(result["runners"]), RUNNERS)
        self.assertEqual(len(RUNNERS), 4)
        self.assertIs(result["matcher"], self.base)
        self.assertEqual(before, {80: learned_assets_fingerprint(self.small),
                                  100: learned_assets_fingerprint(self.full)})
        self.assertEqual(source_builders, (self.small.feature_builder, self.full.feature_builder))
        for name, runner in result["runners"].items():
            levels = FACTOR_LEVELS[name]
            search = result["search_pipelines"][levels["search"]]
            category = result["category_models"][levels["category"]]
            self.assertIs(runner.ranker, self.base.ranker)
            self.assertIs(runner.selection, self.base.selection)
            self.assertIs(runner.catalog_ids, self.base.catalog_ids)
            self.assertIs(runner.pipeline.retriever, search.retriever)
            self.assertIs(runner.pipeline.memory, search.memory)
            self.assertIs(runner.pipeline.categories, category)
            self.assertIs(runner.pipeline.feature_builder.category_model, category)
            self.assertIsNot(runner.pipeline, search)
            self.assertIsNot(runner.pipeline.feature_builder, search.feature_builder)
        self.assertTrue(validate_factorial_wiring(result)["whole_category_models_crossed"])

    def test_training_scopes_exclude_validation_and_proof_is_serializable(self):
        original_proof = copy.deepcopy(self.context["proof"])
        result = self.prepare()
        heldout = set(result["heldout"].target_vessel_id)
        for runner in result["runners"].values():
            self.assertFalse(set(runner.pipeline.memory.training_vessels) & heldout)
        self.assertLess(self.small.memory.training_vessels, self.full.memory.training_vessels)
        self.assertEqual(result["proof"], original_proof)
        proof = result["factorial_proof"]
        self.assertEqual(proof["training_scope_proof"], original_proof)
        self.assertIsNot(proof["training_scope_proof"], result["proof"])
        self.assertTrue(proof["source_assets_preserved_during_construction"])
        self.assertEqual(proof["factor_levels"], FACTOR_LEVELS)
        json.dumps(proof, ensure_ascii=False, allow_nan=False)

    def test_same_search_keeps_candidates_and_noncategory_features_identical(self):
        result = self.prepare()
        before = factorial_assets_fingerprint(result)
        query = result["heldout"].iloc[0].to_dict()
        category_columns = None
        for level in (80, 100):
            left = result["runners"][f"search{level}_category80"].pipeline
            right = result["runners"][f"search{level}_category100"].pipeline
            left_candidates, right_candidates = left.retrieve(query), right.retrieve(query)
            pd.testing.assert_frame_equal(left_candidates, right_candidates)
            left_features = left.features(query, left_candidates)
            right_features = right.features(query, right_candidates)
            category_columns = [name for name in left_features if name.startswith(("flag_", "type_"))]
            self.assertEqual(len(category_columns), 16)
            pd.testing.assert_frame_equal(left_features.drop(columns=category_columns),
                                          right_features.drop(columns=category_columns))
            self.assertFalse(left_features[category_columns].equals(right_features[category_columns]))
        # TF-IDF transform and feature construction must not mutate fitted state.
        self.assertEqual(before, factorial_assets_fingerprint(result))

    def test_mixed_category_binding_does_not_rewire_base_or_other_cell(self):
        result = self.prepare()
        mixed = result["runners"]["search100_category80"]
        self.assertIs(mixed.pipeline.categories, self.small.categories)
        self.assertIs(self.full.feature_builder.category_model, self.full.categories)
        self.assertIs(result["runners"]["search100_category100"].pipeline.categories, self.full.categories)
        self.assertIs(result["runners"]["search80_category100"].pipeline.categories, self.full.categories)
        self.assertIs(self.small.feature_builder.category_model, self.small.categories)

    def test_fingerprint_detects_search_categories_and_catalog_feature_mutations(self):
        result = self.prepare()
        before = factorial_assets_fingerprint(result)
        self.small.categories.tables["flag_name"]["total"] += 1
        self.assertNotEqual(before, factorial_assets_fingerprint(result))
        self.small.categories.tables["flag_name"]["total"] -= 1
        self.assertEqual(before, factorial_assets_fingerprint(result))
        matrix = self.small.retriever._name_index.matrix
        value = matrix.data[0]
        matrix.data[0] = value + .01
        self.assertNotEqual(before, factorial_assets_fingerprint(result))
        matrix.data[0] = value
        self.assertEqual(before, factorial_assets_fingerprint(result))
        names = self.full.feature_builder.names
        previous = names[0]
        names[0] = "changed"
        self.assertNotEqual(before, factorial_assets_fingerprint(result))
        names[0] = previous
        self.assertEqual(before, factorial_assets_fingerprint(result))
        result["runners"][RUNNERS[0]].pipeline._query_cache = {"transient": "allowed"}
        self.assertEqual(before, factorial_assets_fingerprint(result))

    def test_wrong_scope_or_sharing_is_rejected(self):
        result = self.prepare()
        runner = result["runners"]["search80_category100"]
        cases = [
            (runner, "ranker", object()),
            (runner, "selection", object()),
            (runner.pipeline, "memory", self.full.memory),
            (runner.pipeline, "retriever", self.full.retriever),
            (runner.pipeline, "categories", self.small.categories),
            (runner.pipeline, "feature_builder", self.small.feature_builder),
            (runner.pipeline.feature_builder, "category_model", self.small.categories),
            (runner.pipeline.feature_builder, "names", list(self.small.feature_builder.names)),
        ]
        for owner, name, bad in cases:
            previous = getattr(owner, name)
            setattr(owner, name, bad)
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_factorial_wiring(result)
            setattr(owner, name, previous)
        original = result["runners"][RUNNERS[0]]
        result["runners"][RUNNERS[0]] = result["runners"][RUNNERS[1]]
        with self.assertRaises(ValueError):
            validate_factorial_wiring(result)
        result["runners"][RUNNERS[0]] = original
        self.assertTrue(validate_factorial_wiring(result)["verified"])

    def test_source_rewiring_is_rejected(self):
        result = self.prepare()
        self.full.feature_builder.category_model = self.small.categories
        with self.assertRaisesRegex(ValueError, "Source components"):
            validate_factorial_wiring(result)

    def test_unsupported_fold_rejected_and_protected_files_checked(self):
        for fold in (1, 3, 4, -1, True, "0"):
            with self.subTest(fold=fold), patch(
                    "portmatch.index_factorial_assets.prepare_index_scale") as upstream:
                with self.assertRaisesRegex(ValueError, "only frozen outer folds"):
                    prepare_index_factorial("model", "train", "catalog", expected_fold=fold)
                upstream.assert_not_called()
        self.context["proof"]["coverage"]["fold"] = 1
        with self.assertRaisesRegex(ValueError, "only frozen outer folds"):
            self.prepare()
        self.context["proof"]["coverage"]["fold"] = 0
        self.protected.write_text("different", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Protected input/model file changed"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
