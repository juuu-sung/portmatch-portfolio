"""Small synthetic cache lineage and full-training-scope ablation safeguards."""
import contextlib
import copy
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib

from portmatch.data import read_catalog, read_queries
from portmatch.feature_selection import FeatureSelection
from portmatch.gt_augmentation import GTMissingAugmentationConfig
from portmatch.name_feature_cache import transform_cache
from portmatch.ranker_commands import _training_weighting_summary, feature_code_hash
from portmatch.ranking import RankerConfig, VesselRanker
from portmatch.retrieval_comparison import _hash
from portmatch.robust_retrieval import RobustRetrievalConfig
from portmatch.support_ablation_assets import (
    SUPPORT_FEATURES, _load_experiment, _verify_shared_cache, assert_assets_unchanged,
    check_support_ablation_metadata, feature_schemas, pipeline_fingerprints,
    preflight_support_training, prepare_support_ablation,
)
from portmatch.training_data import load_training_arrays, prepare_training_data


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class SupportAblationAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        spec = importlib.util.spec_from_file_location(
            "support_assets_fixture", Path(__file__).with_name("test_robust_training.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        train, catalog = module.fixture()
        cls.train_path, cls.catalog_path = cls.root / "train.csv", cls.root / "catalog.csv"
        train.to_csv(cls.train_path, index=False, encoding="utf-8-sig")
        catalog.to_csv(cls.catalog_path, index=False, encoding="utf-8-sig")
        cls.train = read_queries(cls.train_path, labeled=True)
        cls.catalog = read_catalog(cls.catalog_path)
        cls.raw, cls.cache = cls.root / "raw", cls.root / "cache"
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_training_data(cls.train, cls.catalog, cls.raw,
                retrieval_config=RobustRetrievalConfig(),
                augmentation_config=GTMissingAugmentationConfig(),
                provenance={"feature_code_sha256": feature_code_hash(),
                    "train_sha256": _hash(cls.train_path), "catalog_sha256": _hash(cls.catalog_path)})
            cls.metadata = transform_cache(cls.raw, cls.cache, cls.train_path, cls.catalog_path)
        _, _, groups, language, _, mass = load_training_arrays(cls.cache, return_query_mass=True)
        cls.weighting = _training_weighting_summary(groups, language, mass, .7)

    def setUp(self):
        self.case = self.root / self._testMethodName
        self.case.mkdir()
        self.directories, self.summaries, self.runners = {}, {}, {}
        schemas = feature_schemas()
        for name, selected in (("baseline86", schemas["baseline_features"]),
                               ("no_support82", schemas["experiment_features"])):
            directory = self.case / name
            directory.mkdir()
            selection = FeatureSelection.from_names(schemas["source_features"], selected)
            summary = {"schema_version": 1, "complete": True,
                "dataset": copy.deepcopy(self.metadata), "cache_directory": str(self.cache.resolve()),
                "cache_metadata_sha256": _hash(self.cache / "metadata.json"),
                "feature_selection": selection.to_dict(), "trained_feature_names": selected,
                "ranker": asdict(RankerConfig()), "training_weighting": copy.deepcopy(self.weighting)}
            shutil.copyfile(self.cache / "index.joblib", directory / "index.joblib")
            for filename in ("model.txt", "ranker.json", "validation_predictions.csv"):
                (directory / filename).write_text("synthetic test only", encoding="utf-8")
            write_json(directory / "training_summary.json", summary)
            runner = SimpleNamespace(pipeline=joblib.load(directory / "index.joblib"),
                catalog_ids=self.catalog.vessel_id.to_numpy(), selection=selection,
                feature_transform=self.metadata["feature_transform"],
                ranker=SimpleNamespace(config=RankerConfig(), feature_names=selected,
                    _booster=SimpleNamespace(model_to_string=lambda: "synthetic fixed weights")))
            self.directories[name], self.summaries[name], self.runners[name] = directory, summary, runner

    def save_summary(self, name):
        write_json(self.directories[name] / "training_summary.json", self.summaries[name])

    def patch_file(self, path, content):
        old = path.read_bytes()
        self.addCleanup(path.write_bytes, old)
        path.write_bytes(content)

    def prepare(self, **kwargs):
        def load(name):
            return (self.runners[name], copy.deepcopy(self.summaries[name]),
                {"load_and_integrity_seconds": 0.001, "index_sha256": self.metadata["retrieval_index_sha256"]})
        with patch("portmatch.support_ablation_assets._load_model", return_value=load("baseline86")), \
                patch("portmatch.support_ablation_assets._load_experiment", return_value=load("no_support82")), \
                patch.object(VesselRanker, "fit", side_effect=AssertionError("must not train")):
            return prepare_support_ablation(self.directories["baseline86"], self.directories["no_support82"],
                self.train_path, self.catalog_path, **kwargs)

    def test_exact_schema_and_strict_metadata_profile(self):
        schema = feature_schemas()
        self.assertEqual([len(schema[key]) for key in
            ("source_features", "baseline_features", "experiment_features")], [87, 86, 82])
        self.assertEqual(set(schema["baseline_features"]) - set(schema["experiment_features"]),
                         set(SUPPORT_FEATURES))
        check_support_ablation_metadata(self.metadata)
        for key, value in (("outer_fold", True), ("outer_fold", 5), ("sampled", True),
                           ("outer_split", "row"), ("seed", 20261023), ("inner_folds", 2),
                           ("max_valid_queries", 1), ("training_reweighting", None),
                           ("training_scope", "full_public_train")):
            changed = copy.deepcopy(self.metadata)
            changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check_support_ablation_metadata(changed)
        changed = copy.deepcopy(self.metadata)
        changed["feature_names"] = schema["experiment_features"]
        with self.assertRaisesRegex(ValueError, "87-column"):
            check_support_ablation_metadata(changed)

    def test_preflight_binds_the_existing_cache_without_retrieval_or_training(self):
        with patch.object(VesselRanker, "fit", side_effect=AssertionError("must not train")):
            result = preflight_support_training(self.cache, self.directories["baseline86"], expected_fold=0)
        self.assertTrue(result["cache_proof"]["same_candidates_features_labels_query_mass"])
        self.assertIn(str(self.raw / self.metadata["train_shards"][0]), result["protected_files"])
        with self.assertRaisesRegex(ValueError, "baseline cache directory"):
            preflight_support_training(self.raw, self.directories["baseline86"])
        with self.assertRaisesRegex(ValueError, "expected_fold"):
            preflight_support_training(self.cache, self.directories["baseline86"], expected_fold=2)

    def test_actual_derived_and_parent_cache_bytes_are_checked(self):
        for root in (self.cache, self.raw):
            path = root / self.metadata["train_shards"][0]
            old = path.read_bytes()
            try:
                path.write_bytes(old + b"changed")
                with self.subTest(root=root), self.assertRaisesRegex(ValueError, "asset hash mismatch"):
                    _verify_shared_cache(self.summaries["baseline86"])
            finally:
                path.write_bytes(old)
        path = self.cache / "derivation.json"
        self.patch_file(path, path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            _verify_shared_cache(self.summaries["baseline86"])

    def test_full_T_assets_and_training_population_are_verified(self):
        result = self.prepare(expected_fold=0)
        self.assertEqual(set(result["runners"]), {"baseline86", "no_support82"})
        self.assertEqual(result["coverage"]["full"]["rows"], self.metadata["counts"]["outer_training_rows"])
        membership = result["membership"]
        self.assertTrue(membership.search100.equals(membership.category100))
        self.assertFalse(membership.loc[membership.role.eq("valid"), "search100"].any())
        self.assertTrue(result["proof"]["integrity"]["full_training_assets_recomputed_and_equal"])
        self.assertTrue(assert_assets_unchanged(result)["shared_cache_unchanged"])
        json.dumps(result["proof"], allow_nan=False)
        self.runners["no_support82"].pipeline.categories.tables["flag_name"]["total"] += 1
        with self.assertRaisesRegex(ValueError, "categories differs"):
            self.prepare()

    def test_wrong_cache_selection_weighting_or_index_fails(self):
        original = copy.deepcopy(self.summaries["no_support82"])
        for change in ("path", "schema", "weight", "config"):
            altered = copy.deepcopy(original)
            if change == "path":
                altered["cache_directory"] = str(self.raw)
            elif change == "schema":
                names = feature_schemas()["baseline_features"]
                altered["feature_selection"] = FeatureSelection.from_names(self.metadata["feature_names"], names).to_dict()
                altered["trained_feature_names"] = names
            elif change == "weight":
                altered["training_weighting"]["policy"] = "changed policy"
            else:
                altered["ranker"]["n_estimators"] = 301
            self.summaries["no_support82"] = altered
            self.save_summary("no_support82")
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.prepare()
        self.summaries["no_support82"] = original
        self.save_summary("no_support82")
        path = self.directories["no_support82"] / "index.joblib"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "same cache search index"):
            self.prepare()

    def test_mutation_guards_cover_cache_weights_selection_and_query_cache_exception(self):
        result = self.prepare()
        runner = result["runners"]["no_support82"]
        runner.pipeline._query_feature_cache = {"allowed": True}
        self.assertEqual(pipeline_fingerprints(result["runners"]), result["asset_fingerprints"])
        runner.ranker._booster.model_to_string = lambda: "changed weights"
        with self.assertRaisesRegex(ValueError, "fitted weights changed"):
            assert_assets_unchanged(result)
        runner.ranker._booster.model_to_string = lambda: "synthetic fixed weights"
        path = self.cache / self.metadata["valid_shards"][0]
        self.patch_file(path, path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Protected input/model"):
            assert_assets_unchanged(result)

    def test_experiment_loader_checks_native_schema_and_input_provenance(self):
        with patch("portmatch.ranker_commands.RankedMatcher", return_value=self.runners["no_support82"]):
            matcher, _, _ = _load_experiment(self.directories["no_support82"],
                _hash(self.train_path), _hash(self.catalog_path))
            self.assertIs(matcher, self.runners["no_support82"])
            with self.assertRaisesRegex(ValueError, "train/catalog"):
                _load_experiment(self.directories["no_support82"], "0" * 64, _hash(self.catalog_path))
            matcher.ranker.feature_names = feature_schemas()["baseline_features"]
            with self.assertRaisesRegex(ValueError, "exactly 82"):
                _load_experiment(self.directories["no_support82"],
                    _hash(self.train_path), _hash(self.catalog_path))


if __name__ == "__main__":
    unittest.main()
