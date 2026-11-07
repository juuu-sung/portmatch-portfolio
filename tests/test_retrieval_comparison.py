import argparse
import copy
from contextlib import redirect_stdout
from dataclasses import asdict
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import joblib

from portmatch.features import CategoryCompatibility
from portmatch.memory import TrainingMemory

from portmatch.retrieval_comparison import (
    VARIANTS, _check_saved_predictions, _validate_model_pair, _validate_training_assets,
    _validate_reference_pair, _load_model,
    add_retrieval_comparison_parser, paired_changes, select_queries,
    summarize_predictions, timed_prediction, variant_query, variant_effect_counts, metric_deltas,
    compare_retrieval,
)


def frame():
    return pd.DataFrame([
        {"sample_id": str(i), "vessel_name": "한글" if i < 6 else "LATIN",
         "target_vessel_id": str(i // 2), "gross_tonnage": 100.,
         "international_gross_tonnage": 120., "call_sign": "CALL",
         "flag_name": "FLAG", "vessel_type": "TYPE", "arrival_year": 2024}
        for i in range(12)
    ])


def result(sample, vessel, variant, model, correct, language=True):
    return {"sample_id": sample, "target_vessel_id": vessel,
        "variant": variant, "model": model, "vessel_name": vessel,
        "has_hangul": language, "predicted_vessel_id": vessel if correct else "OTHER",
        "candidate_found": True, "target_rank": 1 if correct else 2,
        "candidate_count": 3 if model == "baseline" else 5,
        "retrieval_seconds": 1., "feature_seconds": 2., "rank_seconds": 3., "total_seconds": 6.}


def training_assets_fixture(vessels=3):
    catalog = pd.DataFrame([
        {"vessel_id": f"V{i}", "vessel_name": f"NAME {i}",
         "flag_name": "Catalog Flag", "vessel_type": "Catalog Type"}
        for i in range(vessels)
    ])
    train = pd.DataFrame([
        {"target_vessel_id": f"V{i}", "vessel_name": f"Name {i}",
         "call_sign": f"CALL{i}", "flag_name": "Query Flag", "vessel_type": "Query Type"}
        for i in range(vessels) for _ in range(2)
    ])
    return SimpleNamespace(memory=TrainingMemory().fit(train, catalog),
                           categories=CategoryCompatibility().fit(train, catalog))


class RetrievalComparisonTests(unittest.TestCase):
    def test_variants_preserve_original_and_strip_labels_ids(self):
        row = frame().iloc[0].to_dict()
        row["injected_label"] = "BAD"
        original = copy.deepcopy(row)
        transformed = {name: variant_query(row, name) for name in VARIANTS}
        self.assertEqual(row, original)
        for query in transformed.values():
            self.assertNotIn("sample_id", query)
            self.assertNotIn("target_vessel_id", query)
            self.assertNotIn("injected_label", query)
            self.assertEqual(query["vessel_name"], "한글")
        self.assertEqual(transformed["original"]["international_gross_tonnage"], 120.)
        self.assertEqual(transformed["international_gt_x1_25"]["international_gross_tonnage"], 150.)
        self.assertEqual(transformed["international_gt_x1_25"]["gross_tonnage"], 100.)
        self.assertTrue(np.isnan(transformed["both_gt_missing"]["gross_tonnage"]))
        self.assertTrue(np.isnan(transformed["both_gt_missing"]["international_gross_tonnage"]))
        self.assertTrue(np.isnan(transformed["international_gt_missing"]["international_gross_tonnage"]))
        self.assertEqual(transformed["international_gt_missing"]["gross_tonnage"], 100.)
        self.assertTrue(np.isnan(transformed["gross_gt_missing"]["gross_tonnage"]))
        self.assertEqual(transformed["gross_gt_missing"]["international_gross_tonnage"], 120.)

    def test_multiplier_does_not_invent_tonnage_for_invalid_values(self):
        for value in (None, np.nan, 0, -1, "unknown", np.inf):
            with self.subTest(value=value):
                query = variant_query({"international_gross_tonnage": value}, "international_gt_x1_25")
                if value is not None and isinstance(value, float) and np.isnan(value):
                    self.assertTrue(np.isnan(query["international_gross_tonnage"]))
                else:
                    self.assertEqual(query["international_gross_tonnage"], value)
        with self.assertRaisesRegex(ValueError, "Unknown input"):
            variant_query({}, "invented")

    def test_default_selection_is_full_and_diagnostic_is_stable_under_reorder(self):
        source = frame()
        pd.testing.assert_frame_equal(select_queries(source), source)
        for kwargs in ({"max_queries": 4}, {"max_vessels_per_language": 2},
                       {"max_vessels_per_language": 2, "one_row_per_vessel": True}):
            left = select_queries(source, **kwargs, seed=3)
            right = select_queries(source.iloc[::-1], **kwargs, seed=3)
            self.assertEqual(set(left.sample_id), set(right.sample_id))
            self.assertTrue(left.vessel_name.eq("한글").any())
            self.assertTrue(left.vessel_name.eq("LATIN").any())
        selected = select_queries(source, max_vessels_per_language=2, one_row_per_vessel=True)
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected.target_vessel_id.nunique(), 4)
        self.assertEqual(len(select_queries(source, max_vessels_per_language=2)), 8)

    def test_invalid_selection_options_rejected(self):
        for kwargs in ({"max_queries": 0}, {"max_vessels_per_language": -1},
                       {"max_queries": 1, "max_vessels_per_language": 1},
                       {"one_row_per_vessel": True}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_queries(frame(), **kwargs)

    def test_variant_effect_counts_exclude_already_missing_and_invalid_values(self):
        source = frame().iloc[[0, 1, 6]].copy()
        source.loc[1, ["gross_tonnage", "international_gross_tonnage"]] = np.nan
        source.loc[6, "international_gross_tonnage"] = 0
        effects = variant_effect_counts(source, VARIANTS)
        self.assertEqual(effects["original"]["all"]["changed"], 0)
        self.assertEqual(effects["both_gt_missing"]["all"]["changed"], 2)
        self.assertEqual(effects["both_gt_missing"]["hangul"]["changed"], 1)
        self.assertEqual(effects["international_gt_x1_25"]["all"]["changed"], 1)
        self.assertEqual(effects["international_gt_x1_25"]["non_hangul"]["unchanged"], 1)

    def test_pairing_uses_same_variant_and_identity_not_csv_order(self):
        rows = []
        for variant, base_good, experiment_good in (("original", True, True),
                ("both_gt_missing", False, True), ("gross_gt_missing", True, False)):
            rows += [result("1", "VESSEL", variant, "baseline", base_good),
                     result("1", "VESSEL", variant, "robust", experiment_good)]
        changes, vessels = paired_changes(pd.DataFrame(rows[::-1]))
        status = changes.set_index("variant").status.to_dict()
        self.assertEqual(status, {"original": "unchanged_correct", "both_gt_missing": "corrected",
                                  "gross_gt_missing": "regressed"})
        self.assertEqual(vessels.set_index("variant").net_corrected.to_dict(),
                         {"original": 0, "both_gt_missing": 1, "gross_gt_missing": -1})
        self.assertTrue(changes.candidate_count_delta.eq(2).all())

    def test_pairing_rejects_missing_rows_or_different_truth(self):
        rows = [result("1", "VESSEL", "original", "baseline", True),
                result("1", "OTHER", "original", "robust", True)]
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            paired_changes(pd.DataFrame(rows))
        rows[-1]["sample_id"] = "2"
        with self.assertRaisesRegex(ValueError, "identical variant"):
            paired_changes(pd.DataFrame(rows))

    def test_direct_pairing_uses_reference_predictions_and_excludes_other_runners(self):
        rows = [result("1", "A", "original", "baseline", False),
                result("1", "A", "original", "robust", False),
                result("1", "A", "original", "reference", True),
                result("1", "A", "original", "trained", False)]
        changes, vessels = paired_changes(pd.DataFrame(rows), reference="reference", experiments=("trained",))
        self.assertEqual(changes.model.tolist(), ["trained"])
        self.assertEqual(changes.status.tolist(), ["regressed"])
        self.assertEqual(changes.predicted_vessel_id_reference.tolist(), ["A"])
        self.assertEqual(vessels.net_corrected.tolist(), [-1])
        rows.append(result("2", "B", "original", "trained", True))
        with self.assertRaisesRegex(ValueError, "identical variant"):
            paired_changes(pd.DataFrame(rows), reference="reference", experiments=("trained",))
        with self.assertRaisesRegex(ValueError, "Missing paired reference"):
            paired_changes(pd.DataFrame(rows), reference="unknown")
        with self.assertRaisesRegex(ValueError, "Missing paired experiment"):
            paired_changes(pd.DataFrame(rows), experiments=("unknown",))

    def test_metrics_are_language_weighted_and_include_counts_and_stage_times(self):
        rows = [result("h", "H", "original", "baseline", True),
                result("n1", "N1", "original", "baseline", False, False),
                result("n2", "N2", "original", "baseline", True, False)]
        rows[2]["candidate_count"] = 9
        metrics = summarize_predictions(pd.DataFrame(rows))
        self.assertAlmostEqual(metrics["weighted_top1"], .85)
        self.assertEqual(metrics["all"]["candidate_count"]["mean"], 5.)
        self.assertEqual(metrics["all"]["candidate_count"]["max"], 9)
        self.assertEqual(metrics["all"]["timing"]["total_seconds"]["sum"], 18.)
        self.assertEqual(metrics["hangul"]["timing"]["feature_seconds"]["sum"], 2.)
        corrected = copy.deepcopy(rows)
        corrected[1]["target_rank"] = 1
        corrected[1]["predicted_vessel_id"] = corrected[1]["target_vessel_id"]
        change = metric_deltas(metrics, summarize_predictions(pd.DataFrame(corrected)))
        self.assertAlmostEqual(change["weighted_top1_pp"], 15.)
        self.assertEqual(change["non_hangul"]["top1_accuracy_pp"], 50.)
        self.assertEqual(change["all"]["timing"]["total_seconds"]["sum_ratio"], 1.)

    def test_live_rank_tie_is_stable_row_independent_and_labels_never_enter_features(self):
        seen = []
        class Pipeline:
            def retrieve(self, row):
                seen.append(dict(row))
                return pd.DataFrame({"catalog_index": [1, 0]})
            def features(self, row, candidates):
                seen.append(dict(row))
                return pd.DataFrame({"x": [1., 1.]})
        matcher = SimpleNamespace(pipeline=Pipeline(), catalog_ids=np.array(["A", "Z"]),
            feature_transform={}, selection=SimpleNamespace(selected_features=("x",), apply=lambda matrix, names: matrix),
            ranker=SimpleNamespace(predict=lambda matrix, names: np.ones(len(matrix))))
        queries = [variant_query(row, "both_gt_missing") for row in frame().iloc[:3].to_dict("records")]
        with patch("portmatch.feature_transforms.transform_features", side_effect=lambda spec, pipeline, row, candidates, features: features):
            first = [timed_prediction(matcher, query, "A") for query in queries]
            reverse = [timed_prediction(matcher, query, "A") for query in reversed(queries)]
            single = timed_prediction(matcher, queries[0], "A")
        for item in [*first, *reverse, single]:
            self.assertEqual(item["predicted_vessel_id"], "A")
            self.assertEqual(item["target_rank"], 1)
            self.assertEqual(item["candidate_ids_sha256"], first[0]["candidate_ids_sha256"])
            self.assertAlmostEqual(item["total_seconds"], sum(item[key] for key in (
                "retrieval_seconds", "feature_seconds", "rank_seconds")))
        self.assertTrue(all("target_vessel_id" not in query and "sample_id" not in query for query in seen))

    def test_live_rank_rejects_duplicate_candidates(self):
        matcher = SimpleNamespace(pipeline=SimpleNamespace(
            retrieve=lambda row: pd.DataFrame({"catalog_index": [0, 0]})), catalog_ids=np.array(["A"]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            timed_prediction(matcher, {}, "A")

    def test_original_parity_checks_complete_saved_fold_and_prediction_identity(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = pd.DataFrame([result("a", "A", "original", "baseline", True),
                                 result("b", "B", "original", "baseline", True, False)])
            rows.iloc[::-1].to_csv(directory / "validation_predictions.csv", index=False)
            report = _check_saved_predictions(directory, rows.iloc[:1], rows.sample_id)
            self.assertEqual(report["matched_queries"], 1)
            wrong = rows.copy()
            wrong.loc[0, "predicted_vessel_id"] = "WRONG"
            with self.assertRaisesRegex(ValueError, "predicted_vessel_id"):
                _check_saved_predictions(directory, wrong, rows.sample_id)
            with self.assertRaisesRegex(ValueError, "Live original trained differs"):
                _check_saved_predictions(directory, wrong, rows.sample_id, "trained")
            with self.assertRaisesRegex(ValueError, "complete held-out"):
                _check_saved_predictions(directory, rows, ["a", "b", "c"])

    def test_trained_comparison_rejects_changed_split_ranker_and_memory(self):
        settings = {"outer_split": "vessel", "outer_folds": 5, "outer_fold": 0,
            "inner_split": "mixed", "inner_folds": 3, "seed": 1, "sampling": {"hard_negatives": 96},
            "max_train_queries": None, "max_valid_queries": None, "sampled": False,
            "provenance": {"train_sha256": "a", "catalog_sha256": "b", "feature_code_sha256": "c"},
            "retrieval": {"name_top_k": 80}}
        summary = {"dataset": settings, "ranker": {"n_estimators": 300}}
        other = copy.deepcopy(summary)
        other["dataset"]["retrieval_extension"] = {"name": "robust"}
        matcher = SimpleNamespace(ranker=SimpleNamespace(feature_names=["x"]),
            pipeline=training_assets_fixture())
        _validate_model_pair(matcher, summary, matcher, other)
        for change in (lambda x: x["dataset"].update(outer_fold=1),
                       lambda x: x["ranker"].update(n_estimators=301),
                       lambda x: x["dataset"]["retrieval"].update(name_top_k=81)):
            changed = copy.deepcopy(other)
            change(changed)
            with self.assertRaises(ValueError):
                _validate_model_pair(matcher, summary, matcher, changed)
        wrong_matcher = copy.deepcopy(matcher)
        wrong_matcher.pipeline.memory.training_rows += 1
        with self.assertRaisesRegex(ValueError, "memory differs"):
            _validate_model_pair(matcher, summary, wrong_matcher, other)

    def test_training_assets_accept_equal_state_saved_with_different_python_hash_seeds(self):
        # The actual regression arose after fitting/saving in separate Python
        # processes. Equal frozensets can retain different pickle iteration
        # orders; comparing their values must survive that round trip.
        script = (
            "import joblib, runpy, sys; "
            "fixture = runpy.run_path(sys.argv[1])['training_assets_fixture']; "
            "assets = fixture(500); "
            "joblib.dump(assets, sys.argv[2])"
        )
        with TemporaryDirectory() as temporary:
            saved = []
            for seed in ("1", "2"):
                output = Path(temporary) / f"assets_{seed}.joblib"
                subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), str(output)],
                    env={**os.environ, "PYTHONHASHSEED": seed}, check=True,
                    capture_output=True, text=True, timeout=30)
                saved.append(joblib.load(output))
            self.assertEqual(saved[0].memory.training_vessels, saved[1].memory.training_vessels)
            _validate_training_assets(*saved)
            # Re-reading a frame can populate pandas caches without changing
            # the learned aliases or the identity of the retrieval index.
            saved[1].memory.aliases["vessel_name"]
            _validate_training_assets(*saved)

    def test_augmented_reference_contract_rejects_changed_controls_and_unknown_spec(self):
        from portmatch.gt_augmentation import GTMissingAugmentationConfig, augmentation_spec
        from portmatch.robust_retrieval import RobustRetrievalConfig, retrieval_spec
        settings = {"outer_split": "vessel", "outer_folds": 5, "outer_fold": 0,
            "inner_split": "mixed", "inner_folds": 3, "seed": 1, "sampling": {"hard_negatives": 96},
            "max_train_queries": None, "max_valid_queries": None, "sampled": False,
            "provenance": {"train_sha256": "a", "catalog_sha256": "b", "feature_code_sha256": "c"},
            "retrieval": asdict(RobustRetrievalConfig()),
            "retrieval_extension": retrieval_spec(RobustRetrievalConfig()),
            "feature_transform": {"name": "same"}, "feature_names": ["x"]}
        reference = {"dataset": settings, "ranker": {"n_estimators": 300}}
        experiment = copy.deepcopy(reference)
        experiment["dataset"].update(
            counts={"selected_training_queries": 1, "train_groups": 4},
            training_augmentation=augmentation_spec(GTMissingAugmentationConfig()),
            training_augmentation_diagnostics={
                "mass_policy_verified": True, "selected_sources": 1, "retained_sources": 1,
                "dropped_sources": 0, "retained_views": 1, "retained_groups": 4,
                "views_without_original": 0, "views_without_masks": 0,
                "source_mass_min": 1., "source_mass_max": 1., "total_query_mass": 1.,
                "hangul_query_mass": 1., "non_hangul_query_mass": 0.,
                "variant_query_mass": {"original": .5, "both_gt_missing": 1 / 6,
                                       "international_gt_missing": 1 / 6, "gross_gt_missing": 1 / 6},
                "variants": {variant: {"stored": 1} for variant in
                    ("original", "both_gt_missing", "international_gt_missing", "gross_gt_missing")}},
            training_augmentation_assets={key: "a" * 64 for key in
                ("querylog.csv", "partitions.csv", "training_groups_sha256")})
        matcher = SimpleNamespace(ranker=SimpleNamespace(feature_names=["x"]), pipeline=training_assets_fixture())
        _validate_reference_pair(matcher, reference, matcher, experiment)
        for name, mutate in (
            ("split", lambda x: x["dataset"].update(outer_fold=1)),
            ("ranker", lambda x: x["ranker"].update(n_estimators=301)),
            ("retrieval", lambda x: x["dataset"]["retrieval"].update(context_top_k=161)),
            ("extension", lambda x: x["dataset"]["retrieval_extension"].update(code_sha256="bad")),
            ("features", lambda x: x["dataset"].update(feature_names=["different"])),
            ("transform", lambda x: x["dataset"]["feature_transform"].update(name="different")),
            ("unknown_augmentation", lambda x: x["dataset"]["training_augmentation"].update(name="unknown")),
        ):
            changed = copy.deepcopy(experiment)
            mutate(changed)
            with self.subTest(name=name), self.assertRaises(ValueError):
                _validate_reference_pair(matcher, reference, matcher, changed)
        with self.assertRaisesRegex(ValueError, "unaugmented"):
            _validate_reference_pair(matcher, experiment, matcher, experiment)
        with self.assertRaisesRegex(ValueError, "declared augmented"):
            _validate_reference_pair(matcher, reference, matcher, reference)
        changed_matcher = copy.deepcopy(matcher)
        changed_matcher.pipeline.memory.training_rows += 1
        with self.assertRaisesRegex(ValueError, "memory differs"):
            _validate_reference_pair(matcher, reference, changed_matcher, experiment)

    def test_reference_option_requires_an_experiment_before_loading(self):
        with self.assertRaisesRegex(ValueError, "requires an augmented experiment"):
            compare_retrieval("absent-baseline", "absent-output", reference_model="absent-reference")

    def test_training_assets_reject_changed_lookup_counts_including_zero_count_keys(self):
        original = training_assets_fixture()
        for field in ("call_signs", "names"):
            for zero_count_key in (False, True):
                with self.subTest(field=field, zero_count_key=zero_count_key):
                    changed = copy.deepcopy(original)
                    table = getattr(changed.memory, field)
                    key = next(iter(table))
                    if zero_count_key:
                        table[key][999] = 0
                        # Counter's equality ignores zero-count keys, whereas
                        # lookup treats membership itself as a candidate hit.
                        self.assertEqual(table[key], getattr(original.memory, field)[key])
                    else:
                        table[key][next(iter(table[key]))] += 1
                    with self.assertRaisesRegex(ValueError, f"memory differs.*{field}"):
                        _validate_training_assets(original, changed)

    def test_training_assets_reject_changed_alias_content_order_and_dtype(self):
        original = training_assets_fixture()
        for mutation in ("content", "catalog_position", "order", "dtype", "index", "columns"):
            with self.subTest(mutation=mutation):
                changed = copy.deepcopy(original)
                aliases = changed.memory.aliases
                if mutation == "content":
                    aliases.loc[0, "vessel_name"] = "Different name"
                elif mutation == "catalog_position":
                    aliases.loc[0, "catalog_index"] = 2
                elif mutation == "order":
                    changed.memory.aliases = aliases.iloc[::-1].reset_index(drop=True)
                elif mutation == "dtype":
                    changed.memory.aliases = aliases.astype({"catalog_index": "float64"})
                elif mutation == "index":
                    aliases.index = pd.Index([10, 11, 12])
                else:
                    changed.memory.aliases = aliases[["catalog_index", "vessel_name"]]
                with self.assertRaisesRegex(ValueError, "memory differs.*aliases"):
                    _validate_training_assets(original, changed)

    def test_training_assets_reject_changed_training_partition(self):
        original = training_assets_fixture()
        for field, value in (("training_rows", original.memory.training_rows + 1),
                             ("training_vessels", frozenset({"V0", "V1", "OTHER"}))):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                setattr(changed.memory, field, value)
                with self.assertRaisesRegex(ValueError, f"memory differs.*{field}"):
                    _validate_training_assets(original, changed)

    def test_training_assets_reject_changed_category_statistics_and_smoothing(self):
        original = training_assets_fixture()
        for field in ("smoothing", "universe", "source_counts", "target_counts", "pairs", "total"):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                table = changed.categories.tables["flag_name"]
                if field == "smoothing":
                    changed.categories.smoothing += 1
                elif field == "universe":
                    table[field].append("other flag")
                elif field == "total":
                    table[field] += 1
                elif field == "pairs":
                    counts = table[field][next(iter(table[field]))]
                    counts[next(iter(counts))] += 1
                else:
                    table[field][next(iter(table[field]))] += 1
                with self.assertRaisesRegex(ValueError, "categories differs"):
                    _validate_training_assets(original, changed)

    def test_training_assets_reject_unexpected_asset_schema_and_type(self):
        original = training_assets_fixture()
        for asset in ("memory", "categories"):
            for mutation in ("added_field", "missing_field", "wrong_type"):
                with self.subTest(asset=asset, mutation=mutation):
                    changed = copy.deepcopy(original)
                    value = getattr(changed, asset)
                    if mutation == "added_field":
                        value.unchecked_future_state = {}
                    elif mutation == "missing_field":
                        delattr(value, next(iter(vars(value))))
                    else:
                        setattr(changed, asset, dict(vars(value)))
                    with self.assertRaisesRegex(ValueError, f"{asset} differs.*unsupported state schema"):
                        _validate_training_assets(original, changed)

    def test_parser_default_full_fold_and_optional_diagnostic_limits(self):
        parser = argparse.ArgumentParser()
        add_retrieval_comparison_parser(parser.add_subparsers(dest="command"))
        args = parser.parse_args(["compare-retrieval", "--baseline", "base", "--output", "out"])
        self.assertEqual(args.profiles, ["hangul", "dual-gt", "robust"])
        self.assertEqual(args.variants, list(VARIANTS))
        self.assertIsNone(args.max_queries)
        self.assertIsNone(args.max_vessels_per_language)
        self.assertIsNone(args.reference_model)
        args = parser.parse_args(["compare-retrieval", "--baseline", "base", "--output", "out",
            "--profile", "robust", "--max-vessels-per-language", "48", "--one-row-per-vessel", "--selection-seed", "9"])
        self.assertEqual(args.profiles, ["robust"])
        self.assertTrue(args.one_row_per_vessel)
        self.assertEqual(args.seed, 9)
        args = parser.parse_args(["compare-retrieval", "--baseline", "base", "--output", "out",
            "--reference-model", "robust", "--experiment-model", "augmented"])
        self.assertEqual(args.reference_model, "robust")
        self.assertEqual(args.experiment_model, "augmented")


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class RetrievalComparisonIntegrationTests(unittest.TestCase):
    def test_unsampled_fold_live_comparison_includes_retrained_model_and_rejects_bad_provenance(self):
        # Reuse the 24-row/12-vessel synthetic fixture without importing tests
        # as a package (unittest discovery and direct file runs differ there).
        spec = importlib.util.spec_from_file_location(
            "_robust_training_fixture", Path(__file__).with_name("test_robust_training.py"))
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        from portmatch.name_feature_cache import transform_cache
        from portmatch.ranker_commands import add_training_parsers, run_training_command

        with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            train, catalog = fixture_module.fixture()
            train_path, catalog_path = root / "train.csv", root / "catalog.csv"
            train.to_csv(train_path, index=False, encoding="utf-8-sig")
            catalog.to_csv(catalog_path, index=False, encoding="utf-8-sig")
            parser = argparse.ArgumentParser()
            add_training_parsers(parser.add_subparsers(dest="command", required=True))
            models = {}
            for profile in ("legacy", "robust", "augmented"):
                source, derived, model = [root / f"{profile}_{suffix}"
                                          for suffix in ("source", "derived", "model")]
                prepare_arguments = [
                    "prepare-ranker", "--train", str(train_path), "--catalog", str(catalog_path),
                    "--output", str(source), "--retrieval-profile", "robust" if profile == "augmented" else profile,
                    "--folds", "3", "--inner-folds", "2", "--seed", "42"]
                if profile == "augmented":
                    prepare_arguments += ["--gt-augmentation", "missing", "--original-mass", "0.5"]
                run_training_command(parser.parse_args(prepare_arguments))
                transform_cache(source, derived, train_path, catalog_path)
                if profile == "augmented":
                    source_metadata = json.loads((source / "metadata.json").read_text("utf-8"))
                    derived_metadata = json.loads((derived / "metadata.json").read_text("utf-8"))
                    for field in ("training_augmentation", "training_augmentation_diagnostics", "training_augmentation_assets"):
                        self.assertEqual(source_metadata[field], derived_metadata[field])
                    for relative in source_metadata["train_shards"]:
                        with np.load(source / relative, allow_pickle=False) as original, np.load(derived / relative, allow_pickle=False) as transformed:
                            for field in ("sample_id", "query_id", "augmentation_variant", "query_mass", "group", "y"):
                                np.testing.assert_array_equal(original[field], transformed[field])
                    for relative in source_metadata["valid_shards"]:
                        with np.load(derived / relative, allow_pickle=False) as validation:
                            self.assertNotIn("query_mass", validation.files)
                            self.assertNotIn("augmentation_variant", validation.files)
                run_training_command(parser.parse_args([
                    "train-ranker", "--data", str(derived), "--output", str(model),
                    "--rounds", "3", "--threads", "1", "--num-leaves", "4",
                    "--min-child-samples", "1", "--seed", "42",
                    "--exclude-features", "retrieval_tonnage_rank"]))
                models[profile] = model
            output = root / "comparison"
            result = compare_retrieval(models["legacy"], output, train_path=train_path,
                catalog_path=catalog_path, experiment_model=models["robust"])
            self.assertTrue(result["complete"])
            self.assertFalse(result["coverage"]["diagnostic_sample"])
            self.assertEqual(result["coverage"]["queries"], 8)
            self.assertEqual(result["baseline_parity"]["matched_queries"], 8)
            self.assertEqual(result["baseline_parity"]["saved_fold_queries"], 8)
            self.assertTrue(result["baseline_parity"]["all_selected_original_predictions_match"])
            self.assertTrue(result["trained_parity"]["all_selected_original_predictions_match"])
            self.assertEqual(len(result["provenance"]["feature_names"]), 86)
            self.assertEqual(result["provenance"]["ranker"]["n_estimators"], 3)
            for variant in VARIANTS:
                self.assertEqual(set(result["metrics"][variant]),
                                 {"baseline", "hangul", "dual-gt", "robust", "trained"})
                self.assertEqual(result["metrics"][variant]["trained"]["all"]["queries"], 8)
                self.assertEqual(sum(result["transitions"][variant]["trained"].values()), 8)
                self.assertIn("trained", result["paired_metric_deltas"][variant])
            self.assertEqual(result["variant_effects"]["original"]["all"]["changed"], 0)
            self.assertEqual(result["variant_effects"]["both_gt_missing"]["all"]["changed"], 8)
            self.assertEqual(result["variant_effects"]["international_gt_x1_25"]["hangul"]["changed"], 4)
            for name in ("summary.json", "report.md", "predictions.csv", "row_changes.csv", "vessel_changes.csv"):
                self.assertTrue((output / name).is_file())
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(len(predictions), 8 * 5 * 5)
            self.assertTrue(predictions.groupby(["variant", "model"]).size().eq(8).all())
            self.assertEqual(json.loads((output / "summary.json").read_text("utf-8")), result)

            for name, experiment, reference, message in (
                ("missing_reference", models["augmented"], None, "requires an unaugmented robust reference"),
                ("augmented_reference", models["augmented"], models["augmented"], "must be the unaugmented"),
                ("unaugmented_experiment", models["robust"], models["robust"], "declared augmented"),
            ):
                rejected = root / name
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    compare_retrieval(models["legacy"], rejected, train_path=train_path,
                        catalog_path=catalog_path, profiles=("robust",),
                        experiment_model=experiment, reference_model=reference)
                self.assertFalse(rejected.exists())

            augmented_output = root / "augmentation_comparison"
            augmented = compare_retrieval(models["legacy"], augmented_output, train_path=train_path,
                catalog_path=catalog_path, profiles=("robust",), reference_model=models["robust"],
                experiment_model=models["augmented"])
            self.assertTrue(augmented["complete"])
            for runner in ("baseline", "reference", "trained"):
                self.assertEqual(augmented[runner + "_parity"]["matched_queries"], 8)
                self.assertTrue(augmented[runner + "_parity"]["all_selected_original_predictions_match"])
            direct = augmented["trained_vs_reference"]
            self.assertEqual(direct["reference_runner"], "reference")
            for variant in VARIANTS:
                self.assertEqual(set(augmented["metrics"][variant]), {"baseline", "robust", "reference", "trained"})
                self.assertEqual(sum(direct["transitions"][variant].values()), 8)
                self.assertTrue(direct["candidate_parity"][variant]["all_candidate_sets_identical"])
                expected = metric_deltas(augmented["metrics"][variant]["reference"],
                                         augmented["metrics"][variant]["trained"])
                self.assertEqual(direct["metric_deltas"][variant], expected)
            direct_rows = pd.read_csv(augmented_output / direct["row_changes_file"])
            self.assertEqual(len(direct_rows), 8 * 5)
            self.assertTrue(direct_rows.candidate_count_delta.eq(0).all())
            for variant in VARIANTS:
                rows = direct_rows.loc[direct_rows.variant.eq(variant)]
                self.assertEqual(direct["transitions"][variant]["corrected"], int(rows.status.eq("corrected").sum()))
                self.assertEqual(direct["transitions"][variant]["regressed"], int(rows.status.eq("regressed").sum()))
            self.assertTrue((augmented_output / direct["vessel_changes_file"]).is_file())
            self.assertIn("GT 결측 증강 자체의 효과", (augmented_output / "report.md").read_text("utf-8"))
            # Comparison loads must prove that the augmented model actually
            # declared source-mass weighting, rather than silently training
            # every replicated query at its old full weight.
            summary_path = models["augmented"] / "training_summary.json"
            saved_summary = json.loads(summary_path.read_text("utf-8"))
            train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
            catalog_hash = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
            for name, mutate in (
                ("missing", lambda x: x.pop("training_weighting")),
                ("policy", lambda x: x["training_weighting"].update(name="equal_replicas")),
                ("code", lambda x: x["training_weighting"].update(ranking_code_sha256="0" * 64)),
                ("groups", lambda x: x["training_weighting"].update(groups=0)),
                ("pairs", lambda x: x["training_weighting"].update(pairs=0)),
                ("mass", lambda x: x["training_weighting"].update(source_mass_total=0)),
                ("minimum", lambda x: x["training_weighting"].update(query_mass_min=0)),
                ("maximum", lambda x: x["training_weighting"].update(query_mass_max=2)),
                ("language_mass", lambda x: x["training_weighting"]["languages"]["hangul"].update(source_mass=0)),
                ("language_budget", lambda x: x["training_weighting"]["languages"]["hangul"].update(normalized_weight_share=.5)),
            ):
                changed = copy.deepcopy(saved_summary)
                mutate(changed)
                try:
                    summary_path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
                    with self.subTest(weighting=name), self.assertRaisesRegex(ValueError, "weighting proof"):
                        _load_model(models["augmented"], train_hash, catalog_hash)
                finally:
                    summary_path.write_text(json.dumps(saved_summary, ensure_ascii=False), encoding="utf-8")
            reference_path = models["robust"] / "training_summary.json"
            reference_summary = json.loads(reference_path.read_text("utf-8"))
            changed = copy.deepcopy(reference_summary)
            changed["training_weighting"] = saved_summary["training_weighting"]
            try:
                reference_path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "weighting requires augmentation"):
                    _load_model(models["robust"], train_hash, catalog_hash)
            finally:
                reference_path.write_text(json.dumps(reference_summary, ensure_ascii=False), encoding="utf-8")
            # Check the public-input provenance guard before any second live run.
            with train_path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            rejected = root / "rejected_changed_train"
            with self.assertRaisesRegex(ValueError, "content differs from model provenance"):
                compare_retrieval(models["legacy"], rejected, train_path=train_path,
                    catalog_path=catalog_path, experiment_model=models["robust"])
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
