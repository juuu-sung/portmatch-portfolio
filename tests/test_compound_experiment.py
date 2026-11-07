import argparse
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from portmatch.compound_protocol import (EVAL_RECIPES, TRAIN_RECIPES, delete_letter,
    evaluation_variant, input_key, protocol, swap_letters, training_variant)
from portmatch.compound_mining import expanded_masses, mined_positions, teacher_features
from portmatch.text import has_hangul


class CompoundProtocolTests(unittest.TestCase):
    def row(self, name="오션 파이오니어 II 20"):
        return dict(sample_id="A", vessel_name=name, call_sign="ABC", gross_tonnage=1000.,
                    international_gross_tonnage=1100., flag_name="Korea", vessel_type="Cargo", arrival_year=2025)

    def test_training_has_no_labels_ids_or_batch_dependencies(self):
        row = self.row()
        reference = [training_variant(row, view) for view in ("vessel", "row")]
        row.update(sample_id="other", target_vessel_id="forbidden", arrival_year=1900)
        for view, (recipe, expected) in zip(("vessel", "row"), reference):
            actual_recipe, actual = training_variant(row, view)
            self.assertEqual(recipe, actual_recipe)
            self.assertEqual(input_key(expected), input_key(actual))
            self.assertNotIn("target_vessel_id", actual)

    def test_every_variant_preserves_numeric_identity_language_and_input(self):
        for name in ("오션 파이오니어 II 20", "Ocean Pioneer IV 1 0", "A가BC 12", "AB12", "X", "시더 7"):
            row = self.row(name)
            initial = input_key(row)
            for variant in EVAL_RECIPES:
                result = evaluation_variant(row, variant)
                self.assertEqual(re.findall(r"\d+", result["vessel_name"]), re.findall(r"\d+", name))
                self.assertEqual(has_hangul(result["vessel_name"]), has_hangul(name))
                if " II " in name:
                    self.assertIn(" II ", result["vessel_name"])
                self.assertEqual(input_key(row), initial)

    def test_corruption_is_effective_and_unknown_families_are_held_out(self):
        self.assertNotEqual(delete_letter("오션 파이오니어"), "오션 파이오니어")
        self.assertNotEqual(swap_letters("Ocean Pioneer"), "Ocean Pioneer")
        self.assertFalse(any("swap" in r for r in TRAIN_RECIPES))
        self.assertEqual(protocol()["training_gt_factors"], [.85, 1.15])
        changed = evaluation_variant(self.row(), "unseen_swap_dual_gt_x0_70")
        self.assertAlmostEqual(changed["gross_tonnage"], 700.)

    def test_metadata_and_gt_recipes_change_only_requested_fields(self):
        row = self.row()
        masked = evaluation_variant(row, "flag_type_missing")
        self.assertTrue(pd.isna(masked["flag_name"]))
        self.assertEqual(masked["gross_tonnage"], row["gross_tonnage"])
        gt = evaluation_variant(row, "international_gt_x1_25")
        self.assertEqual(gt["gross_tonnage"], 1000.)
        self.assertEqual(gt["international_gross_tonnage"], 1375.)
        with self.assertRaises(ValueError):
            evaluation_variant(row, "unknown")

    def test_teacher_excludes_all_fitted_feature_families(self):
        names = ["name_ratio", "retrieval_phonetic_score", "flag_lift_log", "type_pair_support_log", "best_tonnage_gap"]
        self.assertEqual(teacher_features(names), ["name_ratio", "best_tonnage_gap"])

    def test_mining_keeps_positive_budget_and_finds_new_hard_negative(self):
        ids = np.array([10, 11, 12, 13, 14, 15])
        heuristic = np.array([0, 1, 2, 3])
        selected = mined_positions(ids, 10, heuristic, np.array([0., 1., 2., 3., 4., 5.]), max_mined=2)
        self.assertEqual(len(selected), len(heuristic))
        self.assertIn(0, selected); self.assertIn(4, selected); self.assertIn(5, selected)
        with self.assertRaises(ValueError):
            mined_positions(ids, 99, heuristic, np.ones(6))
        with self.assertRaises(ValueError):
            mined_positions(ids, 10, heuristic, np.full(6, np.nan))

    def test_source_mass_original_is_retained_and_missing_recipes_redistribute(self):
        sources = ["A", "A", "A", "A", "B", "C", "C"]
        variants = ["original", "original", "both_gt_missing", "both_gt_missing", "original", "original", "both_gt_missing"]
        old = [.25, .25, .25, .25, 1., .5, .5]
        base, extra = expanded_masses(sources, variants, old, ["A", "A", "B"])
        np.testing.assert_allclose(base[:4], [.25, .25, .125, .125])
        np.testing.assert_allclose(extra, [.125, .125, .25])
        self.assertEqual(base[4], .75)
        np.testing.assert_allclose(base[5:], [.5, .5])
        self.assertAlmostEqual(sum(base) + sum(extra), 3.)
        with self.assertRaises(ValueError):
            expanded_masses(["A"], ["original"], [1.], ["unknown"])


class CompoundIntegrationTests(unittest.TestCase):
    def test_end_to_end_disjoint_teachers_preservation_shared_comparison_and_reload(self):
        from portmatch.compound_experiment import NEW_MODULES, read_json, run_fold, sha, write_json, report, prepare_teacher_pool
        from portmatch.data import read_catalog, read_queries
        from portmatch.gt_augmentation import GTMissingAugmentationConfig
        from portmatch.name_feature_cache import transform_cache
        from portmatch.ranker_commands import add_training_parsers, run_training_command, feature_code_hash
        from portmatch.robust_retrieval import RobustRetrievalConfig
        from portmatch.training_data import prepare_training_data
        spec = importlib.util.spec_from_file_location("compound_fixture", Path(__file__).with_name("test_robust_training.py"))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        train, catalog = module.fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_path, catalog_path = root / "train.csv", root / "catalog.csv"
            train.to_csv(train_path, index=False, encoding="utf-8-sig")
            catalog.to_csv(catalog_path, index=False, encoding="utf-8-sig")
            train, catalog = read_queries(train_path, labeled=True), read_catalog(catalog_path)
            source, cache, baseline = root / "source", root / "cache", root / "baseline"
            prepare_training_data(train, catalog, source, outer_folds=5, outer_fold=0, inner_folds=3,
                retrieval_config=RobustRetrievalConfig(), augmentation_config=GTMissingAugmentationConfig(),
                provenance={"train_sha256": sha(train_path), "catalog_sha256": sha(catalog_path),
                            "feature_code_sha256": feature_code_hash()})
            transform_cache(source, cache, train_path, catalog_path)
            parser = argparse.ArgumentParser(); add_training_parsers(parser.add_subparsers(dest="command", required=True))
            run_training_command(parser.parse_args(["train-ranker", "--data", str(cache), "--output", str(baseline),
                                                   "--exclude-features", "retrieval_tonnage_rank"]))
            output = root / "experiment"; output.mkdir()
            protected = {str(p): sha(p) for folder in (baseline, cache) for p in folder.rglob("*") if p.is_file()}
            write_json(output / "protocol.json", {"version": protocol()["version"], "protocol": protocol(),
                "implementation": {name: sha(Path(__file__).parents[1] / "portmatch" / name) for name in NEW_MODULES},
                "planned_folds": [0], "protected_files": protected,
                "data": {name: {"path": str(p), "sha256": sha(p)} for name, p in
                         (("train.csv", train_path), ("vessel_catalog.csv", catalog_path))},
                "baselines": {"0": {"directory": str(baseline), "cache": str(cache),
                    "files": {name: sha(baseline / name) for name in
                             ("model.txt", "ranker.json", "index.joblib", "training_summary.json", "validation_predictions.csv")}}}})
            prepare_teacher_pool(output)
            run_fold(output, 0)
            comparison = report(output)
            self.assertEqual(comparison["complete_for_folds"], [0])
            self.assertFalse(comparison["full_five_fold"])
            self.assertEqual(comparison["protected_files_verified"], len(protected))
            for inner in range(3):
                proof = read_json(output / f"fold0/teachers/{inner}/crossfit.json")
                self.assertFalse(set(proof["training_vessel_ids"]) & set(proof["excluded_vessel_ids"]))
                self.assertEqual(len(proof["feature_names"]), 54)
            logs = pd.read_csv(output / "fold0/augmentation_log.csv")
            self.assertTrue(set(logs.status).issubset({"stored", "skipped_noop", "skipped_missing_target"}))
            result = read_json(output / "fold0/evaluation.json")
            self.assertTrue(result["baseline_original_parity"])
            self.assertTrue(result["same_candidate_and_feature_arrays"])
            analysis_spec = importlib.util.spec_from_file_location("compound_analysis", Path(__file__).parents[1] / "scripts/analyze_compound.py")
            analysis_module = importlib.util.module_from_spec(analysis_spec)
            analysis_spec.loader.exec_module(analysis_module)
            analysis_module.analyze(output, replicates=100)
            self.assertTrue(read_json(output / "analysis_audit.json")["old_models_and_submissions_unchanged"])
            self.assertEqual(len(read_json(output / "confidence_intervals.json")["comparisons"]), 36)
            runner_spec = importlib.util.spec_from_file_location("compound_runner", Path(__file__).parents[1] / "scripts/run_compound_study.py")
            runner = importlib.util.module_from_spec(runner_spec); runner_spec.loader.exec_module(runner)
            self.assertEqual(runner.status(output)["folds"], {"0": "complete"})
            (output / "fold1").mkdir()
            partial = output / "fold1/partial.txt"
            partial.write_text("keep this partial result", encoding="utf-8")
            archived = runner.archive_interrupted(output, ["fold1"])
            self.assertEqual((archived / "fold1/partial.txt").read_text(), "keep this partial result")
            self.assertFalse((output / "fold1").exists())
            self.assertTrue((output / "fold0/status.json").exists())
            with self.assertRaises(ValueError):
                runner.archive_interrupted(output, ["../baseline"])
            model_before = sha(output / "fold0/expanded/model.txt")
            with runner.runner_lock(output):
                self.assertEqual(runner.run(output, workers=1), 0)
            self.assertEqual(sha(output / "fold0/expanded/model.txt"), model_before)
            with self.assertRaises(FileExistsError):
                run_fold(output, 0)


if __name__ == "__main__":
    unittest.main()
