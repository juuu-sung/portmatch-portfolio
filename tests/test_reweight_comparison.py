import argparse
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from portmatch.retrieval_comparison import (
    VARIANTS, _direct_comparison, _load_model, _validate_control_pair,
    add_retrieval_comparison_parser, compare_retrieval, metric_deltas,
    run_comparison, summarize_predictions,
)


def paired_fixture():
    rows = []
    for variant in ("original", "both_gt_missing"):
        for sample, (before, after) in enumerate(((False, True), (True, False),
                                                  (False, False), (True, True))):
            for model, correct in (("control", before), ("trained", after)):
                rows.append({"sample_id": f"S{sample}", "target_vessel_id": f"V{sample}",
                    "variant": variant, "model": model, "vessel_name": f"Name {sample}",
                    "has_hangul": sample % 2 == 0,
                    "predicted_vessel_id": f"V{sample}" if correct else "OTHER",
                    "candidate_found": True, "target_rank": 1 if correct else 2,
                    "candidate_count": 3, "candidate_ids_sha256": "a" * 64,
                    "retrieval_seconds": .1, "feature_seconds": .2,
                    "rank_seconds": .3, "total_seconds": .6})
    frame = pd.DataFrame(rows)
    variants = ("original", "both_gt_missing")
    metrics = {variant: {model: summarize_predictions(
        frame.loc[frame.variant.eq(variant) & frame.model.eq(model)])
        for model in ("control", "trained")} for variant in variants}
    return frame, metrics, variants


class ReweightComparisonTests(unittest.TestCase):
    def test_control_parser_and_command_forwarding(self):
        parser = argparse.ArgumentParser()
        add_retrieval_comparison_parser(parser.add_subparsers(dest="command"))
        base = ["compare-retrieval", "--baseline", "old86", "--output", "out"]
        self.assertIsNone(parser.parse_args(base).control_model)
        args = parser.parse_args(base + ["--profiles", "robust", "--reference-model", "unaugmented",
            "--control-model", "mass05", "--experiment-model", "mass075"])
        with patch("portmatch.retrieval_comparison.compare_retrieval", return_value={"coverage": {}}) as compare:
            with redirect_stdout(io.StringIO()):
                run_comparison(args)
        self.assertEqual(compare.call_args.kwargs["control_model"], "mass05")
        self.assertEqual(compare.call_args.kwargs["reference_model"], "unaugmented")

    def test_control_requires_both_other_models_before_reading_input(self):
        for kwargs in ({}, {"experiment_model": "mass075"}, {"reference_model": "unaugmented"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "requires"):
                compare_retrieval("missing", "missing_out", control_model="mass05", **kwargs)

    def test_direct_control_metrics_and_transitions_keep_same_input_pairing(self):
        rows, metrics, variants = paired_fixture()
        result, changes, vessels = _direct_comparison(rows, metrics, variants, "control")
        self.assertEqual(len(changes), 8)
        self.assertEqual(result["row_changes_file"], "trained_vs_control_row_changes.csv")
        for variant in variants:
            self.assertEqual(result["transitions"][variant], {
                "corrected": 1, "regressed": 1, "still_wrong": 1, "unchanged_correct": 1})
            self.assertEqual(result["metric_deltas"][variant],
                             metric_deltas(metrics[variant]["control"], metrics[variant]["trained"]))
            self.assertEqual(result["candidate_parity"][variant], {
                "queries": 4, "all_candidate_sets_identical": True})
        self.assertTrue(changes.candidate_count_delta.eq(0).all())
        self.assertEqual(int(vessels.corrected.sum()), 2)
        self.assertEqual(int(vessels.regressed.sum()), 2)

    def test_direct_control_rejects_candidate_or_identity_drift(self):
        rows, metrics, variants = paired_fixture()
        for field, value in (("candidate_ids_sha256", "b" * 64),
                             ("candidate_count", 4), ("candidate_found", False),
                             ("target_vessel_id", "UNRELATED")):
            changed = rows.copy()
            index = changed.loc[changed.model.eq("trained")].index[0]
            changed.loc[index, field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "candidates|identity"):
                _direct_comparison(changed, metrics, variants, "control")
        changed = rows.drop(rows.loc[rows.model.eq("trained")].index[0])
        with self.assertRaisesRegex(ValueError, "identical variant/sample"):
            _direct_comparison(changed, metrics, variants, "control")


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class ReweightComparisonIntegrationTests(unittest.TestCase):
    def test_reused_cache_has_exact_control_lineage_and_five_runner_comparison(self):
        # This miniature pipeline verifies real saved model/cache contracts;
        # each model has three trees and only 24 synthetic training rows.
        spec = importlib.util.spec_from_file_location(
            "_reweight_comparison_fixture", Path(__file__).with_name("test_robust_training.py"))
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        from portmatch.gt_reweight_cache import reweight_cache
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
            models, caches = {}, {}
            for profile in ("legacy", "robust", "augmented", "reweighted"):
                source, derived, model = [root / f"{profile}_{suffix}"
                                          for suffix in ("source", "derived", "model")]
                if profile == "reweighted":
                    reweight_cache(caches["augmented"], derived, original_mass=.75)
                else:
                    args = ["prepare-ranker", "--train", str(train_path), "--catalog", str(catalog_path),
                        "--output", str(source), "--retrieval-profile", "legacy" if profile == "legacy" else "robust",
                        "--folds", "3", "--inner-folds", "2", "--seed", "42"]
                    if profile == "augmented":
                        args += ["--gt-augmentation", "missing", "--original-mass", "0.5"]
                    run_training_command(parser.parse_args(args))
                    transform_cache(source, derived, train_path, catalog_path)
                run_training_command(parser.parse_args([
                    "train-ranker", "--data", str(derived), "--output", str(model),
                    "--rounds", "3", "--threads", "1", "--num-leaves", "4",
                    "--min-child-samples", "1", "--seed", "42",
                    "--exclude-features", "retrieval_tonnage_rank"]))
                models[profile], caches[profile] = model, derived

            train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
            catalog_hash = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
            control, control_summary, _ = _load_model(models["augmented"], train_hash, catalog_hash)
            experiment, experiment_summary, _ = _load_model(models["reweighted"], train_hash, catalog_hash)
            _validate_control_pair(control, control_summary, experiment, experiment_summary)
            self.assertEqual(experiment_summary["dataset"]["training_reweighting"]["source_metadata_sha256"],
                             control_summary["cache_metadata_sha256"])
            self.assertEqual(control_summary["dataset"]["training_augmentation"]["config"]["original_mass"], .5)
            self.assertEqual(experiment_summary["dataset"]["training_augmentation"]["config"]["original_mass"], .75)

            for field, value in (("outer_fold", 1), ("sampling", {"changed": True}),
                                 ("seed", 99), ("sampled", True)):
                changed = copy.deepcopy(experiment_summary)
                changed["dataset"][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "controlled setting"):
                    _validate_control_pair(control, control_summary, experiment, changed)
            for name, mutate, message in (
                ("ranker", lambda x: x["ranker"].update(n_estimators=999), "identical ranker"),
                ("retrieval", lambda x: x["dataset"]["retrieval_extension"]["config"].update(profile="hangul"), "controlled metadata"),
                ("pairs", lambda x: x["dataset"]["counts"].update(train_pairs=999), "controlled metadata"),
                ("lineage", lambda x: x["dataset"]["training_reweighting"].update(source_metadata_sha256="0" * 64), "not derived"),
                ("missing_proof", lambda x: x["dataset"].pop("training_reweighting"), "missing.*declaration"),
            ):
                changed = copy.deepcopy(experiment_summary)
                mutate(changed)
                with self.subTest(setting=name), self.assertRaisesRegex(ValueError, message):
                    _validate_control_pair(control, control_summary, experiment, changed)
            changed_control = copy.deepcopy(control_summary)
            changed_control.pop("cache_metadata_sha256")
            with self.assertRaisesRegex(ValueError, "source cache metadata hash"):
                _validate_control_pair(control, changed_control, experiment, experiment_summary)

            arguments = dict(train_path=train_path, catalog_path=catalog_path, profiles=("robust",),
                             experiment_model=models["reweighted"], reference_model=models["robust"])
            with self.assertRaisesRegex(ValueError, "requires its original augmented control"):
                compare_retrieval(models["legacy"], root / "missing_control", **arguments)
            self.assertFalse((root / "missing_control").exists())
            for label, wrong_control, message in (
                ("unaugmented", models["robust"], "requires an augmented"),
                ("chained", models["reweighted"], "reweight chains"),
            ):
                rejected = root / label
                with self.subTest(control=label), self.assertRaisesRegex(ValueError, message):
                    compare_retrieval(models["legacy"], rejected, control_model=wrong_control, **arguments)
                self.assertFalse(rejected.exists())

            output = root / "comparison"
            summary = compare_retrieval(models["legacy"], output, control_model=models["augmented"], **arguments)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["coverage"]["queries"], 8)
            self.assertFalse(summary["coverage"]["diagnostic_sample"])
            for name in ("baseline", "reference", "control", "trained"):
                self.assertEqual(summary[name + "_parity"]["matched_queries"], 8)
                self.assertTrue(summary[name + "_parity"]["all_selected_original_predictions_match"])
            for name in ("reference", "control"):
                direct = summary["trained_vs_" + name]
                rows = pd.read_csv(output / direct["row_changes_file"])
                self.assertEqual(len(rows), 8 * len(VARIANTS))
                self.assertTrue(rows.candidate_count_delta.eq(0).all())
                self.assertTrue((output / direct["vessel_changes_file"]).is_file())
                for variant in VARIANTS:
                    self.assertEqual(set(summary["metrics"][variant]),
                                     {"baseline", "robust", "reference", "control", "trained"})
                    self.assertTrue(direct["candidate_parity"][variant]["all_candidate_sets_identical"])
                    self.assertEqual(direct["metric_deltas"][variant], metric_deltas(
                        summary["metrics"][variant][name], summary["metrics"][variant]["trained"]))
                    part = rows.loc[rows.variant.eq(variant)]
                    for status, count in direct["transitions"][variant].items():
                        self.assertEqual(count, int(part.status.eq(status).sum()))
            self.assertEqual(len(pd.read_csv(output / "predictions.csv")), 8 * len(VARIANTS) * 5)
            self.assertEqual(json.loads((output / "summary.json").read_text("utf-8")), summary)
            report = (output / "report.md").read_text("utf-8")
            self.assertIn("원본 질량 재가중치 비교", report)
            self.assertIn("질량 0.5", report)
            self.assertIn("0.75", report)


if __name__ == "__main__":
    unittest.main()
