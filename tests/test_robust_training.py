"""Synthetic end-to-end checks for optional, fold-safe candidate expansion."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.data import read_catalog, read_queries
from portmatch.name_feature_cache import transform_cache
from portmatch.ranker_commands import (
    RankedMatcher, _hash_file, add_training_parsers, feature_code_hash,
    run_training_command,
)
from portmatch.ranker_comparison import compare_rankers
from portmatch.robust_retrieval import (
    RobustMatchingPipeline, RobustRetrievalConfig, retrieval_spec,
)
from portmatch.sampling import SamplingConfig
from portmatch.training_data import prepare_training_data, read_metadata


def fixture():
    names = [("Ocean Cedar", "오션 시더"), ("Ocean Meadow", "오션 메도"),
             ("Coastal Light", "코스탈 라이트"), ("Coastal River", "코스탈 리버"),
             ("Blue Pearl", "블루 펄"), ("Blue Peak", "블루 피크"),
             ("Silver Knight", "실버 나이트"), ("Silver Kite", "실버 카이트"),
             ("Cedar No 7", "시더 7"), ("Cedar No 8", "시더 8"),
             ("Jeong Won", "정원호"), ("Jeong Wol", "정월호")]
    catalog = pd.DataFrame([
        dict(vessel_id=f"V{i:02d}", vessel_name=name[0], gross_tonnage=1000 + 100 * i,
             flag_name="Korea", vessel_type="Cargo", size=10., unit="m", deadweight_tonnage=1200.)
        for i, name in enumerate(names)])
    train = pd.DataFrame([
        dict(sample_id=f"S{i:02d}_{arrival}", target_vessel_id=f"V{i:02d}",
             vessel_name=name[arrival], call_sign=f"C{i:02d}", gross_tonnage=1000 + 100 * i,
             international_gross_tonnage=1000 + 100 * i, flag_name="한국",
             vessel_type="화물선", arrival_year=2020 + arrival)
        for i, name in enumerate(names) for arrival in range(2)])
    return train, catalog


class RobustTrainingPartitionTests(unittest.TestCase):
    def test_every_inner_and_outer_pipeline_excludes_held_out_labels(self):
        train, catalog = fixture()
        fits = []

        original_init = RobustMatchingPipeline.__init__

        def build(instance, fitting, source_catalog, config):
            fits.append((set(fitting.sample_id), set(fitting.target_vessel_id)))
            original_init(instance, fitting, source_catalog, config)

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(RobustMatchingPipeline, "__init__", new=build):
                metadata = prepare_training_data(
                    train, catalog, directory, outer_folds=3, inner_folds=2, seed=42,
                    retrieval_config=RobustRetrievalConfig(),
                    sampling_config=SamplingConfig(hard_negatives=2, random_negatives=1, seed=42))
            partitions = pd.read_csv(Path(directory) / "partitions.csv")
            outer = partitions[partitions.role == "valid"]
            training = partitions[partitions.role == "train"]
            self.assertEqual(len(fits), 5)
            for fit_ids, targets in fits:
                self.assertFalse(fit_ids & set(outer.sample_id))
                self.assertFalse(targets & set(outer.target_vessel_id))
            for position, view in enumerate(("vessel", "row")):
                for fold in range(2):
                    held = training[training["inner_" + view] == fold]
                    fit_ids, targets = fits[position * 2 + fold]
                    self.assertFalse(fit_ids & set(held.sample_id))
                    if view == "vessel":
                        self.assertFalse(targets & set(held.target_vessel_id))
            self.assertEqual(fits[-1][0], set(training.sample_id))
            self.assertEqual(metadata["retrieval_extension"], retrieval_spec(RobustRetrievalConfig()))
            self.assertEqual(metadata["retrieval_index_sha256"], _hash_file(Path(directory) / "index.joblib"))


@unittest.skipUnless(importlib.util.find_spec("lightgbm") is not None, "LightGBM is not installed")
class RobustTrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.parser = argparse.ArgumentParser()
        add_training_parsers(cls.parser.add_subparsers(dest="command", required=True))
        train, catalog = fixture()
        train.to_csv(cls.root / "train.csv", index=False, encoding="utf-8-sig")
        catalog.to_csv(cls.root / "catalog.csv", index=False, encoding="utf-8-sig")
        cls.train, cls.catalog = read_queries(cls.root / "train.csv", labeled=True), read_catalog(cls.root / "catalog.csv")
        cls.models, cls.sources, cls.derived = {}, {}, {}
        for profile in ("legacy", "robust"):
            source, derived, model = [cls.root / f"{profile}_{suffix}" for suffix in ("source", "derived", "model")]
            cls.run_cli(["prepare-ranker", "--train", str(cls.root / "train.csv"),
                         "--catalog", str(cls.root / "catalog.csv"), "--output", str(source),
                         "--retrieval-profile", profile, "--folds", "3", "--inner-folds", "2",
                         "--seed", "42", "--max-train-queries", "4", "--max-valid-queries", "4"])
            transform_cache(source, derived, cls.root / "train.csv", cls.root / "catalog.csv")
            cls.run_cli(["train-ranker", "--data", str(derived), "--output", str(model),
                         "--rounds", "3", "--threads", "1", "--num-leaves", "4",
                         "--min-child-samples", "1", "--seed", "42",
                         "--exclude-features", "retrieval_tonnage_rank"])
            cls.sources[profile], cls.derived[profile], cls.models[profile] = source, derived, model

    @classmethod
    def run_cli(cls, arguments):
        run_training_command(cls.parser.parse_args(arguments))

    def test_default_is_legacy_and_new_profile_options_are_explicit(self):
        args = self.parser.parse_args(["prepare-ranker", "--output", "unused"])
        self.assertEqual(args.retrieval_profile, "legacy")
        args = self.parser.parse_args(["prepare-ranker", "--output", "unused", "--retrieval-profile", "hangul",
                                      "--context-top-k", "12", "--vowel-top-k", "13", "--context-edit-top-k", "14"])
        self.assertEqual((args.context_top_k, args.vowel_top_k, args.context_edit_top_k), (12, 13, 14))
        legacy = read_metadata(self.sources["legacy"])
        self.assertNotIn("retrieval_extension", legacy)
        self.assertEqual(legacy["provenance"]["feature_code_sha256"], feature_code_hash())

    def test_name_transform_preserves_extension_and_trained_schema_is_86(self):
        source = read_metadata(self.sources["robust"])
        derived = read_metadata(self.derived["robust"])
        self.assertEqual(len(source["feature_names"]), 70)
        self.assertEqual(len(derived["feature_names"]), 87)
        for field in ("retrieval", "retrieval_extension", "retrieval_index_sha256"):
            self.assertEqual(derived[field], source[field])
        self.assertEqual(_hash_file(self.derived["robust"] / "index.joblib"), source["retrieval_index_sha256"])
        matcher = RankedMatcher(self.models["robust"])
        self.assertIsInstance(matcher.pipeline, RobustMatchingPipeline)
        self.assertEqual(asdict(matcher.pipeline.retriever.config), derived["retrieval"])
        self.assertEqual(len(matcher.ranker.feature_names), 86)
        self.assertNotIn("retrieval_tonnage_rank", matcher.ranker.feature_names)

    def test_restored_predictions_match_cache_and_remain_row_independent(self):
        matcher = RankedMatcher(self.models["robust"])
        saved = pd.read_csv(self.models["robust"] / "validation_predictions.csv")
        rows = self.train.set_index("sample_id")
        expected = {}
        for record in saved.to_dict("records"):
            row = {"sample_id": record["sample_id"], **rows.loc[record["sample_id"]].to_dict()}
            ranked = matcher.rank(row)
            self.assertEqual(len(ranked), record["candidate_count"])
            self.assertEqual(ranked.vessel_id.iloc[0], record["predicted_vessel_id"])
            self.assertAlmostEqual(float(ranked.score.iloc[0]), record["top_score"], places=12)
            expected[row["sample_id"]] = ranked
        for sample_id in saved.sample_id.tolist()[::-1]:
            row = {"sample_id": sample_id, **rows.loc[sample_id].to_dict(), "target_vessel_id": "IGNORED"}
            pd.testing.assert_frame_equal(matcher.rank(row), expected[sample_id], check_exact=True)
            row["gross_tonnage"] = row["international_gross_tonnage"] = np.nan
            ranked = matcher.rank(row)
            pd.testing.assert_frame_equal(RankedMatcher(self.models["robust"]).rank(row), ranked, check_exact=True)

    def test_wrong_extension_config_and_index_are_rejected_before_prediction(self):
        for mutation in ("extension", "missing", "config", "index", "missing_hash", "legacy_index"):
            with self.subTest(mutation=mutation):
                destination = self.root / ("invalid_" + mutation)
                shutil.copytree(self.models["robust"], destination)
                path = destination / "training_summary.json"
                summary = json.loads(path.read_text("utf-8"))
                dataset = summary["dataset"]
                if mutation == "extension":
                    dataset["retrieval_extension"]["code_sha256"] = "0" * 64
                elif mutation == "missing":
                    dataset.pop("retrieval_extension")
                elif mutation == "config":
                    dataset["retrieval"]["context_top_k"] += 1
                elif mutation == "index":
                    with (destination / "index.joblib").open("ab") as stream:
                        stream.write(b"changed")
                elif mutation == "missing_hash":
                    dataset.pop("retrieval_index_sha256")
                else:
                    shutil.copyfile(self.models["legacy"] / "index.joblib", destination / "index.joblib")
                    dataset["retrieval_index_sha256"] = _hash_file(destination / "index.joblib")
                path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(ValueError):
                    RankedMatcher(destination)

    def test_feature_only_comparison_still_rejects_changed_retrieval(self):
        with self.assertRaisesRegex(ValueError, "retrieval|configur|dataset|candidate|cache"):
            compare_rankers([self.models["legacy"]], [self.models["robust"]], self.root / "invalid_comparison")

    def test_cache_reader_rejects_missing_retrieval_version(self):
        invalid = self.root / "invalid_cache"
        invalid.mkdir()
        metadata = read_metadata(self.sources["robust"])
        metadata.pop("retrieval_extension")
        (invalid / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "retrieval_extension"):
            read_metadata(invalid)


if __name__ == "__main__":
    unittest.main()
