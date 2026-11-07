"""Submission contract tests; real artifact timing is measured separately."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from submission import script


class SubmissionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = pd.DataFrame([
            dict(vessel_id="V01", vessel_name="ALPHA", vessel_type="Cargo", size=10.,
                 unit="m", deadweight_tonnage=100., gross_tonnage=90., flag_name="Korea"),
            dict(vessel_id="V02", vessel_name="BETA", vessel_type="Tanker", size=np.nan,
                 unit=None, deadweight_tonnage=200., gross_tonnage=180., flag_name="Japan"),
        ])
        self.queries = pd.DataFrame([
            dict(sample_id="0001", vessel_name="ALPHA", call_sign="CALL1", gross_tonnage=90.,
                 international_gross_tonnage=90., flag_name="Korea", vessel_type="Cargo",
                 arrival_year=2023, target_vessel_id="WRONG", extra_private_column="IGNORED"),
            dict(sample_id="0002", vessel_name="BETA", call_sign="CALL2", gross_tonnage=180.,
                 international_gross_tonnage=180., flag_name="Japan", vessel_type="Tanker",
                 arrival_year=2024, target_vessel_id="WRONG", extra_private_column="IGNORED"),
        ])
        self.query_path, self.catalog_path = self.root / "test.csv", self.root / "catalog.csv"
        self.output_path = self.root / "output" / "submission.csv"
        self.catalog.to_csv(self.catalog_path, index=False)
        self.received = []

    def matcher(self):
        def predict(row):
            self.received.append(dict(row))
            return "V01" if row["vessel_name"] == "ALPHA" else "V02"
        return SimpleNamespace(
            pipeline=SimpleNamespace(catalog=self.catalog.copy()), predict=predict,
            ranker=SimpleNamespace(feature_names=[f"f{i}" for i in range(86)],
                                   config=SimpleNamespace(n_jobs=6)),
            feature_transform={"name": "token_phonetic_v1"})

    def predict(self, queries=None, matcher=None):
        queries = self.queries if queries is None else queries
        queries.to_csv(self.query_path, index=False)
        with patch.object(script, "RankedMatcher", return_value=matcher or self.matcher()):
            metrics = script.run(self.query_path, self.catalog_path, self.root / "model", self.output_path)
        return pd.read_csv(self.output_path, dtype=str), metrics

    def test_exact_output_contract_leading_zero_ids_and_no_extra_inputs(self):
        output, metrics = self.predict()
        self.assertEqual(output.columns.tolist(), ["sample_id", "target_vessel_id"])
        self.assertEqual(output.to_dict("list"), {"sample_id": ["0001", "0002"],
                                                 "target_vessel_id": ["V01", "V02"]})
        self.assertTrue(all(set(row) == set(script.QUERY_COLUMNS) for row in self.received))
        self.assertEqual(metrics["rows"], 2)
        self.assertGreater(metrics["rows_per_second"], 0)
        self.assertEqual([path.name for path in self.output_path.parent.iterdir()], ["submission.csv"])

    def test_full_reordered_subset_and_singleton_outputs_agree(self):
        full, _ = self.predict()
        expected = full.set_index("sample_id").target_vessel_id.to_dict()
        for indices in ([1, 0], [0], [1]):
            result, _ = self.predict(self.queries.iloc[indices])
            self.assertEqual(result.set_index("sample_id").target_vessel_id.to_dict(),
                             {self.queries.iloc[i].sample_id: expected[self.queries.iloc[i].sample_id]
                              for i in indices})

    def test_empty_input_writes_header_only(self):
        result, metrics = self.predict(self.queries.iloc[:0])
        self.assertTrue(result.empty)
        self.assertEqual(self.output_path.read_text(), "sample_id,target_vessel_id\n")
        self.assertEqual(metrics["rows"], 0)
        self.assertIsNone(metrics["rows_per_second"])
        self.assertEqual(self.received, [])

    def test_catalog_reorder_and_numeric_serialization_are_accepted(self):
        reordered = self.catalog.iloc[::-1].copy()
        reordered["gross_tonnage"] = reordered.gross_tonnage.map(lambda v: f"{v:.3f}")
        reordered.to_csv(self.catalog_path, index=False)
        result, _ = self.predict()
        self.assertEqual(result.target_vessel_id.tolist(), ["V01", "V02"])

    def test_changed_catalog_ids_text_and_numbers_are_rejected(self):
        for column, replacement in (("vessel_id", "NEW"), ("vessel_name", "CHANGED"),
                                    ("gross_tonnage", 91.), ("flag_name", None)):
            with self.subTest(column=column):
                changed = self.catalog.copy()
                changed.loc[0, column] = replacement
                changed.to_csv(self.catalog_path, index=False)
                with self.assertRaisesRegex(ValueError, "catalog"):
                    self.predict()

    def test_duplicate_or_null_sample_ids_and_missing_columns_are_rejected(self):
        invalid = [self.queries.assign(sample_id=["same", "same"]),
                   self.queries.assign(sample_id=[None, "0002"]),
                   self.queries.drop(columns="call_sign")]
        for queries in invalid:
            with self.subTest(columns=queries.columns.tolist()), self.assertRaises(ValueError):
                self.predict(queries)

    def test_duplicate_catalog_ids_and_missing_columns_are_rejected(self):
        invalid = [self.catalog.assign(vessel_id=["same", "same"]),
                   self.catalog.assign(vessel_id=[None, "V02"]),
                   self.catalog.drop(columns="unit")]
        for catalog in invalid:
            with self.subTest(columns=catalog.columns.tolist()):
                catalog.to_csv(self.catalog_path, index=False)
                with self.assertRaises(ValueError):
                    self.predict()

    def test_prediction_must_be_in_official_catalog(self):
        for value in (None, np.nan, "UNKNOWN", 123):
            matcher = self.matcher()
            matcher.predict = lambda row, value=value: value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "absent"):
                self.predict(matcher=matcher)
        self.assertFalse(self.output_path.exists())

    def test_rejects_other_model_schema_and_thread_configuration(self):
        for change in ("feature_count", "transform", "tonnage", "threads"):
            matcher = self.matcher()
            if change == "feature_count":
                matcher.ranker.feature_names.append("extra")
            elif change == "transform":
                matcher.feature_transform = {"name": "token_phonetic_ch_v1"}
            elif change == "tonnage":
                matcher.ranker.feature_names[0] = "retrieval_tonnage_rank"
            else:
                matcher.ranker.config.n_jobs = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.predict(matcher=matcher)

    def test_default_paths_and_logging_contain_only_aggregate_timings(self):
        metrics = {"rows": 2, "run_seconds": 1.0}
        stream = io.StringIO()
        with patch.object(script, "ROOT", self.root), patch.object(script, "run", return_value=metrics) as run:
            with redirect_stdout(stream):
                script.main()
        run.assert_called_once_with(self.root / "data" / "test.csv", self.root / "data" / "vessel_catalog.csv",
                                    self.root / "model" / "ranker", self.root / "output" / "submission.csv")
        logged = json.loads(stream.getvalue())
        self.assertEqual(set(logged), {"rows", "run_seconds", "script_seconds_including_imports"})

    def package(self):
        names = ["script.py", "requirements.txt", "model/portmatch/__init__.py",
                 "model/portmatch/ranker_commands.py", "model/ranker/model.txt",
                 "model/ranker/ranker.json", "model/ranker/index.joblib",
                 "model/ranker/training_summary.json"]
        files = {}
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"frozen artifact")
            files[name] = {"size_bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest = {"schema_version": 1, "baseline": "token_phonetic_v1_86", "feature_count": 86,
                    "files": files}
        self.save_manifest(manifest)
        return manifest

    def save_manifest(self, manifest):
        (self.root / "model" / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_manifest_accepts_complete_frozen_payload_and_rejects_changed_bytes(self):
        self.package()
        script.verify_package(self.root)
        (self.root / "model" / "ranker" / "model.txt").write_bytes(b"changed content")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            script.verify_package(self.root)

    def test_manifest_rejects_missing_files_wrong_schema_and_extra_code(self):
        manifest = self.package()
        manifest["feature_count"] = 93
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "manifest"):
            script.verify_package(self.root)
        manifest = self.package()
        del manifest["files"]["script.py"]
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "missing"):
            script.verify_package(self.root)
        self.package()
        (self.root / "model" / "portmatch" / "unexpected.py").write_text("pass", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unmanifested"):
            script.verify_package(self.root)

    def test_manifest_rejects_path_traversal(self):
        manifest = self.package()
        manifest["files"]["../outside.txt"] = {"size_bytes": 0, "sha256": "ignored"}
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "escapes"):
            script.verify_package(self.root)

    def test_progress_contains_only_aggregate_row_counts(self):
        queries = pd.concat([self.queries.iloc[:1]] * 100, ignore_index=True)
        queries["sample_id"] = [f"S{i:03d}" for i in range(100)]
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.predict(queries)
        self.assertEqual(json.loads(stream.getvalue()), {"predicted_rows": 100, "total_rows": 100})


if __name__ == "__main__":
    unittest.main()
