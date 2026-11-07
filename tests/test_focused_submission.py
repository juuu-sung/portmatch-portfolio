import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from portmatch.compound_experiment import sha
from portmatch.focused_runtime import FocusedMatcher
from portmatch.focused_submission_package import build, write_archive
from portmatch.joint_search_experiment import query_arrays
from portmatch.tonnage_repair_experiment import TonnageRepairMatcher
from portmatch.tonnage_repair_protocol import EVAL_RECIPES, evaluation
from scripts.verify_focused_submission import runtime_module
from scripts.verify_submission import extract_archive, run_case


class FocusedSubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("focused_repair_fixture", Path(__file__).with_name("test_tonnage_repair.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.fixture = module.RepairIntegrationTests
        cls.fixture.setUpClass()
        cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.root = cls.fixture.root
        cls.study = cls.fixture.out
        cls.archive = cls.root / "focused_submission/submit.zip"
        before = {str(p): sha(p) for p in cls.fixture.fixture.baseline.iterdir() if p.is_file()}
        with patch("portmatch.focused_submission_package.protected_snapshot", return_value=before):
            cls.report = build(cls.study, cls.archive)
        cls.extracted = cls.root / "extracted"
        extract_archive(cls.archive, cls.extracted)
        cls.matcher = FocusedMatcher(cls.extracted / "model")
        cls.reference = TonnageRepairMatcher(cls.study, 0, "focused_search")
        cls.entry = runtime_module(cls.extracted / "script.py")

    def test_native_scores_and_independence_all_recipes(self):
        for source in self.fixture.fixture.train.iloc[:3].to_dict("records"):
            for variant in EVAL_RECIPES:
                row = evaluation(source, variant)
                _, ids, _, matrix, _, _, _ = query_arrays(self.reference.baseline, self.reference.expansion, row)
                scores = self.reference.ranker.predict(matrix, self.reference.ranker.feature_names)
                actual_ids, actual_scores = self.matcher.candidate_scores(row)
                np.testing.assert_array_equal(actual_ids, ids)
                np.testing.assert_array_equal(actual_scores, scores)
                a, b = self.matcher.candidate_scores({**row, "sample_id": "OTHER", "arrival_year": 1900,
                                                    "target_vessel_id": "OTHER"})
                np.testing.assert_array_equal(a, ids)
                np.testing.assert_array_equal(b, scores)

    def test_isolated_process_loads_only_packaged_dependencies(self):
        root = self.root / "isolated_checks"
        root.mkdir()
        rows = self.fixture.fixture.train.head(4)
        result, predictions = run_case(self.archive, root, "isolated", rows,
            self.root / "catalog.csv", Path(__import__("sys").executable), 60)
        self.assertTrue(result["input_files_unchanged"])
        self.assertEqual(predictions, {r["sample_id"]: self.reference.predict(r) for r in rows.to_dict("records")})
        self.entry.verify_package(root / "isolated")

    def test_manifest_rejects_tampering_and_unlisted_code(self):
        self.entry.verify_package(self.extracted)
        path = self.extracted / "model/ranker/model.txt"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                self.entry.verify_package(self.extracted)
            with self.assertRaises(ValueError):
                FocusedMatcher(self.extracted / "model")
        finally:
            path.write_bytes(original)
        unlisted = self.extracted / "model/portmatch/unlisted.py"
        try:
            unlisted.write_text("raise RuntimeError('unexpected source')", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.entry.verify_package(self.extracted)
        finally:
            unlisted.unlink()

    def test_catalog_reordering_allowed_but_changed_identity_rejected(self):
        catalog = self.matcher.pipeline.catalog.copy()
        self.entry.validate_catalog(catalog.iloc[::-1], catalog)
        changed = catalog.copy()
        changed.loc[0, "vessel_name"] = "CHANGED VESSEL"
        with self.assertRaises(ValueError):
            self.entry.validate_catalog(changed, catalog)
        with self.assertRaises(ValueError):
            self.entry.validate_catalog(catalog.iloc[:-1], catalog)

    def test_reproducible_zip_and_no_overwrite(self):
        with zipfile.ZipFile(self.archive) as archive:
            payload = {name: archive.read(name) for name in archive.namelist()}
            self.assertEqual({name.split("/")[0] for name in payload}, {"model", "script.py", "requirements.txt"})
        second = self.root / "repeat.zip"
        write_archive(second, payload)
        self.assertEqual(sha(second), sha(self.archive))
        with self.assertRaises(FileExistsError):
            write_archive(self.archive, payload)
        with self.assertRaises(FileExistsError):
            build(self.study, self.archive)
        metadata = json.loads(payload["model/runtime.json"])
        self.assertEqual(metadata["training"]["branch"], "focused")
        self.assertNotIn("training_summary.json", " ".join(payload))
        self.assertEqual(payload["model/ranker/model.txt"], (self.study / "fold0/focused/model.txt").read_bytes())


if __name__ == "__main__":
    unittest.main()
