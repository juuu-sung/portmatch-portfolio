import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from portmatch.support_ablation_report import RUNNERS, SUPPORT_FEATURES
from portmatch.support_ablation_evidence import SupportFeatureEvidenceWriter, validate_shared_features


def schema():
    shared = ["feature_" + str(i) for i in range(82)]
    return {"baseline_features": [*SUPPORT_FEATURES, *shared],
            "experiment_features": shared, "excluded_features": list(SUPPORT_FEATURES)}


def example_pair():
    result = {}
    for i, runner in enumerate(RUNNERS):
        key = "baseline_features" if i == 0 else "experiment_features"
        frame = pd.DataFrame(0., columns=schema()[key], index=["A", "B", "UNEXPORTED"], dtype=np.float32)
        frame["feature_1"] = np.nan
        frame["score"] = [3. - 2 * i, 2. + i, 0.]
        frame["candidate_rank"] = [1 + i, 2 - i, 3]
        frame["candidate_catalog_name"] = ["Name " + vessel for vessel in frame.index]
        result[runner] = ({"predicted_vessel_id": "A" if i == 0 else "B"}, frame)
    return result


class SupportAblationEvidenceTests(unittest.TestCase):
    def test_invariance_covers_unexported_candidates_and_aligns_by_id(self):
        pair = example_pair()
        record, frame = pair[RUNNERS[1]]
        pair[RUNNERS[1]] = (record, frame.iloc[::-1])
        self.assertEqual(validate_shared_features(pair, schema()),
            {"same_search_pairs": 1, "shared_feature_candidate_pairs": 3})
        pair[RUNNERS[1]][1].loc["UNEXPORTED", "feature_5"] = .125
        with self.assertRaisesRegex(ValueError, "shared 82"):
            validate_shared_features(pair, schema())

    def test_candidate_membership_nan_changes_and_extra_excluded_inputs_rejected(self):
        pair = example_pair()
        pair[RUNNERS[1]][1].rename(index={"UNEXPORTED": "OTHER"}, inplace=True)
        with self.assertRaisesRegex(ValueError, "candidate set"):
            validate_shared_features(pair, schema())
        pair = example_pair()
        pair[RUNNERS[1]][1].loc["UNEXPORTED", "feature_1"] = 0.
        with self.assertRaisesRegex(ValueError, "shared 82"):
            validate_shared_features(pair, schema())
        pair = example_pair()
        pair[RUNNERS[1]][1][SUPPORT_FEATURES[0]] = 0.
        with self.assertRaisesRegex(ValueError, "actual model inputs"):
            validate_shared_features(pair, schema())

    def test_schema_requires_exact_removals_and_preserved_order(self):
        for key in ("baseline_features", "experiment_features", "excluded_features"):
            broken = schema()
            broken[key][0] = "OTHER"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Feature schema"):
                validate_shared_features(example_pair(), broken)
        broken = schema()
        broken["experiment_features"].reverse()
        with self.assertRaisesRegex(ValueError, "preserving order"):
            validate_shared_features(example_pair(), broken)

    def test_evidence_distinguishes_absence_nan_and_omitted_model_input(self):
        pair = example_pair()
        pair[RUNNERS[0]][1].loc["A", SUPPORT_FEATURES[0]] = np.nan
        row = {"sample_id": "q1", "target_vessel_id": "ABSENT", "vessel_name": "한글"}
        with TemporaryDirectory() as temp:
            writer = SupportFeatureEvidenceWriter(Path(temp), schema())
            try:
                writer.write(pair, row, "original")
            finally:
                writer.close()
            detail = pd.read_csv(Path(temp) / "candidate_features.csv")
            comparison = pd.read_csv(Path(temp) / "candidate_comparisons.csv")
            self.assertFalse((Path(temp) / "feature_changes.csv").exists())
        self.assertEqual(len(detail), 6)
        self.assertEqual(len(comparison), 3)
        self.assertEqual(set(detail.candidate_vessel_id), {"A", "B", "ABSENT"})
        missing = detail.loc[detail.candidate_vessel_id.eq("ABSENT")]
        self.assertFalse(missing.present.any())
        self.assertTrue(missing[schema()["baseline_features"] + ["score", "candidate_rank"]].isna().all().all())
        self.assertTrue(comparison.loc[comparison.candidate_vessel_id.eq("ABSENT"), "shared_changed_features"].isna().all())
        baseline = detail.loc[detail.model.eq(RUNNERS[0])]
        experiment = detail.loc[detail.model.eq(RUNNERS[1])]
        self.assertTrue(baseline.feature_count.eq(86).all())
        self.assertTrue(experiment.feature_count.eq(82).all())
        self.assertTrue(baseline.omitted_features.map(json.loads).map(lambda names: names == []).all())
        self.assertTrue(experiment.omitted_features.map(json.loads).map(lambda names: names == list(SUPPORT_FEATURES)).all())
        self.assertTrue(experiment[list(SUPPORT_FEATURES)].isna().all().all())
        self.assertTrue(detail.loc[detail.present, "feature_1"].isna().all())
        delta = comparison.loc[comparison.candidate_vessel_id.eq("A")].iloc[0]
        self.assertEqual(delta.reference_runner, RUNNERS[0])
        self.assertEqual(delta.experiment_runner, RUNNERS[1])
        self.assertEqual(delta.score_delta, -2.)
        self.assertEqual(delta.shared_changed_features, 0)
        self.assertEqual(delta.shared_feature_count, 82)
        self.assertEqual(delta.omitted_feature_count, 4)
        self.assertEqual(writer.counts, {"candidate_feature_rows": 6,
            "candidate_comparison_rows": 3, "selected_candidate_groups": 3})


if __name__ == "__main__":
    unittest.main()
