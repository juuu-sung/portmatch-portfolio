from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from portmatch.category_scope_assets import RUNNERS
from portmatch.category_scope_evidence import CategoryFeatureEvidenceWriter, validate_feature_isolation


def schema():
    categories = [prefix + '_' + str(i) for prefix in ('flag', 'type') for i in range(8)]
    others = ['feature_' + str(i) for i in range(70)]
    return {'selected_features': categories + others, 'category_features': categories,
            'non_category_features': others}


def example_pair():
    result = {}
    for i, runner in enumerate(RUNNERS):
        frame = pd.DataFrame(0., columns=schema()['selected_features'], index=['A', 'B', 'UNEXPORTED'], dtype=np.float32)
        frame.loc[:, schema()['category_features']] = float(i)
        frame['feature_1'] = np.nan
        frame['score'] = [3. - 2 * i, 2. + i, 0.]
        frame['candidate_rank'] = [1 + i, 2 - i, 3]
        frame['candidate_catalog_name'] = ['Name ' + vessel for vessel in frame.index]
        result[runner] = ({'predicted_vessel_id': 'A' if i == 0 else 'B'}, frame)
    return result


class CategoryScopeEvidenceTests(unittest.TestCase):
    def test_isolation_checks_unexported_candidates_and_aligns_by_id(self):
        pair = example_pair()
        record, frame = pair[RUNNERS[1]]
        pair[RUNNERS[1]] = (record, frame.iloc[::-1])
        self.assertEqual(validate_feature_isolation(pair, schema()),
            {'same_search_pairs': 1, 'non_category_candidate_pairs': 3})
        pair[RUNNERS[1]][1].loc['UNEXPORTED', 'feature_5'] = .125
        with self.assertRaisesRegex(ValueError, 'non-category'):
            validate_feature_isolation(pair, schema())

    def test_candidate_membership_and_nan_transitions_cannot_change_noncategory_features(self):
        pair = example_pair()
        pair[RUNNERS[1]][1].rename(index={'UNEXPORTED': 'OTHER'}, inplace=True)
        with self.assertRaisesRegex(ValueError, 'candidate set'):
            validate_feature_isolation(pair, schema())
        pair = example_pair()
        pair[RUNNERS[1]][1].loc['UNEXPORTED', 'feature_1'] = 0.
        with self.assertRaisesRegex(ValueError, 'non-category'):
            validate_feature_isolation(pair, schema())

    def test_evidence_uses_c_minus_d_and_distinguishes_absence_from_nan(self):
        pair = example_pair()
        pair[RUNNERS[0]][1].loc['A', 'flag_0'] = np.nan
        row = {'sample_id': 'q1', 'target_vessel_id': 'ABSENT', 'vessel_name': '한글'}
        with TemporaryDirectory() as temp:
            writer = CategoryFeatureEvidenceWriter(Path(temp), schema())
            try:
                writer.write(pair, row, 'original')
            finally:
                writer.close()
            detail = pd.read_csv(Path(temp) / 'candidate_features.csv')
            comparison = pd.read_csv(Path(temp) / 'candidate_comparisons.csv')
            changed = pd.read_csv(Path(temp) / 'feature_changes.csv')
        self.assertEqual(len(detail), 6)
        self.assertEqual(len(comparison), 3)
        self.assertEqual(set(detail.candidate_vessel_id), {'A', 'B', 'ABSENT'})
        missing = detail.loc[detail.candidate_vessel_id.eq('ABSENT')]
        self.assertFalse(missing.present.any())
        self.assertTrue(missing[schema()['selected_features'] + ['score', 'candidate_rank']].isna().all().all())
        missing_comparison = comparison.loc[comparison.candidate_vessel_id.eq('ABSENT')]
        self.assertTrue(missing_comparison.changed_features.isna().all())
        self.assertFalse(changed.candidate_vessel_id.eq('ABSENT').any())
        delta = comparison.loc[comparison.candidate_vessel_id.eq('A')].iloc[0]
        self.assertEqual(delta.reference_runner, RUNNERS[0])
        self.assertEqual(delta.experiment_runner, RUNNERS[1])
        self.assertEqual(delta.score_delta, -2.)
        self.assertEqual(delta.non_category_changed_features, 0)
        nan = changed.loc[changed.candidate_vessel_id.eq('A') & changed.feature.eq('flag_0')].iloc[0]
        self.assertTrue(nan.reference_is_nan)
        self.assertFalse(nan.experiment_is_nan)
        self.assertEqual(nan.experiment_value, 1.)
        self.assertTrue(pd.isna(nan.delta))
        self.assertEqual(writer.counts['candidate_feature_rows'], 6)
        self.assertEqual(writer.counts['candidate_comparison_rows'], 3)
        self.assertEqual(writer.counts['changed_feature_rows'], len(changed))


if __name__ == '__main__':
    unittest.main()
