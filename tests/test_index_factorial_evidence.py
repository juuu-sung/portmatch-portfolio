"""All-candidate isolation and explicit missing/NaN feature evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from portmatch.index_factorial_assets import RUNNERS
from portmatch.index_factorial_comparison import FeatureEvidenceWriter, validate_feature_isolation


def schema():
    categories = [prefix + '_' + str(i) for prefix in ('flag', 'type') for i in range(8)]
    others = ['feature_' + str(i) for i in range(70)]
    return {'selected_features': categories + others, 'category_features': categories,
            'non_category_features': others}


def example_pair():
    pair = {}
    for i, name in enumerate(RUNNERS):
        ids = ['A', 'B'] if i < 2 else ['B', 'C']
        frame = pd.DataFrame(0., columns=schema()['selected_features'], index=ids, dtype=np.float32)
        frame.loc[:, schema()['category_features']] = float(i % 2)
        frame.loc[:, 'feature_0'] = float(i // 2)
        frame['feature_1'] = np.nan
        frame['candidate_catalog_name'] = ['Name ' + vessel for vessel in ids]
        frame['score'] = [2., 1.]
        frame['candidate_rank'] = [1, 2]
        pair[name] = ({'predicted_vessel_id': ids[0]}, frame)
    return pair


class IndexFactorialEvidenceTests(unittest.TestCase):
    def test_isolation_checks_all_candidates_and_only_allows_the_intended_factor(self):
        pair = example_pair()
        counts = validate_feature_isolation(pair, schema())
        self.assertEqual(counts, {'same_search_pairs': 2, 'non_category_candidate_pairs': 4,
            'same_category_pairs': 2, 'category_candidate_pairs': 2})
        pair[RUNNERS[1]][1].loc['B', 'feature_3'] = .125
        with self.assertRaisesRegex(ValueError, 'non-category'):
            validate_feature_isolation(pair, schema())

    def test_search_cannot_change_category_values_on_common_candidates(self):
        pair = example_pair()
        pair[RUNNERS[2]][1].loc['B', 'flag_4'] = 7.
        with self.assertRaisesRegex(ValueError, 'category features'):
            validate_feature_isolation(pair, schema())

    def test_category_cannot_change_candidate_membership(self):
        pair = example_pair()
        pair[RUNNERS[1]][1].rename(index={'B': 'C'}, inplace=True)
        with self.assertRaisesRegex(ValueError, 'candidate set'):
            validate_feature_isolation(pair, schema())

    def test_feature_evidence_separates_missing_candidates_from_nan_values(self):
        pair = example_pair()
        # Both candidates are present but one selected input changes NaN -> finite.
        pair[RUNNERS[2]][1].loc['B', 'feature_1'] = .5
        pair[RUNNERS[3]][1].loc['B', 'feature_1'] = .5
        row = {'sample_id': 'q1', 'target_vessel_id': 'ABSENT', 'vessel_name': '한글'}
        with TemporaryDirectory() as temp:
            writer = FeatureEvidenceWriter(Path(temp), schema())
            try:
                writer.write(pair, row, 'original')
            finally:
                writer.close()
            detail = pd.read_csv(Path(temp) / 'candidate_features.csv')
            comparisons = pd.read_csv(Path(temp) / 'candidate_comparisons.csv')
            changed = pd.read_csv(Path(temp) / 'feature_changes.csv')
        self.assertEqual(len(detail), 12)  # target ABSENT and winners A/B across four cells.
        self.assertEqual(set(detail.candidate_vessel_id), {'A', 'B', 'ABSENT'})
        missing = detail.loc[detail.candidate_vessel_id.eq('ABSENT')]
        self.assertFalse(missing.present.any())
        self.assertTrue(missing[schema()['selected_features'] + ['score', 'candidate_rank']].isna().all().all())
        absent_comparisons = comparisons.loc[comparisons.candidate_vessel_id.eq('ABSENT')]
        self.assertTrue(absent_comparisons.changed_features.isna().all())
        self.assertFalse(changed.candidate_vessel_id.eq('ABSENT').any())
        transition = changed.loc[changed.candidate_vessel_id.eq('B') & changed.feature.eq('feature_1')
            & changed.contrast.eq('search_at_category80')].iloc[0]
        self.assertTrue(transition.reference_is_nan)
        self.assertFalse(transition.experiment_is_nan)
        self.assertTrue(pd.isna(transition.reference_value))
        self.assertEqual(transition.experiment_value, .5)
        self.assertTrue(pd.isna(transition.delta))
        row_b = detail.loc[detail.candidate_vessel_id.eq('B') & detail.model.eq(RUNNERS[0])].iloc[0]
        self.assertTrue(row_b.present)
        self.assertTrue(pd.isna(row_b.feature_1))
        self.assertEqual(writer.counts['candidate_feature_rows'], 12)
        self.assertEqual(writer.counts['candidate_comparison_rows'], 15)
        self.assertEqual(writer.counts['changed_feature_rows'], len(changed))


if __name__ == '__main__':
    unittest.main()
