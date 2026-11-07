"""All-candidate category isolation and bounded exact-input evidence."""
from __future__ import annotations

import csv

import numpy as np

from .category_scope_assets import RUNNERS
from .index_factorial_comparison import _cell
from .text import has_hangul


def validate_feature_isolation(pair, schema):
    """Require the same candidates and all70 non-category inputs, including NaNs."""
    if set(pair) != set(RUNNERS):
        raise ValueError('Exactly two category-scope runners are required')
    left, right = (pair[name][1] for name in RUNNERS)
    if (not left.index.is_unique or not right.index.is_unique
            or set(left.index) != set(right.index)):
        raise ValueError('Category-only change altered the search candidate set')
    columns = schema['non_category_features']
    if not np.array_equal(left[columns].to_numpy(), right.loc[left.index, columns].to_numpy(), equal_nan=True):
        raise ValueError('Category-only change altered non-category model features')
    return {'same_search_pairs': 1, 'non_category_candidate_pairs': len(left)}


validate_category_feature_isolation = validate_feature_isolation


class CategoryFeatureEvidenceWriter:
    """Stream bounded target/winner evidence; never retain all candidates across queries."""
    def __init__(self, output, schema):
        from .category_scope_assets import RUNNERS
        self.runners, self.schema, self.streams = RUNNERS, schema, []
        self.counts = {'candidate_feature_rows': 0, 'candidate_comparison_rows': 0,
            'changed_feature_rows': 0, 'selected_candidate_groups': 0}
        identity = ['variant', 'sample_id', 'target_vessel_id', 'vessel_name', 'has_hangul',
            'candidate_vessel_id', 'candidate_catalog_name', 'is_target']
        self.feature_writer = self._open(output / 'candidate_features.csv', identity +
            ['model', 'present', 'is_top1', 'score', 'candidate_rank', *schema['selected_features']])
        comparisons = identity + ['contrast', 'reference_runner', 'experiment_runner',
            'reference_present', 'experiment_present', 'reference_score', 'experiment_score',
            'score_delta', 'reference_rank', 'experiment_rank', 'changed_features',
            'category_changed_features', 'non_category_changed_features']
        self.comparison_writer = self._open(output / 'candidate_comparisons.csv', comparisons)
        self.change_writer = self._open(output / 'feature_changes.csv', identity +
            ['contrast', 'reference_runner', 'experiment_runner', 'feature', 'is_category_feature',
             'reference_value', 'experiment_value', 'reference_is_nan', 'experiment_is_nan', 'delta'])

    def _open(self, path, columns):
        stream = path.open('w', encoding='utf-8-sig', newline='')
        self.streams.append(stream)
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        return writer

    def close(self):
        for stream in self.streams:
            stream.close()

    def write(self, pair, row, variant):
        from .category_scope_report import CONTRAST, REFERENCE, EXPERIMENT
        contrasts = {CONTRAST: {"reference": REFERENCE, "experiment": EXPERIMENT}}
        selected = sorted({str(row['target_vessel_id']),
            *(record['predicted_vessel_id'] for record, _ in pair.values())})
        names, category_names = self.schema['selected_features'], set(self.schema['category_features'])
        for vessel in selected:
            self.counts['selected_candidate_groups'] += 1
            known_names = [details.at[vessel, 'candidate_catalog_name'] for _, details in pair.values()
                if vessel in details.index]
            identity = {'variant': variant, 'sample_id': str(row['sample_id']),
                'target_vessel_id': str(row['target_vessel_id']),
                'vessel_name': row['vessel_name'] if isinstance(row['vessel_name'], str) else '',
                'has_hangul': has_hangul(row['vessel_name']), 'candidate_vessel_id': vessel,
                'candidate_catalog_name': known_names[0] if known_names else None,
                'is_target': vessel == str(row['target_vessel_id'])}
            for runner in self.runners:
                record, details = pair[runner]
                present = vessel in details.index
                value = details.loc[vessel] if present else None
                self.feature_writer.writerow({**identity, 'model': runner, 'present': present,
                    'is_top1': vessel == record['predicted_vessel_id'],
                    'score': _cell(value['score']) if present else None,
                    'candidate_rank': int(value['candidate_rank']) if present else None,
                    **{name: _cell(value[name]) if present else None for name in names}})
                self.counts['candidate_feature_rows'] += 1
            for contrast, spec in contrasts.items():
                left, right = spec['reference'], spec['experiment']
                lf, rf = pair[left][1], pair[right][1]
                lp, rp = vessel in lf.index, vessel in rf.index
                common = {**identity, 'contrast': contrast, 'reference_runner': left, 'experiment_runner': right}
                changed = []
                if lp and rp:
                    for name in names:
                        lv, rv = float(lf.at[vessel, name]), float(rf.at[vessel, name])
                        if lv == rv or (np.isnan(lv) and np.isnan(rv)):
                            continue
                        changed.append(name)
                        self.change_writer.writerow({**common, 'feature': name,
                            'is_category_feature': name in category_names,
                            'reference_value': _cell(lv), 'experiment_value': _cell(rv),
                            'reference_is_nan': bool(np.isnan(lv)), 'experiment_is_nan': bool(np.isnan(rv)),
                            'delta': _cell(rv - lv)})
                        self.counts['changed_feature_rows'] += 1
                self.comparison_writer.writerow({**common, 'reference_present': lp, 'experiment_present': rp,
                    'reference_score': _cell(lf.at[vessel, 'score']) if lp else None,
                    'experiment_score': _cell(rf.at[vessel, 'score']) if rp else None,
                    'score_delta': float(rf.at[vessel, 'score'] - lf.at[vessel, 'score']) if lp and rp else None,
                    'reference_rank': int(lf.at[vessel, 'candidate_rank']) if lp else None,
                    'experiment_rank': int(rf.at[vessel, 'candidate_rank']) if rp else None,
                    'changed_features': len(changed) if lp and rp else None,
                    'category_changed_features': sum(n in category_names for n in changed) if lp and rp else None,
                    'non_category_changed_features': sum(n not in category_names for n in changed) if lp and rp else None})
                self.counts['candidate_comparison_rows'] += 1
