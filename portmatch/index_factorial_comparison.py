"""Frozen-ranker 2x2 search/category diagnostics with exact model-input evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .data import QUERY_COLUMNS
from .index_scale_comparison import DIAGNOSTIC_FEATURES, _ranker_fingerprint
from .index_scale_assets import assert_unchanged
from .index_scale_report import _validate_records
from .retrieval_comparison import (VARIANTS, _check_saved_predictions, _hash,
    _initialize_name_features, _write_json, select_queries, variant_effect_counts, variant_query)
from .text import has_hangul


def feature_schema(matcher):
    names = list(matcher.selection.selected_features)
    categories = [name for name in names if name.startswith(('flag_', 'type_'))]
    if len(names) != 86 or len(set(names)) != 86 or len(categories) != 16:
        raise ValueError('Expected the unchanged 86-feature schema with all 16 category features')
    return {'selected_features': names, 'category_features': categories,
        'non_category_features': [name for name in names if name not in categories],
        'dtype': 'float32', 'nan_policy': 'blank CSV values with explicit presence and NaN flags',
        'detail_scope': 'every query/variant: union of target and four winners, aligned across all runners'}


def full_feature_prediction(matcher, query, target):
    """Never forward IDs/labels; retain the exact float32 array scored by the model."""
    from .feature_transforms import transform_features
    from .ranker_commands import score_order

    query = {key: query.get(key) for key in QUERY_COLUMNS if key != 'sample_id'}
    started = time.perf_counter()
    candidates = matcher.pipeline.retrieve(query)
    retrieved = time.perf_counter()
    positions = candidates.catalog_index.to_numpy()
    if (not len(positions) or positions.dtype.kind not in 'iuf'
            or not np.isfinite(positions).all() or not np.equal(positions, np.floor(positions)).all()
            or len(set(positions)) != len(positions) or (positions < 0).any()
            or (positions >= len(matcher.catalog_ids)).any()):
        raise ValueError('Retriever produced invalid, duplicate or empty catalog positions')
    positions = positions.astype(np.int64)
    features = matcher.pipeline.features(query, candidates)
    features = transform_features(matcher.feature_transform, matcher.pipeline, query, candidates, features)
    names = list(matcher.selection.selected_features)
    selected = matcher.selection.apply(features.to_numpy(dtype=np.float32), features.columns.tolist())
    if selected.shape != (len(positions), 86) or np.isinf(selected).any():
        raise ValueError('Invalid 86-feature model input')
    featured = time.perf_counter()
    scores = np.asarray(matcher.ranker.predict(selected, names))
    if scores.shape != (len(positions),) or not np.isfinite(scores).all():
        raise ValueError('Nonfinite or misaligned rank scores')
    order = score_order(scores, positions, matcher.catalog_ids)
    finished = time.perf_counter()
    ids = matcher.catalog_ids[positions].astype(str)
    ranked = ids[order]
    locations = np.flatnonzero(ranked == str(target))
    result = {'predicted_vessel_id': str(ranked[0]), 'candidate_found': bool(len(locations)),
        'target_rank': int(locations[0] + 1) if len(locations) else 0,
        'candidate_count': len(positions), 'top_score': float(scores[order[0]]),
        'score_margin': float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else None,
        'retrieval_seconds': retrieved - started, 'feature_seconds': featured - retrieved,
        'rank_seconds': finished - featured, 'total_seconds': finished - started,
        'candidate_ids_sha256': hashlib.sha256(json.dumps(sorted(ids.tolist()),
            ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()}
    details = pd.DataFrame(selected.copy(), columns=names, index=pd.Index(ids, name='candidate_vessel_id'))
    details['score'] = scores
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    details['candidate_rank'] = ranks
    details['candidate_catalog_name'] = matcher.pipeline.catalog.iloc[positions].vessel_name.to_numpy()
    for prefix, vessel in [('top', str(ranked[0])), ('target', str(target))]:
        present = vessel in details.index
        result[prefix + '_catalog_name'] = str(details.at[vessel, 'candidate_catalog_name']) if present else None
        for name in DIAGNOSTIC_FEATURES:
            value = float(details.at[vessel, name]) if present else None
            result[prefix + '_' + name] = value if value is not None and np.isfinite(value) else None
    return result, details


def validate_feature_isolation(pair, schema):
    """Check all retrieved candidates, not just the exported target/winner subset."""
    from .index_factorial_assets import RUNNERS
    a, b, c, d = RUNNERS
    counts = {'same_search_pairs': 0, 'non_category_candidate_pairs': 0,
        'same_category_pairs': 0, 'category_candidate_pairs': 0}
    for left, right in [(a, b), (c, d)]:
        lf, rf = pair[left][1], pair[right][1]
        if set(lf.index) != set(rf.index):
            raise ValueError('Category-only change altered the search candidate set')
        columns = schema['non_category_features']
        if not np.array_equal(lf[columns].to_numpy(), rf.loc[lf.index, columns].to_numpy(), equal_nan=True):
            raise ValueError('Category-only change altered non-category model features')
        counts['same_search_pairs'] += 1
        counts['non_category_candidate_pairs'] += len(lf)
    for left, right in [(a, c), (b, d)]:
        lf, rf = pair[left][1], pair[right][1]
        common = lf.index.intersection(rf.index)
        columns = schema['category_features']
        if not np.array_equal(lf.loc[common, columns].to_numpy(), rf.loc[common, columns].to_numpy(), equal_nan=True):
            raise ValueError('Search-only change altered same-candidate category features')
        counts['same_category_pairs'] += 1
        counts['category_candidate_pairs'] += len(common)
    return counts


def _read_reference(directory, context):
    """Bind completed previous diagonal results to exact data, model and subset."""
    directory = Path(directory).resolve()
    summary_path = directory / 'summary.json'
    summary = json.loads(summary_path.read_text('utf-8'))
    flags = ('complete', 'weights_unchanged', 'training_assets_unchanged',
        'protected_files_unchanged', 'ranker_shared_by_both_indexes')
    if not all(summary.get(flag) is True for flag in flags):
        raise ValueError('Reference index-scale diagnostic is incomplete or unverified')
    if (summary.get('variants') != list(VARIANTS)
            or summary.get('coverage', {}).get('diagnostic_sample') is not False
            or summary.get('training_performed') is not False
            or summary.get('submission_created') is not False):
        raise ValueError('Reference must contain the complete five-variant held-out comparison')
    for key, value in context['proof']['coverage'].items():
        if summary.get('coverage', {}).get(key) != value:
            raise ValueError('Reference fold/population differs from current fixed partition')
    if (summary.get('proof', {}).get('selection') != context['proof']['selection']
            or summary['coverage'].get('queries') != len(context['heldout'])):
        raise ValueError('Reference subset or held-out coverage differs')
    for key in ['full_learned_assets_sha256', 'small_learned_assets_sha256', 'protected_files']:
        if summary['proof']['integrity'].get(key) != context['proof']['integrity'][key]:
            raise ValueError('Reference training assets or protected file hashes differ')
    if summary['provenance'].get('in_memory_ranker_sha256') != _ranker_fingerprint(context['matcher'].ranker):
        raise ValueError('Reference ranker weights differ')
    protected = {str(summary_path): _hash(summary_path)}
    expected_files = {'predictions.csv', 'row_changes.csv', 'vessel_changes.csv',
        'changed_candidate_features.csv', 'partitions.csv', 'report.md'}
    if set(summary.get('output_files_sha256', {})) != expected_files:
        raise ValueError('Reference artifact manifest is incomplete')
    for name, digest in summary['output_files_sha256'].items():
        path = directory / name
        if _hash(path) != digest:
            raise ValueError(f'Reference artifact hash differs: {name}')
        protected[str(path)] = digest
    for name in ('index_scale_assets.py', 'index_scale_comparison.py', 'index_scale_report.py'):
        if summary['provenance'].get('code_sha256', {}).get(name) != _hash(Path(__file__).with_name(name)):
            raise ValueError('Reference diagnostic code differs')
    data = pd.read_csv(directory / 'predictions.csv', dtype={'sample_id': str,
        'target_vessel_id': str, 'predicted_vessel_id': str})
    data['vessel_name'] = data.vessel_name.fillna('')
    data = _validate_records(data, VARIANTS, summary['coverage'])
    if set(data.sample_id) != set(context['heldout'].sample_id.astype(str)):
        raise ValueError('Reference query IDs do not match fixed held-out partition')
    return data, protected


def check_reference_parity(predictions, reference):
    from .index_factorial_assets import RUNNERS
    exact = ['target_vessel_id', 'predicted_vessel_id', 'vessel_name', 'has_hangul',
        'candidate_found', 'target_rank', 'candidate_count', 'candidate_ids_sha256']
    numeric = ['top_score', 'score_margin',
        *(f'{side}_{name}' for side in ('top', 'target') for name in DIAGNOSTIC_FEATURES)]
    counts = {}
    for runner, old in [(RUNNERS[0], 'index_80'), (RUNNERS[3], 'index_100')]:
        actual = predictions.loc[predictions.model.eq(runner)].set_index(['variant', 'sample_id']).sort_index()
        expected = reference.loc[reference.model.eq(old)].set_index(['variant', 'sample_id'])
        if not actual.index.isin(expected.index).all():
            raise ValueError('Reference diagonal rows are missing')
        expected = expected.loc[actual.index]
        for column in exact:
            if not actual[column].equals(expected[column]):
                raise ValueError(f'Reference diagonal parity differs: {runner}/{column}')
        for column in numeric:
            if not np.allclose(actual[column].to_numpy(dtype=float), expected[column].to_numpy(dtype=float),
                               rtol=0, atol=1e-10, equal_nan=True):
                raise ValueError(f'Reference diagonal score/feature parity differs: {runner}/{column}')
        counts[runner] = {'matched_predictions': len(actual), 'matched_queries': actual.index.get_level_values(1).nunique()}
    return {'verified': True, 'timing_compared': False, 'score_absolute_tolerance': 1e-10, 'runners': counts}


def _cell(value):
    return float(value) if value is not None and np.isfinite(float(value)) else None


class FeatureEvidenceWriter:
    """Stream bounded target/winner evidence; never retain all candidates across queries."""
    def __init__(self, output, schema):
        from .index_factorial_assets import RUNNERS
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
        from .index_factorial_report import CONTRASTS
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
            for contrast, spec in CONTRASTS.items():
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


def compare_index_factorial(model_directory, reference_directory, output, *,
        train_path='work/data/train.csv', catalog_path='work/data/vessel_catalog.csv',
        subset_seed=20261027, expected_fold=None, variants=VARIANTS, max_queries=None, progress_every=100):
    from .index_factorial_assets import (RUNNERS, prepare_index_factorial,
        validate_factorial_wiring, factorial_assets_fingerprint)
    from .index_factorial_report import build_index_factorial_report
    variants = tuple(variants)
    if (not variants or len(set(variants)) != len(variants) or 'original' not in variants
            or set(variants) - set(VARIANTS)):
        raise ValueError('Use unique known variants including original')
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError('progress-every must be a positive integer')
    if max_queries is not None and (type(max_queries) is not int or max_queries < 1):
        raise ValueError('max-queries must be a positive integer')
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Choose a new or empty output directory: {output}')
    started = time.perf_counter()
    context = prepare_index_factorial(model_directory, train_path, catalog_path,
        subset_seed=subset_seed, expected_fold=expected_fold)
    reference, reference_files = _read_reference(reference_directory, context)
    runners, full = context['runners'], context['matcher']
    schema = feature_schema(full)
    selected = select_queries(context['heldout'], max_queries=max_queries, seed=subset_seed)
    if selected.empty:
        raise ValueError('No held-out rows selected')
    coverage = dict(context['proof']['coverage'])
    coverage.update(queries=len(selected), heldout_queries=len(context['heldout']),
        vessels=int(selected.target_vessel_id.nunique()),
        diagnostic_sample=len(selected) != len(context['heldout']), max_queries=max_queries)
    startup = dict(context['proof']['startup'])
    startup['name_feature_index_seconds'] = _initialize_name_features(full)
    for runner in runners.values():
        runner.pipeline._token_phonetic_computer_v1 = full.pipeline._token_phonetic_computer_v1
    weights_before = _ranker_fingerprint(full.ranker)
    validate_factorial_wiring(context)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'schema_version': 1, 'complete': False, 'coverage': coverage,
        'proof': context['factorial_proof'], 'startup': startup, 'variants': list(variants),
        'feature_schema': schema, 'variant_effects': variant_effect_counts(selected, variants),
        'provenance': {'model_directory': str(Path(model_directory).resolve()),
            'reference_directory': str(Path(reference_directory).resolve()),
            'reference_files': reference_files, 'in_memory_ranker_sha256': weights_before,
            'code_sha256': {name: _hash(Path(__file__).with_name(name)) for name in
                ['index_factorial_assets.py', 'index_factorial_report.py', 'index_factorial_comparison.py']}},
        'environment': {'python': platform.python_version(), 'platform': platform.platform()},
        'training_performed': False, 'submission_created': False}
    _write_json(output / 'summary.json', metadata)
    context['membership'].to_csv(output / 'partitions.csv', index=False, encoding='utf-8-sig')
    assets_before = factorial_assets_fingerprint(context)
    startup['warmup_seconds'] = {}
    for name, runner in runners.items():
        warmup = time.perf_counter()
        full_feature_prediction(runner, variant_query(selected.iloc[0].to_dict(), 'original'), '')
        startup['warmup_seconds'][name] = time.perf_counter() - warmup
    records, isolation = [], {'verified': True, 'same_search_pairs': 0, 'non_category_candidate_pairs': 0,
        'same_category_pairs': 0, 'category_candidate_pairs': 0}
    writer = FeatureEvidenceWriter(output, schema)
    try:
        for position, row in enumerate(selected.to_dict('records')):
            for variant_position, variant in enumerate(variants):
                query = variant_query(row, variant)
                rotation = (position * len(variants) + variant_position) % len(RUNNERS)
                order = RUNNERS[rotation:] + RUNNERS[:rotation]
                pair = {}
                for name in order:
                    result, details = full_feature_prediction(runners[name], query, row['target_vessel_id'])
                    pair[name] = (result, details)
                    records.append({'variant': variant, 'model': name, 'sample_id': str(row['sample_id']),
                        'target_vessel_id': str(row['target_vessel_id']),
                        'vessel_name': row['vessel_name'] if isinstance(row['vessel_name'], str) else '',
                        'has_hangul': has_hangul(row['vessel_name']), 'runner_position': order.index(name), **result})
                for key, count in validate_feature_isolation(pair, schema).items():
                    isolation[key] += count
                writer.write(pair, row, variant)
            if (position + 1) % progress_every == 0 or position + 1 == len(selected):
                print(f'Index factorial: {position + 1:,}/{len(selected):,} held-out rows, '
                      f'{len(records):,} predictions', flush=True)
    finally:
        writer.close()
    predictions = pd.DataFrame(records).sort_values(['variant', 'sample_id', 'model'], kind='stable')
    parity = {'reference_diagonals': check_reference_parity(predictions, reference),
        'saved_original': _check_saved_predictions(model_directory,
            predictions.loc[predictions.variant.eq('original') & predictions.model.eq(RUNNERS[3])],
            context['heldout'].sample_id, RUNNERS[3])}
    validate_factorial_wiring(context)
    assert_unchanged(context['protected_files'])
    assert_unchanged(reference_files)
    if weights_before != _ranker_fingerprint(full.ranker):
        raise AssertionError('Ranker weights changed during the diagnostic')
    if assets_before != factorial_assets_fingerprint(context):
        raise AssertionError('Search/category assets changed during the diagnostic')
    summary, changes, vessels, markdown = build_index_factorial_report(predictions,
        coverage=coverage, proof=context['factorial_proof'], startup=startup, parity=parity, variants=variants)
    summary.update({key: value for key, value in metadata.items() if key not in
        {'complete', 'coverage', 'proof', 'startup'}})
    summary.update(complete=True, weights_unchanged=True, training_assets_unchanged=True,
        protected_files_unchanged=True, reference_files_unchanged=True,
        ranker_shared_by_all_runners=True, feature_isolation=isolation,
        feature_evidence_counts=writer.counts, total_seconds=time.perf_counter() - started)
    predictions.to_csv(output / 'predictions.csv', index=False, encoding='utf-8-sig')
    changes.to_csv(output / 'row_changes.csv', index=False, encoding='utf-8-sig')
    vessels.to_csv(output / 'vessel_changes.csv', index=False, encoding='utf-8-sig')
    markdown += ('\n\n동일 후보 전체86열은 candidate_features.csv, 대조별 점수·변경 열 수는 '
        'candidate_comparisons.csv, 실제 바뀐 열과 NaN 전이는 feature_changes.csv에 있습니다. '
        '모든 질의·조건의 정답과 네1위 후보 합집합을 네 조합에 정렬하며 후보 부재는 present=false입니다. '
        '모든 검색 후보의 특징 분리 불변 조건도 실행 중 검사합니다. 파일 추출·저장은 행별 추론 시간에 포함하지 않습니다.\n')
    (output / 'report.md').write_text(markdown, encoding='utf-8')
    summary['output_files_sha256'] = {path.name: _hash(path) for path in output.iterdir()
        if path.is_file() and path.name != 'summary.json'}
    _write_json(output / 'summary.json', summary)
    print(f'Completed fixed-weight 2x2 diagnostic without training: {output}', flush=True)
    return summary


def add_index_factorial_parser(subparsers):
    parser = subparsers.add_parser('compare-index-factorial', help='Frozen robust86: search/category 80/100 factorial')
    parser.add_argument('--model', required=True)
    parser.add_argument('--reference', required=True, help='Completed full previous compare-index-scale directory')
    parser.add_argument('--fold', required=True, type=int, choices=[0, 2])
    parser.add_argument('--train', default='work/data/train.csv')
    parser.add_argument('--catalog', default='work/data/vessel_catalog.csv')
    parser.add_argument('--output', required=True, help='New or empty output directory')
    parser.add_argument('--subset-seed', type=int, default=20261027)
    parser.add_argument('--max-queries', type=int, help='Optional smoke subset; omit for full validation')
    parser.add_argument('--progress-every', type=int, default=100)


def run_index_factorial(args):
    return compare_index_factorial(args.model, args.reference, args.output, train_path=args.train,
        catalog_path=args.catalog, subset_seed=args.subset_seed, expected_fold=args.fold,
        max_queries=args.max_queries, progress_every=args.progress_every)
