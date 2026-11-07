"""Frozen robust86 comparison of category scope with full search held constant."""
from __future__ import annotations

from pathlib import Path
import platform
import time

import pandas as pd

from .index_factorial_comparison import feature_schema, full_feature_prediction
from .index_scale_assets import SUBSET_SEED, assert_unchanged
from .index_scale_comparison import _ranker_fingerprint
from .retrieval_comparison import (VARIANTS, _check_saved_predictions, _hash,
    _initialize_name_features, _write_json, select_queries, variant_effect_counts, variant_query)
from .text import has_hangul


def compare_category_scope(model_directory, output, *,
        train_path='work/data/train.csv', catalog_path='work/data/vessel_catalog.csv',
        subset_seed=SUBSET_SEED, expected_fold=None, variants=VARIANTS,
        max_queries=None, progress_every=100):
    from .category_scope_assets import (RUNNERS, prepare_category_scope,
        validate_category_wiring, category_assets_fingerprint)
    from .category_scope_evidence import (CategoryFeatureEvidenceWriter,
        validate_category_feature_isolation)
    from .category_scope_report import build_category_scope_report

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
    context = prepare_category_scope(model_directory, train_path, catalog_path,
        subset_seed=subset_seed, expected_fold=expected_fold)
    runners, full = context['runners'], context['matcher']
    schema = feature_schema(full)
    schema['detail_scope'] = 'every query/variant: union of target and two winners, aligned across both runners'
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
    validate_category_wiring(context)
    code_files = ['category_scope_assets.py', 'category_scope_comparison.py',
        'category_scope_report.py', 'category_scope_evidence.py',
        'index_scale_assets.py', 'index_scale_comparison.py', 'index_scale_report.py',
        'index_factorial_assets.py', 'index_factorial_comparison.py',
        'index_factorial_report.py', 'retrieval_comparison.py']
    code_paths = {str(Path(__file__).with_name(name)): _hash(Path(__file__).with_name(name))
        for name in code_files}
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'schema_version': 1, 'complete': False, 'coverage': coverage,
        'proof': context['category_proof'], 'startup': startup, 'variants': list(variants),
        'feature_schema': schema, 'variant_effects': variant_effect_counts(selected, variants),
        'provenance': {'model_directory': str(Path(model_directory).resolve()),
            'in_memory_ranker_sha256': weights_before,
            'code_sha256': {Path(path).name: digest for path, digest in code_paths.items()}},
        'environment': {'python': platform.python_version(), 'platform': platform.platform()},
        'training_performed': False, 'submission_created': False}
    _write_json(output / 'summary.json', metadata)
    context['membership'].to_csv(output / 'partitions.csv', index=False, encoding='utf-8-sig')
    assets_before = category_assets_fingerprint(context)
    startup['warmup_seconds'] = {}
    for name, runner in runners.items():
        warmup = time.perf_counter()
        full_feature_prediction(runner, variant_query(selected.iloc[0].to_dict(), 'original'), '')
        startup['warmup_seconds'][name] = time.perf_counter() - warmup
    records, isolation = [], {'verified': True, 'same_search_pairs': 0, 'non_category_candidate_pairs': 0}
    writer = CategoryFeatureEvidenceWriter(output, schema)
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
                for key, count in validate_category_feature_isolation(pair, schema).items():
                    isolation[key] += count
                writer.write(pair, row, variant)
            if (position + 1) % progress_every == 0 or position + 1 == len(selected):
                print(f'Category scope: {position + 1:,}/{len(selected):,} held-out rows, '
                      f'{len(records):,} predictions', flush=True)
    finally:
        writer.close()
    predictions = pd.DataFrame(records).sort_values(['variant', 'sample_id', 'model'], kind='stable')
    parity = {'saved_original': _check_saved_predictions(model_directory,
        predictions.loc[predictions.variant.eq('original') & predictions.model.eq(RUNNERS[0])],
        context['heldout'].sample_id, RUNNERS[0])}
    validate_category_wiring(context)
    assert_unchanged(context['protected_files'])
    assert_unchanged(code_paths)
    if weights_before != _ranker_fingerprint(full.ranker):
        raise AssertionError('Ranker weights changed during the diagnostic')
    if assets_before != category_assets_fingerprint(context):
        raise AssertionError('Search/category assets changed during the diagnostic')
    summary, changes, vessels, markdown = build_category_scope_report(predictions,
        coverage=coverage, proof=context['category_proof'], startup=startup, parity=parity, variants=variants)
    summary.update({key: value for key, value in metadata.items() if key not in
        {'complete', 'coverage', 'proof', 'startup'}})
    summary.update(complete=True, weights_unchanged=True, training_assets_unchanged=True,
        protected_files_unchanged=True, diagnostic_code_unchanged=True,
        ranker_shared_by_both_runners=True, feature_isolation=isolation,
        feature_evidence_counts=writer.counts, total_seconds=time.perf_counter() - started)
    predictions.to_csv(output / 'predictions.csv', index=False, encoding='utf-8-sig')
    changes.to_csv(output / 'row_changes.csv', index=False, encoding='utf-8-sig')
    vessels.to_csv(output / 'vessel_changes.csv', index=False, encoding='utf-8-sig')
    markdown += ('\n\n동일 후보 전체86열은 candidate_features.csv, 점수·변경 열 수는 '
        'candidate_comparisons.csv, 실제 변경 열과 NaN 전이는 feature_changes.csv에 있습니다. '
        '각 질의·조건의 정답과 두1위 후보 합집합을 양쪽 조합에 정렬하며 후보 부재는 present=false입니다. '
        '실행 중 모든 검색 후보의 범주 외70열 일치를 검사합니다. 파일 추출·저장은 행별 추론 시간에 포함하지 않습니다.\n')
    (output / 'report.md').write_text(markdown, encoding='utf-8')
    summary['output_files_sha256'] = {path.name: _hash(path) for path in output.iterdir()
        if path.is_file() and path.name != 'summary.json'}
    _write_json(output / 'summary.json', summary)
    print(f'Completed fixed-weight category-scope diagnostic without training: {output}', flush=True)
    return summary


def add_category_scope_parser(subparsers):
    parser = subparsers.add_parser('compare-category-scope',
        help='Frozen robust86: full search with category80 versus category100')
    parser.add_argument('--model', required=True)
    parser.add_argument('--fold', required=True, type=int, choices=range(5))
    parser.add_argument('--train', default='work/data/train.csv')
    parser.add_argument('--catalog', default='work/data/vessel_catalog.csv')
    parser.add_argument('--output', required=True, help='New or empty output directory')
    parser.add_argument('--subset-seed', type=int, choices=[SUBSET_SEED], default=SUBSET_SEED)
    parser.add_argument('--max-queries', type=int, help='Optional smoke subset; omit for full validation')
    parser.add_argument('--progress-every', type=int, default=100)


def run_category_scope(args):
    return compare_category_scope(args.model, args.output, train_path=args.train,
        catalog_path=args.catalog, subset_seed=args.subset_seed, expected_fold=args.fold,
        max_queries=args.max_queries, progress_every=args.progress_every)
