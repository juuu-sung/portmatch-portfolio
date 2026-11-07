"""Compare retrained 86/82 inputs with identical full-T search and category assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .data import QUERY_COLUMNS
from .index_scale_assets import assert_unchanged
from .index_scale_comparison import DIAGNOSTIC_FEATURES, _ranker_fingerprint
from .retrieval_comparison import (VARIANTS, _check_saved_predictions, _hash,
    _initialize_name_features, _write_json, select_queries, variant_effect_counts, variant_query)
from .text import has_hangul


def full_feature_prediction(matcher, query, target):
    """Capture the exact selected float32 input; never forward labels or row IDs."""
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
    if len(names) not in (86, 82) or selected.shape != (len(positions), len(names)) or np.isinf(selected).any():
        raise ValueError('Invalid 86/82-feature model input')
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


def compare_support_ablation(baseline_directory, experiment_directory, output, *,
        train_path='work/data/train.csv', catalog_path='work/data/vessel_catalog.csv',
        expected_fold=None, variants=VARIANTS, max_queries=None, progress_every=100):
    from .support_ablation_assets import (RUNNERS, assert_assets_unchanged,
        feature_schemas, prepare_support_ablation)
    from .support_ablation_evidence import SupportFeatureEvidenceWriter, validate_shared_features
    from .support_ablation_report import build_support_ablation_report

    variants = tuple(variants)
    if (not variants or len(set(variants)) != len(variants) or 'original' not in variants
            or set(variants) - set(VARIANTS)):
        raise ValueError('Use unique known variants including original')
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError('progress-every must be a positive integer')
    if max_queries is not None and (type(max_queries) is not int or max_queries < 1):
        raise ValueError('max-queries must be a positive integer')
    if expected_fold is not None and (type(expected_fold) is not int or expected_fold not in range(5)):
        raise ValueError('fold must be an integer from 0 to 4')
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Choose a new or empty output directory: {output}')
    started = time.perf_counter()
    print('Verifying both model bundles and the unchanged shared augmentation cache', flush=True)
    context = prepare_support_ablation(baseline_directory, experiment_directory,
        train_path, catalog_path, expected_fold=expected_fold)
    runners = context['runners']
    schema = {**feature_schemas(), 'dtype': 'float32',
        'nan_policy': 'blank values; present=false means absent candidate; omitted_features marks noninputs',
        'detail_scope': 'every query/variant: union of target and two winners, aligned across both runners'}
    selected = select_queries(context['heldout'], max_queries=max_queries, seed=20261022)
    if selected.empty:
        raise ValueError('No held-out rows selected')
    coverage = dict(context['coverage'])
    coverage.update(queries=len(selected), heldout_queries=len(context['heldout']),
        vessels=int(selected.target_vessel_id.nunique()),
        diagnostic_sample=len(selected) != len(context['heldout']), max_queries=max_queries)
    startup = dict(context['startup'])
    startup['name_feature_index_seconds'] = {
        name: _initialize_name_features(runner) for name, runner in runners.items()}
    weights_before = {name: _ranker_fingerprint(runner.ranker) for name, runner in runners.items()}
    code_files = ['support_ablation_assets.py', 'support_ablation_comparison.py',
        'support_ablation_report.py', 'support_ablation_evidence.py', 'ranker_commands.py',
        'index_scale_assets.py', 'index_scale_comparison.py', 'index_scale_report.py',
        'index_factorial_comparison.py', 'index_factorial_report.py', 'retrieval_comparison.py']
    code_paths = {str(Path(__file__).with_name(name)): _hash(Path(__file__).with_name(name))
        for name in code_files}
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'schema_version': 1, 'complete': False, 'coverage': coverage,
        'proof': context['proof'], 'startup': startup, 'variants': list(variants),
        'feature_schema': schema, 'variant_effects': variant_effect_counts(selected, variants),
        'provenance': {'model_directories': dict(zip(RUNNERS,
            map(lambda path: str(Path(path).resolve()), (baseline_directory, experiment_directory)))),
            'in_memory_ranker_sha256': weights_before,
            'code_sha256': {Path(path).name: digest for path, digest in code_paths.items()}},
        'environment': {'python': platform.python_version(), 'platform': platform.platform()},
        'training_performed': False, 'submission_created': False,
        'training_note': 'Comparison does not fit models; the 82-feature model was retrained separately.'}
    _write_json(output / 'summary.json', metadata)
    _write_json(output / 'feature_schema.json', schema)
    context['membership'].to_csv(output / 'partitions.csv', index=False, encoding='utf-8-sig')
    startup['warmup_seconds'] = {}
    for name, runner in runners.items():
        warmup = time.perf_counter()
        full_feature_prediction(runner, variant_query(selected.iloc[0].to_dict(), 'original'), '')
        startup['warmup_seconds'][name] = time.perf_counter() - warmup
    records, isolation = [], {'verified': True, 'same_search_pairs': 0, 'shared_feature_candidate_pairs': 0}
    writer = SupportFeatureEvidenceWriter(output, schema)
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
                for key, count in validate_shared_features(pair, schema).items():
                    isolation[key] += count
                writer.write(pair, row, variant)
            if (position + 1) % progress_every == 0 or position + 1 == len(selected):
                print(f'Support ablation: {position + 1:,}/{len(selected):,} held-out rows, '
                      f'{len(records):,} predictions', flush=True)
    finally:
        writer.close()
    predictions = pd.DataFrame(records).sort_values(['variant', 'sample_id', 'model'], kind='stable')
    directories = dict(zip(RUNNERS, (baseline_directory, experiment_directory)))
    parity = {'saved_original': {name: _check_saved_predictions(directories[name],
        predictions.loc[predictions.variant.eq('original') & predictions.model.eq(name)],
        context['heldout'].sample_id, name) for name in RUNNERS}}
    preservation = assert_assets_unchanged(context)
    assert_unchanged(code_paths)
    if weights_before != {name: _ranker_fingerprint(runner.ranker) for name, runner in runners.items()}:
        raise AssertionError('Ranker weights changed during the comparison')
    summary, changes, vessels, markdown = build_support_ablation_report(predictions,
        coverage=coverage, proof=context['proof'], startup=startup, parity=parity, variants=variants)
    summary.update({key: value for key, value in metadata.items() if key not in
        {'complete', 'coverage', 'proof', 'startup'}})
    summary.update(complete=True, weights_unchanged_during_comparison=True,
        training_assets_unchanged=True, protected_files_unchanged=True, diagnostic_code_unchanged=True,
        preservation=preservation, feature_isolation=isolation,
        feature_evidence_counts=writer.counts, total_seconds=time.perf_counter() - started)
    predictions.to_csv(output / 'predictions.csv', index=False, encoding='utf-8-sig')
    changes.to_csv(output / 'row_changes.csv', index=False, encoding='utf-8-sig')
    vessels.to_csv(output / 'vessel_changes.csv', index=False, encoding='utf-8-sig')
    (output / 'report.md').write_text(markdown, encoding='utf-8')
    summary['output_files_sha256'] = {path.name: _hash(path) for path in output.iterdir()
        if path.is_file() and path.name != 'summary.json'}
    _write_json(output / 'summary.json', summary)
    print(f'Completed retrained support-feature comparison without fitting: {output}', flush=True)
    return summary


def add_support_ablation_parser(subparsers):
    parser = subparsers.add_parser('compare-support-ablation',
        help='Compare retrained canonical86 versus no-support82 using the exact same augmented cache')
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--fold', required=True, type=int, choices=range(5))
    parser.add_argument('--train', default='work/data/train.csv')
    parser.add_argument('--catalog', default='work/data/vessel_catalog.csv')
    parser.add_argument('--output', required=True, help='New or empty output directory')
    parser.add_argument('--max-queries', type=int, help='Optional smoke subset; omit for full validation')
    parser.add_argument('--progress-every', type=int, default=100)


def run_support_ablation(args):
    return compare_support_ablation(args.baseline, args.experiment, args.output,
        train_path=args.train, catalog_path=args.catalog, expected_fold=args.fold,
        max_queries=args.max_queries, progress_every=args.progress_every)
