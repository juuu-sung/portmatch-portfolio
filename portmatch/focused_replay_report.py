"""Audit replay inputs and report target gains, regressions and factorial effects."""
from pathlib import Path
import numpy as np
import pandas as pd

from .compound_experiment import read_json, sha, write_json, progress
from .ranking import ranking_metrics
from .text import has_hangul
from .focused_replay_experiment import check, previous_groups, mass_records, verify_fold_assets
from .focused_replay_protocol import (ARMS, NEW_ARMS, FAMILIES, EVAL_RECIPES, REGRESSION_RECIPES,
    FRESH_RECIPES, PAIRS, TARGETS, CORE_RETENTION, RETENTION, TOLERANCE_PP, SEED, replay_masses, training_query)


def audit_fold(root, fold, frozen, train, catalog):
    out = root / f'fold{fold}'
    state = read_json(out / 'status.json')
    verify_fold_assets(out, state)
    assert state['complete'] and state['fold'] == fold
    assert {x['arm'] for x in state['independence_checks']} == set(NEW_ARMS)
    assert all(x['native_reload_order_id_year_label_parity'] for x in state['independence_checks'])
    parent = Path(frozen['parent']) / out.name
    cache = Path(frozen['baselines'][str(fold)]['cache'])
    parts = pd.read_csv(cache / 'partitions.csv').set_index('sample_id')
    fitting = set(parts.index[parts.role.eq('train')])
    valid = set(parts.index[parts.role.eq('valid')])
    sources = train.set_index('sample_id', drop=False)
    targets = sources.target_vessel_id
    assert not set(targets.loc[list(fitting)]) & set(targets.loc[list(valid)])
    previous, _, _, _, _ = previous_groups(cache, parent)
    extra = pd.read_csv(out / 'extra_groups.csv')
    logs = pd.read_csv(out / 'replay_log.csv')
    manifest = read_json(out / 'extra_manifest.json')
    assert logs['query'].is_unique and set(logs.source) == fitting
    assert set(logs['query']) == {f'{s}::{v}::replay_{f}' for s in fitting for v in ('vessel', 'row') for f in FAMILIES}
    assert extra['query'].is_unique and set(extra.source).issubset(fitting)
    assert set(logs.loc[logs.status.eq('stored'), 'query']) == set(extra['query'])
    for row in logs.itertuples():
        _, detail = training_query(sources.loc[row.source].to_dict(), row.view, row.recipe)
        for field, expected in detail.items():
            assert getattr(row, field) == expected, (row.query, field)
    for proof in read_json(out / 'inner_partitions.json'):
        held = set(parts.index[parts.role.eq('train') & parts['inner_' + proof['view']].eq(proof['inner'])])
        fit = fitting - held
        assert set(proof['held_sample_ids']) == held and set(proof['fitting_sample_ids']) == fit
        if proof['view'] == 'vessel':
            assert not set(targets.loc[list(held)]) & set(targets.loc[list(fit)])
    target_indices = dict(zip(catalog.vessel_id, range(len(catalog))))
    observed, counts, families = [], [], []
    for name, expected in manifest['files'].items():
        assert sha(out / 'extra' / name) == expected
        with np.load(out / 'extra' / name, allow_pickle=False) as z:
            offset = 0
            for i, length in enumerate(z['union_group']):
                section = slice(offset, offset + int(length))
                offset += int(length)
                ids, labels = z['catalog_index'][section], z['y'][section]
                h, m = z['heuristic'][section], z['mined'][section]
                source = str(z['source'][i])
                assert source in fitting and len(set(ids)) == len(ids)
                assert np.array_equal(labels, (ids == target_indices[targets[source]]).astype(np.uint8))
                assert labels[m].sum() == 1 and m.sum() >= 2 and np.array_equal(h, m)
                observed.append(str(z['query'][i]))
                counts.append(int(m.sum()))
                families.append(str(z['recipe'][i]))
    assert observed == extra['query'].tolist() and counts == extra.group.tolist() and families == extra.recipe.tolist()
    assert len(observed) == manifest['groups'] and sum(counts) == manifest['pairs']
    all_masses = {}
    for arm in NEW_ARMS:
        old, added = replay_masses(previous, extra, arm)
        all_masses[arm] = added
        expected = mass_records(previous, extra, old, added)
        actual = pd.read_csv(out / arm / 'query_masses.csv')
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=0)
        summary = read_json(out / arm / 'replay_model.json')
        assert summary['groups'] == len(expected) and summary['retained_parent_groups'] == len(previous)
        assert summary['pairs'] == int(previous.group.sum() + extra.loc[added > 0, 'group'].sum())
        assert summary['query_masses_sha256'] == sha(out / arm / 'query_masses.csv')
        rank = read_json(out / arm / 'ranker.json')
        prior = read_json(parent / 'focused/ranker.json')
        assert rank['config'] == prior['config'] and rank['feature_names'] == prior['feature_names']
    for family in FAMILIES:
        mask = extra.recipe.eq(family)
        np.testing.assert_array_equal(all_masses['combined'][mask], all_masses[family][mask])
    predictions = pd.read_csv(out / 'predictions.csv').assign(fold=fold)
    assert set(predictions.sample_id) == valid and len(predictions) == len(valid) * len(EVAL_RECIPES) * len(ARMS)
    assert set(predictions.model) == set(ARMS) and set(predictions.variant) == set(EVAL_RECIPES)
    assert not predictions.duplicated(['sample_id', 'variant', 'model']).any()
    assert predictions.target_vessel_id.equals(predictions.sample_id.map(targets))
    assert predictions.has_hangul.equals(predictions.sample_id.map(sources.vessel_name.map(has_hangul)))
    assert predictions.predicted_vessel_id.isin(catalog.vessel_id).all()
    old_predictions = pd.read_csv(parent / 'predictions.csv')
    old_predictions = old_predictions[old_predictions.model.eq('focused_search')].set_index(['sample_id', 'variant'])
    baseline = predictions[predictions.model.eq('focused_search') & predictions.variant.isin(REGRESSION_RECIPES)].set_index(['sample_id', 'variant'])
    prior = old_predictions.loc[baseline.index]
    for field in ('predicted_vessel_id', 'candidate_found', 'target_rank', 'candidate_count'):
        np.testing.assert_array_equal(baseline[field], prior[field])
    np.testing.assert_allclose(baseline.top_score, prior.top_score, atol=1e-10, rtol=0)
    for field in ('candidate_found', 'candidate_count'):
        wide = predictions.pivot(index=['sample_id', 'variant'], columns='model', values=field)
        assert all(wide[arm].equals(wide.focused_search) for arm in NEW_ARMS)
    return predictions, {'fold': fold, 'retained_groups': len(previous), 'new_groups': len(extra),
                         'new_pairs': sum(counts), 'statuses': logs.status.value_counts().to_dict()}, logs


def report(root, replicates=5000):
    if replicates < 100:
        raise ValueError('Use at least 100 bootstrap draws')
    root = Path(root).resolve()
    frozen = check(root, protection=True)
    train = pd.read_csv(frozen['data']['train.csv']['path'])
    catalog = pd.read_csv(frozen['data']['vessel_catalog.csv']['path'])
    frames, audits, logs, folds = [], [], [], []
    for fold in frozen['planned_folds']:
        path = root / f'fold{fold}/status.json'
        if not path.exists() or not read_json(path).get('complete'):
            continue
        frame, audit, log = audit_fold(root, fold, frozen, train, catalog)
        frames.append(frame)
        audits.append(audit)
        logs.append(log.assign(fold=fold))
        folds.append(fold)
    if not frames:
        raise ValueError('No complete folds')
    p = pd.concat(frames, ignore_index=True)
    p['correct'] = p.predicted_vessel_id.eq(p.target_vessel_id)
    clean = p[p.variant.eq('original') & p.model.eq('focused_search')]
    assert clean.sample_id.is_unique
    vessels = sorted(clean.target_vessel_id.unique())
    counts = clean.assign(h=clean.has_hangul.astype(int), e=(~clean.has_hangul).astype(int)).groupby('target_vessel_id')[['h', 'e']].sum().reindex(vessels).to_numpy(float)
    metrics, fold_metrics, errors, paired, changes, interactions, terms, vectors = [], [], [], [], [], [], [], []
    names = train.set_index('sample_id').vessel_name.to_dict()
    catalog_names = catalog.set_index('vessel_id').vessel_name.to_dict()
    for variant, data in p.groupby('variant', sort=False):
        measured = {}
        for arm, rows in data.groupby('model'):
            m = ranking_metrics(rows)
            measured[arm] = m['weighted_top1']
            metrics.append({'variant': variant, 'evaluation_set': 'fresh' if variant in FRESH_RECIPES else 'regression',
                'model': arm, 'weighted_top1': m['weighted_top1'], 'weighted_candidate_recall': m['weighted_candidate_recall'],
                'hangul_correct': m['hangul']['correct_top1'], 'hangul_queries': m['hangul']['queries'],
                'non_hangul_correct': m['non_hangul']['correct_top1'], 'non_hangul_queries': m['non_hangul']['queries'],
                'name_changed_queries': int(rows.name_changed.sum()), 'two_letter_deletions': int(rows.deleted_characters.eq(2).sum())})
            for fold, sample in rows.groupby('fold'):
                fold_metrics.append({'variant': variant, 'model': arm, 'fold': int(fold),
                                     'weighted_top1': ranking_metrics(sample)['weighted_top1']})
            for (language, changed), sample in rows.groupby(['has_hangul', 'name_changed']):
                errors.append({'variant': variant, 'model': arm, 'has_hangul': bool(language), 'name_changed': bool(changed),
                    'queries': len(sample), 'correct': int(sample.correct.sum()), 'candidate_missing': int((~sample.candidate_found).sum()),
                    'ranking_errors': int((sample.candidate_found & ~sample.correct).sum())})
        wide = data.pivot(index=['fold', 'sample_id', 'target_vessel_id', 'has_hangul'], columns='model', values='predicted_vessel_id').reset_index()
        success = {arm: wide[arm].eq(wide.target_vessel_id).astype(int) for arm in ARMS}
        for before, after in PAIRS:
            d = success[after] - success[before]
            term = {'variant': variant, 'reference': before, 'experiment': after,
                    'delta_pp': 100 * (measured[after] - measured[before]),
                    'corrected': int(d.eq(1).sum()), 'regressed': int(d.eq(-1).sum()),
                    'corrected_vessels': int(wide.loc[d.eq(1), 'target_vessel_id'].nunique()),
                    'regressed_vessels': int(wide.loc[d.eq(-1), 'target_vessel_id'].nunique())}
            paired.append(term)
            terms.append(term)
            vectors.append(wide.assign(hd=d*wide.has_hangul, ed=d*(~wide.has_hangul)).groupby('target_vessel_id')[['hd', 'ed']].sum().reindex(vessels, fill_value=0).to_numpy(float))
            for row in wide.loc[wide[before].ne(wide[after])].to_dict('records'):
                changes.append({'variant': variant, 'reference': before, 'experiment': after, 'sample_id': row['sample_id'],
                    'fold': row['fold'], 'target_vessel_id': row['target_vessel_id'], 'has_hangul': row['has_hangul'],
                    'query_name': names[row['sample_id']], 'target_name': catalog_names[row['target_vessel_id']],
                    'reference_prediction': catalog_names[row[before]], 'experiment_prediction': catalog_names[row[after]],
                    'status': 'corrected' if row[after] == row['target_vessel_id'] else 'regressed' if row[before] == row['target_vessel_id'] else 'still_wrong'})
        d = success['combined'] - success['names'] - success['moderate'] + success['focused_search']
        term = {'variant': variant, 'reference': 'additive', 'experiment': 'interaction',
                'delta_pp': 100 * (measured['combined'] - measured['names'] - measured['moderate'] + measured['focused_search'])}
        interactions.append(term)
        terms.append(term)
        vectors.append(wide.assign(hd=d*wide.has_hangul, ed=d*(~wide.has_hangul)).groupby('target_vessel_id')[['hd', 'ed']].sum().reindex(vessels, fill_value=0).to_numpy(float))
    if not (counts.sum(axis=0) > 0).all():
        raise ValueError('Both language strata required')
    matrix = np.concatenate([counts, *vectors], axis=1)
    rng = np.random.default_rng(SEED)
    draws, accepted = [], 0
    while accepted < replicates:
        z = rng.poisson(1, size=(min(100, replicates-accepted), len(vessels))) @ matrix
        z = z[(z[:, 0] > 0) & (z[:, 1] > 0)]
        if len(z):
            draws.append(100 * (.7*z[:, 2::2]/z[:, 0, None] + .3*z[:, 3::2]/z[:, 1, None]))
            accepted += len(z)
    draws = np.concatenate(draws)
    intervals = [{**term, 'vessel_bootstrap_95pct_pp': np.quantile(draws[:, i], [.025, .975]).tolist()} for i, term in enumerate(terms)]
    lookup = {(x['variant'], x['reference'], x['experiment']): x for x in intervals}
    fm = pd.DataFrame(fold_metrics).set_index(['fold', 'variant', 'model'])
    gates = []
    for arm in NEW_ARMS:
        target = [lookup[v, 'focused_search', arm] for v in TARGETS]
        keep = [lookup[v, 'focused_search', arm] for v in RETENTION]
        core = [lookup[v, 'focused_search', arm] for v in CORE_RETENTION]
        regressions = [x['variant'] for x in keep if x['delta_pp'] < -TOLERANCE_PP]
        uncertain = [x['variant'] for x in keep if x['vessel_bootstrap_95pct_pp'][0] < -TOLERANCE_PP]
        core_loss = [x['variant'] for x in core if x['delta_pp'] < -1e-10]
        fold0_core = {v: 100*(fm.loc[(0, v, arm), 'weighted_top1']-fm.loc[(0, v, 'focused_search'), 'weighted_top1']) for v in CORE_RETENTION} if 0 in folds else {}
        fold0_targets = {v: 100*(fm.loc[(0, v, arm), 'weighted_top1']-fm.loc[(0, v, 'focused_search'), 'weighted_top1']) for v in TARGETS} if 0 in folds else {}
        full = folds == [0, 1, 2, 3, 4]
        point = (all(x['delta_pp'] > 0 for x in target) and not core_loss and not regressions
                 and bool(fold0_core) and min(fold0_core.values()) >= -1e-10 and min(fold0_targets.values()) >= -1e-10)
        gates.append({'model': arm, 'reference': 'focused_search', 'target_comparisons': target,
            'core_retention_comparisons': core, 'core_point_loss_conditions': core_loss,
            'retention_margin_pp': TOLERANCE_PP, 'regressed_conditions': regressions,
            'retention_not_confirmed_conditions': uncertain, 'fold0_core_deltas_pp': fold0_core,
            'fold0_target_deltas_pp': fold0_targets, 'all_five_folds': full,
            'development_point_gate_pass': bool(full and point),
            'strict_development_gate_pass': bool(full and point and not uncertain and all(x['vessel_bootstrap_95pct_pp'][0] > 0 for x in target)),
            'automatic_promotion': False})
    change_columns = ['variant', 'reference', 'experiment', 'sample_id', 'fold', 'target_vessel_id', 'has_hangul',
                      'query_name', 'target_name', 'reference_prediction', 'experiment_prediction', 'status']
    for name, records in [('metrics', metrics), ('fold_metrics', fold_metrics), ('error_breakdown', errors),
                          ('paired_changes', paired), ('changed_rows', changes), ('interactions', interactions)]:
        pd.DataFrame(records, columns=change_columns if name == 'changed_rows' else None).to_csv(root / f'{name}.csv', index=False, encoding='utf-8-sig')
    pd.concat(logs, ignore_index=True).groupby(['fold', 'recipe', 'language', 'operation', 'factor', 'status']).agg(
        groups=('query', 'size'), rescued_targets=('target_rescued', 'sum'), added_negatives=('added_negatives', 'sum')).reset_index().to_csv(root / 'training_coverage.csv', index=False)
    write_json(root / 'confidence_intervals.json', {'replicates': replicates, 'seed': SEED,
        'method': 'Paired Poisson vessel-cluster bootstrap', 'comparisons': intervals,
        'caveat': 'Development data; not adjusted for model selection/multiple tests/Public distribution changes'})
    write_json(root / 'retention_gates.json', {'targets': list(TARGETS), 'core_retention': list(CORE_RETENTION),
        'retention': list(RETENTION), 'gates': gates, 'automatic_promotion': False})
    write_json(root / 'comparison.json', {'complete_for_folds': folds, 'full_five_fold': folds == [0,1,2,3,4],
        'queries': len(clean), 'predictions': len(p), 'baseline': 'focused_search', 'public_submission_performed': False})
    write_json(root / 'analysis_audit.json', {'complete': True, 'folds': folds, 'per_fold': audits, 'predictions': len(p),
        'protected_files_verified': len(frozen['protected_files']), 'old_assets_unchanged': True,
        'previous_twenty_conditions_exact': True, 'original_and_prior_repair_mass_unchanged': True,
        'family_mass_equal_single_and_combined': True, 'candidate_sets_identical_all_arms': True,
        'analysis_code_sha256': sha(__file__)})
    progress('report_complete', folds=folds, predictions=len(p), protected_files=len(frozen['protected_files']))
