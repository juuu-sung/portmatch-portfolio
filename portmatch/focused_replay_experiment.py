"""Preserved focused baseline plus controlled name/moderate-GT replay arms."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import copy
import gc
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .compound_experiment import ExtraWriter, _group_meta, heavy_training_lock, progress, read_json, sha, write_json
from .data import QUERY_COLUMNS, read_catalog, read_queries
from .hangul_expansion import HangulExpansion
from .joint_search_experiment import iter_extra, query_arrays
from .ranking import VesselRanker
from .ranker_commands import RankedMatcher, score_order
from .retrieval_comparison import _load_model, _validate_partition, _initialize_name_features
from .robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig
from .sampling import SamplingConfig, sample_training_candidates
from .text import has_hangul
from .training_data import iter_shards
from .tonnage_repair_experiment import check as check_parent, repair_shards, verify_fold_assets
from .tonnage_repair_protocol import focused_positions, replay_masses as parent_replay_masses
from .focused_replay_protocol import (ARMS, NEW_ARMS, FAMILIES, EVAL_RECIPES, REGRESSION_RECIPES, VERSION,
    evaluation, input_key, protocol, replay_masses, training_query)

CODE_ROOT = Path(__file__).resolve().parents[1]
FILES = ('portmatch/focused_replay_protocol.py', 'portmatch/focused_replay_experiment.py',
         'portmatch/focused_replay_report.py', 'scripts/run_focused_replay_study.py',
         'scripts/run_compound_study.py', 'scripts/verify_submission.py')


def current_assets():
    paths = [p for directory in ('artifacts', 'submissions') for p in (CODE_ROOT / directory).rglob('*') if p.is_file()]
    paths += [CODE_ROOT / p for p in ('portmatch/focused_runtime.py', 'portmatch/focused_submission_package.py',
                                    'submission_focused/script.py', 'scripts/verify_focused_submission.py')]
    return {str(p.resolve()): sha(p) for p in paths if p.exists()}


def freeze(root, parent):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root.exists():
        raise FileExistsError('Use a new replay study directory')
    old = check_parent(parent, protection=True)
    protected = dict(old['protected_files'])
    protected.update(current_assets())
    if parent in root.parents or root in parent.parents:
        raise ValueError('New study must be separate from its parent')
    for fold in old['planned_folds']:
        directory = parent / f'fold{fold}'
        state = read_json(directory / 'status.json')
        if not state.get('complete') or state['fold'] != fold:
            raise ValueError('All parent folds must be complete')
        verify_fold_assets(directory, state)
    for path in parent.rglob('*'):
        if path.is_file() and path.suffix != '.lock' and '__pycache__' not in path.parts:
            protected[str(path)] = sha(path)
    for info in old['baselines'].values():
        for field in ('directory', 'cache'):
            directory = Path(info[field]).resolve()
            if root == directory or directory in root.parents:
                raise ValueError('Output cannot be inside old model/cache')
    frozen = {'version': VERSION, 'protocol': protocol(), 'parent': str(parent),
              'parent_protocol_sha256': sha(parent / 'protocol.json'), 'planned_folds': old['planned_folds'],
              'data': old['data'], 'baselines': old['baselines'], 'protected_files': protected,
              'implementation': {name: sha(CODE_ROOT / name) for name in FILES}, 'frozen_before_training': True}
    root.mkdir(parents=True)
    write_json(root / 'protocol.json', frozen)
    progress('frozen', output=str(root), protected_files=len(protected))
    return frozen


def check(root, protection=False):
    frozen = read_json(Path(root) / 'protocol.json')
    if frozen['version'] != VERSION or frozen['protocol'] != protocol():
        raise ValueError('Replay protocol changed')
    for relative, expected in frozen['implementation'].items():
        if sha(CODE_ROOT / relative) != expected:
            raise ValueError(f'Implementation changed: {relative}')
    parent = Path(frozen['parent'])
    if sha(parent / 'protocol.json') != frozen['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    check_parent(parent)
    if protection:
        for path, expected in frozen['protected_files'].items():
            if sha(path) != expected:
                raise ValueError(f'Protected asset changed: {path}')
    return frozen


def previous_groups(cache, parent_fold):
    """Bind every old feature group to the exact focused-model training masses."""
    parent_fold = Path(parent_fold)
    grand = Path(read_json(parent_fold.parent / 'protocol.json')['parent']) / parent_fold.name
    base = _group_meta(cache)
    joint = pd.read_csv(grand / 'extra_groups.csv')
    repair = pd.read_csv(parent_fold / 'extra_groups.csv')
    masses = parent_replay_masses(base, joint, repair)
    pieces = []
    for groups, mass, kind in zip((base, joint, repair), masses, ('base', 'previous_joint', 'repair')):
        piece = groups[['source', 'query', 'group', 'language']].copy()
        piece['kind'] = kind
        piece['locked'] = groups.variant.eq('original').to_numpy() if kind == 'base' else kind == 'repair'
        piece['mass'] = mass
        pieces.append(piece)
    previous = pd.concat(pieces, ignore_index=True)
    actual = pd.read_csv(parent_fold / 'focused/query_masses.csv')
    pd.testing.assert_frame_equal(actual, previous[['source', 'query', 'kind', 'mass']], check_dtype=False, atol=1e-12, rtol=0)
    previous['mass'] = actual.mass.to_numpy()  # exactly the saved new-best model's masses
    if previous['query'].duplicated().any():
        raise ValueError('Parent group identities collide')
    return previous, base, joint, repair, grand


def prepare_extra(train, catalog, parts, baseline, dataset, previous, out):
    writer = ExtraWriter(out / 'extra', baseline.ranker.feature_names)
    available = set(previous.loc[~previous.locked, 'source'])
    fitting_ids = set(parts.loc[parts.role.eq('train'), 'sample_id'])
    outer = train[train.sample_id.isin(fitting_ids)].copy()
    partition = parts.set_index('sample_id')
    valid_vessels = set(parts.loc[parts.role.eq('valid'), 'target_vessel_id'])
    targets = dict(zip(catalog.vessel_id, range(len(catalog))))
    expansion = HangulExpansion(catalog)
    sampling = SamplingConfig(**dataset['sampling'])
    logs, proofs = [], []
    for view in ('vessel', 'row'):
        for inner in range(3):
            held = outer[outer.sample_id.map(partition['inner_' + view]).eq(inner)].copy()
            fit = outer[outer.sample_id.map(partition['inner_' + view]).ne(inner)]
            if held.empty:
                continue
            assert not set(fit.sample_id) & set(held.sample_id)
            assert not set(fit.target_vessel_id) & valid_vessels
            if view == 'vessel':
                assert not set(fit.target_vessel_id) & set(held.target_vessel_id)
            proofs.append({'view': view, 'inner': inner, 'fitting_sample_ids': sorted(fit.sample_id),
                           'held_sample_ids': sorted(held.sample_id)})
            pipeline = RobustMatchingPipeline(fit, catalog, RobustRetrievalConfig(**dataset['retrieval']))
            holder = copy.copy(baseline)
            holder.pipeline = pipeline
            _initialize_name_features(holder)
            held['_key'] = held.apply(input_key, axis=1)
            for number, source in enumerate(held.sort_values(['_key', 'sample_id']).to_dict('records'), 1):
                for family in FAMILIES:
                    query, details = training_query(source, view, family)
                    qid = f"{source['sample_id']}::{view}::replay_{family}"
                    record = {'source': source['sample_id'], 'query': qid, 'recipe': family,
                              'language': bool(has_hangul(source['vessel_name'])), 'teacher_fold': -1}
                    log = {**record, **details, 'view': view, 'inner': inner, 'status': 'stored',
                           'target_rescued': False, 'candidates': 0, 'group': 0, 'added_negatives': 0}
                    if source['sample_id'] not in available:
                        log['status'] = 'skipped_no_budget'
                    elif not details['effective']:
                        log['status'] = 'skipped_no_effective_change'
                    else:
                        old_ids, ids, _, matrix, added, _, _ = query_arrays(holder, expansion, query)
                        target = targets[source['target_vessel_id']]
                        log.update(candidates=len(ids), target_rescued=bool(target in ids and target not in old_ids))
                        if target not in ids:
                            log['status'] = 'skipped_missing_target'
                        elif len(ids) < 2:
                            log['status'] = 'skipped_no_negative'
                        else:
                            candidates = pd.DataFrame({'catalog_index': ids})
                            features = pd.DataFrame(matrix, columns=baseline.ranker.feature_names)
                            usual = sample_training_candidates(candidates, features, target, qid, sampling)
                            chosen = focused_positions(candidates, features, target, usual, old_ids)
                            writer.add(matrix, candidates, target, chosen, chosen, record)
                            log.update(group=len(chosen), added_negatives=int(np.sum(added[chosen] & (ids[chosen] != target))))
                    logs.append(log)
                if number % 200 == 0:
                    progress('replay_pool', view=view, inner=inner, done=number, total=len(held))
            del holder, pipeline
            gc.collect()
    writer.flush()
    groups = pd.DataFrame(writer.rows)
    if groups.empty or set(groups.recipe) != set(FAMILIES):
        raise ValueError('Both replay families require usable groups')
    groups.to_csv(out / 'extra_groups.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(logs).to_csv(out / 'replay_log.csv', index=False, encoding='utf-8-sig')
    write_json(out / 'inner_partitions.json', proofs)
    manifest = {'complete': True, 'feature_names': writer.names, 'groups': len(groups),
                'pairs': int(groups.group.sum()), 'files': {f: sha(out / 'extra' / f) for f in writer.files},
                'statuses': dict(Counter(x['status'] for x in logs)), 'same_groups_across_arms': True}
    write_json(out / 'extra_manifest.json', manifest)
    progress('replay_pool_complete', groups=len(groups), pairs=manifest['pairs'])
    return groups, manifest


def fit_arm(cache, parent_fold, previous, base, joint, repair, grand, extra, manifest, baseline, arm, out):
    with heavy_training_lock(out.parent):
        old_mass, new_mass = replay_masses(previous, extra, arm)
        active = new_mass > 0
        groups = np.r_[previous.group, extra.loc[active, 'group']].astype(np.int32)
        languages = np.r_[previous.language, extra.loc[active, 'language']].astype(bool)
        matrix = np.empty((int(groups.sum()), len(baseline.ranker.feature_names)), np.float32)
        labels = np.empty(len(matrix), np.uint8)
        offset = 0
        positions = [read_json(cache / 'metadata.json')['feature_names'].index(n) for n in baseline.ranker.feature_names]
        for shard in iter_shards(cache, 'train'):
            n = len(shard['y'])
            matrix[offset:offset+n] = shard['X'][:, positions]
            labels[offset:offset+n] = shard['y']
            offset += n
        assert offset == int(base.group.sum())
        for group in iter_extra(grand, read_json(grand / 'extra_manifest.json')):
            n = len(group['y'])
            matrix[offset:offset+n] = group['X']
            labels[offset:offset+n] = group['y']
            offset += n
        assert offset == int(base.group.sum() + joint.group.sum())
        observed = []
        for x, y, g in repair_shards(parent_fold, read_json(parent_fold / 'extra_manifest.json'), 'mined'):
            n = len(y)
            matrix[offset:offset+n] = x
            labels[offset:offset+n] = y
            offset += n
            observed.extend(g.tolist())
        assert observed == repair.group.tolist() and offset == int(previous.group.sum())
        cursor = 0
        for x, y, g in repair_shards(out, manifest, 'mined'):
            keep = active[cursor:cursor+len(g)]
            assert np.array_equal(g, extra.group.iloc[cursor:cursor+len(g)])
            mask = np.repeat(keep, g)
            n = int(mask.sum())
            matrix[offset:offset+n] = x[mask]
            labels[offset:offset+n] = y[mask]
            offset += n
            cursor += len(g)
        assert offset == len(matrix) and cursor == len(extra)
        started = time.perf_counter()
        ranker = VesselRanker(baseline.ranker.config)
        ranker.fit(matrix, labels, groups, languages, baseline.ranker.feature_names,
                   query_mass=np.r_[old_mass, new_mass[active]])
        directory = out / arm
        ranker.save(directory)
        mass_frame = mass_records(previous, extra, old_mass, new_mass)
        mass_frame.to_csv(directory / 'query_masses.csv', index=False)
        summary = {'complete': True, 'branch': arm, 'fold': int(out.name.removeprefix('fold')),
                   'protocol_path': str(out.parent / 'protocol.json'), 'groups': len(groups), 'pairs': len(matrix),
                   'retained_parent_groups': len(previous), 'new_groups': int(active.sum()),
                   'original_and_prior_repair_mass_unchanged': True, 'feature_names': ranker.feature_names,
                   'query_masses_sha256': sha(directory / 'query_masses.csv'), 'train_seconds': time.perf_counter()-started}
        write_json(directory / 'replay_model.json', summary)
        ranker.feature_importance().to_csv(directory / 'feature_importance.csv', index=False)
        del matrix, labels
        gc.collect()
        progress('model_complete', branch=arm, seconds=summary['train_seconds'])
        return ranker


def mass_records(previous, extra, old_mass, new_mass):
    active = new_mass > 0
    return pd.concat([previous[['source', 'query', 'kind']].assign(mass=old_mass),
                      extra.loc[active, ['source', 'query']].assign(kind=extra.loc[active, 'recipe'].to_numpy(),
                                                                 mass=new_mass[active])], ignore_index=True)


def score_query(baseline, rankers, expansion, query):
    old_ids, ids, _, matrix, added, base_seconds, extra_seconds = query_arrays(baseline, expansion, query)
    result = {}
    for arm, ranker in rankers.items():
        scores = ranker.predict(matrix, ranker.feature_names)
        order = score_order(scores, ids, baseline.catalog_ids)
        result[arm] = (baseline.catalog_ids[ids[order]], scores[order], bool(added[order[0]]))
    return result, {'base_seconds': base_seconds, 'extra_seconds': extra_seconds,
                    'base_candidates': len(old_ids), 'added_candidates': int(added.sum())}


def evaluate(baseline, rankers, heldout, parent_fold, out):
    expansion = HangulExpansion(baseline.pipeline.catalog)
    _initialize_name_features(baseline)
    saved = pd.read_csv(parent_fold / 'predictions.csv')
    saved = saved[saved.model.eq('focused_search')].set_index(['sample_id', 'variant'])
    memo, rows, timings = OrderedDict(), [], []
    heldout = heldout.copy()
    heldout['_key'] = heldout.apply(input_key, axis=1)
    for number, source in enumerate(heldout.sort_values(['_key', 'sample_id']).to_dict('records'), 1):
        for variant in EVAL_RECIPES:
            query = evaluation(source, variant)
            key = input_key(query)
            if key not in memo:
                memo[key], timing = score_query(baseline, rankers, expansion, query)
                timings.append({'has_hangul': bool(has_hangul(source['vessel_name'])), **timing})
                if len(memo) > 64:
                    memo.popitem(last=False)
            else:
                memo.move_to_end(key)
            for arm, (ordered, scores, added) in memo[key].items():
                location = np.flatnonzero(ordered == source['target_vessel_id'])
                record = {'variant': variant, 'model': arm, 'sample_id': source['sample_id'],
                    'target_vessel_id': source['target_vessel_id'], 'has_hangul': bool(has_hangul(source['vessel_name'])),
                    'predicted_vessel_id': str(ordered[0]), 'candidate_found': bool(len(location)),
                    'target_rank': int(location[0]+1) if len(location) else 0, 'candidate_count': len(ordered),
                    'top_score': float(scores[0]), 'score_margin': float(scores[0]-scores[1]) if len(scores)>1 else None,
                    'prediction_from_added_candidate': added, 'name_changed': str(query['vessel_name']) != str(source['vessel_name']),
                    'deleted_characters': max(0, len(str(source['vessel_name']))-len(str(query['vessel_name'])))}
                if arm == 'focused_search' and variant in REGRESSION_RECIPES:
                    old = saved.loc[(source['sample_id'], variant)]
                    for field in ('predicted_vessel_id', 'candidate_found', 'target_rank', 'candidate_count'):
                        assert record[field] == old[field], (source['sample_id'], variant, field)
                    assert np.isclose(record['top_score'], old.top_score, atol=1e-10, rtol=0)
                rows.append(record)
        if number % 100 == 0:
            progress('evaluate', done=number, total=len(heldout), predictions=len(rows))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(timings).to_csv(out / 'timings.csv', index=False)
    write_json(out / 'evaluation.json', {'complete': True, 'queries': len(heldout), 'predictions': len(frame),
        'previous_twenty_conditions_exact': True, 'same_candidates_and_features_all_arms': True,
        'unique_inputs_computed': len(timings), 'timing_note': 'Shared cached comparison, not a submission benchmark'})
    return frame


class FocusedReplayMatcher:
    def __init__(self, root, fold, arm):
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        root = Path(root).resolve()
        frozen = check(root)
        info = frozen['baselines'][str(fold)]
        for name, expected in info['files'].items():
            if sha(Path(info['directory']) / name) != expected:
                raise ValueError('Baseline bundle changed')
        self.baseline = RankedMatcher(info['directory'])
        if arm == 'focused_search':
            directory = Path(frozen['parent']) / f'fold{fold}/focused'
            for name in ('model.txt', 'ranker.json'):
                if sha(directory / name) != frozen['protected_files'][str(directory / name)]:
                    raise ValueError('Focused baseline changed')
        else:
            directory = root / f'fold{fold}' / arm
            summary = read_json(directory / 'replay_model.json')
            if (not summary['complete'] or summary['branch'] != arm or summary['fold'] != fold
                    or Path(summary['protocol_path']) != root / 'protocol.json'):
                raise ValueError('Wrong/incomplete replay model')
        self.ranker = VesselRanker.load(directory)
        if self.ranker.feature_names != self.baseline.ranker.feature_names or self.ranker.config != self.baseline.ranker.config:
            raise ValueError('Changed ranker schema/config')
        _initialize_name_features(self.baseline)
        self.expansion = HangulExpansion(self.baseline.pipeline.catalog)

    def predict(self, source):
        query = {key: source.get(key) for key in QUERY_COLUMNS}
        _, ids, _, matrix, _, _, _ = query_arrays(self.baseline, self.expansion, query)
        scores = self.ranker.predict(matrix, self.ranker.feature_names)
        order = score_order(scores, ids, self.baseline.catalog_ids)
        return str(self.baseline.catalog_ids[ids[order[0]]])


def run_fold(root, fold):
    root = Path(root).resolve()
    frozen = check(root)
    if fold not in frozen['planned_folds']:
        raise ValueError('Unplanned fold')
    info = frozen['baselines'][str(fold)]
    cache = Path(info['cache'])
    parent = Path(frozen['parent']) / f'fold{fold}'
    grand = Path(read_json(parent.parent / 'protocol.json')['parent']) / parent.name
    for path, expected in frozen['protected_files'].items():
        if any(Path(path).is_relative_to(p) for p in (parent, grand, cache, Path(info['directory']))):
            if sha(path) != expected:
                raise ValueError(f'Parent training asset changed: {path}')
    out = root / f'fold{fold}'
    if out.exists():
        raise FileExistsError('Use runner to preserve/restart an interrupted fold')
    out.mkdir()
    write_json(out / 'status.json', {'complete': False})
    started = time.perf_counter()
    train = read_queries(frozen['data']['train.csv']['path'], labeled=True)
    catalog = read_catalog(frozen['data']['vessel_catalog.csv']['path'])
    baseline, summary, _ = _load_model(info['directory'], frozen['data']['train.csv']['sha256'], frozen['data']['vessel_catalog.csv']['sha256'])
    if summary['dataset']['outer_fold'] != fold or summary['dataset']['inner_folds'] != 3:
        raise ValueError('Wrong partition')
    heldout = _validate_partition(baseline, summary, train, catalog)
    parts = pd.read_csv(cache / 'partitions.csv')
    assert set(heldout.sample_id) == set(parts.loc[parts.role.eq('valid'), 'sample_id'])
    previous, base, joint, repair, grand = previous_groups(cache, parent)
    extra, manifest = prepare_extra(train, catalog, parts, baseline, summary['dataset'], previous, out)
    rankers = {'focused_search': VesselRanker.load(parent / 'focused')}
    for arm in NEW_ARMS:
        rankers[arm] = fit_arm(cache, parent, previous, base, joint, repair, grand, extra, manifest, baseline, arm, out)
    predictions = evaluate(baseline, rankers, heldout, parent, out)
    del rankers
    gc.collect()
    sample = pd.concat([heldout[heldout.vessel_name.map(has_hangul).eq(lang)].head(1) for lang in (True, False)])
    variants = ('original', 'name_delete', 'unseen_swap_dual_gt_x0_70', 'fresh_double_delete_gt_x0_55', 'fresh_double_delete_gt_x1_50')
    records = [(r, v) for r in sample.to_dict('records') for v in variants]
    proofs = []
    for arm in NEW_ARMS:
        matcher = FocusedReplayMatcher(root, fold, arm)
        inputs = [evaluation(r, v) for r, v in records]
        expected = predictions[predictions.model.eq(arm)].set_index(['sample_id', 'variant']).predicted_vessel_id
        forward = [matcher.predict(q) for q in inputs]
        reverse = [matcher.predict(q) for q in reversed(inputs)][::-1]
        changed = [matcher.predict({**q, 'sample_id': 'CHANGED', 'arrival_year': 1900, 'target_vessel_id': 'IGNORED'}) for q in inputs]
        assert forward == reverse == changed == [expected.loc[(r['sample_id'], v)] for r, v in records]
        proofs.append({'arm': arm, 'queries': len(inputs), 'native_reload_order_id_year_label_parity': True})
        del matcher
    files = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'status.json'}
    write_json(out / 'status.json', {'complete': True, 'fold': fold, 'seconds': time.perf_counter()-started,
                                    'files': files, 'independence_checks': proofs})
    progress('fold_complete', fold=fold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'report'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    if args.command == 'run':
        if args.fold is None:
            parser.error('--fold required')
        run_fold(args.output, args.fold)
    else:
        from .focused_replay_report import report
        report(args.output)


if __name__ == '__main__':
    main()
