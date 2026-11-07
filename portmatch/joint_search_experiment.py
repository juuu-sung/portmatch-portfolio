"""Isolated 2x2 study: joint-error training versus append-only Hangul search."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from dataclasses import asdict
import gc
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .compound_experiment import (ExtraWriter, _group_meta, check_protocol as check_parent,
    fit_branch, progress, read_json, sha, write_json)
from .compound_mining import expanded_masses
from .compound_protocol import input_key, EVAL_RECIPES as OLD_RECIPES
from .data import QUERY_COLUMNS, read_catalog, read_queries
from .feature_transforms import transform_features
from .hangul_expansion import ExpansionConfig, HangulExpansion
from .joint_search_protocol import ARMS, EVAL_RECIPES, VERSION, evaluation_query, joint_variant, protocol
from .ranker_commands import RankedMatcher, score_order
from .ranking import VesselRanker, ranking_metrics
from .retrieval_comparison import _load_model, _validate_partition, _initialize_name_features
from .robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig
from .sampling import SamplingConfig, sample_training_candidates
from .text import has_hangul

CODE_ROOT = Path(__file__).resolve().parents[1]
FILES = ('portmatch/joint_search_protocol.py', 'portmatch/hangul_expansion.py',
         'portmatch/joint_search_experiment.py', 'portmatch/joint_search_report.py',
         'scripts/run_joint_search_study.py', 'scripts/run_compound_study.py')


def freeze(root, parent):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if root.exists():
        raise FileExistsError('Use a new experiment directory')
    old = check_parent(parent, protection=True)
    if parent in root.parents:
        raise ValueError('New study must be outside the preserved parent')
    protected = dict(old['protected_files'])
    for path in parent.iterdir():
        if path.is_file() and path.suffix != '.lock':
            protected[str(path)] = sha(path)
    for fold in old['planned_folds']:
        out = parent / f'fold{fold}'
        if not read_json(out/'status.json').get('complete'):
            raise ValueError('Parent study must have completed every planned fold')
        control = VesselRanker.load(out/'expanded')
        baseline = old['baselines'][str(fold)]
        for directory in (Path(baseline['directory']), Path(baseline['cache'])):
            if directory == root or directory in root.parents:
                raise ValueError('Output would be inside an existing model/cache')
        assert len(control.feature_names) == 86
        for path in out.rglob('*'):
            if path.is_file():
                protected[str(path)] = sha(path)
        cache = Path(baseline['cache'])
        metadata = read_json(cache/'metadata.json')
        for relative in ('metadata.json', 'partitions.csv', *metadata['train_shards']):
            protected[str(cache/relative)] = sha(cache/relative)
    frozen = {'version': VERSION, 'protocol': protocol(), 'parent': str(parent),
        'parent_protocol_sha256': sha(parent/'protocol.json'),
        'planned_folds': old['planned_folds'], 'data': old['data'], 'baselines': old['baselines'],
        'protected_files': protected, 'implementation': {f: sha(CODE_ROOT/f) for f in FILES},
        'expansion': asdict(ExpansionConfig()), 'frozen_before_training': True}
    root.mkdir(parents=True)
    write_json(root/'protocol.json', frozen)
    progress('frozen', protected_files=len(protected), output=str(root))
    return frozen


def check(root, protection=False):
    frozen = read_json(Path(root)/'protocol.json')
    if frozen['version'] != VERSION or frozen['protocol'] != protocol():
        raise ValueError('Study protocol changed')
    if frozen['expansion'] != asdict(ExpansionConfig()):
        raise ValueError('Search configuration differs')
    for relative, expected in frozen['implementation'].items():
        if sha(CODE_ROOT/relative) != expected:
            raise ValueError(f'Implementation changed after freeze: {relative}')
    parent = Path(frozen['parent'])
    if sha(parent/'protocol.json') != frozen['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    check_parent(parent)
    if protection:
        for path, expected in frozen['protected_files'].items():
            if sha(path) != expected:
                raise ValueError(f'Protected file changed: {path}')
    return frozen


def iter_extra(directory, manifest):
    for relative, expected in manifest['files'].items():
        path = Path(directory)/'extra'/relative
        if sha(path) != expected:
            raise ValueError(f'Extra shard changed: {path}')
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        offset = 0
        for i, count in enumerate(arrays['union_group']):
            section = slice(offset, offset+int(count))
            chosen = arrays['heuristic'][section]
            record = {key: arrays[key][i].item() for key in ('source','query','recipe','language','teacher_fold')}
            yield {**record, 'X': arrays['X'][section][chosen], 'ids': arrays['catalog_index'][section][chosen],
                   'y': arrays['y'][section][chosen]}
            offset += int(count)


def matrix_for(baseline, pipeline, query, candidates):
    features = pipeline.features(query, candidates)
    features = transform_features(baseline.feature_transform, pipeline, query, candidates, features)
    matrix = baseline.selection.apply(features.to_numpy(dtype=np.float32), features.columns.tolist())
    return features, matrix


def prepare_joint(train, catalog, partitions, baseline, dataset, parent_fold, output):
    original = pd.read_csv(parent_fold/'extra_groups.csv')
    manifest = read_json(parent_fold/'extra_manifest.json')
    lookup = train.set_index('sample_id').to_dict('index')
    partition = partitions.set_index('sample_id')
    training_ids = set(partitions.loc[partitions.role.eq('train'), 'sample_id'])
    assert set(original.source).issubset(training_ids)
    selected = original[original.recipe.eq('dual_gt_noise')].copy()
    selected['view'] = selected['query'].str.split('::').str[1]
    ids = dict(zip(catalog.vessel_id, range(len(catalog))))
    sampling = SamplingConfig(**dataset['sampling'])
    patches, logs, proofs = {}, [], []
    for view in ('vessel','row'):
        for inner in range(3):
            held = selected[selected.view.eq(view) & selected.source.map(partition['inner_'+view]).eq(inner)]
            if held.empty:
                continue
            fitting = train[train.sample_id.isin(training_ids) & train.sample_id.map(partition['inner_'+view]).ne(inner)]
            assert not set(fitting.sample_id) & set(held.source)
            valid_vessels = set(partitions.loc[partitions.role.eq('valid'),'target_vessel_id'])
            assert not set(fitting.target_vessel_id) & valid_vessels
            if view == 'vessel':
                assert not set(fitting.target_vessel_id) & {lookup[s]['target_vessel_id'] for s in held.source}
            proofs.append({'view':view, 'inner':inner, 'fitting_sample_ids':sorted(fitting.sample_id),
                           'held_sample_ids':sorted(held.source)})
            pipeline = RobustMatchingPipeline(fitting, catalog, RobustRetrievalConfig(**dataset['retrieval']))
            _initialize_name_features(type('Holder',(),{'pipeline':pipeline})())
            for number, old in enumerate(held.to_dict('records'),1):
                source = {'sample_id':old['source'], **lookup[old['source']]}
                query, info = joint_variant(source, view)
                state, candidates_count = 'fallback_no_joint_change', 0
                if info['effective']:
                    candidates = pipeline.retrieve(query)
                    candidates_count = len(candidates)
                    target = ids[source['target_vessel_id']]
                    if target not in set(candidates.catalog_index):
                        state = 'fallback_missing_target'
                    else:
                        features, matrix = matrix_for(baseline, pipeline, query, candidates)
                        positions = sample_training_candidates(candidates, features, target, old['query'], sampling)
                        state = 'fallback_group_budget'
                        if len(positions) == old['group']:
                            patches[old['query']] = (matrix[positions].copy(), candidates.catalog_index.to_numpy()[positions].copy())
                            state = 'replaced'
                logs.append({'source':old['source'], 'query':old['query'], 'view':view, 'inner':inner,
                             'status':state, 'candidate_count':candidates_count, **info})
                if number % 100 == 0:
                    progress('joint_data', view=view, inner=inner, done=number, total=len(held))
            del pipeline
            gc.collect()
    writer = ExtraWriter(output/'extra', baseline.ranker.feature_names)
    unchanged = 0
    for old in iter_extra(parent_fold, manifest):
        matrix, indices = patches.get(old['query'], (old['X'], old['ids']))
        replaced = old['query'] in patches
        unchanged += not replaced
        row = {key:old[key] for key in ('source','query','recipe','language')}
        row.update(recipe='joint_name_gt' if replaced else old['recipe'], teacher_fold=-1)
        positions = np.arange(len(indices))
        writer.add(matrix, pd.DataFrame({'catalog_index':indices}), ids[lookup[old['source']]['target_vessel_id']],
                   positions, positions, row)
    writer.flush()
    extra = pd.DataFrame(writer.rows)
    for field in ('source','query','group','language'):
        assert extra[field].equals(original[field]), field
    log = pd.DataFrame(logs, columns=['source','query','view','inner','status','candidate_count','operation','factor','effective'])
    log.to_csv(output/'joint_log.csv', index=False, encoding='utf-8-sig')
    extra.to_csv(output/'extra_groups.csv', index=False, encoding='utf-8-sig')
    write_json(output/'inner_partitions.json', proofs)
    result = {'complete':True, 'feature_names':writer.names, 'files':{f:sha(output/'extra'/f) for f in writer.files},
              'groups':len(extra), 'pairs_per_branch':int(extra.group.sum()), 'eligible_groups':len(selected),
              'replaced_groups':len(patches), 'unchanged_groups':unchanged, 'statuses':dict(Counter(log.status)),
              'group_order_pair_count_and_mass_preserved':True, 'learned_hard_negative_mining':False}
    write_json(output/'extra_manifest.json',result)
    progress('joint_data_complete', replaced=len(patches), eligible=len(selected), total=len(extra))
    return extra, result


def query_arrays(baseline, expansion, query):
    start = time.perf_counter()
    old = baseline.pipeline.retrieve(query)
    _, matrix = matrix_for(baseline, baseline.pipeline, query, old)
    old_seconds = time.perf_counter()-start
    start = time.perf_counter()
    full = expansion.append(query, old) if expansion is not None else old.copy()
    old_ids = old.catalog_index.to_numpy(dtype=np.int64)
    full_ids = full.catalog_index.to_numpy(dtype=np.int64)
    old_positions = np.searchsorted(full_ids, old_ids)
    added_mask = ~np.isin(full_ids, old_ids)
    full_matrix = np.empty((len(full), matrix.shape[1]), dtype=np.float32)
    full_matrix[old_positions] = matrix
    if added_mask.any():
        _, added = matrix_for(baseline, baseline.pipeline, query, full.loc[added_mask].reset_index(drop=True))
        full_matrix[added_mask] = added
    assert np.array_equal(full_matrix[old_positions], matrix, equal_nan=True)
    return old_ids, full_ids, matrix, full_matrix, added_mask, old_seconds, time.perf_counter()-start


def score_query(baseline, control, joint, expansion, query):
    old_ids, full_ids, old_x, full_x, added, old_time, new_time = query_arrays(baseline, expansion, query)
    catalog_ids = baseline.catalog_ids
    result = {}; old_set=set(old_ids)
    prior_scores = {}
    for arm, ranker, indices, matrix in (
        ('best',baseline.ranker,old_ids,old_x), ('control',control,old_ids,old_x),
        ('augmentation',joint,old_ids,old_x), ('retrieval',control,full_ids,full_x), ('both',joint,full_ids,full_x)):
        if arm in ('retrieval','both') and not added.any():
            scores = prior_scores['control' if arm=='retrieval' else 'augmentation']
        else:
            scores = ranker.predict(matrix, ranker.feature_names)
        if arm in ('retrieval','both'):
            reference = 'control' if arm == 'retrieval' else 'augmentation'
            assert np.allclose(scores[np.searchsorted(full_ids,old_ids)],prior_scores[reference],atol=1e-10,rtol=0)
        else:
            prior_scores[arm] = scores
        order = score_order(scores, indices, catalog_ids)
        result[arm] = (catalog_ids[indices[order]], scores[order], bool(indices[order[0]] not in old_set))
    return result, {'base_retrieval_features_seconds':old_time, 'extra_retrieval_features_seconds':new_time,
                    'base_candidates':len(old_ids), 'added_candidates':int(added.sum())}


def evaluate(baseline, control, joint, heldout, parent_fold, output):
    expansion = HangulExpansion(baseline.pipeline.catalog)
    saved = pd.read_csv(parent_fold/'predictions.csv').set_index(['sample_id','variant','model'])
    _initialize_name_features(baseline)
    memo, rows, timings, effects = OrderedDict(), [], [], Counter()
    heldout = heldout.copy(); heldout['_key'] = heldout.apply(input_key, axis=1)
    for number, source in enumerate(heldout.sort_values(['_key','sample_id']).to_dict('records'),1):
        for variant in EVAL_RECIPES:
            query = evaluation_query(source, variant); key = input_key(query)
            effects[variant] += input_key(source) != key
            if key not in memo:
                memo[key], timing = score_query(baseline,control,joint,expansion,query)
                timings.append({'has_hangul':bool(has_hangul(source['vessel_name'])), **timing})
                if len(memo)>64:
                    memo.popitem(last=False)
            else:
                memo.move_to_end(key)
            for arm,(ordered,scores,added) in memo[key].items():
                location = np.flatnonzero(ordered == source['target_vessel_id'])
                record = {'variant':variant,'model':arm,'sample_id':source['sample_id'],
                    'target_vessel_id':source['target_vessel_id'],'has_hangul':bool(has_hangul(source['vessel_name'])),
                    'predicted_vessel_id':str(ordered[0]),'candidate_found':bool(len(location)),
                    'target_rank':int(location[0]+1) if len(location) else 0,'candidate_count':len(ordered),
                    'top_score':float(scores[0]),'score_margin':float(scores[0]-scores[1]) if len(scores)>1 else None,
                    'prediction_from_added_candidate':added,
                    'name_changed':str(query.get('vessel_name'))!=str(source.get('vessel_name')),
                    'deleted_characters':max(0,len(str(source.get('vessel_name')))-len(str(query.get('vessel_name'))))}
                if arm in ('best','control') and variant in OLD_RECIPES:
                    old = saved.loc[(source['sample_id'],variant,'baseline' if arm=='best' else 'expanded')]
                    for field in ('predicted_vessel_id','candidate_found','target_rank','candidate_count'):
                        assert record[field] == old[field], (source['sample_id'],variant,field)
                    assert np.isclose(record['top_score'], old.top_score, atol=1e-10,rtol=0)
                rows.append(record)
        if number%100==0:
            progress('evaluate',done=number,total=len(heldout),predictions=len(rows))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(timings).to_csv(output/'timings.csv',index=False)
    metrics = {v:{m:ranking_metrics(frame[frame.variant.eq(v)&frame.model.eq(m)]) for m in ARMS} for v in EVAL_RECIPES}
    write_json(output/'evaluation.json',{'complete':True,'metrics':metrics,'queries':len(heldout),
        'changed_inputs_per_variant':dict(effects),'old_twelve_condition_prediction_parity':True,
        'common_candidate_features_and_scores_unchanged':True,'unique_inputs_computed':len(timings),
        'timing_note':'Memoized shared feature comparison, not submission runtime; extra cost measured per computed input'})
    return frame


class JointSearchMatcher:
    def __init__(self, root, fold, arm):
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        self.arm = arm
        frozen = check(root)
        info = frozen['baselines'][str(fold)]
        for name,expected in info['files'].items():
            if sha(Path(info['directory'])/name) != expected:
                raise ValueError('Baseline bundle changed')
        self.baseline = RankedMatcher(info['directory'])
        directory = Path(root)/f'fold{fold}'/'joint' if arm in ('augmentation','both') else Path(frozen['parent'])/f'fold{fold}'/'expanded'
        if arm in ('augmentation','both'):
            summary=read_json(directory/'compound_model.json')
            if not summary.get('complete') or summary['fold']!=fold or Path(summary['protocol_path']).resolve()!=Path(root).resolve()/'protocol.json':
                raise ValueError('Joint model belongs to a different or incomplete study')
        elif arm!='best':
            for name in ('model.txt','ranker.json'):
                if sha(directory/name)!=frozen['protected_files'][str(directory/name)]:
                    raise ValueError('Control model changed')
        self.ranker = self.baseline.ranker if arm=='best' else VesselRanker.load(directory)
        if self.ranker.feature_names != self.baseline.ranker.feature_names or self.ranker.config!=self.baseline.ranker.config:
            raise ValueError('Feature schema differs')
        _initialize_name_features(self.baseline)
        self.expansion = HangulExpansion(self.baseline.pipeline.catalog) if arm in ('retrieval','both') else None

    def predict(self, source):
        query = {key:source.get(key) for key in QUERY_COLUMNS}
        _,ids,_,matrix,_,_,_ = query_arrays(self.baseline,self.expansion,query)
        scores = self.ranker.predict(matrix,self.ranker.feature_names)
        order = score_order(scores,ids,self.baseline.catalog_ids)
        return str(self.baseline.catalog_ids[ids[order[0]]])


def run_fold(root,fold):
    root=Path(root).resolve(); frozen=check(root)
    if fold not in frozen['planned_folds']:
        raise ValueError('Unplanned fold')
    parent_fold=Path(frozen['parent'])/f'fold{fold}'
    info=frozen['baselines'][str(fold)]; cache=Path(info['cache'])
    # Validate this fold's consumed arrays and models before fitting.
    for path,expected in frozen['protected_files'].items():
        if any(Path(path).is_relative_to(p) for p in (parent_fold,cache,Path(info['directory']))):
            if sha(path)!=expected:
                raise ValueError(f'Parent asset changed: {path}')
    out=root/f'fold{fold}'
    if out.exists():
        raise FileExistsError('Use the terminal runner to preserve and restart incomplete folds')
    out.mkdir(); write_json(out/'status.json',{'complete':False})
    started=time.perf_counter()
    train=read_queries(frozen['data']['train.csv']['path'],labeled=True)
    catalog=read_catalog(frozen['data']['vessel_catalog.csv']['path'])
    baseline,summary,_=_load_model(info['directory'],frozen['data']['train.csv']['sha256'],frozen['data']['vessel_catalog.csv']['sha256'])
    if summary['dataset']['outer_fold']!=fold or summary['dataset']['inner_folds']!=3:
        raise ValueError('Baseline fold differs')
    heldout=_validate_partition(baseline,summary,train,catalog)
    parts=pd.read_csv(cache/'partitions.csv')
    assert set(heldout.sample_id)==set(parts.loc[parts.role.eq('valid'),'sample_id'])
    control=VesselRanker.load(parent_fold/'expanded')
    assert control.config==baseline.ranker.config and control.feature_names==baseline.ranker.feature_names
    groups,manifest=prepare_joint(train,catalog,parts,baseline,summary['dataset'],parent_fold,out)
    base_groups=_group_meta(cache)
    prior_groups=pd.read_csv(parent_fold/'extra_groups.csv')
    for left,right in zip(expanded_masses(base_groups.source,base_groups.variant,base_groups.mass,groups.source),
                          expanded_masses(base_groups.source,base_groups.variant,base_groups.mass,prior_groups.source)):
        assert np.array_equal(left,right)
    joint=fit_branch(cache,base_groups,groups,manifest,baseline,'joint',out)
    state=read_json(out/'joint/compound_model.json'); prior=read_json(parent_fold/'expanded/compound_model.json')
    for field in ('groups','pairs','sources','feature_names'):
        assert state[field]==prior[field]
    predictions=evaluate(baseline,control,joint,heldout,parent_fold,out)
    checks=[]
    sample=pd.concat([heldout[heldout.vessel_name.map(has_hangul).eq(lang)].head(1) for lang in (True,False)])
    inputs=[(evaluation_query(r,v),v) for r in sample.to_dict('records') for v in ('original','unseen_swap_dual_gt_x0_70','fresh_double_delete_gt_x1_50')]
    for arm in ('augmentation','retrieval','both'):
        restored=JointSearchMatcher(root,fold,arm)
        forward=[restored.predict(q) for q,_ in inputs]
        reverse=[restored.predict(q) for q,_ in reversed(inputs)][::-1]
        expected=predictions[predictions.model.eq(arm)].set_index(['sample_id','variant']).predicted_vessel_id
        assert forward==reverse==[expected.loc[(q['sample_id'],v)] for q,v in inputs]
        checks.append({'arm':arm,'queries':len(inputs),'native_reload_and_order_parity':True})
        del restored
    write_json(out/'status.json',{'complete':True,'fold':fold,'seconds':time.perf_counter()-started,
        'independence_checks':checks,'predictions_sha256':sha(out/'predictions.csv'),
        'joint_model_sha256':sha(out/'joint/model.txt'),'extra_manifest_sha256':sha(out/'extra_manifest.json'),
        'joint_log_sha256':sha(out/'joint_log.csv')})
    progress('fold_complete',fold=fold)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','run','report','status'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    if args.command=='freeze':
        if args.parent is None: parser.error('--parent required')
        freeze(args.output,args.parent)
    elif args.command=='run':
        if args.fold is None: parser.error('--fold required')
        run_fold(args.output,args.fold)
    elif args.command=='report':
        from .joint_search_report import report
        report(args.output)
    else:
        f=check(args.output,protection=True)
        progress('status',folds={str(i):read_json(args.output/f'fold{i}'/'status.json').get('complete',False)
                  if (args.output/f'fold{i}'/'status.json').exists() else False for i in f['planned_folds']})


if __name__=='__main__':
    main()
