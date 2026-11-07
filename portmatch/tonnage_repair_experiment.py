"""Add a balanced GT replay pool while preserving all earlier study artifacts."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import copy
import gc
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .compound_experiment import (ExtraWriter, _group_meta, heavy_training_lock, progress,
                                  read_json, sha, write_json)
from .data import QUERY_COLUMNS, read_catalog, read_queries
from .hangul_expansion import HangulExpansion
from .joint_search_experiment import check as check_parent, iter_extra, query_arrays
from .ranking import VesselRanker
from .ranker_commands import RankedMatcher, score_order
from .retrieval_comparison import _load_model, _validate_partition, _initialize_name_features
from .robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig
from .sampling import SamplingConfig, sample_training_candidates
from .text import has_hangul
from .training_data import iter_shards
from .tonnage_repair_protocol import (ARMS, EVAL_RECIPES, NEW_ARMS, REGRESSION_RECIPES, VERSION,
    evaluation, focused_positions, input_key, protocol, replay_masses, training_query)

CODE_ROOT=Path(__file__).resolve().parents[1]
FILES=('portmatch/tonnage_repair_protocol.py','portmatch/tonnage_repair_experiment.py',
       'portmatch/tonnage_repair_report.py','scripts/run_tonnage_repair_study.py')


def freeze(root,parent):
    root,parent=Path(root).resolve(),Path(parent).resolve()
    if root.exists():raise FileExistsError('Use a new study directory')
    old=check_parent(parent,protection=True)
    protected=dict(old['protected_files'])
    if parent in root.parents:raise ValueError('New study must be outside preserved parent')
    for fold in old['planned_folds']:
        directory=parent/f'fold{fold}';state=read_json(directory/'status.json')
        if not state['complete']:
            raise ValueError('Every planned parent fold must be complete')
        for name,key in (('predictions.csv','predictions_sha256'),('joint/model.txt','joint_model_sha256'),
                         ('extra_manifest.json','extra_manifest_sha256'),('joint_log.csv','joint_log_sha256')):
            if sha(directory/name)!=state[key]:raise ValueError(f'Parent completion evidence changed: {name}')
        for name,expected in read_json(directory/'extra_manifest.json')['files'].items():
            if sha(directory/'extra'/name)!=expected:raise ValueError('Parent joint training shard changed')
    for path in parent.rglob('*'):
        if path.is_file() and path.suffix!='.lock' and '__pycache__' not in path.parts:
            protected[str(path)]=sha(path)
    for info in old['baselines'].values():
        for folder in ('directory','cache'):
            directory=Path(info[folder]).resolve()
            if root==directory or directory in root.parents:
                raise ValueError('Output cannot be inside baseline or cache')
    frozen={'version':VERSION,'protocol':protocol(),'parent':str(parent),'parent_protocol_sha256':sha(parent/'protocol.json'),
        'planned_folds':old['planned_folds'],'data':old['data'],'baselines':old['baselines'],
        'protected_files':protected,'implementation':{f:sha(CODE_ROOT/f) for f in FILES},'frozen_before_training':True}
    root.mkdir(parents=True);write_json(root/'protocol.json',frozen)
    progress('frozen',output=str(root),protected_files=len(protected))
    return frozen


def check(root,protection=False):
    frozen=read_json(Path(root)/'protocol.json')
    if frozen['version']!=VERSION or frozen['protocol']!=protocol():raise ValueError('Study protocol changed')
    for relative,expected in frozen['implementation'].items():
        if sha(CODE_ROOT/relative)!=expected:raise ValueError(f'Implementation changed: {relative}')
    parent=Path(frozen['parent'])
    if sha(parent/'protocol.json')!=frozen['parent_protocol_sha256']:raise ValueError('Parent protocol changed')
    check_parent(parent)
    if protection:
        for path,expected in frozen['protected_files'].items():
            if sha(path)!=expected:raise ValueError(f'Protected file changed: {path}')
    return frozen


def prepare_repair(train,catalog,parts,baseline,dataset,base,parent_groups,out):
    writer=ExtraWriter(out/'extra',baseline.ranker.feature_names)
    budget_sources=set(base.loc[base.variant.ne('original'),'source']) | set(parent_groups.source)
    fitting_ids=set(parts.loc[parts.role.eq('train'),'sample_id'])
    outer=train[train.sample_id.isin(fitting_ids)].copy()
    partition=parts.set_index('sample_id')
    valid_vessels=set(parts.loc[parts.role.eq('valid'),'target_vessel_id'])
    targets=dict(zip(catalog.vessel_id,range(len(catalog))))
    expansion=HangulExpansion(catalog); sampling=SamplingConfig(**dataset['sampling'])
    logs,proofs=[],[]
    for view in ('vessel','row'):
        for inner in range(3):
            held=outer[outer.sample_id.map(partition['inner_'+view]).eq(inner)].copy()
            fit=outer[outer.sample_id.map(partition['inner_'+view]).ne(inner)]
            if held.empty:continue
            assert not set(fit.sample_id)&set(held.sample_id)
            assert not set(fit.target_vessel_id)&valid_vessels
            if view=='vessel':assert not set(fit.target_vessel_id)&set(held.target_vessel_id)
            proofs.append({'view':view,'inner':inner,'fitting_sample_ids':sorted(fit.sample_id),
                           'held_sample_ids':sorted(held.sample_id)})
            pipeline=RobustMatchingPipeline(fit,catalog,RobustRetrievalConfig(**dataset['retrieval']))
            holder=copy.copy(baseline);holder.pipeline=pipeline;_initialize_name_features(holder)
            held['_key']=held.apply(input_key,axis=1)
            for number,source in enumerate(held.sort_values(['_key','sample_id']).to_dict('records'),1):
                query,details=training_query(source,view)
                query_id=f"{source['sample_id']}::{view}::tonnage_repair"
                record={'source':source['sample_id'],'query':query_id,'recipe':'tonnage_repair',
                        'language':bool(has_hangul(source['vessel_name'])),'teacher_fold':-1}
                log={**record,**details,'view':view,'inner':inner,'status':'stored','target_rescued':False,
                     'candidates':0,'group':0,'broad_added_negatives':0,'focused_added_negatives':0,'sampling_changed':False}
                if source['sample_id'] not in budget_sources:log['status']='skipped_no_replay_budget'
                elif not details['effective']:log['status']='skipped_no_valid_gt'
                else:
                    old_ids,ids,_,matrix,added,_,_=query_arrays(holder,expansion,query)
                    log['candidates']=len(ids)
                    target=targets[source['target_vessel_id']]
                    log['target_rescued']=bool(target in ids and target not in old_ids)
                    if target not in ids:log['status']='skipped_missing_target'
                    elif len(ids)<2:log['status']='skipped_no_negative'
                    else:
                        candidates=pd.DataFrame({'catalog_index':ids})
                        features=pd.DataFrame(matrix,columns=baseline.ranker.feature_names)
                        usual=sample_training_candidates(candidates,features,target,query_id,sampling)
                        focused=focused_positions(candidates,features,target,usual,old_ids)
                        assert len(usual)==len(focused) and np.sum(ids[usual]==target)==np.sum(ids[focused]==target)==1
                        writer.add(matrix,candidates,target,usual,focused,record)
                        log.update(group=len(usual),broad_added_negatives=int(np.sum(added[usual]&(ids[usual]!=target))),
                            focused_added_negatives=int(np.sum(added[focused]&(ids[focused]!=target))),
                            sampling_changed=not np.array_equal(usual,focused))
                logs.append(log)
                if number%200==0:progress('repair_pool',view=view,inner=inner,done=number,total=len(held))
            del holder,pipeline;gc.collect()
    writer.flush()
    groups=pd.DataFrame(writer.rows)
    if groups.empty:raise ValueError('No usable repair groups')
    groups.to_csv(out/'extra_groups.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(logs).to_csv(out/'repair_log.csv',index=False,encoding='utf-8-sig')
    write_json(out/'inner_partitions.json',proofs)
    manifest={'complete':True,'feature_names':writer.names,'groups':len(groups),'pairs_per_branch':int(groups.group.sum()),
        'files':{f:sha(out/'extra'/f) for f in writer.files},'statuses':dict(Counter(x['status'] for x in logs)),
        'sampling_changed_groups':int(groups.mining_changed.sum()),'expanded_candidates_in_both_branches':True,
        'positives_rescued_by_expansion':sum(x['target_rescued'] and x['status']=='stored' for x in logs)}
    write_json(out/'extra_manifest.json',manifest)
    progress('repair_pool_complete',groups=len(groups),pairs=manifest['pairs_per_branch'])
    return groups,manifest


def repair_shards(out,manifest,selection):
    for relative,expected in manifest['files'].items():
        path=out/'extra'/relative
        if sha(path)!=expected:raise ValueError('Repair shard changed')
        with np.load(path,allow_pickle=False) as z:
            chosen=z[selection]
            starts=np.r_[0,np.cumsum(z['union_group'])[:-1]]
            yield z['X'][chosen],z['y'][chosen],np.add.reduceat(chosen.astype(int),starts)


def fit_repair(cache,base,parent_fold,parent_groups,extra,manifest,baseline,branch,out):
    with heavy_training_lock(out.parent):
        masses=replay_masses(base,parent_groups,extra)
        groups=np.r_[base.group,parent_groups.group,extra.group].astype(np.int32)
        languages=np.r_[base.language,parent_groups.language,extra.language].astype(bool)
        matrix=np.empty((int(groups.sum()),len(baseline.ranker.feature_names)),np.float32)
        labels=np.empty(len(matrix),np.uint8);offset=0
        cache_names=read_json(cache/'metadata.json')['feature_names']
        positions=[cache_names.index(n) for n in baseline.ranker.feature_names]
        for shard in iter_shards(cache,'train'):
            n=len(shard['y']);matrix[offset:offset+n]=shard['X'][:,positions];labels[offset:offset+n]=shard['y'];offset+=n
        assert offset==int(base.group.sum())
        for group in iter_extra(parent_fold,read_json(parent_fold/'extra_manifest.json')):
            n=len(group['y']);matrix[offset:offset+n]=group['X'];labels[offset:offset+n]=group['y'];offset+=n
        assert offset==int(base.group.sum()+parent_groups.group.sum())
        observed=[]
        for x,y,g in repair_shards(out,manifest,'heuristic' if branch=='broad' else 'mined'):
            n=len(y);matrix[offset:offset+n]=x;labels[offset:offset+n]=y;offset+=n;observed.extend(g.tolist())
        assert offset==len(matrix) and observed==extra.group.tolist()
        started=time.perf_counter()
        ranker=VesselRanker(baseline.ranker.config)
        ranker.fit(matrix,labels,groups,languages,baseline.ranker.feature_names,query_mass=np.concatenate(masses))
        directory=out/branch;ranker.save(directory)
        mass_frame=pd.concat([pd.DataFrame({'source':g.source,'query':g['query'],'kind':kind,'mass':mass})
            for g,mass,kind in zip((base,parent_groups,extra),masses,('base','previous_joint','repair'))],ignore_index=True)
        mass_frame.to_csv(directory/'query_masses.csv',index=False)
        summary={'complete':True,'branch':branch,'fold':int(out.name.removeprefix('fold')),
            'protocol_path':str(out.parent/'protocol.json'),'groups':len(groups),'pairs':len(matrix),
            'retained_base_groups':len(base),'retained_joint_groups':len(parent_groups),'repair_groups':len(extra),
            'feature_names':ranker.feature_names,'all_previous_examples_retained':True,'original_mass_unchanged':True,
            'query_masses_sha256':sha(directory/'query_masses.csv'),'train_seconds':time.perf_counter()-started}
        write_json(directory/'repair_model.json',summary)
        ranker.feature_importance().to_csv(directory/'feature_importance.csv',index=False)
        del matrix,labels;gc.collect()
        progress('model_complete',branch=branch,seconds=summary['train_seconds'])
        return ranker


def score_query(baseline,previous,broad,focused,expansion,query):
    old,full,x,y,added,old_seconds,new_seconds=query_arrays(baseline,expansion,query)
    result={};common={};old_set=set(old)
    for arm,ranker in [('best',baseline.ranker),('previous',previous),('previous_search',previous),
                       ('broad',broad),('broad_search',broad),('focused',focused),('focused_search',focused)]:
        expanded=arm.endswith('_search');indices,matrix=(full,y) if expanded else (old,x)
        key=arm.removesuffix('_search')
        scores=common[key] if expanded and not added.any() else ranker.predict(matrix,ranker.feature_names)
        if expanded:
            assert np.allclose(scores[np.searchsorted(full,old)],common[key],atol=1e-10,rtol=0)
        else:common[key]=scores
        order=score_order(scores,indices,baseline.catalog_ids)
        result[arm]=(baseline.catalog_ids[indices[order]],scores[order],bool(indices[order[0]] not in old_set))
    return result,{'base_seconds':old_seconds,'extra_seconds':new_seconds,'base_candidates':len(old),'added_candidates':int(added.sum())}


def evaluate(baseline,previous,broad,focused,heldout,parent_fold,out):
    expansion=HangulExpansion(baseline.pipeline.catalog);_initialize_name_features(baseline)
    saved=pd.read_csv(parent_fold/'predictions.csv').set_index(['sample_id','variant','model'])
    memo,rows,timings=OrderedDict(),[],[]
    heldout=heldout.copy();heldout['_key']=heldout.apply(input_key,axis=1)
    for number,source in enumerate(heldout.sort_values(['_key','sample_id']).to_dict('records'),1):
        for variant in EVAL_RECIPES:
            query=evaluation(source,variant);key=input_key(query)
            if key not in memo:
                memo[key],timing=score_query(baseline,previous,broad,focused,expansion,query)
                timings.append({'has_hangul':bool(has_hangul(source['vessel_name'])),**timing})
                if len(memo)>64:memo.popitem(last=False)
            else:memo.move_to_end(key)
            for arm,(ordered,scores,added) in memo[key].items():
                location=np.flatnonzero(ordered==source['target_vessel_id'])
                record={'variant':variant,'model':arm,'sample_id':source['sample_id'],'target_vessel_id':source['target_vessel_id'],
                    'has_hangul':bool(has_hangul(source['vessel_name'])),'predicted_vessel_id':str(ordered[0]),
                    'candidate_found':bool(len(location)),'target_rank':int(location[0]+1) if len(location) else 0,
                    'candidate_count':len(ordered),'top_score':float(scores[0]),
                    'score_margin':float(scores[0]-scores[1]) if len(scores)>1 else None,
                    'prediction_from_added_candidate':added,'name_changed':str(query['vessel_name'])!=str(source['vessel_name']),
                    'deleted_characters':max(0,len(str(source['vessel_name']))-len(str(query['vessel_name'])))}
                if arm in ARMS[:3] and variant in REGRESSION_RECIPES:
                    old=saved.loc[(source['sample_id'],variant,{'best':'best','previous':'augmentation','previous_search':'both'}[arm])]
                    for field in ('predicted_vessel_id','candidate_found','target_rank','candidate_count'):
                        assert record[field]==old[field],(source['sample_id'],variant,field)
                    assert np.isclose(record['top_score'],old.top_score,atol=1e-10,rtol=0)
                rows.append(record)
        if number%100==0:progress('evaluate',done=number,total=len(heldout),predictions=len(rows))
    frame=pd.DataFrame(rows);frame.to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    write_json(out/'evaluation.json',{'complete':True,'queries':len(heldout),'predictions':len(frame),
        'previous_sixteen_conditions_exact':True,'common_candidate_features_and_scores_unchanged':True,
        'unique_inputs_computed':len(timings),'timing_note':'Shared memoized comparison; not a submission benchmark'})
    return frame


class TonnageRepairMatcher:
    def __init__(self,root,fold,arm):
        if arm not in ARMS:raise ValueError('Unknown arm')
        root=Path(root).resolve();frozen=check(root);info=frozen['baselines'][str(fold)]
        for name,expected in info['files'].items():
            if sha(Path(info['directory'])/name)!=expected:raise ValueError('Baseline bundle changed')
        self.baseline=RankedMatcher(info['directory']);key=arm.removesuffix('_search')
        if key=='best':self.ranker=self.baseline.ranker
        elif key=='previous':
            directory=Path(frozen['parent'])/f'fold{fold}'/'joint'
            for name in ('model.txt','ranker.json'):
                if sha(directory/name)!=frozen['protected_files'][str(directory/name)]:raise ValueError('Parent model changed')
            self.ranker=VesselRanker.load(directory)
        else:
            directory=root/f'fold{fold}'/key;summary=read_json(directory/'repair_model.json')
            if not summary['complete'] or summary['fold']!=fold or Path(summary['protocol_path'])!=root/'protocol.json':
                raise ValueError('Wrong or incomplete repair model')
            self.ranker=VesselRanker.load(directory)
        if self.ranker.feature_names!=self.baseline.ranker.feature_names or self.ranker.config!=self.baseline.ranker.config:
            raise ValueError('Model schema/config changed')
        _initialize_name_features(self.baseline)
        self.expansion=HangulExpansion(self.baseline.pipeline.catalog) if arm.endswith('_search') else None

    def predict(self,source):
        query={key:source.get(key) for key in QUERY_COLUMNS}
        _,ids,_,matrix,_,_,_=query_arrays(self.baseline,self.expansion,query)
        scores=self.ranker.predict(matrix,self.ranker.feature_names)
        order=score_order(scores,ids,self.baseline.catalog_ids)
        return str(self.baseline.catalog_ids[ids[order[0]]])


def verify_fold_assets(out,state):
    for name,expected in state['files'].items():
        if sha(out/name)!=expected:raise ValueError(f'Completed fold file changed: {name}')


def run_fold(root,fold):
    root=Path(root).resolve();frozen=check(root)
    if fold not in frozen['planned_folds']:raise ValueError('Unplanned fold')
    info=frozen['baselines'][str(fold)];cache=Path(info['cache']);parent=Path(frozen['parent'])/f'fold{fold}'
    # Verify every consumed parent group and baseline before allocating training matrices.
    for path,expected in frozen['protected_files'].items():
        if any(Path(path).is_relative_to(p) for p in (parent,cache,Path(info['directory']))):
            if sha(path)!=expected:raise ValueError(f'Parent asset changed: {path}')
    out=root/f'fold{fold}'
    if out.exists():raise FileExistsError('Use runner to preserve and restart incomplete fold')
    out.mkdir();write_json(out/'status.json',{'complete':False});start=time.perf_counter()
    train=read_queries(frozen['data']['train.csv']['path'],labeled=True);catalog=read_catalog(frozen['data']['vessel_catalog.csv']['path'])
    baseline,summary,_=_load_model(info['directory'],frozen['data']['train.csv']['sha256'],frozen['data']['vessel_catalog.csv']['sha256'])
    if summary['dataset']['outer_fold']!=fold or summary['dataset']['inner_folds']!=3:raise ValueError('Wrong fold')
    heldout=_validate_partition(baseline,summary,train,catalog);parts=pd.read_csv(cache/'partitions.csv')
    assert set(heldout.sample_id)==set(parts.loc[parts.role.eq('valid'),'sample_id'])
    previous=VesselRanker.load(parent/'joint')
    assert previous.config==baseline.ranker.config and previous.feature_names==baseline.ranker.feature_names
    base=_group_meta(cache);parent_groups=pd.read_csv(parent/'extra_groups.csv')
    extra,manifest=prepare_repair(train,catalog,parts,baseline,summary['dataset'],base,parent_groups,out)
    broad=fit_repair(cache,base,parent,parent_groups,extra,manifest,baseline,'broad',out)
    focused=fit_repair(cache,base,parent,parent_groups,extra,manifest,baseline,'focused',out)
    predictions=evaluate(baseline,previous,broad,focused,heldout,parent,out)
    del broad,focused;gc.collect()
    sample=pd.concat([heldout[heldout.vessel_name.map(has_hangul).eq(lang)].head(1) for lang in (True,False)])
    records=[(r,v) for r in sample.to_dict('records') for v in ('original','fresh_double_delete_gt_x0_55','unseen_swap_dual_gt_x0_70')]
    proofs=[]
    for arm in NEW_ARMS:
        matcher=TonnageRepairMatcher(root,fold,arm)
        inputs=[evaluation(r,v) for r,v in records]
        forward=[matcher.predict(q) for q in inputs];reverse=[matcher.predict(q) for q in reversed(inputs)][::-1]
        expected=predictions[predictions.model.eq(arm)].set_index(['sample_id','variant']).predicted_vessel_id
        assert forward==reverse==[expected.loc[(r['sample_id'],v)] for r,v in records]
        proofs.append({'arm':arm,'queries':len(inputs),'native_reload_and_order_parity':True})
        del matcher
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='status.json'}
    write_json(out/'status.json',{'complete':True,'fold':fold,'seconds':time.perf_counter()-start,'files':files,
                                 'independence_checks':proofs})
    progress('fold_complete',fold=fold)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','report'))
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    if args.command=='run':
        if args.fold is None:parser.error('--fold required')
        run_fold(args.output,args.fold)
    else:
        from .tonnage_repair_report import report
        report(args.output)


if __name__=='__main__':main()
