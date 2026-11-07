"""Verify saved experiments and report repair gains alongside every regression."""
from pathlib import Path
import numpy as np
import pandas as pd

from .compound_experiment import _group_meta, read_json, sha, write_json, progress
from .ranking import ranking_metrics
from .text import has_hangul
from .tonnage_repair_experiment import check, verify_fold_assets
from .tonnage_repair_protocol import (ARMS, EVAL_RECIPES, NEW_ARMS, NEW_RECIPES, PAIRS, REGRESSION_RECIPES,
    RETENTION, SEED, TARGET, TOLERANCE_PP, replay_masses)


def audit_fold(root,fold,frozen,train,catalog):
    out=root/f'fold{fold}';state=read_json(out/'status.json');verify_fold_assets(out,state)
    assert state['complete'] and state['fold']==fold
    assert all(x['native_reload_and_order_parity'] for x in state['independence_checks'])
    parent=Path(frozen['parent'])/f'fold{fold}';info=frozen['baselines'][str(fold)]
    parts=pd.read_csv(Path(info['cache'])/'partitions.csv').set_index('sample_id')
    fitting=set(parts.index[parts.role.eq('train')]);valid=set(parts.index[parts.role.eq('valid')])
    targets=train.set_index('sample_id').target_vessel_id
    assert not set(targets.loc[list(fitting)])&set(targets.loc[list(valid)])
    base=_group_meta(Path(info['cache']));prior=pd.read_csv(parent/'extra_groups.csv');extra=pd.read_csv(out/'extra_groups.csv')
    manifest=read_json(out/'extra_manifest.json');logs=pd.read_csv(out/'repair_log.csv')
    assert logs['query'].is_unique and set(logs.source)==fitting
    assert set(logs['query'])=={f'{s}::{v}::tonnage_repair' for s in fitting for v in ('vessel','row')}
    stored=logs[logs.status.eq('stored')]
    assert set(stored['query'])==set(extra['query']) and extra['query'].is_unique
    assert set(extra.source).issubset(fitting)
    for proof in read_json(out/'inner_partitions.json'):
        held=set(parts.index[parts.role.eq('train') & parts['inner_'+proof['view']].eq(proof['inner'])])
        fit=fitting-held
        assert set(proof['held_sample_ids'])==held and set(proof['fitting_sample_ids'])==fit
        if proof['view']=='vessel':assert not set(targets.loc[list(held)])&set(targets.loc[list(fit)])
    masses=replay_masses(base,prior,extra)
    source_index=dict(zip(catalog.vessel_id,range(len(catalog))))
    observed=[];pair_counts=[];changed=0
    for relative,expected in manifest['files'].items():
        path=out/'extra'/relative
        assert sha(path)==expected
        with np.load(path,allow_pickle=False) as z:
            offset=0
            for i,n in enumerate(z['union_group']):
                section=slice(offset,offset+int(n));offset+=int(n)
                ids=z['catalog_index'][section];y=z['y'][section];h=z['heuristic'][section];m=z['mined'][section]
                source=str(z['source'][i]);target=source_index[targets[source]]
                assert np.array_equal(y,(ids==target).astype(np.uint8)) and len(set(ids))==len(ids)
                assert y[h].sum()==y[m].sum()==1 and h.sum()==m.sum() and h.sum()>=2
                observed.append(str(z['query'][i]));pair_counts.append(int(h.sum()));changed+=not np.array_equal(h,m)
    assert observed==extra['query'].tolist() and pair_counts==extra.group.tolist()
    assert len(observed)==manifest['groups'] and sum(pair_counts)==manifest['pairs_per_branch']
    assert changed==manifest['sampling_changed_groups']
    expected_masses=pd.concat([pd.DataFrame({'source':g.source,'query':g['query'],'kind':kind,'mass':mass})
        for g,mass,kind in zip((base,prior,extra),masses,('base','previous_joint','repair'))],ignore_index=True)
    for branch in ('broad','focused'):
        summary=read_json(out/branch/'repair_model.json')
        assert summary['groups']==len(expected_masses)
        assert summary['pairs']==int(base.group.sum()+prior.group.sum()+extra.group.sum())
        actual=pd.read_csv(out/branch/'query_masses.csv')
        pd.testing.assert_frame_equal(actual,expected_masses,check_dtype=False,atol=1e-12,rtol=0)
        rank=read_json(out/branch/'ranker.json');old=read_json(parent/'joint/ranker.json')
        assert rank['config']==old['config'] and rank['feature_names']==old['feature_names']
    p=pd.read_csv(out/'predictions.csv').assign(fold=fold)
    assert set(p.sample_id)==valid and len(p)==len(valid)*len(EVAL_RECIPES)*len(ARMS)
    assert set(p.model)==set(ARMS) and set(p.variant)==set(EVAL_RECIPES)
    assert not p.duplicated(['sample_id','variant','model']).any()
    assert p.target_vessel_id.equals(p.sample_id.map(targets))
    language=train.set_index('sample_id').vessel_name.map(has_hangul)
    assert p.has_hangul.equals(p.sample_id.map(language))
    assert p.predicted_vessel_id.isin(catalog.vessel_id).all()
    old=pd.read_csv(parent/'predictions.csv').set_index(['sample_id','variant','model'])
    for arm,old_arm in [('best','best'),('previous','augmentation'),('previous_search','both')]:
        a=p[p.model.eq(arm)&p.variant.isin(REGRESSION_RECIPES)].set_index(['sample_id','variant']).sort_index()
        b=old.xs(old_arm,level='model').loc[a.index]
        for field in ('predicted_vessel_id','candidate_found','target_rank','candidate_count'):
            assert np.array_equal(a[field],b[field])
        assert np.allclose(a.top_score,b.top_score,atol=1e-10,rtol=0)
    return p,{'fold':fold,'new_groups':len(extra),'pairs_per_branch':sum(pair_counts),'sampling_changed_groups':changed,
              'previous_base_groups_retained':len(base),'previous_joint_groups_retained':len(prior),
              'training_statuses':logs.status.value_counts().to_dict()},logs


def report(root,replicates=5000):
    if replicates<100:raise ValueError('Use at least 100 bootstrap draws')
    root=Path(root).resolve();frozen=check(root,protection=True)
    train=pd.read_csv(frozen['data']['train.csv']['path']);catalog=pd.read_csv(frozen['data']['vessel_catalog.csv']['path'])
    frames,audits,logs,folds=[],[],[],[]
    for fold in frozen['planned_folds']:
        path=root/f'fold{fold}/status.json'
        if not path.exists() or not read_json(path).get('complete'):continue
        p,a,l=audit_fold(root,fold,frozen,train,catalog)
        frames.append(p);audits.append(a);logs.append(l.assign(fold=fold));folds.append(fold)
    if not frames:raise ValueError('No complete folds')
    p=pd.concat(frames,ignore_index=True);p['correct']=p.predicted_vessel_id.eq(p.target_vessel_id)
    clean=p[p.variant.eq('original')&p.model.eq('best')]
    assert clean.sample_id.is_unique
    vessels=sorted(clean.target_vessel_id.unique())
    counts=clean.assign(h=clean.has_hangul.astype(int),e=(~clean.has_hangul).astype(int)).groupby('target_vessel_id')[['h','e']].sum().reindex(vessels).to_numpy(float)
    metrics,fold_metrics,errors,paired,changes,search,terms,vectors=[],[],[],[],[],[],[],[]
    names=train.set_index('sample_id').vessel_name.to_dict();catalog_names=catalog.set_index('vessel_id').vessel_name.to_dict()
    for variant,g in p.groupby('variant',sort=False):
        metric={}
        for arm,a in g.groupby('model'):
            metric[arm]=ranking_metrics(a)
            m=metric[arm]
            metrics.append({'variant':variant,'evaluation_set':'fresh' if variant in NEW_RECIPES else 'regression',
                'model':arm,'weighted_top1':m['weighted_top1'],'weighted_candidate_recall':m['weighted_candidate_recall'],
                'hangul_correct':m['hangul']['correct_top1'],'hangul_queries':m['hangul']['queries'],
                'non_hangul_correct':m['non_hangul']['correct_top1'],'non_hangul_queries':m['non_hangul']['queries'],
                'name_changed_queries':int(a.name_changed.sum()),'two_letter_deletions':int(a.deleted_characters.eq(2).sum())})
            for fold,b in a.groupby('fold'):
                fold_metrics.append({'variant':variant,'model':arm,'fold':int(fold),**{'weighted_top1':ranking_metrics(b)['weighted_top1']}})
            for (lang,changed),b in a.groupby(['has_hangul','name_changed']):
                errors.append({'variant':variant,'model':arm,'has_hangul':bool(lang),'name_changed':bool(changed),
                    'queries':len(b),'correct':int(b.correct.sum()),'candidate_missing':int((~b.candidate_found).sum()),
                    'ranking_errors':int((b.candidate_found&~b.correct).sum())})
        wide=g.pivot(index=['fold','sample_id','target_vessel_id','has_hangul'],columns='model',values='predicted_vessel_id').reset_index()
        found=g.pivot(index='sample_id',columns='model',values='candidate_found')
        count=g.pivot(index='sample_id',columns='model',values='candidate_count')
        for field in (found,count):
            for arm in ('previous','broad','focused'):assert field.best.equals(field[arm])
            for arm in ('broad_search','focused_search'):assert field.previous_search.equals(field[arm])
        assert (count.previous_search-count.previous).between(0,256).all()
        assert not (found.previous&~found.previous_search).any()
        for before,after in [('previous','previous_search'),('broad','broad_search'),('focused','focused_search')]:
            non=~wide.has_hangul;assert wide.loc[non,before].equals(wide.loc[non,after])
            for lang in (True,False):
                w=wide[wide.has_hangul.eq(lang)].set_index('sample_id');f=found.loc[w.index]
                a=w[before].eq(w.target_vessel_id);b=w[after].eq(w.target_vessel_id)
                assert not (f[before]&~a&b).any()
                search.append({'variant':variant,'reference':before,'experiment':after,'has_hangul':lang,
                    'recovered_targets':int((~f[before]&f[after]).sum()),'recovered_ranked_first':int((~f[before]&b).sum()),
                    'new_distractor_regressions':int((a&~b).sum())})
        for before,after in PAIRS:
            d=wide[after].eq(wide.target_vessel_id).astype(int)-wide[before].eq(wide.target_vessel_id).astype(int)
            term={'variant':variant,'reference':before,'experiment':after,
                'delta_pp':100*(metric[after]['weighted_top1']-metric[before]['weighted_top1']),
                'corrected':int(d.eq(1).sum()),'regressed':int(d.eq(-1).sum()),
                'corrected_vessels':int(wide.loc[d.eq(1),'target_vessel_id'].nunique()),
                'regressed_vessels':int(wide.loc[d.eq(-1),'target_vessel_id'].nunique())}
            paired.append(term);terms.append(term)
            v=wide.assign(hd=d*wide.has_hangul,ed=d*(~wide.has_hangul))
            vectors.append(v.groupby('target_vessel_id')[['hd','ed']].sum().reindex(vessels,fill_value=0).to_numpy(float))
            for row in wide.loc[wide[before].ne(wide[after])].to_dict('records'):
                changes.append({'variant':variant,'reference':before,'experiment':after,'sample_id':row['sample_id'],
                    'fold':row['fold'],'target_vessel_id':row['target_vessel_id'],'has_hangul':row['has_hangul'],
                    'query_name':names[row['sample_id']],'target_name':catalog_names[row['target_vessel_id']],
                    'reference_prediction':catalog_names[row[before]],'experiment_prediction':catalog_names[row[after]],
                    'status':'corrected' if row[after]==row['target_vessel_id'] else 'regressed' if row[before]==row['target_vessel_id'] else 'still_wrong'})
    matrix=np.concatenate([counts,*vectors],axis=1);rng=np.random.default_rng(SEED)
    if not (counts.sum(axis=0)>0).all():raise ValueError('Both language strata needed')
    draws=[];accepted=0
    while accepted<replicates:
        z=rng.poisson(1,size=(min(100,replicates-accepted),len(vessels)))@matrix
        z=z[(z[:,0]>0)&(z[:,1]>0)]
        if len(z):
            draws.append(100*(.7*z[:,2::2]/z[:,0,None]+.3*z[:,3::2]/z[:,1,None]));accepted+=len(z)
    draws=np.concatenate(draws)
    intervals=[{**t,'vessel_bootstrap_95pct_pp':np.quantile(draws[:,i],[.025,.975]).tolist()} for i,t in enumerate(terms)]
    lookup={(x['variant'],x['reference'],x['experiment']):x for x in intervals}
    fm=pd.DataFrame(fold_metrics).set_index(['fold','variant','model']);gates=[]
    for arm in NEW_ARMS:
        reference='previous_search' if arm.endswith('_search') else 'previous'
        target=lookup[TARGET,reference,arm]
        keep=[lookup[v,reference,arm] for v in RETENTION]
        regressions=[x['variant'] for x in keep if x['delta_pp'] < -TOLERANCE_PP]
        uncertain=[x['variant'] for x in keep if x['vessel_bootstrap_95pct_pp'][0] < -TOLERANCE_PP]
        fold0_losses=[v for v in RETENTION if 0 in folds and 100*(fm.loc[(0,v,arm),'weighted_top1']-fm.loc[(0,v,reference),'weighted_top1']) < -TOLERANCE_PP]
        gates.append({'model':arm,'reference':reference,'target_delta_pp':target['delta_pp'],
            'target_95pct_pp':target['vessel_bootstrap_95pct_pp'],'target_gain_positive':target['delta_pp']>0,
            'target_ci_lower_positive':target['vessel_bootstrap_95pct_pp'][0]>0,
            'retention_margin_pp':TOLERANCE_PP,'retention_point_pass':not regressions,
            'retention_ci_noninferiority_pass':not uncertain,'regressed_conditions':regressions,
            'retention_not_confirmed_conditions':uncertain,'fold0_regressed_conditions':fold0_losses,
            'all_five_folds':folds==[0,1,2,3,4],
            'strict_development_gate_pass':bool(folds==[0,1,2,3,4] and target['vessel_bootstrap_95pct_pp'][0]>0 and not uncertain and not fold0_losses),
            'automatic_promotion':False})
    for name,rows in [('metrics',metrics),('fold_metrics',fold_metrics),('error_breakdown',errors),
                      ('paired_changes',paired),('changed_rows',changes),('retrieval_effects',search)]:
        columns=['variant','reference','experiment','sample_id','fold','target_vessel_id','has_hangul',
                 'query_name','target_name','reference_prediction','experiment_prediction','status'] if name=='changed_rows' else None
        pd.DataFrame(rows,columns=columns).to_csv(root/f'{name}.csv',index=False,encoding='utf-8-sig')
    log=pd.concat(logs,ignore_index=True)
    log.groupby(['fold','language','operation','factor','status','name_changed']).agg(
        groups=('query','size'),rescued_targets=('target_rescued','sum'),
        broad_added_negatives=('broad_added_negatives','sum'),focused_added_negatives=('focused_added_negatives','sum')).reset_index().to_csv(root/'training_coverage.csv',index=False)
    write_json(root/'confidence_intervals.json',{'replicates':replicates,'seed':SEED,'method':'Paired Poisson vessel-cluster bootstrap',
        'caveat':'Development-set uncertainty; no correction for repeated model selection, multiple comparisons or Public shift','comparisons':intervals})
    write_json(root/'retention_gates.json',{'target':TARGET,'retention_conditions':list(RETENTION),'gates':gates,
        'caveat':'Predeclared descriptive gate, not proof of Public gain; no automatic promotion'})
    write_json(root/'comparison.json',{'complete_for_folds':folds,'full_five_fold':folds==[0,1,2,3,4],
        'queries':int(clean.sample_id.nunique()),'predictions':len(p),'public_submission_performed':False,'automatic_promotion':False})
    write_json(root/'analysis_audit.json',{'complete':True,'folds':folds,'predictions':len(p),'per_fold':audits,
        'protected_files_verified':len(frozen['protected_files']),'old_assets_unchanged':True,
        'previous_sixteen_conditions_exact':True,'original_mass_unchanged':True,'branch_groups_pair_counts_and_mass_equal':True,
        'analysis_code_sha256':sha(__file__)})
    progress('report_complete',folds=folds,predictions=len(p),protected_files=len(frozen['protected_files']))
