"""Audit the factorial controls and report separate paired effects."""
from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
import numpy as np
import pandas as pd

from .compound_experiment import read_json, sha, write_json, progress
from .joint_search_experiment import check, iter_extra
from .joint_search_protocol import ARMS, EVAL_RECIPES, FRESH_RECIPES, PAIRS
from .ranking import ranking_metrics
from .text import has_hangul


def report(root, replicates=5000):
    if replicates < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    root=Path(root).resolve(); frozen=check(root,protection=True)
    train=pd.read_csv(frozen['data']['train.csv']['path'])
    catalog=pd.read_csv(frozen['data']['vessel_catalog.csv']['path'])
    target=train.set_index('sample_id').target_vessel_id.to_dict()
    language=train.set_index('sample_id').vessel_name.map(has_hangul).to_dict()
    query_names=train.set_index('sample_id').vessel_name.to_dict()
    catalog_names=catalog.set_index('vessel_id').vessel_name.to_dict()
    index_of=dict(zip(catalog.vessel_id,range(len(catalog))))
    frames, audits, folds, training_rows=[] ,[],[],[]
    for fold in frozen['planned_folds']:
        out=root/f'fold{fold}'
        if not (out/'status.json').exists() or not read_json(out/'status.json').get('complete'):
            continue
        state=read_json(out/'status.json'); parent=Path(frozen['parent'])/f'fold{fold}'
        for name,key in (('predictions.csv','predictions_sha256'),('joint/model.txt','joint_model_sha256'),
                         ('extra_manifest.json','extra_manifest_sha256'),('joint_log.csv','joint_log_sha256')):
            assert sha(out/name)==state[key], name
        parts=pd.read_csv(Path(frozen['baselines'][str(fold)]['cache'])/'partitions.csv').set_index('sample_id')
        fitting=set(parts.index[parts.role.eq('train')]); valid=set(parts.index[parts.role.eq('valid')])
        assert not {target[s] for s in fitting} & {target[s] for s in valid}
        logs=pd.read_csv(out/'joint_log.csv')
        parent_groups=pd.read_csv(parent/'extra_groups.csv')
        eligible=parent_groups[parent_groups.recipe.eq('dual_gt_noise')]
        assert logs['query'].is_unique and set(logs['query'])==set(eligible['query'])
        assert set(logs.source).issubset(fitting)
        assert set(logs.status).issubset({'replaced','fallback_no_joint_change','fallback_missing_target','fallback_group_budget'})
        applied=set(logs.loc[logs.status.eq('replaced'),'query'])
        for proof in read_json(out/'inner_partitions.json'):
            held=set(logs.loc[logs.view.eq(proof['view']) & logs.inner.eq(proof['inner']),'source'])
            fit=set(parts.index[parts.role.eq('train') & parts['inner_'+proof['view']].ne(proof['inner'])])
            assert set(proof['held_sample_ids'])==held and set(proof['fitting_sample_ids'])==fit
            assert not fit & held and not fit & valid
            if proof['view']=='vessel':
                assert not {target[s] for s in fit} & {target[s] for s in held}
        old_manifest=read_json(parent/'extra_manifest.json'); new_manifest=read_json(out/'extra_manifest.json')
        assert new_manifest['replaced_groups']==len(applied)
        seen=[]; unchanged=0; pairs=0
        for old,new in zip_longest(iter_extra(parent,old_manifest),iter_extra(out,new_manifest)):
            assert old is not None and new is not None
            for field in ('source','query','language'):
                assert old[field]==new[field]
            assert len(old['y'])==len(new['y'])
            assert len(set(new['ids']))==len(new['ids'])
            assert np.array_equal(new['y'],(new['ids']==index_of[target[new['source']]]).astype(np.uint8))
            assert new['y'].sum()==1
            if new['query'] not in applied:
                for field in ('X','ids','y'):
                    assert np.array_equal(old[field],new[field],equal_nan=True), field
                unchanged+=1
            seen.append(new['query']); pairs+=len(new['y'])
        assert len(seen)==new_manifest['groups'] and set(applied).issubset(seen)
        prior_summary=read_json(parent/'expanded/compound_model.json')
        new_summary=read_json(out/'joint/compound_model.json')
        for field in ('groups','pairs','sources','source_mass_min','source_mass_max','feature_names'):
            assert np.allclose(new_summary[field],prior_summary[field]) if isinstance(new_summary[field],float) else new_summary[field]==prior_summary[field]
        for field in ('config','feature_names'):
            assert read_json(out/'joint/ranker.json')[field]==read_json(parent/'expanded/ranker.json')[field]
        p=pd.read_csv(out/'predictions.csv')
        assert set(p.sample_id)==valid and len(p)==len(valid)*len(ARMS)*len(EVAL_RECIPES)
        assert set(p.model)==set(ARMS) and set(p.variant)==set(EVAL_RECIPES)
        assert not p.duplicated(['sample_id','variant','model']).any()
        assert p.target_vessel_id.equals(p.sample_id.map(target))
        assert p.has_hangul.equals(p.sample_id.map(language))
        assert p.predicted_vessel_id.isin(catalog.vessel_id).all()
        assert all(c['native_reload_and_order_parity'] for c in state['independence_checks'])
        frames.append(p.assign(fold=fold));folds.append(fold)
        audits.append({'fold':fold,'groups_checked':len(seen),'pairs_checked':pairs,'unchanged_groups_checked':unchanged,
                       'replaced_groups':len(applied),'predictions_sha256':sha(out/'predictions.csv')})
        training_rows.append({'fold':fold,'eligible_groups':len(logs),'replaced_groups':len(applied),
                              **{str(k):int(v) for k,v in logs.status.value_counts().items()}})
    if not frames:
        raise ValueError('No completed folds')
    p=pd.concat(frames,ignore_index=True)
    clean=p[p.variant.eq('original') & p.model.eq('control')]
    assert clean.sample_id.is_unique
    vessels=sorted(clean.target_vessel_id.unique())
    denominator=clean.assign(h=clean.has_hangul.astype(int),e=(~clean.has_hangul).astype(int))
    counts=denominator.groupby('target_vessel_id')[['h','e']].sum().reindex(vessels).to_numpy(float)
    metrics, flat, paired, changed, retrieval, vectors, terms={ },[],[],[],[],[],[]
    for variant in EVAL_RECIPES:
        part=p[p.variant.eq(variant)]
        metrics[variant]={}
        for model,g in part.groupby('model'):
            m=ranking_metrics(g);metrics[variant][model]=m
            flat.append({'variant':variant,'evaluation_set':'fresh' if variant in FRESH_RECIPES else 'regression',
                'model':model,'weighted_top1':m['weighted_top1'],'weighted_candidate_recall':m['weighted_candidate_recall'],
                'hangul_correct':m['hangul']['correct_top1'],'hangul_queries':m['hangul']['queries'],
                'non_hangul_correct':m['non_hangul']['correct_top1'],'non_hangul_queries':m['non_hangul']['queries'],
                'mean_candidates':float(g.candidate_count.mean()),'p95_candidates':float(g.candidate_count.quantile(.95)),
                'name_changed_queries':int(g.name_changed.sum()),'two_letter_deletions':int(g.deleted_characters.eq(2).sum())})
        index=['fold','sample_id','target_vessel_id','has_hangul']
        wide=part.pivot(index=index,columns='model',values='predicted_vessel_id').reset_index()
        found=part.pivot(index='sample_id',columns='model',values='candidate_found')
        count=part.pivot(index='sample_id',columns='model',values='candidate_count')
        for field in (found,count):
            assert field['best'].equals(field['control']) and field['control'].equals(field['augmentation'])
            assert field['retrieval'].equals(field['both'])
        assert (count.retrieval>=count.control).all()
        assert not (found.control & ~found.retrieval).any()
        assert (count.retrieval-count.control).le(frozen['expansion']['max_new']).all()
        non=~wide.has_hangul
        assert wide.loc[non,'control'].equals(wide.loc[non,'retrieval'])
        assert wide.loc[non,'augmentation'].equals(wide.loc[non,'both'])
        for lang in (True,False):
            sources=wide.loc[wide.has_hangul.eq(lang),'sample_id']
            f=found.loc[sources]; n=count.loc[sources]
            outcomes=wide.set_index('sample_id').loc[sources]
            before_correct=outcomes.control.eq(outcomes.target_vessel_id)
            after_correct=outcomes.retrieval.eq(outcomes.target_vessel_id)
            assert not (f.control & ~before_correct & after_correct).any()
            retrieval.append({'variant':variant,'has_hangul':lang,'queries':len(sources),
                'recovered_targets':int((~f.control & f.retrieval).sum()),'lost_targets':int((f.control & ~f.retrieval).sum()),
                'recovered_targets_ranked_first':int((~f.control & f.retrieval & after_correct).sum()),
                'new_distractor_regressions':int((before_correct & ~after_correct).sum()),
                'mean_added_candidates':float((n.retrieval-n.control).mean()) if len(n) else 0.})
        correct={m:wide[m].eq(wide.target_vessel_id).astype(int) for m in ARMS}
        local=[]
        for before,after in PAIRS:
            d=correct[after]-correct[before]; fixed=d.eq(1); lost=d.eq(-1)
            effect={'variant':variant,'reference':before,'experiment':after,
                'delta_pp':100*(metrics[variant][after]['weighted_top1']-metrics[variant][before]['weighted_top1']),
                'corrected':int(fixed.sum()),'regressed':int(lost.sum()),
                'corrected_vessels':int(wide.loc[fixed,'target_vessel_id'].nunique()),'regressed_vessels':int(wide.loc[lost,'target_vessel_id'].nunique())}
            paired.append(effect); local.append((effect,d))
            for row in wide.loc[wide[before].ne(wide[after])].to_dict('records'):
                changed.append({'variant':variant,'reference':before,'experiment':after,'sample_id':row['sample_id'],
                    'fold':row['fold'],'target_vessel_id':row['target_vessel_id'],'has_hangul':row['has_hangul'],
                    'query_name':query_names[row['sample_id']],'target_name':catalog_names[row['target_vessel_id']],
                    'reference_prediction':catalog_names[row[before]],'experiment_prediction':catalog_names[row[after]],
                    'status':'corrected' if row[after]==row['target_vessel_id'] else 'regressed' if row[before]==row['target_vessel_id'] else 'still_wrong'})
        d=correct['both']-correct['augmentation']-correct['retrieval']+correct['control']
        interaction={'variant':variant,'reference':'factorial','experiment':'interaction',
            'delta_pp':100*(metrics[variant]['both']['weighted_top1']-metrics[variant]['augmentation']['weighted_top1']-metrics[variant]['retrieval']['weighted_top1']+metrics[variant]['control']['weighted_top1'])}
        local.append((interaction,d))
        for term,d in local:
            v=wide.assign(hd=d*wide.has_hangul,ed=d*(~wide.has_hangul))
            vectors.append(v.groupby('target_vessel_id')[['hd','ed']].sum().reindex(vessels,fill_value=0).to_numpy(float));terms.append(term)
    matrix=np.concatenate([counts,*vectors],axis=1)
    if not (counts.sum(axis=0)>0).all():
        raise ValueError('Both language strata required for clustered comparison')
    rng=np.random.default_rng(20261031); draws=[]; accepted=0
    while accepted<replicates:
        value=rng.poisson(1,size=(min(100,replicates-accepted),len(vessels)))@matrix
        value=value[(value[:,0]>0)&(value[:,1]>0)]
        if not len(value):continue
        draws.append(100*(.7*value[:,2::2]/value[:,0,None]+.3*value[:,3::2]/value[:,1,None]));accepted+=len(value)
    draws=np.concatenate(draws)
    intervals=[{**term,'vessel_bootstrap_95pct_pp':np.quantile(draws[:,i],[.025,.975]).tolist()} for i,term in enumerate(terms)]
    pd.DataFrame(flat).to_csv(root/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(paired).to_csv(root/'paired_changes.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([x for x in intervals if x['experiment']=='interaction']).to_csv(root/'interactions.csv',index=False)
    pd.DataFrame(changed).to_csv(root/'changed_rows.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(retrieval).to_csv(root/'retrieval_effects.csv',index=False)
    pd.DataFrame(training_rows).to_csv(root/'training_effective_groups.csv',index=False)
    write_json(root/'confidence_intervals.json',{'replicates':replicates,'method':'Paired Poisson vessel-cluster bootstrap',
        'seed':20261031,'caveat':'Development-set uncertainty; no correction for model selection, multiple comparisons or Public shift','comparisons':intervals})
    write_json(root/'comparison.json',{'complete_for_folds':folds,'full_five_fold':folds==[0,1,2,3,4],
        'queries':int(clean.sample_id.nunique()),'predictions':len(p),'metrics':metrics,'paired_effects':paired,
        'public_submission_performed':False,'automatic_promotion':False})
    write_json(root/'analysis_audit.json',{'complete':True,'folds':folds,'predictions':len(p),'per_fold':audits,
        'protected_files_verified':len(frozen['protected_files']),'old_assets_unchanged':True,
        'common_feature_and_score_checks':'Performed on every computed input during evaluation; protected code and status evidence',
        'analysis_code_sha256':sha(Path(__file__))})
    progress('report_complete',folds=folds,predictions=len(p),protected_files=len(frozen['protected_files']))
