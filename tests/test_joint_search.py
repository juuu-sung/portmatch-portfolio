import argparse
import importlib.util
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.compound_experiment import read_json, sha, write_json
from portmatch.compound_protocol import input_key
from portmatch.joint_search_protocol import (ARMS, EVAL_RECIPES, FRESH_RECIPES, TRAIN_FACTORS,
    edit_name, evaluation_query, joint_variant)
from portmatch.hangul_expansion import ExpansionConfig, HangulExpansion
from portmatch.retrieval import RESULT_COLUMNS
from portmatch.text import has_hangul


def fixture():
    spec=importlib.util.spec_from_file_location('joint_fixture',Path(__file__).with_name('test_robust_training.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.fixture()


class JointRecipeTests(unittest.TestCase):
    def row(self,name='오션 파이오니어 II 20'):
        return dict(sample_id='S',vessel_name=name,call_sign='ABC',gross_tonnage=1000.,international_gross_tonnage=1200.,
                    flag_name='Korea',vessel_type='Cargo',arrival_year=2020,target_vessel_id='DO_NOT_USE')

    def test_joint_changes_both_fields_without_target_id_or_year_dependency(self):
        factors,operations=set(),set()
        for i in range(80):
            row=self.row(f'오션 파이오니어 {i}'); before=input_key(row)
            q,info=joint_variant(row,'vessel')
            self.assertTrue(info['effective']);self.assertNotEqual(q['vessel_name'],row['vessel_name'])
            self.assertAlmostEqual(q['gross_tonnage']/row['gross_tonnage'],info['factor'])
            self.assertNotIn('target_vessel_id',q);self.assertEqual(input_key(row),before)
            other={**row,'target_vessel_id':'OTHER','sample_id':'OTHER','arrival_year':1900}
            q2,info2=joint_variant(other,'vessel')
            self.assertEqual(input_key(q),input_key(q2));self.assertEqual(info,info2)
            factors.add(info['factor']);operations.add(info['operation'])
        self.assertEqual(factors,set(TRAIN_FACTORS));self.assertEqual(operations,{'delete','swap'})

    def test_digits_roman_tokens_language_and_missing_values(self):
        for name in ('오션 파이오니어 II 20','Ocean 1A2 Pioneer IV','A가BC 12','Cedar No 8','시더 7','X'):
            for operation in ('delete','swap'):
                for seed in range(20):
                    changed=edit_name(name,seed,operation)
                    self.assertEqual(re.findall(r'\d+',changed),re.findall(r'\d+',name))
                    self.assertEqual(has_hangul(changed),has_hangul(name))
                    if ' II ' in name:self.assertIn(' II ',changed)
                    if name.endswith(' IV'):self.assertTrue(changed.endswith(' IV'))
        q,info=joint_variant({**self.row(),'gross_tonnage':None,'international_gross_tonnage':np.nan},'row')
        self.assertFalse(info['effective']);self.assertIsNone(q['gross_tonnage'])
        self.assertFalse(joint_variant(self.row(None),'row')[1]['effective'])

    def test_fresh_double_deletion_and_factors_are_not_training_recipes(self):
        for recipe in FRESH_RECIPES:
            row=self.row(); q=evaluation_query(row,recipe)
            self.assertEqual(len(row['vessel_name'])-len(q['vessel_name']),2)
            self.assertEqual(re.findall(r'\d+',q['vessel_name']),['20'])
            self.assertEqual(input_key(q),input_key(evaluation_query(row,recipe)))
        self.assertNotIn(.55,TRAIN_FACTORS);self.assertNotIn(1.5,TRAIN_FACTORS)
        self.assertEqual(evaluation_query(self.row(),'fresh_double_delete_gt_x0_55')['gross_tonnage'],550.)
        self.assertEqual(evaluation_query(self.row('가'),'fresh_double_delete')['vessel_name'],'가')
        with self.assertRaises(ValueError):evaluation_query(self.row(),'invalid')


class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train,cls.catalog=fixture();cls.index=HangulExpansion(cls.catalog)

    def old(self):
        frame=pd.DataFrame(0,index=range(1),columns=RESULT_COLUMNS)
        frame['catalog_index']=1;frame['phonetic_hit']=1;frame['phonetic_rank']=1
        frame['phonetic_score']=.9;frame['channel_count']=3
        return frame

    def test_recovers_missing_vessel_and_preserves_old_evidence(self):
        old=self.old();out=self.index.append({'vessel_name':'오션 시더'},old)
        self.assertIn(0,set(out.catalog_index));self.assertGreater(len(out),len(old))
        common=out.set_index('catalog_index').loc[old.catalog_index].reset_index()
        pd.testing.assert_frame_equal(common,old,check_dtype=False)
        self.assertTrue(out.catalog_index.is_unique)
        self.assertTrue(out.phonetic_score.between(0,1).all())
        self.assertTrue((out.loc[out.catalog_index.ne(1),'phonetic_rank']>1).all())

    def test_catalog_only_no_tonnage_or_label_or_query_order_dependencies(self):
        old=self.old();q={'vessel_name':'오션 시더','gross_tonnage':1.}
        a=self.index.append(q,old)
        self.index.append({'vessel_name':'블루 펄'},old)
        b=self.index.append({**q,'gross_tonnage':999999.,'target_vessel_id':'V01','sample_id':'OTHER'},old)
        pd.testing.assert_frame_equal(a,b)
        pd.testing.assert_frame_equal(old,self.index.append({'vessel_name':'Ocean Cedar'},old))

    def test_bound_disabled_and_unique_vessel_ties(self):
        small=HangulExpansion(self.catalog,ExpansionConfig(max_new=1))
        self.assertLessEqual(len(small.append({'vessel_name':'오션 시더'},self.old())),2)
        off=HangulExpansion(self.catalog,ExpansionConfig(max_new=0))
        pd.testing.assert_frame_equal(self.old(),off.append({'vessel_name':'오션 시더'},self.old()))
        shuffled=self.catalog.sample(frac=1,random_state=9).reset_index(drop=True)
        other=HangulExpansion(shuffled)
        a,sa=self.index.search('오션 시더');b,sb=other.search('오션 시더')
        self.assertEqual(self.catalog.iloc[a].vessel_id.tolist(),shuffled.iloc[b].vessel_id.tolist())
        np.testing.assert_allclose(sa,sb,atol=1e-6)
        with self.assertRaises(ValueError):ExpansionConfig(max_new=-1)


class FactorialIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from portmatch.compound_experiment import NEW_MODULES, prepare_teacher_pool,run_fold,report
        from portmatch.compound_protocol import protocol
        from portmatch.data import read_catalog,read_queries
        from portmatch.gt_augmentation import GTMissingAugmentationConfig
        from portmatch.name_feature_cache import transform_cache
        from portmatch.ranker_commands import add_training_parsers,run_training_command,feature_code_hash
        from portmatch.robust_retrieval import RobustRetrievalConfig
        from portmatch.training_data import prepare_training_data
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        root=Path(cls.temp.name);cls.root=root
        t,c=fixture();tp=root/'train.csv';cp=root/'catalog.csv'
        t.to_csv(tp,index=False);c.to_csv(cp,index=False)
        cls.train=read_queries(tp,labeled=True);cls.catalog=read_catalog(cp)
        source,cache,baseline=root/'source',root/'cache',root/'baseline'
        prepare_training_data(cls.train,cls.catalog,source,outer_folds=5,outer_fold=0,inner_folds=3,
            retrieval_config=RobustRetrievalConfig(),augmentation_config=GTMissingAugmentationConfig(),
            provenance={'train_sha256':sha(tp),'catalog_sha256':sha(cp),'feature_code_sha256':feature_code_hash()})
        transform_cache(source,cache,tp,cp)
        parser=argparse.ArgumentParser();add_training_parsers(parser.add_subparsers(dest='command',required=True))
        run_training_command(parser.parse_args(['train-ranker','--data',str(cache),'--output',str(baseline),'--exclude-features','retrieval_tonnage_rank']))
        parent=root/'parent';parent.mkdir();cls.parent=parent;cls.cache=cache;cls.baseline=baseline
        protected={str(p):sha(p) for d in (baseline,cache) for p in d.rglob('*') if p.is_file()}
        write_json(parent/'protocol.json',{'version':protocol()['version'],'protocol':protocol(),'implementation':{
            n:sha(Path(__file__).parents[1]/'portmatch'/n) for n in NEW_MODULES},'planned_folds':[0],'protected_files':protected,
            'data':{n:{'path':str(p),'sha256':sha(p)} for n,p in (('train.csv',tp),('vessel_catalog.csv',cp))},
            'baselines':{'0':{'directory':str(baseline),'cache':str(cache),'files':{n:sha(baseline/n) for n in
                ('model.txt','ranker.json','index.joblib','training_summary.json','validation_predictions.csv')}}}})
        prepare_teacher_pool(parent);run_fold(parent,0);report(parent)

    def test_end_to_end_factorial_native_reload_audit_and_parent_preservation(self):
        from portmatch.joint_search_experiment import freeze,run_fold,check
        from portmatch.joint_search_report import report
        from scripts.run_joint_search_study import status
        out=self.root/'study';frozen=freeze(out,self.parent)
        run_fold(out,0);report(out,replicates=100)
        self.assertTrue(read_json(out/'analysis_audit.json')['complete'])
        extra=read_json(out/'fold0/extra_manifest.json')
        self.assertGreater(extra['replaced_groups'],0)
        self.assertGreater(extra['unchanged_groups'],0)
        p=pd.read_csv(out/'fold0/predictions.csv')
        self.assertEqual(set(p.model),set(ARMS));self.assertEqual(set(p.variant),set(EVAL_RECIPES))
        self.assertEqual(len(p),p.sample_id.nunique()*5*16)
        self.assertEqual(len(read_json(out/'confidence_intervals.json')['comparisons']),16*10)
        self.assertEqual(status(out)['folds'],{'0':'complete'})
        from scripts.run_joint_search_study import run
        from scripts.run_compound_study import runner_lock
        model_hash=sha(out/'fold0/joint/model.txt')
        with runner_lock(out):self.assertEqual(run(out,1),0)
        self.assertEqual(sha(out/'fold0/joint/model.txt'),model_hash)
        check(out,protection=True)
        for path,expected in frozen['protected_files'].items():self.assertEqual(sha(path),expected)
        with self.assertRaises(FileExistsError):run_fold(out,0)
        # Artifact tampering must be caught without rerunning retrieval or fitting.
        log=out/'fold0/joint_log.csv';before=log.read_bytes()
        try:
            log.write_bytes(before+b'\n')
            with self.assertRaises(AssertionError):report(out,replicates=100)
        finally:log.write_bytes(before)

    def test_common_86_features_and_scores_survive_real_candidate_append(self):
        from portmatch.joint_search_experiment import query_arrays,score_query
        from portmatch.ranker_commands import RankedMatcher
        from portmatch.ranking import VesselRanker
        from portmatch.retrieval_comparison import _initialize_name_features
        model=RankedMatcher(self.baseline);_initialize_name_features(model)
        ranker=VesselRanker.load(self.parent/'fold0/expanded')
        expansion=HangulExpansion(self.catalog)
        query=self.train[self.train.vessel_name.map(has_hangul)].iloc[0].to_dict()
        original=model.pipeline.retrieve
        with patch.object(model.pipeline,'retrieve',side_effect=lambda r:original(r).head(1).reset_index(drop=True)):
            old,full,x,y,added,_,_=query_arrays(model,expansion,query)
            self.assertGreater(int(added.sum()),0)
            np.testing.assert_array_equal(x,y[np.searchsorted(full,old)])
            outcomes,_=score_query(model,ranker,ranker,expansion,query)
            self.assertEqual(len(outcomes),5)

    def test_missing_positive_falls_back_to_exact_control_groups(self):
        from portmatch.joint_search_experiment import prepare_joint,iter_extra
        from portmatch.ranker_commands import RankedMatcher
        from portmatch.robust_retrieval import RobustMatchingPipeline
        out=self.root/'fallback';out.mkdir()
        parent=self.parent/'fold0';model=RankedMatcher(self.baseline)
        dataset=read_json(self.baseline/'training_summary.json')['dataset']
        parts=pd.read_csv(self.cache/'partitions.csv')
        with patch.object(RobustMatchingPipeline,'retrieve',return_value=pd.DataFrame(columns=RESULT_COLUMNS)):
            _,manifest=prepare_joint(self.train,self.catalog,parts,model,dataset,parent,out)
        self.assertEqual(manifest['replaced_groups'],0)
        self.assertGreater(manifest['statuses'].get('fallback_missing_target',0),0)
        for a,b in zip(iter_extra(parent,read_json(parent/'extra_manifest.json')),iter_extra(out,manifest)):
            for field in ('X','ids','y'):np.testing.assert_array_equal(a[field],b[field])

    def test_changed_pair_budget_falls_back_without_altering_training_mass(self):
        from portmatch.joint_search_experiment import prepare_joint
        from portmatch.ranker_commands import RankedMatcher
        out=self.root/'budget_fallback';out.mkdir()
        model=RankedMatcher(self.baseline)
        with patch('portmatch.joint_search_experiment.sample_training_candidates',return_value=np.array([0])):
            groups,manifest=prepare_joint(self.train,self.catalog,pd.read_csv(self.cache/'partitions.csv'),model,
                read_json(self.baseline/'training_summary.json')['dataset'],self.parent/'fold0',out)
        self.assertEqual(manifest['replaced_groups'],0)
        self.assertGreater(manifest['statuses'].get('fallback_group_budget',0),0)
        prior=pd.read_csv(self.parent/'fold0/extra_groups.csv')
        for field in ('source','query','group','language'):
            pd.testing.assert_series_equal(groups[field],prior[field])


if __name__=='__main__':unittest.main()
