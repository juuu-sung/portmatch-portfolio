import importlib.util
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.compound_experiment import read_json, sha
from portmatch.compound_protocol import input_key
from portmatch.joint_search_protocol import evaluation_query
from portmatch.text import has_hangul
from portmatch.tonnage_repair_protocol import (ARMS, EVAL_RECIPES, NAME_OPERATIONS, NEW_ARMS, NEW_RECIPES,
    REGRESSION_RECIPES, TRAIN_FACTORS, change_name, evaluation, focused_positions, replay_masses, training_query)


class RepairPolicyTests(unittest.TestCase):
    def row(self,name='오션 파이오니어 II 20'):
        return dict(sample_id='S',vessel_name=name,call_sign='ABC',gross_tonnage=1000.,international_gross_tonnage=1200.,
                    flag_name='Korea',vessel_type='Cargo',arrival_year=2020,target_vessel_id='TARGET')

    def test_broad_balanced_recipes_are_input_only_and_do_not_mutate(self):
        factors,operations=set(),set()
        for i in range(512):
            row=self.row(f'오션 파이오니어 II {i}');before=input_key(row)
            q,info=training_query(row,'vessel');other={**row,'sample_id':'OTHER','arrival_year':1900,'target_vessel_id':'OTHER'}
            z,other_info=training_query(other,'vessel')
            self.assertEqual(input_key(q),input_key(z));self.assertEqual(info,other_info)
            self.assertEqual(input_key(row),before);self.assertTrue(info['effective'])
            self.assertAlmostEqual(q['gross_tonnage']/row['gross_tonnage'],info['factor'])
            self.assertNotIn('target_vessel_id',q)
            factors.add(info['factor']);operations.add(info['operation'])
        self.assertEqual(factors,set(TRAIN_FACTORS));self.assertEqual(operations,set(NAME_OPERATIONS))
        self.assertTrue(all(any(np.isclose(x*y,1) for y in TRAIN_FACTORS) for x in TRAIN_FACTORS))

    def test_preserves_numeric_identity_language_and_short_names(self):
        for name in ('오션 파이오니어 II 20','Ocean 1A2 Pioneer IV','A가BC 12','시더 7','가',None):
            for op in NAME_OPERATIONS:
                for n in range(15):
                    q=change_name(name,op,n)
                    self.assertEqual(re.findall(r'\d+',str(q)),re.findall(r'\d+',str(name)))
                    self.assertEqual(has_hangul(q),has_hangul(name))
                    if ' II ' in str(name):self.assertIn(' II ',q)
                    if str(name).endswith(' IV'):self.assertTrue(q.endswith(' IV'))
        self.assertEqual(change_name('시더 7','double_delete',1),'시더 7')
        self.assertFalse(training_query({**self.row(),'gross_tonnage':np.nan,'international_gross_tonnage':None},'row')[1]['effective'])
        q,info=training_query({**self.row(),'gross_tonnage':0.,'international_gross_tonnage':1200.},'row')
        self.assertTrue(info['effective']);self.assertEqual(q['gross_tonnage'],0.)

    def test_prior_sixteen_inputs_exact_and_new_recipes_predeclared(self):
        row=self.row()
        for recipe in REGRESSION_RECIPES:self.assertEqual(input_key(evaluation(row,recipe)),input_key(evaluation_query(row,recipe)))
        for recipe in NEW_RECIPES:
            q=evaluation(row,recipe)
            self.assertAlmostEqual(q['gross_tonnage'],450. if recipe.endswith('x0_45') else 1800.)
            self.assertEqual(len(row['vessel_name'])-len(q['vessel_name']),2 if 'double_delete' in recipe else 0)
        self.assertNotIn(.45,TRAIN_FACTORS);self.assertNotIn(1.8,TRAIN_FACTORS)
        self.assertNotIn(.55,TRAIN_FACTORS);self.assertEqual(len(EVAL_RECIPES),20)
        with self.assertRaises(ValueError):evaluation(row,'invalid')

    def test_replay_keeps_original_mass_and_every_parent_group(self):
        base=pd.DataFrame({'source':['a','a','a','b','c'],'variant':['original','original','missing','original','original'],
                           'mass':[.25,.25,.5,1.,1.]})
        prior=pd.DataFrame({'source':['a','b']});extra=pd.DataFrame({'source':['a','a','b']})
        b,p,e=replay_masses(base,prior,extra)
        np.testing.assert_allclose(b,[.25,.25,.15,.75,1.])
        np.testing.assert_allclose(p,[.15,.15]);np.testing.assert_allclose(e,[.1,.1,.1])
        self.assertTrue((np.r_[b,p,e]>0).all())
        with self.assertRaises(ValueError):replay_masses(base,prior,pd.DataFrame({'source':['c']}))
        with self.assertRaises(ValueError):replay_masses(base,prior,pd.DataFrame({'source':['unknown']}))

    def sampling_data(self):
        candidates=pd.DataFrame({'catalog_index':np.arange(200)})
        features=pd.DataFrame({'name_context_ratio':np.arange(200)/200,'name_vowel_fold_ratio':np.arange(200)/200,
                              'phonetic_ratio':np.arange(200)/200,'best_tonnage_gap':1-np.arange(200)/200})
        return candidates,features

    def test_added_negatives_change_exposure_without_changing_positive_or_budget(self):
        c,f=self.sampling_data();usual=np.arange(129);old=np.arange(150)
        chosen=focused_positions(c,f,0,usual,old)
        self.assertEqual(len(chosen),len(usual));self.assertIn(0,chosen)
        self.assertEqual(sum(c.iloc[chosen].catalog_index>=150),32)
        self.assertEqual(len(set(chosen)),len(chosen))
        np.testing.assert_array_equal(c.catalog_index,np.arange(200))
        # A target recovered by actual append is valid training evidence, never a fabricated row.
        usual=np.r_[np.arange(128),199]
        chosen=focused_positions(c,f,199,usual,old)
        self.assertIn(199,chosen);self.assertEqual(len(chosen),129)
        with self.assertRaises(ValueError):focused_positions(c,f,300,usual,old)
        with self.assertRaises(ValueError):focused_positions(c,f,0,usual,old,quota=-1)

    def test_sampling_ties_permutation_nonhangul_and_no_added_candidates(self):
        c,f=self.sampling_data();f[:]=1.;old=np.arange(150);usual=np.arange(129)
        a=c.iloc[focused_positions(c,f,0,usual,old)].catalog_index.tolist()
        perm=np.random.default_rng(2).permutation(200);d=c.iloc[perm].reset_index(drop=True);g=f.iloc[perm].reset_index(drop=True)
        chosen=np.flatnonzero(d.catalog_index.isin(c.iloc[usual].catalog_index))
        b=d.iloc[focused_positions(d,g,0,chosen,old)].catalog_index.tolist()
        self.assertEqual(sorted(a),sorted(b))
        np.testing.assert_array_equal(focused_positions(c,f,0,usual,c.catalog_index.to_numpy()),usual)


class RepairIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('joint_repair_fixture',Path(__file__).with_name('test_joint_search.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        fixture=module.FactorialIntegrationTests;fixture.setUpClass();cls.addClassCleanup(fixture.doClassCleanups)
        cls.root=fixture.root;cls.fixture=fixture
        from portmatch.joint_search_experiment import freeze,run_fold
        from portmatch.joint_search_report import report
        cls.parent=cls.root/'joint_parent';freeze(cls.parent,fixture.parent);run_fold(cls.parent,0);report(cls.parent,100)
        from portmatch.tonnage_repair_experiment import freeze as repair_freeze,run_fold as repair_run
        cls.out=cls.root/'repair';repair_freeze(cls.out,cls.parent);repair_run(cls.out,0)

    def test_end_to_end_audit_native_models_and_preservation(self):
        from portmatch.tonnage_repair_experiment import check
        from portmatch.tonnage_repair_report import report
        report(self.out,replicates=100)
        audit=read_json(self.out/'analysis_audit.json');self.assertTrue(audit['complete'])
        self.assertTrue(audit['original_mass_unchanged']);self.assertTrue(audit['previous_sixteen_conditions_exact'])
        frozen=check(self.out,protection=True)
        for path,expected in frozen['protected_files'].items():self.assertEqual(sha(path),expected)
        p=pd.read_csv(self.out/'fold0/predictions.csv')
        self.assertEqual(len(p),p.sample_id.nunique()*20*7);self.assertEqual(set(p.model),set(ARMS))
        self.assertEqual(set(p.variant),set(EVAL_RECIPES))
        self.assertEqual(len(read_json(self.out/'confidence_intervals.json')['comparisons']),20*12)
        gate=read_json(self.out/'retention_gates.json')
        self.assertEqual(len(gate['retention_conditions']),15);self.assertEqual(len(gate['gates']),4)
        self.assertTrue(all(not x['strict_development_gate_pass'] for x in gate['gates']))
        a=pd.read_csv(self.out/'fold0/broad/query_masses.csv');b=pd.read_csv(self.out/'fold0/focused/query_masses.csv')
        pd.testing.assert_frame_equal(a,b)
        self.assertGreater((a.kind=='repair').sum(),0)
        np.testing.assert_allclose(a.groupby('source').mass.sum(),1)

    def test_restored_predictions_ignore_labels_ids_year_and_order(self):
        from portmatch.tonnage_repair_experiment import TonnageRepairMatcher
        rows=self.fixture.train.head(2).to_dict('records')
        for arm in NEW_ARMS:
            matcher=TonnageRepairMatcher(self.out,0,arm)
            queries=[evaluation(r,'fresh_double_delete_gt_x0_55') for r in rows]
            expected=[matcher.predict(q) for q in queries]
            changed=[{**q,'sample_id':'OTHER','arrival_year':1900,'target_vessel_id':'DO_NOT_USE'} for q in reversed(queries)]
            self.assertEqual(expected,[matcher.predict(q) for q in changed][::-1])

    def test_completed_runner_reuses_models_and_refuses_tampered_evidence(self):
        from scripts.run_tonnage_repair_study import run,status
        from scripts.run_compound_study import runner_lock
        model=self.out/'fold0/focused/model.txt';before=sha(model)
        with runner_lock(self.out):self.assertEqual(run(self.out,1),0)
        self.assertEqual(sha(model),before);self.assertEqual(status(self.out)['folds'],{'0':'complete'})
        log=self.out/'fold0/repair_log.csv';saved=log.read_bytes()
        try:
            log.write_bytes(saved+b'\n')
            with self.assertRaises(ValueError):run(self.out,1)
        finally:log.write_bytes(saved)

    def test_freeze_rejects_changed_parent_completion_evidence(self):
        from portmatch.tonnage_repair_experiment import freeze
        path=self.parent/'fold0/joint_log.csv';saved=path.read_bytes()
        try:
            path.write_bytes(saved+b'\n')
            with self.assertRaisesRegex(ValueError,'Parent completion evidence changed'):
                freeze(self.root/'invalid_parent_snapshot',self.parent)
            self.assertFalse((self.root/'invalid_parent_snapshot').exists())
        finally:path.write_bytes(saved)

    def test_missing_positive_is_not_injected(self):
        from portmatch.tonnage_repair_experiment import prepare_repair
        from portmatch.compound_experiment import _group_meta
        from portmatch.ranker_commands import RankedMatcher
        from portmatch.joint_search_experiment import query_arrays
        out=self.root/'missing_positive';out.mkdir()
        baseline=RankedMatcher(self.fixture.baseline)
        def empty_target(*args):
            old,ids,x,y,added,a,b=query_arrays(*args)
            return old,np.full_like(ids,999),x,y,added,a,b
        with patch('portmatch.tonnage_repair_experiment.query_arrays',side_effect=empty_target):
            with self.assertRaisesRegex(ValueError,'No usable repair groups'):
                prepare_repair(self.fixture.train,self.fixture.catalog,pd.read_csv(self.fixture.cache/'partitions.csv'),
                    baseline,read_json(self.fixture.baseline/'training_summary.json')['dataset'],_group_meta(self.fixture.cache),
                    pd.read_csv(self.parent/'fold0/extra_groups.csv'),out)
        self.assertEqual(list((out/'extra').glob('*.npz')),[])


if __name__=='__main__':unittest.main()
