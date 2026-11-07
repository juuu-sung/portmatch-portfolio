import importlib.util
from pathlib import Path
import re
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from portmatch.compound_experiment import read_json, sha
from portmatch.compound_protocol import input_key
from portmatch.focused_replay_protocol import (ARMS, CORE_RETENTION, EVAL_RECIPES, FAMILIES, FRESH_RECIPES,
    MODERATE_FACTORS, NEW_ARMS, OPERATIONS, REGRESSION_RECIPES, TARGETS, evaluation, replay_masses, training_query)
from portmatch.text import has_hangul


class ReplayPolicyTests(unittest.TestCase):
    def test_interrupt_stops_owned_interpreter(self):
        from scripts.run_focused_replay_study import stop_child
        from scripts.verify_submission import ProcessMemoryMonitor
        child = subprocess.Popen([sys.executable, '-u', '-c', "import time; print('ready',flush=True); time.sleep(20)"],
            stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        monitor = ProcessMemoryMonitor(child.pid)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            monitor.sample()
            stop_child(child, monitor)
            self.assertIsNotNone(child.poll())
            if os.name == 'nt':
                import ctypes
                from ctypes import wintypes as w
                monitor.kernel.GetExitCodeProcess.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
                monitor.kernel.GetExitCodeProcess.restype = w.BOOL
                for handle in monitor.handles.values():
                    code = w.DWORD()
                    self.assertTrue(monitor.kernel.GetExitCodeProcess(handle, ctypes.byref(code)))
                    self.assertNotEqual(code.value, 259)  # STILL_ACTIVE
        finally:
            stop_child(child, monitor); monitor.close(); child.stdout.close()

    def source(self, name='오션 파이오니어 II 20'):
        return dict(sample_id='S', vessel_name=name, call_sign='ABC', gross_tonnage=1000.,
                    international_gross_tonnage=1200., flag_name='Korea', vessel_type='Cargo',
                    arrival_year=2020, target_vessel_id='TARGET')

    def test_families_are_input_only_and_share_name_operations(self):
        factors, operations = set(), set()
        for i in range(150):
            source = self.source(f'오션 파이오니어 II {i}')
            before = input_key(source)
            name, ni = training_query(source, 'vessel', 'names')
            moderate, mi = training_query(source, 'vessel', 'moderate')
            changed = {**source, 'sample_id': 'OTHER', 'target_vessel_id': 'OTHER', 'arrival_year': 1900}
            other, oi = training_query(changed, 'vessel', 'moderate')
            self.assertEqual(input_key(moderate), input_key(other))
            self.assertEqual(mi, oi)
            self.assertEqual(name['vessel_name'], moderate['vessel_name'])
            self.assertEqual(ni['operation'], mi['operation'])
            self.assertEqual(name['gross_tonnage'], source['gross_tonnage'])
            self.assertEqual(name['international_gross_tonnage'], source['international_gross_tonnage'])
            self.assertAlmostEqual(moderate['gross_tonnage'] / source['gross_tonnage'], mi['factor'])
            self.assertNotIn('target_vessel_id', name)
            self.assertEqual(input_key(source), before)
            factors.add(mi['factor']); operations.add(mi['operation'])
        self.assertEqual(factors, set(MODERATE_FACTORS))
        self.assertEqual(operations, set(OPERATIONS))
        self.assertFalse(set(MODERATE_FACTORS) & {.55, .70, 1.50})

    def test_missing_gt_still_allows_name_only_replay(self):
        source = {**self.source(), 'gross_tonnage': np.nan, 'international_gross_tonnage': None}
        names, detail = training_query(source, 'row', 'names')
        self.assertTrue(detail['effective'])
        self.assertTrue(pd.isna(names['gross_tonnage']))
        self.assertIsNone(names['international_gross_tonnage'])
        _, detail = training_query(source, 'row', 'moderate')
        self.assertFalse(detail['effective'])

    def test_identity_digits_roman_numbers_language_and_input_independence(self):
        for name in ('오션 파이오니어 II 20', 'Ocean 1A2 Pioneer IV', 'A가BC 12', '시더 7', '가', None):
            for view in ('row', 'vessel'):
                for family in FAMILIES:
                    source = self.source(name)
                    row, _ = training_query(source, view, family)
                    if name is not None:
                        self.assertEqual(re.findall(r'\d+', str(row['vessel_name'])), re.findall(r'\d+', name))
                        self.assertEqual(has_hangul(row['vessel_name']), has_hangul(name))
                        if ' II ' in name: self.assertIn(' II ', row['vessel_name'])
                        if name.endswith(' IV'): self.assertTrue(row['vessel_name'].endswith(' IV'))
        with self.assertRaises(ValueError): training_query(self.source(), 'unknown', 'names')

    def test_fixed_regressions_and_predeclared_fresh_recipes(self):
        from portmatch.tonnage_repair_protocol import evaluation as old
        self.assertEqual(len(EVAL_RECIPES), 25)
        self.assertEqual(len(set(EVAL_RECIPES)), 25)
        self.assertTrue(set(CORE_RETENTION).isdisjoint(TARGETS))
        for recipe in REGRESSION_RECIPES:
            self.assertEqual(input_key(old(self.source(), recipe)), input_key(evaluation(self.source(), recipe)))
        for recipe in FRESH_RECIPES:
            row = evaluation(self.source(), recipe)
            self.assertNotIn('target_vessel_id', row)
            self.assertEqual(input_key(row), input_key(evaluation({**self.source(), 'target_vessel_id': 'OTHER'}, recipe)))

    def masses(self):
        previous = pd.DataFrame({'source': ['a']*4 + ['b']*3 + ['c'],
            'query': ['a0','a1','a2','a3','b0','b1','b2','c0'],
            'mass': [.5,.2,.15,.15,.75,.1,.15,1.], 'locked': [True,True,False,False,True,True,False,True]})
        extra = pd.DataFrame({'source': ['a','a','a','b'], 'query': ['an0','an1','am','bn'],
                              'recipe': ['names','names','moderate','names']})
        return previous, extra

    def test_mass_conservation_locked_repair_and_factorial_budgets(self):
        previous, extra = self.masses()
        old, added = replay_masses(previous, extra, 'combined')
        np.testing.assert_allclose(old, [.5,.2,.09,.09,.75,.1,.12,1.], atol=1e-12)
        np.testing.assert_allclose(added, [.03,.03,.06,.03], atol=1e-12)
        np.testing.assert_array_equal(old[previous.locked], previous.loc[previous.locked,'mass'])
        for family in FAMILIES:
            _, single = replay_masses(previous, extra, family)
            mask = extra.recipe.eq(family)
            np.testing.assert_array_equal(single[mask], added[mask])
            self.assertTrue((single[~mask] == 0).all())
        totals = pd.DataFrame({'source': np.r_[previous.source, extra.source], 'mass': np.r_[old,added]}).groupby('source').mass.sum()
        np.testing.assert_allclose(totals, 1., atol=1e-12)

    def test_invalid_sources_budget_and_duplicate_groups_rejected(self):
        previous, extra = self.masses()
        bad = extra.copy(); bad.loc[0, 'source'] = 'unknown'
        with self.assertRaises(ValueError): replay_masses(previous, bad, 'combined')
        bad.loc[0, 'source'] = 'c'
        with self.assertRaises(ValueError): replay_masses(previous, bad, 'combined')
        bad = pd.concat([extra, extra.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError): replay_masses(previous, bad, 'combined')
        with self.assertRaises(ValueError): replay_masses(previous, extra, 'focused_search')


class ReplayIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('replay_parent_fixture', Path(__file__).with_name('test_tonnage_repair.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cls.fixture = module.RepairIntegrationTests
        cls.fixture.setUpClass(); cls.addClassCleanup(cls.fixture.doClassCleanups)
        cls.root = cls.fixture.root
        cls.parent = cls.fixture.out
        cls.out = cls.root / 'focused_replay'
        from portmatch.focused_replay_experiment import freeze, run_fold
        from portmatch.focused_replay_report import report
        with patch('portmatch.focused_replay_experiment.current_assets', return_value={}):
            cls.frozen = freeze(cls.out, cls.parent)
        run_fold(cls.out, 0)
        report(cls.out, replicates=100)

    def test_full_pipeline_audit_regressions_and_native_independence(self):
        from portmatch.focused_replay_experiment import check
        from scripts.run_focused_replay_study import status
        check(self.out, protection=True)
        audit = read_json(self.out / 'analysis_audit.json')
        for key in ('complete', 'previous_twenty_conditions_exact', 'original_and_prior_repair_mass_unchanged',
                    'family_mass_equal_single_and_combined', 'candidate_sets_identical_all_arms'):
            self.assertTrue(audit[key])
        p = pd.read_csv(self.out / 'fold0/predictions.csv')
        self.assertEqual(len(p), p.sample_id.nunique()*len(EVAL_RECIPES)*len(ARMS))
        self.assertEqual(set(p.model), set(ARMS))
        self.assertEqual(set(p.variant), set(EVAL_RECIPES))
        self.assertEqual(status(self.out)['folds'], {'0':'complete'})
        gates = read_json(self.out / 'retention_gates.json')
        self.assertEqual(set(gates['core_retention']), set(CORE_RETENTION))
        self.assertTrue(all(not g['automatic_promotion'] and not g['all_five_folds'] for g in gates['gates']))
        self.assertEqual(len(read_json(self.out/'confidence_intervals.json')['comparisons']), 25*7)
        self.assertEqual(len(pd.read_csv(self.out/'interactions.csv')), 25)

    def test_no_parent_mutation_or_overwrite_and_completed_fold_reuse(self):
        from portmatch.focused_replay_experiment import freeze, run_fold
        from scripts.run_focused_replay_study import run
        for path, expected in self.frozen['protected_files'].items():
            self.assertEqual(sha(path), expected)
        with self.assertRaises(FileExistsError): freeze(self.out, self.parent)
        with self.assertRaises(FileExistsError): run_fold(self.out, 0)
        model_hash = sha(self.out / 'fold0/combined/model.txt')
        self.assertEqual(run(self.out, 1), 0)
        self.assertEqual(sha(self.out / 'fold0/combined/model.txt'), model_hash)

    def test_altered_parent_and_new_model_fail_closed(self):
        from portmatch.focused_replay_experiment import FocusedReplayMatcher, check
        path = self.parent / 'fold0/focused/model.txt'; original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with self.assertRaises(ValueError): check(self.out, protection=True)
            with self.assertRaises(ValueError): FocusedReplayMatcher(self.out, 0, 'focused_search')
        finally:
            path.write_bytes(original)
        path = self.out / 'fold0/combined/model.txt'; original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with self.assertRaises(ValueError): FocusedReplayMatcher(self.out, 0, 'combined')
        finally:
            path.write_bytes(original)

    def test_missing_targets_are_skipped_without_injection(self):
        from portmatch.focused_replay_experiment import prepare_extra, previous_groups
        from portmatch.ranker_commands import RankedMatcher
        from portmatch.joint_search_experiment import query_arrays
        fixture = self.fixture.fixture
        previous, *_ = previous_groups(fixture.cache, self.parent/'fold0')
        out = self.root/'no_target'; out.mkdir()
        def wrong_ids(*args):
            old, ids, x, y, added, a, b = query_arrays(*args)
            return old, np.full_like(ids, 99999), x, y, added, a, b
        with patch('portmatch.focused_replay_experiment.query_arrays', side_effect=wrong_ids):
            with self.assertRaisesRegex(ValueError, 'Both replay families'):
                prepare_extra(fixture.train, fixture.catalog, pd.read_csv(fixture.cache/'partitions.csv'),
                    RankedMatcher(fixture.baseline), read_json(fixture.baseline/'training_summary.json')['dataset'], previous, out)
        self.assertEqual(list((out/'extra').glob('*.npz')), [])


if __name__ == '__main__':
    unittest.main()
