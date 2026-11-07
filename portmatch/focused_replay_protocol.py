"""Predeclared name/moderate-GT replay ablation on the focused_search baseline."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from .compound_protocol import input_key
from .data import QUERY_COLUMNS
from .joint_search_protocol import scale
from .tonnage_repair_protocol import change_name, evaluation as previous_evaluation
from .tonnage_repair_protocol import EVAL_RECIPES as REGRESSION_RECIPES

VERSION = 'focused_replay_v1'
SEED = 20261102
OPERATIONS = ('delete', 'swap', 'double_delete')
MODERATE_FACTORS = (.625, .675, .75, .80)
FAMILIES = ('names', 'moderate')
NEW_ARMS = ('names', 'moderate', 'combined')
ARMS = ('focused_search', *NEW_ARMS)
FRESH_RECIPES = ('replay_fresh_delete', 'replay_fresh_swap', 'replay_fresh_double_delete',
                 'replay_fresh_swap_gt_x0_72', 'replay_fresh_double_delete_gt_x0_68')
EVAL_RECIPES = (*REGRESSION_RECIPES, *FRESH_RECIPES)
TARGETS = ('unseen_swap_dual_gt_x0_70', 'name_delete', 'unseen_name_swap', 'fresh_double_delete')
CORE_RETENTION = ('original', 'fresh_double_delete_gt_x0_55', 'fresh_double_delete_gt_x1_50')
RETENTION = tuple(v for v in REGRESSION_RECIPES if v not in TARGETS)
TOLERANCE_PP = .25
PAIRS = (('focused_search', 'names'), ('focused_search', 'moderate'), ('focused_search', 'combined'),
         ('names', 'combined'), ('moderate', 'combined'), ('names', 'moderate'))


def selector(row, salt):
    return int.from_bytes(hashlib.sha256(f'{SEED}:{salt}:{input_key(row)}'.encode()).digest()[:8], 'big')


def training_query(source, view, family):
    if view not in ('vessel', 'row') or family not in FAMILIES:
        raise ValueError('Unknown inner view or replay family')
    row = {key: source.get(key) for key in QUERY_COLUMNS}
    number = selector(row, 'training:' + view)
    operation = OPERATIONS[number % len(OPERATIONS)]
    factor = MODERATE_FACTORS[(number // len(OPERATIONS)) % len(MODERATE_FACTORS)] if family == 'moderate' else 1.
    row['vessel_name'] = change_name(row['vessel_name'], operation, number // 12)
    name_changed = str(row['vessel_name']) != str(source.get('vessel_name'))
    gt_changed = scale(row, factor) if family == 'moderate' else False
    return row, {'operation': operation, 'factor': factor, 'name_changed': name_changed,
                 'gt_changed': bool(gt_changed), 'effective': bool(name_changed and (gt_changed or family == 'names'))}


def evaluation(source, recipe):
    if recipe in REGRESSION_RECIPES:
        return previous_evaluation(source, recipe)
    if recipe not in FRESH_RECIPES:
        raise ValueError('Unknown evaluation recipe')
    row = {key: source.get(key) for key in QUERY_COLUMNS}
    operation = 'double_delete' if 'double_delete' in recipe else 'swap' if 'swap' in recipe else 'delete'
    row['vessel_name'] = change_name(row['vessel_name'], operation, selector(row, 'held-out-name'))
    if recipe.endswith('x0_72'):
        scale(row, .72)
    elif recipe.endswith('x0_68'):
        scale(row, .68)
    return row


def replay_masses(previous, extra, arm):
    """Preserve original AND prior GT-repair masses; each active family has its own budget.

    Allocate min(.06, .20 * available older-corruption mass) per source/family.
    Budgets are independent: adding the second family cannot change the first
    family's mass. Unusable new groups simply leave their budget with old data.
    """
    if arm not in NEW_ARMS:
        raise ValueError('Unknown replay arm')
    if previous['query'].duplicated().any() or extra['query'].duplicated().any():
        raise ValueError('Duplicate query groups')
    if not set(extra.recipe).issubset(FAMILIES) or not set(extra.source).issubset(set(previous.source)):
        raise ValueError('Unknown replay source/family')
    if (previous.mass <= 0).any() or not np.isfinite(previous.mass).all():
        raise ValueError('Invalid saved parent masses')
    if not np.allclose(previous.groupby('source').mass.sum(), 1., atol=1e-10, rtol=0):
        raise ValueError('Parent source masses must sum to one')
    active = set(FAMILIES) if arm == 'combined' else {arm}
    eligible = extra.recipe.isin(active)
    available = previous.loc[~previous.locked].groupby('source').mass.sum()
    counts = extra.loc[eligible].groupby(['source', 'recipe']).size()
    budget = {(source, family): min(.06, .20 * float(available.get(source, 0.))) for source, family in counts.index}
    if any(value <= 0 for value in budget.values()):
        raise ValueError('No unlocked corruption mass available for replay')
    used = {}
    for (source, _), value in budget.items():
        used[source] = used.get(source, 0.) + value
    old_mass = previous.mass.to_numpy(float).copy()
    for i, row in enumerate(previous.itertuples()):
        if not row.locked and row.source in used:
            old_mass[i] *= 1. - used[row.source] / available[row.source]
    new_mass = np.array([budget[(r.source, r.recipe)] / counts.loc[(r.source, r.recipe)]
                         if r.recipe in active else 0. for r in extra.itertuples()], dtype=float)
    totals = pd.DataFrame({'source': np.r_[previous.source, extra.source],
                           'mass': np.r_[old_mass, new_mass]}).groupby('source').mass.sum()
    if (not np.allclose(totals, 1., atol=1e-10, rtol=0) or (old_mass <= 0).any()
            or (new_mass[eligible] <= 0).any() or not np.isfinite(new_mass).all()):
        raise ValueError('Replay source mass conservation failed')
    if not np.array_equal(old_mass[previous.locked], previous.loc[previous.locked, 'mass']):
        raise AssertionError('Original/previous-repair masses changed')
    return old_mass, new_mass


def protocol():
    return {'version': VERSION, 'seed': SEED, 'baseline': 'Frozen tonnage_repair focused_search; Public .8515414798 user report',
        'arms': list(ARMS), 'operations': list(OPERATIONS), 'moderate_factors': list(MODERATE_FACTORS),
        'names': 'Name errors only; GT and other metadata unchanged, including missing GT',
        'moderate': 'Same name operations/positions plus .625/.675/.75/.80 GT; .70/.55/1.50 excluded from new training factors',
        'combined': 'Both independently budgeted families; same groups reused from the single-family arms',
        'budget': 'Per source/family min(.06,.20*older corrupted mass); original and all prior GT-repair groups/masses unchanged; all old examples retained',
        'retrieval': 'Unchanged fixed parent index and HangulExpansion; same candidate sets/features for every arm',
        'negatives': 'Existing focused quota32 added distractors; same 86 features, query pair budget and ranker config; no teacher, no target injection',
        'split': 'Parent outer vessel folds; inner vessel/row pipelines exclude held input rows; outer validation labels never used for training decisions',
        'regression': list(REGRESSION_RECIPES), 'fresh': list(FRESH_RECIPES), 'targets': list(TARGETS),
        'core_retention': list(CORE_RETENTION), 'retention': list(RETENTION), 'noninferiority_margin_pp': TOLERANCE_PP,
        'gate': 'All four targets improve pooled; three core conditions have no point loss pooled or fold0; all16 retention deltas >=-.25pp; report CIs/fold0 separately. Never auto-promote.',
        'selection': 'Known targets are regression cases, not independent holdout; five extra recipes predeclared on same public vessels, no Public guarantee',
        'pairs': [list(pair) for pair in PAIRS]}
