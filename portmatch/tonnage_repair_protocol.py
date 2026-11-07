"""Frozen replay and candidate-exposure policy; previous studies stay immutable."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from .compound_protocol import input_key
from .data import QUERY_COLUMNS
from .joint_search_protocol import EVAL_RECIPES as REGRESSION_RECIPES, edit_name, evaluation_query, scale

VERSION = 'tonnage_repair_v1'
SEED = 20261101
TRAIN_FACTORS = (.4, .5, .625, .8, 1.25, 1.6, 2., 2.5)
NAME_OPERATIONS = ('keep', 'delete', 'double_delete', 'swap')
NEW_RECIPES = ('repair_keep_gt_x0_45', 'repair_double_delete_gt_x0_45',
               'repair_keep_gt_x1_80', 'repair_double_delete_gt_x1_80')
EVAL_RECIPES = (*REGRESSION_RECIPES, *NEW_RECIPES)
ARMS = ('best', 'previous', 'previous_search', 'broad', 'broad_search', 'focused', 'focused_search')
NEW_ARMS = ARMS[3:]
TARGET = 'fresh_double_delete_gt_x0_55'
RETENTION = tuple(v for v in REGRESSION_RECIPES if v != TARGET)
TOLERANCE_PP = .25
PAIRS = (('previous','broad'), ('previous','focused'),
         ('previous_search','broad_search'), ('previous_search','focused_search'),
         ('broad','focused'), ('broad_search','focused_search'),
         ('broad','broad_search'), ('focused','focused_search'),
         *(( 'best', a) for a in NEW_ARMS))


def selector(row, salt):
    return int.from_bytes(hashlib.sha256(f'{SEED}:{salt}:{input_key(row)}'.encode()).digest()[:8], 'big')


def change_name(name, operation, number):
    if operation == 'keep':
        return name
    if operation == 'double_delete':
        first = edit_name(name, number)
        second = edit_name(first, number // 13)
        return second if isinstance(name, str) and len(name)-len(second) == 2 else name
    return edit_name(name, number, operation)


def training_query(source, view):
    row = {k:source.get(k) for k in QUERY_COLUMNS}
    n = selector(row, 'training:' + view)
    operation = NAME_OPERATIONS[n % len(NAME_OPERATIONS)]
    factor = TRAIN_FACTORS[(n // len(NAME_OPERATIONS)) % len(TRAIN_FACTORS)]
    row['vessel_name'] = change_name(row['vessel_name'], operation, n // 32)
    effective = scale(row, factor)
    return row, {'operation':operation, 'factor':factor, 'effective':bool(effective),
                 'name_changed':str(row['vessel_name']) != str(source.get('vessel_name'))}


def evaluation(source, recipe):
    if recipe in REGRESSION_RECIPES:
        return evaluation_query(source, recipe)
    if recipe not in NEW_RECIPES:
        raise ValueError('Unknown evaluation recipe')
    row = {k:source.get(k) for k in QUERY_COLUMNS}
    if 'double_delete' in recipe:
        row['vessel_name'] = change_name(row['vessel_name'], 'double_delete', selector(row, 'held-out-name'))
    scale(row, .45 if recipe.endswith('x0_45') else 1.8)
    return row


def replay_masses(base, parent, extra):
    """Keep original masses; keep every prior group, fund additions from nonoriginal mass."""
    from .compound_mining import expanded_masses
    b, p = expanded_masses(base.source, base.variant, base.mass, parent.source)
    meta = pd.DataFrame({'source':np.r_[base.source, parent.source],
                         'original':np.r_[base.variant.eq('original'), np.zeros(len(parent),bool)],
                         'mass':np.r_[b,p]})
    available = meta.loc[~meta.original].groupby('source').mass.sum()
    counts = extra.source.value_counts() if len(extra) else pd.Series(dtype=int)
    if not set(counts.index).issubset(set(meta.source)):
        raise ValueError('Repair source absent from retained training data')
    budget = {s:min(.20, .4*float(available.get(s,0.))) for s in counts.index}
    if any(v <= 0 for v in budget.values()):
        raise ValueError('No nonoriginal mass available for repair source')
    factor = meta.source.map({s:1-budget[s]/available[s] for s in budget}).fillna(1.)
    masses = meta.mass.to_numpy() * np.where(meta.original, 1., factor)
    new = extra.source.map({s:budget[s]/counts[s] for s in budget}).to_numpy(float)
    total = pd.DataFrame({'source':np.r_[meta.source,extra.source], 'mass':np.r_[masses,new]}).groupby('source').mass.sum()
    if not np.allclose(total,1.,atol=1e-10,rtol=0) or (masses<=0).any() or (new<=0).any():
        raise ValueError('Replay mass conservation failed')
    return masses[:len(base)], masses[len(base):], new


def focused_positions(candidates, features, target, usual, old_ids, quota=32):
    """Same positive/budget; expose added distractors using label-free text/GT signals."""
    ids = candidates.catalog_index.to_numpy(dtype=np.int64)
    usual = np.asarray(usual,dtype=np.int64)
    positive = np.flatnonzero(ids == target)
    if len(set(ids)) != len(ids) or len(positive)!=1 or positive[0] not in usual:
        raise ValueError('Never inject a missing positive')
    if quota < 0:
        raise ValueError('Negative quota')
    added = np.flatnonzero(~np.isin(ids,old_ids) & (ids != target))
    streams=[]
    for column, ascending in [('name_context_ratio',False),('name_vowel_fold_ratio',False),
                              ('phonetic_ratio',False),('best_tonnage_gap',True)]:
        if column not in features:
            continue
        values = features[column].to_numpy(dtype=float)
        eligible = added[np.isfinite(values[added])]
        scores = values[eligible] if ascending else -values[eligible]
        streams.append(list(eligible[np.lexsort((ids[eligible],scores))]))
    chosen=set(); pointers=[0]*len(streams); limit=min(quota,len(usual)-1,len(added))
    while len(chosen)<limit:
        advanced=False
        for i,stream in enumerate(streams):
            while pointers[i]<len(stream) and stream[pointers[i]] in chosen:
                pointers[i]+=1
            if pointers[i]<len(stream):
                chosen.add(int(stream[pointers[i]]));pointers[i]+=1;advanced=True
            if len(chosen)==limit:break
        if not advanced:break
    for pos in added[np.argsort(ids[added],kind='stable')]:
        if len(chosen)==limit:break
        chosen.add(int(pos))
    chosen.add(int(positive[0]))
    for pos in usual[np.argsort(ids[usual],kind='stable')]:
        if len(chosen)==len(usual):break
        chosen.add(int(pos))
    result=np.array(sorted(chosen),dtype=np.int64)
    if len(result)!=len(usual):raise AssertionError('Changed candidate budget')
    return result


def protocol():
    return {'version':VERSION,'seed':SEED,'train_factors':list(TRAIN_FACTORS),'name_operations':list(NAME_OPERATIONS),
        'replay':'Every prior base and joint group retained verbatim; original mass unchanged; up to .20/source transferred from 40% of nonoriginal mass',
        'new_groups':'Same OOF queries, expanded candidate union, positives, pair budgets and masses in both branches; no positive injection',
        'broad':'Usual heuristic sampling on expanded candidate union',
        'focused':'Same pool with up to 32 added negatives prioritized by text/GT feature ranks, then usual samples; no learned teacher',
        'scope':'Broad combines wider GT/name augmentation and training on expanded candidates; focused isolates added-negative sampling, not retrieval itself',
        'arms':list(ARMS),'regression':list(REGRESSION_RECIPES),'fresh':list(NEW_RECIPES),
        'target':TARGET,'retention':list(RETENTION),'noninferiority_margin_pp':TOLERANCE_PP,
        'gate':'Require target gain, all 15 retention deltas >= -.25pp; report paired CIs and fold0 separately. Never auto-promote.',
        'selection':'Known .55 failure is now regression; new .45/1.80 inputs predeclared, same public vessels, no Public claim'}
