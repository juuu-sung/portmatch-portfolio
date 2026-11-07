"""Predeclared augmentation/retrieval factorial; prior protocols stay immutable."""
from __future__ import annotations

import hashlib
import re
import numpy as np
import pandas as pd

from .compound_protocol import EVAL_RECIPES as REGRESSION_RECIPES, evaluation_variant, input_key
from .data import QUERY_COLUMNS
from .text import has_hangul

VERSION = 'joint_search_v1'
SEED = 20261031
TRAIN_FACTORS = (.65, .80, 1.20, 1.35)
FRESH_RECIPES = ('fresh_double_delete', 'fresh_double_delete_missing',
                 'fresh_double_delete_gt_x0_55', 'fresh_double_delete_gt_x1_50')
EVAL_RECIPES = (*REGRESSION_RECIPES, *FRESH_RECIPES)
ARMS = ('best', 'control', 'augmentation', 'retrieval', 'both')
PAIRS = (('control', 'augmentation'), ('control', 'retrieval'), ('control', 'both'),
         ('augmentation', 'both'), ('retrieval', 'both'),
         ('best', 'control'), ('best', 'augmentation'), ('best', 'retrieval'), ('best', 'both'))


def selector(row, salt):
    return int.from_bytes(hashlib.sha256(f'{SEED}:{salt}:{input_key(row)}'.encode()).digest()[:8], 'big')


def edit_name(value, number, operation='delete'):
    text = value if isinstance(value, str) else ''
    protected = set()
    for match in re.finditer(r'\b[IVXLCDM]+\b', text, flags=re.I):
        protected.update(range(match.start(), match.end()))
    positions = [i for i, c in enumerate(text) if c.isalpha() and i not in protected]
    if operation == 'swap':
        pairs = [i for i in positions if i+1 in positions and text[i] != text[i+1]]
        if not pairs:
            return text
        i = pairs[number % len(pairs)]
        result = text[:i] + text[i+1] + text[i] + text[i+2:]
    else:
        positions = [i for i in positions if not (has_hangul(text[i]) and sum(has_hangul(c) for c in text) == 1)]
        if len(positions) < 4:
            return text
        i = positions[number % len(positions)]
        result = text[:i] + text[i+1:]
    if re.findall(r'\d+', result) != re.findall(r'\d+', text) or has_hangul(result) != has_hangul(text):
        return text
    return result


def scale(row, factor):
    changed = False
    for field in ('gross_tonnage', 'international_gross_tonnage'):
        value = pd.to_numeric(row.get(field), errors='coerce')
        if value is not None and pd.notna(value) and np.isfinite(value) and value > 0:
            row[field] = float(value * factor)
            changed = True
    return changed


def joint_variant(source, view):
    row = {key: source.get(key) for key in QUERY_COLUMNS}
    n = selector(row, 'training:' + view)
    operation = ('delete', 'swap')[n % 2]
    factor = TRAIN_FACTORS[(n // 2) % len(TRAIN_FACTORS)]
    row['vessel_name'] = edit_name(row.get('vessel_name'), n // 8, operation)
    valid_gt = scale(row, factor)
    effective = valid_gt and isinstance(source.get('vessel_name'),str) and bool(source['vessel_name'].strip()) and row['vessel_name'] != source['vessel_name']
    return row, {'operation': operation, 'factor': factor, 'effective': bool(effective)}


def evaluation_query(source, recipe):
    if recipe in REGRESSION_RECIPES:
        return evaluation_variant(source, recipe)
    if recipe not in FRESH_RECIPES:
        raise ValueError('Unknown evaluation recipe')
    row = {key: source.get(key) for key in QUERY_COLUMNS}
    n = selector(row, 'fresh-evaluation-double-delete')
    original = row.get('vessel_name')
    name = edit_name(original, n)
    result = edit_name(name, n // 11)
    row['vessel_name'] = result if isinstance(original,str) and len(original)-len(result)==2 else original
    if recipe.endswith('_missing'):
        row['gross_tonnage'] = row['international_gross_tonnage'] = np.nan
    elif recipe.endswith('_x0_55'):
        scale(row, .55)
    elif recipe.endswith('_x1_50'):
        scale(row, 1.50)
    return row


def protocol():
    return {'version': VERSION, 'seed': SEED, 'arms': list(ARMS),
        'control': 'Completed compound-v1 expanded (heuristic, no learned mining)',
        'augmentation': 'Replace only retained dual_gt_noise groups with one letter deletion/transposition AND wrong positive GT; factors .65/.80/1.20/1.35',
        'fallback': 'If no effective joint change, missing positive, or different sampled group size, retain the original control group; never inject a positive',
        'training_controls': 'Identical base cache, group order, pair count, source/language masses, 86 features and 300-round config; change only eligible extra group contents',
        'scope': 'Measures the joint-error recipe policy including wider GT severity, not a pure name-versus-GT interaction within augmentation',
        'retrieval': 'Catalog-only Hangul char 1-2 TF-IDF top512 + OSA edit top256; at most256 new vessels; no GT gates; append only',
        'retrieval_features': 'Common candidate records/features/scores unchanged; new rows use phonetic_hit=1, score=max(new channels), rank after old maximum, channel_count=1',
        'regression_recipes': list(REGRESSION_RECIPES), 'fresh_recipes': list(FRESH_RECIPES),
        'held_out': 'Two letter deletions and GT .55/1.50 absent from training; fresh means predeclared synthetic cases, not independent vessels or Public',
        'pairs': [list(p) for p in PAIRS], 'interaction': 'both - augmentation - retrieval + control',
        'selection': 'No auto promotion or submission; review clean, regression and fresh conditions separately, with candidate recall and paired vessel uncertainty'}
