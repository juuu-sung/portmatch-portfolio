"""Frozen, row-independent corruption recipes for the compound-error study.

Only official query values enter recipe selection. Targets are never accepted.
Training uses one of six recipes per source/view. Evaluation additionally holds
out transpositions and GT factors, so training and testing are not identical.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from .data import QUERY_COLUMNS
from .text import has_hangul

VERSION = "compound_errors_v1"
SEED = 20261030
ACTIVE_FIELDS = ("vessel_name", "call_sign", "gross_tonnage",
                 "international_gross_tonnage", "flag_name", "vessel_type")
TRAIN_RECIPES = ("delete", "spacing", "categories", "dual_gt_noise",
                 "delete_missing_gt", "missing_metadata")
EVAL_RECIPES = ("original", "both_gt_missing", "international_gt_missing",
                "gross_gt_missing", "international_gt_x1_25", "both_gt_x1_25",
                "flag_type_missing", "all_metadata_missing", "name_delete",
                "name_delete_both_gt_missing", "unseen_name_swap",
                "unseen_swap_dual_gt_x0_70")


def input_key(row):
    values = []
    for field in ACTIVE_FIELDS:
        value = row.get(field)
        if value is None or pd.isna(value):
            value = None
        elif isinstance(value, (np.integer, np.floating)):
            value = value.item()
        values.append(value)
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def choice(row, salt):
    return int.from_bytes(hashlib.sha256(
        f"{SEED}:{salt}:{input_key(row)}".encode("utf-8")).digest()[:8], "big")


def _letter_positions(text):
    # Roman numeral tokens, digit sequences, and the final Hangul character
    # of an otherwise Latin name must survive corruption.
    protected = set()
    for match in re.finditer(r"\b[IVXLCDM]+\b", text):
        protected.update(range(match.start(), match.end()))
    hangul_count = sum(has_hangul(c) for c in text)
    return [i for i, c in enumerate(text) if c.isalpha() and i not in protected
            and not (has_hangul(c) and hangul_count == 1)]


def delete_letter(text, selector=None):
    text = text if isinstance(text, str) else ""
    positions = _letter_positions(text)
    if len(positions) < 4:
        return text
    pos = positions[len(positions) // 2 if selector is None else selector % len(positions)]
    return text[:pos] + text[pos + 1:]


def swap_letters(text):
    text = text if isinstance(text, str) else ""
    eligible = set(_letter_positions(text))
    positions = [i for i in eligible if i + 1 in eligible and text[i] != text[i + 1]]
    if not positions:
        return text
    pos = sorted(positions)[len(positions) // 2]
    return text[:pos] + text[pos + 1] + text[pos] + text[pos + 2:]


def mask_gt(row):
    for field in ("gross_tonnage", "international_gross_tonnage"):
        row[field] = np.nan


def scale_gt(row, factor, fields=("gross_tonnage", "international_gross_tonnage")):
    for field in fields:
        value = pd.to_numeric(row.get(field), errors="coerce")
        if np.isfinite(value) and value > 0:
            row[field] = float(value * factor)


def training_variant(row, view):
    query = {key: row.get(key) for key in QUERY_COLUMNS}
    selector = choice(query, "training:" + view)
    recipe = TRAIN_RECIPES[selector % len(TRAIN_RECIPES)]
    if recipe in ("delete", "delete_missing_gt"):
        query["vessel_name"] = delete_letter(query.get("vessel_name"), selector // 6)
    if recipe == "spacing":
        query["vessel_name"] = re.sub(r"(?<=[^\W\d_])\s+(?=[^\W\d_])", "",
                                      str(query.get("vessel_name", "")))
    if recipe == "dual_gt_noise":
        scale_gt(query, .85 if (selector // 6) % 2 else 1.15)
    if recipe in ("delete_missing_gt", "missing_metadata"):
        mask_gt(query)
    if recipe in ("categories", "missing_metadata"):
        for field in ("flag_name", "vessel_type"):
            query[field] = np.nan
    if recipe == "missing_metadata":
        query["call_sign"] = np.nan
    assert has_hangul(query.get("vessel_name")) == has_hangul(row.get("vessel_name"))
    return recipe, query


def evaluation_variant(row, recipe):
    if recipe not in EVAL_RECIPES:
        raise ValueError("Unknown evaluation recipe")
    query = {key: row.get(key) for key in QUERY_COLUMNS}
    if recipe in ("both_gt_missing", "all_metadata_missing", "name_delete_both_gt_missing"):
        mask_gt(query)
    if recipe == "international_gt_missing":
        query["international_gross_tonnage"] = np.nan
    if recipe == "gross_gt_missing":
        query["gross_tonnage"] = np.nan
    if recipe == "international_gt_x1_25":
        scale_gt(query, 1.25, ("international_gross_tonnage",))
    if recipe == "both_gt_x1_25":
        scale_gt(query, 1.25)
    if recipe in ("flag_type_missing", "all_metadata_missing"):
        query["flag_name"] = query["vessel_type"] = np.nan
    if recipe == "all_metadata_missing":
        query["call_sign"] = np.nan
    if recipe in ("name_delete", "name_delete_both_gt_missing"):
        query["vessel_name"] = delete_letter(query.get("vessel_name"))
    if recipe in ("unseen_name_swap", "unseen_swap_dual_gt_x0_70"):
        query["vessel_name"] = swap_letters(query.get("vessel_name"))
    if recipe == "unseen_swap_dual_gt_x0_70":
        scale_gt(query, .70)
    assert has_hangul(query.get("vessel_name")) == has_hangul(row.get("vessel_name"))
    return query


def protocol():
    return {
        "version": VERSION, "seed": SEED,
        "evaluation_recipes": list(EVAL_RECIPES), "training_recipes": list(TRAIN_RECIPES),
        "training_budget": "One deterministic added recipe per original source and OOF view; all training sources eligible",
        "held_out_corruptions": ["adjacent-letter transposition", "GT x1.25", "GT x0.70"],
        "training_gt_factors": [.85, 1.15],
        "source_mass": "Original 0.5, legacy GT masks 0.25, added views 0.25; absent legacy share goes to original; without added view retain baseline weights",
        "teacher": "3 vessel-disjoint teachers using 54 query/catalog features; pairs retrieved from catalog-only indexes with zero training rows, so even candidate sampling has no held-out label dependency; one source per fitting vessel, original and both-GT-missing groups; 100 rounds",
        "mining": "Added views only: up to 64 highest-scoring teacher negatives plus retained heuristic negatives to preserve the original candidate budget; base cache groups unchanged",
        "ranker": "86 features; same 300-round ranker configuration as baseline",
        "metrics": "0.7 Hangul Top1 + 0.3 non-Hangul Top1, reported separately for every condition, with candidate recall and errors by vessel",
        "selection": "No automatic promotion; clean accuracy, all corruption conditions, new errors, fold consistency and clustered uncertainty reviewed together",
        "evaluation_note": "All existing folds are development folds; frozen new corruption tests are not independent hidden-test estimates",
    }
