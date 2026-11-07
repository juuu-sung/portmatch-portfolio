"""Stateless mining and source-mass utilities, independent of old cache formats."""
from __future__ import annotations

import numpy as np


def teacher_features(names):
    return [name for name in names if not name.startswith(("retrieval_", "flag_", "type_"))]


def mined_positions(indices, target, heuristic, scores, max_mined=64):
    indices = np.asarray(indices)
    scores = np.asarray(scores)
    heuristic = np.asarray(heuristic, dtype=np.int64)
    if len(indices) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid teacher scores")
    positive = np.flatnonzero(indices == target)
    if len(positive) != 1 or positive[0] not in heuristic:
        raise ValueError("Never inject a missing positive")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("Duplicate candidates")
    negatives = np.flatnonzero(indices != target)
    order = negatives[np.lexsort((indices[negatives], -scores[negatives]))]
    budget = len(heuristic) - 1
    chosen = set(int(p) for p in order[:min(max_mined, budget)])
    for pos in heuristic[np.argsort(indices[heuristic], kind="stable")]:
        if indices[pos] != target and len(chosen) < budget:
            chosen.add(int(pos))
    chosen.add(int(positive[0]))
    result = np.array(sorted(chosen), dtype=np.int64)
    if len(result) != len(heuristic):
        raise AssertionError("Mining changed the group budget")
    return result


def expanded_masses(sources, variants, old_mass, added_sources):
    sources, variants = np.asarray(sources), np.asarray(variants)
    old_mass = np.asarray(old_mass, dtype=float)
    if not (len(sources) == len(variants) == len(old_mass)) or (old_mass <= 0).any():
        raise ValueError("Invalid base group metadata")
    added_sources = np.asarray(added_sources)
    new_mass = old_mass.copy()
    extra_mass = np.zeros(len(added_sources), dtype=float)
    for source in np.unique(sources):
        base = sources == source
        if not np.isclose(old_mass[base].sum(), 1, atol=1e-10):
            raise ValueError("Original source mass must equal one")
        added = added_sources == source
        if not added.any():
            continue
        original = base & (variants == "original")
        legacy = base & ~original
        if not original.any():
            raise ValueError("Source has no original training group")
        original_budget = .5 if legacy.any() else .75
        new_mass[original] = original_budget * old_mass[original] / old_mass[original].sum()
        if legacy.any():
            new_mass[legacy] = .25 * old_mass[legacy] / old_mass[legacy].sum()
        extra_mass[added] = .25 / added.sum()
    if len(added_sources) and ((extra_mass <= 0).any() or not set(added_sources).issubset(set(sources))):
        raise ValueError("Added group does not belong to a training source")
    return new_mass, extra_mass
