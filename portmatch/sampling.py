"""Query-local hard-negative sampling for training groups only.

Evaluation must keep the original candidate union. This module never adds a
missing positive: a failed retrieval returns an empty selection for the caller
to count and skip. Returned values are row positions for ``DataFrame.iloc``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SamplingConfig:
    hard_negatives: int = 96
    random_negatives: int = 32
    seed: int = 20261022

    def __post_init__(self) -> None:
        for name in ("hard_negatives", "random_negatives", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.hard_negatives < 0 or self.random_negatives < 0:
            raise ValueError("Negative sample counts must be nonnegative")
        if self.hard_negatives + self.random_negatives < 1:
            raise ValueError("At least one negative must be requested")


# Each view contributes its own difficult distractors instead of selecting only
# tonnage-nearest vessels or allowing one scale to dominate a combined score.
_HARD_VIEWS = (
    ("name_edit_similarity", False),
    ("romanized_ratio", False),
    ("phonetic_ratio", False),
    ("international_relative_gap", True),
    ("best_tonnage_gap", True),
    ("flag_lift_log", False),
    ("type_lift_log", False),
)


def _values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _metadata(candidates: pd.DataFrame, features: pd.DataFrame, name: str) -> np.ndarray:
    if name in candidates:
        values = _values(candidates, name)
    elif "retrieval_" + name in features:
        values = _values(features, "retrieval_" + name)
    else:
        return np.zeros(len(candidates), dtype=float)
    return np.where(np.isfinite(values), values, 0.0)


def sample_training_candidates(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    target_catalog_index: int,
    query_id: str,
    config: SamplingConfig | None = None,
) -> np.ndarray:
    """Keep the positive, diverse hard negatives, then random remaining negatives.

    Hard negatives start with conflicting call-sign/name lookups, then round-robin
    the text, phonetic, tonnage, flag and type views. Uninformative constant views
    are skipped. When no view distinguishes remaining candidates, catalog-index
    order fills the hard quota. Random sampling uses a stable query-specific seed
    and catalog-index order, so input execution order and candidate permutation
    cannot change the selected vessels. All ties are deterministic.

    Feature rows must have the same index/order as candidate rows. Target IDs and
    labels are used only to preserve the positive and exclude it from negatives.
    The input frames are never modified.
    """
    config = config or SamplingConfig()
    if len(candidates) != len(features) or not candidates.index.equals(features.index):
        raise ValueError("Feature rows must align with candidate rows")
    if not candidates.index.is_unique or not features.columns.is_unique:
        raise ValueError("Candidate index and feature columns must be unique")
    if not candidates.columns.is_unique or "catalog_index" not in candidates:
        raise ValueError("Candidates need a unique catalog_index column")
    if (isinstance(target_catalog_index, bool) or
            not isinstance(target_catalog_index, Integral) or target_catalog_index < 0):
        raise ValueError("target_catalog_index must be a nonnegative integer")
    if not isinstance(query_id, str):
        raise ValueError("query_id must be a stable string identifier")
    raw_ids = _values(candidates, "catalog_index")
    if (not np.isfinite(raw_ids).all() or (raw_ids < 0).any() or
            not np.equal(raw_ids, np.floor(raw_ids)).all() or
            (raw_ids >= np.iinfo(np.int64).max).any()):
        raise ValueError("catalog_index values must be nonnegative integer positions")
    ids = raw_ids.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Candidates must not repeat a catalog_index")
    positive = np.flatnonzero(ids == target_catalog_index)
    if not len(positive):
        return np.empty(0, dtype=np.int64)
    negatives = np.flatnonzero(ids != target_catalog_index)
    budget = config.hard_negatives + config.random_negatives
    if len(negatives) <= budget:
        return np.arange(len(candidates), dtype=np.int64)

    hard_budget = min(config.hard_negatives, len(negatives))
    selected: set[int] = set()
    # A false lookup can be especially dangerous, even when text or GT disagree.
    call = _metadata(candidates, features, "call_match")
    alias = _metadata(candidates, features, "alias_match")
    purity = _metadata(candidates, features, "lookup_purity")
    support = _metadata(candidates, features, "lookup_support")
    false_lookup = negatives[(call[negatives] > 0) | (alias[negatives] > 0)]
    lookup_order = np.lexsort((ids[false_lookup], -support[false_lookup],
                              -purity[false_lookup], -call[false_lookup]))
    selected.update(int(index) for index in false_lookup[lookup_order[:hard_budget]])

    streams = []
    for name, ascending in _HARD_VIEWS:
        if name not in features:
            continue
        values = _values(features, name)[negatives]
        valid = np.isfinite(values)
        # Missing values are not evidence of a hard negative, and a constant
        # score carries no preference. The fallback still fills the total quota.
        if not valid.any() or (valid.all() and np.ptp(values) == 0):
            continue
        eligible = negatives[valid]
        scores = values[valid] if ascending else -values[valid]
        streams.append(eligible[np.lexsort((ids[eligible], scores))])
    pointers = [0] * len(streams)
    while len(selected) < hard_budget:
        advanced = False
        for stream_index, stream in enumerate(streams):
            pointer = pointers[stream_index]
            while pointer < len(stream) and int(stream[pointer]) in selected:
                pointer += 1
            if pointer < len(stream):
                selected.add(int(stream[pointer]))
                pointer += 1
                advanced = True
            pointers[stream_index] = pointer
            if len(selected) == hard_budget:
                break
        if not advanced:
            break
    if len(selected) < hard_budget:
        ordered = negatives[np.argsort(ids[negatives], kind="stable")]
        for position in ordered:
            selected.add(int(position))
            if len(selected) == hard_budget:
                break

    remaining = np.array([int(position) for position in negatives
                          if int(position) not in selected], dtype=np.int64)
    remaining = remaining[np.argsort(ids[remaining], kind="stable")]
    digest = hashlib.sha256(f"{config.seed}\0{query_id}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:16], "big"))
    random_count = min(config.random_negatives, len(remaining))
    if random_count:
        selected.update(int(position) for position in
                        rng.choice(remaining, size=random_count, replace=False))
    selected.add(int(positive[0]))
    return np.array(sorted(selected), dtype=np.int64)
