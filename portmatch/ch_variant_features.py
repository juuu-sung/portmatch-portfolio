"""Additive CH pronunciation views for token and residual comparisons.

The v1 token feature family remains unchanged. This family carries its two
existing CH alternatives (J and K) through the entire token comparison. Each
name has at most two coherent views: all CH occurrences use the same rule in
that view. We do not independently choose a pronunciation for each token or
maximize each output feature using different views.

Only the raw query and fixed catalog names enter computation. No labels,
aliases, candidate statistics, or ship-specific pronunciation rules are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

import numpy as np

from .name_features import (
    _TOKENS, _alignment, _clean, _ratio, _spelling_fold, _syllable_roman,
    _token_ratio,
)


FEATURE_NAMES = (
    "name_ch_token_context_alignment",
    "name_ch_token_context_weakest",
    "name_ch_token_context_first",
    "name_ch_token_context_last",
    "name_ch_token_residual_context_ratio",
    "name_ch_token_residual_vowel_ratio",
    "name_ch_token_residual_present",
)


@dataclass(frozen=True, slots=True)
class _Pronunciation:
    context: tuple[str, ...]
    vowel: tuple[str, ...]
    hard_ch: bool


@lru_cache(maxsize=20_000)
def _variants(raw: str) -> tuple[_Pronunciation, ...]:
    source = tuple(_syllable_roman(token) for token in _TOKENS.findall(raw))
    if not source:
        return ()
    views = []
    for hard_ch in (False, True):
        vowel = tuple(_spelling_fold(token, hard_ch=hard_ch) for token in source)
        context = tuple(re.sub(r"([A-Z])\1+", r"\1", re.sub(r"[AEIOUY]", "", token))
                        for token in vowel)
        if views and (context, vowel) == (views[0].context, views[0].vowel):
            continue
        views.append(_Pronunciation(context, vowel, hard_ch))
    return tuple(views)


def _residual(left: _Pronunciation, right: _Pronunciation) -> tuple[float, float, float]:
    """Use one pronunciation and one set of boundaries for both residuals.

    The removal rule and joined-string ratios are identical to v1. In
    particular, the boundary matcher retains exact numeric-token matching and
    at least one token on each side. Presence is computed for the selected
    pronunciation; a different pronunciation cannot supply its flag or slices.
    """
    li = ri = 0
    lj, rj = len(left.context), len(right.context)
    removed = False
    while lj - li > 1 and rj - ri > 1:
        if _token_ratio(left.context[li], right.context[ri]) < .8:
            break
        li += 1
        ri += 1
        removed = True
    while lj - li > 1 and rj - ri > 1:
        if _token_ratio(left.context[lj - 1], right.context[rj - 1]) < .8:
            break
        lj -= 1
        rj -= 1
        removed = True
    if not removed:
        return 0.0, 0.0, 0.0
    return (
        _ratio("".join(left.context[li:lj]), "".join(right.context[ri:rj])),
        _ratio("".join(left.vowel[li:lj]), "".join(right.vowel[ri:rj])),
        1.0,
    )


def _pair(left: tuple[_Pronunciation, ...], right: tuple[_Pronunciation, ...]) -> tuple[float, ...]:
    if not left or not right:
        return (0.0,) * len(FEATURE_NAMES)
    if len(left) == len(right) == 1:
        a, b = left[0], right[0]
        selected = a, b, *_alignment(a.context, b.context)
    else:
        selected = None
        selected_key = None
        for a in left:
            for b in right:
                alignment, weakest = _alignment(a.context, b.context)
                if selected_key is not None and (alignment, weakest) < selected_key[:2]:
                    continue
                # Choose one complete pronunciation pair. Context alignment is
                # the objective; weakest-token and vowel alignment resolve ties.
                # Final spelling-only tie breakers prefer fewer CH->K variants
                # and are symmetric when the two names are exchanged.
                key = (
                    alignment, weakest, *_alignment(a.vowel, b.vowel),
                    -int(a.hard_ch) - int(b.hard_ch),
                    tuple(sorted(((a.context, a.vowel), (b.context, b.vowel)))),
                )
                if selected_key is None or key > selected_key:
                    selected_key = key
                    selected = a, b, alignment, weakest
    assert selected is not None
    a, b, alignment, weakest = selected
    return (
        alignment, weakest,
        _token_ratio(a.context[0], b.context[0]),
        _token_ratio(a.context[-1], b.context[-1]),
        *_residual(a, b),
    )


class ChVariantFeatureComputer:
    """Compute seven fixed-rule features without changing the v1 feature view.

    Catalog order, candidate subset, repeated calls, and unrelated queries do
    not fit or alter this computer. The string-view cache is bounded, and only
    the requested N-by-7 matrix is allocated for a query.
    """

    def __init__(self, catalog_names):
        self.views = tuple(_variants(_clean(name)) for name in catalog_names)

    def compute(self, query_name: object, catalog_indices) -> np.ndarray:
        raw_indices = np.asarray(catalog_indices)
        if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iuf":
            raise ValueError("catalog_indices must be a one-dimensional numeric array")
        if (not np.isfinite(raw_indices).all()
                or not np.equal(raw_indices, np.floor(raw_indices)).all()
                or np.any(raw_indices < 0) or np.any(raw_indices >= len(self.views))):
            raise ValueError("catalog_indices must contain valid finite integer positions")
        indices = raw_indices.astype(np.int64)
        result = np.empty((len(indices), len(FEATURE_NAMES)), dtype=np.float32)
        query = _variants(_clean(query_name))
        for position, index in enumerate(indices):
            result[position] = _pair(query, self.views[index])
        return result
