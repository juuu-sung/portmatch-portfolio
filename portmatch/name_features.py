"""Additive, label-independent token and transliteration comparisons.

These are spelling views, not a translation/pronunciation dictionary. Every rule
is fixed and applies to all names. Only the raw query name and fixed catalog names
enter computation: aliases, target IDs and category/validation statistics do not.
Legacy text/features remain unchanged, including their vowel-bearing romanized
comparison and number-conflict features. Digits survive all added views.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
import unicodedata

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from unidecode import unidecode


FEATURE_NAMES = (
    "name_token_count_ratio",
    "name_token_roman_alignment",
    "name_token_context_alignment",
    "name_token_context_weakest",
    "name_token_context_first",
    "name_token_context_last",
    "name_token_residual_context_ratio",
    "name_token_residual_vowel_ratio",
    "name_token_residual_present",
    "name_context_ratio",
    "name_context_edit_similarity",
    "name_context_variant_ratio",
    "name_vowel_fold_ratio",
    "name_vowel_fold_edit_similarity",
    "name_suffix_variant_context_ratio",
    "name_suffix_variant_vowel_ratio",
    "name_suffix_variant_present",
)

# Syllable decomposition retains Korean final consonants which plain Unidecode
# often renders differently (e.g. syllable-final ㅅ is a stopped T, not initial S).
_INITIAL = ("G", "GG", "N", "D", "DD", "R", "M", "B", "BB", "S", "SS", "",
            "J", "JJ", "CH", "K", "T", "P", "H")
_VOWEL = ("A", "AE", "YA", "YAE", "EO", "E", "YEO", "YE", "O", "WA", "WAE",
          "OE", "YO", "U", "WEO", "WE", "WI", "YU", "EU", "YI", "I")
_FINAL = ("", "K", "K", "K", "N", "N", "N", "T", "L", "K", "M", "P", "L",
          "L", "P", "L", "M", "P", "P", "T", "T", "NG", "T", "T", "K", "T", "P", "T")
_TOKENS = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)
_CONSONANT_FOLD = str.maketrans({"F": "P", "V": "B", "Z": "S", "R": "L",
                               "D": "T", "G": "K", "Q": "K"})


def _clean(value: object) -> str:
    return unicodedata.normalize("NFKC", value).upper() if isinstance(value, str) else ""


def _latin(text: str) -> str:
    return "".join(c for c in unidecode(text).upper() if c.isascii() and c.isalnum())


def _syllable_roman(text: str) -> str:
    parts = []
    for char in text:
        code = ord(char) - 0xAC00
        if 0 <= code < 11172:
            initial, remainder = divmod(code, 588)
            vowel, final = divmod(remainder, 28)
            parts.append(_INITIAL[initial] + _VOWEL[vowel] + _FINAL[final])
        else:
            parts.append(_latin(char))
    return "".join(parts)


def _spelling_fold(roman: str, hard_ch: bool = False) -> str:
    """Conservative shared spelling folds, with CH ambiguity kept as a variant.

    Soft C before E/I/Y and affricate CH are generic alternatives to the old
    all-C-to-K key. This cannot model all English pronunciations. NG is protected
    from G/K folding. The original feature family is retained by the caller.
    """
    text = roman.replace("WOO", "U").replace("WU", "U")
    text = text.replace("PH", "F").replace("CK", "K").replace("QU", "KW")
    text = text.replace("CH", "K" if hard_ch else "J").replace("SH", "S")
    text = re.sub(r"C(?=[EIY])", "S", text).replace("C", "K")
    text = text.replace("NG", "~").translate(_CONSONANT_FOLD).replace("~", "NG")
    # These are additive vowel-bearing spelling alternatives, not replacements
    # for the original Romanization. Terminal EU may be an epenthetic Korean vowel.
    text = text.replace("OO", "U").replace("EE", "I").replace("EA", "I")
    text = text.replace("EO", "O").replace("AE", "E")
    text = re.sub(r"(?<=[A-Z])EU$", "", text)
    text = re.sub(r"(?<=[A-Z])E$", "", text)
    return re.sub(r"([A-Z])\1+", r"\1", text)


def contextual_key(value: object, hard_ch: bool = False) -> str:
    """Expose the exact whole-name consonant view used by pair features."""
    _, context, hard, _ = _parts(_clean(value))
    return "".join(hard if hard_ch else context)


def _ratio(left: str, right: str) -> float:
    return fuzz.ratio(left, right) / 100.0 if left and right else 0.0


def _edit(left: str, right: str) -> float:
    return Levenshtein.normalized_similarity(left, right) if left and right else 0.0


def _token_ratio(left: str, right: str) -> float:
    # Token matching never treats vessel number 1 as a partial match for 10.
    if left.isdigit() or right.isdigit():
        return float(bool(left) and left == right)
    return _ratio(left, right)


def _alignment(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[float, float]:
    """Order-preserving one-to-one soft alignment; unmatched tokens score zero.

    DP maximizes summed similarities, then weakest-token score on exact ties.
    Duplicate tokens cannot reuse the same opposite token. The denominator is
    max token count, so neither extra tokens nor numeric suffixes disappear.
    """
    if not left or not right:
        return 0.0, 0.0
    previous = [(0.0, 0.0)] * (len(right) + 1)
    previous[0] = (0.0, 1.0)
    for i, a in enumerate(left, 1):
        current = [(0.0, 0.0)]
        for j, b in enumerate(right, 1):
            similarity = _token_ratio(a, b)
            score, weakest = previous[j - 1]
            match = (score + similarity, min(weakest, similarity))
            skip_left = (previous[j][0], 0.0)
            skip_right = (current[j - 1][0], 0.0)
            current.append(max(match, skip_left, skip_right))
        previous = current
    score, weakest = previous[-1]
    return score / max(len(left), len(right)), weakest


@dataclass(frozen=True, slots=True)
class _View:
    roman: tuple[str, ...]
    context: tuple[str, ...]
    hard_context: tuple[str, ...]
    vowel: tuple[str, ...]
    context_joined: str
    hard_joined: str
    vowel_joined: str
    suffix_context: str
    suffix_vowel: str
    suffix_present: bool


def _parts(raw: str):
    tokens = _TOKENS.findall(raw)
    roman = tuple(_latin(token) for token in tokens)
    phonetic_source = tuple(_syllable_roman(token) for token in tokens)
    vowel = tuple(_spelling_fold(token) for token in phonetic_source)
    context = tuple(re.sub(r"([A-Z])\1+", r"\1", re.sub(r"[AEIOUY]", "", token))
                    for token in vowel)
    hard = tuple(re.sub(r"([A-Z])\1+", r"\1", re.sub(r"[AEIOUY]", "", _spelling_fold(token, True)))
                 for token in phonetic_source)
    return roman, context, hard, vowel


@lru_cache(maxsize=20_000)
def _view(raw: str) -> _View:
    roman, context, hard, vowel = _parts(raw)
    # A terminal Korean 호 is only an alternate view when at least two preceding
    # Hangul syllables exist. Short words such as 산호 and the original name remain.
    trimmed = raw.rstrip()
    suffix = trimmed.endswith("호") and len(re.findall(r"[가-힣]", trimmed[:-1])) >= 2
    if suffix:
        _, suffix_context, _, suffix_vowel = _parts(trimmed[:-1].rstrip())
        suffix_context, suffix_vowel = "".join(suffix_context), "".join(suffix_vowel)
    else:
        suffix_context = suffix_vowel = ""
    return _View(roman, context, hard, vowel, "".join(context), "".join(hard),
                 "".join(vowel), suffix_context, suffix_vowel, suffix)


def _residual(left: _View, right: _View) -> tuple[float, float, float]:
    """Compare remaining words after strong common boundary words are removed.

    No corpus-derived stop list: a boundary is shared only within this pair.
    Keep at least one token on each side. A flag distinguishes unavailable
    residual evidence from an observed mismatch.
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
    return (_ratio("".join(left.context[li:lj]), "".join(right.context[ri:rj])),
            _ratio("".join(left.vowel[li:lj]), "".join(right.vowel[ri:rj])), 1.0)


def _pair(left: _View, right: _View) -> tuple[float, ...]:
    if not left.roman or not right.roman:
        return (0.0,) * len(FEATURE_NAMES)
    context_score, weakest = _alignment(left.context, right.context)
    residual = _residual(left, right)
    suffix = left.suffix_present or right.suffix_present
    suffix_context = suffix_vowel = 0.0
    if suffix:
        a = (left.context_joined, left.suffix_context) if left.suffix_present else (left.context_joined,)
        b = (right.context_joined, right.suffix_context) if right.suffix_present else (right.context_joined,)
        suffix_context = max(_ratio(x, y) for x in a for y in b)
        a = (left.vowel_joined, left.suffix_vowel) if left.suffix_present else (left.vowel_joined,)
        b = (right.vowel_joined, right.suffix_vowel) if right.suffix_present else (right.vowel_joined,)
        suffix_vowel = max(_ratio(x, y) for x in a for y in b)
    return (
        min(len(left.roman), len(right.roman)) / max(len(left.roman), len(right.roman)),
        _alignment(left.roman, right.roman)[0],
        context_score, weakest,
        _token_ratio(left.context[0], right.context[0]),
        _token_ratio(left.context[-1], right.context[-1]),
        *residual,
        _ratio(left.context_joined, right.context_joined),
        _edit(left.context_joined, right.context_joined),
        max(_ratio(a, b) for a in (left.context_joined, left.hard_joined)
            for b in (right.context_joined, right.hard_joined)),
        _ratio(left.vowel_joined, right.vowel_joined),
        _edit(left.vowel_joined, right.vowel_joined),
        suffix_context, suffix_vowel, float(suffix),
    )


class NameFeatureComputer:
    """Precompute fixed catalog views; candidate subsets and order cannot fit it.

    The bounded string-view cache only memoizes deterministic transformations.
    No pair cache or query-wide matrix is retained, limiting memory to catalog
    views, at most 20,000 cached names and the requested N-by-17 output.
    """

    def __init__(self, catalog_names):
        self.views = tuple(_view(_clean(name)) for name in catalog_names)

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
        query = _view(_clean(query_name))
        for position, index in enumerate(indices):
            result[position] = _pair(query, self.views[index])
        return result
