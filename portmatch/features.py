"""Row-independent comparison features for a query and its vessel candidates.

Category tables must be fitted on the training partition only. This module never
fits on transform, reads other evaluation rows, or compares GT directly to DWT.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein

from .text import normalize_name, romanize_name, phonetic_key, name_numbers, has_hangul


CATEGORY_FIELDS = {"flag_name": "flag", "vessel_type": "type"}
RETRIEVAL_COLUMNS = (
    "call_match", "alias_match", "lookup_support", "lookup_purity",
    "name_score", "name_rank", "phonetic_score", "phonetic_rank",
    "tonnage_rank", "channel_count", "exact_hit", "name_hit", "phonetic_hit", "tonnage_hit",
    "edit_hit", "edit_score", "edit_rank",
)


def _category(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).casefold().split())


def _positive(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) and number > 0 else float("nan")


class CategoryCompatibility:
    """Smoothed Korean/English category compatibility learned from labeled pairs.

    Repeated training records contribute counts. Unknown/missing query categories
    back off to the training target prior and have zero log lift. They are not
    treated as evidence that a candidate is incompatible.
    """

    def __init__(self, smoothing: float = 5.0) -> None:
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be positive and finite")
        self.smoothing = float(smoothing)
        self.tables: dict[str, dict[str, Any]] = {}

    def fit(self, train: pd.DataFrame, catalog: pd.DataFrame) -> "CategoryCompatibility":
        required_train = {"target_vessel_id", *CATEGORY_FIELDS}
        required_catalog = {"vessel_id", *CATEGORY_FIELDS}
        if not required_train.issubset(train.columns):
            raise ValueError(f"Training columns missing: {sorted(required_train - set(train.columns))}")
        if not required_catalog.issubset(catalog.columns):
            raise ValueError(f"Catalog columns missing: {sorted(required_catalog - set(catalog.columns))}")
        if catalog.vessel_id.isna().any() or not catalog.vessel_id.is_unique:
            raise ValueError("Catalog vessel_id must be non-null and unique")
        if train.target_vessel_id.isna().any() or not train.target_vessel_id.isin(catalog.vessel_id).all():
            raise ValueError("All training targets must exist in the catalog")
        joined = train[list(required_train)].merge(
            catalog[list(required_catalog)],
            left_on="target_vessel_id", right_on="vessel_id",
            how="left", validate="many_to_one", suffixes=("_query", "_catalog"),
        )
        tables = {}
        for field in CATEGORY_FIELDS:
            universe = sorted({x for x in catalog[field].map(_category) if x})
            source_counts: Counter[str] = Counter()
            target_counts: Counter[str] = Counter()
            pairs: dict[str, Counter[str]] = defaultdict(Counter)
            for source, target in zip(joined[field + "_query"].map(_category),
                                      joined[field + "_catalog"].map(_category)):
                if target:
                    target_counts[target] += 1
                if source and target:
                    source_counts[source] += 1
                    pairs[source][target] += 1
            tables[field] = {
                "universe": universe,
                "target_counts": dict(target_counts),
                "source_counts": dict(source_counts),
                "pairs": {source: dict(counts) for source, counts in pairs.items()},
                "total": sum(target_counts.values()),
            }
        self.tables = tables
        return self

    def transform(self, row: dict, candidate_rows: pd.DataFrame) -> pd.DataFrame:
        if set(self.tables) != set(CATEGORY_FIELDS):
            raise ValueError("CategoryCompatibility must be fitted before transform")
        result: dict[str, np.ndarray] = {}
        n = len(candidate_rows)
        for field, prefix in CATEGORY_FIELDS.items():
            table = self.tables[field]
            source = _category(row.get(field))
            targets = candidate_rows[field].map(_category).tolist()
            support = table["source_counts"].get(source, 0)
            pair_counts = table["pairs"].get(source, {})
            prior_denom = table["total"] + max(len(table["universe"]), 1)
            prior = np.array([(table["target_counts"].get(target, 0) + 1) / prior_denom
                              for target in targets], dtype=float)
            joint = np.array([pair_counts.get(target, 0) for target in targets], dtype=float)
            probability = (joint + self.smoothing * prior) / (support + self.smoothing)
            missing = np.array([not target for target in targets], dtype=bool)
            # Missing candidate values carry no compatibility evidence.
            probability[missing] = prior[missing]
            lift_log = np.log(probability / prior)
            if not support:
                lift_log[:] = 0.0
            result.update({
                f"{prefix}_probability": probability,
                f"{prefix}_lift_log": lift_log,
                f"{prefix}_source_support_log": np.full(n, np.log1p(support)),
                f"{prefix}_pair_support_log": np.log1p(joint),
                f"{prefix}_source_seen": np.full(n, float(support > 0)),
                f"{prefix}_pair_seen": (joint > 0).astype(float),
                f"{prefix}_query_missing": np.full(n, float(not source)),
                f"{prefix}_candidate_missing": missing.astype(float),
            })
        return pd.DataFrame(result, index=candidate_rows.index).astype(np.float32)

    def to_dict(self) -> dict:
        return {"version": 1, "smoothing": self.smoothing, "tables": self.tables}

    @classmethod
    def from_dict(cls, state: dict) -> "CategoryCompatibility":
        if state.get("version") != 1:
            raise ValueError("Unsupported category model version")
        model = cls(state["smoothing"])
        if set(state.get("tables", {})) != set(CATEGORY_FIELDS):
            raise ValueError("Category model fields are incomplete")
        model.tables = state["tables"]
        return model


class PairFeatureBuilder:
    """Precompute catalog text once, then compare only one query at a time.

    `candidates` must contain `catalog_index` (zero-based positions in the catalog
    passed to this constructor). Other supported retrieval metadata are optional.
    The returned float32 DataFrame preserves candidate order and contains no IDs or
    labels. Missing numeric comparisons remain NaN and have explicit missing flags.
    """

    def __init__(self, catalog: pd.DataFrame, category_model: CategoryCompatibility) -> None:
        required = {"vessel_id", "vessel_name", "gross_tonnage", *CATEGORY_FIELDS}
        if not required.issubset(catalog.columns):
            raise ValueError(f"Catalog columns missing: {sorted(required - set(catalog.columns))}")
        if catalog.vessel_id.isna().any() or not catalog.vessel_id.is_unique:
            raise ValueError("Catalog vessel_id must be non-null and unique")
        self.catalog = catalog.reset_index(drop=True).copy()
        self.category_model = category_model
        self.names = [normalize_name(x) for x in self.catalog.vessel_name]
        self.romanized = [romanize_name(x) for x in self.catalog.vessel_name]
        self.phonetic = [phonetic_key(x) for x in self.catalog.vessel_name]
        self.numbers = [tuple(name_numbers(x)) for x in self.catalog.vessel_name]
        self.gross = np.array([_positive(x) for x in self.catalog.gross_tonnage])

    @staticmethod
    def _ratio(query: str, choices: list[str], scorer=fuzz.ratio) -> np.ndarray:
        if not query:
            return np.zeros(len(choices), dtype=float)
        return np.array([scorer(query, choice) / 100 if choice else 0.0 for choice in choices])

    def transform(self, row: dict, candidates: pd.DataFrame) -> pd.DataFrame:
        if "catalog_index" not in candidates:
            raise ValueError("Candidates must include catalog_index")
        positions = pd.to_numeric(candidates.catalog_index, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(positions).all() or not np.equal(positions, np.floor(positions)).all():
            raise ValueError("catalog_index must contain finite integer positions")
        if (positions < 0).any() or (positions >= len(self.catalog)).any():
            raise ValueError("catalog_index is outside the catalog")
        indices = positions.astype(np.int64)
        n = len(indices)
        names = [self.names[i] for i in indices]
        romanized = [self.romanized[i] for i in indices]
        phonetic = [self.phonetic[i] for i in indices]
        numbers = [self.numbers[i] for i in indices]
        query = normalize_name(row.get("vessel_name"))
        roman = romanize_name(row.get("vessel_name"))
        sound = phonetic_key(row.get("vessel_name"))
        q_numbers = tuple(name_numbers(row.get("vessel_name")))
        q_number_set = set(q_numbers)
        max_lengths = np.maximum([max(len(query), len(name)) for name in names], 1)
        prefix_lengths = []
        for name in names:
            count = 0
            for left, right in zip(query, name):
                if left != right:
                    break
                count += 1
            prefix_lengths.append(count)
        features: dict[str, np.ndarray] = {
            "name_exact": np.array([bool(query) and query == name for name in names], dtype=float),
            "name_ratio": self._ratio(query, names),
            "name_partial_ratio": self._ratio(query, names, fuzz.partial_ratio),
            "name_edit_similarity": np.array([
                DamerauLevenshtein.normalized_similarity(query, name) if query and name else 0.0
                for name in names]),
            "name_prefix_ratio": np.asarray(prefix_lengths) / max_lengths,
            "name_length_ratio": np.minimum([len(query)] * n, [len(name) for name in names]) / max_lengths,
            "romanized_exact": np.array([bool(roman) and roman == name for name in romanized], dtype=float),
            "romanized_ratio": self._ratio(roman, romanized),
            "phonetic_exact": np.array([bool(sound) and sound == name for name in phonetic], dtype=float),
            "phonetic_ratio": self._ratio(sound, phonetic),
            "query_has_hangul": np.full(n, float(has_hangul(row.get("vessel_name")))),
            "query_name_missing": np.full(n, float(not query)),
            "candidate_name_missing": np.array([not name for name in names], dtype=float),
            "numbers_both_present": np.array([bool(q_numbers) and bool(x) for x in numbers], dtype=float),
            "numbers_exact": np.array([bool(q_numbers) and q_numbers == x for x in numbers], dtype=float),
            "numbers_conflict": np.array([bool(q_numbers) and bool(x) and q_numbers != x for x in numbers], dtype=float),
            "numbers_jaccard": np.array([
                len(q_number_set & set(x)) / len(q_number_set | set(x)) if q_number_set | set(x) else 0.0
                for x in numbers]),
            "query_numbers_missing": np.full(n, float(not q_numbers)),
            "candidate_numbers_missing": np.array([not x for x in numbers], dtype=float),
        }
        gt = self.gross[indices]
        valid_gt = np.isfinite(gt)
        query_gross = _positive(row.get("gross_tonnage"))
        query_international = _positive(row.get("international_gross_tonnage"))
        relative_gaps = []
        for prefix, value in [("gross", query_gross), ("international", query_international)]:
            valid = valid_gt & np.isfinite(value)
            relative = np.full(n, np.nan)
            relative[valid] = np.abs(value - gt[valid]) / np.maximum(value, gt[valid])
            log_gap = np.full(n, np.nan)
            log_gap[valid] = np.abs(np.log1p(value) - np.log1p(gt[valid]))
            features[f"{prefix}_relative_gap"] = relative
            features[f"{prefix}_log_gap"] = log_gap
            features[f"{prefix}_exact"] = (valid & (value == gt)).astype(float)
            features[f"{prefix}_within_1pct"] = (valid & (relative <= .01)).astype(float)
            features[f"{prefix}_within_5pct"] = (valid & (relative <= .05)).astype(float)
            features[f"{prefix}_query_missing"] = np.full(n, float(not np.isfinite(value)))
            features[f"{prefix}_pair_valid"] = valid.astype(float)
            relative_gaps.append(relative)
        features["best_tonnage_gap"] = np.fmin(*relative_gaps)
        features["catalog_tonnage_missing"] = (~valid_gt).astype(float)
        both_query_valid = np.isfinite(query_gross) and np.isfinite(query_international)
        features["query_tonnages_both_valid"] = np.full(n, float(both_query_valid))
        features["query_tonnages_relative_gap"] = np.full(n,
            abs(query_gross - query_international) / max(query_gross, query_international)
            if both_query_valid else np.nan)
        # Missing channel ranks/scores are zero; real ranks are one-based.
        # No statistics depend on other rows in an evaluation file.
        for column in RETRIEVAL_COLUMNS:
            default = 0.0
            if column in candidates:
                values = pd.to_numeric(candidates[column], errors="coerce").to_numpy(dtype=float)
                values = np.where(np.isfinite(values), values, default)
            else:
                values = np.full(n, default)
            features["retrieval_" + column] = values
        result = pd.DataFrame(features, index=candidates.index).astype(np.float32)
        candidate_rows = self.catalog.iloc[indices].copy()
        candidate_rows.index = candidates.index
        categories = self.category_model.transform(row, candidate_rows)
        return pd.concat([result, categories], axis=1)
