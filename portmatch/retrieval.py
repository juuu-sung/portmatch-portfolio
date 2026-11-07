"""Independent, high-recall candidate retrieval for one port arrival at a time.

Only the supplied catalog and TrainingMemory are indexed. In validation, build a
new memory and retriever from the training partition for each split. No query
batch is used to build an index, learn aliases, or select retrieval parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import has_hangul, normalize_name, phonetic_key, romanize_name

if TYPE_CHECKING:
    from .memory import TrainingMemory


@dataclass(frozen=True)
class RetrievalConfig:
    name_top_k: int = 80
    phonetic_top_k: int = 80
    edit_top_k: int = 40
    gt_relative_window: float = 0.10
    numeric_fallback_k: int = 50
    # None deliberately preserves the complete tonnage window, often thousands
    # of candidates. A cap is an explicit recall/runtime tradeoff, not a default.
    max_total: int | None = None

    def __post_init__(self) -> None:
        for name in ("name_top_k", "phonetic_top_k", "edit_top_k"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.numeric_fallback_k < 1:
            raise ValueError("numeric_fallback_k must be positive")
        if not 0 <= self.gt_relative_window < 1:
            raise ValueError("gt_relative_window must be in [0, 1)")
        if self.max_total is not None and self.max_total < 1:
            raise ValueError("max_total must be positive or None")


RESULT_COLUMNS = [
    "catalog_index", "call_match", "alias_match", "lookup_support",
    "lookup_purity", "exact_hit", "name_hit", "edit_hit", "phonetic_hit",
    "tonnage_hit", "name_score", "name_rank", "edit_score", "edit_rank",
    "phonetic_score", "phonetic_rank", "tonnage_rank", "channel_count",
]


def _has_letter(value: str) -> bool:
    """Do not search noisy grams from blank or purely numeric vessel names."""
    return bool(value) and any(char.isalpha() for char in value)


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


class _TextIndex:
    """Sparse document search; take maximum alias similarity for each vessel."""

    def __init__(self, documents: list[tuple[str, int]], size: int,
                 tie_keys: np.ndarray) -> None:
        self.size = size
        self.tie_keys = tie_keys
        documents = sorted({(name, index) for name, index in documents
                            if _has_letter(name)})
        self.owner = np.array([index for _, index in documents], dtype=np.int64)
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if not documents:
            return
        texts = [name for name, _ in documents]
        # Single-character catalogs are unusual but must remain valid inputs.
        grams = (2, 4) if any(len(name) > 1 for name in texts) else (1, 3)
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=grams, lowercase=False,
            sublinear_tf=True, norm="l2", dtype=np.float32,
        )
        self.matrix = self.vectorizer.fit_transform(texts).T.tocsr()

    def search(self, queries: list[str], top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = sorted({query for query in queries if _has_letter(query)})
        if not queries or top_k == 0 or self.vectorizer is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        similarities = (self.vectorizer.transform(queries) @ self.matrix).tocsr()
        scores = np.zeros(self.size, dtype=np.float32)
        np.maximum.at(scores, self.owner[similarities.indices], similarities.data)
        candidates = np.flatnonzero(scores > 0)
        order = np.lexsort((candidates, self.tie_keys[candidates], -scores[candidates]))
        candidates = candidates[order[:top_k]]
        return candidates, np.minimum(scores[candidates], 1.0)


class CandidateRetriever:
    """Union exact lookup, spelling, phonetic, and broad tonnage candidates.

    The index uses canonical catalog names plus the aliases in ``memory``.
    ``memory`` must have been fitted exclusively on the current training split.
    Missing channel ranks are 0; present ranks start at 1. Similarities use [0,1].
    Results are returned in canonical catalog-index order, independent of query
    execution order. The optional cap always preserves exact lookup candidates.
    """

    def __init__(self, catalog: pd.DataFrame, memory: TrainingMemory,
                 config: RetrievalConfig | None = None) -> None:
        self.config = config or RetrievalConfig()
        self.catalog = catalog.reset_index(drop=True).copy()
        required = {"vessel_id", "vessel_name", "gross_tonnage"}
        missing = required.difference(self.catalog.columns)
        if missing:
            raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
        if self.catalog.empty:
            raise ValueError("Catalog must contain at least one vessel")
        if self.catalog["vessel_id"].isna().any() or self.catalog["vessel_id"].duplicated().any():
            raise ValueError("Catalog vessel_id values must be present and unique")
        self.memory = memory
        self.size = len(self.catalog)
        self.tie_keys = self.catalog["vessel_id"].astype(str).to_numpy()
        self._stable_order = np.argsort(self.tie_keys, kind="stable")

        documents = [(name, index) for index, name in
                     enumerate(self.catalog["vessel_name"].tolist())]
        aliases = memory.aliases
        if not aliases.empty:
            for alias in aliases[["vessel_name", "catalog_index"]].itertuples(index=False):
                index = int(alias.catalog_index)
                if index < 0 or index >= self.size:
                    raise ValueError("Training alias refers to an invalid catalog_index")
                documents.append((alias.vessel_name, index))

        self._exact: dict[str, list[int]] = {}
        for index, value in enumerate(self.catalog["vessel_name"]):
            name = normalize_name(value)
            if name:
                self._exact.setdefault(name, []).append(index)
        normalized = [(normalize_name(name), index) for name, index in documents]
        self._name_index = _TextIndex(normalized, self.size, self.tie_keys)
        phonetic_documents = []
        for name, index in documents:
            phonetic_documents.append((romanize_name(name), index))
            phonetic_documents.append((phonetic_key(name), index))
        self._phonetic_index = _TextIndex(phonetic_documents, self.size, self.tie_keys)

        # A separate edit-distance search recovers short/typo names that do not
        # share enough character grams. Names and ties are ordered deterministically.
        self._edit_owners: dict[str, list[int]] = {}
        for name, index in sorted(set(normalized)):
            if _has_letter(name) and not has_hangul(name):
                self._edit_owners.setdefault(name, []).append(index)
        self._edit_names = sorted(
            self._edit_owners,
            key=lambda name: (min(self.tie_keys[self._edit_owners[name]]), name),
        )

        self._gt = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)
        valid = np.flatnonzero(np.isfinite(self._gt) & (self._gt > 0))
        gt_order = np.lexsort((valid, self.tie_keys[valid], self._gt[valid]))
        self._gt_indices = valid[gt_order]
        self._sorted_gt = self._gt[self._gt_indices]

    def _edit_search(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        if (not _has_letter(query) or has_hangul(query) or
                self.config.edit_top_k == 0 or not self._edit_names):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        matches = process.extract(
            query, self._edit_names, scorer=Levenshtein.normalized_similarity,
            limit=self.config.edit_top_k, score_cutoff=0.01,
        )
        scores: dict[int, float] = {}
        for name, score, _ in matches:
            for index in self._edit_owners[name]:
                scores[index] = max(scores.get(index, 0.0), score)
        ids = np.array(list(scores), dtype=np.int64)
        values = np.array([scores[index] for index in ids], dtype=np.float32)
        order = np.lexsort((ids, self.tie_keys[ids], -values))
        return ids[order], values[order]

    def _window(self, query_gt: float) -> np.ndarray:
        tolerance = self.config.gt_relative_window
        # Relative error uses the catalog tonnage as denominator, consistent
        # with the data audit: abs(query - catalog) / catalog <= tolerance.
        lower = np.nextafter(query_gt / (1 + tolerance), -np.inf)
        upper = np.nextafter(query_gt / (1 - tolerance), np.inf)
        left = np.searchsorted(self._sorted_gt, lower, side="left")
        right = np.searchsorted(self._sorted_gt, upper, side="right")
        return self._gt_indices[left:right]

    def _numeric_search(self, row: Mapping[str, Any]) -> np.ndarray:
        international = _positive_number(row.get("international_gross_tonnage"))
        gross = _positive_number(row.get("gross_tonnage"))
        query_gt = international if international is not None else gross
        if query_gt is None or not len(self._gt_indices):
            return np.empty(0, dtype=np.int64)
        candidates = self._window(query_gt)
        if not len(candidates) and gross is not None and gross != query_gt:
            candidates = self._window(gross)
            if len(candidates):
                query_gt = gross
        if not len(candidates):
            candidates = self._gt_indices
            order = np.lexsort((candidates, self.tie_keys[candidates],
                                np.abs(self._gt[candidates] - query_gt)))
            return candidates[order[:self.config.numeric_fallback_k]]
        order = np.lexsort((candidates, self.tie_keys[candidates],
                            np.abs(self._gt[candidates] - query_gt)))
        return candidates[order]

    def retrieve(self, row: Mapping[str, Any]) -> pd.DataFrame:
        query = normalize_name(row.get("vessel_name"))
        lookup = self.memory.lookup(row)
        exact_ids = np.array(self._exact.get(query, []), dtype=np.int64)
        name_ids, name_scores = self._name_index.search([query], self.config.name_top_k)
        edit_ids, edit_scores = self._edit_search(query)
        phonetic_ids, phonetic_scores = self._phonetic_index.search(
            [romanize_name(row.get("vessel_name")), phonetic_key(row.get("vessel_name"))],
            self.config.phonetic_top_k,
        )
        tonnage_ids = self._numeric_search(row)
        lookup_ids = lookup["catalog_index"].to_numpy(dtype=np.int64)
        if np.any((lookup_ids < 0) | (lookup_ids >= self.size)):
            raise ValueError("Training lookup refers to an invalid catalog_index")
        ids = np.unique(np.concatenate([
            lookup_ids, exact_ids, name_ids, edit_ids, phonetic_ids, tonnage_ids,
        ]))
        if not len(ids):
            ids = np.sort(self._stable_order[:self.config.numeric_fallback_k])
        result = pd.DataFrame(0.0, index=pd.Index(ids, name="catalog_index"),
                              columns=RESULT_COLUMNS[1:])
        if len(lookup):
            columns = ["call_match", "alias_match", "lookup_support", "lookup_purity"]
            result.loc[lookup_ids, columns] = lookup[columns].to_numpy(dtype=float)
        result.loc[exact_ids, "exact_hit"] = 1
        for prefix, selected, scores in [
            ("name", name_ids, name_scores), ("edit", edit_ids, edit_scores),
            ("phonetic", phonetic_ids, phonetic_scores),
        ]:
            result.loc[selected, f"{prefix}_hit"] = 1
            result.loc[selected, f"{prefix}_score"] = scores
            result.loc[selected, f"{prefix}_rank"] = np.arange(1, len(selected) + 1)
        result.loc[tonnage_ids, "tonnage_hit"] = 1
        result.loc[tonnage_ids, "tonnage_rank"] = np.arange(1, len(tonnage_ids) + 1)
        flags = ["call_match", "alias_match", "exact_hit", "name_hit", "edit_hit",
                 "phonetic_hit", "tonnage_hit"]
        result["channel_count"] = result[flags].sum(axis=1)
        if self.config.max_total is not None and len(result) > self.config.max_total:
            # Exact sources must survive a caller-requested cap, even if their
            # count alone exceeds max_total. The default never performs pruning.
            forced = result.index[result[["call_match", "alias_match", "exact_hit"]].any(axis=1)]
            remaining = result.drop(index=forced).copy()
            remaining["_score"] = remaining[["name_score", "edit_score", "phonetic_score"]].max(axis=1)
            remaining["_tonnage"] = remaining["tonnage_rank"].replace(0, np.inf)
            remaining["_id"] = self.tie_keys[remaining.index.to_numpy(dtype=int)]
            remaining = remaining.sort_values(
                ["channel_count", "_score", "_tonnage", "_id"],
                ascending=[False, False, True, True], kind="stable",
            )
            limit = max(0, self.config.max_total - len(forced))
            result = result.loc[np.sort(np.concatenate([forced.to_numpy(), remaining.index[:limit].to_numpy()]))]
        integer_columns = flags + ["name_rank", "edit_rank", "phonetic_rank",
                                   "tonnage_rank", "channel_count", "lookup_support"]
        result[integer_columns] = result[integer_columns].astype(np.int64)
        return result.reset_index()[RESULT_COLUMNS]
