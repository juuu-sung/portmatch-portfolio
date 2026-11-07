"""Catalog-only typo-tolerant search; existing candidates are immutable."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import OSA
from sklearn.feature_extraction.text import TfidfVectorizer

from .robust_retrieval import _name_views
from .retrieval import RESULT_COLUMNS, _has_letter
from .text import has_hangul


@dataclass(frozen=True)
class ExpansionConfig:
    bag_top_k: int = 512
    edit_top_k: int = 256
    max_new: int = 256

    def __post_init__(self):
        for value in (self.bag_top_k, self.edit_top_k, self.max_new):
            if type(value) is not int or value < 0:
                raise ValueError('Search limits must be nonnegative integers')


class HangulExpansion:
    """Fit only fixed canonical catalog names; no training or query batch input."""
    def __init__(self, catalog, config=None):
        self.config = config or ExpansionConfig()
        self.size = len(catalog)
        self.tie_keys = catalog.vessel_id.astype(str).to_numpy()
        if len(set(self.tie_keys)) != self.size:
            raise ValueError('Duplicate catalog vessel ids')
        documents = set()
        for owner, name in enumerate(catalog.vessel_name):
            contexts, vowels = _name_views(name)
            documents.update((text, owner) for text in contexts + vowels if _has_letter(text))
        documents = sorted(documents)
        self.texts = [t for t, _ in documents]
        self.owners = np.array([o for _, o in documents], dtype=np.int64)
        self.vectorizer, self.matrix = None, None
        if documents:
            self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2),
                lowercase=False, sublinear_tf=True, norm='l2', dtype=np.float32)
            self.matrix = self.vectorizer.fit_transform(self.texts).T.tocsr()

    def _top(self, scores, k):
        ids = np.flatnonzero(scores > 0)
        return ids[np.lexsort((ids, self.tie_keys[ids], -scores[ids]))[:k]]

    def search(self, name):
        if not has_hangul(name) or not self.texts or self.config.max_new == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        contexts, vowels = _name_views(name)
        queries = sorted(set(contexts + vowels))
        if not queries:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        bag, edit = np.zeros(self.size, np.float32), np.zeros(self.size, np.float32)
        if self.config.bag_top_k:
            similarities = (self.vectorizer.transform(queries) @ self.matrix).tocsr()
            np.maximum.at(bag, self.owners[similarities.indices], similarities.data)
        if self.config.edit_top_k:
            similarities = process.cdist(queries, self.texts, scorer=OSA.normalized_similarity,
                                          dtype=np.float32, workers=1)
            np.maximum.at(edit, self.owners, similarities.max(axis=0))
        ids = np.union1d(self._top(bag, self.config.bag_top_k), self._top(edit, self.config.edit_top_k))
        scores = np.minimum(np.maximum(bag[ids], edit[ids]), 1.)
        order = np.lexsort((ids, self.tie_keys[ids], -scores))
        return ids[order], scores[order]

    def append(self, row, old):
        """Preserve every common row, even its old rank and channel count."""
        if not has_hangul(row.get('vessel_name')):
            return old.copy()
        ids, scores = self.search(row.get('vessel_name'))
        keep = ~np.isin(ids, old.catalog_index.to_numpy())
        ids, scores = ids[keep][:self.config.max_new], scores[keep][:self.config.max_new]
        if not len(ids):
            return old.copy()
        new = pd.DataFrame(0, index=range(len(ids)), columns=RESULT_COLUMNS)
        new['catalog_index'] = ids
        new['phonetic_hit'] = new['channel_count'] = 1
        new['phonetic_score'] = scores
        new['phonetic_rank'] = (int(old.phonetic_rank.max()) if len(old) else 0) + np.arange(1, len(ids)+1)
        merged = pd.concat([old, new], ignore_index=True).sort_values('catalog_index').reset_index(drop=True)
        assert merged.catalog_index.is_unique
        common = merged.set_index('catalog_index').loc[old.catalog_index].reset_index()
        assert np.array_equal(common.to_numpy(), old[RESULT_COLUMNS].to_numpy())
        return merged[RESULT_COLUMNS]
