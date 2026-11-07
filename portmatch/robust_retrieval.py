"""Versioned candidate expansion for noisy Hangul names and conflicting tonnage.

This optional retriever leaves the legacy implementation and feature schema
untouched. All text indexes use the supplied catalog plus aliases from the
supplied training partition. Query rows never fit or update an index.

For Hangul inputs, context, vowel-bearing and edit searches are merged into the
existing phonetic channel: similarity is the maximum across its searches and
rank is the deterministic descending-similarity order. This is a changed
feature *meaning*, despite the unchanged 70/87-column schema, so this module's
version/hash is recorded in caches and trained-model metadata.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from .features import CategoryCompatibility, PairFeatureBuilder
from .memory import TrainingMemory
from .name_features import _clean, _view
from .pipeline import MatchingPipeline
from .retrieval import (
    RESULT_COLUMNS, CandidateRetriever, RetrievalConfig, _TextIndex,
    _has_letter, _positive_number,
)
from .text import has_hangul


RETRIEVAL_NAME = "robust_retrieval_v1"
_EXTRA_FIELDS = {"profile", "context_top_k", "vowel_top_k", "context_edit_top_k"}
_FLAGS = ["call_match", "alias_match", "exact_hit", "name_hit", "edit_hit",
          "phonetic_hit", "tonnage_hit"]
_INTEGER_COLUMNS = _FLAGS + ["name_rank", "edit_rank", "phonetic_rank",
                            "tonnage_rank", "channel_count", "lookup_support"]


@dataclass(frozen=True)
class RobustRetrievalConfig(RetrievalConfig):
    """Independent ablations: Hangul-only, dual-tonnage-only, or their union."""

    profile: str = "robust"
    context_top_k: int = 160
    vowel_top_k: int = 160
    context_edit_top_k: int = 80

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.profile not in {"hangul", "dual-gt", "robust"}:
            raise ValueError("profile must be hangul, dual-gt or robust")
        for name in ("context_top_k", "vowel_top_k", "context_edit_top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def retrieval_spec(config: RetrievalConfig) -> dict | None:
    """Provenance for this extension; legacy configurations need no extension."""
    if not isinstance(config, RobustRetrievalConfig):
        return None
    digest = hashlib.sha256()
    for filename in ("robust_retrieval.py", "retrieval.py", "pipeline.py",
                     "memory.py", "features.py", "text.py", "name_features.py"):
        digest.update(filename.encode("utf-8"))
        digest.update(Path(__file__).with_name(filename).read_bytes())
    return {"name": RETRIEVAL_NAME, "code_sha256": digest.hexdigest(),
            "config": asdict(config)}


def check_retrieval_metadata(metadata: dict, pipeline=None) -> None:
    """Reject unversioned, modified or mismatched retrieval assets on load."""
    spec = metadata.get("retrieval_extension")
    values = metadata.get("retrieval", {})
    retriever = getattr(pipeline, "retriever", None)
    if spec is None:
        if (_EXTRA_FIELDS.intersection(values)
                or isinstance(retriever, RobustCandidateRetriever)
                or isinstance(getattr(retriever, "config", None), RobustRetrievalConfig)):
            raise ValueError("Robust retrieval is missing its retrieval_extension metadata")
        return
    if not isinstance(spec, dict) or not isinstance(values, dict):
        raise ValueError("Invalid robust retrieval metadata")
    try:
        config = RobustRetrievalConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid robust retrieval configuration") from error
    if spec != retrieval_spec(config):
        raise ValueError("Unknown or changed robust retrieval; regenerate its feature cache")
    if pipeline is not None:
        if (not isinstance(retriever, RobustCandidateRetriever)
                or asdict(retriever.config) != values):
            raise ValueError("Saved pipeline disagrees with robust retrieval metadata")


def _name_views(value: object) -> tuple[list[str], list[str]]:
    view = _view(_clean(value))
    contexts = [view.context_joined, view.hard_joined]
    vowels = [view.vowel_joined]
    if view.suffix_present:
        contexts.append(view.suffix_context)
        vowels.append(view.suffix_vowel)
    return (sorted({name for name in contexts if _has_letter(name)}),
            sorted({name for name in vowels if _has_letter(name)}))


class RobustCandidateRetriever(CandidateRetriever):
    """Keep legacy candidates and add tonnage-independent Hangul searches.

All candidates survive by default. An explicit ``max_total`` remains an
experimental recall/runtime tradeoff and may prune the combined union. Exact
lookup candidates survive that cap, as they do in the legacy implementation.
"""

    def __init__(self, catalog: pd.DataFrame, memory: TrainingMemory,
                 config: RobustRetrievalConfig | None = None) -> None:
        config = config or RobustRetrievalConfig()
        if not isinstance(config, RobustRetrievalConfig):
            raise TypeError("RobustCandidateRetriever requires RobustRetrievalConfig")
        super().__init__(catalog, memory, config)
        self._context_index = self._vowel_index = None
        self._context_edit_owners: dict[str, list[int]] = {}
        self._context_edit_names: list[str] = []
        if config.profile == "dual-gt" or not any((config.context_top_k,
                                                   config.vowel_top_k,
                                                   config.context_edit_top_k)):
            return
        documents = [(name, index) for index, name in
                     enumerate(self.catalog.vessel_name.tolist())]
        documents.extend((row.vessel_name, int(row.catalog_index))
                         for row in memory.aliases.itertuples(index=False))
        contexts, vowels = [], []
        for name, index in documents:
            context_views, vowel_views = _name_views(name)
            contexts.extend((text, index) for text in context_views)
            vowels.extend((text, index) for text in vowel_views)
        if config.context_top_k:
            self._context_index = _TextIndex(contexts, self.size, self.tie_keys)
        if config.vowel_top_k:
            self._vowel_index = _TextIndex(vowels, self.size, self.tie_keys)
        if config.context_edit_top_k:
            # Edit similarity bridges short/no-shared-bigram views. Scores are
            # reduced to unique vessels before applying the final top-k bound.
            for name, index in sorted(set(contexts + vowels)):
                self._context_edit_owners.setdefault(name, []).append(index)
            self._context_edit_names = sorted(
                self._context_edit_owners,
                key=lambda name: (min(self.tie_keys[self._context_edit_owners[name]]), name),
            )

    def _numeric_search(self, row: Mapping[str, Any]) -> np.ndarray:
        legacy = super()._numeric_search(row)
        if self.config.profile == "hangul":
            return legacy
        values = sorted({value for field in ("international_gross_tonnage", "gross_tonnage")
                         if (value := _positive_number(row.get(field))) is not None})
        # One usable measurement gives exactly the legacy candidates and ranks.
        if len(values) < 2 or not len(self._gt_indices):
            return legacy
        channels = [legacy]
        for value in values:
            selected = self._window(value)
            if not len(selected):
                ids = self._gt_indices
                order = np.lexsort((ids, self.tie_keys[ids], np.abs(self._gt[ids] - value)))
                selected = ids[order[:self.config.numeric_fallback_k]]
            channels.append(selected)
        ids = np.unique(np.concatenate(channels))
        # Unlike positional concatenation, this rank is independent of which
        # measurement is named international. The 86-feature baseline excludes
        # tonnage rank, but all experiments keep a well-defined finite column.
        distance = np.min(np.abs(self._gt[ids, None] - np.array(values)[None, :])
                          / self._gt[ids, None], axis=1)
        order = np.lexsort((ids, self.tie_keys[ids], distance))
        return ids[order]

    def _hangul_search(self, value: object) -> tuple[np.ndarray, np.ndarray]:
        contexts, vowels = _name_views(value)
        scores: dict[int, float] = {}
        for index, queries, limit in (
            (self._context_index, contexts, self.config.context_top_k),
            (self._vowel_index, vowels, self.config.vowel_top_k),
        ):
            if index is None:
                continue
            ids, values = index.search(queries, limit)
            for owner, score in zip(ids, values):
                scores[int(owner)] = max(scores.get(int(owner), 0.0), float(score))
        if self._context_edit_names and self.config.context_edit_top_k:
            edit_scores: dict[int, float] = {}
            for query in sorted(set(contexts + vowels)):
                for name, score, _ in process.extract(
                    query, self._context_edit_names,
                    scorer=Levenshtein.normalized_similarity,
                    limit=self.config.context_edit_top_k, score_cutoff=0.01,
                ):
                    for owner in self._context_edit_owners[name]:
                        edit_scores[owner] = max(edit_scores.get(owner, 0.0), float(score))
            ids = sorted(edit_scores, key=lambda owner: (-edit_scores[owner], self.tie_keys[owner]))
            for owner in ids[:self.config.context_edit_top_k]:
                scores[owner] = max(scores.get(owner, 0.0), edit_scores[owner])
        ids = np.array(sorted(scores), dtype=np.int64)
        return ids, np.array([scores[int(owner)] for owner in ids], dtype=np.float32)

    def retrieve(self, row: Mapping[str, Any]) -> pd.DataFrame:
        result = super().retrieve(row)
        if (self.config.profile == "dual-gt" or not has_hangul(row.get("vessel_name"))
                or not any((self.config.context_top_k, self.config.vowel_top_k,
                            self.config.context_edit_top_k))):
            return result
        ids, scores = self._hangul_search(row.get("vessel_name"))
        if not len(ids):
            return result
        result = result.set_index("catalog_index")
        result = result.reindex(np.union1d(result.index.to_numpy(), ids), fill_value=0)
        result.loc[ids, "phonetic_hit"] = 1
        result.loc[ids, "phonetic_score"] = np.maximum(
            result.loc[ids, "phonetic_score"].to_numpy(), scores,
        )
        selected = result.index[result.phonetic_hit != 0].to_numpy(dtype=np.int64)
        order = np.lexsort((selected, self.tie_keys[selected],
                            -result.loc[selected, "phonetic_score"].to_numpy()))
        result.loc[selected[order], "phonetic_rank"] = np.arange(1, len(selected) + 1)
        result["channel_count"] = result[_FLAGS].sum(axis=1)
        if self.config.max_total is not None and len(result) > self.config.max_total:
            forced = result.index[result[["call_match", "alias_match", "exact_hit"]].any(axis=1)]
            remaining = result.drop(index=forced).copy()
            remaining["_score"] = remaining[["name_score", "edit_score", "phonetic_score"]].max(axis=1)
            remaining["_tonnage"] = remaining.tonnage_rank.replace(0, np.inf)
            remaining["_id"] = self.tie_keys[remaining.index.to_numpy(dtype=np.int64)]
            remaining = remaining.sort_values(
                ["channel_count", "_score", "_tonnage", "_id"],
                ascending=[False, False, True, True], kind="stable",
            )
            count = max(0, self.config.max_total - len(forced))
            result = result.loc[np.sort(np.concatenate((forced.to_numpy(),
                                                       remaining.index[:count].to_numpy())))]
        result[_INTEGER_COLUMNS] = result[_INTEGER_COLUMNS].astype(np.int64)
        result.index.name = "catalog_index"
        return result.reset_index()[RESULT_COLUMNS]


class RobustMatchingPipeline(MatchingPipeline):
    """Build every fitted asset only from the caller-supplied training partition."""

    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame,
                 config: RobustRetrievalConfig | None = None) -> None:
        self.catalog = catalog.reset_index(drop=True).copy()
        self.memory = TrainingMemory().fit(train, self.catalog)
        self.categories = CategoryCompatibility().fit(train, self.catalog)
        self.retriever = RobustCandidateRetriever(self.catalog, self.memory, config)
        self.feature_builder = PairFeatureBuilder(self.catalog, self.categories)


def build_pipeline(train: pd.DataFrame, catalog: pd.DataFrame,
                   config: RetrievalConfig | None = None, extension=None) -> MatchingPipeline:
    """Construct legacy or versioned pipelines without changing legacy modules."""
    if extension is not None:
        if isinstance(extension, RobustRetrievalConfig):
            robust = extension
        elif isinstance(extension, dict):
            robust = RobustRetrievalConfig(**extension.get("config", {}))
            if extension != retrieval_spec(robust):
                raise ValueError("Unknown or changed robust retrieval extension")
        else:
            raise TypeError("extension must be a robust config or retrieval spec")
        if config is not None:
            supplied = asdict(config)
            wanted = asdict(robust)
            if any(wanted.get(key) != value for key, value in supplied.items()):
                raise ValueError("Base config disagrees with robust retrieval extension")
        config = robust
    if isinstance(config, RobustRetrievalConfig):
        return RobustMatchingPipeline(train, catalog, config)
    return MatchingPipeline(train, catalog, config)
