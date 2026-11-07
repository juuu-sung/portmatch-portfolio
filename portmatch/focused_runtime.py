"""Standalone, fixed fold-0 focused ranker with append-only Hangul search.

Only inference dependencies are imported. No experiment directories, validation
labels or feature shards are required to restore this bundle.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .data import QUERY_COLUMNS
from .feature_selection import FeatureSelection
from .hangul_expansion import ExpansionConfig, HangulExpansion
from .name_feature_cache import transform_features, transform_spec
from .name_features import NameFeatureComputer
from .ranking import VesselRanker
from .robust_retrieval import check_retrieval_metadata


PROFILE = "tonnage_repair_v1_focused_search_fold0"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class FocusedMatcher:
    def __init__(self, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "runtime.json").read_text("utf-8"))
        if (metadata.get("schema_version") != 1 or metadata.get("profile") != PROFILE
                or metadata.get("fold") != 0 or metadata.get("complete") is not True):
            raise ValueError("Expected a complete focused_search fold0 bundle")
        names = ("index.joblib", "hangul_expansion.joblib", "ranker/model.txt", "ranker/ranker.json")
        if set(metadata["assets"]) != set(names):
            raise ValueError("Unexpected fixed assets")
        # Check all bytes before deserializing any object.
        for name in names:
            if file_hash(directory / name) != metadata["assets"][name]:
                raise ValueError(f"Fixed asset changed: {name}")
        self.feature_transform = metadata["feature_transform"]
        if self.feature_transform != transform_spec():
            raise ValueError("Name feature implementation differs from training")
        digest = hashlib.sha256()
        for name in ("features.py", "text.py", "memory.py", "retrieval.py", "pipeline.py"):
            digest.update(name.encode())
            digest.update(Path(__file__).with_name(name).read_bytes())
        if digest.hexdigest() != metadata["feature_code_sha256"]:
            raise ValueError("Base feature implementation differs from training")
        self.selection = FeatureSelection.from_dict(metadata["feature_selection"])
        self.ranker = VesselRanker.load(directory / "ranker")
        if (len(self.ranker.feature_names) != 86
                or "retrieval_tonnage_rank" in self.ranker.feature_names
                or self.selection.selected_features != tuple(self.ranker.feature_names)
                or self.ranker.config.n_jobs != 6):
            raise ValueError("Invalid focused ranker schema or thread configuration")
        check_retrieval_metadata(metadata["retrieval_metadata"])
        self.pipeline = joblib.load(directory / "index.joblib")
        check_retrieval_metadata(metadata["retrieval_metadata"], pipeline=self.pipeline)
        self.catalog_ids = self.pipeline.catalog.vessel_id.astype(str).to_numpy()
        self.expansion = joblib.load(directory / "hangul_expansion.joblib")
        if (not isinstance(self.expansion, HangulExpansion)
                or asdict(self.expansion.config) != asdict(ExpansionConfig())
                or metadata["expansion_config"] != asdict(self.expansion.config)
                or self.expansion.size != len(self.catalog_ids)
                or not np.array_equal(self.expansion.tie_keys, self.catalog_ids)):
            raise ValueError("Search expansion differs from the frozen catalog/configuration")
        self.pipeline._token_phonetic_computer_v1 = NameFeatureComputer(
            self.pipeline.catalog.vessel_name.tolist())

    def _matrix(self, query, candidates):
        features = self.pipeline.features(query, candidates)
        features = transform_features(self.pipeline, query, candidates, features)
        return self.selection.apply(features.to_numpy(dtype=np.float32), features.columns.tolist())

    def candidate_scores(self, row):
        query = {key: row.get(key) for key in QUERY_COLUMNS}
        old = self.pipeline.retrieve(query)
        old_x = self._matrix(query, old)
        full = self.expansion.append(query, old)
        old_ids = old.catalog_index.to_numpy(dtype=np.int64)
        ids = full.catalog_index.to_numpy(dtype=np.int64)
        positions = np.searchsorted(ids, old_ids)
        added = ~np.isin(ids, old_ids)
        matrix = np.empty((len(full), old_x.shape[1]), dtype=np.float32)
        matrix[positions] = old_x
        if added.any():
            matrix[added] = self._matrix(query, full.loc[added].reset_index(drop=True))
        # Existing candidates keep their original, pool-dependent feature values.
        if not np.array_equal(matrix[positions], old_x, equal_nan=True):
            raise ValueError("Search expansion changed existing candidate features")
        scores = self.ranker.predict(matrix, self.ranker.feature_names)
        if len(scores) != len(ids) or not np.isfinite(scores).all():
            raise ValueError("Invalid prediction scores")
        return ids, scores

    def predict(self, row):
        ids, scores = self.candidate_scores(row)
        # Same score -> same fixed vessel ID, independently of batch order.
        order = np.lexsort((ids, self.catalog_ids[ids], -scores))
        return str(self.catalog_ids[ids[order[0]]])
