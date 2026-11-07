"""Build fixed training assets; queries never update them."""
from __future__ import annotations

import pandas as pd

from .features import CategoryCompatibility, PairFeatureBuilder
from .memory import TrainingMemory
from .retrieval import CandidateRetriever, RetrievalConfig


class MatchingPipeline:
    def __init__(self, train: pd.DataFrame, catalog: pd.DataFrame,
                 config: RetrievalConfig | None = None):
        self.catalog = catalog.reset_index(drop=True).copy()
        self.memory = TrainingMemory().fit(train, self.catalog)
        self.categories = CategoryCompatibility().fit(train, self.catalog)
        self.retriever = CandidateRetriever(self.catalog, self.memory, config)
        self.feature_builder = PairFeatureBuilder(self.catalog, self.categories)

    def retrieve(self, row: dict) -> pd.DataFrame:
        return self.retriever.retrieve(row)

    def features(self, row: dict, candidates: pd.DataFrame) -> pd.DataFrame:
        return self.feature_builder.transform(row, candidates)
