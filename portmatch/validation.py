"""Validation splits and retrieval metrics, separate from inference."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .text import has_hangul


def make_folds(train: pd.DataFrame, split: str, folds: int, seed: int) -> np.ndarray:
    if folds < 2:
        raise ValueError("At least two folds are required")
    language = train.vessel_name.map(has_hangul).to_numpy(dtype=int)
    if split == "vessel":
        if train.target_vessel_id.nunique() < folds:
            raise ValueError("More folds than distinct vessels")
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        partitions = splitter.split(train, language, groups=train.target_vessel_id)
    elif split == "row":
        if min(np.bincount(language, minlength=2)) < folds:
            raise ValueError("Each language group needs at least one row per fold")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        partitions = splitter.split(train, language)
    else:
        raise ValueError(f"Unknown split: {split}")
    assignments = np.full(len(train), -1, dtype=int)
    for fold, (fitting, validation) in enumerate(partitions):
        assignments[validation] = fold
        if split == "vessel":
            left = set(train.iloc[fitting].target_vessel_id)
            right = set(train.iloc[validation].target_vessel_id)
            if left & right:
                raise AssertionError("A validation vessel leaked into training")
    if (assignments < 0).any():
        raise AssertionError("Incomplete validation split")
    return assignments


def sample_validation(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """A diagnostic sample only; preserve both languages when possible."""
    if limit is None or limit >= len(frame):
        return frame
    if limit < 1:
        raise ValueError("max-queries must be positive")
    rng = np.random.default_rng(seed)
    language = frame.vessel_name.map(has_hangul).to_numpy()
    chosen: list[int] = []
    if limit >= 2:
        for value in (False, True):
            positions = np.flatnonzero(language == value)
            if len(positions):
                chosen.append(int(rng.choice(positions)))
    remaining = np.setdiff1d(np.arange(len(frame)), chosen)
    chosen.extend(rng.choice(remaining, size=limit - len(chosen), replace=False).tolist())
    return frame.iloc[sorted(chosen)]


def summarize(records: pd.DataFrame) -> dict:
    metrics = {}
    for name, part in [("all", records),
                       ("hangul", records[records.has_hangul]),
                       ("non_hangul", records[~records.has_hangul])]:
        metrics[name] = {"queries": len(part)}
        for column in records:
            if column.startswith("recall_"):
                metrics[name][column] = float(part[column].mean()) if len(part) else None
        for percentile in (50, 95, 100):
            metrics[name][f"candidate_count_p{percentile}"] = (
                float(np.percentile(part.candidate_count, percentile)) if len(part) else None)
    korean = metrics["hangul"].get("recall_union")
    other = metrics["non_hangul"].get("recall_union")
    metrics["weighted_candidate_recall"] = (
        .7 * korean + .3 * other if korean is not None and other is not None else None)
    metrics["note"] = "Candidate recall is an upper bound, not top-1 accuracy or a competition score."
    return metrics
