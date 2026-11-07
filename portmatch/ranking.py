"""Fixed-round ranking on training features, with independent query metrics.

No validation rows or validation scores are accepted by ``fit``. Model selection
and any later tuning must keep their own held-out evaluation separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_FORBIDDEN_FEATURES = {
    "sample_id", "vessel_id", "target_vessel_id", "candidate_vessel_id",
    "catalog_index", "target_catalog_index", "label", "target", "group", "query_id",
}


@dataclass(frozen=True)
class RankerConfig:
    n_estimators: int = 300
    learning_rate: float = .05
    num_leaves: int = 31
    min_child_samples: int = 20
    reg_lambda: float = 1.0
    n_jobs: int = 6
    seed: int = 20261022
    hangul_weight: float = .7

    def __post_init__(self) -> None:
        for field in ("n_estimators", "num_leaves", "min_child_samples", "n_jobs"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.num_leaves < 2:
            raise ValueError("num_leaves must be at least two")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.reg_lambda) or self.reg_lambda < 0:
            raise ValueError("reg_lambda must be finite and nonnegative")
        if not np.isfinite(self.hangul_weight) or not 0 < self.hangul_weight < 1:
            raise ValueError("hangul_weight must be strictly between zero and one")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _feature_schema(feature_names: list[str]) -> list[str]:
    names = list(feature_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Feature names must be nonempty strings")
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique")
    forbidden = [name for name in names if name in _FORBIDDEN_FEATURES or name.endswith("_id")]
    if forbidden:
        raise ValueError(f"Identifiers or targets cannot be features: {forbidden}")
    if any(any(character in name for character in '[]{}:,"') for name in names):
        raise ValueError("Feature names contain characters unsupported by LightGBM")
    return names


def _matrix(X: np.ndarray, names: list[str]) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("Feature matrix shape does not match feature names")
    if np.isinf(matrix).any():
        raise ValueError("Feature matrix contains infinity; NaN is allowed for missing values")
    return matrix


def _groups(group: np.ndarray) -> np.ndarray:
    values = np.asarray(group)
    if values.ndim != 1 or not len(values):
        raise ValueError("group must be a nonempty one-dimensional array")
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("group must contain positive integer lengths") from error
    if (not np.isfinite(numeric).all() or (numeric < 1).any()
            or not np.equal(numeric, np.floor(numeric)).all()
            or (numeric > np.iinfo(np.int32).max).any()):
        raise ValueError("group must contain positive int32 lengths")
    return numeric.astype(np.int32)


def _booleans(values: np.ndarray, name: str, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have one value per query")
    if not np.isin(array, [False, True]).all():
        raise ValueError(f"{name} must contain only boolean values")
    return array.astype(bool)


def query_sample_weights(group: np.ndarray, query_has_hangul: np.ndarray,
                         hangul_weight: float = .7, *,
                         query_mass: np.ndarray | None = None) -> np.ndarray:
    """Balance query mass within each language, then balance the languages.

    LightGBM receives row weights. A query's weight is divided by its candidate
    count and expanded over its rows; the final row-weight mean is one. When both
    languages are present their weight totals are ``hangul_weight`` and its
    complement. Without ``query_mass``, each query has equal weight within its
    language. Explicit masses can divide one source sample's weight between its
    original input, masked inputs, and out-of-fold views without giving sources
    with more retained inputs extra total weight. This is a training heuristic,
    not the competition metric or a claim about exact per-query LambdaRank
    gradient contributions.
    """
    if not np.isfinite(hangul_weight) or not 0 < hangul_weight < 1:
        raise ValueError("hangul_weight must be strictly between zero and one")
    groups = _groups(group)
    languages = _booleans(query_has_hangul, "query_has_hangul", len(groups))
    masses = None
    if query_mass is not None:
        if np.iscomplexobj(query_mass):
            raise ValueError("query_mass must contain finite positive numbers")
        try:
            masses = np.asarray(query_mass, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("query_mass must contain finite positive numbers") from error
        if masses.ndim != 1 or len(masses) != len(groups):
            raise ValueError("query_mass must have one value per query")
        if not np.isfinite(masses).all() or (masses <= 0).any():
            raise ValueError("query_mass must contain finite positive numbers")
    query_weight = np.zeros(len(groups), dtype=np.float64)
    for language, mass in ((True, hangul_weight), (False, 1 - hangul_weight)):
        selected = languages == language
        count = int(selected.sum())
        if count:
            if masses is None:
                query_weight[selected] = mass / count
            else:
                # Scaling first avoids overflow when valid finite masses are large.
                relative = masses[selected] / masses[selected].max()
                query_weight[selected] = mass * relative / relative.sum()
    # Normalize also covers training subsets containing just one language.
    query_weight /= query_weight.sum()
    weights = np.repeat(query_weight / groups, groups)
    weights *= len(weights) / weights.sum()
    return weights.astype(np.float32)


class VesselRanker:
    """A CPU LightGBM ranker with an exact feature schema and native model file."""

    def __init__(self, config: RankerConfig | None = None):
        self.config = config or RankerConfig()
        self.feature_names: list[str] = []
        self._booster: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray,
            query_has_hangul: np.ndarray, feature_names: list[str], *,
            query_mass: np.ndarray | None = None) -> "VesselRanker":
        import lightgbm as lgb

        names = _feature_schema(feature_names)
        matrix = _matrix(X, names)
        groups = _groups(group)
        labels = np.asarray(y)
        if not len(matrix) or labels.ndim != 1 or len(labels) != len(matrix):
            raise ValueError("Labels must have one entry per nonempty feature row")
        if int(groups.sum(dtype=np.int64)) != len(matrix):
            raise ValueError("Group lengths must sum to the number of feature rows")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Ranking labels must be binary")
        starts = np.r_[0, np.cumsum(groups, dtype=np.int64)[:-1]]
        positives = np.add.reduceat(labels.astype(np.int64), starts)
        if (positives != 1).any() or (groups < 2).any():
            raise ValueError("Each training query needs exactly one positive and at least one negative")
        weights = query_sample_weights(groups, query_has_hangul, self.config.hangul_weight,
                                       query_mass=query_mass)
        model = lgb.LGBMRanker(
            objective="lambdarank", label_gain=[0, 1],
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            reg_lambda=self.config.reg_lambda,
            n_jobs=self.config.n_jobs, random_state=self.config.seed,
            deterministic=True, force_col_wise=True, device_type="cpu", verbosity=-1,
        )
        model.fit(matrix, labels.astype(np.uint8), group=groups,
                  sample_weight=weights, feature_name=names)
        self._booster = model.booster_
        self.feature_names = names
        return self

    def predict(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        if self._booster is None:
            raise ValueError("Ranker must be fitted or loaded before prediction")
        names = _feature_schema(feature_names)
        if names != self.feature_names:
            raise ValueError("Feature names or their order differ from the fitted schema")
        matrix = _matrix(X, names)
        if not len(matrix):
            return np.empty(0, dtype=np.float64)
        result = np.asarray(self._booster.predict(matrix, num_threads=self.config.n_jobs), dtype=np.float64)
        if result.shape != (len(matrix),) or not np.isfinite(result).all():
            raise ValueError("Ranker produced invalid prediction scores")
        return result

    def save(self, directory: str | Path) -> None:
        import lightgbm as lgb

        if self._booster is None:
            raise ValueError("Ranker must be fitted before saving")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_path, state_path = directory / "model.txt", directory / "ranker.json"
        if model_path.exists() or state_path.exists():
            raise FileExistsError("Ranker files already exist; choose a new output directory")
        self._booster.save_model(str(model_path))
        state = {"artifact_version": 1, "lightgbm_version": lgb.__version__,
                 "config": asdict(self.config), "feature_names": self.feature_names,
                 "iterations": self._booster.current_iteration(),
                 "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "VesselRanker":
        import lightgbm as lgb

        directory = Path(directory)
        state = json.loads((directory / "ranker.json").read_text(encoding="utf-8"))
        if state.get("artifact_version") != 1:
            raise ValueError("Unsupported ranker artifact version")
        model_path = directory / "model.txt"
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != state.get("model_sha256"):
            raise ValueError("Native ranker file does not match its metadata")
        model = cls(RankerConfig(**state["config"]))
        model.feature_names = _feature_schema(state["feature_names"])
        model._booster = lgb.Booster(model_file=str(model_path))
        if model._booster.feature_name() != model.feature_names:
            raise ValueError("Native ranker feature schema does not match its metadata")
        return model

    def feature_importance(self) -> pd.DataFrame:
        if self._booster is None:
            raise ValueError("Ranker must be fitted or loaded before feature importance")
        result = pd.DataFrame({"feature": self.feature_names,
                               "gain": self._booster.feature_importance(importance_type="gain"),
                               "split": self._booster.feature_importance(importance_type="split")})
        return result.sort_values(["gain", "feature"], ascending=[False, True], kind="stable").reset_index(drop=True)


def ranking_metrics(records: pd.DataFrame) -> dict:
    """Measure every query, including retrieval failures with target_rank = 0."""
    required = {"has_hangul", "candidate_found", "target_rank", "candidate_count"}
    if not required.issubset(records.columns):
        raise ValueError(f"Ranking metric columns missing: {sorted(required - set(records.columns))}")
    languages = _booleans(records.has_hangul.to_numpy(), "has_hangul", len(records))
    found = _booleans(records.candidate_found.to_numpy(), "candidate_found", len(records))
    ranks = pd.to_numeric(records.target_rank, errors="raise").to_numpy(dtype=float)
    counts = pd.to_numeric(records.candidate_count, errors="raise").to_numpy(dtype=float)
    if (not np.isfinite(ranks).all() or not np.isfinite(counts).all()
            or (ranks < 0).any() or (counts < 0).any()
            or not np.equal(ranks, np.floor(ranks)).all()
            or not np.equal(counts, np.floor(counts)).all()
            or (ranks > counts).any() or not np.array_equal(found, ranks > 0)):
        raise ValueError("Target ranks must match candidate_found and candidate_count")
    metrics = {}
    for name, selected in (("all", np.ones(len(records), dtype=bool)),
                           ("hangul", languages), ("non_hangul", ~languages)):
        count = int(selected.sum())
        success = found[selected]
        rank = ranks[selected]
        part = {"queries": count, "candidate_hits": int(success.sum()),
                "correct_top1": int((success & (rank == 1)).sum()),
                "candidate_recall": float(success.mean()) if count else None,
                "top1_accuracy": float((success & (rank == 1)).mean()) if count else None}
        for top_k in (5, 10, 50, 100, 200):
            part[f"recall_at_{top_k}"] = float((success & (rank <= top_k)).mean()) if count else None
        metrics[name] = part
    for output, field in (("weighted_top1", "top1_accuracy"),
                          ("weighted_candidate_recall", "candidate_recall")):
        korean, other = metrics["hangul"][field], metrics["non_hangul"][field]
        metrics[output] = .7 * korean + .3 * other if korean is not None and other is not None else None
    return metrics
