"""Training-only assets for fixed-ranker, vessel-level index-size diagnostics.

The saved 100% index remains the reference. Only the small index is rebuilt;
all of its lookup, aliases and category statistics use the same selected rows.
No ranker is fitted and no validation label participates in subset selection.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .data import read_catalog, read_queries
from .features import CategoryCompatibility
from .gt_augmentation import GTMissingAugmentationConfig, augmentation_spec
from .memory import TrainingMemory
from .ranking import RankerConfig
from .retrieval_comparison import (
    _hash, _load_model, _validate_partition, _validate_training_assets,
)
from .robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig, retrieval_spec
from .sampling import SamplingConfig
from .text import has_hangul
from .validation import make_folds


SUBSET_SEED = 20261027
SUBSET_ALGORITHM = "sha256-seed-vessel-sort-floor-v1"
_MODEL_FILES = ("model.txt", "ranker.json", "index.joblib",
                "training_summary.json", "validation_predictions.csv")


def _ids_digest(values):
    payload = json.dumps(sorted(str(value) for value in values), ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_training_vessels(fitting, *, fraction=0.8, seed=SUBSET_SEED):
    """Choose entire vessels, independently of row order or model performance."""
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("fraction must be finite and strictly between 0 and 1")
    if type(seed) is not int or seed < 0:
        raise ValueError("subset seed must be a nonnegative integer")
    if "target_vessel_id" not in fitting or fitting.target_vessel_id.isna().any():
        raise ValueError("Training vessel IDs must be present")
    vessels = set(fitting.target_vessel_id.astype(str))
    count = math.floor(fraction * len(vessels))
    if not 1 <= count < len(vessels):
        raise ValueError("fraction must select at least one but fewer than all training vessels")

    def priority(vessel):
        payload = json.dumps([seed, vessel], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest(), vessel

    selected = frozenset(sorted(vessels, key=priority)[:count])
    small = fitting.loc[fitting.target_vessel_id.astype(str).isin(selected)].copy()
    return small, selected


def assert_unchanged(protected_files):
    """Reject changed inputs or model artifacts at the end of a diagnostic."""
    if not protected_files:
        raise ValueError("Protected input/model file hashes must not be empty")
    for path, digest in protected_files.items():
        source = Path(path)
        if not source.is_file() or _hash(source) != digest:
            raise ValueError(f"Protected input/model file changed: {source}")
    return {"verified": True, "files": len(protected_files)}


def learned_assets_fingerprint(pipeline):
    """Digest catalog, every memory field and categories, excluding query caches.

    Frame order, index and dtypes are bound alongside values. TF-IDF matrices
    are protected by the source index file hash, not covered by this digest.
    """
    def frame_digest(frame):
        return {"columns": frame.columns.tolist(), "dtypes": frame.dtypes.astype(str).tolist(),
                "index_dtype": str(frame.index.dtype),
                "values_sha256": hashlib.sha256(
                    pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()}

    memory = pipeline.memory
    if (set(vars(memory)) != {"call_signs", "names", "aliases", "training_rows", "training_vessels"}
            or set(vars(pipeline.categories)) != {"smoothing", "tables"}):
        raise ValueError("Unsupported learned training asset state schema")
    state = {
        "training_rows": memory.training_rows,
        "training_vessels": sorted(memory.training_vessels),
        "call_signs": {key: dict(counts) for key, counts in memory.call_signs.items()},
        "names": {key: dict(counts) for key, counts in memory.names.items()},
        "aliases": frame_digest(memory.aliases),
        "categories": pipeline.categories.to_dict(),
        "catalog": frame_digest(pipeline.catalog),
    }
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


training_assets_fingerprint = learned_assets_fingerprint


def _validate_scope(summary, expected_fold):
    data = summary.get("dataset", {})
    if ("training_scope" in summary or "training_scope" in data
            or any(key.startswith("full_training") for key in summary)
            or any(key.startswith("full_training") for key in data)):
        raise ValueError("Index-size diagnostics require an outer-fold model, not full training")
    if data.get("outer_split") != "vessel":
        raise ValueError("Index-size diagnostics require a vessel-disjoint outer split")
    if expected_fold is not None:
        if type(expected_fold) is not int or expected_fold < 0:
            raise ValueError("expected_fold must be a nonnegative integer")
        if data.get("outer_fold") != expected_fold:
            raise ValueError("Saved model outer fold differs from expected_fold")
    if (summary.get("complete") is not True or data.get("complete") is not True
            or data.get("sampled") is not False
            or data.get("max_train_queries") is not None
            or data.get("max_valid_queries") is not None):
        raise ValueError("Use a complete, unsampled outer-fold model")
    if (data.get("outer_folds") != 5 or data.get("inner_split") != "mixed"
            or data.get("inner_folds") != 3 or data.get("seed") != 20261022):
        raise ValueError("Index-size diagnostics require the frozen 5-fold/mixed-3 profile")
    if (data.get("retrieval") != asdict(RobustRetrievalConfig())
            or data.get("retrieval_extension") != retrieval_spec(RobustRetrievalConfig())):
        raise ValueError("Index-size diagnostics require the unchanged default robust profile")
    if ("training_reweighting" in data
            or data.get("training_augmentation") != augmentation_spec(GTMissingAugmentationConfig())):
        raise ValueError("Index-size diagnostics require original GT-missing mass 0.5 without reweighting")
    if (summary.get("ranker") != asdict(RankerConfig())
            or data.get("sampling") != asdict(SamplingConfig())):
        raise ValueError("Index-size diagnostics require the frozen ranker and sampling configuration")


def prepare_index_scale(model_directory, train_path, catalog_path, *, fraction=0.8,
                        subset_seed=SUBSET_SEED, expected_fold=None):
    """Load fixed weights and construct a strict training-only smaller index.

    The result contains in-memory pipelines and a serializable audit proof.
    Neither the saved model nor its index/metadata is rewritten or relaxed.
    """
    directory, train_path, catalog_path = (
        Path(value).resolve() for value in (model_directory, train_path, catalog_path))
    paths = [directory / filename for filename in _MODEL_FILES] + [train_path, catalog_path]
    protected_files = {str(path): _hash(path) for path in paths}
    summary = json.loads((directory / "training_summary.json").read_text("utf-8"))
    _validate_scope(summary, expected_fold)
    matcher, loaded_summary, model_provenance = _load_model(
        directory, protected_files[str(train_path)], protected_files[str(catalog_path)])
    if summary != loaded_summary:
        raise ValueError("Saved summary changed while loading the fixed model")
    train, catalog = read_queries(train_path, labeled=True), read_catalog(catalog_path)
    heldout = _validate_partition(matcher, summary, train, catalog)
    data = summary["dataset"]
    assignments = make_folds(train, "vessel", data["outer_folds"], data["seed"])
    fitting = train.loc[assignments != data["outer_fold"]].copy()
    if data.get("counts", {}).get("selected_training_queries") != len(fitting):
        raise ValueError("Saved model did not select the complete outer training population")
    small, selected_vessels = select_training_vessels(
        fitting, fraction=fraction, seed=subset_seed)
    full_vessels = set(fitting.target_vessel_id.astype(str))
    heldout_vessels = set(heldout.target_vessel_id.astype(str))
    full_ids, small_ids, heldout_ids = (set(frame.sample_id.astype(str))
        for frame in (fitting, small, heldout))
    if (not full_ids or not heldout_ids or not small_ids < full_ids
            or full_ids & heldout_ids or full_vessels & heldout_vessels
            or not selected_vessels < full_vessels):
        raise ValueError("Invalid index membership or held-out leakage")

    started = time.perf_counter()
    expected_full = SimpleNamespace(memory=TrainingMemory().fit(fitting, catalog),
                                    categories=CategoryCompatibility().fit(fitting, catalog))
    _validate_training_assets(matcher.pipeline, expected_full)
    full_assets_seconds = time.perf_counter() - started
    if (matcher.pipeline.retriever.memory is not matcher.pipeline.memory
            or matcher.pipeline.feature_builder.category_model is not matcher.pipeline.categories):
        raise ValueError("Saved pipeline components disagree on their learned training assets")
    started = time.perf_counter()
    small_pipeline = RobustMatchingPipeline(small, catalog, RobustRetrievalConfig(**data["retrieval"]))
    small_fit_seconds = time.perf_counter() - started
    if (small_pipeline.memory.training_rows != len(small)
            or set(small_pipeline.memory.training_vessels) != selected_vessels):
        raise ValueError("Small index fitting population differs from selected training vessels")

    membership = train[["sample_id", "target_vessel_id"]].copy()
    membership["outer_fold"] = assignments
    membership["role"] = np.where(assignments == data["outer_fold"], "valid", "train")
    membership["selected_full"] = membership.role.eq("train")
    membership["selected_small"] = membership.sample_id.astype(str).isin(small_ids)
    membership["has_hangul"] = train.vessel_name.map(has_hangul)
    counts = lambda frame: {"rows": len(frame), "vessels": int(frame.target_vessel_id.nunique()),
        "hangul_rows": int(frame.vessel_name.map(has_hangul).sum()),
        "non_hangul_rows": int((~frame.vessel_name.map(has_hangul)).sum())}
    proof = {
        "name": "fixed_ranker_index_scale_v1", "verified": True,
        "coverage": {"fold": data["outer_fold"], "folds": data["outer_folds"],
                     "split": "vessel", "seed": data["seed"],
                     "full": counts(fitting), "small": counts(small), "heldout": counts(heldout)},
        "selection": {"algorithm": SUBSET_ALGORITHM, "fraction": fraction, "seed": subset_seed,
            "rounding": "floor(fraction * training_vessel_count)",
            "full_vessels_sha256": _ids_digest(full_vessels),
            "small_vessels_sha256": _ids_digest(selected_vessels),
            "full_sample_ids_sha256": _ids_digest(full_ids),
            "small_sample_ids_sha256": _ids_digest(small_ids),
            "heldout_sample_ids_sha256": _ids_digest(heldout_ids),
            "realized_vessel_fraction": len(selected_vessels) / len(full_vessels),
            "realized_row_fraction": len(small) / len(fitting)},
        "integrity": {"heldout_rows_excluded": True, "heldout_vessels_excluded": True,
            "small_is_strict_subset": True, "full_learned_assets_recomputed_and_equal": True,
            "full_index_source": "unchanged saved training index",
            "small_asset_scope": "lookup, aliases, TF-IDF and categories fitted together on selected original rows",
            "full_learned_assets_sha256": training_assets_fingerprint(matcher.pipeline),
            "small_learned_assets_sha256": training_assets_fingerprint(small_pipeline),
            "protected_files": protected_files},
        "model_provenance": model_provenance,
        "startup": {"model_load_and_integrity_seconds": model_provenance["load_and_integrity_seconds"],
            "full_training_assets_verification_seconds": full_assets_seconds,
            "small_index_build_seconds": small_fit_seconds},
        "interpretation": "Sensitivity to training-asset size with fixed weights; held-out target vessels are unknown to both indexes. This does not isolate retraining or reproduce the Public score change.",
    }
    assert_unchanged(protected_files)
    return {"matcher": matcher, "summary": summary, "small_pipeline": small_pipeline, "fitting": fitting,
            "heldout": heldout, "membership": membership, "proof": proof,
            "protected_files": protected_files}
