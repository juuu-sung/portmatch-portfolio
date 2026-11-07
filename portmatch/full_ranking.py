"""Fixed-profile final training from full-public internal-OOF feature caches.

The final model has no external holdout. Training deliberately writes neither
validation predictions nor accuracy metrics; the preceding five-fold experiments
are the model-selection evidence. Legacy cache and model implementations remain
unchanged so their source hashes and reproducibility proofs stay valid.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

from .feature_selection import FeatureSelection
from .ranking import RankerConfig, VesselRanker


TRAINING_SCOPE = "full_public_train"
_NO_VALIDATION = {
    "available": False,
    "held_out_queries": 0,
    "reason": "no_external_holdout_full_training_scope",
}


def _hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def full_ranker_spec():
    """Version the new orchestration separately from frozen legacy weighting."""
    digest = hashlib.sha256()
    for name in ("full_ranking.py", "ranking.py"):
        digest.update(name.encode("utf-8"))
        digest.update(Path(__file__).with_name(name).read_bytes())
    return {"name": "full_public_ranker_v1", "code_sha256": digest.hexdigest()}


def _canonical_features():
    from .name_feature_cache import _legacy_feature_names
    from .name_features import FEATURE_NAMES

    source = list(_legacy_feature_names()) + list(FEATURE_NAMES)
    return source, [name for name in source if name != "retrieval_tonnage_rank"]


def _check_fixed_profile(metadata):
    """Reject a new experiment presented as the selected final-training profile."""
    from .full_training_data import check_full_training_metadata
    from .gt_augmentation import check_augmentation_metadata
    from .robust_retrieval import RobustRetrievalConfig
    from .sampling import SamplingConfig

    if check_full_training_metadata(metadata) is None:
        raise ValueError("Full training requires a full_public_train cache")
    augmentation = check_augmentation_metadata(metadata)
    if (augmentation is None or augmentation.original_mass != .5
            or metadata.get("retrieval") != asdict(RobustRetrievalConfig())
            or metadata.get("sampling") != asdict(SamplingConfig())
            or metadata.get("seed") != RankerConfig().seed
            or metadata.get("inner_split") != "mixed"
            or metadata.get("inner_folds") != 3
            or metadata.get("training_reweighting") is not None):
        raise ValueError("Full training fixes robust retrieval, mixed three-fold OOF, "
                         "original-mass 0.5, default sampling and seed")


def check_full_model_summary(summary, model_state=None):
    """Validate full-model metadata both locally and in a minimal runtime bundle.

    This is an inference-time metadata check. Full source-cache lineage is
    verified before training and again by the submission builder; runtime never
    needs public labels, feature shards, local file paths or another query row.
    """
    from .full_training_data import check_full_training_metadata
    from .name_feature_cache import transform_spec
    from .ranker_commands import _check_training_weighting

    dataset = summary.get("dataset", {})
    full = check_full_training_metadata(dataset)
    if full is None:
        if ("training_scope" in summary
                or "full_training_trainer" in summary or "full_training_proof" in summary):
            raise ValueError("Full model summary is missing its full-training dataset")
        return None
    _check_fixed_profile(dataset)
    if (summary.get("schema_version") != 1 or summary.get("complete") is not True
            or summary.get("training_scope") != TRAINING_SCOPE
            or summary.get("validation") != _NO_VALIDATION
            or any(key in summary for key in ("metrics", "validation_metrics", "validation_predictions"))
            or summary.get("full_training_trainer") != full_ranker_spec()
            or summary.get("ranker") != asdict(RankerConfig())):
        raise ValueError("Invalid full-model scope, fixed profile or no-validation declaration")
    source, selected = _canonical_features()
    selection = FeatureSelection.from_dict(summary["feature_selection"])
    if (dataset.get("feature_transform") != transform_spec()
            or dataset.get("feature_names") != source
            or selection.source_feature_names != tuple(source)
            or selection.selected_features != tuple(selected)
            or summary.get("trained_feature_names") != selected):
        raise ValueError("Full model must retain exactly the canonical 86 features")
    _check_training_weighting(summary)
    if model_state is not None:
        if (model_state.get("config") != asdict(RankerConfig())
                or model_state.get("feature_names") != selected
                or type(model_state.get("iterations")) is not int
                or not 0 < model_state["iterations"] <= RankerConfig().n_estimators):
            raise ValueError("Native full model disagrees with its fixed training profile")
    return full


def transform_full_cache(data, output, train_path, catalog_path):
    """Reuse the frozen name transform after checking the full-data contract."""
    from .full_training_data import verify_full_training_cache
    from .name_feature_cache import transform_cache, verify_derivation
    from .training_data import read_metadata

    metadata = read_metadata(data)
    _check_fixed_profile(metadata)
    if metadata.get("feature_transform") is not None or "derivation" in metadata:
        raise ValueError("Full name transform requires the original 70-column cache")
    verify_full_training_cache(data, train_path=train_path, catalog_path=catalog_path)
    result = transform_cache(data, output, train_path, catalog_path)
    verify_derivation(output, result["derivation"]["parent_metadata_sha256"])
    verify_full_training_cache(output, train_path=train_path, catalog_path=catalog_path)
    return result


def train_full_ranker(data, output, *, config=None, train_path=None, catalog_path=None):
    """Train the selected fixed profile, without scoring its own source rows."""
    from .full_training_data import verify_full_training_cache
    from .name_feature_cache import verify_derivation
    from .ranker_commands import (
        _check_augmented_features, _check_feature_code, _load_retrieval_index,
        _new_output, _training_weighting_summary,
    )
    from .training_data import load_training_arrays, read_metadata

    config = RankerConfig() if config is None else config
    if not isinstance(config, RankerConfig) or asdict(config) != asdict(RankerConfig()):
        raise ValueError("Full model training fixes RankerConfig defaults: 300 rounds and six threads")
    data = Path(data).resolve()
    destination = Path(output).resolve()
    if (data == destination or data.is_relative_to(destination)
            or destination.is_relative_to(data)):
        raise ValueError("Full cache and model output directories must be separate")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Choose a new or empty output directory: {destination}")
    metadata = read_metadata(data)
    _check_fixed_profile(metadata)
    _check_feature_code(metadata)
    source, selected = _canonical_features()
    selection = FeatureSelection.from_names(metadata["feature_names"], selected)
    _check_augmented_features(metadata, selection)
    if metadata["feature_names"] != source or "derivation" not in metadata:
        raise ValueError("Full training requires the canonical token-phonetic derived cache")
    print("Verifying full-public OOF cache, source masses and unchanged name-feature lineage", flush=True)
    input_paths = {"train_path": train_path, "catalog_path": catalog_path}
    for key, value in input_paths.items():
        if value is None:
            value = metadata.get("provenance", {}).get(key)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("Full training requires original train and catalog paths for population verification")
        input_paths[key] = value
    full_proof = verify_full_training_cache(data, **input_paths)
    if full_proof.get("original_population_verified") is not True:
        raise ValueError("Full training requires verified original public rows and final inference assets")
    derivation_proof = verify_derivation(data, metadata["derivation"]["parent_metadata_sha256"])
    pipeline = _load_retrieval_index(data / metadata["index_path"], metadata)
    del pipeline
    cache_hash = _hash_file(data / "metadata.json")
    trainer_spec = full_ranker_spec()
    destination = _new_output(destination)
    started = time.perf_counter()
    print("Loading full-public internal-OOF training pairs; no external validation is scored", flush=True)
    X, y, groups, language, names, masses = load_training_arrays(
        data, feature_names=selected, return_query_mass=True)
    if masses is None or names != selected:
        raise ValueError("Full training lost source masses or the canonical feature schema")
    weighting = _training_weighting_summary(groups, language, masses, config.hangul_weight)
    print(f"Training {len(groups):,} OOF groups, {len(y):,} pairs, {len(names)} features", flush=True)
    ranker = VesselRanker(config).fit(X, y, groups, language, names, query_mass=masses)
    fit_seconds = time.perf_counter() - started
    del X, y, groups, language, masses
    if _hash_file(data / "metadata.json") != cache_hash or full_ranker_spec() != trainer_spec:
        raise ValueError("Cache metadata or training code changed during full-model training")
    _check_feature_code(metadata)
    ranker.save(destination)
    shutil.copy2(data / metadata["index_path"], destination / "index.joblib")
    if _hash_file(destination / "index.joblib") != metadata["retrieval_index_sha256"]:
        raise ValueError("Full model index changed during copy")
    ranker.feature_importance().to_csv(destination / "feature_importance.csv", index=False, encoding="utf-8-sig")
    summary = {
        "schema_version": 1, "complete": True, "training_scope": TRAINING_SCOPE,
        "dataset": metadata, "cache_directory": str(data), "cache_metadata_sha256": cache_hash,
        "feature_selection": selection.to_dict(), "trained_feature_names": names,
        "ranker": asdict(config), "training_weighting": weighting,
        "full_training_trainer": trainer_spec, "full_training_proof": full_proof,
        "full_derivation_proof": derivation_proof,
        "validation": dict(_NO_VALIDATION),
        "train_load_and_fit_seconds": fit_seconds, "total_seconds": time.perf_counter() - started,
        "note": "Fixed profile selected before final training. Selected original public rows supply "
                "internal-OOF features; final inference assets fit all original public rows. "
                "Diagnostic query limits are explicitly recorded by dataset.sampled. No held-out "
                "accuracy is available, and source rows are not rescored as validation.",
    }
    state = json.loads((destination / "ranker.json").read_text(encoding="utf-8"))
    check_full_model_summary(summary, model_state=state)
    _write_json(destination / "training_summary.json", summary)
    print(f"Saved full-public model without validation metrics: {destination}", flush=True)
    return summary
