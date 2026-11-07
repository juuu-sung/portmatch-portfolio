"""Strict assets for retraining canonical86 without four category support logs.

The experiment consumes the exact existing 87-column, mass-0.5 cache. Search
and category assets remain the full outer training population T. Only selected
model columns and newly fitted weights differ; no index is rebuilt here.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import joblib
import numpy as np

from .data import read_catalog, read_queries
from .feature_selection import FeatureSelection
from .feature_transforms import check_transform_metadata
from .features import CategoryCompatibility
from .index_scale_assets import (
    _ids_digest, _validate_scope, assert_unchanged, learned_assets_fingerprint,
)
from .memory import TrainingMemory
from .name_feature_cache import _asset_names, _inside, _legacy_feature_names, transform_spec
from .name_features import FEATURE_NAMES
from .ranking import RankerConfig
from .retrieval_comparison import (
    _hash, _json, _load_model, _validate_partition, _validate_training_assets, _verify_index,
)
from .text import has_hangul
from .validation import make_folds


SUPPORT_FEATURES = (
    "flag_source_support_log", "flag_pair_support_log",
    "type_source_support_log", "type_pair_support_log",
)
RUNNERS = ("baseline86", "no_support82")
_MODEL_FILES = ("model.txt", "ranker.json", "index.joblib",
                "training_summary.json", "validation_predictions.csv")


def feature_schemas():
    source = list(_legacy_feature_names()) + list(FEATURE_NAMES)
    baseline = [name for name in source if name != "retrieval_tonnage_rank"]
    experiment = [name for name in baseline if name not in SUPPORT_FEATURES]
    if (len(source), len(baseline), len(experiment)) != (87, 86, 82):
        raise ValueError("Support-log ablation requires the frozen 87/86/82 schemas")
    if not set(SUPPORT_FEATURES).issubset(baseline):
        raise ValueError("Canonical schema is missing a support-log feature")
    return {"source_features": source, "baseline_features": baseline,
            "experiment_features": experiment, "excluded_features": list(SUPPORT_FEATURES)}


def check_support_ablation_metadata(metadata):
    """Permit only unsampled vessel5/mixed3 robust GT-missing mass-0.5 caches."""
    if not isinstance(metadata, dict):
        raise ValueError("Support-log ablation requires cache metadata")
    _validate_scope({"complete": True, "dataset": metadata,
                     "ranker": asdict(RankerConfig())}, None)
    fold = metadata.get("outer_fold")
    if type(fold) is not int or fold not in range(5):
        raise ValueError("Support-log ablation outer_fold must be an integer from 0 to 4")
    if (metadata.get("feature_transform") != transform_spec()
            or metadata.get("feature_names") != feature_schemas()["source_features"]):
        raise ValueError("Support-log ablation requires the unchanged token-phonetic 87-column cache")
    check_transform_metadata(metadata)
    from .ranker_commands import feature_code_hash

    if metadata.get("provenance", {}).get("feature_code_sha256") != feature_code_hash():
        raise ValueError("Support-log ablation requires unchanged feature-code provenance")


def _check_selection(summary, kind):
    schemas = feature_schemas()
    expected = schemas["baseline_features" if kind == "baseline86" else "experiment_features"]
    selection = FeatureSelection.from_dict(summary.get("feature_selection", {}))
    if (list(selection.source_feature_names) != schemas["source_features"]
            or list(selection.selected_features) != expected
            or summary.get("trained_feature_names") != expected):
        raise ValueError(f"{kind} does not have its exact frozen feature selection")
    return selection


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _valid_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _verify_shared_cache(summary, cache_directory=None):
    """Bind the saved derivation proof to actual bytes with bounded-memory hashes.

    Training already checks unchanged arrays through verify_derivation. Here the
    hash-bound manifest and every original/derived asset are checked again, without
    decoding multi-gigabyte feature arrays or calculating replacement features.
    """
    started = time.perf_counter()
    recorded = summary.get("cache_directory")
    if not isinstance(recorded, str) or not recorded:
        raise ValueError("Saved model is missing its cache directory")
    root = Path(recorded).resolve()
    if cache_directory is not None and Path(cache_directory).resolve() != root:
        raise ValueError("Support-log ablation must reuse the baseline cache directory")
    metadata = _json(root / "metadata.json")
    metadata_hash = _hash(root / "metadata.json")
    if metadata_hash != summary.get("cache_metadata_sha256") or metadata != summary.get("dataset"):
        raise ValueError("Actual cache metadata differs from the baseline training summary")
    check_support_ablation_metadata(metadata)
    derivation = metadata["derivation"]
    if (set(derivation) != {"manifest_path", "manifest_sha256", "parent_metadata_sha256"}
            or not all(_valid_hash(derivation.get(key))
                       for key in ("manifest_sha256", "parent_metadata_sha256"))):
        raise ValueError("Invalid support-ablation cache derivation")
    manifest_path = _inside(root, derivation["manifest_path"])
    if _hash(manifest_path) != derivation["manifest_sha256"]:
        raise ValueError("Support-ablation derivation manifest hash mismatch")
    manifest = _json(manifest_path)
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or manifest.get("transform") != transform_spec()
            or manifest.get("parent_metadata_sha256") != derivation["parent_metadata_sha256"]):
        raise ValueError("Invalid support-ablation derivation manifest")
    source = Path(manifest["source_directory"]).resolve()
    if (source == root or source.is_relative_to(root) or root.is_relative_to(source)
            or _hash(source / "metadata.json") != derivation["parent_metadata_sha256"]):
        raise ValueError("Support-ablation parent metadata hash mismatch")
    parent = _json(source / "metadata.json")
    if ("feature_transform" in parent or "derivation" in parent
            or parent.get("feature_names") != list(_legacy_feature_names())):
        raise ValueError("Support-ablation parent must be the original 70-column cache")
    expected = dict(parent)
    expected.update(feature_names=feature_schemas()["source_features"],
                    feature_transform=transform_spec(), derivation=derivation)
    if metadata != expected:
        raise ValueError("Derived cache changed metadata beyond the name-feature transform")
    assets = _asset_names(parent)
    shards = parent["train_shards"] + parent["valid_shards"]
    if any(set(manifest.get(key, {})) != set(assets)
           for key in ("source_file_sha256", "derived_file_sha256")):
        raise ValueError("Support-ablation derivation asset list mismatch")
    if (set(manifest.get("unchanged_arrays_sha256", {})) != set(shards)
            or any(not _valid_hash(value) for value in manifest["unchanged_arrays_sha256"].values())):
        raise ValueError("Support-ablation derivation is missing unchanged-array proofs")
    counts = {key: parent["counts"][key]
              for key in ("train_pairs", "train_groups", "valid_pairs", "valid_groups")}
    if manifest.get("counts") != counts:
        raise ValueError("Support-ablation derivation counts mismatch")
    protected = {str(root / "metadata.json"): metadata_hash,
        str(source / "metadata.json"): derivation["parent_metadata_sha256"],
        str(manifest_path): derivation["manifest_sha256"]}
    for relative in assets:
        old = manifest["source_file_sha256"][relative]
        new = manifest["derived_file_sha256"][relative]
        if not _valid_hash(old) or not _valid_hash(new):
            raise ValueError("Invalid support-ablation asset hash")
        if relative not in shards and old != new:
            raise ValueError("Support-ablation transform changed a non-feature asset")
        for directory, expected_hash in ((source, old), (root, new)):
            path = _inside(directory, relative)
            if _hash(path) != expected_hash:
                raise ValueError(f"Support-ablation cache asset hash mismatch: {path}")
            protected[str(path)] = expected_hash
    for relative in ("querylog.csv", "partitions.csv"):
        if metadata["training_augmentation_assets"][relative] != manifest["derived_file_sha256"][relative]:
            raise ValueError("Augmentation audit asset disagrees with the derivation")
    index_path = _inside(root, metadata.get("index_path", "index.joblib"))
    if protected[str(index_path)] != metadata.get("retrieval_index_sha256"):
        raise ValueError("Support-ablation cache search index differs from retrieval provenance")
    return {"verified": True, "cache_directory": str(root), "source_directory": str(source),
        "cache_metadata_sha256": metadata_hash, "manifest_sha256": derivation["manifest_sha256"],
        "index_sha256": protected[str(index_path)], "verified_assets": len(assets),
        "verified_shards": len(shards), "counts": counts,
        "verification": "hash-bound derivation manifest and streamed SHA256 of every original/derived asset",
        "same_candidates_features_labels_query_mass": True,
        "seconds": time.perf_counter() - started}, protected


def preflight_support_training(cache_directory, baseline_directory, *, expected_fold=None):
    """Optional read-only preflight binding an 82-column run to its existing control."""
    directory = Path(baseline_directory).resolve()
    summary = _json(directory / "training_summary.json")
    _validate_scope(summary, expected_fold)
    check_support_ablation_metadata(summary["dataset"])
    _check_selection(summary, "baseline86")
    from .ranker_commands import _check_training_weighting

    _check_training_weighting(summary)
    proof, protected = _verify_shared_cache(summary, cache_directory)
    for name in _MODEL_FILES:
        path = directory / name
        protected[str(path)] = _hash(path)
    if protected[str(directory / "index.joblib")] != proof["index_sha256"]:
        raise ValueError("Baseline saved index differs from the reused cache")
    return {"summary": summary, "cache_proof": proof, "protected_files": protected}


def _load_experiment(directory, train_hash, catalog_hash):
    from .ranker_commands import RankedMatcher

    started = time.perf_counter()
    directory = Path(directory).resolve()
    summary = _json(directory / "training_summary.json")
    _validate_scope(summary, None)
    check_support_ablation_metadata(summary["dataset"])
    _check_selection(summary, "no_support82")
    provenance = summary["dataset"].get("provenance", {})
    if provenance.get("train_sha256") != train_hash or provenance.get("catalog_sha256") != catalog_hash:
        raise ValueError("Public train/catalog content differs from experiment provenance")
    index_hash = _verify_index(summary, directory)
    matcher = RankedMatcher(directory)
    if (matcher.ranker.feature_names != feature_schemas()["experiment_features"]
            or asdict(matcher.ranker.config) != asdict(RankerConfig())):
        raise ValueError("Native experiment must retain exactly 82 features and the frozen ranker configuration")
    return matcher, summary, {"model_directory": str(directory),
        "model_sha256": _hash(directory / "model.txt"), "index_sha256": index_hash,
        "training_summary_sha256": _hash(directory / "training_summary.json"),
        "load_and_integrity_seconds": time.perf_counter() - started}


def pipeline_fingerprints(runners):
    """Cover learned values, search arrays, selected columns and each model's weights."""
    if set(runners) != set(RUNNERS):
        raise ValueError("Support ablation requires exactly baseline86 and no_support82")
    result = {}
    for name, runner in runners.items():
        pipeline = runner.pipeline
        if (pipeline.retriever.memory is not pipeline.memory
                or pipeline.feature_builder.category_model is not pipeline.categories):
            raise ValueError("Support-ablation pipeline components disagree on learned assets")
        result[name] = {
            "learned_assets_sha256": learned_assets_fingerprint(pipeline),
            "retriever_joblib_sha1": joblib.hash(pipeline.retriever, hash_name="sha1"),
            "catalog_feature_assets_joblib_sha1": joblib.hash({key: value for key, value
                in vars(pipeline.feature_builder).items() if key != "category_model"}, hash_name="sha1"),
            "catalog_ids_joblib_sha1": joblib.hash(runner.catalog_ids, hash_name="sha1"),
            "selection_sha256": _digest(runner.selection.to_dict()),
            "feature_transform_sha256": _digest(runner.feature_transform),
            "ranker_config_sha256": _digest(asdict(runner.ranker.config)),
            "ranker_feature_names_sha256": _digest(runner.ranker.feature_names),
            "ranker_weights_sha256": hashlib.sha256(
                runner.ranker._booster.model_to_string().encode("utf-8")).hexdigest(),
        }
    return result


def assert_assets_unchanged(context):
    current = pipeline_fingerprints(context["runners"])
    if current != context["asset_fingerprints"]:
        raise ValueError("Support-ablation learned assets, selected features or fitted weights changed")
    files = assert_unchanged(context["protected_files"])
    return {"verified": True, "runners": len(RUNNERS), "files": files["files"],
            "models_and_full_T_assets_unchanged": True, "shared_cache_unchanged": True}


def prepare_support_ablation(baseline_directory, experiment_directory, train_path,
                             catalog_path, *, expected_fold=None):
    """Load the two retrained models and prove identical cache and full-T assets."""
    directories = {name: Path(path).resolve() for name, path in zip(
        RUNNERS, (baseline_directory, experiment_directory))}
    if len(set(directories.values())) != 2:
        raise ValueError("Baseline and experiment model directories must be separate")
    train_path, catalog_path = Path(train_path).resolve(), Path(catalog_path).resolve()
    protected = {str(path): _hash(path) for path in (train_path, catalog_path)}
    summaries = {name: _json(path / "training_summary.json") for name, path in directories.items()}
    for name, summary in summaries.items():
        _validate_scope(summary, expected_fold)
        check_support_ablation_metadata(summary["dataset"])
        _check_selection(summary, name)
        for filename in _MODEL_FILES:
            path = directories[name] / filename
            protected[str(path)] = _hash(path)
    baseline, experiment = (summaries[name] for name in RUNNERS)
    if baseline["dataset"] != experiment["dataset"]:
        raise ValueError("Support ablation requires identical cache metadata for both models")
    if (baseline.get("cache_metadata_sha256") != experiment.get("cache_metadata_sha256")
            or Path(baseline.get("cache_directory", "")).resolve()
               != Path(experiment.get("cache_directory", "")).resolve()):
        raise ValueError("Support ablation requires the exact same cache path and metadata hash")
    from .ranker_commands import _check_training_weighting

    for summary in summaries.values():
        _check_training_weighting(summary)
    if baseline.get("training_weighting") != experiment.get("training_weighting"):
        raise ValueError("Support ablation requires identical source/language/candidate weighting")
    cache_proof, cache_files = _verify_shared_cache(baseline)
    protected.update(cache_files)
    index_hashes = {protected[str(path / "index.joblib")] for path in directories.values()}
    if index_hashes != {cache_proof["index_sha256"]}:
        raise ValueError("Both saved models must reuse the exact same cache search index")
    runners, provenance = {}, {}
    for name, loader in zip(RUNNERS, (_load_model, _load_experiment)):
        runner, loaded_summary, model_proof = loader(directories[name],
            protected[str(train_path)], protected[str(catalog_path)])
        if loaded_summary != summaries[name]:
            raise ValueError("Saved model summary changed while loading")
        runners[name], provenance[name] = runner, model_proof
    train, catalog = read_queries(train_path, labeled=True), read_catalog(catalog_path)
    heldout = _validate_partition(runners["baseline86"], baseline, train, catalog)
    other_heldout = _validate_partition(runners["no_support82"], experiment, train, catalog)
    if not heldout.equals(other_heldout):
        raise ValueError("Support-ablation validation populations differ")
    data = baseline["dataset"]
    assignments = make_folds(train, "vessel", data["outer_folds"], data["seed"])
    fitting = train.loc[assignments != data["outer_fold"]].copy()
    if data.get("counts", {}).get("selected_training_queries") != len(fitting):
        raise ValueError("Support ablation requires the complete outer training population")
    full_ids, heldout_ids = (set(frame.sample_id.astype(str)) for frame in (fitting, heldout))
    full_vessels, heldout_vessels = (set(frame.target_vessel_id.astype(str)) for frame in (fitting, heldout))
    if not full_ids or not heldout_ids or full_ids & heldout_ids or full_vessels & heldout_vessels:
        raise ValueError("Support-ablation training/validation membership leaks or is empty")
    started = time.perf_counter()
    expected = SimpleNamespace(memory=TrainingMemory().fit(fitting, catalog),
        categories=CategoryCompatibility().fit(fitting, catalog))
    for runner in runners.values():
        _validate_training_assets(runner.pipeline, expected)
    _validate_training_assets(runners["baseline86"].pipeline, runners["no_support82"].pipeline)
    asset_seconds = time.perf_counter() - started
    membership = train[["sample_id", "target_vessel_id"]].copy()
    membership["outer_fold"] = assignments
    membership["role"] = np.where(assignments == data["outer_fold"], "valid", "train")
    membership["search100"] = membership.role.eq("train")
    membership["category100"] = membership.role.eq("train")
    membership["has_hangul"] = train.vessel_name.map(has_hangul)
    counts = lambda frame: {"rows": len(frame), "vessels": int(frame.target_vessel_id.nunique()),
        "hangul_rows": int(frame.vessel_name.map(has_hangul).sum()),
        "non_hangul_rows": int((~frame.vessel_name.map(has_hangul)).sum())}
    coverage = {"fold": data["outer_fold"], "folds": 5, "split": "vessel", "seed": data["seed"],
                "full": counts(fitting), "heldout": counts(heldout)}
    startup = {"cache_integrity_seconds": cache_proof["seconds"],
        "full_training_assets_verification_seconds": asset_seconds,
        "model_load_and_integrity_seconds": {name: proof["load_and_integrity_seconds"]
                                             for name, proof in provenance.items()}}
    proof = {"name": "support_log4_ablation_v1", "verified": True,
        "coverage": coverage, "schemas": feature_schemas(), "cache": cache_proof,
        "training_weighting": baseline["training_weighting"], "ranker": baseline["ranker"],
        "model_provenance": provenance, "startup": startup,
        "integrity": {"heldout_rows_excluded": True, "heldout_vessels_excluded": True,
            "search_scope": "full outer training T (100%)", "category_scope": "full outer training T (100%)",
            "full_training_assets_recomputed_and_equal": True, "same_saved_index_bytes": True,
            "same_cache_directory_and_metadata": True, "same_training_weighting": True,
            "full_sample_ids_sha256": _ids_digest(full_ids),
            "heldout_sample_ids_sha256": _ids_digest(heldout_ids),
            "full_vessels_sha256": _ids_digest(full_vessels), "protected_files": protected},
        "interpretation": "Retrained 86 versus 82 selected features using identical mass-0.5 OOF pairs and full-T search/categories; no validation-driven refit or subset index."}
    context = {"runners": runners, "summaries": summaries, "heldout": heldout,
        "membership": membership, "proof": proof, "protected_files": protected,
        "startup": startup, "coverage": coverage}
    context["asset_fingerprints"] = pipeline_fingerprints(runners)
    return context
