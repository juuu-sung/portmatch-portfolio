"""Append CH-alternative features to the immutable token/phonetic v1 cache.

The 87 parent columns, candidates, OOF statistics and sampled negatives stay
unchanged. Derivation checks follow both links back to the original cache.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from . import name_feature_cache as parent_transform
from .ch_variant_features import FEATURE_NAMES, ChVariantFeatureComputer
from .name_feature_cache import (
    _asset_names, _hash_file, _indices, _inside, _invariant_digest,
    _legacy_feature_names, _write_json,
)


TRANSFORM_NAME = "token_phonetic_ch_v1"


def _parent_feature_names():
    return list(_legacy_feature_names()) + list(parent_transform.FEATURE_NAMES)


def transform_spec():
    digest = hashlib.sha256()
    for filename in ("ch_variant_cache.py", "ch_variant_features.py"):
        digest.update(filename.encode())
        digest.update(Path(__file__).with_name(filename).read_bytes())
    return {"name": TRANSFORM_NAME, "appended_features": list(FEATURE_NAMES),
            "code_sha256": digest.hexdigest(),
            "parent_transform": parent_transform.transform_spec()}


def check_transform_metadata(metadata):
    if metadata.get("feature_transform") != transform_spec():
        raise ValueError("Unknown or changed CH feature transform; regenerate the derived cache")
    if metadata.get("feature_names") != _parent_feature_names() + list(FEATURE_NAMES):
        raise ValueError("CH feature schema disagrees with transform metadata")
    if not isinstance(metadata.get("derivation"), dict):
        raise ValueError("CH cache is missing derivation metadata")


def _check_parent(metadata):
    spec = metadata.get("feature_transform")
    if not isinstance(spec, dict) or spec.get("name") != parent_transform.TRANSFORM_NAME:
        raise ValueError("CH transform requires a token_phonetic_v1 cache as its immediate parent")
    parent_transform.check_transform_metadata(metadata)
    if metadata["feature_names"] != _parent_feature_names():
        raise ValueError("CH transform requires the 87-column token_phonetic_v1 schema")


def _new_features(computer, query_name, indices):
    values = computer.compute(query_name, indices)
    if (values.shape != (len(indices), len(FEATURE_NAMES)) or values.dtype != np.float32
            or not np.isfinite(values).all()):
        raise ValueError("CH feature computer returned an invalid feature matrix")
    return values


def transform_features(pipeline, row, candidates, features):
    """Build both immutable parent and CH columns from a legacy inference frame."""
    expanded = parent_transform.transform_features(pipeline, row, candidates, features)
    indices = _indices(candidates.catalog_index.to_numpy(), len(pipeline.catalog))
    computer = getattr(pipeline, "_token_phonetic_ch_computer_v1", None)
    if computer is None:
        computer = ChVariantFeatureComputer(pipeline.catalog.vessel_name.tolist())
        pipeline._token_phonetic_ch_computer_v1 = computer
    added = _new_features(computer, row.get("vessel_name"), indices)
    return pd.concat([expanded, pd.DataFrame(added, columns=FEATURE_NAMES, index=features.index)], axis=1)


def transform_cache(data, output, train_path, catalog_path):
    """Append CH columns to a verified v1 cache without refitting or resampling."""
    from .data import read_catalog, read_queries
    from .ranker_commands import _check_feature_code
    from .training_data import iter_shards, read_metadata

    started = time.perf_counter()
    source, destination = Path(data).resolve(), Path(output).resolve()
    if (source == destination or destination.is_relative_to(source)
            or source.is_relative_to(destination)):
        raise ValueError("Source and derived cache directories must be separate")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Choose a new or empty dataset directory: {destination}")
    metadata = read_metadata(source)
    _check_parent(metadata)
    _check_feature_code(metadata)
    provenance = metadata.get("provenance", {})
    for field, path in (("train_sha256", train_path), ("catalog_sha256", catalog_path)):
        if not provenance.get(field) or _hash_file(path) != provenance[field]:
            raise ValueError(f"Input {field} differs from source cache provenance")
    if not provenance.get("feature_code_sha256"):
        raise ValueError("Source cache lacks feature-code provenance")
    parent_proof = parent_transform.verify_derivation(
        source, metadata["derivation"]["parent_metadata_sha256"])
    source_metadata_hash = _hash_file(source / "metadata.json")
    train, catalog = read_queries(train_path, labeled=True), read_catalog(catalog_path)
    assets = _asset_names(metadata)
    source_hashes = {name: _hash_file(_inside(source, name)) for name in assets}
    pipeline = joblib.load(_inside(source, metadata.get("index_path", "index.joblib")))
    try:
        pd.testing.assert_frame_equal(pipeline.catalog, catalog)
    except AssertionError as error:
        raise ValueError("Saved index catalog differs from source catalog") from error
    if asdict(pipeline.retriever.config) != metadata["retrieval"]:
        raise ValueError("Saved retrieval settings differ from cache metadata")
    catalog_ids = np.load(_inside(source, metadata.get("catalog_ids_path", "catalog_ids.npy")), allow_pickle=False)
    if not np.array_equal(catalog_ids, catalog.vessel_id.astype(str)):
        raise ValueError("Saved catalog IDs differ from source catalog")
    computer = ChVariantFeatureComputer(catalog.vessel_name.tolist())
    query_names = train.set_index("sample_id").vessel_name.to_dict()
    targets = train.set_index("sample_id").target_vessel_id.to_dict()
    names = metadata["feature_names"]
    original_columns = len(names)
    destination.mkdir(parents=True, exist_ok=True)
    destination_hashes, proofs, totals = {}, {}, {}
    for kind in ("train", "valid"):
        count_pairs = count_groups = 0
        for number, (relative, shard) in enumerate(zip(metadata[kind + "_shards"], iter_shards(source, kind)), 1):
            transformed = dict(shard)
            expanded = np.empty((len(shard["X"]), original_columns + len(FEATURE_NAMES)), dtype=np.float32)
            expanded[:, :original_columns] = shard["X"]
            offset = 0
            for q, group in enumerate(shard["group"]):
                sample_id = str(shard["sample_id"][q])
                if sample_id not in query_names:
                    raise ValueError(f"Cached sample_id missing from source queries: {sample_id}")
                target = int(shard["target_catalog_index"][q])
                if target < 0 or target >= len(catalog_ids) or targets[sample_id] != catalog_ids[target]:
                    raise ValueError("Cached target differs from original training data")
                section = slice(offset, offset + int(group))
                indices = _indices(shard["catalog_index"][section], len(catalog_ids))
                expanded[section, original_columns:] = _new_features(computer, query_names[sample_id], indices)
                offset += int(group)
            transformed["X"] = expanded
            original_digest = _invariant_digest(shard, original_columns)
            if original_digest != _invariant_digest(transformed, original_columns):
                raise AssertionError("CH cache transform changed an invariant array")
            target_path = _inside(destination, relative)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target_path, **transformed)
            destination_hashes[relative] = _hash_file(target_path)
            proofs[relative] = original_digest
            count_groups += len(shard["group"])
            count_pairs += len(shard["y"])
            print(f"CH transform {kind}: {number}/{len(metadata[kind + '_shards'])} shards, "
                  f"{count_groups:,} groups, {count_pairs:,} pairs", flush=True)
        if (count_pairs != metadata["counts"][kind + "_pairs"]
                or count_groups != metadata["counts"][kind + "_groups"]):
            raise ValueError("Shard totals disagree with source metadata")
        totals[kind + "_pairs"], totals[kind + "_groups"] = count_pairs, count_groups
    for relative in assets:
        if relative not in destination_hashes:
            target_path = _inside(destination, relative)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_inside(source, relative), target_path)
            destination_hashes[relative] = _hash_file(target_path)
        if _hash_file(_inside(source, relative)) != source_hashes[relative]:
            raise ValueError("Source cache changed during transformation")
    if (_hash_file(source / "metadata.json") != source_metadata_hash
            or _hash_file(_inside(source, metadata["derivation"]["manifest_path"]))
            != parent_proof["manifest_sha256"]):
        raise ValueError("Source metadata or derivation changed during transformation")
    manifest = {"schema_version": 1, "complete": True, "transform": transform_spec(),
                "source_directory": str(source), "parent_metadata_sha256": source_metadata_hash,
                "parent_derivation": parent_proof,
                "source_file_sha256": source_hashes, "derived_file_sha256": destination_hashes,
                "unchanged_arrays_sha256": proofs, "counts": totals,
                "seconds": time.perf_counter() - started,
                "invariants": "All 87 parent features and all other arrays byte-identical; "
                              "CH features appended using only query name and fixed catalog names. "
                              "Candidate retrieval, OOF statistics and sampled negatives unchanged. "
                              "Parent token_phonetic_v1 derivation verified back to the original cache."}
    _write_json(destination / "derivation.json", manifest)
    derived = dict(metadata)
    derived["feature_names"] = names + list(FEATURE_NAMES)
    derived["feature_transform"] = transform_spec()
    derived["derivation"] = {"manifest_path": "derivation.json",
                             "manifest_sha256": _hash_file(destination / "derivation.json"),
                             "parent_metadata_sha256": source_metadata_hash}
    _write_json(destination / "metadata.json", derived)
    return derived


def verify_derivation(experiment_cache, parent_metadata_sha256, expected_parent_dataset=None):
    """Verify unchanged parent columns and the complete original -> v1 -> CH chain."""
    from .training_data import iter_shards, read_metadata

    root = Path(experiment_cache).resolve()
    metadata = read_metadata(root)
    check_transform_metadata(metadata)
    derivation = metadata["derivation"]
    if (set(derivation) != {"manifest_path", "manifest_sha256", "parent_metadata_sha256"}
            or derivation["parent_metadata_sha256"] != parent_metadata_sha256):
        raise ValueError("CH cache does not identify the expected baseline parent")
    manifest_path = _inside(root, derivation["manifest_path"])
    if _hash_file(manifest_path) != derivation["manifest_sha256"]:
        raise ValueError("CH derivation manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or manifest.get("transform") != transform_spec()
            or manifest.get("parent_metadata_sha256") != parent_metadata_sha256):
        raise ValueError("Unsupported or mismatching CH derivation manifest")
    source = Path(manifest["source_directory"]).resolve()
    if (source == root or source.is_relative_to(root) or root.is_relative_to(source)
            or _hash_file(source / "metadata.json") != parent_metadata_sha256):
        raise ValueError("Parent metadata hash mismatch")
    parent = read_metadata(source)
    _check_parent(parent)
    if expected_parent_dataset is not None and parent != expected_parent_dataset:
        raise ValueError("Derivation parent metadata differs from baseline model")
    expected = dict(parent)
    expected["feature_names"] = parent["feature_names"] + list(FEATURE_NAMES)
    expected.update(feature_transform=transform_spec(), derivation=derivation)
    if metadata != expected:
        raise ValueError("Derived cache changed metadata beyond the declared feature transform")
    parent_proof = parent_transform.verify_derivation(
        source, parent["derivation"]["parent_metadata_sha256"])
    if manifest.get("parent_derivation") != parent_proof:
        raise ValueError("CH derivation does not match its verified parent derivation")
    assets = _asset_names(parent)
    shards = parent["train_shards"] + parent["valid_shards"]
    for field in ("source_file_sha256", "derived_file_sha256"):
        if set(manifest.get(field, {})) != set(assets):
            raise ValueError("Derivation asset list differs from parent metadata")
    if set(manifest.get("unchanged_arrays_sha256", {})) != set(shards):
        raise ValueError("Derivation is missing invariant shard proofs")
    counts = {key: parent["counts"][key] for key in ("train_pairs", "train_groups", "valid_pairs", "valid_groups")}
    if manifest.get("counts") != counts:
        raise ValueError("Derivation counts differ from parent metadata")
    for relative in assets:
        old_hash = manifest["source_file_sha256"][relative]
        new_hash = manifest["derived_file_sha256"][relative]
        if (_hash_file(_inside(source, relative)) != old_hash
                or _hash_file(_inside(root, relative)) != new_hash):
            raise ValueError(f"Derived cache asset hash mismatch: {relative}")
        if relative not in shards and old_hash != new_hash:
            raise ValueError("Derived cache changed a non-feature asset")
    columns = len(parent["feature_names"])
    for kind in ("train", "valid"):
        for relative, old, new in zip(parent[kind + "_shards"], iter_shards(source, kind), iter_shards(root, kind)):
            expected_digest = manifest["unchanged_arrays_sha256"][relative]
            if (_invariant_digest(old, columns) != expected_digest
                    or _invariant_digest(new, columns) != expected_digest):
                raise ValueError(f"Derived cache changed invariant arrays: {relative}")
            if not np.isfinite(new["X"][:, columns:]).all():
                raise ValueError("Derived CH features must be finite")
    return {"verified": True, "transform": TRANSFORM_NAME,
            "source_directory": str(source), "derived_directory": str(root),
            "parent_metadata_sha256": parent_metadata_sha256,
            "derived_metadata_sha256": _hash_file(root / "metadata.json"),
            "manifest_sha256": derivation["manifest_sha256"],
            "verified_assets": len(assets), "verified_shards": len(shards),
            "parent_derivation": parent_proof, "invariants": manifest["invariants"]}
