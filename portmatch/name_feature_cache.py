"""Append versioned, label-independent name features to original OOF caches.

The original 70 columns, query groups, candidate order and sampled negatives are
kept byte-for-byte. New features only read the query's raw name and fixed catalog
names, so no outer-training alias index is applied to inner training queries.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from .name_features import FEATURE_NAMES, NameFeatureComputer


TRANSFORM_NAME = "token_phonetic_v1"


def _hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


@lru_cache(maxsize=1)
def _legacy_feature_names():
    """Read the legacy schema without fitting any real query/label statistics."""
    from .features import CategoryCompatibility, PairFeatureBuilder

    catalog = pd.DataFrame(columns=["vessel_id", "vessel_name", "gross_tonnage", "flag_name", "vessel_type"])
    train = pd.DataFrame(columns=["target_vessel_id", "flag_name", "vessel_type"])
    builder = PairFeatureBuilder(catalog, CategoryCompatibility().fit(train, catalog))
    empty = pd.DataFrame({"catalog_index": pd.Series(dtype=np.int64)})
    names = tuple(builder.transform({}, empty).columns)
    if len(names) != 70 or "retrieval_tonnage_rank" not in names:
        raise ValueError("Name feature transform requires the original 70-column schema")
    return names


def transform_spec():
    digest = hashlib.sha256()
    for filename in ("name_feature_cache.py", "name_features.py"):
        digest.update(filename.encode())
        digest.update(Path(__file__).with_name(filename).read_bytes())
    return {"name": TRANSFORM_NAME, "appended_features": list(FEATURE_NAMES),
            "code_sha256": digest.hexdigest()}


def check_transform_metadata(metadata):
    names = metadata.get("feature_names", [])
    spec = metadata.get("feature_transform")
    if spec is None:
        if set(FEATURE_NAMES).intersection(names) or "derivation" in metadata:
            raise ValueError("Transformed feature schema is missing its transform metadata")
        return
    if spec != transform_spec():
        raise ValueError("Unknown or changed name feature transform; regenerate the derived cache")
    if names != list(_legacy_feature_names()) + list(FEATURE_NAMES):
        raise ValueError("Transformed feature schema disagrees with transform metadata")
    if not isinstance(metadata.get("derivation"), dict):
        raise ValueError("Transformed cache is missing derivation metadata")


def _indices(values, size):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= size):
        raise ValueError("Invalid catalog indices for name feature transform")
    return values.astype(np.int64, copy=False)


def _new_features(computer, query_name, indices):
    values = computer.compute(query_name, indices)
    if values.shape != (len(indices), len(FEATURE_NAMES)) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Name feature computer returned an invalid feature matrix")
    return values


def transform_features(pipeline, row, candidates, features):
    """Apply identical name features to one independent inference query."""
    if tuple(features.columns) != _legacy_feature_names():
        raise ValueError("Name transform requires the original 70 feature columns in order")
    if len(candidates) != len(features) or not candidates.index.equals(features.index):
        raise ValueError("Candidates and features are misaligned")
    indices = _indices(candidates.catalog_index.to_numpy(), len(pipeline.catalog))
    computer = getattr(pipeline, "_token_phonetic_computer_v1", None)
    if computer is None:
        computer = NameFeatureComputer(pipeline.catalog.vessel_name.tolist())
        pipeline._token_phonetic_computer_v1 = computer
    added = _new_features(computer, row.get("vessel_name"), indices)
    return pd.concat([features, pd.DataFrame(added, columns=FEATURE_NAMES, index=features.index)], axis=1)


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Derived-cache asset path escapes its directory")
    return path


def _asset_names(metadata):
    names = [*metadata["train_shards"], *metadata["valid_shards"],
             metadata.get("index_path", "index.joblib"),
             metadata.get("catalog_ids_path", "catalog_ids.npy"), "partitions.csv", "querylog.csv"]
    if len(set(names)) != len(names) or "metadata.json" in names or "derivation.json" in names:
        raise ValueError("Invalid or duplicate cache asset paths")
    return names


def _invariant_digest(shard, columns):
    """Hash every source array and only the unchanged prefix of expanded X."""
    digest = hashlib.sha256()
    for key in sorted(shard):
        value = shard[key][:, :columns] if key == "X" else shard[key]
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def transform_cache(data, output, train_path, catalog_path):
    """Append features without refitting indices or changing negative sampling."""
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
    if "feature_transform" in metadata or "derivation" in metadata:
        raise ValueError("Use the original cache as the transformation source")
    _check_feature_code(metadata)
    names = metadata["feature_names"]
    if names != list(_legacy_feature_names()):
        raise ValueError("Source cache must contain the original 70-column feature schema")
    provenance = metadata.get("provenance", {})
    for field, path in (("train_sha256", train_path), ("catalog_sha256", catalog_path)):
        if not provenance.get(field) or _hash_file(path) != provenance[field]:
            raise ValueError(f"Input {field} differs from source cache provenance")
    if not provenance.get("feature_code_sha256"):
        raise ValueError("Source cache lacks feature-code provenance")
    train, catalog = read_queries(train_path, labeled=True), read_catalog(catalog_path)
    source_metadata_hash = _hash_file(source / "metadata.json")
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
    computer = NameFeatureComputer(catalog.vessel_name.tolist())
    query_names = train.set_index("sample_id").vessel_name.to_dict()
    targets = train.set_index("sample_id").target_vessel_id.to_dict()
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
                raise AssertionError("Cache transform changed an invariant array")
            target_path = _inside(destination, relative)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target_path, **transformed)
            destination_hashes[relative] = _hash_file(target_path)
            proofs[relative] = original_digest
            count_groups += len(shard["group"])
            count_pairs += len(shard["y"])
            print(f"Name transform {kind}: {number}/{len(metadata[kind + '_shards'])} shards, "
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
    if _hash_file(source / "metadata.json") != source_metadata_hash:
        raise ValueError("Source metadata changed during transformation")
    manifest = {"schema_version": 1, "complete": True, "transform": transform_spec(),
                "source_directory": str(source), "parent_metadata_sha256": source_metadata_hash,
                "source_file_sha256": source_hashes, "derived_file_sha256": destination_hashes,
                "unchanged_arrays_sha256": proofs, "counts": totals,
                "seconds": time.perf_counter() - started,
                "invariants": "All original 70 features and all other arrays byte-identical; "
                              "name features appended using only query name and fixed catalog names. "
                              "Candidate retrieval, OOF statistics and sampled negatives unchanged."}
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
    """Check actual parent/derived bytes and shared arrays before training/comparison."""
    from .training_data import iter_shards, read_metadata

    root = Path(experiment_cache).resolve()
    metadata = read_metadata(root)
    check_transform_metadata(metadata)
    if "feature_transform" not in metadata:
        raise ValueError("Comparison requires a recognized derived feature cache")
    derivation = metadata["derivation"]
    if (set(derivation) != {"manifest_path", "manifest_sha256", "parent_metadata_sha256"}
            or derivation["parent_metadata_sha256"] != parent_metadata_sha256):
        raise ValueError("Derived cache does not identify the expected baseline parent")
    manifest_path = _inside(root, derivation["manifest_path"])
    if _hash_file(manifest_path) != derivation["manifest_sha256"]:
        raise ValueError("Derivation manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or manifest.get("transform") != transform_spec()
            or manifest.get("parent_metadata_sha256") != parent_metadata_sha256):
        raise ValueError("Unsupported or mismatching derivation manifest")
    source = Path(manifest["source_directory"]).resolve()
    if (source == root or source.is_relative_to(root) or root.is_relative_to(source)
            or _hash_file(source / "metadata.json") != parent_metadata_sha256):
        raise ValueError("Parent metadata hash mismatch")
    parent = read_metadata(source)
    if "feature_transform" in parent or "derivation" in parent:
        raise ValueError("Derivation parent must be an original cache")
    if parent["feature_names"] != list(_legacy_feature_names()):
        raise ValueError("Derivation parent has a different feature schema")
    if expected_parent_dataset is not None and parent != expected_parent_dataset:
        raise ValueError("Derivation parent metadata differs from baseline model")
    expected = dict(parent)
    expected["feature_names"] = parent["feature_names"] + list(FEATURE_NAMES)
    expected.update(feature_transform=transform_spec(), derivation=derivation)
    if metadata != expected:
        raise ValueError("Derived cache changed metadata beyond the declared feature transform")
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
                raise ValueError("Derived name features must be finite")
    return {"verified": True, "transform": TRANSFORM_NAME,
            "source_directory": str(source), "derived_directory": str(root),
            "parent_metadata_sha256": parent_metadata_sha256,
            "derived_metadata_sha256": _hash_file(root / "metadata.json"),
            "manifest_sha256": derivation["manifest_sha256"],
            "verified_assets": len(assets), "verified_shards": len(shards),
            "invariants": manifest["invariants"]}
