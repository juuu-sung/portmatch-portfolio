"""Versioned, label-independent tonnage tie ranks over the complete numeric route.

Legacy retrieval/features deliberately remain unchanged.  A derived cache replaces
one column, preserves the sampled pairs, and carries a verifiable parent lineage.
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

from .retrieval import _positive_number


SOURCE_FEATURE = "retrieval_tonnage_rank"
TARGET_FEATURE = "retrieval_tonnage_min_rank"
TRANSFORM_NAME = "tonnage_min_rank_v1"


def _hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def transform_spec():
    return {"name": TRANSFORM_NAME, "source_feature": SOURCE_FEATURE,
            "target_feature": TARGET_FEATURE, "code_sha256": _hash_file(__file__)}


def check_transform_metadata(metadata):
    """Legacy schemas stay valid; derived schemas require the exact transform."""
    names = metadata.get("feature_names", [])
    spec = metadata.get("feature_transform")
    if spec is None:
        if TARGET_FEATURE in names or "derivation" in metadata:
            raise ValueError("Transformed feature schema is missing its transform metadata")
        return
    if spec != transform_spec():
        raise ValueError("Unknown or changed feature transform; regenerate the derived cache")
    if names.count(TARGET_FEATURE) != 1 or SOURCE_FEATURE in names:
        raise ValueError("Transformed feature schema disagrees with transform metadata")
    if not isinstance(metadata.get("derivation"), dict):
        raise ValueError("Transformed cache is missing derivation metadata")


def numeric_ranks(retriever, row, catalog_indices):
    """Return old ordinals and min ranks, computed before any candidate sampling.

The existing numeric search defines the route including fallback-K boundaries.
Its ranking anchor switches from IGT to GT only if the GT window is nonempty.
No labels, aliases, category statistics, or other queries enter this calculation.
"""
    indices = np.asarray(catalog_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= retriever.size):
        raise ValueError("Invalid candidate indices for tonnage rank transform")
    selected = retriever._numeric_search(row)
    old = np.zeros(retriever.size, dtype=np.int64)
    tied = np.zeros(retriever.size, dtype=np.int64)
    if len(selected):
        international = _positive_number(row.get("international_gross_tonnage"))
        gross = _positive_number(row.get("gross_tonnage"))
        anchor = international if international is not None else gross
        if (not len(retriever._window(anchor)) and gross is not None and gross != anchor
                and len(retriever._window(gross))):
            anchor = gross
        gaps = np.abs(retriever._gt[selected] - anchor)
        if np.any(gaps[1:] < gaps[:-1]):
            raise ValueError("Numeric candidate order disagrees with tonnage distance")
        old[selected] = np.arange(1, len(selected) + 1)
        tied[selected] = np.searchsorted(gaps, gaps, side="left") + 1
    return old[indices], tied[indices]


def transform_features(pipeline, row, candidates, features):
    """Apply the same transformation used by caches to one inference query."""
    if (not features.columns.is_unique or SOURCE_FEATURE not in features
            or TARGET_FEATURE in features):
        raise ValueError("Expected the original tonnage rank feature exactly once")
    indices = candidates.catalog_index.to_numpy(dtype=np.int64)
    if len(indices) != len(features) or not candidates.index.equals(features.index):
        raise ValueError("Candidates and features are misaligned")
    old, tied = numeric_ranks(pipeline.retriever, row, indices)
    if not np.array_equal(features[SOURCE_FEATURE].to_numpy(), old):
        raise ValueError("Original tonnage ranks disagree with fixed numeric retrieval")
    if "retrieval_tonnage_hit" in features and not np.array_equal(
            features.retrieval_tonnage_hit.to_numpy(), old > 0):
        raise ValueError("Original tonnage hits disagree with fixed numeric retrieval")
    result = features.rename(columns={SOURCE_FEATURE: TARGET_FEATURE}).copy()
    result[TARGET_FEATURE] = tied.astype(features[SOURCE_FEATURE].dtype)
    return result


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


def _invariant_digest(shard, column):
    """Hash every array except the replaced feature, retaining shape and dtype."""
    digest = hashlib.sha256()
    for key in sorted(shard):
        value = np.delete(shard[key], column, axis=1) if key == "X" else shard[key]
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def transform_cache(data, output, train_path, catalog_path):
    """Derive a separate cache without rebuilding OOF indices or choosing negatives."""
    from .data import QUERY_COLUMNS, read_catalog, read_queries
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
    _check_feature_code(metadata)
    check_transform_metadata(metadata)
    if "feature_transform" in metadata:
        raise ValueError("Use the original cache as the transformation source")
    names = metadata["feature_names"]
    if names.count(SOURCE_FEATURE) != 1 or TARGET_FEATURE in names:
        raise ValueError("Source cache must contain the original tonnage rank feature")
    provenance = metadata.get("provenance", {})
    for field, path in (("train_sha256", train_path), ("catalog_sha256", catalog_path)):
        if not provenance.get(field) or _hash_file(path) != provenance[field]:
            raise ValueError(f"Input {field} differs from source cache provenance")
    if not provenance.get("feature_code_sha256"):
        raise ValueError("Source cache lacks feature-code provenance")
    train = read_queries(train_path, labeled=True)
    catalog = read_catalog(catalog_path)
    source_metadata_hash = _hash_file(source / "metadata.json")
    assets = _asset_names(metadata)
    source_hashes = {name: _hash_file(_inside(source, name)) for name in assets}
    pipeline = joblib.load(_inside(source, metadata.get("index_path", "index.joblib")))
    try:
        pd.testing.assert_frame_equal(pipeline.catalog, catalog)
    except AssertionError as error:
        raise ValueError("Saved index catalog differs from source catalog") from error
    if asdict(pipeline.retriever.config) != metadata["retrieval"]:
        raise ValueError("Saved numeric retrieval settings differ from cache metadata")
    catalog_ids = np.load(_inside(source, metadata.get("catalog_ids_path", "catalog_ids.npy")), allow_pickle=False)
    if not np.array_equal(catalog_ids, catalog.vessel_id.astype(str)):
        raise ValueError("Saved catalog IDs differ from source catalog")
    queries = {row["sample_id"]: {key: row.get(key) for key in QUERY_COLUMNS}
               for row in train.to_dict("records")}
    targets = train.set_index("sample_id").target_vessel_id.to_dict()
    column = names.index(SOURCE_FEATURE)
    hit_column = names.index("retrieval_tonnage_hit") if "retrieval_tonnage_hit" in names else None
    destination.mkdir(parents=True, exist_ok=True)
    destination_hashes, proofs, totals = {}, {}, {}
    for kind in ("train", "valid"):
        count_pairs = count_groups = 0
        for number, (relative, shard) in enumerate(zip(metadata[kind + "_shards"], iter_shards(source, kind)), 1):
            transformed = {key: value.copy() for key, value in shard.items()}
            offset = 0
            for q, group in enumerate(shard["group"]):
                sample_id = str(shard["sample_id"][q])
                if sample_id not in queries:
                    raise ValueError(f"Cached sample_id missing from source queries: {sample_id}")
                target = int(shard["target_catalog_index"][q])
                if target < 0 or target >= len(catalog_ids) or targets[sample_id] != catalog_ids[target]:
                    raise ValueError("Cached target differs from original training data")
                section = slice(offset, offset + int(group))
                old, tied = numeric_ranks(pipeline.retriever, queries[sample_id], shard["catalog_index"][section])
                if not np.array_equal(shard["X"][section, column], old):
                    raise ValueError(f"Cached tonnage ordinal differs from fixed numeric retrieval: {sample_id}")
                if hit_column is not None and not np.array_equal(shard["X"][section, hit_column], old > 0):
                    raise ValueError("Cached tonnage hit differs from fixed numeric retrieval")
                transformed["X"][section, column] = tied
                offset += int(group)
            original_digest = _invariant_digest(shard, column)
            if original_digest != _invariant_digest(transformed, column):
                raise AssertionError("Cache transform changed an invariant array")
            target_path = _inside(destination, relative)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target_path, **transformed)
            destination_hashes[relative] = _hash_file(target_path)
            proofs[relative] = original_digest
            count_groups += len(shard["group"])
            count_pairs += len(shard["y"])
            print(f"Transform {kind}: {number}/{len(metadata[kind + '_shards'])} shards, "
                  f"{count_groups:,} groups", flush=True)
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
                "invariants": "All arrays and shared features byte-identical; only one rank column replaced. "
                              "Ranks computed from complete numeric route before existing sampling."}
    _write_json(destination / "derivation.json", manifest)
    derived = dict(metadata)
    derived["feature_names"] = [TARGET_FEATURE if name == SOURCE_FEATURE else name for name in names]
    derived["feature_transform"] = transform_spec()
    derived["derivation"] = {"manifest_path": "derivation.json",
                             "manifest_sha256": _hash_file(destination / "derivation.json"),
                             "parent_metadata_sha256": source_metadata_hash}
    _write_json(destination / "metadata.json", derived)
    return derived


def verify_derivation(experiment_cache, parent_metadata_sha256, expected_parent_dataset=None):
    """Fail closed unless a derived cache has the recorded, intact parent/assets.

This reads file bytes but does not regenerate candidate indices or refit models.
The source cache must remain available to establish lineage to the baseline.
"""
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
    if source == root or _hash_file(source / "metadata.json") != parent_metadata_sha256:
        raise ValueError("Parent metadata hash mismatch")
    parent = read_metadata(source)
    check_transform_metadata(parent)
    if "feature_transform" in parent:
        raise ValueError("Derivation parent must be an original cache")
    if expected_parent_dataset is not None and parent != expected_parent_dataset:
        raise ValueError("Derivation parent metadata differs from baseline model")
    expected = dict(parent)
    expected["feature_names"] = [TARGET_FEATURE if name == SOURCE_FEATURE else name
                                 for name in parent["feature_names"]]
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
    # Recompute rather than merely trusting the conversion-time proof. This
    # explicitly establishes identical selected negatives and validation IDs,
    # labels, groups, query order, and all 69 shared feature values.
    column = parent["feature_names"].index(SOURCE_FEATURE)
    for kind in ("train", "valid"):
        for relative, old, new in zip(parent[kind + "_shards"], iter_shards(source, kind), iter_shards(root, kind)):
            expected_digest = manifest["unchanged_arrays_sha256"][relative]
            if (_invariant_digest(old, column) != expected_digest
                    or _invariant_digest(new, column) != expected_digest):
                raise ValueError(f"Derived cache changed invariant arrays: {relative}")
    return {"verified": True, "transform": TRANSFORM_NAME,
            "source_directory": str(source), "derived_directory": str(root),
            "parent_metadata_sha256": parent_metadata_sha256,
            "derived_metadata_sha256": _hash_file(root / "metadata.json"),
            "manifest_sha256": derivation["manifest_sha256"],
            "verified_assets": len(assets), "verified_shards": len(shards),
            "invariants": manifest["invariants"]}
