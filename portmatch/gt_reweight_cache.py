"""Derive a GT-augmented cache by changing only its training query masses.

The feature-transform lineage is retained separately from this mass-only
lineage. Neither retrieval, fitting, sampling nor feature calculation runs here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from itertools import zip_longest
from pathlib import Path
import shutil
import time

import numpy as np

from .gt_augmentation import (
    GTMissingAugmentationConfig, VARIANTS, augmentation_spec,
    check_augmentation_metadata, expected_query_masses, query_identity,
    update_group_digest,
)


REWEIGHT_NAME = "gt_original_mass_reweight_v1"
MANIFEST_NAME = "reweighting.json"
_MASS_DIAGNOSTICS = ("source_mass_min", "source_mass_max", "total_query_mass",
                     "hangul_query_mass", "non_hangul_query_mass")


def _hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def _code_hash():
    return _hash_file(__file__)


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _inside(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Invalid reweight cache asset path")
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Reweight cache asset path escapes its directory")
    return path


def _separate(source, destination):
    if (source == destination or source.is_relative_to(destination)
            or destination.is_relative_to(source)):
        raise ValueError("Source and reweighted cache directories must be separate")


def _file_names(root):
    result = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Reweight cache assets must not be symbolic links")
        if path.is_file():
            result.append(path.relative_to(root).as_posix())
    return sorted(result)


def _expected_files(metadata):
    from .name_feature_cache import _asset_names

    names = [*_asset_names(metadata), "metadata.json", metadata["derivation"]["manifest_path"]]
    if len(set(names)) != len(names) or MANIFEST_NAME in names:
        raise ValueError("Invalid or duplicate reweight cache assets")
    return sorted(names)


def _check_source_metadata(metadata):
    from .name_feature_cache import check_transform_metadata, transform_spec

    if "training_reweighting" in metadata:
        raise ValueError("Use the original augmented cache; reweight chains are not supported")
    augmentation = check_augmentation_metadata(metadata)
    if augmentation is None:
        raise ValueError("Reweighting requires an augmented robust source cache")
    check_transform_metadata(metadata)
    if metadata.get("feature_transform") != transform_spec():
        raise ValueError("Reweighting requires the canonical token-phonetic 87-column cache")
    return augmentation


def check_reweighting_metadata(metadata):
    """Validate a model/cache declaration without requiring training files at inference."""
    proof = metadata.get("training_reweighting")
    if proof is None:
        return None
    required = {"name", "version", "code_sha256", "source_metadata_sha256",
                "source_original_mass", "target_original_mass", "manifest_path", "manifest_sha256"}
    if (not isinstance(proof, dict) or set(proof) != required
            or proof.get("name") != REWEIGHT_NAME or type(proof.get("version")) is not int
            or proof["version"] != 1 or proof.get("code_sha256") != _code_hash()
            or proof.get("manifest_path") != MANIFEST_NAME
            or not _sha(proof.get("source_metadata_sha256"))
            or not _sha(proof.get("manifest_sha256"))):
        raise ValueError("Unknown or changed training reweighting declaration")
    source_config = GTMissingAugmentationConfig(proof["source_original_mass"])
    target_config = GTMissingAugmentationConfig(proof["target_original_mass"])
    if source_config == target_config:
        raise ValueError("Reweighting must change original_mass")
    without_proof = dict(metadata)
    without_proof.pop("training_reweighting")
    actual_config = _check_source_metadata(without_proof)
    if actual_config != target_config:
        raise ValueError("Reweighting target mass differs from augmentation metadata")
    return proof


def validate_reweight_pair(source_metadata, target_metadata, source_metadata_sha256=None):
    """Bind a mass-only experiment to an exact control dataset declaration."""
    source_config = _check_source_metadata(source_metadata)
    proof = check_reweighting_metadata(target_metadata)
    if proof is None:
        raise ValueError("Experiment is missing its training reweighting declaration")
    if proof["source_original_mass"] != source_config.original_mass:
        raise ValueError("Reweighting source mass differs from control metadata")
    if (source_metadata_sha256 is not None
            and proof["source_metadata_sha256"] != source_metadata_sha256):
        raise ValueError("Reweighted experiment is not derived from this control cache")
    normalized = copy.deepcopy(target_metadata)
    normalized.pop("training_reweighting")
    normalized["training_augmentation"] = copy.deepcopy(source_metadata["training_augmentation"])
    normalized["training_augmentation_assets"]["training_groups_sha256"] = (
        source_metadata["training_augmentation_assets"]["training_groups_sha256"])
    old_diagnostics = source_metadata["training_augmentation_diagnostics"]
    diagnostics = normalized["training_augmentation_diagnostics"]
    diagnostics["variant_query_mass"] = copy.deepcopy(old_diagnostics["variant_query_mass"])
    for field in _MASS_DIAGNOSTICS:
        if not math.isclose(diagnostics[field], old_diagnostics[field], abs_tol=1e-10, rel_tol=0):
            raise ValueError("Reweighting changed source or language mass totals")
        diagnostics[field] = old_diagnostics[field]
    if normalized != source_metadata:
        raise ValueError("Reweighted cache changed controlled metadata beyond query masses")
    return proof


def _invariant_digest(shard):
    """Bitwise digest of every array except the intentionally changed query_mass."""
    digest = hashlib.sha256()
    for key in sorted(shard):
        if key == "query_mass":
            continue
        value = shard[key]
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _records(metadata, shard):
    views = ["vessel", "row"] if metadata["inner_split"] == "mixed" else [metadata["inner_split"]]
    for source, query, variant in zip(shard["sample_id"], shard["query_id"], shard["augmentation_variant"]):
        source, view = query_identity(str(source), str(query), str(variant), views)
        yield source, view, str(variant)


def _mass_diagnostics(records, languages, masses):
    source_totals = {}
    for identity in records:
        source = identity[0]
        source_totals[source] = source_totals.get(source, 0.0) + masses[identity]
    if not source_totals or not all(math.isclose(value, 1, abs_tol=1e-12, rel_tol=0)
                                    for value in source_totals.values()):
        raise ValueError("Reweighted source masses are not normalized")
    return {
        "source_mass_min": min(source_totals.values()),
        "source_mass_max": max(source_totals.values()),
        "total_query_mass": float(sum(source_totals.values())),
        "hangul_query_mass": float(sum(value for source, value in source_totals.items() if languages[source])),
        "non_hangul_query_mass": float(sum(value for source, value in source_totals.items() if not languages[source])),
        "variant_query_mass": {variant: float(sum(masses[identity] for identity in records
                                                   if identity[2] == variant)) for variant in VARIANTS},
    }


def reweight_cache(data, output, original_mass):
    """Copy a complete 87-column augmented cache, changing only train query_mass."""
    from .name_feature_cache import verify_derivation
    from .ranker_commands import _check_feature_code
    from .training_data import iter_shards, read_metadata

    started = time.perf_counter()
    config = GTMissingAugmentationConfig(original_mass)
    source, destination = Path(data).resolve(), Path(output).resolve()
    _separate(source, destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Choose a new or empty dataset directory: {destination}")
    metadata = read_metadata(source)
    source_config = _check_source_metadata(metadata)
    if config == source_config:
        raise ValueError("Reweighting must change original_mass")
    _check_feature_code(metadata)
    files = _expected_files(metadata)
    if _file_names(source) != files:
        raise ValueError("Source cache contains unexpected or missing assets")
    hashes = {name: _hash_file(_inside(source, name)) for name in files}
    print("Verifying source name-feature lineage and augmented arrays", flush=True)
    verify_derivation(source, metadata["derivation"]["parent_metadata_sha256"])
    records, languages = [], {}
    invariants, counts = {}, {"train_groups": 0, "train_pairs": 0, "valid_groups": 0, "valid_pairs": 0}
    for kind in ("train", "valid"):
        for position, shard in enumerate(iter_shards(source, kind)):
            relative = metadata[kind + "_shards"][position]
            invariants[relative] = _invariant_digest(shard)
            counts[kind + "_groups"] += len(shard["group"])
            counts[kind + "_pairs"] += len(shard["y"])
            if kind == "train":
                records.extend(_records(metadata, shard))
                for sample, language in zip(shard["sample_id"], shard["has_hangul"]):
                    languages[str(sample)] = bool(language)
    if any(counts[key] != metadata["counts"][key] for key in counts):
        raise ValueError("Source shard totals differ from metadata")
    masses = expected_query_masses(records, config)
    diagnostics = _mass_diagnostics(records, languages, masses)
    destination.mkdir(parents=True, exist_ok=True)
    output_hashes, group_digest = {}, hashlib.sha256()
    for number, shard in enumerate(iter_shards(source, "train"), 1):
        relative = metadata["train_shards"][number - 1]
        changed = dict(shard)
        changed["query_mass"] = np.asarray([masses[identity] for identity in _records(metadata, shard)], dtype=np.float64)
        if _invariant_digest(changed) != invariants[relative]:
            raise AssertionError("Reweighting changed an invariant array")
        update_group_digest(group_digest, changed)
        path = _inside(destination, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **changed)
        output_hashes[relative] = _hash_file(path)
        print(f"Reweight train: {number}/{len(metadata['train_shards'])} shards", flush=True)
    for relative in files:
        if relative == "metadata.json" or relative in output_hashes:
            continue
        path = _inside(destination, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_inside(source, relative), path)
        output_hashes[relative] = _hash_file(path)
        if output_hashes[relative] != hashes[relative]:
            raise ValueError("Reweighting changed a copied validation or audit asset")
    if _file_names(source) != files or any(_hash_file(_inside(source, name)) != hashes[name] for name in files):
        raise ValueError("Source cache changed during reweighting")
    manifest = {
        "schema_version": 1, "complete": True, "name": REWEIGHT_NAME, "code_sha256": _code_hash(),
        "source_directory": str(source), "source_metadata_sha256": hashes["metadata.json"],
        "source_original_mass": source_config.original_mass, "target_original_mass": config.original_mass,
        "source_file_sha256": hashes, "derived_file_sha256": output_hashes,
        "unchanged_arrays_sha256": invariants, "counts": counts,
        "seconds": time.perf_counter() - started,
        "invariants": "Only training query_mass changed. All other NPZ arrays bitwise identical; "
                      "validation shards, catalog, fitted index, partitions, query log and feature derivation copied byte-for-byte. "
                      "Source cache unchanged; no retrieval, fitting, feature calculation or sampling.",
    }
    _write_json(destination / MANIFEST_NAME, manifest)
    derived = copy.deepcopy(metadata)
    derived["training_augmentation"] = augmentation_spec(config)
    derived["training_augmentation_diagnostics"].update(diagnostics)
    derived["training_augmentation_assets"]["training_groups_sha256"] = group_digest.hexdigest()
    derived["training_reweighting"] = {
        "name": REWEIGHT_NAME, "version": 1, "code_sha256": _code_hash(),
        "source_metadata_sha256": hashes["metadata.json"],
        "source_original_mass": source_config.original_mass, "target_original_mass": config.original_mass,
        "manifest_path": MANIFEST_NAME, "manifest_sha256": _hash_file(destination / MANIFEST_NAME),
    }
    validate_reweight_pair(metadata, derived, hashes["metadata.json"])
    _write_json(destination / "metadata.json", derived)
    try:
        verify_reweighting(destination, hashes["metadata.json"], metadata)
    except Exception:
        derived["complete"] = False
        _write_json(destination / "metadata.json", derived)
        raise
    return derived


def verify_reweighting(directory, expected_source_metadata_sha256=None, expected_source_dataset=None):
    """Recheck declared and actual source/child bytes, then all non-mass arrays."""
    from .name_feature_cache import verify_derivation
    from .ranker_commands import _check_feature_code
    from .training_data import iter_shards, read_metadata

    root = Path(directory).resolve()
    metadata = read_metadata(root)
    proof = check_reweighting_metadata(metadata)
    if proof is None:
        raise ValueError("Cache is missing its training reweighting declaration")
    manifest_path = _inside(root, proof["manifest_path"])
    if _hash_file(manifest_path) != proof["manifest_sha256"]:
        raise ValueError("Reweighting manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or manifest.get("name") != REWEIGHT_NAME or manifest.get("code_sha256") != _code_hash()
            or any(manifest.get(key) != proof[key] for key in
                   ("source_metadata_sha256", "source_original_mass", "target_original_mass"))):
        raise ValueError("Unsupported or mismatching reweighting manifest")
    source = Path(manifest["source_directory"]).resolve()
    _separate(source, root)
    source_hash = _hash_file(source / "metadata.json")
    if source_hash != proof["source_metadata_sha256"]:
        raise ValueError("Reweighting source metadata hash mismatch")
    if expected_source_metadata_sha256 is not None and source_hash != expected_source_metadata_sha256:
        raise ValueError("Reweighting source differs from expected control cache")
    parent = read_metadata(source)
    validate_reweight_pair(parent, metadata, source_hash)
    if expected_source_dataset is not None and parent != expected_source_dataset:
        raise ValueError("Reweighting source metadata differs from expected control dataset")
    _check_feature_code(parent)
    verify_derivation(source, parent["derivation"]["parent_metadata_sha256"])
    files = _expected_files(parent)
    derived_files = sorted(set(files) - {"metadata.json"})
    if (_file_names(source) != files
            or _file_names(root) != sorted(files + [MANIFEST_NAME])):
        raise ValueError("Reweighting cache has unexpected or missing assets")
    if (set(manifest.get("source_file_sha256", {})) != set(files)
            or set(manifest.get("derived_file_sha256", {})) != set(derived_files)):
        raise ValueError("Reweighting manifest asset lists differ from cache metadata")
    shards = parent["train_shards"] + parent["valid_shards"]
    if set(manifest.get("unchanged_arrays_sha256", {})) != set(shards):
        raise ValueError("Reweighting manifest is missing unchanged array proofs")
    counts = {key: parent["counts"][key] for key in ("train_groups", "train_pairs", "valid_groups", "valid_pairs")}
    if manifest.get("counts") != counts:
        raise ValueError("Reweighting manifest counts disagree")
    for relative in files:
        old_hash = manifest["source_file_sha256"][relative]
        if _hash_file(_inside(source, relative)) != old_hash:
            raise ValueError(f"Reweighting source asset hash mismatch: {relative}")
        if relative == "metadata.json":
            continue
        new_hash = manifest["derived_file_sha256"][relative]
        if _hash_file(_inside(root, relative)) != new_hash:
            raise ValueError(f"Reweighted asset hash mismatch: {relative}")
        if relative not in parent["train_shards"] and old_hash != new_hash:
            raise ValueError(f"Reweighting changed a validation or audit asset: {relative}")
    for kind in ("train", "valid"):
        # zip_longest exhausts both generators, including their final mass-policy checks.
        for position, (old, new) in enumerate(zip_longest(iter_shards(source, kind), iter_shards(root, kind))):
            if old is None or new is None:
                raise ValueError("Source and reweighted cache shard counts differ")
            relative = parent[kind + "_shards"][position]
            expected = manifest["unchanged_arrays_sha256"][relative]
            if _invariant_digest(old) != expected or _invariant_digest(new) != expected:
                raise ValueError(f"Reweighted cache changed invariant arrays: {relative}")
    return {
        "verified": True, "name": REWEIGHT_NAME, "source_directory": str(source),
        "derived_directory": str(root), "source_metadata_sha256": source_hash,
        "derived_metadata_sha256": _hash_file(root / "metadata.json"),
        "manifest_sha256": proof["manifest_sha256"], "source_original_mass": proof["source_original_mass"],
        "target_original_mass": proof["target_original_mass"], "verified_assets": len(files),
        "verified_shards": len(shards), "invariants": manifest["invariants"],
    }
