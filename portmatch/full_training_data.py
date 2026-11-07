"""Full-public-training caches with internal OOF features and no holdout score.

This is a separate generation path so completed external-fold experiments keep
their source hashes. The final inference index is fitted only after every OOF
query has been processed, from all unmodified public training rows.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .data import QUERY_COLUMNS, read_catalog, read_queries
from .gt_augmentation import (
    GTMissingAugmentationConfig, VARIANTS, augmentation_spec, query_variants,
)
from .robust_retrieval import RobustMatchingPipeline, RobustRetrievalConfig, retrieval_spec
from .sampling import SamplingConfig, sample_training_candidates
from .text import has_hangul
from .training_data import _LOG_FIELDS, _ShardWriter, _finalize_query_masses, _write_json
from .validation import make_folds, sample_validation


FULL_TRAINING_SCOPE = "full_public_train"
_ASSETS = ("index.joblib", "catalog_ids.npy", "partitions.csv", "querylog.csv")


def _hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _hash_values(values):
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _population(train):
    # to_json normalizes missing scalars and NumPy values without permitting NaN
    # JSON tokens. Column and row order are deliberately part of the proof.
    rows = train[[*QUERY_COLUMNS, "target_vessel_id"]].to_json(
        orient="split", index=False, force_ascii=False, double_precision=15)
    return {"rows": len(train), "vessels": int(train.target_vessel_id.nunique()),
            "sample_ids_sha256": _hash_values(train.sample_id.astype(str)),
            "target_ids_sha256": _hash_values(sorted(set(train.target_vessel_id.astype(str)))),
            "original_rows_sha256": hashlib.sha256(rows.encode("utf-8")).hexdigest()}


def full_training_spec():
    digest = hashlib.sha256()
    for name in ("full_training_data.py", "training_data.py", "validation.py", "sampling.py",
                 "gt_augmentation.py", "name_feature_cache.py"):
        digest.update(name.encode("utf-8"))
        digest.update(Path(__file__).with_name(name).read_bytes())
    return {"name": "full_public_training_v1", "version": 1,
            "generation_code_sha256": digest.hexdigest(),
            "features": "mixed vessel and row OOF; each query excluded from its fitting rows",
            "augmentation": "mask queries after splitting; fit only original rows; source mass 1",
            "inference_index": "all original public training rows; no augmented or prediction rows",
            "validation": "none; use separate completed external-fold experiments"}


def check_full_training_metadata(metadata):
    """Validate the scope without changing ordinary external-fold metadata."""
    fields = {"full_training", "full_training_assets", "full_training_population", "full_training_pair_proof"}
    scope = metadata.get("training_scope")
    if "training_scope" not in metadata and not fields.intersection(metadata):
        if metadata.get("outer_split") == "none":
            raise ValueError("Full-training cache is missing its training scope")
        return None
    if scope != FULL_TRAINING_SCOPE or not fields.issubset(metadata):
        raise ValueError("Invalid or incomplete full-training scope")
    if metadata["full_training"] != full_training_spec():
        raise ValueError("Unknown or changed full-training generation; regenerate the cache")
    if (metadata.get("outer_split") != "none" or metadata.get("outer_folds") != 0
            or metadata.get("outer_fold") is not None or metadata.get("valid_shards") != []
            or metadata.get("inner_split") != "mixed" or metadata.get("inner_folds", 0) < 2
            or metadata.get("max_valid_queries") is not None
            or metadata.get("index_path") != "index.joblib"
            or metadata.get("catalog_ids_path") != "catalog_ids.npy"):
        raise ValueError("Full-training cache must have internal mixed OOF and no outer validation")
    counts = metadata.get("counts", {})
    population = metadata["full_training_population"]
    assets = metadata["full_training_assets"]
    pair_proof = metadata["full_training_pair_proof"]
    if (not isinstance(pair_proof, dict)
            or set(pair_proof) != {"algorithm", "source_feature_count", "shards", "sha256"}
            or pair_proof["algorithm"] != "sha256-original-shard-arrays-v1"
            or pair_proof["source_feature_count"] != 70
            or type(pair_proof["shards"]) is not int or pair_proof["shards"] < 1):
        raise ValueError("Invalid full-training pair-array proof")
    if (not isinstance(population, dict)
            or set(population) != {"rows", "vessels", "sample_ids_sha256", "target_ids_sha256", "original_rows_sha256"}
            or any(type(population.get(key)) is not int or population[key] < 1 for key in ("rows", "vessels"))
            or population["vessels"] > population["rows"]
            or not isinstance(assets, dict) or set(assets) != set(_ASSETS)):
        raise ValueError("Invalid full-training population or asset proof")
    for value in [*assets.values(), pair_proof["sha256"], *(population[key] for key in
                  ("sample_ids_sha256", "target_ids_sha256", "original_rows_sha256"))]:
        if (not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)):
            raise ValueError("Invalid full-training proof hash")
    if (counts.get("outer_training_rows") != population["rows"]
            or counts.get("training_views") != 2
            or any(counts.get(key) != 0 for key in
                   ("outer_validation_rows", "selected_validation_queries", "valid_pairs", "valid_groups", "valid_missing_target"))
            or metadata.get("sampled") is not (metadata.get("max_train_queries") is not None)
            or not 0 < counts.get("selected_training_queries", 0) <= population["rows"]
            or (not metadata["sampled"] and counts["selected_training_queries"] != population["rows"])):
        raise ValueError("Full-training counts disagree with the declared population")
    if (metadata.get("retrieval", {}).get("profile") != "robust"
            or metadata.get("retrieval", {}).get("max_total") is not None
            or metadata.get("training_augmentation") != augmentation_spec(GTMissingAugmentationConfig(.5))
            or metadata.get("training_reweighting") is not None):
        raise ValueError("Full training requires robust retrieval and original-mass 0.5 augmentation")
    return metadata["full_training"]


def _update_pair_proof(digest, relative, shard):
    from .name_feature_cache import _invariant_digest

    digest.update(relative.encode("utf-8") + b"\0")
    digest.update(_invariant_digest(shard, 70).encode("ascii") + b"\n")


def prepare_full_training_data(
    train, catalog, output, *, inner_folds=3, inner_split="mixed", seed=20261022,
    retrieval_config=None, sampling_config=None, augmentation_config=None,
    max_train_queries=None, shard_queries=64, provenance=None,
):
    """Create original 70-column training-only shards; no training-set score.

    ``max_train_queries`` marks diagnostic caches. It limits source queries,
    never the final index population, and cannot produce a submission profile.
    """
    started = time.perf_counter()
    retrieval = retrieval_config or RobustRetrievalConfig()
    sampling = sampling_config or SamplingConfig(seed=seed)
    augmentation = augmentation_config or GTMissingAugmentationConfig(.5)
    if (not isinstance(retrieval, RobustRetrievalConfig) or retrieval.profile != "robust"
            or retrieval.max_total is not None):
        raise ValueError("Full training requires robust profile and max_total=None")
    if not isinstance(augmentation, GTMissingAugmentationConfig) or augmentation.original_mass != .5:
        raise ValueError("Full training fixes original_mass=0.5")
    if inner_split != "mixed" or type(inner_folds) is not int or inner_folds < 2:
        raise ValueError("Full training requires mixed internal OOF with at least two folds")
    if type(shard_queries) is not int or shard_queries < 1:
        raise ValueError("shard_queries must be positive")
    required = set(QUERY_COLUMNS) | {"target_vessel_id"}
    if not required.issubset(train.columns):
        raise ValueError(f"Training columns missing: {sorted(required - set(train.columns))}")
    if (train.empty or train.sample_id.isna().any() or not train.sample_id.astype(str).is_unique
            or train.target_vessel_id.isna().any()):
        raise ValueError("Full training needs nonempty original rows and unique sample IDs")
    if catalog.vessel_id.isna().any() or not catalog.vessel_id.astype(str).is_unique:
        raise ValueError("Catalog vessel_id must be present and unique")
    if not train.target_vessel_id.isin(catalog.vessel_id).all():
        raise ValueError("All training targets must exist in the catalog")
    train, catalog = train.reset_index(drop=True).copy(), catalog.reset_index(drop=True).copy()
    selected = sample_validation(train, max_train_queries, seed + 101)
    selected_ids = set(selected.sample_id)
    views = ["vessel", "row"]
    assignments = {view: make_folds(train, view, inner_folds, seed + 1000 + number)
                   for number, view in enumerate(views)}
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new or empty dataset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    partitions = train[["sample_id", "target_vessel_id"]].copy()
    partitions["outer_fold"] = -1
    partitions["role"] = "train"
    for view in views:
        partitions["inner_" + view] = assignments[view]
    partitions.to_csv(output / "partitions.csv", index=False, encoding="utf-8-sig")
    np.save(output / "catalog_ids.npy", catalog.vessel_id.astype(str).to_numpy(dtype=str), allow_pickle=False)
    id_to_index = {value: index for index, value in enumerate(catalog.vessel_id)}
    writer = _ShardWriter(output, "train", shard_queries, augmented=True)
    counts = {"outer_training_rows": len(train), "outer_validation_rows": 0,
              "selected_training_queries": len(selected), "selected_validation_queries": 0,
              "training_views": 2, "training_attempts": 0, "train_missing_target": 0,
              "train_without_negative": 0, "valid_missing_target": 0,
              "valid_groups": 0, "valid_pairs": 0}
    diagnostics = {"variants": {name: {} for name in VARIANTS}}
    with (output / "querylog.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        logger = csv.DictWriter(stream, fieldnames=[*_LOG_FIELDS, "augmentation_variant"])
        logger.writeheader()
        for view in views:
            inner = assignments[view]
            for fold in range(inner_folds):
                fitting, held = train[inner != fold], train[inner == fold]
                queries = held[held.sample_id.isin(selected_ids)]
                if queries.empty:
                    continue
                if set(fitting.sample_id) & set(held.sample_id):
                    raise AssertionError("An OOF query was included in feature fitting")
                if view == "vessel" and set(fitting.target_vessel_id) & set(held.target_vessel_id):
                    raise AssertionError("An OOF vessel was included in feature fitting")
                if fitting.empty:
                    raise ValueError("Empty internal OOF fitting partition")
                print(f"Prepare full {view} inner fold {fold}: fit {len(fitting):,}, queries {len(queries):,}", flush=True)
                pipeline = RobustMatchingPipeline(fitting, catalog, retrieval)
                fit_vessels = fitting.target_vessel_id.nunique()
                for number, full in enumerate(queries.to_dict("records"), 1):
                    for variant, masked, initial_status in query_variants(full):
                        row = {key: masked[key] for key in QUERY_COLUMNS}
                        target = id_to_index[full["target_vessel_id"]]
                        query_id = str(row["sample_id"]) + "::" + view
                        if variant != "original":
                            query_id += "::" + variant
                        candidates = (pipeline.retrieve(row) if initial_status == "ready" else
                                      pd.DataFrame({"catalog_index": pd.Series(dtype=np.int64)}))
                        target_found = bool(candidates.catalog_index.eq(target).any())
                        status, selected_count = initial_status, 0
                        if initial_status == "ready":
                            counts["training_attempts"] += 1
                            if not target_found:
                                counts["train_missing_target"] += 1
                                status = "skipped_missing_target"
                            else:
                                features = pipeline.features(row, candidates)
                                positions = sample_training_candidates(candidates, features, target, query_id, sampling)
                                if len(positions) < 2:
                                    counts["train_without_negative"] += 1
                                    status = "skipped_without_negative"
                                else:
                                    writer.add(features.iloc[positions], candidates.iloc[positions],
                                               sample_id=row["sample_id"], query_id=query_id,
                                               language=has_hangul(row["vessel_name"]), target=target,
                                               augmentation_variant=variant)
                                    status, selected_count = "stored", len(positions)
                        stats = diagnostics["variants"][variant]
                        stats[status] = stats.get(status, 0) + 1
                        logger.writerow({"kind": "train", "view": view, "inner_fold": fold,
                                         "sample_id": row["sample_id"], "query_id": query_id,
                                         "has_hangul": has_hangul(row["vessel_name"]),
                                         "candidate_count": len(candidates), "selected_count": selected_count,
                                         "target_found": target_found, "status": status,
                                         "fit_rows": len(fitting), "fit_vessels": fit_vessels,
                                         "augmentation_variant": variant})
                    if number % 100 == 0:
                        print(f"Prepare full {view}/{fold}: {number:,}/{len(queries):,}", flush=True)
                        stream.flush()
                del pipeline
    writer.flush()
    if writer.groups == 0:
        raise ValueError("No usable full-training groups; targets were not injected")
    group_proof, mass_diagnostics = _finalize_query_masses(
        output, writer.files, views, augmentation, selected)
    diagnostics.update(mass_diagnostics)
    pair_digest = hashlib.sha256()
    for relative in writer.files:
        with np.load(output / relative, allow_pickle=False) as archive:
            _update_pair_proof(pair_digest, relative, {key: archive[key] for key in archive.files})
    print(f"Fit final inference index on all {len(train):,} original public training rows", flush=True)
    final_pipeline = RobustMatchingPipeline(train, catalog, retrieval)
    joblib.dump(final_pipeline, output / "index.joblib", compress=3)
    assets = {name: _hash_file(output / name) for name in _ASSETS}
    counts.update(train_pairs=writer.pairs, train_groups=writer.groups)
    metadata = {"schema_version": 1, "complete": True,
                "training_scope": FULL_TRAINING_SCOPE, "full_training": full_training_spec(),
                "full_training_assets": assets, "full_training_population": _population(train),
                "full_training_pair_proof": {"algorithm": "sha256-original-shard-arrays-v1",
                    "source_feature_count": 70, "shards": len(writer.files), "sha256": pair_digest.hexdigest()},
                "catalog_rows": len(catalog), "feature_names": writer.feature_names,
                "train_shards": writer.files, "valid_shards": [], "counts": counts,
                "outer_split": "none", "outer_folds": 0, "outer_fold": None,
                "inner_split": inner_split, "inner_folds": inner_folds, "seed": seed,
                "retrieval": asdict(retrieval), "sampling": asdict(sampling),
                "max_train_queries": max_train_queries, "max_valid_queries": None,
                "sampled": max_train_queries is not None, "shard_queries": shard_queries,
                "index_path": "index.joblib", "catalog_ids_path": "catalog_ids.npy",
                "provenance": provenance or {}, "retrieval_extension": retrieval_spec(retrieval),
                "retrieval_index_sha256": assets["index.joblib"],
                "training_augmentation": augmentation_spec(augmentation),
                "training_augmentation_diagnostics": diagnostics,
                "training_augmentation_assets": {"training_groups_sha256": group_proof,
                    "querylog.csv": assets["querylog.csv"], "partitions.csv": assets["partitions.csv"]},
                "validation_sampling": "none; no held-out validation in full training",
                "training_missing_target_policy": "skip group, count and log; never inject target",
                "seconds": time.perf_counter() - started}
    check_full_training_metadata(metadata)
    _write_json(output / "metadata.json", metadata)
    return metadata


def verify_full_training_cache(directory, *, train_path=None, catalog_path=None, train=None, catalog=None):
    """Verify the actual full population, OOF log, final index and stored groups.

    Supply original inputs (or paths) for exact population and final fitted-state
    comparison. Without them, returns a self-contained provenance audit only.
    """
    from .features import CategoryCompatibility
    from .memory import TrainingMemory
    from .ranker_commands import _load_retrieval_index
    from .training_data import iter_shards, read_metadata

    root = Path(directory)
    metadata = read_metadata(root)
    if check_full_training_metadata(metadata) is None:
        raise ValueError("Expected an explicit full-public-training cache")
    for name, expected in metadata["full_training_assets"].items():
        if _hash_file(root / name) != expected:
            raise ValueError(f"Full-training asset changed: {name}")
    if train_path is not None:
        if _hash_file(train_path) != metadata.get("provenance", {}).get("train_sha256"):
            raise ValueError("Original training file differs from full-cache provenance")
        train = read_queries(train_path, labeled=True)
    if catalog_path is not None:
        if _hash_file(catalog_path) != metadata.get("provenance", {}).get("catalog_sha256"):
            raise ValueError("Original catalog differs from full-cache provenance")
        catalog = read_catalog(catalog_path)
    partitions = pd.read_csv(root / "partitions.csv", encoding="utf-8-sig",
                             dtype={"sample_id": str, "target_vessel_id": str})
    population = metadata["full_training_population"]
    if (set(partitions.columns) != {"sample_id", "target_vessel_id", "outer_fold", "role", "inner_vessel", "inner_row"}
            or len(partitions) != population["rows"] or not partitions.sample_id.is_unique
            or partitions.sample_id.isna().any() or partitions.target_vessel_id.isna().any()
            or not partitions.role.eq("train").all() or not partitions.outer_fold.eq(-1).all()
            or _hash_values(partitions.sample_id) != population["sample_ids_sha256"]
            or _hash_values(sorted(set(partitions.target_vessel_id))) != population["target_ids_sha256"]
            or partitions.target_vessel_id.nunique() != population["vessels"]):
        raise ValueError("Full-training partitions do not cover the declared original population")
    for view in ("vessel", "row"):
        values = partitions["inner_" + view]
        if not values.between(0, metadata["inner_folds"] - 1).all():
            raise ValueError("Invalid internal full-training fold assignments")
    if partitions.groupby("target_vessel_id").inner_vessel.nunique().max() != 1:
        raise ValueError("A vessel crosses internal vessel OOF partitions")
    pipeline = _load_retrieval_index(root / "index.joblib", metadata)
    ids = np.load(root / "catalog_ids.npy", allow_pickle=False)
    if (not np.array_equal(ids, pipeline.catalog.vessel_id.astype(str))
            or pipeline.memory.training_rows != population["rows"]
            or set(pipeline.memory.training_vessels) != set(partitions.target_vessel_id)):
        raise ValueError("Final inference index was not fitted on the declared full population")
    if (train is None) != (catalog is None):
        raise ValueError("Provide both original train and catalog for exact full-fit verification")
    if train is not None:
        train, catalog = train.reset_index(drop=True), catalog.reset_index(drop=True)
        if _population(train) != population:
            raise ValueError("Full cache original row population differs from supplied training rows")
        try:
            pd.testing.assert_frame_equal(pipeline.catalog, catalog)
        except AssertionError as error:
            raise ValueError("Full index catalog differs from supplied catalog") from error
        memory = TrainingMemory().fit(train, catalog)
        categories = CategoryCompatibility().fit(train, catalog)
        if (memory.call_signs != pipeline.memory.call_signs or memory.names != pipeline.memory.names
                or categories.tables != pipeline.categories.tables
                or categories.smoothing != pipeline.categories.smoothing):
            raise ValueError("Final lookup/category state differs from all original training rows")
        try:
            pd.testing.assert_frame_equal(memory.aliases, pipeline.memory.aliases)
        except AssertionError as error:
            raise ValueError("Final alias state differs from all original training rows") from error
        for number, view in enumerate(("vessel", "row")):
            expected = make_folds(train, view, metadata["inner_folds"], metadata["seed"] + 1000 + number)
            if not np.array_equal(expected, partitions["inner_" + view]):
                raise ValueError("Internal OOF assignments differ from declared seed and original rows")
    log = pd.read_csv(root / "querylog.csv", encoding="utf-8-sig", dtype={"sample_id": str, "query_id": str})
    if (not log.kind.eq("train").all() or not log.view.isin(["vessel", "row"]).all()
            or not log.augmentation_variant.isin(VARIANTS).all() or not log.query_id.is_unique
            or not log.sample_id.isin(partitions.sample_id).all()):
        raise ValueError("Full-training query log contains invalid identities")
    lookup = partitions.set_index("sample_id")
    for (view, fold), part in log.groupby(["view", "inner_fold"]):
        held = partitions[partitions["inner_" + view] == fold]
        fitting = partitions[partitions["inner_" + view] != fold]
        if (not part.sample_id.isin(held.sample_id).all()
                or not part.fit_rows.eq(len(fitting)).all()
                or not part.fit_vessels.eq(fitting.target_vessel_id.nunique()).all()):
            raise ValueError("Query log does not match internal OOF fitting boundaries")
    selected_ids = set(log.sample_id)
    expected_log_groups = {(source, view, variant) for source in selected_ids
                           for view in ("vessel", "row") for variant in VARIANTS}
    if (set(zip(log.sample_id, log.view, log.augmentation_variant)) != expected_log_groups
            or len(log) != len(expected_log_groups)
            or len(selected_ids) != metadata["counts"]["selected_training_queries"]):
        raise ValueError("Full-training source/view/variant log coverage is incomplete")
    if train is not None:
        expected_selected = sample_validation(train, metadata["max_train_queries"], metadata["seed"] + 101)
        if selected_ids != set(expected_selected.sample_id):
            raise ValueError("Diagnostic source selection differs from the recorded limit")
    stored = log[log.status.eq("stored")].set_index("query_id")
    seen, pairs = set(), 0
    pair_digest = hashlib.sha256()
    for relative, shard in zip(metadata["train_shards"], iter_shards(root, "train")):
        _update_pair_proof(pair_digest, relative, shard)
        for i, query in enumerate(shard["query_id"]):
            query = str(query)
            source = str(shard["sample_id"][i])
            if query not in stored.index or source not in lookup.index:
                raise ValueError("Cached group is absent from full-training OOF log")
            record = stored.loc[query]
            if (record.sample_id != source or record.selected_count != int(shard["group"][i])
                    or record.augmentation_variant != str(shard["augmentation_variant"][i])
                    or ids[int(shard["target_catalog_index"][i])] != lookup.loc[source, "target_vessel_id"]):
                raise ValueError("Cached group differs from full-training OOF identity/label")
            seen.add(query)
        pairs += len(shard["y"])
    if (seen != set(stored.index) or len(seen) != metadata["counts"]["train_groups"]
            or pairs != metadata["counts"]["train_pairs"]):
        raise ValueError("Full-training cached group totals disagree with metadata/log")
    pair_proof = metadata["full_training_pair_proof"]
    if (len(metadata["train_shards"]) != pair_proof["shards"]
            or pair_digest.hexdigest() != pair_proof["sha256"]):
        raise ValueError("Full-training original feature or pair arrays changed")
    return {"verified": True, "training_scope": FULL_TRAINING_SCOPE,
            "original_population_verified": train is not None,
            "public_training_rows": population["rows"], "public_training_vessels": population["vessels"],
            "selected_sources": len(selected_ids),
            "retained_sources": metadata["training_augmentation_diagnostics"]["retained_sources"],
            "dropped_sources": metadata["training_augmentation_diagnostics"]["dropped_sources"],
            "training_groups": len(seen), "training_pairs": pairs,
            "held_out_validation_rows": 0, "final_index_original_rows": pipeline.memory.training_rows,
            "sampled": metadata["sampled"]}
