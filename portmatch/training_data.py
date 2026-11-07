"""Nested, out-of-fold feature caches for supervised candidate ranking.

Outer validation labels never fit any lookup, alias index, or category table.
Each training query is represented using an inner-fold pipeline that excludes
the query itself (and its whole vessel for the vessel view). Only training
groups receive negative sampling; outer validation keeps every retrieved row.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Iterator

import joblib
import numpy as np
import pandas as pd

from .data import QUERY_COLUMNS
from .gt_augmentation import (
    GTMissingAugmentationConfig, VARIANTS, augmentation_spec,
    check_augmentation_metadata, expected_query_masses, query_identity,
    query_variants, update_group_digest,
)
from .pipeline import MatchingPipeline
from .retrieval import RetrievalConfig
from .sampling import SamplingConfig, sample_training_candidates
from .text import has_hangul
from .validation import make_folds, sample_validation


_FORBIDDEN_FEATURES = {"sample_id", "query_id", "vessel_id", "target_vessel_id",
                       "target_catalog_index", "catalog_index", "label", "y"}
_LOG_FIELDS = ["kind", "view", "inner_fold", "sample_id", "query_id", "has_hangul",
               "candidate_count", "selected_count", "target_found", "status",
               "fit_rows", "fit_vessels"]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _check_features(features: pd.DataFrame) -> None:
    if not features.columns.is_unique or _FORBIDDEN_FEATURES.intersection(features.columns):
        raise ValueError("Feature names must be unique and contain no identifiers or labels")
    if np.isinf(features.to_numpy(dtype=np.float32)).any():
        raise ValueError("Feature matrix contains infinity")


class _ShardWriter:
    """Keep at most shard_queries candidate groups in memory before compression."""

    def __init__(self, root: Path, kind: str, shard_queries: int, *, augmented: bool = False):
        self.root, self.kind, self.shard_queries = root, kind, shard_queries
        self.augmented = augmented and kind == "train"
        self.files: list[str] = []
        self.feature_names: list[str] | None = None
        self.pending: list[dict] = []
        self.pairs, self.groups = 0, 0
        (root / kind).mkdir()

    def add(self, features: pd.DataFrame, candidates: pd.DataFrame, *,
            sample_id: str, query_id: str, language: bool, target: int,
            augmentation_variant: str = "original") -> None:
        _check_features(features)
        names = features.columns.tolist()
        if self.feature_names is None:
            self.feature_names = names
        elif self.feature_names != names:
            raise ValueError("Feature schema changed between queries")
        indices = candidates.catalog_index.to_numpy(dtype=np.int64)
        if len(features) != len(indices) or not len(indices):
            raise ValueError("Each cached group needs aligned, nonempty candidates/features")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("A candidate occurs more than once in a group")
        labels = (indices == target).astype(np.uint8)
        if self.kind == "train" and (labels.sum() != 1 or len(labels) < 2):
            raise ValueError("Training groups need exactly one positive and at least one negative")
        self.pending.append({"X": features.to_numpy(dtype=np.float32), "y": labels,
                             "catalog_index": indices, "sample_id": str(sample_id),
                             "query_id": str(query_id), "has_hangul": bool(language),
                             "target_catalog_index": int(target)})
        if self.augmented:
            self.pending[-1]["augmentation_variant"] = augmentation_variant
        self.pairs += len(indices)
        self.groups += 1
        if len(self.pending) >= self.shard_queries:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        relative = f"{self.kind}/part_{len(self.files):05d}.npz"
        arrays = dict(
            X=np.concatenate([item["X"] for item in self.pending]),
            y=np.concatenate([item["y"] for item in self.pending]),
            catalog_index=np.concatenate([item["catalog_index"] for item in self.pending]),
            group=np.asarray([len(item["y"]) for item in self.pending], dtype=np.int32),
            sample_id=np.asarray([item["sample_id"] for item in self.pending], dtype=str),
            query_id=np.asarray([item["query_id"] for item in self.pending], dtype=str),
            has_hangul=np.asarray([item["has_hangul"] for item in self.pending], dtype=bool),
            target_catalog_index=np.asarray([item["target_catalog_index"] for item in self.pending], dtype=np.int64),
        )
        if self.augmented:
            arrays["augmentation_variant"] = np.asarray(
                [item["augmentation_variant"] for item in self.pending], dtype=str)
            # Final source/view normalization runs after every usable group is known.
            arrays["query_mass"] = np.ones(len(self.pending), dtype=np.float64)
        np.savez_compressed(self.root / relative, **arrays)
        self.files.append(relative)
        self.pending.clear()


def _finalize_query_masses(output, files, views, config, selected_train):
    records = []
    for relative in files:
        with np.load(output / relative, allow_pickle=False) as shard:
            for source, query, variant in zip(shard["sample_id"], shard["query_id"],
                                              shard["augmentation_variant"]):
                source, view = query_identity(str(source), str(query), str(variant), views)
                records.append((source, view, str(variant)))
    masses = expected_query_masses(records, config)
    digest = hashlib.sha256()
    source_totals, source_views = {}, {}
    views_without_original = views_without_masks = 0
    for source, view, variant in records:
        source_totals[source] = source_totals.get(source, 0.0) + masses[source, view, variant]
        source_views.setdefault((source, view), []).append(variant)
    for variants in source_views.values():
        views_without_original += "original" not in variants
        views_without_masks += variants == ["original"]
    for relative in files:
        with np.load(output / relative, allow_pickle=False) as archive:
            shard = {key: archive[key] for key in archive.files}
        values = []
        for source, query, variant in zip(shard["sample_id"], shard["query_id"],
                                          shard["augmentation_variant"]):
            source, view = query_identity(str(source), str(query), str(variant), views)
            values.append(masses[source, view, str(variant)])
        shard["query_mass"] = np.asarray(values, dtype=np.float64)
        update_group_digest(digest, shard)
        np.savez_compressed(output / relative, **shard)
    if not all(np.isclose(total, 1, atol=1e-12, rtol=0) for total in source_totals.values()):
        raise AssertionError("Training augmentation source mass is not normalized")
    languages = dict(zip(selected_train.sample_id.astype(str),
                         selected_train.vessel_name.map(has_hangul)))
    return digest.hexdigest(), {
        "mass_policy_verified": True, "selected_sources": len(selected_train),
        "retained_sources": len(source_totals),
        "dropped_sources": len(selected_train) - len(source_totals),
        "retained_views": len(source_views), "retained_groups": len(records),
        "views_without_original": views_without_original,
        "views_without_masks": views_without_masks,
        "source_mass_min": min(source_totals.values()),
        "source_mass_max": max(source_totals.values()),
        "total_query_mass": float(sum(source_totals.values())),
        "hangul_query_mass": float(sum(total for source, total in source_totals.items() if languages[source])),
        "non_hangul_query_mass": float(sum(total for source, total in source_totals.items() if not languages[source])),
        "variant_query_mass": {variant: float(sum(mass for identity, mass in masses.items()
                                                   if identity[2] == variant)) for variant in VARIANTS},
    }


def prepare_training_data(
    train: pd.DataFrame, catalog: pd.DataFrame, output: str | Path, *,
    outer_split: str = "vessel", outer_folds: int = 5, outer_fold: int = 0,
    inner_folds: int = 3, inner_split: str = "mixed", seed: int = 20261022,
    retrieval_config: RetrievalConfig | None = None,
    sampling_config: SamplingConfig | None = None,
    augmentation_config: GTMissingAugmentationConfig | None = None,
    max_train_queries: int | None = None, max_valid_queries: int | None = None,
    shard_queries: int = 64, provenance: dict | None = None,
) -> dict:
    """Write a self-contained ranking dataset; limits are for smoke tests only.

    ``mixed`` produces two OOF views of each selected outer-training query:
    vessel-disjoint retrieval and row-disjoint retrieval with legitimate previous
    arrivals. The source query never contributes its own label to either view.
    All fold dictionaries may use the unselected training queries when a smoke
    limit is set, but never that view's held-out queries or outer-validation rows.
    """
    started = time.perf_counter()
    retrieval = retrieval_config or RetrievalConfig()
    sampling = sampling_config or SamplingConfig(seed=seed)
    if augmentation_config is not None:
        from .robust_retrieval import RobustRetrievalConfig

        if not isinstance(augmentation_config, GTMissingAugmentationConfig):
            raise TypeError("augmentation_config must be GTMissingAugmentationConfig")
        if not isinstance(retrieval, RobustRetrievalConfig) or retrieval.profile != "robust":
            raise ValueError("GT missing augmentation requires retrieval profile robust")
    if retrieval.max_total is not None:
        raise ValueError("Training-data preparation requires max_total=None to preserve validation candidates")
    if inner_split not in {"vessel", "row", "mixed"}:
        raise ValueError("inner_split must be vessel, row, or mixed")
    if shard_queries < 1:
        raise ValueError("shard_queries must be positive")
    if not 0 <= outer_fold < outer_folds:
        raise ValueError("outer_fold is outside outer_folds")
    required = set(QUERY_COLUMNS) | {"target_vessel_id"}
    if not required.issubset(train.columns):
        raise ValueError(f"Training columns missing: {sorted(required - set(train.columns))}")
    if train.sample_id.isna().any() or not train.sample_id.is_unique:
        raise ValueError("Training sample_id must be present and unique")
    if augmentation_config is not None and not train.sample_id.astype(str).is_unique:
        raise ValueError("Augmented sample_id must remain unique after string normalization")
    if catalog.vessel_id.isna().any() or not catalog.vessel_id.is_unique:
        raise ValueError("Catalog vessel_id must be present and unique")
    if train.target_vessel_id.isna().any() or not train.target_vessel_id.isin(catalog.vessel_id).all():
        raise ValueError("All training targets must exist in the catalog")
    train, catalog = train.reset_index(drop=True).copy(), catalog.reset_index(drop=True).copy()
    assignments = make_folds(train, outer_split, outer_folds, seed)
    outer_train = train[assignments != outer_fold].reset_index(drop=True)
    outer_valid = train[assignments == outer_fold].reset_index(drop=True)
    if outer_train.empty or outer_valid.empty:
        raise ValueError("Outer training and validation partitions must both be nonempty")
    selected_train = sample_validation(outer_train, max_train_queries, seed + 101)
    selected_valid = sample_validation(outer_valid, max_valid_queries, seed + 102)
    selected_ids = set(selected_train.sample_id)
    views = ["vessel", "row"] if inner_split == "mixed" else [inner_split]
    inner_assignments = {
        view: make_folds(outer_train, view, inner_folds, seed + 1000 + view_number)
        for view_number, view in enumerate(views)
    }
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new or empty dataset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    partition_log = train[["sample_id", "target_vessel_id"]].copy()
    partition_log["outer_fold"] = assignments
    partition_log["role"] = np.where(assignments == outer_fold, "valid", "train")
    for view, inner in inner_assignments.items():
        mapping = dict(zip(outer_train.sample_id, inner))
        partition_log["inner_" + view] = partition_log.sample_id.map(mapping).fillna(-1).astype(int)
    partition_log.to_csv(output / "partitions.csv", index=False, encoding="utf-8-sig")
    np.save(output / "catalog_ids.npy", catalog.vessel_id.astype(str).to_numpy(dtype=str), allow_pickle=False)
    id_to_index = {value: index for index, value in enumerate(catalog.vessel_id)}
    validation_ids = set(outer_valid.sample_id)
    validation_targets = set(outer_valid.target_vessel_id)
    writers = {kind: _ShardWriter(output, kind, shard_queries, augmented=augmentation_config is not None)
               for kind in ("train", "valid")}
    counts = {"outer_training_rows": len(outer_train), "outer_validation_rows": len(outer_valid),
              "selected_training_queries": len(selected_train), "selected_validation_queries": len(selected_valid),
              "training_views": len(views), "training_attempts": 0,
              "train_missing_target": 0, "train_without_negative": 0, "valid_missing_target": 0}
    augmentation_diagnostics = {"variants": {name: {} for name in VARIANTS}}

    def fitted_pipeline(fitting: pd.DataFrame, held: pd.DataFrame, view: str) -> MatchingPipeline:
        fitting_ids = set(fitting.sample_id)
        if fitting_ids & validation_ids or fitting_ids & set(held.sample_id):
            raise AssertionError("A held-out query was included in feature fitting")
        if outer_split == "vessel" and set(fitting.target_vessel_id) & validation_targets:
            raise AssertionError("An outer-validation vessel was included in feature fitting")
        if view == "vessel" and set(fitting.target_vessel_id) & set(held.target_vessel_id):
            raise AssertionError("An inner-validation vessel was included in feature fitting")
        from .robust_retrieval import RobustRetrievalConfig, RobustMatchingPipeline

        if isinstance(retrieval, RobustRetrievalConfig):
            return RobustMatchingPipeline(fitting, catalog, retrieval)
        return MatchingPipeline(fitting, catalog, retrieval)

    with (output / "querylog.csv").open("w", newline="", encoding="utf-8-sig") as log_file:
        logger = csv.DictWriter(log_file, fieldnames=_LOG_FIELDS +
                                (["augmentation_variant"] if augmentation_config is not None else []))
        logger.writeheader()

        def process_group(pipeline, full: dict, kind: str, view: str,
                          fold: int, fit_rows: int, fit_vessels: int,
                          variant: str = "original", initial_status: str = "ready") -> None:
            row = {key: full[key] for key in QUERY_COLUMNS}
            target = id_to_index[full["target_vessel_id"]]
            query_id = str(full["sample_id"]) + "::" + view
            if variant != "original":
                query_id += "::" + variant
            language = has_hangul(row["vessel_name"])
            candidates = (pipeline.retrieve(row) if initial_status == "ready"
                          else pd.DataFrame({"catalog_index": pd.Series(dtype=np.int64)}))
            target_found = bool(candidates.catalog_index.eq(target).any())
            status = "stored" if initial_status == "ready" else initial_status
            selected_count = 0
            if initial_status != "ready":
                pass
            elif kind == "train":
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
                        writers[kind].add(features.iloc[positions], candidates.iloc[positions],
                                          sample_id=row["sample_id"], query_id=query_id,
                                          language=language, target=target,
                                          augmentation_variant=variant)
                        selected_count = len(positions)
            else:
                if not target_found:
                    counts["valid_missing_target"] += 1
                features = pipeline.features(row, candidates)
                writers[kind].add(features, candidates, sample_id=row["sample_id"], query_id=query_id,
                                  language=language, target=target)
                selected_count = len(candidates)
            log = {"kind": kind, "view": view, "inner_fold": fold,
                             "sample_id": row["sample_id"], "query_id": query_id,
                             "has_hangul": language, "candidate_count": len(candidates),
                             "selected_count": selected_count, "target_found": target_found,
                             "status": status, "fit_rows": fit_rows,
                             "fit_vessels": fit_vessels}
            if augmentation_config is not None:
                log["augmentation_variant"] = variant
                if kind == "train":
                    stats = augmentation_diagnostics["variants"][variant]
                    stats[status] = stats.get(status, 0) + 1
            logger.writerow(log)

        for view in views:
            inner = inner_assignments[view]
            for fold in range(inner_folds):
                fitting = outer_train[inner != fold]
                held = outer_train[inner == fold]
                queries = held[held.sample_id.isin(selected_ids)]
                if queries.empty:
                    continue
                print(f"Prepare {view} inner fold {fold}: fit {len(fitting):,}, queries {len(queries):,}", flush=True)
                pipeline = fitted_pipeline(fitting, held, view)
                fit_vessels = fitting.target_vessel_id.nunique()
                for number, full in enumerate(queries.to_dict("records"), 1):
                    variants = (query_variants(full) if augmentation_config is not None
                                else [("original", full, "ready")])
                    for variant, query, status in variants:
                        process_group(pipeline, query, "train", view, fold, len(fitting), fit_vessels,
                                      variant, status)
                    if number % 100 == 0:
                        print(f"Prepare {view}/{fold}: {number:,}/{len(queries):,}", flush=True)
                        log_file.flush()
                del pipeline
        writers["train"].flush()
        if writers["train"].groups == 0:
            raise ValueError("No usable training groups: check retrieval coverage and training limits")
        print(f"Prepare outer validation: fit {len(outer_train):,}, queries {len(selected_valid):,}; all candidates retained", flush=True)
        pipeline = fitted_pipeline(outer_train, outer_valid, "outer")
        fit_vessels = outer_train.target_vessel_id.nunique()
        for number, full in enumerate(selected_valid.to_dict("records"), 1):
            process_group(pipeline, full, "valid", "valid", -1, len(outer_train), fit_vessels)
            if number % 100 == 0:
                print(f"Prepare validation: {number:,}/{len(selected_valid):,}", flush=True)
                log_file.flush()
        writers["valid"].flush()
        if writers["train"].feature_names != writers["valid"].feature_names:
            raise ValueError("Training and validation feature schemas differ")
        joblib.dump(pipeline, output / "index.joblib", compress=3)
    counts.update({f"{kind}_pairs": writer.pairs for kind, writer in writers.items()})
    counts.update({f"{kind}_groups": writer.groups for kind, writer in writers.items()})
    metadata = {"schema_version": 1, "complete": True,
                "catalog_rows": len(catalog),
                "feature_names": writers["train"].feature_names,
                "train_shards": writers["train"].files, "valid_shards": writers["valid"].files,
                "counts": counts, "outer_split": outer_split, "outer_folds": outer_folds,
                "outer_fold": outer_fold, "inner_split": inner_split, "inner_folds": inner_folds,
                "seed": seed, "retrieval": asdict(retrieval), "sampling": asdict(sampling),
                "max_train_queries": max_train_queries, "max_valid_queries": max_valid_queries,
                "sampled": max_train_queries is not None or max_valid_queries is not None,
                "shard_queries": shard_queries, "index_path": "index.joblib",
                "catalog_ids_path": "catalog_ids.npy", "provenance": provenance or {},
                "seconds": time.perf_counter() - started,
                "validation_sampling": "none; every retrieved candidate retained",
                "training_missing_target_policy": "skip group, count and log; never inject target"}
    from .robust_retrieval import retrieval_spec

    extension = retrieval_spec(retrieval)
    if extension is not None:
        metadata["retrieval_extension"] = extension
        with (output / "index.joblib").open("rb") as stream:
            metadata["retrieval_index_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    if augmentation_config is not None:
        group_proof, mass_diagnostics = _finalize_query_masses(
            output, writers["train"].files, views, augmentation_config, selected_train)
        augmentation_diagnostics.update(mass_diagnostics)
        metadata["training_augmentation"] = augmentation_spec(augmentation_config)
        metadata["training_augmentation_diagnostics"] = augmentation_diagnostics
        metadata["training_augmentation_assets"] = {"training_groups_sha256": group_proof}
        for filename in ("querylog.csv", "partitions.csv"):
            with (output / filename).open("rb") as stream:
                metadata["training_augmentation_assets"][filename] = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata["seconds"] = time.perf_counter() - started
    _write_json(output / "metadata.json", metadata)
    return metadata


def read_metadata(directory: str | Path) -> dict:
    metadata = json.loads((Path(directory) / "metadata.json").read_text("utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("complete") is not True:
        raise ValueError("Dataset is incomplete or has an unsupported schema")
    names = metadata.get("feature_names", [])
    if not names or len(set(names)) != len(names) or _FORBIDDEN_FEATURES.intersection(names):
        raise ValueError("Invalid cached feature schema")
    from .robust_retrieval import check_retrieval_metadata

    check_retrieval_metadata(metadata)
    augmentation = check_augmentation_metadata(metadata)
    if augmentation is not None:
        for filename in ("querylog.csv", "partitions.csv"):
            try:
                with (Path(directory) / filename).open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError as error:
                raise ValueError("Missing training augmentation audit asset") from error
            if actual != metadata["training_augmentation_assets"][filename]:
                raise ValueError(f"Training augmentation audit asset changed: {filename}")
    return metadata


def iter_shards(directory: str | Path, kind: str = "train") -> Iterator[dict[str, np.ndarray]]:
    """Read bounded arrays; no pickled Python objects are accepted."""
    if kind not in {"train", "valid"}:
        raise ValueError("kind must be train or valid")
    directory = Path(directory).resolve()
    metadata = read_metadata(directory)
    augmentation = check_augmentation_metadata(metadata)
    augmented_train = augmentation is not None and kind == "train"
    augmentation_records, observed_masses = [], []
    augmentation_digest = hashlib.sha256()
    source_consistency = {}
    views = ["vessel", "row"] if metadata["inner_split"] == "mixed" else [metadata["inner_split"]]
    seen_query_ids = set()
    paths = metadata[kind + "_shards"]
    for shard_number, relative in enumerate(paths):
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Shard path escapes the dataset directory")
        with np.load(path, allow_pickle=False) as archive:
            item = {key: archive[key] for key in archive.files}
        required = {"X", "y", "group", "sample_id", "query_id", "has_hangul", "target_catalog_index", "catalog_index"}
        if augmented_train:
            required.update({"augmentation_variant", "query_mass"})
        if set(item) != required:
            raise ValueError(f"Shard arrays are missing or unexpected: {path.name}")
        groups, x = item["group"], item["X"]
        if groups.ndim != 1 or groups.dtype.kind not in "iu" or (groups <= 0).any():
            raise ValueError("Invalid cached group sizes")
        rows, query_count = int(groups.sum()), len(groups)
        if x.shape != (rows, len(metadata["feature_names"])) or x.dtype != np.float32 or np.isinf(x).any():
            raise ValueError("Invalid cached feature matrix")
        if any(item[name].shape != (query_count,) for name in
               ("sample_id", "query_id", "has_hangul", "target_catalog_index")):
            raise ValueError("Cached query metadata is misaligned")
        if item["y"].shape != (rows,) or item["catalog_index"].shape != (rows,):
            raise ValueError("Cached pair metadata is misaligned")
        if not np.isin(item["y"], [0, 1]).all():
            raise ValueError("Labels must be binary")
        query_ids = item["query_id"].tolist()
        if len(set(query_ids)) != len(query_ids) or seen_query_ids.intersection(query_ids):
            raise ValueError("Duplicate cached query groups")
        seen_query_ids.update(query_ids)
        if augmented_train:
            if (item["augmentation_variant"].shape != (query_count,)
                    or item["augmentation_variant"].dtype.kind != "U"
                    or item["query_mass"].shape != (query_count,)
                    or item["query_mass"].dtype != np.float64
                    or not np.isfinite(item["query_mass"]).all()
                    or (item["query_mass"] <= 0).any()
                    or item["sample_id"].dtype.kind != "U" or item["query_id"].dtype.kind != "U"):
                raise ValueError("Invalid augmentation variant or query_mass arrays")
            for index, (source, query, variant) in enumerate(zip(
                    item["sample_id"], item["query_id"], item["augmentation_variant"])):
                source, view = query_identity(str(source), str(query), str(variant), views)
                identity = (bool(item["has_hangul"][index]), int(item["target_catalog_index"][index]))
                if source_consistency.setdefault(source, identity) != identity:
                    raise ValueError("Augmentation changed a source language or target")
                augmentation_records.append((source, view, str(variant)))
                observed_masses.append(float(item["query_mass"][index]))
            update_group_digest(augmentation_digest, item)
        offsets = np.r_[0, np.cumsum(groups)]
        for index, target in enumerate(item["target_catalog_index"]):
            section = slice(offsets[index], offsets[index + 1])
            catalog_indices = item["catalog_index"][section]
            labels = item["y"][section]
            if len(np.unique(catalog_indices)) != len(catalog_indices) or not np.array_equal(labels, catalog_indices == target):
                raise ValueError("Candidate identities and labels disagree")
            if kind == "train" and (labels.sum() != 1 or len(labels) < 2):
                raise ValueError("Invalid sampled training group")
        if augmented_train and shard_number == len(paths) - 1:
            expected = expected_query_masses(augmentation_records, augmentation)
            expected_values = [expected[identity] for identity in augmentation_records]
            if not np.allclose(observed_masses, expected_values, atol=1e-12, rtol=0):
                raise ValueError("Augmented query_mass violates source/view normalization policy")
            if augmentation_digest.hexdigest() != metadata["training_augmentation_assets"]["training_groups_sha256"]:
                raise ValueError("Training augmentation group proof differs from cached arrays")
            diagnostics = metadata["training_augmentation_diagnostics"]
            retained_views = {(source, view) for source, view, _ in augmentation_records}
            retained_originals = {(source, view) for source, view, variant in augmentation_records
                                  if variant == "original"}
            retained_masks = {(source, view) for source, view, variant in augmentation_records
                             if variant != "original"}
            variant_counts = {variant: sum(record[2] == variant for record in augmentation_records)
                              for variant in VARIANTS}
            variant_masses = {variant: sum(mass for record, mass in zip(augmentation_records, observed_masses)
                                          if record[2] == variant) for variant in VARIANTS}
            if (len(augmentation_records) != diagnostics["retained_groups"]
                    or len(source_consistency) != diagnostics["retained_sources"]
                    or len(retained_views) != diagnostics["retained_views"]
                    or len(retained_views - retained_originals) != diagnostics["views_without_original"]
                    or len(retained_views - retained_masks) != diagnostics["views_without_masks"]
                    or not np.isclose(sum(identity[0] for identity in source_consistency.values()),
                                      diagnostics["hangul_query_mass"], atol=1e-10, rtol=0)
                    or any(variant_counts[name] != diagnostics["variants"][name].get("stored", 0)
                           for name in VARIANTS)
                    or any(not np.isclose(variant_masses[name], diagnostics["variant_query_mass"][name],
                                          atol=1e-10, rtol=1e-12) for name in VARIANTS)):
                raise ValueError("Training augmentation counts disagree with cached groups")
        yield item


def load_training_arrays(directory: str | Path, *, feature_names: list[str] | None = None,
                         return_query_mass: bool = False) -> tuple:
    """Preallocate only selected columns; source shards stay unchanged on disk."""
    from .feature_selection import FeatureSelection

    metadata = read_metadata(directory)
    selection = FeatureSelection.from_names(metadata["feature_names"], feature_names)
    rows = int(metadata["counts"]["train_pairs"])
    queries = int(metadata["counts"]["train_groups"])
    x = np.empty((rows, len(selection.selected_features)), dtype=np.float32)
    y = np.empty(rows, dtype=np.uint8)
    groups = np.empty(queries, dtype=np.int32)
    language = np.empty(queries, dtype=bool)
    query_mass = (np.empty(queries, dtype=np.float64)
                  if metadata.get("training_augmentation") is not None else None)
    row_offset = query_offset = 0
    for shard in iter_shards(directory, "train"):
        row_stop = row_offset + len(shard["y"])
        query_stop = query_offset + len(shard["group"])
        if row_stop > rows or query_stop > queries:
            raise ValueError("Shard totals exceed metadata counts")
        x[row_offset:row_stop] = selection.apply(shard["X"], metadata["feature_names"])
        y[row_offset:row_stop] = shard["y"]
        groups[query_offset:query_stop] = shard["group"]
        language[query_offset:query_stop] = shard["has_hangul"]
        if query_mass is not None:
            query_mass[query_offset:query_stop] = shard["query_mass"]
        row_offset, query_offset = row_stop, query_stop
    if row_offset != rows or query_offset != queries:
        raise ValueError("Shard totals do not match metadata counts")
    result = x, y, groups, language, list(selection.selected_features)
    return (*result, query_mass) if return_query_mass else result
