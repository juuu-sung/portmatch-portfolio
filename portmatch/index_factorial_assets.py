"""Cross training-only search indexes and complete category models in memory.

Every runner shares the same trained ranker and feature selection. Search assets
come from one of the two already verified M/T pipelines; category statistics
come from either M or T independently. The saved pipeline is never rewired.
"""
from __future__ import annotations

import copy
import hashlib
import json

import joblib

from .index_scale_assets import (
    SUBSET_SEED, assert_unchanged, learned_assets_fingerprint, prepare_index_scale,
)


FACTOR_LEVELS = {
    "search80_category80": {"search": 80, "category": 80},
    "search80_category100": {"search": 80, "category": 100},
    "search100_category80": {"search": 100, "category": 80},
    "search100_category100": {"search": 100, "category": 100},
}
RUNNERS = tuple(FACTOR_LEVELS)


def _category_fingerprint(model):
    payload = json.dumps(model.to_dict(), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_factorial_wiring(context):
    """Reject detached or accidentally mutated references before/after inference.

    Catalog-only arrays may be shared, but each runner owns a distinct pipeline
    and feature-builder wrapper. Thus replacing categories cannot change the
    saved pipeline or the other runner using the same search index.
    """
    runners = context["runners"]
    sources, categories = context["search_pipelines"], context["category_models"]
    snapshots = context["_source_components"]
    base = context["matcher"]
    if (set(runners) != set(RUNNERS) or set(sources) != {80, 100}
            or set(categories) != {80, 100} or set(snapshots) != {80, 100}):
        raise ValueError("Factorial asset registries must contain exactly the four fixed cells")
    for level in (80, 100):
        source, snapshot = sources[level], snapshots[level]
        for attr, original in snapshot.items():
            if getattr(source, attr) is not original:
                raise ValueError("Source pipeline was rewired during factorial comparison")
        if (source.memory is not source.retriever.memory
                or source.categories is not categories[level]
                or source.feature_builder.category_model is not categories[level]):
            raise ValueError("Source components disagree on their training-only asset scope")
    if base.pipeline is not sources[100] or context["small_pipeline"] is not sources[80]:
        raise ValueError("Saved/full and selected/small pipeline references changed")
    if len({id(runner) for runner in runners.values()}) != 4:
        raise ValueError("Each factorial cell requires its own matcher wrapper")
    if len({id(runner.pipeline) for runner in runners.values()}) != 4:
        raise ValueError("Each factorial cell requires its own pipeline wrapper")
    if len({id(runner.pipeline.feature_builder) for runner in runners.values()}) != 4:
        raise ValueError("Each factorial cell requires its own feature-builder wrapper")
    for name, runner in runners.items():
        levels = FACTOR_LEVELS[name]
        source, category = sources[levels["search"]], categories[levels["category"]]
        pipeline, builder = runner.pipeline, runner.pipeline.feature_builder
        if (runner is base or pipeline is source or builder is source.feature_builder
                or runner.ranker is not base.ranker or runner.selection is not base.selection
                or runner.feature_transform is not base.feature_transform
                or runner.catalog_ids is not base.catalog_ids):
            raise ValueError("Factorial cells must share fixed weights/schema with independent wrappers")
        if (pipeline.catalog is not source.catalog or pipeline.memory is not source.memory
                or pipeline.retriever is not source.retriever
                or pipeline.retriever.memory is not pipeline.memory
                or pipeline.categories is not category or builder.category_model is not category):
            raise ValueError(f"Factorial cell {name} disagrees with its search/category levels")
        original_builder = source.feature_builder
        if set(vars(builder)) != set(vars(original_builder)):
            raise ValueError("Factorial feature builder has unexpected learned or mutable fields")
        for attr, value in vars(original_builder).items():
            if attr != "category_model" and getattr(builder, attr) is not value:
                raise ValueError("Catalog-only feature assets changed while mixing categories")
    return {"verified": True, "cells": 4, "shared_ranker_and_selection": True,
            "independent_pipeline_and_builder_wrappers": True,
            "source_pipeline_wiring_preserved": True,
            "whole_category_models_crossed": True}


def factorial_assets_fingerprint(context):
    """Snapshot learned assets and full retriever/TF-IDF state, excluding query caches.

    Canonical memory/category/catalog hashes use SHA-256 as in index-scale.
    Joblib's SHA-1 hashing covers numeric arrays, sparse TF-IDF matrices,
    vectorizers and lookup/edit/numeric retriever state in this local process.
    Name-feature query caches attached to pipeline wrappers are not fitted assets.
    """
    validate_factorial_wiring(context)
    return {
        "search_retriever_joblib_sha1": {str(level): joblib.hash(pipeline.retriever,
            hash_name="sha1") for level, pipeline in context["search_pipelines"].items()},
        "category_sha256": {str(level): _category_fingerprint(model)
            for level, model in context["category_models"].items()},
        "catalog_feature_assets_joblib_sha1": {str(level): joblib.hash({
            name: value for name, value in vars(pipeline.feature_builder).items()
            if name != "category_model"}, hash_name="sha1")
            for level, pipeline in context["search_pipelines"].items()},
        "runner_learned_assets_sha256": {name: learned_assets_fingerprint(runner.pipeline)
            for name, runner in context["runners"].items()},
    }


def prepare_index_factorial(model_directory, train_path, catalog_path, *,
                            subset_seed=SUBSET_SEED, expected_fold=None):
    """Build the fixed 80/100 search × 80/100 category diagnostic without training.

    The existing scale preparation owns canonical86, split/provenance checks,
    T/M fitting and V exclusion. This wrapper changes only in-memory bindings;
    it never fits on queries or serializes a new deployment index.
    """
    if expected_fold is not None and (type(expected_fold) is not int or expected_fold not in (0, 2)):
        raise ValueError("Factorial diagnostics support only frozen outer folds 0 and 2")
    context = prepare_index_scale(model_directory, train_path, catalog_path,
        fraction=0.8, subset_seed=subset_seed, expected_fold=expected_fold)
    if context["proof"]["coverage"]["fold"] not in (0, 2):
        raise ValueError("Factorial diagnostics support only frozen outer folds 0 and 2")
    base = context["matcher"]
    sources = {80: context["small_pipeline"], 100: base.pipeline}
    categories = {level: pipeline.categories for level, pipeline in sources.items()}
    source_components = {level: {name: getattr(pipeline, name) for name in
        ("catalog", "memory", "retriever", "categories", "feature_builder")}
        for level, pipeline in sources.items()}
    before_sources = {level: learned_assets_fingerprint(pipeline)
                      for level, pipeline in sources.items()}
    runners = {}
    for name, levels in FACTOR_LEVELS.items():
        runner = copy.copy(base)
        pipeline = copy.copy(sources[levels["search"]])
        pipeline.feature_builder = copy.copy(pipeline.feature_builder)
        pipeline.categories = categories[levels["category"]]
        pipeline.feature_builder.category_model = pipeline.categories
        runner.pipeline = pipeline
        runners[name] = runner
    context.update(runners=runners, search_pipelines=sources,
                   category_models=categories, _source_components=source_components)
    wiring = validate_factorial_wiring(context)
    if before_sources != {level: learned_assets_fingerprint(pipeline)
                          for level, pipeline in sources.items()}:
        raise ValueError("Source training assets changed while constructing factorial cells")
    context["factorial_proof"] = {
        "name": "fixed_ranker_search_category_factorial_v1", "verified": True,
        "factor_levels": copy.deepcopy(FACTOR_LEVELS), "wiring": wiring,
        "training_scope_proof": copy.deepcopy(context["proof"]),
        "asset_fingerprints": factorial_assets_fingerprint(context),
        "source_assets_preserved_during_construction": True,
        "search_scope": "lookup, aliases, TF-IDF and all retrieval channels from selected M or full T",
        "category_scope": "entire CategoryCompatibility model from selected M or full T; not four overwritten columns",
        "heldout_exclusion": "Both search and category scopes exclude every held-out V row and vessel",
        "interpretation": "Conditional asset effects at fixed model weights and feature schema; not a retraining experiment or direct Public score explanation.",
    }
    assert_unchanged(context["protected_files"])
    return context
