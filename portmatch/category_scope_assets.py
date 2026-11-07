"""Frozen full-search runners with training-only 80/100% category statistics.

The verified scale preparation is reused unchanged, including its exact vessel
subset. Its smaller search index is built for provenance but never queried here.
Only the entire CategoryCompatibility object is exchanged between wrappers.
"""
from __future__ import annotations

import copy

import joblib

from .index_factorial_assets import _category_fingerprint
from .index_scale_assets import (
    SUBSET_SEED, assert_unchanged, learned_assets_fingerprint, prepare_index_scale,
)


REFERENCE = "search100_category100"
EXPERIMENT = "search100_category80"
FACTOR_LEVELS = {
    REFERENCE: {"search": 100, "category": 100},
    EXPERIMENT: {"search": 100, "category": 80},
}
RUNNERS = tuple(FACTOR_LEVELS)


def validate_category_wiring(context):
    """Require one full search source, fixed ranker and separate category wrappers."""
    runners = context["runners"]
    sources, categories = context["search_pipelines"], context["category_models"]
    snapshots = context["_source_components"]
    base, small = context["matcher"], context["small_pipeline"]
    if (set(runners) != set(RUNNERS) or set(sources) != {100}
            or set(categories) != {80, 100} or set(snapshots) != {80, 100}):
        raise ValueError("Category-scope registries require exactly two fixed full-search cells")
    if base.pipeline is not sources[100]:
        raise ValueError("Saved full-search pipeline reference changed")
    for level, source in ((80, small), (100, base.pipeline)):
        for attr, original in snapshots[level].items():
            if getattr(source, attr) is not original:
                raise ValueError("Source pipeline was rewired during category comparison")
        if (source.memory is not source.retriever.memory
                or source.categories is not categories[level]
                or source.feature_builder.category_model is not categories[level]):
            raise ValueError("Source components disagree on their training-only asset scope")
    for accessor in (lambda runner: runner, lambda runner: runner.pipeline,
                     lambda runner: runner.pipeline.feature_builder):
        if len({id(accessor(runner)) for runner in runners.values()}) != 2:
            raise ValueError("Each category cell requires independent matcher/pipeline/builder wrappers")
    source = sources[100]
    for name, runner in runners.items():
        pipeline, builder = runner.pipeline, runner.pipeline.feature_builder
        category = categories[FACTOR_LEVELS[name]["category"]]
        if (runner is base or pipeline is source or builder is source.feature_builder
                or runner.ranker is not base.ranker or runner.selection is not base.selection
                or runner.feature_transform is not base.feature_transform
                or runner.catalog_ids is not base.catalog_ids):
            raise ValueError("Category cells must share frozen weights/schema with independent wrappers")
        if (pipeline.catalog is not source.catalog or pipeline.memory is not source.memory
                or pipeline.retriever is not source.retriever
                or pipeline.retriever.memory is not pipeline.memory
                or pipeline.categories is not category or builder.category_model is not category):
            raise ValueError(f"Category cell {name} disagrees with its fixed search/category scope")
        original_builder = source.feature_builder
        if set(vars(builder)) != set(vars(original_builder)):
            raise ValueError("Category feature builder has unexpected learned or mutable fields")
        for attr, value in vars(original_builder).items():
            if attr != "category_model" and getattr(builder, attr) is not value:
                raise ValueError("Catalog-only feature assets changed while replacing categories")
    return {"verified": True, "cells": 2, "search_fixed_at_100": True,
            "shared_full_retriever_and_memory": True, "shared_ranker_and_selection": True,
            "independent_pipeline_and_builder_wrappers": True,
            "source_pipeline_wiring_preserved": True, "whole_category_models_crossed": True}


def category_assets_fingerprint(context):
    """Check fitted state before/after inference in the same process.

    The joblib hashes cover the full retriever including TF-IDF arrays. They are
    not portable cross-process canonical hashes; source file SHA-256 and category
    SHA-256 provide the portable provenance. Query caches are excluded.
    """
    validate_category_wiring(context)
    full = context["search_pipelines"][100]
    return {
        "search_retriever_joblib_sha1": {"100": joblib.hash(full.retriever, hash_name="sha1")},
        "category_sha256": {str(level): _category_fingerprint(model)
            for level, model in context["category_models"].items()},
        "catalog_feature_assets_joblib_sha1": {"100": joblib.hash({
            name: value for name, value in vars(full.feature_builder).items()
            if name != "category_model"}, hash_name="sha1")},
        "runner_learned_assets_sha256": {name: learned_assets_fingerprint(runner.pipeline)
            for name, runner in context["runners"].items()},
    }


def prepare_category_scope(model_directory, train_path, catalog_path, *,
                           subset_seed=SUBSET_SEED, expected_fold=None):
    """Prepare C/D for any saved outer fold, preserving the frozen selection seed."""
    if expected_fold is not None and (type(expected_fold) is not int or expected_fold not in range(5)):
        raise ValueError("Category-scope diagnostics require a frozen outer fold from 0 through 4")
    if type(subset_seed) is not int or subset_seed != SUBSET_SEED:
        raise ValueError(f"Category-scope diagnostics require the frozen subset seed {SUBSET_SEED}")
    context = prepare_index_scale(model_directory, train_path, catalog_path,
        fraction=.8, subset_seed=subset_seed, expected_fold=expected_fold)
    fold = context["proof"]["coverage"]["fold"]
    if type(fold) is not int or fold not in range(5):
        raise ValueError("Category-scope diagnostics require a frozen outer fold from 0 through 4")
    base, small = context["matcher"], context["small_pipeline"]
    source = base.pipeline
    sources = {80: small, 100: source}
    before = {level: learned_assets_fingerprint(pipeline) for level, pipeline in sources.items()}
    categories = {level: pipeline.categories for level, pipeline in sources.items()}
    snapshots = {level: {name: getattr(pipeline, name) for name in
        ("catalog", "memory", "retriever", "categories", "feature_builder")}
        for level, pipeline in sources.items()}
    runners = {}
    for name, levels in FACTOR_LEVELS.items():
        runner, pipeline = copy.copy(base), copy.copy(source)
        pipeline.feature_builder = copy.copy(source.feature_builder)
        pipeline.categories = categories[levels["category"]]
        pipeline.feature_builder.category_model = pipeline.categories
        runner.pipeline = pipeline
        runners[name] = runner
    context.update(runners=runners, search_pipelines={100: source},
                   category_models=categories, _source_components=snapshots)
    wiring = validate_category_wiring(context)
    if before != {level: learned_assets_fingerprint(pipeline) for level, pipeline in sources.items()}:
        raise ValueError("Source assets changed while constructing category-scope cells")
    context["category_proof"] = {
        "name": "fixed_ranker_full_search_category_scope_v1", "verified": True,
        "factor_levels": copy.deepcopy(FACTOR_LEVELS), "wiring": wiring,
        "training_scope_proof": copy.deepcopy(context["proof"]),
        "asset_fingerprints": category_assets_fingerprint(context),
        "source_assets_preserved_during_construction": True,
        "reference_runner": REFERENCE, "experiment_runner": EXPERIMENT,
        "effect_direction": "category80 minus category100 at fixed search100 (C minus D)",
        "search_scope": "Both runners use the same unchanged full-T lookup, aliases, TF-IDF and every retrieval channel",
        "category_scope": "Whole CategoryCompatibility from selected M (80% vessels) or full T; not overwritten feature columns",
        "preparation_note": "Verified scale preparation also builds the M search index for provenance; neither runner queries it",
        "heldout_exclusion": "Both search and category models exclude all held-out V rows and vessels",
        "interpretation": "Conditional category-scope effect with frozen weights/schema; not retraining or a direct Public-score explanation",
    }
    assert_unchanged(context["protected_files"])
    return context
