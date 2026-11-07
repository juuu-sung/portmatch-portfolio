"""Explicit dispatch for versioned derived caches without changing legacy code.

Each implementation owns its schema, code hash and parent/array proof. Unknown
transform names fail closed; legacy caches and models keep their original hashes.
"""
from __future__ import annotations

import json
from pathlib import Path


def implementation(spec):
    if not isinstance(spec, dict):
        raise ValueError("Missing or invalid feature transform metadata")
    if spec.get("name") == "tonnage_min_rank_v1":
        from . import tonnage_rank
        return tonnage_rank
    if spec.get("name") == "token_phonetic_v1":
        from . import name_feature_cache
        return name_feature_cache
    if spec.get("name") == "token_phonetic_ch_v1":
        from . import ch_variant_cache
        return ch_variant_cache
    raise ValueError("Unknown or changed feature transform; regenerate with a supported version")


def check_transform_metadata(metadata):
    from .gt_augmentation import check_augmentation_metadata
    from .robust_retrieval import check_retrieval_metadata

    check_retrieval_metadata(metadata)
    augmentation = check_augmentation_metadata(metadata)
    spec = metadata.get("feature_transform")
    if spec is not None:
        module = implementation(spec)
        if augmentation is not None and spec.get("name") != "token_phonetic_v1":
            raise ValueError("GT augmentation only supports the fixed token-phonetic feature transform")
        module.check_transform_metadata(metadata)
        return
    from .tonnage_rank import check_transform_metadata as check_legacy
    from .name_features import FEATURE_NAMES
    from .ch_variant_features import FEATURE_NAMES as CH_FEATURE_NAMES

    check_legacy(metadata)
    if (set(FEATURE_NAMES) | set(CH_FEATURE_NAMES)).intersection(metadata.get("feature_names", [])):
        raise ValueError("Name feature schema is missing its transform metadata")


def check_parent_transform(parent_spec, child_spec):
    """Only explicitly supported immediate-parent relationships are comparable."""
    if child_spec is None:
        raise ValueError("Only a source-cache baseline and its recognized derived experiment may differ")
    module = implementation(child_spec)
    if child_spec["name"] == "token_phonetic_ch_v1":
        from .name_feature_cache import transform_spec

        valid = parent_spec == transform_spec() and child_spec == module.transform_spec()
    else:
        valid = parent_spec is None
    if not valid:
        raise ValueError("Only a source-cache baseline and its recognized derived experiment may differ")


def transform_features(spec, pipeline, row, candidates, features):
    return implementation(spec).transform_features(pipeline, row, candidates, features)


def verify_derivation(directory, parent_metadata_sha256, expected_parent_dataset=None):
    metadata = json.loads((Path(directory) / "metadata.json").read_text(encoding="utf-8"))
    check_transform_metadata(metadata)
    return implementation(metadata.get("feature_transform")).verify_derivation(
        directory, parent_metadata_sha256, expected_parent_dataset)


def check_selected_features(spec, baseline, experiment):
    """Require identical shared selected columns, permitting only declared changes."""
    module = implementation(spec)
    if spec["name"] == "tonnage_min_rank_v1":
        left = [n for n in baseline if n != module.SOURCE_FEATURE]
        right = [n for n in experiment if n != module.TARGET_FEATURE]
        valid = left == right and module.TARGET_FEATURE in experiment
        kind = "verified_tonnage_min_rank_derivation"
    elif spec["name"] == "token_phonetic_v1":
        from .name_features import FEATURE_NAMES
        added = set(FEATURE_NAMES)
        # Selecting none of the new columns is the exact baseline69 control.
        valid = baseline == [n for n in experiment if n not in added]
        kind = "verified_token_phonetic_derivation"
    else:
        from .ch_variant_features import FEATURE_NAMES

        added = set(FEATURE_NAMES)
        valid = baseline == [n for n in experiment if n not in added]
        kind = "verified_token_phonetic_ch_derivation"
    if not valid:
        raise ValueError("Derived feature comparison must retain the same shared selected features")
    return kind
