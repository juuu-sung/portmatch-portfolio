"""Versioned training-only GT masks and source-balanced query masses.

Masks never fit lookup assets and never modify query names. They are applied
only after the original rows have been assigned to their outer/inner folds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


AUGMENTATION_NAME = "gt_missing_v1"
VARIANTS = ("original", "both_gt_missing", "international_gt_missing", "gross_gt_missing")
_MASKS = {
    "both_gt_missing": ("gross_tonnage", "international_gross_tonnage"),
    "international_gt_missing": ("international_gross_tonnage",),
    "gross_gt_missing": ("gross_tonnage",),
}


@dataclass(frozen=True)
class GTMissingAugmentationConfig:
    original_mass: float = 0.5

    def __post_init__(self):
        value = self.original_mass
        if (isinstance(value, (bool, str)) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 < value < 1):
            raise ValueError("GT augmentation original_mass must be finite and strictly between 0 and 1")


def augmentation_spec(config: GTMissingAugmentationConfig) -> dict:
    if not isinstance(config, GTMissingAugmentationConfig):
        raise TypeError("Expected GTMissingAugmentationConfig")
    digest = hashlib.sha256()
    for name in ("gt_augmentation.py", "training_data.py"):
        digest.update(name.encode("utf-8"))
        digest.update(Path(__file__).with_name(name).read_bytes())
    return {
        "name": AUGMENTATION_NAME, "version": 1, "config": asdict(config),
        "generation_code_sha256": digest.hexdigest(), "variants": list(VARIANTS),
        "scope": "training queries only; split and fit original rows before masks",
        "retrieval": "robust profile; fresh retrieval/features/negative sampling for every retained mask",
        "deduplication": "skip unchanged masks; first unique GT input in listed variant order per source/view",
        "missing_target": "skip without target injection; log every attempted or skipped variant",
        "mass_policy": "source total 1; equal retained OOF views; original_mass to original and remaining mass equally to retained masks; redistribute within view when original or masks unavailable",
        "validation": "original inputs only; unchanged all-candidate validation",
        "group_arrays": ["augmentation_variant", "query_mass"],
    }


def validate_training_augmentation(spec: dict) -> GTMissingAugmentationConfig:
    if not isinstance(spec, dict) or not isinstance(spec.get("config"), dict):
        raise ValueError("Invalid training_augmentation specification")
    try:
        config = GTMissingAugmentationConfig(**spec["config"])
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid GT augmentation configuration") from error
    if spec != augmentation_spec(config):
        raise ValueError("Unknown or changed training augmentation; regenerate its feature cache")
    return config


def check_augmentation_metadata(metadata: dict) -> GTMissingAugmentationConfig | None:
    spec = metadata.get("training_augmentation")
    companions = {"training_augmentation_diagnostics", "training_augmentation_assets"}
    if spec is None:
        if companions.intersection(metadata):
            raise ValueError("Augmentation diagnostics/assets are missing training_augmentation metadata")
        return None
    config = validate_training_augmentation(spec)
    if (metadata.get("retrieval", {}).get("profile") != "robust"
            or not isinstance(metadata.get("retrieval_extension"), dict)):
        raise ValueError("GT augmentation requires the versioned robust retrieval profile")
    if not companions.issubset(metadata):
        raise ValueError("Augmented cache is missing diagnostics or asset proof")
    assets = metadata["training_augmentation_assets"]
    if (not isinstance(assets, dict)
            or set(assets) != {"querylog.csv", "partitions.csv", "training_groups_sha256"}
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in assets.values())):
        raise ValueError("Invalid training augmentation asset proof")
    diagnostics = metadata["training_augmentation_diagnostics"]
    required_counts = {"selected_sources", "retained_sources", "dropped_sources", "retained_views",
                       "retained_groups", "views_without_original", "views_without_masks"}
    required_masses = {"source_mass_min", "source_mass_max", "total_query_mass",
                       "hangul_query_mass", "non_hangul_query_mass"}
    if (not isinstance(diagnostics, dict) or diagnostics.get("mass_policy_verified") is not True
            or set(diagnostics) != required_counts | required_masses | {"variants", "variant_query_mass", "mass_policy_verified"}
            or any(type(diagnostics.get(key)) is not int or diagnostics[key] < 0 for key in required_counts)
            or any(isinstance(diagnostics.get(key), bool)
                   or not isinstance(diagnostics.get(key), (float, int))
                   or not math.isfinite(diagnostics[key]) or diagnostics[key] < 0 for key in required_masses)):
        raise ValueError("Missing verified training augmentation mass diagnostics")
    statuses = {"stored", "skipped_missing_target", "skipped_without_negative",
                "skipped_noop", "skipped_duplicate"}
    variants = diagnostics["variants"]
    variant_mass = diagnostics["variant_query_mass"]
    if (not isinstance(variants, dict) or set(variants) != set(VARIANTS)
            or any(not isinstance(counts, dict) or not set(counts).issubset(statuses)
                   or any(type(count) is not int or count < 0 for count in counts.values())
                   for counts in variants.values())):
        raise ValueError("Invalid training augmentation variant diagnostics")
    if (not isinstance(variant_mass, dict) or set(variant_mass) != set(VARIANTS)
            or any(isinstance(mass, bool) or not isinstance(mass, (float, int))
                   or not math.isfinite(mass) or mass < 0 for mass in variant_mass.values())
            or not math.isclose(sum(variant_mass.values()), diagnostics["total_query_mass"], abs_tol=1e-10)):
        raise ValueError("Invalid training augmentation variant mass diagnostics")
    if (diagnostics["selected_sources"] != diagnostics["retained_sources"] + diagnostics["dropped_sources"]
            or diagnostics["retained_groups"] != sum(counts.get("stored", 0) for counts in variants.values())
            or diagnostics["selected_sources"] != metadata.get("counts", {}).get("selected_training_queries")
            or diagnostics["retained_groups"] != metadata.get("counts", {}).get("train_groups")
            or not math.isclose(diagnostics["total_query_mass"], diagnostics["retained_sources"], abs_tol=1e-10)
            or not math.isclose(diagnostics["hangul_query_mass"] + diagnostics["non_hangul_query_mass"],
                                diagnostics["total_query_mass"], abs_tol=1e-10)
            or not math.isclose(diagnostics["source_mass_min"], 1, abs_tol=1e-12)
            or not math.isclose(diagnostics["source_mass_max"], 1, abs_tol=1e-12)):
        raise ValueError("Training augmentation diagnostics violate source mass/count invariants")
    return config


def _gt_key(row: dict) -> tuple:
    result = []
    for field in ("gross_tonnage", "international_gross_tonnage"):
        value = row.get(field)
        missing = bool(pd.isna(value)) or (isinstance(value, str) and not value.strip())
        result.append(None if missing else value)
    return tuple(result)


def query_variants(row: dict):
    """Yield (name, independent row copy, status) including rejected masks."""
    seen = {_gt_key(row)}
    original_key = _gt_key(row)
    yield "original", dict(row), "ready"
    for name, fields in _MASKS.items():
        masked = dict(row)
        for field in fields:
            masked[field] = np.nan
        key = _gt_key(masked)
        status = ("skipped_noop" if key == original_key else
                  "skipped_duplicate" if key in seen else "ready")
        seen.add(key)
        yield name, masked, status


def query_identity(sample_id: str, query_id: str, variant: str, views) -> tuple[str, str]:
    """Parse after the full source prefix, so source IDs may contain '::'."""
    if variant not in VARIANTS:
        raise ValueError("Unknown cached augmentation variant")
    matches = [view for view in views if query_id == sample_id + "::" + view
               + ("" if variant == "original" else "::" + variant)]
    if len(matches) != 1:
        raise ValueError("Augmented query identity disagrees with source, view or variant")
    return sample_id, matches[0]


def expected_query_masses(records, config: GTMissingAugmentationConfig) -> dict:
    """Only group identities are retained in memory, never candidate matrices."""
    by_source = {}
    identities = set()
    for source, view, variant in records:
        identity = source, view, variant
        if identity in identities:
            raise ValueError("Duplicate augmented source/view/variant identity")
        identities.add(identity)
        by_source.setdefault(source, {}).setdefault(view, []).append(variant)
    result = {}
    for source, views in by_source.items():
        view_mass = 1.0 / len(views)
        for view, variants in views.items():
            masks = [variant for variant in variants if variant != "original"]
            has_original = "original" in variants
            original_mass = config.original_mass if masks else 1.0
            if has_original:
                result[source, view, "original"] = view_mass * original_mass
            if masks:
                mask_mass = view_mass * ((1 - original_mass) if has_original else 1.0) / len(masks)
                for variant in masks:
                    result[source, view, variant] = mask_mass
    return result


def update_group_digest(digest, shard) -> None:
    """Hash ordered query metadata independently of shard boundaries/features."""
    import json

    for index in range(len(shard["group"])):
        value = [str(shard["sample_id"][index]), str(shard["query_id"][index]),
                 str(shard["augmentation_variant"][index]), float(shard["query_mass"][index]),
                 bool(shard["has_hangul"][index]), int(shard["target_catalog_index"][index]),
                 int(shard["group"][index])]
        digest.update(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8") + b"\n")
