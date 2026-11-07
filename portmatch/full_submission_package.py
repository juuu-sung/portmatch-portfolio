"""Full-public-training submission profile and minimal runtime metadata.

Packaging verifies the completed model's full-training proof and unchanged
shared cache assets. The trainer already checked every training shard; no
training matrices or source rows are copied into the submission.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


_FULL_DATASET_FIELDS = (
    "training_scope", "full_training", "full_training_assets", "full_training_population",
    "full_training_pair_proof",
    "outer_split", "outer_folds", "outer_fold", "valid_shards", "inner_split", "inner_folds",
    "max_valid_queries", "max_train_queries", "sampled", "index_path", "catalog_ids_path",
    "counts", "seed", "sampling",
)


def _hash_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected an unchanged regular full-training asset")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def add_full_runtime_metadata(minimal: dict, summary: dict) -> None:
    """Copy only aggregate facts, canonical basenames and proof hashes."""
    from .full_ranking import check_full_model_summary

    check_full_model_summary(summary)
    for name in _FULL_DATASET_FIELDS:
        minimal["dataset"][name] = copy.deepcopy(summary["dataset"][name])
    for name in ("training_scope", "full_training_trainer", "validation", "ranker"):
        minimal[name] = copy.deepcopy(summary[name])
    check_full_model_summary(minimal)


def validate_full_submission(summary: dict, model_state: dict, model_directory: Path,
                             cache_directory: Path | None = None) -> None:
    """Fail closed on incomplete/limited caches, modified assets or wrong profile."""
    from .full_ranking import check_full_model_summary
    from .full_training_data import check_full_training_metadata

    check_full_model_summary(summary, model_state=model_state)
    dataset = summary["dataset"]
    check_full_training_metadata(dataset)
    if dataset.get("sampled") is not False or dataset.get("max_train_queries") is not None:
        raise ValueError("Diagnostic full-training caches cannot be packaged")
    if dataset["counts"]["selected_training_queries"] != dataset["full_training_population"]["rows"]:
        raise ValueError("Full-training model did not select every public training source")
    proof = summary.get("full_training_proof", {})
    population = dataset["full_training_population"]
    if (proof.get("verified") is not True or proof.get("original_population_verified") is not True
            or proof.get("training_scope") != "full_public_train"
            or proof.get("public_training_rows") != population["rows"]
            or proof.get("public_training_vessels") != population["vessels"]
            or proof.get("selected_sources") != population["rows"]
            or proof.get("final_index_original_rows") != population["rows"]
            or proof.get("training_groups") != dataset["counts"]["train_groups"]
            or proof.get("training_pairs") != dataset["counts"]["train_pairs"]
            or proof.get("retained_sources") != dataset["training_augmentation_diagnostics"]["retained_sources"]
            or proof.get("dropped_sources") != dataset["training_augmentation_diagnostics"]["dropped_sources"]
            or proof.get("held_out_validation_rows") != 0 or proof.get("sampled") is not False):
        raise ValueError("Full-training model lacks a complete original-population training proof")
    derivation = summary.get("full_derivation_proof", {})
    if (derivation.get("verified") is not True
            or derivation.get("transform") != "token_phonetic_v1"
            or derivation.get("derived_metadata_sha256") != summary.get("cache_metadata_sha256")
            or derivation.get("parent_metadata_sha256") != dataset["derivation"]["parent_metadata_sha256"]
            or derivation.get("manifest_sha256") != dataset["derivation"]["manifest_sha256"]
            or derivation.get("verified_shards") != dataset["full_training_pair_proof"]["shards"]):
        raise ValueError("Full-training model lacks its checked feature-derivation proof")
    # A full model has no external validation labels or reported accuracy.
    if any(name in summary for name in ("metrics", "validation_metrics", "validation_ids")):
        raise ValueError("A full-training bundle must not claim held-out accuracy")
    if (Path(model_directory) / "validation_predictions.csv").exists():
        raise ValueError("Full-training bundle contains unexpected held-out predictions")
    cache = Path(cache_directory or summary.get("cache_directory", ""))
    metadata = json.loads((cache / "metadata.json").read_text("utf-8"))
    if metadata != dataset or _hash_file(cache / "metadata.json") != summary.get("cache_metadata_sha256"):
        raise ValueError("Full-training cache metadata differs from its trained proof")
    for name, expected in dataset["full_training_assets"].items():
        if name not in {"index.joblib", "catalog_ids.npy", "partitions.csv", "querylog.csv"}:
            raise ValueError("Unexpected full-training proof asset")
        if _hash_file(cache / name) != expected:
            raise ValueError("Full-training shared asset changed after training")
    if (_hash_file(Path(model_directory) / "index.joblib")
            != dataset["full_training_assets"]["index.joblib"]):
        raise ValueError("Full-model index differs from all-original-public-training proof")


def build_full_submission(model_directory, output, *, runtime_directory=None, cache_directory=None):
    """Build only the fixed full-public-training robust/86-feature/0.5 profile."""
    from .submission_package import build_submission

    return build_submission(model_directory, output, runtime_directory=runtime_directory,
                            cache_directory=cache_directory, allow_full=True)


def verify_full_copied_code(payload: dict[str, bytes], summary: dict) -> None:
    """Bind the copied preparation/training implementation to its saved proof."""
    checks = [
        (("full_training_data.py", "training_data.py", "validation.py", "sampling.py",
          "gt_augmentation.py", "name_feature_cache.py"),
         summary["dataset"]["full_training"]["generation_code_sha256"]),
        (("full_ranking.py", "ranking.py"), summary["full_training_trainer"]["code_sha256"]),
    ]
    for filenames, expected in checks:
        digest = hashlib.sha256()
        for name in filenames:
            digest.update(name.encode("utf-8"))
            digest.update(payload[f"model/portmatch/{name}"])
        if digest.hexdigest() != expected:
            raise ValueError("Full-training source changed while packaging")
