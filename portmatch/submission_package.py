"""Build a frozen 86-feature DACON submission from trusted local artifacts.

This command never trains, predicts, or reads feature shards. It verifies the
saved native model, feature-code hashes, and index hash in the existing cache
derivation proof, then packages only executable source and fixed model assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

from .feature_selection import FeatureSelection
from .name_feature_cache import transform_spec
from .gt_augmentation import check_augmentation_metadata
from .ranker_commands import _check_feature_code, _check_training_weighting, feature_code_hash
from .ranking import VesselRanker


_HEX256 = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_FILES = ("model.txt", "ranker.json", "index.joblib")
_HASHED_FEATURE_FILES = ("features.py", "text.py", "memory.py", "retrieval.py", "pipeline.py")
_HASHED_NAME_FILES = ("name_feature_cache.py", "name_features.py")
_HASHED_ROBUST_FILES = ("robust_retrieval.py", "retrieval.py", "pipeline.py", "memory.py",
                        "features.py", "text.py", "name_features.py")
_HASHED_AUGMENTATION_FILES = ("gt_augmentation.py", "training_data.py")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def _read_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular local file: {path.name}")
    return path.read_bytes()


def _hash_field(value, name: str) -> str:
    if not isinstance(value, str) or not _HEX256.fullmatch(value):
        raise ValueError(f"Missing or invalid {name} hash")
    return value


def _verified_index_hash(summary: dict, index_bytes: bytes, cache_directory: Path | None) -> str:
    """Verify the saved index without rereading large train/validation shards."""
    cache = Path(cache_directory or summary.get("cache_directory", ""))
    expected_metadata = _hash_field(summary.get("cache_metadata_sha256"), "cache metadata")
    metadata_bytes = _read_file(cache / "metadata.json")
    if _sha256(metadata_bytes) != expected_metadata:
        raise ValueError("Cache metadata hash differs from the trained bundle")
    metadata = json.loads(metadata_bytes)
    if metadata != summary["dataset"]:
        raise ValueError("Saved dataset metadata differs from its cache")
    derivation = metadata["derivation"]
    relative = derivation.get("manifest_path")
    # This builder accepts the project's known fixed layout, never arbitrary
    # paths embedded in a model's JSON metadata.
    if relative != "derivation.json" or metadata.get("index_path", "index.joblib") != "index.joblib":
        raise ValueError("Unexpected cache asset path")
    proof_bytes = _read_file(cache / relative)
    if _sha256(proof_bytes) != _hash_field(derivation.get("manifest_sha256"), "derivation manifest"):
        raise ValueError("Cache derivation proof hash mismatch")
    proof = json.loads(proof_bytes)
    if (proof.get("schema_version") != 1 or proof.get("complete") is not True
            or proof.get("transform") != metadata["feature_transform"]
            or proof.get("parent_metadata_sha256") != derivation.get("parent_metadata_sha256")):
        raise ValueError("Invalid or incomplete cache derivation proof")
    expected = _hash_field(proof.get("derived_file_sha256", {}).get("index.joblib"), "derived index")
    if expected != proof.get("source_file_sha256", {}).get("index.joblib"):
        raise ValueError("The transformed cache changed the frozen retrieval index")
    if _sha256(index_bytes) != expected:
        raise ValueError("Saved model index differs from the verified training index")
    if metadata.get("training_augmentation") is not None:
        if metadata.get("retrieval_index_sha256") != expected:
            raise ValueError("Robust index hash differs from the verified training index")
        for name in ("querylog.csv", "partitions.csv"):
            asset_hash = metadata["training_augmentation_assets"][name]
            if (proof.get("source_file_sha256", {}).get(name) != asset_hash
                    or proof.get("derived_file_sha256", {}).get(name) != asset_hash
                    or _sha256(_read_file(cache / name)) != asset_hash):
                raise ValueError("Augmentation asset proof differs from the verified cache")
    return expected


def _minimal_summary(summary: dict, model_state: dict) -> dict:
    """Allowlist inference metadata; omit paths, reports, query IDs and metrics."""
    dataset = summary["dataset"]
    provenance = {name: _hash_field(dataset.get("provenance", {}).get(name), name)
                  for name in ("train_sha256", "catalog_sha256", "feature_code_sha256")}
    derivation = {name: _hash_field(dataset.get("derivation", {}).get(name), name)
                  for name in ("manifest_sha256", "parent_metadata_sha256")}
    minimal = {
        "schema_version": 1,
        "complete": True,
        "dataset": {
            "schema_version": 1, "complete": True,
            "feature_names": dataset["feature_names"],
            "feature_transform": dataset["feature_transform"],
            "derivation": derivation,
            "provenance": provenance,
        },
        "feature_selection": summary["feature_selection"],
        "trained_feature_names": model_state["feature_names"],
    }
    if dataset.get("retrieval_extension") is not None:
        # These are fixed configurations, hashes and aggregate training counts.
        # Runtime checks require them; no source paths, records or labels travel.
        for name in ("retrieval", "retrieval_extension", "retrieval_index_sha256",
                     "training_augmentation", "training_augmentation_diagnostics",
                     "training_augmentation_assets"):
            minimal["dataset"][name] = dataset[name]
        minimal["dataset"]["counts"] = {
            name: dataset["counts"][name]
            for name in ("selected_training_queries", "train_groups", "train_pairs")}
        minimal["ranker"] = {"hangul_weight": summary["ranker"]["hangul_weight"]}
        minimal["training_weighting"] = summary["training_weighting"]
    if dataset.get("training_scope") == "full_public_train":
        from .full_submission_package import add_full_runtime_metadata

        add_full_runtime_metadata(minimal, summary)
    return minimal


def _verify_copied_code(payload: dict[str, bytes], expected_base: str, expected_names: str,
                        summary: dict | None = None) -> None:
    checks = [(_HASHED_FEATURE_FILES, expected_base), (_HASHED_NAME_FILES, expected_names)]
    if summary is not None and summary["dataset"].get("retrieval_extension") is not None:
        dataset = summary["dataset"]
        checks.extend(((_HASHED_ROBUST_FILES, dataset["retrieval_extension"]["code_sha256"]),
                       (_HASHED_AUGMENTATION_FILES,
                        dataset["training_augmentation"]["generation_code_sha256"])))
        if (_sha256(payload["model/portmatch/ranking.py"])
                != summary["training_weighting"]["ranking_code_sha256"]):
            raise ValueError("Training weighting source changed while packaging")
    for files, expected in checks:
        digest = hashlib.sha256()
        for name in files:
            digest.update(name.encode())
            digest.update(payload[f"model/portmatch/{name}"])
        if digest.hexdigest() != expected:
            raise ValueError("Source files changed while the submission was being packaged")


def build_submission(model_directory: str | Path, output: str | Path, *,
                     runtime_directory: str | Path | None = None,
                     cache_directory: str | Path | None = None,
                     allow_robust: bool = False, allow_full: bool = False) -> dict:
    """Create one deterministic ZIP; an existing destination is never replaced.

    Model artifacts must be trusted local project outputs. The index is copied
    verbatim and verified by its training proof; this builder does not unpickle
    it. The runtime loads it only after verifying the package manifest.
    """
    model_directory = Path(model_directory).resolve()
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Submission already exists: {output}")
    if output.suffix.lower() != ".zip":
        raise ValueError("Submission output must end in .zip")
    if model_directory.is_symlink() or not model_directory.is_dir():
        raise ValueError("Choose a trusted local model directory")
    source = Path(__file__).parent
    runtime = Path(runtime_directory) if runtime_directory is not None else source.parent / "submission"
    summary_bytes = _read_file(model_directory / "training_summary.json")
    summary = json.loads(summary_bytes)
    dataset = summary.get("dataset", {})
    full = ("training_scope" in summary or "training_scope" in dataset
            or dataset.get("outer_split") == "none"
            or any(key.startswith("full_training") for key in dataset)
            or any(key.startswith("full_training") for key in summary))
    if full and not allow_full:
        raise ValueError("Full-public-training packaging requires explicit --allow-full")
    if allow_full and not full:
        raise ValueError("--allow-full requires a full-public-training model")
    if (summary.get("complete") is not True or summary.get("schema_version") != 1
            or dataset.get("complete") is not True or dataset.get("schema_version") != 1):
        raise ValueError("Model bundle and source dataset must be complete schema version 1")
    if dataset.get("feature_transform") != transform_spec():
        raise ValueError("Only the frozen token_phonetic_v1 baseline is supported")
    robust = dataset.get("retrieval_extension") is not None
    if robust and not (allow_robust or allow_full):
        raise ValueError("Retrieval experiments require a separately validated submission package")
    if dataset.get("training_reweighting") is not None:
        raise ValueError("Reweighted model packaging needs a separately validated submission profile")
    _check_feature_code(dataset)
    if robust:
        augmentation = check_augmentation_metadata(dataset)
        if (dataset.get("retrieval", {}).get("profile") != "robust"
                or augmentation is None or augmentation.original_mass != 0.5):
            raise ValueError("The opt-in robust submission requires original-mass 0.5 GT augmentation")
        _check_training_weighting(summary)
    selection = FeatureSelection.from_dict(summary["feature_selection"])
    expected = tuple(name for name in dataset["feature_names"] if name != "retrieval_tonnage_rank")
    if (len(expected) != 86 or selection.selected_features != expected
            or selection.source_feature_names != tuple(dataset["feature_names"])):
        raise ValueError("Expected exactly 86 features, excluding only retrieval_tonnage_rank")
    model = VesselRanker.load(model_directory)
    if model.feature_names != list(expected) or summary.get("trained_feature_names") != list(expected):
        raise ValueError("Model feature schema does not match the frozen 86-feature baseline")
    if robust and (model.config.n_jobs != 6
                   or model.config.hangul_weight != summary["ranker"]["hangul_weight"]):
        raise ValueError("Robust submission requires six threads and matching saved language weighting")
    model_bytes = {name: _read_file(model_directory / name) for name in _MODEL_FILES}
    model_state = json.loads(model_bytes["ranker.json"])
    if _sha256(model_bytes["model.txt"]) != model_state.get("model_sha256"):
        raise ValueError("Native model changed during packaging")
    index_hash = _verified_index_hash(summary, model_bytes["index.joblib"],
                                      Path(cache_directory) if cache_directory is not None else None)
    if full:
        from .full_submission_package import validate_full_submission

        validate_full_submission(summary, model_state, model_directory,
                                 Path(cache_directory) if cache_directory is not None else None)
    minimal = _minimal_summary(summary, model_state)
    _check_feature_code(minimal["dataset"])
    if robust:
        _check_training_weighting(minimal)
    payload = {"script.py": _read_file(runtime / "script.py"),
               "requirements.txt": _read_file(runtime / "requirements.txt")}
    for name, content in model_bytes.items():
        payload[f"model/ranker/{name}"] = content
    payload["model/ranker/training_summary.json"] = _json_bytes(minimal)
    for path in sorted(source.glob("*.py")):
        payload[f"model/portmatch/{path.name}"] = _read_file(path)
    _verify_copied_code(payload, feature_code_hash(), transform_spec()["code_sha256"], minimal)
    if full:
        from .full_submission_package import verify_full_copied_code

        verify_full_copied_code(payload, minimal)
    manifest = {
        "schema_version": 1,
        "competition_id": "236771",
        "baseline": ("robust_gt_missing_v1_86_full" if full else
                     "robust_gt_missing_v1_86" if robust else "token_phonetic_v1_86"),
        "feature_count": 86,
        "trained_feature_names": model.feature_names,
        "outer_split": dataset.get("outer_split"),
        "outer_folds": dataset.get("outer_folds"),
        "outer_fold": dataset.get("outer_fold"),
        "model_sha256": model_state["model_sha256"],
        "index_sha256": index_hash,
        "source_training_summary_sha256": _sha256(summary_bytes),
        "source_cache_metadata_sha256": summary["cache_metadata_sha256"],
        "feature_code_sha256": minimal["dataset"]["provenance"]["feature_code_sha256"],
        "feature_transform": minimal["dataset"]["feature_transform"],
        "files": {name: {"sha256": _sha256(content), "size_bytes": len(content)}
                  for name, content in sorted(payload.items())},
        "note": "Frozen fold model; no full-data retraining or test-time fitting. "
                "All payloads except this manifest are hashed. Raw datasets and validation reports are omitted.",
    }
    if robust:
        manifest["retrieval_extension"] = minimal["dataset"]["retrieval_extension"]
        manifest["training_augmentation"] = minimal["dataset"]["training_augmentation"]
    if full:
        manifest["training_scope"] = "full_public_train"
        manifest["full_training_population"] = minimal["dataset"]["full_training_population"]
        manifest["full_training_trainer"] = minimal["full_training_trainer"]
        manifest["note"] = (
            "Fixed full-public-training model with internal OOF training features. "
            "Final index uses all original public training rows; no held-out accuracy is claimed. "
            "No test-time fitting. All payloads except this manifest are hashed. "
            "Raw datasets, feature caches and training records are omitted.")
    payload["model/package_manifest.json"] = _json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        handle, name = tempfile.mkstemp(prefix=".submission-", suffix=".zip.tmp", dir=output.parent)
        os.close(handle)
        temporary = Path(name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, content in sorted(payload.items()):
                item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                item.create_system = 3
                item.external_attr = 0o100644 << 16
                item.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(item, content, compresslevel=6)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or set(archive.namelist()) != set(payload):
                raise ValueError("Written submission ZIP failed its integrity check")
        zip_hash = _sha256(temporary.read_bytes())
        zip_size = temporary.stat().st_size
        # Windows rename is atomic and refuses an existing destination. POSIX
        # rename replaces files, so use an exclusive same-filesystem hard link.
        if os.name == "nt":
            os.rename(temporary, output)
        else:
            os.link(temporary, output)
        return {"output": str(output), "sha256": zip_hash, "size_bytes": zip_size,
                "file_count": len(payload), "feature_count": 86,
                "model_sha256": model_state["model_sha256"], "index_sha256": index_hash}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trusted local frozen 86-feature model directory")
    parser.add_argument("--output", required=True, help="New submission ZIP path (never overwritten)")
    parser.add_argument("--runtime-directory", help="Runtime source folder; defaults to submission/")
    parser.add_argument("--cache-directory", help="Override cache path if the trusted source cache was moved")
    parser.add_argument("--allow-robust", action="store_true",
                        help="Opt in to the validated robust + GT-missing original-mass 0.5 profile")
    parser.add_argument("--allow-full", action="store_true",
                        help="Opt in to the fixed full-public-training robust/86-feature/0.5 profile")
    args = parser.parse_args()
    print(json.dumps(build_submission(args.model, args.output, runtime_directory=args.runtime_directory,
                                     cache_directory=args.cache_directory,
                                     allow_robust=args.allow_robust, allow_full=args.allow_full), indent=2))


if __name__ == "__main__":
    main()
