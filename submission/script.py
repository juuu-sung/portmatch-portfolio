"""DACON 236771: frozen 86-feature inference, one independent query at a time.

The evaluation server provides data/test.csv and data/vessel_catalog.csv beside
this file. Only output/submission.csv is written. No training, network access,
test-set statistics, group voting, or updates to the fixed training index occur.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time


# Set before importing numpy/LightGBM. Six CPU threads match the evaluation host;
# linear algebra uses one thread to avoid nested thread pools for tiny queries.
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.dont_write_bytecode = True
_SCRIPT_STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parent


def verify_package(root: Path) -> None:
    """Check the packaged code and frozen assets before loading executable code."""
    root = Path(root).resolve()
    manifest = json.loads((root / "model" / "package_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("baseline") not in {"token_phonetic_v1_86", "robust_gt_missing_v1_86",
                                                "robust_gt_missing_v1_86_full"}
            or manifest.get("feature_count") != 86):
        raise ValueError("Unsupported submission package manifest")
    files = manifest.get("files")
    required = {"script.py", "requirements.txt", "model/portmatch/__init__.py",
                "model/portmatch/ranker_commands.py", "model/ranker/model.txt",
                "model/ranker/ranker.json", "model/ranker/index.joblib",
                "model/ranker/training_summary.json"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("Submission manifest is missing required payload files")
    for relative, metadata in files.items():
        if not isinstance(relative, str) or "\\" in relative:
            raise ValueError("Invalid path in submission manifest")
        path = root / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts
                or not path.resolve().is_relative_to(root) or path.is_symlink()):
            raise ValueError("Submission manifest path escapes its package")
        if (not isinstance(metadata, dict) or not path.is_file()
                or path.stat().st_size != metadata.get("size_bytes")):
            raise ValueError("Submission payload file is missing or has changed size")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != metadata.get("sha256"):
            raise ValueError("Submission payload SHA256 verification failed")
    if any(path.relative_to(root).as_posix() not in files
           for path in (root / "model" / "portmatch").glob("*.py")):
        raise ValueError("Submission contains unmanifested inference modules")
    if manifest["baseline"] in {"robust_gt_missing_v1_86", "robust_gt_missing_v1_86_full"}:
        summary = json.loads((root / "model" / "ranker" / "training_summary.json").read_text("utf-8"))
        dataset = summary.get("dataset", {})
        if (manifest.get("retrieval_extension") != dataset.get("retrieval_extension")
                or manifest.get("training_augmentation") != dataset.get("training_augmentation")
                or dataset.get("retrieval", {}).get("profile") != "robust"
                or dataset.get("training_augmentation", {}).get("config", {}).get("original_mass") != 0.5
                or dataset.get("training_reweighting") is not None):
            raise ValueError("Robust submission manifest differs from its validated model profile")
        full = manifest["baseline"] == "robust_gt_missing_v1_86_full"
        if full:
            if (manifest.get("training_scope") != "full_public_train"
                    or summary.get("training_scope") != "full_public_train"
                    or dataset.get("training_scope") != "full_public_train"
                    or manifest.get("full_training_population") != dataset.get("full_training_population")
                    or manifest.get("full_training_trainer") != summary.get("full_training_trainer")
                    or summary.get("validation") != {"available": False, "held_out_queries": 0,
                        "reason": "no_external_holdout_full_training_scope"}):
                raise ValueError("Full submission manifest differs from its training scope proof")
        elif summary.get("training_scope") is not None or dataset.get("training_scope") is not None:
            raise ValueError("Full-public-training models require the full submission profile")


if (ROOT / "model" / "portmatch").is_dir():
    verify_package(ROOT)
    sys.path.insert(0, str(ROOT / "model"))

import numpy as np
import pandas as pd

from portmatch.data import CATALOG_COLUMNS, QUERY_COLUMNS, read_catalog, read_queries
from portmatch.ranker_commands import RankedMatcher


OUTPUT_COLUMNS = ["sample_id", "target_vessel_id"]
NUMERIC_CATALOG_COLUMNS = {"size", "deadweight_tonnage", "gross_tonnage"}


def validate_catalog(supplied: pd.DataFrame, frozen: pd.DataFrame) -> None:
    """Confirm the official catalog matches the model, independent of CSV order.

    Candidate indices refer to the frozen catalog's order. We therefore validate
    the server's catalog by vessel ID, then retain that fixed index unchanged.
    Numeric CSV representations such as 100 and 100.0 are equivalent; names and
    categorical text must match exactly (including missing values).
    """
    for catalog in (supplied, frozen):
        if not set(CATALOG_COLUMNS).issubset(catalog.columns):
            raise ValueError("Catalog is missing required columns")
        if catalog.vessel_id.isna().any() or not catalog.vessel_id.is_unique:
            raise ValueError("Catalog vessel_id must be non-null and unique")
    left = supplied.loc[:, CATALOG_COLUMNS].set_index("vessel_id").sort_index()
    right = frozen.loc[:, CATALOG_COLUMNS].set_index("vessel_id").sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Official catalog vessel IDs differ from the frozen model catalog")
    for column in left.columns:
        if column in NUMERIC_CATALOG_COLUMNS:
            a = pd.to_numeric(left[column], errors="raise").to_numpy(dtype=np.float64)
            b = pd.to_numeric(right[column], errors="raise").to_numpy(dtype=np.float64)
            equal = np.allclose(a, b, rtol=1e-12, atol=1e-9, equal_nan=True)
        else:
            a, b = left[column].astype("string"), right[column].astype("string")
            equal = a.equals(b)
        if not equal:
            # Do not echo private input values or records into evaluation logs.
            raise ValueError(f"Official catalog differs from the frozen model in column {column}")


def validate_model(matcher: RankedMatcher) -> None:
    names = matcher.ranker.feature_names
    if (len(names) != 86 or "retrieval_tonnage_rank" in names
            or not isinstance(matcher.feature_transform, dict)
            or matcher.feature_transform.get("name") != "token_phonetic_v1"):
        raise ValueError("This submission requires the frozen 86-feature token-phonetic model")
    if matcher.ranker.config.n_jobs != 6:
        raise ValueError("This submission requires the saved six-thread inference configuration")


def run(test_path: str | Path, catalog_path: str | Path,
        model_path: str | Path, output_path: str | Path) -> dict:
    """Read inputs, predict independently, validate and atomically save the CSV.

    Returns timings only, never input values, predictions, or test-derived
    features. The model and its dictionaries are fixed before the query loop.
    """
    started = time.perf_counter()
    queries = read_queries(test_path).loc[:, QUERY_COLUMNS]
    catalog = read_catalog(catalog_path)
    loaded_inputs = time.perf_counter()
    matcher = RankedMatcher(model_path)
    validate_model(matcher)
    validate_catalog(catalog, matcher.pipeline.catalog)
    loaded_model = time.perf_counter()
    allowed_ids = set(catalog.vessel_id.astype(str))
    records = []
    for number, row in enumerate(queries.to_dict("records"), 1):
        # Neither the remaining rows nor their labels/distribution are passed in.
        predicted = matcher.predict(row)
        if not isinstance(predicted, str) or predicted not in allowed_ids:
            raise ValueError("Model produced a vessel ID absent from the official catalog")
        records.append({"sample_id": row["sample_id"], "target_vessel_id": predicted})
        if number % 100 == 0:
            print(json.dumps({"predicted_rows": number, "total_rows": len(queries)}), flush=True)
    inferred = time.perf_counter()
    result = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    if (len(result) != len(queries) or result.sample_id.tolist() != queries.sample_id.tolist()
            or result.isna().any().any() or not result.sample_id.is_unique):
        raise ValueError("Submission must have exactly one valid output per input row, in input order")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        result.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    finished = time.perf_counter()
    inference_seconds = inferred - loaded_model
    return {
        "rows": len(queries),
        "input_seconds": loaded_inputs - started,
        "model_load_and_catalog_check_seconds": loaded_model - loaded_inputs,
        "inference_seconds": inference_seconds,
        "output_seconds": finished - inferred,
        "run_seconds": finished - started,
        "rows_per_second": len(queries) / inference_seconds if len(queries) else None,
    }


def main() -> None:
    metrics = run(ROOT / "data" / "test.csv", ROOT / "data" / "vessel_catalog.csv",
                  ROOT / "model" / "ranker", ROOT / "output" / "submission.csv")
    metrics["script_seconds_including_imports"] = time.perf_counter() - _SCRIPT_STARTED
    # Aggregate timing only: never log hidden test rows, labels, or predictions.
    print(json.dumps(metrics, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
