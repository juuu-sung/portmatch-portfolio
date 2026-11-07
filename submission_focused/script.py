"""Offline DACON entrypoint for the frozen focused_search fold0 package."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time

_SCRIPT_STARTED = time.perf_counter()
for _key, _value in (("OMP_NUM_THREADS", "6"), ("OPENBLAS_NUM_THREADS", "1"),
                     ("MKL_NUM_THREADS", "1"), ("NUMEXPR_NUM_THREADS", "1")):
    os.environ[_key] = _value
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def verify_package(root):
    """Validate manifest before importing bundled code or loading joblib assets."""
    root = Path(root).resolve()
    manifest = json.loads((root / "model/package_manifest.json").read_text("utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("profile") != "tonnage_repair_v1_focused_search_fold0"
            or manifest.get("competition_id") != 236771 or manifest.get("feature_count") != 86):
        raise ValueError("Unsupported focused submission manifest")
    expected = manifest["files"]
    required = {"script.py", "requirements.txt", "model/runtime.json", "model/index.joblib",
                "model/hangul_expansion.joblib", "model/ranker/model.txt", "model/ranker/ranker.json"}
    if not required.issubset(expected):
        raise ValueError("Required package file missing from manifest")
    for relative, record in expected.items():
        name = PurePosixPath(relative)
        if (name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative
                or str(name) != relative or not name.parts
                or (name.parts[0] != "model" and relative not in {"script.py", "requirements.txt"})):
            raise ValueError("Invalid manifest path")
        path = root / relative
        if (not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file()
                or path.stat().st_size != record["bytes"]):
            raise ValueError(f"Missing or changed package file: {relative}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Package checksum mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for p in (root / "model").rglob("*") if p.is_file()}
    if actual != {p for p in expected if p.startswith("model/")} | {"model/package_manifest.json"}:
        raise ValueError("Unexpected model package files")
    return manifest


if __name__ == "__main__":
    verify_package(ROOT)
    sys.path.insert(0, str(ROOT / "model"))

import numpy as np
import pandas as pd

from portmatch.data import CATALOG_COLUMNS, QUERY_COLUMNS, read_catalog, read_queries
from portmatch.focused_runtime import FocusedMatcher


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
    matcher = FocusedMatcher(model_path)
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
                  ROOT / "model", ROOT / "output" / "submission.csv")
    metrics["script_seconds_including_imports"] = time.perf_counter() - _SCRIPT_STARTED
    # Aggregate timing only: never log hidden test rows, labels, or predictions.
    print(json.dumps(metrics, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
