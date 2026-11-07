"""Check a full-training ZIP's parity, row independence and local runtime.

Public training inputs are used only as interface probes. They overlap model
training and final index fitting, so this command never computes accuracy.
Each ZIP case runs in a fresh isolated interpreter, through the actual script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

# Direct script execution places scripts/, rather than the project root, on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from portmatch.data import QUERY_COLUMNS, read_queries
from portmatch.ranker_commands import RankedMatcher
from portmatch.text import has_hangul
from scripts.verify_submission import file_hash, run_case


def _select_probe_rows(queries: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    if maximum is None or maximum >= len(queries):
        return queries.copy()
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("max_rows must be a positive integer")
    return queries.sample(n=maximum, random_state=20261022).sort_index().reset_index(drop=True)


def _small_subset(queries: pd.DataFrame) -> pd.DataFrame:
    languages = queries.vessel_name.map(has_hangul)
    return pd.concat((queries[languages].head(32), queries[~languages].head(32)), ignore_index=True)


def _mask_both(queries: pd.DataFrame) -> pd.DataFrame:
    result = queries.copy()
    result["gross_tonnage"] = np.nan
    result["international_gross_tonnage"] = np.nan
    return result


def _scale_international(queries: pd.DataFrame) -> pd.DataFrame:
    result = queries.copy()
    result["international_gross_tonnage"] = pd.to_numeric(
        result.international_gross_tonnage, errors="raise") * 1.25
    return result


def verify_full_submission(archive: str | Path, output: str | Path, *,
                           reference_model: str | Path, train: str | Path,
                           catalog: str | Path, python: str | Path = sys.executable,
                           timeout: int = 1800, max_rows: int | None = None) -> dict:
    """Inspect a trusted full model and run six ZIP interface/timing probes.

    Full/reversed/subset/single are compared to the same saved model and each
    other by sample ID. Two GT variants are compared to direct predictions for
    the same transformed inputs; no changed-input invariance is assumed.
    """
    from portmatch.full_ranking import check_full_model_summary

    root, archive = Path(output).resolve(), Path(archive).resolve()
    reference_model, train, catalog = (Path(p).resolve() for p in (reference_model, train, catalog))
    python = Path(python).resolve()
    if root.exists():
        raise FileExistsError("Use a new verification output directory")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout < 1:
        raise ValueError("timeout must be a positive integer")
    if max_rows is not None and (isinstance(max_rows, bool) or not isinstance(max_rows, int)
                                 or max_rows < 1):
        raise ValueError("max_rows must be a positive integer")
    summary = json.loads((reference_model / "training_summary.json").read_text("utf-8"))
    state = json.loads((reference_model / "ranker.json").read_text("utf-8"))
    check_full_model_summary(summary, model_state=state)
    source = summary["dataset"]["provenance"]
    if file_hash(train) != source["train_sha256"] or file_hash(catalog) != source["catalog_sha256"]:
        raise ValueError("Probe train/catalog files differ from the saved full-training provenance")
    archive_hash = file_hash(archive)
    model_files = {name: file_hash(reference_model / name) for name in (
        "model.txt", "ranker.json", "index.joblib", "training_summary.json")}
    with zipfile.ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("model/package_manifest.json"))
    if (manifest.get("baseline") != "robust_gt_missing_v1_86_full"
            or manifest.get("training_scope") != "full_public_train"
            or manifest.get("model_sha256") != model_files["model.txt"]
            or manifest.get("index_sha256") != model_files["index.joblib"]
            or manifest.get("source_training_summary_sha256") != model_files["training_summary.json"]):
        raise ValueError("ZIP is not the selected full-public-training model")
    queries = read_queries(train).loc[:, QUERY_COLUMNS]
    if queries.empty:
        raise ValueError("Public probe inputs must not be empty")
    selected = _select_probe_rows(queries, max_rows)
    subset = _small_subset(selected)
    languages = selected.vessel_name.map(has_hangul)
    catalog_ids = set(pd.read_csv(catalog, usecols=["vessel_id"], dtype=str).vessel_id)
    root.mkdir(parents=True)
    result = {
        "schema_version": 1, "complete": False,
        "purpose": "packaging_parity_and_timing_only_not_validation",
        "training_scope": "full_public_train", "probe_rows_overlap_training": True,
        "accuracy_evaluated": False, "held_out_queries": 0,
        "archive": str(archive), "archive_sha256": archive_hash,
        "archive_bytes": archive.stat().st_size, "reference_model": str(reference_model),
        "reference_model_files": model_files, "public_training_rows": len(queries),
        "rows": len(selected), "max_rows": max_rows,
        "hangul_rows": int(languages.sum()), "non_hangul_rows": int((~languages).sum()),
        "platform": platform.platform(), "python": platform.python_version(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "logical_cpus_visible": os.cpu_count(), "lightgbm_threads": 6,
        "cases": [], "limits": [
            "These public rows were used for training/index fitting; no accuracy is evaluated.",
            "Local OS timings are not DACON server timings or a guarantee for hidden input size.",
            "Every ZIP case starts a new process; filesystem caches were not cleared.",
            "Masked/scaled inputs are interface probes, not a new generalization evaluation."],
    }

    def save():
        (root / "verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    started = time.perf_counter()
    matcher = RankedMatcher(reference_model)
    result["reference_model_load_seconds"] = time.perf_counter() - started

    def expected_predictions(rows):
        started = time.perf_counter()
        predictions = {}
        for row in rows.loc[:, QUERY_COLUMNS].to_dict("records"):
            predicted = matcher.predict(row)
            if not isinstance(predicted, str) or predicted not in catalog_ids:
                raise ValueError("Saved model predicted an invalid catalog ID")
            predictions[row["sample_id"]] = predicted
            if len(predictions) % 100 == 0:
                print(f"Reference probe: {len(predictions):,}/{len(rows):,} rows", flush=True)
        return predictions, time.perf_counter() - started

    expected, seconds = expected_predictions(selected)
    result["reference_original_inference_seconds"] = seconds
    save()
    full = None
    cases = [("full", selected, False), ("reversed", selected.iloc[::-1], False),
             ("subset", subset, False), ("single", subset.iloc[:1], False),
             ("both_gt_missing_subset", _mask_both(subset), True),
             ("international_gt_x1_25_subset", _scale_international(subset), True)]
    for label, rows, changed in cases:
        direct, direct_seconds = (expected_predictions(rows) if changed
                                  else ({sid: expected[sid] for sid in rows.sample_id}, None))
        measured, predictions = run_case(archive, root, label, rows, catalog, python, timeout)
        if not set(predictions.values()).issubset(catalog_ids):
            raise AssertionError("Predicted vessel is outside the official catalog")
        if predictions != direct:
            raise AssertionError("ZIP predictions differ from the saved full model on the same inputs")
        if full is None:
            full = predictions
        if not changed and predictions != {sid: full[sid] for sid in rows.sample_id}:
            raise AssertionError("Predictions depend on input order, subset or batch size")
        measured.update(matches_reference=True, catalog_ids_valid=True,
                        input_variant=label if changed else "original",
                        matches_full_by_id=None if changed else True,
                        reference_inference_seconds=direct_seconds)
        result["cases"].append(measured)
        save()
    if file_hash(archive) != archive_hash or any(
            file_hash(reference_model / name) != digest for name, digest in model_files.items()):
        raise AssertionError("ZIP or saved model changed during verification")
    if file_hash(train) != source["train_sha256"] or file_hash(catalog) != source["catalog_sha256"]:
        raise AssertionError("Public probe source files changed during verification")
    result.update(complete=True, row_independence_verified=True,
                  all_predictions_match_saved_model=True, upload_performed=False)
    save()
    print(f"Verified full model ZIP without accuracy evaluation: {root / 'verification.json'}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--train", default="work/data/train.csv", type=Path)
    parser.add_argument("--catalog", default="work/data/vessel_catalog.csv", type=Path)
    parser.add_argument("--python", default=sys.executable, type=Path)
    parser.add_argument("--timeout", default=1800, type=int)
    parser.add_argument("--max-rows", type=int, help="Optional diagnostic probe limit; default checks every public row")
    args = parser.parse_args()
    verify_full_submission(args.archive, args.output, reference_model=args.reference_model,
                           train=args.train, catalog=args.catalog, python=args.python,
                           timeout=args.timeout, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
