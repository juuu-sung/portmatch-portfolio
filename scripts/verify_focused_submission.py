"""Verify the real focused_search ZIP against frozen fold0 predictions.

Smoke runs cover every corruption recipe on a small public sample. Full runs
add all held-out original rows and their reversal; they are intentionally opt-in.
Every subprocess restores exclusively from the extracted ZIP, with Python -I.
"""
from __future__ import annotations

import argparse
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

for key, value in (("OMP_NUM_THREADS", "6"), ("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")):
    os.environ[key] = value
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from portmatch.compound_experiment import read_json, sha
from portmatch.data import QUERY_COLUMNS, read_queries
from portmatch.focused_runtime import FocusedMatcher, PROFILE
from portmatch.focused_submission_package import check_snapshot, protected_snapshot
from portmatch.joint_search_experiment import query_arrays
from portmatch.text import has_hangul
from portmatch.tonnage_repair_experiment import check, TonnageRepairMatcher
from portmatch.tonnage_repair_protocol import EVAL_RECIPES, evaluation
from scripts.verify_submission import run_case


def runtime_module(path):
    spec = importlib.util.spec_from_file_location("focused_zip_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True, dest="archive")
    parser.add_argument("--study", type=Path, default=Path("runs/tonnage_repair_v1_20261031"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    archive, study, output = args.archive.resolve(), args.study.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new verification output directory")
    if args.timeout < 1:
        raise ValueError("Timeout must be positive")
    before = protected_snapshot()
    frozen = check(study)
    state = read_json(study / "fold0/status.json")
    if not state["complete"] or sha(study / "fold0/predictions.csv") != state["files"]["predictions.csv"]:
        raise ValueError("Fold0 reference predictions are incomplete or modified")
    for item in frozen["data"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("Official public data changed")
    train = read_queries(frozen["data"]["train.csv"]["path"], labeled=True).set_index("sample_id", drop=False)
    catalog = Path(frozen["data"]["vessel_catalog.csv"]["path"])
    saved = pd.read_csv(study / "fold0/predictions.csv", dtype={"sample_id": str, "predicted_vessel_id": str})
    saved = saved[saved.model.eq("focused_search")].set_index(["sample_id", "variant"])
    if not saved.index.is_unique:
        raise ValueError("Duplicate reference predictions")
    original = saved.xs("original", level="variant")
    held = train.loc[original.index].sort_index().copy()
    hangul = held.vessel_name.map(has_hangul)
    sample_count = 1 if args.mode == "smoke" else 8
    seeds = pd.concat([held[hangul].head(sample_count), held[~hangul].head(sample_count)])
    probe_rows, probe_expected, probe_sources = [], {}, []
    for source in seeds.to_dict("records"):
        for variant in EVAL_RECIPES:
            row = evaluation(source, variant)
            source_id = source["sample_id"]
            row["sample_id"] = f"PROBE_{len(probe_rows):05d}"
            probe_rows.append(row)
            probe_expected[row["sample_id"]] = str(saved.loc[(source_id, variant), "predicted_vessel_id"])
            probe_sources.append({"probe_id": row["sample_id"], "source_id": source_id, "variant": variant,
                                  "has_hangul": has_hangul(source["vessel_name"])})
    probes = pd.DataFrame(probe_rows, columns=QUERY_COLUMNS)
    output.mkdir(parents=True)
    results = []

    def execute(label, rows, expected, catalog_path=catalog):
        result, predictions = run_case(archive, output, label, rows, catalog_path, args.python.resolve(), args.timeout)
        wanted = {str(row): expected[str(row)] for row in rows.sample_id}
        if predictions != wanted:
            raise AssertionError(f"{label}: ZIP predictions differ from frozen focused_search results")
        # Check all source/assets again after execution, not just data files.
        runtime_module(output / label / "script.py").verify_package(output / label)
        allowed = set(pd.read_csv(catalog_path, usecols=["vessel_id"], dtype=str).vessel_id)
        if not set(predictions.values()).issubset(allowed):
            raise AssertionError("Unknown output vessel ID")
        result.update(reference_predictions_equal=True, package_files_unchanged=True)
        results.append(result)

    execute("all_recipe_probes", probes, probe_expected)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("model/package_manifest.json"))
        metadata = json.loads(z.read("model/runtime.json"))
        sizes = [i.file_size for i in z.infolist()]
        roots = sorted({i.filename.split("/")[0] for i in z.infolist()})
    if (manifest["profile"] != PROFILE or roots != ["model", "requirements.txt", "script.py"]
            or archive.stat().st_size >= 10_000_000_000 or sum(sizes) >= 32_000_000_000
            or metadata["provenance"]["study_protocol_sha256"] != sha(study / "protocol.json")
            or metadata["provenance"]["reference_predictions_sha256"] != sha(study / "fold0/predictions.csv")
            or metadata["assets"]["ranker/model.txt"] != sha(study / "fold0/focused/model.txt")):
        raise AssertionError("Package format, size or experiment provenance mismatch")

    # Independent candidate-by-candidate parity against the original experiment.
    print("Checking native restored candidate scores against experiment implementation", flush=True)
    restored = FocusedMatcher(output / "all_recipe_probes/model")
    reference = TonnageRepairMatcher(study, 0, "focused_search")
    native_checks = []
    core = {"original", "fresh_double_delete_gt_x0_55", "name_swap_gt_x0_70", "name_double_delete_gt_x1_50"}
    # Resolve the existing recipe spellings by their suffix for the two regression cases.
    core |= {v for v in EVAL_RECIPES if v.endswith(("x0_70", "x1_50"))}
    for source, row in zip(probe_sources, probe_rows):
        if source["variant"] not in core or not source["has_hangul"]:
            continue
        if any(x["variant"] == source["variant"] for x in native_checks):
            continue
        _, expected_ids, _, matrix, added, _, _ = query_arrays(reference.baseline, reference.expansion, row)
        expected_scores = reference.ranker.predict(matrix, reference.ranker.feature_names)
        ids, scores = restored.candidate_scores(row)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_array_equal(scores, expected_scores)
        changed = {**row, "sample_id": "CHANGED_ID", "arrival_year": 1900, "target_vessel_id": "IGNORED"}
        changed_ids, changed_scores = restored.candidate_scores(changed)
        np.testing.assert_array_equal(changed_ids, ids)
        np.testing.assert_array_equal(changed_scores, scores)
        native_checks.append({**source, "candidate_count": len(ids), "added_candidates": int(added.sum()),
                              "candidate_ids_and_scores_exact": True, "id_year_label_independent": True})
    if len(native_checks) < 4 or not any(x["added_candidates"] for x in native_checks):
        raise AssertionError("Native parity probes must exercise original, .55, .70, 1.50 and actual candidate additions")
    del restored, reference

    if args.mode == "full":
        expected_original = original.predicted_vessel_id.astype(str).to_dict()
        execute("original_full", held, expected_original)
        execute("original_reversed", held.iloc[::-1].copy(), expected_original)
        subset = held.iloc[::max(1, len(held) // 64)].head(64).copy()
        execute("original_subset", subset, expected_original)
    else:
        execute("probes_reversed", probes.iloc[::-1].copy(), probe_expected)
        execute("probes_subset", probes.iloc[[0, 1, len(EVAL_RECIPES), len(probes) - 1]].copy(), probe_expected)
    execute("single", probes.iloc[[0]].copy(), probe_expected)
    reordered_catalog = output / "catalog_reordered.csv"
    pd.read_csv(catalog, dtype={"vessel_id": str, "vessel_name": str}).iloc[::-1].to_csv(reordered_catalog, index=False)
    changed = probes.iloc[[0, 1, len(EVAL_RECIPES), len(probes) - 1]].copy()
    changed_expected = {}
    for position, row in changed.iterrows():
        old_id = row.sample_id
        new_id = "RENAMED_" + old_id
        changed.loc[position, "sample_id"] = new_id
        changed.loc[position, "arrival_year"] = 1900
        changed_expected[new_id] = probe_expected[old_id]
    execute("id_year_catalog_order", changed, changed_expected, reordered_catalog)
    check_snapshot(before)
    report = {
        "complete": True, "mode": args.mode, "profile": PROFILE, "zip_sha256": sha(archive),
        "zip_bytes": archive.stat().st_size, "expanded_bytes": sum(sizes), "root_entries": roots,
        "cases": results, "total_subprocess_predictions": sum(x["rows"] for x in results),
        "native_candidate_parity": native_checks, "recipe_probe_sources": probe_sources,
        "preserved_file_count": len(before), "existing_artifacts_unchanged": True,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "rapidfuzz", "Unidecode", "joblib", "lightgbm")},
        "limitations": "Local Windows/Python 3.11 verification; official Ubuntu/Python 3.11.17 timing and hidden test size remain unmeasured.",
    }
    with (output / "verification.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("complete", "mode", "total_subprocess_predictions", "zip_sha256",
                                           "existing_artifacts_unchanged")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
