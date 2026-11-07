"""Export aggregate evidence from completed runs, or check the public snapshot.

No training, inference, row-level predictions, data or model files are written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "compound-errors": ("compound_v1_20261030", 12, 3),
    "joint-search": ("joint_search_v1_20261031", 16, 5),
    "tonnage-repair": ("tonnage_repair_v1_20261031", 20, 7),
    "focused-replay": ("focused_replay_v1_20261031", 25, 4),
}
TEXT_COLUMNS = {"variant", "evaluation_set", "model", "reference", "experiment"}
NUMBER_COLUMNS = {
    "weighted_top1", "weighted_candidate_recall", "hangul_correct", "hangul_queries",
    "non_hangul_correct", "non_hangul_queries", "mean_candidates", "p95_candidates",
    "name_changed_queries", "two_letter_deletions", "fold", "corrected", "regressed",
    "corrected_vessels", "regressed_vessels", "weighted_delta_pp", "delta_pp",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text("utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


def aggregate_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(reader.fieldnames or ()).issubset(TEXT_COLUMNS | NUMBER_COLUMNS):
            raise ValueError(f"Unexpected aggregate columns: {path.name}")
        return [{key: value if key in TEXT_COLUMNS else float(value) if "." in value or "e" in value.lower()
                 else int(value) for key, value in row.items()} for row in reader]


def check_metrics(data):
    rows = data["metrics"]
    assert len(rows) == data["conditions"] * data["models"]
    assert len({(r["variant"], r["model"]) for r in rows}) == len(rows)
    for row in rows:
        assert row["hangul_queries"] == 1191 and row["non_hangul_queries"] == 11031
        expected = .7 * row["hangul_correct"] / row["hangul_queries"] + .3 * row["non_hangul_correct"] / row["non_hangul_queries"]
        assert math.isclose(expected, row["weighted_top1"], abs_tol=1e-12)
    scores = {(r["variant"], r["model"]): r["weighted_top1"] for r in rows}
    for row in data["paired_changes"]:
        expected = 100 * (scores[row["variant"], row["experiment"]] - scores[row["variant"], row["reference"]])
        assert math.isclose(expected, row.get("delta_pp", row.get("weighted_delta_pp")), abs_tol=1e-10)


def export(output):
    output.mkdir(parents=True, exist_ok=True)
    for label, (directory, conditions, models) in STUDIES.items():
        root = ROOT / "runs" / directory
        audit = load(root / "analysis_audit.json")
        assert audit["complete"] and sorted(audit["folds"]) == list(range(5))
        payload = {
            "study": directory, "conditions": conditions, "models": models,
            "folds": audit["folds"], "prediction_evaluations": audit["predictions"],
            "metric": "Pooled 5-fold: 0.7 * Hangul Top-1 + 0.3 * non-Hangul Top-1; not Public",
            "metrics": aggregate_rows(root / "metrics.csv"),
            "paired_changes": aggregate_rows(root / "paired_changes.csv"),
            "confidence_intervals": load(root / "confidence_intervals.json"),
        }
        sources = [root / name for name in ("metrics.csv", "paired_changes.csv", "confidence_intervals.json", "analysis_audit.json", "protocol.json")]
        for name in ("fold_metrics.csv", "retention_gates.json"):
            path = root / name
            if path.exists():
                payload[path.stem] = aggregate_rows(path) if path.suffix == ".csv" else load(path)
                sources.append(path)
        payload["sources"] = [source(path) for path in sources]
        assert payload["prediction_evaluations"] == 12222 * conditions * models
        check_metrics(payload)
        write(output / f"{label}.json", payload)

    review_path = ROOT / "runs/verify_focused_search_fold0_full/review_20261031.json"
    verification_path = ROOT / "runs/verify_focused_search_fold0_full/verification.json"
    public_path = ROOT / "runs/tonnage_repair_v1_20261031/public_submission_result_20261031.json"
    review, verification, public = map(load, (review_path, verification_path, public_path))
    archive = ROOT / "submissions/focused_search_fold0/submit.zip"
    assert digest(archive) == review["zip_sha256"] == verification["zip_sha256"] == public["zip_sha256"]
    assert digest(verification_path) == review["verification_sha256"]
    runtime_sources = []
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if name.startswith("model/portmatch/") and name.endswith(".py"):
                path = ROOT / name.removeprefix("model/")
            elif name == "script.py":
                path = ROOT / "submission_focused/script.py"
            else:
                continue
            assert bundle.read(name) == path.read_bytes(), name
            runtime_sources.append(source(path))
    selected = {key: verification[key] for key in ("complete", "mode", "profile", "zip_sha256", "zip_bytes",
                "root_entries", "total_subprocess_predictions", "existing_artifacts_unchanged", "python", "packages", "limitations")}
    selected.update({"cases": review["cases"], "zip_original_fold0": review["zip_original_fold0"],
                     "runtime_source_bytes_match_submission": True, "runtime_sources": runtime_sources,
                     "public": {key: public[key] for key in ("score", "previous_best_score", "recorded_date", "source",
                                "leaderboard", "submission_id", "server_screen_or_logs_reviewed")},
                     "sources": [source(p) for p in (review_path, verification_path, public_path)]})
    write(output / "submission-verification.json", selected)
    check(output)


def check(output):
    studies = {key: load(output / f"{key}.json") for key in STUDIES}
    for data in studies.values():
        check_metrics(data)
    old = {(r["variant"], r["model"]): r["weighted_top1"] for r in studies["tonnage-repair"]["metrics"]}
    new = {(r["variant"], r["model"]): r["weighted_top1"] for r in studies["focused-replay"]["metrics"]}
    for key, score in old.items():
        if key[1] == "focused_search":
            assert score == new[key], key
    verification = load(output / "submission-verification.json")
    assert verification["complete"] and verification["total_subprocess_predictions"] == 5293
    assert sum(row["rows"] for row in verification["cases"]) == 5293
    assert all(row["predictions_equal"] and row["input_bytes_equal"] and row["package_bytes_equal"] for row in verification["cases"])
    for record in verification["runtime_sources"]:
        assert digest(ROOT / record["path"]) == record["sha256"], record["path"]
    print(json.dumps({"studies": len(studies), "aggregate_metric_rows": sum(len(x["metrics"]) for x in studies.values()),
                      "submission_predictions_verified_in_saved_record": 5293,
                      "runtime_source_files_verified": len(verification["runtime_sources"]),
                      "training_or_inference_rerun": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check published aggregates and runtime source hashes without local runs")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evidence")
    args = parser.parse_args()
    (check if args.check else export)(args.output)
