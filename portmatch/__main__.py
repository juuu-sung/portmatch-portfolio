"""Local retrieval diagnostics, fold-safe training, and row-independent prediction."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .data import QUERY_COLUMNS, read_catalog, read_queries
from .pipeline import MatchingPipeline
from .retrieval import RetrievalConfig
from .text import has_hangul
from .validation import make_folds, sample_validation, summarize


CHANNELS = {"call": "call_match", "alias": "alias_match", "exact": "exact_hit",
            "name": "name_hit", "edit": "edit_hit", "phonetic": "phonetic_hit",
            "tonnage": "tonnage_hit"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path: str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_output(value: str) -> Path:
    output = Path(value)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be an empty/new directory; choose a new --output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def metadata(args, config) -> dict:
    return {"version": __version__, "train_sha256": file_hash(args.train),
            "catalog_sha256": file_hash(args.catalog), "retrieval": asdict(config)}


def validate_features(features: pd.DataFrame) -> None:
    if np.isinf(features.to_numpy()).any():
        raise ValueError("Infinite feature value")
    if any(column in features for column in ("sample_id", "vessel_id", "target_vessel_id", "catalog_index")):
        raise ValueError("An identifier or target leaked into the feature matrix")


def evaluate(args, config: RetrievalConfig) -> None:
    train, catalog = read_queries(args.train, labeled=True), read_catalog(args.catalog)
    assignments = make_folds(train, args.split, args.folds, args.seed)
    selected = list(range(args.folds)) if args.fold is None else [args.fold]
    if any(fold < 0 or fold >= args.folds for fold in selected):
        raise ValueError("fold must be between zero and folds - 1")
    output = prepare_output(args.output)
    manifest = train[["sample_id", "target_vessel_id"]].copy()
    manifest["has_hangul"] = train.vessel_name.map(has_hangul)
    manifest["fold"] = assignments
    manifest.to_csv(output / "splits.csv", index=False, encoding="utf-8-sig")
    id_to_index = {value: index for index, value in enumerate(catalog.vessel_id)}
    records, fold_summaries, feature_names = [], [], []
    for fold in selected:
        started = time.perf_counter()
        fitting = train[assignments != fold].copy()
        validation = train[assignments == fold].copy()
        validation = sample_validation(validation, args.max_queries, args.seed + fold)
        print(f"Fold {fold}: index from {len(fitting):,} training rows; evaluate {len(validation):,} queries", flush=True)
        pipeline = MatchingPipeline(fitting, catalog, config)
        build_seconds = time.perf_counter() - started
        fold_records = []
        feature_checked = 0
        pair_path = output / f"pairs_fold_{fold}.csv"
        for position, full in enumerate(validation.to_dict("records")):
            # Targets are used only after retrieval, never passed into inference.
            row = {key: full[key] for key in QUERY_COLUMNS}
            target = id_to_index[full["target_vessel_id"]]
            query_started = time.perf_counter()
            candidates = pipeline.retrieve(row)
            true_candidate = candidates[candidates.catalog_index == target]
            record = {"sample_id": row["sample_id"], "fold": fold,
                      "has_hangul": has_hangul(row["vessel_name"]),
                      "training_vessel_seen": full["target_vessel_id"] in pipeline.memory.training_vessels,
                      "candidate_count": len(candidates), "recall_union": bool(len(true_candidate)),
                      "retrieval_seconds": time.perf_counter() - query_started}
            for name, column in CHANNELS.items():
                record["recall_" + name] = bool(len(true_candidate) and true_candidate[column].iloc[0] > 0)
            if args.save_pairs or position < args.feature_checks:
                features = pipeline.features(row, candidates)
                validate_features(features)
                feature_checked += 1
                feature_names = features.columns.tolist()
                if args.save_pairs:
                    pairs = features.copy()
                    pairs.insert(0, "label", (candidates.catalog_index.to_numpy() == target).astype(int))
                    pairs.insert(0, "candidate_vessel_id", catalog.iloc[candidates.catalog_index].vessel_id.to_numpy())
                    pairs.insert(0, "sample_id", row["sample_id"])
                    pairs.to_csv(pair_path, mode="w" if position == 0 else "a",
                                 header=position == 0, index=False, encoding="utf-8")
            fold_records.append(record)
            if (position + 1) % 100 == 0:
                print(f"Fold {fold}: {position + 1:,}/{len(validation):,} queries", flush=True)
        fold_frame = pd.DataFrame(fold_records)
        fold_frame.to_csv(output / f"queries_fold_{fold}.csv", index=False, encoding="utf-8-sig")
        fold_summary = {"fold": fold, "training_rows": len(fitting),
                        "validation_rows_total": int((assignments == fold).sum()),
                        "feature_queries_checked": feature_checked,
                        "index_seconds": build_seconds,
                        "total_seconds": time.perf_counter() - started,
                        "metrics": summarize(fold_frame)}
        write_json(output / f"summary_fold_{fold}.json", fold_summary)
        fold_summaries.append(fold_summary)
        records.extend(fold_records)
        print(f"Fold {fold}: candidate recall = {fold_summary['metrics']['weighted_candidate_recall']}", flush=True)
        del pipeline
    report = {**metadata(args, config), "split": args.split, "seed": args.seed,
              "folds": args.folds, "selected_folds": selected, "max_queries_per_fold": args.max_queries,
              "sampled": args.max_queries is not None,
              "feature_names": feature_names, "fold_reports": fold_summaries,
              "metrics": summarize(pd.DataFrame(records))}
    write_json(output / "summary.json", report)
    print(f"Saved diagnostics: {output.resolve()}", flush=True)


def build(args, config: RetrievalConfig) -> None:
    train, catalog = read_queries(args.train, labeled=True), read_catalog(args.catalog)
    output = prepare_output(args.output)
    started = time.perf_counter()
    pipeline = MatchingPipeline(train, catalog, config)
    joblib.dump(pipeline, output / "index.joblib", compress=3)
    write_json(output / "metadata.json", {**metadata(args, config), "training_rows": len(train),
               "catalog_rows": len(catalog), "seconds": time.perf_counter() - started,
               "artifact_kind": "retrieval index and pair features; no trained ranker"})
    print(f"Saved fixed index: {(output / 'index.joblib').resolve()}", flush=True)


def main() -> None:
    from .ranker_commands import COMMANDS, add_training_parsers, run_training_command

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_training_parsers(subparsers)
    for command in ("evaluate", "build"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--train", default="work/data/train.csv")
        sub.add_argument("--catalog", default="work/data/vessel_catalog.csv")
        sub.add_argument("--output", required=True)
        sub.add_argument("--name-top-k", type=int, default=80)
        sub.add_argument("--phonetic-top-k", type=int, default=80)
        sub.add_argument("--edit-top-k", type=int, default=40)
        sub.add_argument("--gt-window", type=float, default=.10)
        sub.add_argument("--max-candidates", type=int, default=None,
                         help="Optional cap, with exact lookups protected; default keeps the full union")
        if command == "evaluate":
            sub.add_argument("--split", choices=["vessel", "row"], default="vessel")
            sub.add_argument("--folds", type=int, default=5)
            sub.add_argument("--fold", type=int, default=None, help="Only this zero-based fold; default all folds")
            sub.add_argument("--seed", type=int, default=20261022)
            sub.add_argument("--max-queries", type=int, default=None, help="Sample at most this many validation queries per fold")
            sub.add_argument("--feature-checks", type=int, default=10,
                             help="Validate features for this many queries per fold when not saving pairs")
            sub.add_argument("--save-pairs", action="store_true", help="Write all candidate features; can create large CSVs")
    args = parser.parse_args()
    if args.command in COMMANDS:
        run_training_command(args)
        return
    if args.command == "evaluate" and args.feature_checks < 0:
        parser.error("feature-checks must be nonnegative")
    config = RetrievalConfig(name_top_k=args.name_top_k, phonetic_top_k=args.phonetic_top_k,
                             edit_top_k=args.edit_top_k, gt_relative_window=args.gt_window,
                             max_total=args.max_candidates)
    (evaluate if args.command == "evaluate" else build)(args, config)


if __name__ == "__main__":
    main()
