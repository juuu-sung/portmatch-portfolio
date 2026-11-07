"""Explicit commands for the frozen full-public-training submission profile."""
from __future__ import annotations

import json
from pathlib import Path


COMMANDS = {"prepare-full-ranker", "train-full-ranker"}


def add_full_training_parsers(subparsers):
    prepare = subparsers.add_parser(
        "prepare-full-ranker",
        help="Build full-public-data inner-OOF pairs; robust/GT-missing mass 0.5 fixed, no held-out score")
    prepare.add_argument("--train", default="work/data/train.csv")
    prepare.add_argument("--catalog", default="work/data/vessel_catalog.csv")
    prepare.add_argument("--output", required=True, help="A new or empty full-training cache directory")
    prepare.add_argument("--shard-queries", type=int, default=64)
    prepare.add_argument("--max-train-queries", type=int, default=None,
                         help="Diagnostic source limit only; limited caches cannot become submission packages")
    train = subparsers.add_parser(
        "train-full-ranker", help="Fit the fixed 86-feature full-data model; no held-out evaluation")
    train.add_argument("--data", required=True, help="Full-training token-phonetic cache")
    train.add_argument("--output", required=True, help="A new or empty full-model directory")
    train.add_argument("--train", default=None, help="Original training CSV; defaults to its recorded provenance path")
    train.add_argument("--catalog", default=None, help="Original catalog CSV; defaults to its recorded provenance path")
    train.add_argument("--rounds", type=int, choices=[300], default=300,
                       help="Frozen full-training profile: 300 rounds")
    train.add_argument("--threads", type=int, choices=[6], default=6,
                       help="Frozen full-training profile: 6 CPU threads")


def run_full_training_command(args):
    if args.command == "prepare-full-ranker":
        from .data import read_catalog, read_queries
        from .full_training_data import prepare_full_training_data
        from .ranker_commands import _hash_file, feature_code_hash

        if args.shard_queries < 1:
            raise ValueError("shard-queries must be positive")
        if args.max_train_queries is not None and args.max_train_queries < 1:
            raise ValueError("max-train-queries must be positive or omitted")
        provenance = {
            "train_sha256": _hash_file(args.train), "catalog_sha256": _hash_file(args.catalog),
            "feature_code_sha256": feature_code_hash(),
            "train_path": str(Path(args.train).resolve()),
            "catalog_path": str(Path(args.catalog).resolve()),
        }
        result = prepare_full_training_data(
            read_queries(args.train, labeled=True), read_catalog(args.catalog), args.output,
            max_train_queries=args.max_train_queries, shard_queries=args.shard_queries,
            provenance=provenance)
        print(f"Completed full-training OOF cache: {Path(args.output).resolve()}", flush=True)
        print(json.dumps(result.get("counts", {}), ensure_ascii=False, indent=2), flush=True)
        print("No held-out validation was created. Transform token-phonetic features before training.", flush=True)
    elif args.command == "train-full-ranker":
        from .full_ranking import train_full_ranker
        from .ranking import RankerConfig

        result = train_full_ranker(args.data, args.output,
                                   config=RankerConfig(n_estimators=args.rounds, n_jobs=args.threads),
                                   train_path=args.train, catalog_path=args.catalog)
        print(f"Saved full-training model: {Path(args.output).resolve()}", flush=True)
        print("No held-out validation accuracy is available for this full-training scope.", flush=True)
        if result.get("dataset", {}).get("sampled"):
            print("Diagnostic source limit was used; this model cannot be packaged for submission.", flush=True)
        return result
    else:
        raise ValueError(f"Unknown full-training command: {args.command}")
