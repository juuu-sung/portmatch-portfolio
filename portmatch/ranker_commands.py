"""CLI workflows for fold-safe training caches and held-out ranker evaluation."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from .data import QUERY_COLUMNS, read_catalog, read_queries
from .feature_selection import FeatureSelection
from .retrieval import RetrievalConfig


COMMANDS = {"prepare-ranker", "transform-ranker-cache", "reweight-ranker-cache", "train-ranker", "predict-ranker", "compare-rankers",
            "compare-retrieval", "prepare-full-ranker", "train-full-ranker", "compare-index-scale", "compare-index-factorial",
            "compare-category-scope", "compare-support-ablation"}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_code_hash():
    digest = hashlib.sha256()
    for filename in ("features.py", "text.py", "memory.py", "retrieval.py", "pipeline.py"):
        digest.update(filename.encode())
        digest.update(Path(__file__).with_name(filename).read_bytes())
    return digest.hexdigest()


def _new_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Choose a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _check_feature_code(metadata):
    expected = metadata.get("provenance", {}).get("feature_code_sha256")
    if expected and expected != feature_code_hash():
        raise ValueError("Feature code changed since cache creation. Regenerate the training cache.")
    from .feature_transforms import check_transform_metadata

    check_transform_metadata(metadata)
    from .gt_reweight_cache import check_reweighting_metadata

    check_reweighting_metadata(metadata)
    if _has_full_training_metadata(metadata):
        from .full_training_data import check_full_training_metadata

        check_full_training_metadata(metadata)


def _has_full_training_metadata(metadata):
    """Recognize partial full markers too, so a malformed bundle fails closed."""
    return (any(key in metadata for key in
                ("training_scope", "full_training", "full_training_assets",
                 "full_training_population", "full_training_pair_proof"))
            or metadata.get("outer_split") == "none")


def _load_retrieval_index(path, metadata):
    """Bind versioned retrieval to its unchanged, trusted local fitted index."""
    from .robust_retrieval import check_retrieval_metadata

    check_retrieval_metadata(metadata)
    if metadata.get("retrieval_extension") is not None:
        expected = metadata.get("retrieval_index_sha256")
        if not isinstance(expected, str) or len(expected) != 64 or expected != _hash_file(path):
            raise ValueError("Saved retrieval index differs from cache metadata")
    pipeline = joblib.load(path)
    check_retrieval_metadata(metadata, pipeline=pipeline)
    return pipeline


def score_order(scores, catalog_indices, catalog_ids):
    """Break equal model scores by fixed vessel ID, independently of other queries."""
    scores = np.asarray(scores)
    indices = np.asarray(catalog_indices, dtype=np.int64)
    if len(scores) != len(indices) or not np.isfinite(scores).all():
        raise ValueError("Invalid prediction scores")
    return np.lexsort((indices, catalog_ids[indices], -scores))


def evaluate_cache(ranker, directory):
    from .training_data import iter_shards
    from .ranking import ranking_metrics

    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text("utf-8"))
    if _has_full_training_metadata(metadata):
        raise ValueError("Full-training caches have no held-out queries or validation accuracy")
    selection = FeatureSelection.from_names(metadata["feature_names"],
                                            getattr(ranker, "feature_names", None))
    catalog_ids = np.load(directory / metadata.get("catalog_ids_path", "catalog_ids.npy"), allow_pickle=False)
    records = []
    for shard in iter_shards(directory, "valid"):
        scores = ranker.predict(selection.apply(shard["X"], metadata["feature_names"]),
                                list(selection.selected_features))
        offset = 0
        for query_index, count in enumerate(shard["group"]):
            count = int(count)
            section = slice(offset, offset + count)
            indices = shard["catalog_index"][section]
            query_scores = scores[section]
            order = score_order(query_scores, indices, catalog_ids)
            target = int(shard["target_catalog_index"][query_index])
            target_locations = np.flatnonzero(indices[order] == target)
            record = {
                "sample_id": str(shard["sample_id"][query_index]),
                "has_hangul": bool(shard["has_hangul"][query_index]),
                "target_vessel_id": str(catalog_ids[target]),
                "predicted_vessel_id": str(catalog_ids[indices[order[0]]]),
                "candidate_found": bool(len(target_locations)),
                "target_rank": int(target_locations[0] + 1) if len(target_locations) else 0,
                "candidate_count": count,
                "top_score": float(query_scores[order[0]]),
                "score_margin": float(query_scores[order[0]] - query_scores[order[1]]) if count > 1 else None,
            }
            records.append(record)
            offset += count
        print(f"Evaluated {len(records):,} held-out queries against all candidates", flush=True)
    if not records:
        raise ValueError("Validation cache contains no queries")
    frame = pd.DataFrame(records)
    return frame, ranking_metrics(frame)


class RankedMatcher:
    """Load a trusted local model bundle; predict each query with fixed assets."""

    def __init__(self, model_directory):
        from .ranking import VesselRanker

        directory = Path(model_directory)
        manifest = json.loads((directory / "training_summary.json").read_text("utf-8"))
        if not manifest.get("complete"):
            raise ValueError("Model bundle is incomplete")
        _check_feature_code(manifest["dataset"])
        _check_training_weighting(manifest)
        self.feature_transform = manifest["dataset"].get("feature_transform")
        self.ranker = VesselRanker.load(directory)
        if (_has_full_training_metadata(manifest["dataset"])
                or "training_scope" in manifest or "full_training_trainer" in manifest
                or "full_training_proof" in manifest):
            from .full_ranking import check_full_model_summary

            check_full_model_summary(
                manifest, model_state=json.loads((directory / "ranker.json").read_text("utf-8")))
        source_names = manifest["dataset"]["feature_names"]
        if "feature_selection" in manifest:
            self.selection = FeatureSelection.from_dict(manifest["feature_selection"])
        else:
            # Legacy bundles used every cached feature. Preserve exact-schema checks.
            self.selection = FeatureSelection.from_names(source_names)
        if (self.selection.source_feature_names != tuple(source_names)
                or self.selection.selected_features != tuple(self.ranker.feature_names)
                or manifest.get("trained_feature_names", self.ranker.feature_names) != self.ranker.feature_names):
            raise ValueError("Model and bundle feature selection schemas disagree")
        _check_augmented_features(manifest["dataset"], self.selection)
        _check_support_ranker_config(manifest["dataset"], self.selection, self.ranker.config)
        self.pipeline = _load_retrieval_index(directory / "index.joblib", manifest["dataset"])
        self.catalog_ids = self.pipeline.catalog.vessel_id.astype(str).to_numpy()

    def rank(self, row):
        # Extra columns, including any labels, are never visible to inference.
        query = {key: row.get(key) for key in QUERY_COLUMNS}
        candidates = self.pipeline.retrieve(query)
        features = self.pipeline.features(query, candidates)
        if self.feature_transform is not None:
            from .feature_transforms import transform_features

            features = transform_features(self.feature_transform, self.pipeline, query, candidates, features)
        selected = self.selection.apply(features.to_numpy(dtype=np.float32), features.columns.tolist())
        scores = self.ranker.predict(selected, list(self.selection.selected_features))
        indices = candidates.catalog_index.to_numpy(dtype=np.int64)
        order = score_order(scores, indices, self.catalog_ids)
        return pd.DataFrame({"vessel_id": self.catalog_ids[indices[order]], "score": scores[order]})

    def predict(self, row):
        return str(self.rank(row).vessel_id.iloc[0])


def run_prepare(args):
    from .gt_augmentation import GTMissingAugmentationConfig
    from .sampling import SamplingConfig
    from .training_data import prepare_training_data

    augmentation = getattr(args, "gt_augmentation", "none")
    original_mass = getattr(args, "original_mass", None)
    if augmentation == "none" and original_mass is not None:
        raise ValueError("--original-mass requires --gt-augmentation missing")
    if augmentation not in {"none", "missing"}:
        raise ValueError("Unknown GT augmentation mode")
    augmentation_config = (GTMissingAugmentationConfig(
        original_mass=.5 if original_mass is None else original_mass)
        if augmentation == "missing" else None)
    settings = dict(name_top_k=args.name_top_k, phonetic_top_k=args.phonetic_top_k,
                    edit_top_k=args.edit_top_k, gt_relative_window=args.gt_window)
    profile = getattr(args, "retrieval_profile", "legacy")
    if augmentation_config is not None and profile != "robust":
        raise ValueError("GT missing augmentation requires --retrieval-profile robust")
    if profile == "legacy":
        config = RetrievalConfig(**settings)
    else:
        from .robust_retrieval import RobustRetrievalConfig

        config = RobustRetrievalConfig(
            **settings, profile=profile,
            context_top_k=args.context_top_k, vowel_top_k=args.vowel_top_k,
            context_edit_top_k=args.context_edit_top_k)
    sampling = SamplingConfig(hard_negatives=args.hard_negatives,
                              random_negatives=args.random_negatives, seed=args.seed)
    provenance = {"train_sha256": _hash_file(args.train), "catalog_sha256": _hash_file(args.catalog),
                  "feature_code_sha256": feature_code_hash(),
                  "train_path": str(Path(args.train).resolve()), "catalog_path": str(Path(args.catalog).resolve())}
    result = prepare_training_data(
        read_queries(args.train, labeled=True), read_catalog(args.catalog), Path(args.output),
        outer_split=args.split, outer_folds=args.folds, outer_fold=args.fold,
        inner_folds=args.inner_folds, inner_split=args.inner_split, seed=args.seed,
        retrieval_config=config, sampling_config=sampling,
        max_train_queries=args.max_train_queries, max_valid_queries=args.max_valid_queries,
        shard_queries=args.shard_queries, provenance=provenance,
        augmentation_config=augmentation_config,
    )
    print(f"Completed training cache: {Path(args.output).resolve()}", flush=True)
    print(json.dumps(result.get("counts", {}), ensure_ascii=False), flush=True)
    if augmentation_config is not None:
        print(json.dumps(result["training_augmentation_diagnostics"], ensure_ascii=False, indent=2), flush=True)


def _check_augmented_features(metadata, selection):
    """Keep canonical86, with one explicitly scoped four-support-log ablation."""
    if metadata.get("training_augmentation") is None:
        return
    from .name_feature_cache import _legacy_feature_names, transform_spec
    from .name_features import FEATURE_NAMES

    source = list(_legacy_feature_names()) + list(FEATURE_NAMES)
    expected = [name for name in source if name != "retrieval_tonnage_rank"]
    from .support_ablation_assets import check_support_ablation_metadata, feature_schemas

    if list(selection.selected_features) == feature_schemas()["experiment_features"]:
        check_support_ablation_metadata(metadata)
        if list(selection.source_feature_names) != source:
            raise ValueError("Support-log ablation source schema differs from the existing cache")
        return
    if (metadata.get("feature_transform") != transform_spec()
            or metadata.get("feature_names") != source
            or list(selection.selected_features) != expected):
        raise ValueError("GT augmentation requires token-phonetic cache and canonical 86 features; "
                         "use --exclude-features retrieval_tonnage_rank")


def _check_support_ranker_config(metadata, selection, config):
    if metadata.get("training_augmentation") is None:
        return
    from .support_ablation_assets import feature_schemas
    from .ranking import RankerConfig

    if (list(selection.selected_features) == feature_schemas()["experiment_features"]
            and config != RankerConfig()):
        raise ValueError("Support-log ablation fixes the baseline ranker configuration (300 rounds, 6 threads)")


def _training_weighting_summary(group, language, query_mass, hangul_weight):
    """Record the mass passed to fit and its intended expanded row-weight budget."""
    if query_mass is None:
        return None
    present_budget = sum(budget for flag, budget in ((True, hangul_weight), (False, 1 - hangul_weight))
                         if np.any(language == flag))
    languages = {}
    for flag, name, budget in ((True, "hangul", hangul_weight), (False, "non_hangul", 1 - hangul_weight)):
        selected = language == flag
        languages[name] = {
            "groups": int(selected.sum()), "source_mass": float(query_mass[selected].sum()),
            "normalized_weight_share": float(budget / present_budget) if selected.any() else 0.,
        }
    return {
        "name": "source_mass_language_candidate_balance_v1",
        "ranking_code_sha256": _hash_file(Path(__file__).with_name("ranking.py")),
        "groups": int(len(group)), "pairs": int(group.sum()),
        "source_mass_total": float(query_mass.sum()),
        "query_mass_min": float(query_mass.min()), "query_mass_max": float(query_mass.max()),
        "languages": languages,
        "policy": "Normalize source masses within language; apply language budget; divide by candidate count; row-weight mean 1",
        "note": "Row sample weights are a training heuristic, not exact per-source LambdaRank gradient contributions.",
    }


def _check_training_weighting(summary):
    """Reject augmented bundles without their recorded source-mass training policy."""
    dataset = summary["dataset"]
    weighting = summary.get("training_weighting")
    if dataset.get("training_augmentation") is None:
        if weighting is not None:
            raise ValueError("Training weighting requires augmentation metadata")
        return
    if not isinstance(weighting, dict):
        raise ValueError("Augmented model is missing training weighting proof")
    diagnostics = dataset["training_augmentation_diagnostics"]
    try:
        budget = float(summary["ranker"]["hangul_weight"])
        language_data = weighting["languages"]
        totals = {name: float(diagnostics[name + "_query_mass"])
                  for name in ("hangul", "non_hangul")}
        denominator = sum(value for name, value in (("hangul", budget), ("non_hangul", 1 - budget))
                          if totals[name] > 0)
        valid = (
            weighting["name"] == "source_mass_language_candidate_balance_v1"
            and weighting["ranking_code_sha256"] == _hash_file(Path(__file__).with_name("ranking.py"))
            and weighting["groups"] == dataset["counts"]["train_groups"]
            and weighting["pairs"] == dataset["counts"]["train_pairs"]
            and np.isclose(weighting["source_mass_total"], diagnostics["total_query_mass"], atol=1e-8, rtol=0)
            and 0 < weighting["query_mass_min"] <= weighting["query_mass_max"] <= 1
            and 0 < budget < 1
            and set(language_data) == set(totals)
            and sum(item["groups"] for item in language_data.values()) == weighting["groups"])
        for name, share in (("hangul", budget), ("non_hangul", 1 - budget)):
            entry = language_data[name]
            expected_share = share / denominator if totals[name] > 0 else 0.
            valid = valid and (
                isinstance(entry["groups"], int) and entry["groups"] >= 0
                and (entry["groups"] > 0) == (totals[name] > 0)
                and np.isclose(entry["source_mass"], totals[name], atol=1e-8, rtol=0)
                and np.isclose(entry["normalized_weight_share"], expected_share, atol=1e-12, rtol=0))
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("Invalid training weighting proof") from error
    if not valid:
        raise ValueError("Training weighting proof differs from augmented dataset or implementation")


def run_train(args):
    from .ranking import RankerConfig, VesselRanker
    from .training_data import load_training_arrays, read_metadata

    data = Path(args.data)
    dataset_metadata = read_metadata(data)
    if _has_full_training_metadata(dataset_metadata):
        raise ValueError("Use train-full-ranker for full-training caches; no held-out score is available")
    _check_feature_code(dataset_metadata)
    if dataset_metadata.get("retrieval_extension") is not None:
        pipeline = _load_retrieval_index(data / dataset_metadata.get("index_path", "index.joblib"),
                                         dataset_metadata)
        del pipeline
    if dataset_metadata.get("training_reweighting") is not None:
        from .gt_reweight_cache import verify_reweighting

        print("Verifying mass-only cache lineage and unchanged candidate/feature/label arrays", flush=True)
        verify_reweighting(data)
    elif dataset_metadata.get("feature_transform") is not None:
        from .feature_transforms import verify_derivation

        print("Verifying transformed cache lineage and file integrity", flush=True)
        verify_derivation(data, dataset_metadata["derivation"]["parent_metadata_sha256"])
    selection = FeatureSelection.resolve(dataset_metadata["feature_names"],
                                         include=getattr(args, "include_features", None),
                                         exclude=getattr(args, "exclude_features", None))
    _check_augmented_features(dataset_metadata, selection)
    config = RankerConfig(n_estimators=args.rounds, learning_rate=args.learning_rate,
                          num_leaves=args.num_leaves, min_child_samples=args.min_child_samples,
                          reg_lambda=args.reg_lambda, n_jobs=args.threads, seed=args.seed,
                          hangul_weight=args.hangul_weight)
    _check_support_ranker_config(dataset_metadata, selection, config)
    output = _new_output(args.output)
    started = time.perf_counter()
    print("Loading sampled training pairs (held-out pairs are excluded)", flush=True)
    X, y, group, language, feature_names, query_mass = load_training_arrays(
        data, feature_names=list(selection.selected_features), return_query_mass=True)
    print(f"Feature selection: {len(feature_names)}/{len(selection.source_feature_names)} retained; "
          f"excluded={selection.to_dict()['excluded_features']}", flush=True)
    print(f"Training {len(group):,} query groups, {len(y):,} pairs, {len(feature_names)} features", flush=True)
    weighting = _training_weighting_summary(group, language, query_mass, config.hangul_weight)
    fit_options = {"query_mass": query_mass} if query_mass is not None else {}
    ranker = VesselRanker(config).fit(X, y, group, language, feature_names, **fit_options)
    training_seconds = time.perf_counter() - started
    del X, y, group, language, query_mass, fit_options
    ranker.save(output)
    shutil.copy2(data / dataset_metadata.get("index_path", "index.joblib"), output / "index.joblib")
    print("Scoring the complete held-out candidate sets", flush=True)
    predictions, metrics = evaluate_cache(ranker, data)
    predictions.to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    _write_json(output / "metrics.json", metrics)
    importance = ranker.feature_importance()
    importance.to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")
    _write_json(output / "training_summary.json", {
        "complete": True, "schema_version": 1, "dataset": dataset_metadata,
        "cache_directory": str(data.resolve()), "cache_metadata_sha256": _hash_file(data / "metadata.json"),
        "feature_selection": selection.to_dict(), "trained_feature_names": feature_names,
        "ranker": asdict(config), "train_load_and_fit_seconds": training_seconds,
        "total_seconds": time.perf_counter() - started,
        "metrics": metrics,
        "note": "Fixed training rounds; held-out validation was never used to fit or early-stop the model.",
        **({"training_weighting": weighting} if weighting is not None else {}),
    })
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved model and validation: {output.resolve()}", flush=True)


def run_predict(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a new output file: {output}")
    queries = read_queries(args.queries)
    matcher = RankedMatcher(args.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for number, row in enumerate(queries.to_dict("records"), 1):
        records.append({"sample_id": row["sample_id"], "target_vessel_id": matcher.predict(row)})
        if number % 100 == 0:
            print(f"Predicted {number:,}/{len(queries):,} queries", flush=True)
    pd.DataFrame(records, columns=["sample_id", "target_vessel_id"]).to_csv(output, index=False)
    print(f"Saved predictions: {output.resolve()}", flush=True)


def run_transform(args):
    transform_name = getattr(args, "transform", "tonnage-min-rank")
    from .training_data import read_metadata

    metadata = read_metadata(args.data)
    if _has_full_training_metadata(metadata):
        if transform_name != "token-phonetic":
            raise ValueError("Full training fixes token-phonetic features; other transforms are unsupported")
        from .full_ranking import transform_full_cache

        result = transform_full_cache(args.data, args.output, args.train, args.catalog)
        print(f"Completed full-training name cache: {Path(args.output).resolve()}", flush=True)
        print(json.dumps(result.get("counts", {}), ensure_ascii=False), flush=True)
        return
    if (metadata.get("training_augmentation") is not None
            and transform_name != "token-phonetic"):
        raise ValueError("GT augmentation fixes token-phonetic features; other transforms require a separate experiment")
    if transform_name == "token-phonetic":
        from .name_feature_cache import transform_cache
    elif transform_name == "ch-variants":
        from .ch_variant_cache import transform_cache
    elif transform_name == "tonnage-min-rank":
        from .tonnage_rank import transform_cache
    else:
        raise ValueError(f"Unsupported cache transform: {transform_name}")

    result = transform_cache(args.data, args.output, args.train, args.catalog)
    print(f"Completed derived feature cache: {Path(args.output).resolve()}", flush=True)
    print(json.dumps(result.get("counts", {}), ensure_ascii=False), flush=True)


def run_reweight(args):
    from .gt_augmentation import GTMissingAugmentationConfig
    from .gt_reweight_cache import reweight_cache
    from .training_data import read_metadata

    GTMissingAugmentationConfig(original_mass=args.original_mass)
    if _has_full_training_metadata(read_metadata(args.data)):
        raise ValueError("Full training fixes original_mass=0.5; do not reweight the final-training cache")

    result = reweight_cache(args.data, args.output, original_mass=args.original_mass)
    print(f"Completed mass-only training cache: {Path(args.output).resolve()}", flush=True)
    print(json.dumps(result["training_augmentation_diagnostics"], ensure_ascii=False, indent=2), flush=True)


def add_training_parsers(subparsers):
    from .ranker_comparison import add_comparison_parser
    from .retrieval_comparison import add_retrieval_comparison_parser
    from .full_training_commands import add_full_training_parsers
    from .index_scale_comparison import add_index_scale_parser
    from .index_factorial_comparison import add_index_factorial_parser
    from .category_scope_comparison import add_category_scope_parser
    from .support_ablation_comparison import add_support_ablation_parser

    add_support_ablation_parser(subparsers)
    add_category_scope_parser(subparsers)
    add_index_factorial_parser(subparsers)
    add_index_scale_parser(subparsers)
    add_full_training_parsers(subparsers)
    add_comparison_parser(subparsers)
    add_retrieval_comparison_parser(subparsers)
    reweight = subparsers.add_parser(
        "reweight-ranker-cache", help="Change original GT augmentation mass while preserving cached pairs")
    reweight.add_argument("--data", required=True, help="Completed augmented token-phonetic source cache")
    reweight.add_argument("--output", required=True, help="A new or empty cache directory; source is never modified")
    reweight.add_argument("--original-mass", required=True, type=float,
                          help="Original share within each retained OOF view; strictly between 0 and 1")
    transform = subparsers.add_parser(
        "transform-ranker-cache", help="Derive versioned features without repeating retrieval or sampling")
    transform.add_argument("--transform", choices=["tonnage-min-rank", "token-phonetic", "ch-variants"],
                           default="tonnage-min-rank", help="Tie rank, name features, or CH variants on a token-phonetic cache")
    transform.add_argument("--data", required=True, help="Completed parent cache (ch-variants requires token-phonetic)")
    transform.add_argument("--output", required=True, help="A new or empty derived cache directory")
    transform.add_argument("--train", default="work/data/train.csv")
    transform.add_argument("--catalog", default="work/data/vessel_catalog.csv")
    prepare = subparsers.add_parser("prepare-ranker", help="Generate inner-OOF training pairs and complete held-out pairs")
    prepare.add_argument("--train", default="work/data/train.csv")
    prepare.add_argument("--catalog", default="work/data/vessel_catalog.csv")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--split", choices=["vessel", "row"], default="vessel")
    prepare.add_argument("--folds", type=int, default=5)
    prepare.add_argument("--fold", type=int, default=0)
    prepare.add_argument("--inner-folds", type=int, default=3)
    prepare.add_argument("--inner-split", choices=["mixed", "vessel", "row"], default="mixed")
    prepare.add_argument("--seed", type=int, default=20261022)
    prepare.add_argument("--hard-negatives", type=int, default=96)
    prepare.add_argument("--random-negatives", type=int, default=32)
    prepare.add_argument("--max-train-queries", type=int, default=None, help="Diagnostic source-query limit, before generating the inner views")
    prepare.add_argument("--max-valid-queries", type=int, default=None, help="Diagnostic held-out query limit")
    prepare.add_argument("--shard-queries", type=int, default=64)
    prepare.add_argument("--name-top-k", type=int, default=80)
    prepare.add_argument("--phonetic-top-k", type=int, default=80)
    prepare.add_argument("--edit-top-k", type=int, default=40)
    prepare.add_argument("--gt-window", type=float, default=.10)
    prepare.add_argument("--retrieval-profile", choices=["legacy", "hangul", "dual-gt", "robust"],
                         default="legacy", help="Versioned candidate routes; legacy preserves previous caches/models")
    prepare.add_argument("--context-top-k", type=int, default=160)
    prepare.add_argument("--vowel-top-k", type=int, default=160)
    prepare.add_argument("--context-edit-top-k", type=int, default=80)
    prepare.add_argument("--gt-augmentation", choices=["none", "missing"], default="none",
                         help="Training-only original plus three GT missing masks; requires robust retrieval")
    prepare.add_argument("--original-mass", type=float, default=None,
                         help="Original share within each retained OOF view (missing augmentation default: 0.5)")
    train = subparsers.add_parser("train-ranker", help="Fit fixed-round LightGBM, then evaluate full held-out candidates")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--rounds", type=int, default=300)
    train.add_argument("--learning-rate", type=float, default=.05)
    train.add_argument("--num-leaves", type=int, default=31)
    train.add_argument("--min-child-samples", type=int, default=20)
    train.add_argument("--reg-lambda", type=float, default=1.0)
    train.add_argument("--threads", type=int, default=6)
    train.add_argument("--seed", type=int, default=20261022)
    train.add_argument("--hangul-weight", type=float, default=.7)
    selection = train.add_mutually_exclusive_group()
    selection.add_argument("--include-features", nargs="+", action="extend", default=None,
                           metavar="NAME_OR_PATTERN", help="Keep matching cached columns, in cache order")
    selection.add_argument("--exclude-features", nargs="+", action="extend", default=None,
                           metavar="NAME_OR_PATTERN", help="Drop matching cached columns without regenerating features")
    predict = subparsers.add_parser("predict-ranker", help="Predict each query independently with a trusted local model bundle")
    predict.add_argument("--model", required=True)
    predict.add_argument("--queries", required=True)
    predict.add_argument("--output", required=True, help="A new CSV file")


def run_training_command(args):
    from .ranker_comparison import run_compare
    from .retrieval_comparison import run_comparison
    from .full_training_commands import COMMANDS as FULL_COMMANDS, run_full_training_command
    from .index_scale_comparison import run_index_scale
    from .index_factorial_comparison import run_index_factorial
    from .category_scope_comparison import run_category_scope
    from .support_ablation_comparison import run_support_ablation

    if args.command in FULL_COMMANDS:
        return run_full_training_command(args)
    {"prepare-ranker": run_prepare, "transform-ranker-cache": run_transform,
     "reweight-ranker-cache": run_reweight, "train-ranker": run_train,
     "predict-ranker": run_predict, "compare-rankers": run_compare,
     "compare-retrieval": run_comparison, "compare-index-scale": run_index_scale,
     "compare-index-factorial": run_index_factorial,
     "compare-category-scope": run_category_scope,
     "compare-support-ablation": run_support_ablation}[args.command](args)
