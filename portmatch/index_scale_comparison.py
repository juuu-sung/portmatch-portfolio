"""Held-out index-size sensitivity with one immutable, already trained ranker.

The saved T-trained model scores both the saved T index and a new index fitted
on a deterministic 80% of T's vessels. V never fits either index or the ranker.
This is a diagnostic, not a full-data model evaluation or a deployable bundle.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .data import QUERY_COLUMNS
from .retrieval_comparison import (
    VARIANTS, _check_saved_predictions, _hash, _initialize_name_features,
    _write_json, select_queries, variant_effect_counts, variant_query,
)
from .text import has_hangul


RUNNERS = ("index_80", "index_100")
DIAGNOSTIC_FEATURES = (
    "retrieval_name_score", "retrieval_name_rank", "retrieval_phonetic_score",
    "retrieval_phonetic_rank", "retrieval_lookup_support", "retrieval_lookup_purity",
    "flag_probability", "flag_lift_log", "type_probability", "type_lift_log",
    "retrieval_channel_count",
)
DETAIL_COLUMNS = ["variant", "sample_id", "target_vessel_id", "vessel_name", "has_hangul",
    "candidate_vessel_id", "candidate_catalog_name", "is_target", "is_index_80_top",
    "is_index_100_top", "model", "present", "score", *DIAGNOSTIC_FEATURES]


def detailed_prediction(matcher, query, target):
    """Rank once; inspect target/candidate features only after all scores exist."""
    from .feature_transforms import transform_features
    from .ranker_commands import score_order

    query = {key: query.get(key) for key in QUERY_COLUMNS if key != "sample_id"}
    started = time.perf_counter()
    candidates = matcher.pipeline.retrieve(query)
    retrieved = time.perf_counter()
    indices = candidates.catalog_index.to_numpy()
    if (not len(indices) or indices.dtype.kind not in "iuf" or not np.isfinite(indices).all()
            or not np.equal(indices, np.floor(indices)).all() or len(set(indices)) != len(indices)
            or (indices < 0).any() or (indices >= len(matcher.catalog_ids)).any()):
        raise ValueError("Retriever produced invalid, duplicate or empty catalog positions")
    indices = indices.astype(np.int64)
    features = matcher.pipeline.features(query, candidates)
    features = transform_features(matcher.feature_transform, matcher.pipeline, query, candidates, features)
    selected = matcher.selection.apply(features.to_numpy(dtype=np.float32), features.columns.tolist())
    featured = time.perf_counter()
    scores = matcher.ranker.predict(selected, list(matcher.selection.selected_features))
    order = score_order(scores, indices, matcher.catalog_ids)
    finished = time.perf_counter()
    ids = matcher.catalog_ids[indices]
    ranked_ids = ids[order]
    locations = np.flatnonzero(ranked_ids == str(target))
    result = {
        "predicted_vessel_id": str(ranked_ids[0]), "candidate_found": bool(len(locations)),
        "target_rank": int(locations[0] + 1) if len(locations) else 0,
        "candidate_count": len(indices), "top_score": float(scores[order[0]]),
        "score_margin": float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else None,
        "retrieval_seconds": retrieved - started, "feature_seconds": featured - retrieved,
        "rank_seconds": finished - featured, "total_seconds": finished - started,
    }
    # Same signature definition as the existing retrieval comparison.
    import json
    result["candidate_ids_sha256"] = hashlib.sha256(json.dumps(
        sorted(ids.tolist()), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    missing = set(DIAGNOSTIC_FEATURES) - set(features.columns)
    if missing:
        raise ValueError(f"Diagnostic feature schema is missing: {sorted(missing)}")
    details = features.loc[:, list(DIAGNOSTIC_FEATURES)].copy()
    details.index = pd.Index(ids.astype(str), name="candidate_vessel_id")
    details["score"] = scores
    details["candidate_catalog_name"] = matcher.pipeline.catalog.iloc[indices].vessel_name.to_numpy()
    for prefix, vessel in (("top", str(ranked_ids[0])), ("target", str(target))):
        present = vessel in details.index
        result[prefix + "_catalog_name"] = str(details.loc[vessel, "candidate_catalog_name"]) if present else None
        for feature in DIAGNOSTIC_FEATURES:
            value = float(details.loc[vessel, feature]) if present else None
            result[prefix + "_" + feature] = value if value is not None and np.isfinite(value) else None
    return result, details


def _changed_details(pair, row, variant):
    """Align the very same candidate across indexes, even when their tops differ."""
    left, right = (pair[name][0]["predicted_vessel_id"] for name in RUNNERS)
    if left == right:
        return []
    selected = sorted({str(row["target_vessel_id"]), left, right})
    output = []
    for vessel in selected:
        names = [details.loc[vessel, "candidate_catalog_name"] for _, details in pair.values()
                 if vessel in details.index]
        for model in RUNNERS:
            details = pair[model][1]
            present = vessel in details.index
            record = {"variant": variant, "sample_id": str(row["sample_id"]),
                "target_vessel_id": str(row["target_vessel_id"]),
                "vessel_name": row["vessel_name"] if isinstance(row["vessel_name"], str) else "",
                "has_hangul": has_hangul(row["vessel_name"]), "candidate_vessel_id": vessel,
                "candidate_catalog_name": names[0] if names else None,
                "is_target": vessel == str(row["target_vessel_id"]),
                "is_index_80_top": vessel == left, "is_index_100_top": vessel == right,
                "model": model, "present": present}
            for field in ("score", *DIAGNOSTIC_FEATURES):
                value = float(details.loc[vessel, field]) if present else None
                record[field] = value if value is not None and np.isfinite(value) else None
            output.append(record)
    return output


def _ranker_fingerprint(ranker):
    return hashlib.sha256(ranker._booster.model_to_string().encode("utf-8")).hexdigest()


def compare_index_scale(model_directory, output, *, train_path="work/data/train.csv",
                        catalog_path="work/data/vessel_catalog.csv", subset_seed=20261027,
                        expected_fold=None, variants=VARIANTS, max_queries=None, progress_every=100):
    """Run fixed 80% versus 100% fitting scopes; never fit any model weights."""
    from .index_scale_assets import prepare_index_scale, assert_unchanged, learned_assets_fingerprint
    from .index_scale_report import build_index_scale_report

    variants = tuple(variants)
    if (not variants or len(set(variants)) != len(variants) or "original" not in variants
            or set(variants) - set(VARIANTS)):
        raise ValueError("Use unique known variants including original for saved-model parity")
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("progress-every must be a positive integer")
    if max_queries is not None and (type(max_queries) is not int or max_queries < 1):
        raise ValueError("max-queries must be a positive integer")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new or empty output directory: {output}")
    started = time.perf_counter()
    context = prepare_index_scale(model_directory, train_path, catalog_path,
        fraction=.8, subset_seed=subset_seed, expected_fold=expected_fold)
    full = context["matcher"]
    small = copy.copy(full)
    small.pipeline = context["small_pipeline"]
    runners = {"index_80": small, "index_100": full}
    if small.ranker is not full.ranker or small.selection is not full.selection:
        raise AssertionError("Both indexes must share the exact frozen ranker and feature selection")
    weights_before = _ranker_fingerprint(full.ranker)
    assets_before = {name: learned_assets_fingerprint(runner.pipeline) for name, runner in runners.items()}
    selected = select_queries(context["heldout"], max_queries=max_queries, seed=subset_seed)
    if selected.empty:
        raise ValueError("No held-out rows selected")
    coverage = dict(context["proof"]["coverage"])
    coverage.update(queries=len(selected), heldout_queries=len(context["heldout"]),
        vessels=int(selected.target_vessel_id.nunique()),
        diagnostic_sample=len(selected) != len(context["heldout"]), max_queries=max_queries)
    startup = dict(context["proof"].get("startup", {}))
    startup["name_feature_index_seconds"] = _initialize_name_features(full)
    # This object depends solely on the unchanged official catalog names.
    small.pipeline._token_phonetic_computer_v1 = full.pipeline._token_phonetic_computer_v1
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1, "complete": False, "purpose": "fixed_ranker_index_scope_sensitivity",
        "coverage": coverage, "proof": context["proof"], "variants": list(variants), "startup": startup,
        "variant_effects": variant_effect_counts(selected, variants),
        "provenance": {"code_sha256": {p: _hash(Path(__file__).with_name(p)) for p in
            ("index_scale_comparison.py", "index_scale_assets.py", "index_scale_report.py")},
            "model_directory": str(Path(model_directory).resolve()),
            "in_memory_ranker_sha256": weights_before},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "training_performed": False, "submission_created": False}
    _write_json(output / "summary.json", metadata)
    context["membership"].to_csv(output / "partitions.csv", index=False, encoding="utf-8-sig")
    startup["warmup_seconds"] = {}
    for name, runner in runners.items():
        warmup = time.perf_counter()
        # No target is needed for warmup; detailed_prediction only uses it post-score.
        detailed_prediction(runner, variant_query(selected.iloc[0].to_dict(), "original"), "")
        startup["warmup_seconds"][name] = time.perf_counter() - warmup
    records, details_records = [], []
    rows = selected.to_dict("records")
    for position, row in enumerate(rows):
        for variant_position, variant in enumerate(variants):
            query = variant_query(row, variant)
            order = RUNNERS if (position * len(variants) + variant_position) % 2 == 0 else RUNNERS[::-1]
            pair = {}
            for name in order:
                result, details = detailed_prediction(runners[name], query, row["target_vessel_id"])
                pair[name] = (result, details)
                records.append({"variant": variant, "model": name, "sample_id": str(row["sample_id"]),
                    "target_vessel_id": str(row["target_vessel_id"]),
                    "vessel_name": row["vessel_name"] if isinstance(row["vessel_name"], str) else "",
                    "has_hangul": has_hangul(row["vessel_name"]),
                    "runner_position": order.index(name), **result})
            details_records.extend(_changed_details(pair, row, variant))
        if (position + 1) % progress_every == 0 or position + 1 == len(rows):
            print(f"Index scope: {position + 1:,}/{len(rows):,} held-out rows, "
                  f"{len(records):,} predictions", flush=True)
    predictions = pd.DataFrame(records).sort_values(["variant", "sample_id", "model"], kind="stable")
    original = predictions.loc[predictions.variant.eq("original") & predictions.model.eq("index_100")]
    parity = _check_saved_predictions(model_directory, original, context["heldout"].sample_id, "index_100")
    assert_unchanged(context["protected_files"])
    if weights_before != _ranker_fingerprint(full.ranker):
        raise AssertionError("Model weights changed during diagnostic inference")
    if assets_before != {name: learned_assets_fingerprint(runner.pipeline) for name, runner in runners.items()}:
        raise AssertionError("Training-only lookup or category assets changed during inference")
    summary, changes, vessels, markdown = build_index_scale_report(predictions, coverage=coverage,
        proof=context["proof"], startup=startup, parity=parity, variants=variants)
    summary.update({key: value for key, value in metadata.items()
                    if key not in {"complete", "coverage", "proof", "startup"}})
    summary.update(complete=True, weights_unchanged=True, training_assets_unchanged=True,
        protected_files_unchanged=True, ranker_shared_by_both_indexes=True,
        paired_candidate_detail_rows=len(details_records),
        changed_candidate_features_file="changed_candidate_features.csv",
        total_seconds=time.perf_counter() - started)
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(output / "row_changes.csv", index=False, encoding="utf-8-sig")
    vessels.to_csv(output / "vessel_changes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(details_records, columns=DETAIL_COLUMNS).to_csv(
        output / "changed_candidate_features.csv", index=False, encoding="utf-8-sig")
    markdown += ("\n\n예측이 바뀐 행의 동일 후보별 점수·특징은 changed_candidate_features.csv에 기록합니다. "
        "정답과 두 사전의 1위 후보를 양쪽 사전에 정렬하고, 후보 부재는 present=false로 표시합니다. "
        "이 진단 추출·파일 저장 시간과 초기화·warmup은 행별 추론 시간에서 제외합니다.\n")
    (output / "report.md").write_text(markdown, encoding="utf-8")
    summary["output_files_sha256"] = {p.name: _hash(p) for p in output.iterdir()
        if p.is_file() and p.name != "summary.json"}
    _write_json(output / "summary.json", summary)
    print(f"Completed index-scope diagnostic without training: {output}", flush=True)
    return summary


def add_index_scale_parser(subparsers):
    parser = subparsers.add_parser("compare-index-scale", help="Fixed robust86 ranker: train-only 80%/100% indexes")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", required=True, type=int, choices=[0, 2], help="Must match the saved vessel fold")
    parser.add_argument("--train", default="work/data/train.csv")
    parser.add_argument("--catalog", default="work/data/vessel_catalog.csv")
    parser.add_argument("--output", required=True, help="New or empty diagnostic output directory")
    parser.add_argument("--subset-seed", type=int, default=20261027)
    parser.add_argument("--max-queries", type=int, help="Optional smoke-test subset; omit for full held-out validation")
    parser.add_argument("--progress-every", type=int, default=100)


def run_index_scale(args):
    return compare_index_scale(args.model, args.output, train_path=args.train, catalog_path=args.catalog,
        subset_seed=args.subset_seed, expected_fold=args.fold, max_queries=args.max_queries,
        progress_every=args.progress_every)
