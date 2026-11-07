"""Live, paired retrieval experiments on one model's public held-out fold.

No feature caches are scored and no labels are used to build experimental
retrievers. Each named input variant is applied identically to every runner.
The default evaluates every held-out query; diagnostic subsampling is explicit.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from .data import QUERY_COLUMNS, read_catalog, read_queries
from .ranking import ranking_metrics
from .text import has_hangul
from .validation import make_folds


PROFILES = ("hangul", "dual-gt", "robust")
VARIANTS = ("original", "both_gt_missing", "international_gt_missing",
            "gross_gt_missing", "international_gt_x1_25")
TIMING_COLUMNS = ("retrieval_seconds", "feature_seconds", "rank_seconds", "total_seconds")
_MATCH_FIELDS = ("outer_split", "outer_folds", "outer_fold", "inner_split", "inner_folds",
                 "seed", "sampling", "max_train_queries", "max_valid_queries", "sampled")


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def variant_query(row, variant):
    """Return only inference inputs, without labels or even sample identifiers."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown input variant: {variant}")
    query = {key: row.get(key) for key in QUERY_COLUMNS if key != "sample_id"}
    if variant in ("both_gt_missing", "gross_gt_missing"):
        query["gross_tonnage"] = np.nan
    if variant in ("both_gt_missing", "international_gt_missing"):
        query["international_gross_tonnage"] = np.nan
    if variant == "international_gt_x1_25":
        value = pd.to_numeric(query.get("international_gross_tonnage"), errors="coerce")
        if pd.notna(value) and np.isfinite(value) and value > 0:
            query["international_gross_tonnage"] = float(value) * 1.25
    return query


def variant_effect_counts(frame, variants):
    """Count real per-row input changes; an already missing tonnage stays missing."""
    def equal(left, right):
        if pd.isna(left) and pd.isna(right):
            return True
        return bool(left == right) if pd.notna(left) and pd.notna(right) else False
    rows = frame.to_dict("records")
    language = np.array([has_hangul(row["vessel_name"]) for row in rows], dtype=bool)
    result = {}
    for variant in variants:
        changed = []
        for row in rows:
            original, transformed = variant_query(row, "original"), variant_query(row, variant)
            changed.append(any(not equal(original[key], transformed[key])
                               for key in ("gross_tonnage", "international_gross_tonnage")))
        changed = np.asarray(changed, dtype=bool)
        result[variant] = {name: {"queries": int(mask.sum()), "changed": int(changed[mask].sum()),
                                  "unchanged": int((~changed[mask]).sum())}
            for name, mask in (("all", np.ones(len(rows), dtype=bool)),
                               ("hangul", language), ("non_hangul", ~language))}
    return result


def select_queries(frame, *, max_vessels_per_language=None, max_queries=None,
                   one_row_per_vessel=False, seed=20261022):
    if max_vessels_per_language is not None and max_queries is not None:
        raise ValueError("Choose only one diagnostic selection limit")
    for value in (max_vessels_per_language, max_queries):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("Diagnostic limits must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("Selection seed must be a nonnegative integer")
    if one_row_per_vessel and max_vessels_per_language is None:
        raise ValueError("one-row-per-vessel requires max-vessels-per-language")
    def priority(value):
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    if max_vessels_per_language is not None:
        language = frame.vessel_name.map(has_hangul)
        selected = pd.Series(False, index=frame.index)
        for flag in (False, True):
            group = frame.loc[language.eq(flag)]
            vessels = sorted(group.target_vessel_id.astype(str).unique(), key=priority)
            selected |= language.eq(flag) & frame.target_vessel_id.astype(str).isin(vessels[:max_vessels_per_language])
        result = frame.loc[selected].copy()
        if one_row_per_vessel:
            result = result.assign(_language=result.vessel_name.map(has_hangul),
                                   _priority=result.sample_id.astype(str).map(priority))
            result = result.sort_values("_priority", kind="stable").drop_duplicates(
                ["_language", "target_vessel_id"]).drop(columns=["_language", "_priority"])
            result = result.loc[frame.index.intersection(result.index)]
        return result
    if max_queries is not None and max_queries < len(frame):
        # Include both languages when the budget permits, then fill by stable ID
        # hash. This selects evaluation rows only; no parameter is fitted here.
        ordered = sorted(frame.sample_id.astype(str), key=priority)
        selected = []
        if max_queries >= 2:
            for flag in (False, True):
                group = set(frame.loc[frame.vessel_name.map(has_hangul).eq(flag), "sample_id"].astype(str))
                if group:
                    selected.append(next(value for value in ordered if value in group))
        selected += [value for value in ordered if value not in selected][:max_queries - len(selected)]
        return frame.loc[frame.sample_id.astype(str).isin(selected)].copy()
    return frame.copy()


def _verify_index(summary, directory):
    """Check the frozen index proof, without loading the large pair shards."""
    dataset = summary["dataset"]
    index_hash = _hash(directory / "index.joblib")
    if dataset.get("retrieval_extension") is not None:
        if index_hash != dataset.get("retrieval_index_sha256"):
            raise ValueError("Experimental saved index hash differs from its training provenance")
    else:
        from .submission_package import _verified_index_hash
        _verified_index_hash(summary, (directory / "index.joblib").read_bytes(), None)
    return index_hash


def _load_model(directory, train_hash, catalog_hash):
    from .name_feature_cache import transform_spec, _legacy_feature_names
    from .name_features import FEATURE_NAMES
    from .ranker_commands import RankedMatcher

    directory = Path(directory).resolve()
    summary = _json(directory / "training_summary.json")
    dataset = summary.get("dataset", {})
    if summary.get("complete") is not True or dataset.get("complete") is not True:
        raise ValueError(f"Incomplete model: {directory}")
    from .gt_augmentation import check_augmentation_metadata
    check_augmentation_metadata(dataset)
    if "training_reweighting" in dataset:
        from .gt_reweight_cache import check_reweighting_metadata
        check_reweighting_metadata(dataset)
    if dataset.get("feature_transform") != transform_spec():
        raise ValueError("Comparison requires unchanged token_phonetic_v1 features")
    source = list(_legacy_feature_names()) + list(FEATURE_NAMES)
    expected = [name for name in source if name != "retrieval_tonnage_rank"]
    if len(expected) != 86 or dataset.get("feature_names") != source:
        raise ValueError("Comparison requires the canonical 86-feature model schema")
    provenance = dataset.get("provenance", {})
    if provenance.get("train_sha256") != train_hash or provenance.get("catalog_sha256") != catalog_hash:
        raise ValueError("Public train/catalog content differs from model provenance")
    if dataset.get("sampled") is not False or dataset.get("max_train_queries") is not None:
        raise ValueError("Use a complete fold model, not a sampled training artifact")
    started = time.perf_counter()
    index_hash = _verify_index(summary, directory)
    matcher = RankedMatcher(directory)
    if matcher.ranker.feature_names != expected or summary.get("trained_feature_names") != expected:
        raise ValueError("Model must retain exactly 86 features, excluding only tonnage rank")
    if asdict(matcher.ranker.config) != summary.get("ranker"):
        raise ValueError("Native ranker settings differ from the training summary")
    return matcher, summary, {"model_directory": str(directory),
        "model_sha256": _hash(directory / "model.txt"), "index_sha256": index_hash,
        "training_summary_sha256": _hash(directory / "training_summary.json"),
        "load_and_integrity_seconds": time.perf_counter() - started}


def _validate_partition(matcher, summary, train, catalog):
    data = summary["dataset"]
    fold, folds, seed = data.get("outer_fold"), data.get("outer_folds"), data.get("seed")
    if (type(fold) is not int or type(folds) is not int or folds < 2 or not 0 <= fold < folds
            or type(seed) is not int or seed < 0):
        raise ValueError("Invalid saved outer split")
    assignment = make_folds(train, data["outer_split"], folds, seed)
    fitting, heldout = train.loc[assignment != fold], train.loc[assignment == fold]
    if not matcher.pipeline.catalog.equals(catalog):
        raise ValueError("Saved catalog positions/content differ from the supplied catalog")
    memory = matcher.pipeline.memory
    if (memory.training_rows != len(fitting)
            or set(memory.training_vessels) != set(fitting.target_vessel_id.astype(str))):
        raise ValueError("Saved lookup memory does not match the model's training partition")
    if data["outer_split"] == "vessel" and set(memory.training_vessels) & set(heldout.target_vessel_id.astype(str)):
        raise ValueError("Held-out vessels leaked into lookup memory")
    if set(train.target_vessel_id.astype(str)) - set(matcher.catalog_ids):
        raise ValueError("Public target is absent from the supplied catalog")
    counts = data.get("counts", {})
    if (counts.get("outer_training_rows") != len(fitting)
            or counts.get("outer_validation_rows") != len(heldout)):
        raise ValueError("Saved partition counts disagree with deterministic split")
    return heldout.copy()


def _validate_training_assets(baseline, experiment):
    """Compare learned values, not the pickle ordering of unordered containers.

    joblib.hash does not canonicalize frozenset iteration, so identical vessel
    sets fitted in separate Python processes can produce different hashes.
    Check every learned field explicitly; retain alias row order and dtypes
    because they determine the retrieval index. Saved file hashes are checked
    separately by _load_model and must not be weakened here.
    """
    from .features import CategoryCompatibility
    from .memory import TrainingMemory

    for asset, cls, fields in (
        ("memory", TrainingMemory,
         {"call_signs", "names", "aliases", "training_rows", "training_vessels"}),
        ("categories", CategoryCompatibility, {"smoothing", "tables"}),
    ):
        left, right = getattr(baseline, asset), getattr(experiment, asset)
        message = f"Experimental {asset} differs from baseline training-only assets"
        # Fail closed if a future asset schema adds state that is not audited.
        if (type(left) is not cls or type(right) is not cls
                or set(vars(left)) != fields or set(vars(right)) != fields):
            raise ValueError(f"{message}: unsupported state schema")
        for field in sorted(fields):
            first, second = getattr(left, field), getattr(right, field)
            if field == "aliases":
                try:
                    pd.testing.assert_frame_equal(first, second, check_exact=True,
                        check_index_type=True, check_column_type=True, check_like=False)
                except AssertionError as error:
                    raise ValueError(f"{message}: {field}") from error
            else:
                if field in ("call_signs", "names"):
                    # Unlike Counter equality, dict equality also detects an
                    # added zero-count target, which lookup treats as a hit.
                    first = {key: dict(counts) for key, counts in first.items()}
                    second = {key: dict(counts) for key, counts in second.items()}
                if first != second:
                    raise ValueError(f"{message}: {field}")


def _validate_model_pair(baseline, baseline_summary, experiment, experiment_summary):
    from .gt_augmentation import check_augmentation_metadata
    left, right = baseline_summary["dataset"], experiment_summary["dataset"]
    check_augmentation_metadata(left)
    check_augmentation_metadata(right)
    for field in _MATCH_FIELDS:
        if field not in left or field not in right or left[field] != right[field]:
            raise ValueError(f"Experimental training differs on controlled setting: {field}")
    if (baseline_summary["ranker"] != experiment_summary["ranker"]
            or baseline.ranker.feature_names != experiment.ranker.feature_names):
        raise ValueError("Experimental model must retain identical ranker settings and 86 feature order")
    for field in ("train_sha256", "catalog_sha256", "feature_code_sha256"):
        if left["provenance"].get(field) != right["provenance"].get(field):
            raise ValueError(f"Experimental provenance differs on {field}")
    if right.get("retrieval_extension") is None:
        raise ValueError("Experimental model must declare its robust retrieval extension")
    for field, value in left["retrieval"].items():
        if right["retrieval"].get(field) != value:
            raise ValueError(f"Experimental legacy retrieval setting differs: {field}")
    _validate_training_assets(baseline.pipeline, experiment.pipeline)


def _validate_reference_pair(reference, reference_summary, experiment, experiment_summary):
    """Isolate augmentation while retaining the fixed retrieval/model controls."""
    from .gt_augmentation import check_augmentation_metadata
    left, right = reference_summary["dataset"], experiment_summary["dataset"]
    if check_augmentation_metadata(left) is not None:
        raise ValueError("Reference model must be the unaugmented robust-trained model")
    if check_augmentation_metadata(right) is None:
        raise ValueError("A reference model is only supported for a declared augmented experiment")
    _validate_model_pair(reference, reference_summary, experiment, experiment_summary)
    for field in ("retrieval", "retrieval_extension", "feature_transform", "feature_names"):
        if field not in left or left[field] != right.get(field):
            raise ValueError(f"Augmented experiment differs from reference on controlled setting: {field}")


def _validate_control_pair(control, control_summary, experiment, experiment_summary):
    """Bind a mass-only experiment to the exact augmented cache it reused."""
    from .gt_reweight_cache import validate_reweight_pair

    source_hash = control_summary.get("cache_metadata_sha256")
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(char not in "0123456789abcdef" for char in source_hash)):
        raise ValueError("Control model is missing its source cache metadata hash")
    _validate_model_pair(control, control_summary, experiment, experiment_summary)
    validate_reweight_pair(control_summary["dataset"], experiment_summary["dataset"],
                           source_metadata_sha256=source_hash)


def _initialize_name_features(matcher):
    from .name_features import NameFeatureComputer
    started = time.perf_counter()
    matcher.pipeline._token_phonetic_computer_v1 = NameFeatureComputer(matcher.pipeline.catalog.vessel_name.tolist())
    return time.perf_counter() - started


def timed_prediction(matcher, query, target):
    """The target is used after ranking only, never passed into the pipeline."""
    from .feature_transforms import transform_features
    from .ranker_commands import score_order
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
    ids = matcher.catalog_ids[indices[order]]
    locations = np.flatnonzero(ids == str(target))
    result = {"predicted_vessel_id": str(ids[0]), "candidate_found": bool(len(locations)),
        "target_rank": int(locations[0] + 1) if len(locations) else 0,
        "candidate_count": len(indices), "top_score": float(scores[order[0]]),
        "score_margin": float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else None,
        "retrieval_seconds": retrieved - started, "feature_seconds": featured - retrieved,
        "rank_seconds": finished - featured, "total_seconds": finished - started}
    # A content signature helps check candidate reproducibility without writing
    # every candidate ID (millions of rows) to reports.
    result["candidate_ids_sha256"] = hashlib.sha256(
        json.dumps(sorted(ids.tolist()), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return result


def summarize_predictions(records):
    result = ranking_metrics(records)
    for language, part in (("all", records), ("hangul", records.loc[records.has_hangul]),
                           ("non_hangul", records.loc[~records.has_hangul])):
        values = part.candidate_count.to_numpy(dtype=float)
        result[language]["candidate_count"] = {
            "mean": float(values.mean()) if len(values) else None,
            "p50": float(np.percentile(values, 50)) if len(values) else None,
            "p95": float(np.percentile(values, 95)) if len(values) else None,
            "max": int(values.max()) if len(values) else None}
        result[language]["timing"] = {column: {
            "sum": float(part[column].sum()),
            "mean": float(part[column].mean()) if len(part) else None,
            "p95": float(part[column].quantile(.95)) if len(part) else None,
        } for column in TIMING_COLUMNS}
    return result


def metric_deltas(baseline, experiment):
    """Experiment minus baseline, always under the same input variant."""
    def difference(left, right, scale=1.):
        return None if left is None or right is None else (right - left) * scale
    result = {"weighted_top1_pp": difference(baseline["weighted_top1"], experiment["weighted_top1"], 100.),
              "weighted_candidate_recall_pp": difference(baseline["weighted_candidate_recall"],
                                                          experiment["weighted_candidate_recall"], 100.)}
    for language in ("all", "hangul", "non_hangul"):
        left, right = baseline[language], experiment[language]
        result[language] = {
            "top1_accuracy_pp": difference(left["top1_accuracy"], right["top1_accuracy"], 100.),
            "candidate_recall_pp": difference(left["candidate_recall"], right["candidate_recall"], 100.),
            "candidate_count_delta": {name: difference(left["candidate_count"][name], right["candidate_count"][name])
                                       for name in ("mean", "p50", "p95", "max")},
            "timing": {name: {"sum_seconds_delta": right["timing"][name]["sum"] - left["timing"][name]["sum"],
                              "sum_ratio": right["timing"][name]["sum"] / left["timing"][name]["sum"]
                              if left["timing"][name]["sum"] > 0 else None}
                       for name in TIMING_COLUMNS}}
    return result


def paired_changes(records, *, reference="baseline", experiments=None):
    keys = ["variant", "sample_id"]
    identity = ["target_vessel_id", "has_hangul", "vessel_name"]
    baseline = records.loc[records.model.eq(reference)].drop(columns="model")
    if baseline.empty:
        raise ValueError(f"Missing paired reference runner: {reference}")
    candidates = records.loc[~records.model.eq(reference)]
    if experiments is not None:
        candidates = candidates.loc[candidates.model.isin(experiments)]
        if set(candidates.model) != set(experiments):
            raise ValueError("Missing paired experiment runner")
    reference_suffix = "_" + reference
    pieces = []
    for model, part in candidates.groupby("model", sort=False):
        joined = baseline.merge(part.drop(columns="model"), on=keys, validate="one_to_one",
                                suffixes=(reference_suffix, "_experiment"))
        if len(joined) != len(baseline) or len(part) != len(baseline):
            raise ValueError("Paired runners do not cover identical variant/sample IDs")
        for field in identity:
            if not joined[field + reference_suffix].equals(joined[field + "_experiment"]):
                raise ValueError(f"Paired identity mismatch: {field}")
            joined[field] = joined.pop(field + reference_suffix)
            joined.drop(columns=field + "_experiment", inplace=True)
        good = joined["predicted_vessel_id" + reference_suffix].eq(joined.target_vessel_id)
        better = joined.predicted_vessel_id_experiment.eq(joined.target_vessel_id)
        joined["status"] = np.select([~good & better, good & ~better, ~good & ~better],
                                      ["corrected", "regressed", "still_wrong"], default="unchanged_correct")
        joined["prediction_changed"] = joined["predicted_vessel_id" + reference_suffix].ne(joined.predicted_vessel_id_experiment)
        joined["candidate_count_delta"] = joined.candidate_count_experiment - joined["candidate_count" + reference_suffix]
        joined["model"] = model
        pieces.append(joined)
    changes = pd.concat(pieces, ignore_index=True)
    vessels = changes.groupby(["variant", "model", "target_vessel_id"], sort=True).agg(
        queries=("sample_id", "size"), corrected=("status", lambda s: int(s.eq("corrected").sum())),
        regressed=("status", lambda s: int(s.eq("regressed").sum())),
        still_wrong=("status", lambda s: int(s.eq("still_wrong").sum())),
        predictions_changed=("prediction_changed", "sum"),
        mean_candidate_delta=("candidate_count_delta", "mean")).reset_index()
    vessels["net_corrected"] = vessels.corrected - vessels.regressed
    return changes, vessels


def _check_saved_predictions(directory, originals, full_ids, runner_name="baseline"):
    path = Path(directory) / "validation_predictions.csv"
    if not path.exists():
        raise ValueError(f"{runner_name} saved validation_predictions.csv is required for original-input parity")
    saved = pd.read_csv(path, dtype={"sample_id": str, "target_vessel_id": str, "predicted_vessel_id": str})
    required = {"sample_id", "target_vessel_id", "predicted_vessel_id", "candidate_count", "target_rank", "has_hangul"}
    if not required.issubset(saved.columns) or saved.sample_id.duplicated().any():
        raise ValueError(f"Invalid {runner_name} saved validation predictions")
    if set(saved.sample_id) != set(full_ids):
        raise ValueError(f"Saved {runner_name} predictions do not cover the complete held-out fold")
    selected = saved.set_index("sample_id").loc[originals.sample_id].reset_index()
    for field in required - {"sample_id"}:
        left, right = selected[field].to_numpy(), originals[field].to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"Live original {runner_name} differs from saved predictions: {field}")
    return {"file_sha256": _hash(path), "matched_queries": len(originals),
            "saved_fold_queries": len(saved), "all_selected_original_predictions_match": True}


def _direct_comparison(predictions, metrics, variants, reference):
    """Summarize an exact-candidate comparison without changing row identities."""
    direct, vessels = paired_changes(predictions, reference=reference, experiments=("trained",))
    parity = {}
    for variant in variants:
        part = direct.loc[direct.variant.eq(variant)]
        equal = (part["candidate_ids_sha256_" + reference].eq(part.candidate_ids_sha256_experiment)
                 & part["candidate_count_" + reference].eq(part.candidate_count_experiment)
                 & part["candidate_found_" + reference].eq(part.candidate_found_experiment))
        if not bool(equal.all()):
            raise ValueError(f"Augmented experiment changed fixed {reference} candidates: {variant}")
        parity[variant] = {"queries": len(part), "all_candidate_sets_identical": True}
    prefix = "trained_vs_" + reference
    summary = {
        "reference_runner": reference, "experiment_runner": "trained",
        "metric_deltas": {variant: metric_deltas(metrics[variant][reference], metrics[variant]["trained"])
                          for variant in variants},
        "transitions": {variant: {status: int(direct.loc[direct.variant.eq(variant), "status"].eq(status).sum())
                                    for status in ("corrected", "regressed", "still_wrong", "unchanged_correct")}
                        for variant in variants},
        "candidate_parity": parity,
        "row_changes_file": prefix + "_row_changes.csv",
        "vessel_changes_file": prefix + "_vessel_changes.csv"}
    return summary, direct, vessels


def _report(summary):
    lines = ["# 후보 검색 원본·톤수 변형 비교", "",
        "공개 학습 데이터의 고정 held-out fold에 대한 개발용 검증입니다. 비공개 입력이나 실제 Public 점수를 재현하지 않습니다.", "",
        f"- fold: {summary['coverage']['fold']} / {summary['coverage']['folds']}",
        f"- 행: {summary['coverage']['queries']} / 전체 fold {summary['coverage']['heldout_queries']}",
        f"- 진단 표본 여부: {summary['coverage']['diagnostic_sample']}",
        "- 동일 변형 입력의 baseline과 각 실험을 짝지어 비교합니다. 원본과 변형 간 점수 차이를 실험 개선으로 세지 않습니다.",
        "- baseline 및 후보 교체 실험은 동일한 86개 순위 모델·훈련 사전·범주 통계를 사용합니다.",
        "- 시간은 실시간 후보 검색·특징 계산·순위 계산입니다. 인덱스/모델 로드와 고정 이름 특징 준비 시간은 summary.json startup에 분리합니다.",
        "- 각 행마다 실행 순서를 순환합니다. 운영체제/문자열 캐시와 반복 입력 효과가 있으므로 서버 실행 시간으로 해석하지 마세요.", "",
        "| 변형 | 모델 | 가중 정확도 | 한글 정확도 | 비한글 정확도 | 한글 후보 재현율 | 후보 평균 / p95 / 최대 | 실행 초 | 개선 / 신규 오답 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "—" if value is None else f"{100 * value:.4f}%"
    for variant, models in summary["metrics"].items():
        for model, item in models.items():
            counts = item["all"]["candidate_count"]
            transition = summary["transitions"].get(variant, {}).get(model, {})
            lines.append(f"| {variant} | {model} | {percent(item['weighted_top1'])} | "
                         f"{percent(item['hangul']['top1_accuracy'])} | {percent(item['non_hangul']['top1_accuracy'])} | "
                         f"{percent(item['hangul']['candidate_recall'])} | {counts['mean']:.1f} / {counts['p95']:.1f} / {counts['max']} | "
                         f"{item['all']['timing']['total_seconds']['sum']:.3f} | "
                         f"{transition.get('corrected', '—')} / {transition.get('regressed', '—')} |")
    if "trained_vs_reference" in summary:
        direct = summary["trained_vs_reference"]
        lines += ["", "## GT 결측 증강 자체의 효과", "",
            "reference는 증강 전 robust 재학습 모델이고 trained는 GT 결측 증강 모델입니다. "
            "아래 개선·신규 오답은 같은 변형의 reference 대비입니다. 두 모델의 후보 집합이 모두 동일함을 검사했습니다.", "",
            "| 변형 | 증강 전 가중 정확도 | 증강 후 가중 정확도 | 증감 (%p) | 개선 / 신규 오답 |",
            "|---|---:|---:|---:|---:|"]
        for variant, models in summary["metrics"].items():
            transition = direct["transitions"][variant]
            delta = direct["metric_deltas"][variant]["weighted_top1_pp"]
            delta_text = "—" if delta is None else f"{delta:+.4f}"
            lines.append(f"| {variant} | {percent(models['reference']['weighted_top1'])} | "
                         f"{percent(models['trained']['weighted_top1'])} | {delta_text} | "
                         f"{transition['corrected']} / {transition['regressed']} |")
        lines += ["", "증강 자체의 행별 변화: trained_vs_reference_row_changes.csv. "
                  "선박별 변화: trained_vs_reference_vessel_changes.csv. "
                  "가중 정확도·언어별 정확도·후보 수·시간 차이와 전이는 summary.json trained_vs_reference에 저장합니다."]
    if "trained_vs_control" in summary:
        direct = summary["trained_vs_control"]
        proof = summary["training_reweighting"]
        lines += ["", "## 원본 질량 재가중치 비교", "",
            f"control은 원본 질량 {proof['source_original_mass']:g}의 기존 증강 모델, "
            f"trained는 같은 캐시의 질량을 {proof['target_original_mass']:g}로 바꾼 모델입니다. "
            "훈련 후보·특징·라벨·샘플 순서와 검증 입력을 보존한 출처를 검증했습니다. "
            "아래 개선·신규 오답은 같은 입력 조건의 control 대비이며 후보 집합도 동일합니다.", "",
            "| 변형 | 기존 질량 가중 정확도 | 새 질량 가중 정확도 | 증감 (%p) | 개선 / 신규 오답 |",
            "|---|---:|---:|---:|---:|"]
        for variant, models in summary["metrics"].items():
            transition = direct["transitions"][variant]
            delta = direct["metric_deltas"][variant]["weighted_top1_pp"]
            delta_text = "—" if delta is None else f"{delta:+.4f}"
            lines.append(f"| {variant} | {percent(models['control']['weighted_top1'])} | "
                         f"{percent(models['trained']['weighted_top1'])} | {delta_text} | "
                         f"{transition['corrected']} / {transition['regressed']} |")
        lines += ["", "질량 변경의 행별 변화: trained_vs_control_row_changes.csv. "
                  "선박별 변화: trained_vs_control_vessel_changes.csv. "
                  "summary.json의 trained_vs_control은 control 대비이며 trained_vs_reference는 증강 없는 robust 대비입니다."]
    lines += ["", "이미 결측인 값을 지우거나 유효하지 않은 국제총톤수에 배수를 적용해도 입력은 바뀌지 않습니다. "
              "아래는 실제로 하나 이상의 톤수 값이 바뀐 행 수입니다.", "",
              "| 변형 | 전체 변경 / 평가 | 한글 변경 / 평가 | 비한글 변경 / 평가 |",
              "|---|---:|---:|---:|"]
    for variant, groups in summary["variant_effects"].items():
        cells = [f"{groups[group]['changed']} / {groups[group]['queries']}"
                 for group in ("all", "hangul", "non_hangul")]
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines += ["", "행별 예측/시간: predictions.csv. 같은 변형의 행별 변화: row_changes.csv. "
              "선박별 개선·회귀: vessel_changes.csv. 모든 설정·해시·세부 언어별 지표: summary.json.", ""]
    return "\n".join(lines)


def compare_retrieval(baseline_directory, output, *, train_path="work/data/train.csv",
                      catalog_path="work/data/vessel_catalog.csv", profiles=PROFILES,
                      experiment_model=None, reference_model=None, control_model=None,
                      variants=VARIANTS, max_vessels_per_language=None,
                      max_queries=None, one_row_per_vessel=False, seed=20261022, progress_every=100):
    from .robust_retrieval import RobustCandidateRetriever, RobustRetrievalConfig, retrieval_spec

    profiles, variants = tuple(profiles), tuple(variants)
    if not profiles or len(profiles) != len(set(profiles)) or set(profiles) - set(PROFILES):
        raise ValueError("Choose unique supported retrieval profiles")
    if (not variants or "original" not in variants or len(variants) != len(set(variants))
            or set(variants) - set(VARIANTS)):
        raise ValueError("Choose unique supported variants, always including original for parity")
    if type(progress_every) is not int or progress_every < 1:
        raise ValueError("progress-every must be positive")
    if reference_model is not None and experiment_model is None:
        raise ValueError("reference-model requires an augmented experiment-model")
    if control_model is not None and (reference_model is None or experiment_model is None):
        raise ValueError("control-model requires both unaugmented reference-model and reweighted experiment-model")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new or empty output directory: {output}")
    started = time.perf_counter()
    train_hash, catalog_hash = _hash(train_path), _hash(catalog_path)
    train, catalog = read_queries(train_path, labeled=True), read_catalog(catalog_path)
    baseline, baseline_summary, baseline_startup = _load_model(baseline_directory, train_hash, catalog_hash)
    if baseline_summary["dataset"].get("retrieval_extension") is not None:
        raise ValueError("Baseline must be the legacy 86-feature retrieval model")
    if baseline_summary["dataset"].get("training_augmentation") is not None:
        raise ValueError("Baseline must retain original unaugmented training")
    full = _validate_partition(baseline, baseline_summary, train, catalog)
    selected = select_queries(full, max_vessels_per_language=max_vessels_per_language,
                              max_queries=max_queries, one_row_per_vessel=one_row_per_vessel, seed=seed)
    if selected.empty:
        raise ValueError("No held-out rows selected")
    runners, startup, specs = {"baseline": baseline}, {"baseline": baseline_startup}, {}
    startup["baseline"]["name_feature_index_seconds"] = _initialize_name_features(baseline)
    for profile in profiles:
        config = RobustRetrievalConfig(**baseline_summary["dataset"]["retrieval"], profile=profile)
        build_started = time.perf_counter()
        matcher = copy.copy(baseline)
        matcher.pipeline = copy.copy(baseline.pipeline)
        matcher.pipeline.retriever = RobustCandidateRetriever(catalog, baseline.pipeline.memory, config)
        startup[profile] = {"retrieval_index_build_seconds": time.perf_counter() - build_started,
                            "shared_baseline_ranker_memory_categories": True,
                            "name_feature_index_seconds": 0.0}
        specs[profile] = retrieval_spec(config)
        runners[profile] = matcher
    saved_models = {"baseline": baseline_directory}
    augmentation = None
    reweighting = None
    if experiment_model is not None:
        model, summary, load = _load_model(experiment_model, train_hash, catalog_hash)
        _validate_partition(model, summary, train, catalog)
        _validate_model_pair(baseline, baseline_summary, model, summary)
        augmentation = summary["dataset"].get("training_augmentation")
        reweighting = summary["dataset"].get("training_reweighting")
        if augmentation is not None and reference_model is None:
            raise ValueError("Augmented experiment-model requires an unaugmented robust reference-model")
        if reweighting is not None and control_model is None:
            raise ValueError("Reweighted experiment-model requires its original augmented control-model")
        if reference_model is not None:
            reference, reference_summary, reference_load = _load_model(reference_model, train_hash, catalog_hash)
            _validate_partition(reference, reference_summary, train, catalog)
            _validate_model_pair(baseline, baseline_summary, reference, reference_summary)
            _validate_reference_pair(reference, reference_summary, model, summary)
            reference_load["name_feature_index_seconds"] = _initialize_name_features(reference)
            runners["reference"] = reference
            startup["reference"] = reference_load
            specs["reference"] = reference_summary["dataset"]["retrieval_extension"]
            saved_models["reference"] = reference_model
        if control_model is not None:
            control, control_summary, control_load = _load_model(control_model, train_hash, catalog_hash)
            _validate_partition(control, control_summary, train, catalog)
            _validate_model_pair(baseline, baseline_summary, control, control_summary)
            _validate_control_pair(control, control_summary, model, summary)
            control_load["name_feature_index_seconds"] = _initialize_name_features(control)
            runners["control"] = control
            startup["control"] = control_load
            specs["control"] = control_summary["dataset"]["retrieval_extension"]
            saved_models["control"] = control_model
        load["name_feature_index_seconds"] = _initialize_name_features(model)
        runners["trained"] = model
        startup["trained"] = load
        specs["trained"] = summary["dataset"]["retrieval_extension"]
        saved_models["trained"] = experiment_model
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1, "complete": False,
        "coverage": {"fold": baseline_summary["dataset"]["outer_fold"],
            "folds": baseline_summary["dataset"]["outer_folds"], "split": baseline_summary["dataset"]["outer_split"],
            "split_seed": baseline_summary["dataset"]["seed"], "heldout_queries": len(full),
            "queries": len(selected), "vessels": int(selected.target_vessel_id.nunique()),
            "diagnostic_sample": len(selected) != len(full), "selection_seed": seed,
            "max_vessels_per_language": max_vessels_per_language, "max_queries": max_queries,
            "one_row_per_language_vessel": one_row_per_vessel},
        "provenance": {"train_sha256": train_hash, "catalog_sha256": catalog_hash,
            "comparison_code_sha256": _hash(__file__), "feature_transform": baseline.feature_transform,
            "feature_names": baseline.ranker.feature_names, "ranker": asdict(baseline.ranker.config)},
        "profiles": specs, "variants": list(variants), "startup": startup,
        "variant_effects": variant_effect_counts(selected, variants),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "notes": ["Public held-out development validation, not hidden-test reconstruction.",
            "Corruptions change only the selected query's two input tonnages; labels never enter inference.",
            "No cache reuse for candidate/features/scores. Each row rotates runner order; string/OS caches remain uncontrolled.",
            "Feature seconds include candidate validation and matrix selection; rank seconds include stable sorting.",
            "Startup/load/build and report IO are excluded from row inference time.",
            "Fixed-model retrieval swaps test immediate effects; a trained model also changes negative sampling/training pairs."]}
    if augmentation is not None:
        metadata["training_augmentation"] = augmentation
        metadata["notes"].append("The trained-vs-reference comparison isolates declared training augmentation; retrieval, features, splits, learned lookup assets and ranker settings are identical.")
    if reweighting is not None:
        metadata["training_reweighting"] = reweighting
        metadata["control_training_augmentation"] = control_summary["dataset"]["training_augmentation"]
        metadata["notes"].append("The trained-vs-control comparison changes only the declared original query mass on the exact reused training cache; candidate pairs, feature values, labels and validation inputs are preserved.")
    _write_json(output / "summary.json", metadata)
    records, names = [], list(runners)
    queries = selected.to_dict("records")
    for variant_index, variant in enumerate(variants):
        for query_index, row in enumerate(queries):
            query = variant_query(row, variant)
            rotation = (query_index + variant_index) % len(names)
            for name in names[rotation:] + names[:rotation]:
                record = timed_prediction(runners[name], query, row["target_vessel_id"])
                record.update({"variant": variant, "model": name, "sample_id": str(row["sample_id"]),
                    "target_vessel_id": str(row["target_vessel_id"]), "vessel_name": row["vessel_name"],
                    "has_hangul": has_hangul(row["vessel_name"]), "runner_order": names.index(name) - rotation})
                record["runner_order"] %= len(names)
                records.append(record)
            if (query_index + 1) % progress_every == 0 or query_index + 1 == len(queries):
                print(f"{variant}: {query_index + 1:,}/{len(queries):,} paired queries", flush=True)
    predictions = pd.DataFrame(records)
    for name, directory in saved_models.items():
        originals = predictions.loc[predictions.model.eq(name) & predictions.variant.eq("original")]
        metadata[name + "_parity"] = _check_saved_predictions(directory, originals, full.sample_id, name)
    metadata["metrics"] = {variant: {name: summarize_predictions(
        predictions.loc[predictions.variant.eq(variant) & predictions.model.eq(name)]) for name in names}
        for variant in variants}
    metadata["paired_metric_deltas"] = {variant: {name: metric_deltas(
        metadata["metrics"][variant]["baseline"], metadata["metrics"][variant][name])
        for name in names if name != "baseline"} for variant in variants}
    changes, vessels = paired_changes(predictions)
    metadata["transitions"] = {variant: {name: {
        status: int(changes.loc[changes.variant.eq(variant) & changes.model.eq(name), "status"].eq(status).sum())
        for status in ("corrected", "regressed", "still_wrong", "unchanged_correct")}
        for name in names if name != "baseline"} for variant in variants}
    for name, directory in (("reference", reference_model), ("control", control_model)):
        if directory is None:
            continue
        direct_summary, direct, direct_vessels = _direct_comparison(predictions, metadata["metrics"], variants, name)
        metadata["trained_vs_" + name] = direct_summary
        direct.to_csv(output / direct_summary["row_changes_file"], index=False, encoding="utf-8-sig")
        direct_vessels.to_csv(output / direct_summary["vessel_changes_file"], index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(output / "row_changes.csv", index=False, encoding="utf-8-sig")
    vessels.to_csv(output / "vessel_changes.csv", index=False, encoding="utf-8-sig")
    metadata["complete"] = True
    metadata["wall_seconds"] = time.perf_counter() - started
    _write_json(output / "summary.json", metadata)
    (output / "report.md").write_text(_report(metadata), encoding="utf-8")
    return metadata


def add_retrieval_comparison_parser(subparsers):
    parser = subparsers.add_parser("compare-retrieval", help="Live paired 86-feature retrieval comparison on public held-out input variants")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--profiles", "--profile", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--experiment-model", help="Optional retrained robust 86-feature model from the same fold/settings")
    parser.add_argument("--reference-model", help="Required unaugmented robust-trained model when experiment-model uses GT augmentation")
    parser.add_argument("--control-model", help="Original augmented model whose exact cache was reused by a mass-reweighted experiment")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--train", default="work/data/train.csv")
    parser.add_argument("--catalog", default="work/data/vessel_catalog.csv")
    parser.add_argument("--output", required=True)
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument("--max-vessels-per-language", type=int)
    sample.add_argument("--max-queries", type=int)
    parser.add_argument("--one-row-per-vessel", action="store_true",
                        help="With max-vessels-per-language, keep one deterministic row per language/vessel")
    parser.add_argument("--seed", "--selection-seed", type=int, default=20261022,
                        help="Selection seed only; outer split is always read from the saved model")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def run_comparison(args):
    result = compare_retrieval(args.baseline, args.output, train_path=args.train, catalog_path=args.catalog,
        profiles=args.profiles, experiment_model=args.experiment_model, reference_model=args.reference_model,
        control_model=getattr(args, "control_model", None), variants=args.variants,
        max_vessels_per_language=args.max_vessels_per_language, max_queries=args.max_queries,
        one_row_per_vessel=args.one_row_per_vessel, seed=args.seed, progress_every=args.progress_every)
    print(f"Saved paired retrieval report: {Path(args.output).resolve()}", flush=True)
    print(json.dumps(result["coverage"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_retrieval_comparison_parser(subparsers)
    run_comparison(parser.parse_args())
