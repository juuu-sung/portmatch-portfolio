"""Paired, read-only evaluation of controlled feature experiments.

Predictions are matched by sample identity, never CSV order. The report rejects
changed data, candidates, splits, sampling or ranker settings: this command is
intentionally for controlled feature selection and verified cache derivations,
not arbitrary model comparison.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ranking import ranking_metrics
from .feature_selection import FeatureSelection


_REQUIRED = {
    "sample_id", "has_hangul", "target_vessel_id", "predicted_vessel_id",
    "candidate_found", "target_rank", "candidate_count",
}
_GLOBAL_DATA_FIELDS = (
    "schema_version", "outer_split", "outer_folds", "inner_split", "inner_folds",
    "seed", "feature_names", "retrieval", "sampling", "max_train_queries",
    "max_valid_queries", "sampled", "validation_sampling", "catalog_rows", "provenance",
    "feature_transform", "training_augmentation",
)
_REQUIRED_DATA_FIELDS = {
    "outer_split", "outer_folds", "outer_fold", "inner_split", "inner_folds",
    "seed", "feature_names", "retrieval", "sampling", "counts", "provenance", "sampled",
}
_STATUSES = ("corrected", "regressed", "still_wrong", "unchanged_correct")
_WARNING = (
    "All observed folds are development validation, not independent confirmation. "
    "Scores are pooled by language, not averaged across fold scores. "
    "Feature exclusion retains the original candidate generation and negative sampling."
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _boolean(series, name):
    normalized = series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if normalized.isna().any():
        raise ValueError(f"{name} must contain only explicit booleans")
    return normalized.astype(bool)


def _feature_names(summary, directory):
    source = summary["dataset"]["feature_names"]
    if "feature_selection" in summary:
        selection = FeatureSelection.from_dict(summary["feature_selection"])
    else:
        selection = FeatureSelection.from_names(source, summary.get("trained_feature_names", source))
    names = list(selection.selected_features)
    if list(selection.source_feature_names) != source:
        raise ValueError(f"Selected feature source differs from dataset: {directory}")
    if "trained_feature_names" in summary and summary["trained_feature_names"] != names:
        raise ValueError(f"Conflicting trained feature schemas: {directory}")
    ranker_path = directory / "ranker.json"
    if ranker_path.exists() and _json(ranker_path).get("feature_names") != names:
        raise ValueError(f"Ranker feature schema differs from summary: {directory}")
    return names


def _load_run(directory):
    directory = Path(directory).resolve()
    summary_path = directory / "training_summary.json"
    summary = _json(summary_path)
    dataset = summary.get("dataset", {})
    if summary.get("complete") is not True or dataset.get("complete") is not True:
        raise ValueError(f"Incomplete model or dataset: {directory}")
    from .feature_transforms import check_transform_metadata
    check_transform_metadata(dataset)
    missing = _REQUIRED_DATA_FIELDS - dataset.keys()
    if missing or not isinstance(summary.get("ranker"), dict) or not summary["ranker"]:
        raise ValueError(f"Missing comparison metadata {sorted(missing)}: {directory}")
    fold, folds = dataset["outer_fold"], dataset["outer_folds"]
    if (type(fold) is not int or type(folds) is not int or folds < 2
            or not 0 <= fold < folds):
        raise ValueError(f"Invalid outer fold metadata: {directory}")
    prediction_path = directory / "validation_predictions.csv"
    frame = pd.read_csv(prediction_path, keep_default_na=False, dtype={
        "sample_id": str, "target_vessel_id": str, "predicted_vessel_id": str})
    if not _REQUIRED.issubset(frame.columns) or frame.empty:
        raise ValueError(f"Missing prediction columns or empty predictions: {directory}")
    for name in ("sample_id", "target_vessel_id", "predicted_vessel_id"):
        if frame[name].str.strip().eq("").any():
            raise ValueError(f"Empty {name}: {directory}")
    if frame.sample_id.duplicated().any():
        raise ValueError(f"Duplicate sample_id within a fold: {directory}")
    for name in ("has_hangul", "candidate_found"):
        frame[name] = _boolean(frame[name], name)
    for name in ("target_rank", "candidate_count"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    ranking_metrics(frame)  # Checks integer bounds and retrieval/rank consistency.
    if (frame.candidate_count < 1).any():
        raise ValueError(f"Prediction has an empty candidate set: {directory}")
    if not np.array_equal(frame.target_rank.eq(1), frame.predicted_vessel_id.eq(frame.target_vessel_id)):
        raise ValueError(f"Predicted identity and target rank disagree: {directory}")
    counts = dataset["counts"]
    if counts.get("valid_groups") != len(frame):
        raise ValueError(f"Validation row count differs from cache metadata: {directory}")
    if counts.get("valid_pairs") != int(frame.candidate_count.sum()):
        raise ValueError(f"Candidate count differs from cache metadata: {directory}")
    if dataset["sampled"] is False and counts.get("outer_validation_rows") != len(frame):
        raise ValueError(f"Unsampled validation does not contain its full fold: {directory}")
    names = _feature_names(summary, directory)
    frame["fold"] = fold
    return {"directory": directory, "summary": summary, "dataset": dataset,
            "features": names, "predictions": frame,
            "summary_sha256": _hash(summary_path), "predictions_sha256": _hash(prediction_path)}


def _load_side(directories, name):
    if not directories:
        raise ValueError(f"At least one {name} directory is required")
    runs = {}
    for directory in directories:
        run = _load_run(directory)
        fold = run["dataset"]["outer_fold"]
        if fold in runs:
            raise ValueError(f"Duplicate {name} outer_fold {fold}")
        runs[fold] = run
    reference = next(iter(runs.values()))
    for run in runs.values():
        for field in _GLOBAL_DATA_FIELDS:
            if run["dataset"].get(field) != reference["dataset"].get(field):
                raise ValueError(f"Inconsistent {name} dataset field across folds: {field}")
        if run["summary"]["ranker"] != reference["summary"]["ranker"]:
            raise ValueError(f"Inconsistent {name} ranker settings across folds")
        if run["features"] != reference["features"]:
            raise ValueError(f"Inconsistent {name} selected features across folds")
    all_rows = pd.concat([run["predictions"] for run in runs.values()], ignore_index=True)
    if all_rows.sample_id.duplicated().any():
        raise ValueError(f"Duplicate {name} sample_id across folds")
    if reference["dataset"]["outer_split"] == "vessel":
        if all_rows.groupby("target_vessel_id").fold.nunique().gt(1).any():
            raise ValueError(f"A {name} target vessel occurs in multiple vessel folds")
    return runs


def _verify_run_derivation(run, parent_hash, parent_dataset=None):
    """Bind the saved model to the exact derived cache before trusting lineage."""
    from .feature_transforms import verify_derivation

    summary = run["summary"]
    directory = summary.get("cache_directory")
    expected_hash = summary.get("cache_metadata_sha256")
    if not directory or not expected_hash or not parent_hash:
        raise ValueError("Derived-cache comparison requires cache paths and metadata hashes")
    path = Path(directory) / "metadata.json"
    if _hash(path) != expected_hash or _json(path) != run["dataset"]:
        raise ValueError("Derived cache metadata differs from the trained model")
    return verify_derivation(directory, parent_hash, parent_dataset)


def _align_pair(baseline, experiment):
    btransform = baseline["dataset"].get("feature_transform")
    etransform = experiment["dataset"].get("feature_transform")
    derived_pair = btransform != etransform
    if derived_pair:
        from .feature_transforms import check_parent_transform

        check_parent_transform(btransform, etransform)
    for field in (*_GLOBAL_DATA_FIELDS, "outer_fold", "counts", "train_shards", "valid_shards",
                  "training_missing_target_policy", "training_augmentation_diagnostics", "training_augmentation_assets"):
        if derived_pair and field in {"feature_names", "feature_transform"}:
            continue
        if baseline["dataset"].get(field) != experiment["dataset"].get(field):
            raise ValueError(f"Controlled ablation requires identical dataset {field}")
    if baseline["summary"]["ranker"] != experiment["summary"]["ranker"]:
        raise ValueError("Controlled ablation requires identical ranker settings")
    verification = {"kind": "same_cache_metadata_and_prediction_alignment"}
    if derived_pair:
        from .feature_transforms import check_selected_features

        kind = check_selected_features(etransform, baseline["features"], experiment["features"])
        proof = _verify_run_derivation(experiment,
                                       baseline["summary"].get("cache_metadata_sha256"),
                                       baseline["dataset"])
        verification = {"kind": kind, "proof": proof}
    elif etransform is not None:
        # A later feature-selection experiment may reuse a derived cache too.
        proofs = [_verify_run_derivation(run, run["dataset"].get("derivation", {}).get("parent_metadata_sha256"))
                  for run in (baseline, experiment)]
        verification = {"kind": "verified_same_derived_cache", "proofs": proofs}
    for field in (() if derived_pair else ("cache_metadata_sha256",)):
        left, right = baseline["summary"].get(field), experiment["summary"].get(field)
        if left is not None and right is not None and left != right:
            raise ValueError(f"Controlled ablation requires identical {field}")
    left, right = baseline["predictions"], experiment["predictions"]
    if set(left.sample_id) != set(right.sample_id):
        raise ValueError("Baseline and experiment sample_id sets differ; no rows may be silently dropped")
    left = left.sort_values("sample_id", kind="stable").reset_index(drop=True)
    right = right.set_index("sample_id").loc[left.sample_id].reset_index()
    for column in ("target_vessel_id", "has_hangul", "candidate_found", "candidate_count", "fold"):
        if not np.array_equal(left[column], right[column]):
            raise ValueError(f"Baseline and experiment {column} differ")
    rows = left[["sample_id", "fold", "target_vessel_id", "has_hangul", "candidate_found", "candidate_count"]].copy()
    for side, values in (("baseline", left), ("experiment", right)):
        for column in ("predicted_vessel_id", "target_rank"):
            rows[f"{side}_{column}"] = values[column].to_numpy()
        rows[f"{side}_correct"] = values.target_rank.eq(1).to_numpy()
    rows["prediction_changed"] = rows.baseline_predicted_vessel_id.ne(rows.experiment_predicted_vessel_id)
    rows["status"] = np.select([
        ~rows.baseline_correct & rows.experiment_correct,
        rows.baseline_correct & ~rows.experiment_correct,
        ~rows.baseline_correct & ~rows.experiment_correct,
    ], _STATUSES[:3], default="unchanged_correct")
    return left, right, rows, verification


def _vessels(rows):
    result = rows.groupby("target_vessel_id", sort=True).agg(
        queries=("sample_id", "size"), baseline_correct=("baseline_correct", "sum"),
        experiment_correct=("experiment_correct", "sum"))
    for status in _STATUSES:
        result[status] = rows.loc[rows.status.eq(status)].groupby("target_vessel_id").size().reindex(result.index, fill_value=0)
    result["net_correct_delta"] = result.experiment_correct - result.baseline_correct
    result["outcome"] = np.select([
        result.corrected.gt(0) & result.regressed.gt(0),
        result.corrected.gt(0), result.regressed.gt(0),
    ], ["mixed", "improved", "regressed"], default="unchanged")
    return result.reset_index()


def _transitions(rows):
    vessel = _vessels(rows)
    counts = {status: int(rows.status.eq(status).sum()) for status in _STATUSES}
    counts["prediction_changed"] = int(rows.prediction_changed.sum())
    counts["net_correct_delta"] = counts["corrected"] - counts["regressed"]
    counts["vessels"] = {name: int(vessel.outcome.eq(name).sum())
                         for name in ("improved", "regressed", "mixed", "unchanged")}
    counts["vessels"].update({"total": len(vessel),
                              "net_improved": int(vessel.net_correct_delta.gt(0).sum()),
                              "net_regressed": int(vessel.net_correct_delta.lt(0).sum()),
                              "net_unchanged": int(vessel.net_correct_delta.eq(0).sum())})
    return counts


def _comparison(left, right, rows):
    baseline, experiment = ranking_metrics(left), ranking_metrics(right)
    delta = {}
    for language in ("all", "hangul", "non_hangul"):
        delta[language] = {field: (None if value is None or experiment[language][field] is None
                                   else 100 * (experiment[language][field] - value))
                           for field, value in baseline[language].items()
                           if field == "top1_accuracy" or field == "candidate_recall" or field.startswith("recall_at_")}
    for field in ("weighted_top1", "weighted_candidate_recall"):
        delta[field] = (None if baseline[field] is None or experiment[field] is None
                        else 100 * (experiment[field] - baseline[field]))
    changes = {"all": _transitions(rows), "hangul": _transitions(rows[rows.has_hangul]),
               "non_hangul": _transitions(rows[~rows.has_hangul])}
    return {"baseline": baseline, "experiment": experiment,
            "delta_percentage_points": delta, "transitions": changes}


def _percent(value):
    return "N/A" if value is None else f"{100 * value:.4f}%"


def _points(value):
    return "N/A" if value is None else f"{value:+.4f}%p"


def _report(summary):
    coverage, pooled = summary["coverage"], summary["pooled"]
    derived = summary["comparison_type"] == "verified_derived_feature_experiment"
    lines = ["# 특징 실험 비교", "",
             ("기준 모델을 자기 자신과 비교한 리포트 동작 확인입니다. 새 실험 학습 결과가 아닙니다."
              if summary["self_comparison"] else "기준 모델과 특징 실험의 저장 예측을 비교합니다."), "",
             f"검증 범위: {'전체 OOF' if coverage['full_oof'] else '부분 검증'} — "
             f"fold {', '.join(map(str, coverage['included_folds']))} / "
             f"전체 {coverage['outer_folds']}개 fold, {coverage['queries']:,}행.", "",
             f"포함하지 않은 fold: {', '.join(map(str, coverage['missing_folds'])) or '(없음)'}. "
             f"캐시 생성 시 질의 수 제한 여부: {'있음' if coverage['sampled'] else '없음'}.", "",
             "전체 합산 점수는 언어별 전체 정답 수로 계산합니다. fold 점수의 단순 평균이 아닙니다.", "",
             "| 구분 | 행 | 기준 Top-1 | 실험 Top-1 | 차이 | 수정 | 신규 오답 | 순개선 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for lang, label in (("hangul", "한글"), ("non_hangul", "비한글"), ("all", "전체")):
        b, e, change = pooled["baseline"][lang], pooled["experiment"][lang], pooled["transitions"][lang]
        lines.append(f"| {label} | {b['queries']} | {_percent(b['top1_accuracy'])} | "
                     f"{_percent(e['top1_accuracy'])} | {_points(pooled['delta_percentage_points'][lang]['top1_accuracy'])} | "
                     f"{change['corrected']} | {change['regressed']} | {change['net_correct_delta']:+d} |")
    lines.extend(["", f"공식 가중 Top-1: **{_percent(pooled['baseline']['weighted_top1'])} → "
                  f"{_percent(pooled['experiment']['weighted_top1'])} "
                  f"({_points(pooled['delta_percentage_points']['weighted_top1'])})**.", "",
                  "| fold | 행 | 기준 가중 Top-1 | 실험 가중 Top-1 | 차이 | 한글 수정/신규 오답 | 비한글 수정/신규 오답 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for item in summary["per_fold"]:
        h, n = item["transitions"]["hangul"], item["transitions"]["non_hangul"]
        lines.append(f"| {item['fold']} | {item['baseline']['all']['queries']} | "
                     f"{_percent(item['baseline']['weighted_top1'])} | {_percent(item['experiment']['weighted_top1'])} | "
                     f"{_points(item['delta_percentage_points']['weighted_top1'])} | "
                     f"{h['corrected']}/{h['regressed']} | {n['corrected']}/{n['regressed']} |")
    lines.extend(["", "## 선박 단위 변화", "",
                  "개선은 수정만 있는 선박, 악화는 신규 오답만 있는 선박, 혼합은 둘 다 있는 선박입니다. "
                  "순개선/순악화는 해당 선박의 정답 행 수 차이로 따로 집계합니다.", "",
                  "| 구분 | 개선 | 악화 | 혼합 | 변화 없음 | 순개선 | 순악화 |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for lang, label in (("hangul", "한글"), ("non_hangul", "비한글"), ("all", "전체")):
        v = pooled["transitions"][lang]["vessels"]
        lines.append(f"| {label} | {v['improved']} | {v['regressed']} | {v['mixed']} | {v['unchanged']} | "
                     f"{v['net_improved']} | {v['net_regressed']} |")
    features = summary["features"]
    lines.extend(["", "## 비교 조건", "",
                  f"- 기준 특징 {len(features['baseline'])}개, 실험 특징 {len(features['experiment'])}개",
                  f"- 기준 대비 제외: {', '.join(features['removed']) or '(없음)'}",
                  f"- 기준 대비 추가: {', '.join(features['added']) or '(없음)'}",
                  ("- 같은 외부 분할·데이터 출처·샘플링·학습 설정을 확인하고, 선언된 특징 변환의 출처를 검증했습니다."
                   if derived else "- 같은 외부 분할·데이터 출처·원본 특징 스키마·샘플링·후보 수/정답 포함 여부·학습 설정을 확인했습니다."),
                  ("- 변환에서 훈련/검증의 후보 ID·그룹·정답과 나머지 특징값이 유지됐다는 기록을 확인하고, "
                   "원본 및 변환 캐시의 모든 관련 파일 해시를 다시 검사했습니다. "
                   "버전별 변환만 적용하며 기존 후보·훈련 오답 샘플은 유지합니다. "
                   "기준 모델에는 과거 학습 당시의 각 캐시 파일 해시가 없으므로, 기준 모델의 저장 메타데이터와 "
                   "현재 원본→변환 캐시의 일치까지 검증합니다. 과거 파일 바이트까지 증명하거나 기준 모델을 재평가하지는 않습니다."
                   if derived else "- 후보 구성 동일성은 저장된 캐시 메타데이터와 질의별 후보 수/정답 포함 여부를 근거로 확인합니다. "
                   "원본 캐시 파일 전체를 다시 해시하거나 후보 ID 목록을 재생성하지 않습니다."),
                  "- 특징 제거는 기존 후보 생성과 어려운 오답 샘플링을 유지합니다. 범주 특징을 빼도 샘플링에 사용된 범주 정보까지 제거하지는 않습니다.",
                  "- 모든 관측 fold는 개발용 검증입니다. 이 결과는 독립 평가나 대회 리더보드 점수가 아닙니다.",
                  "- 동일 선박의 반복 기록을 독립 선박 개선으로 세지 않습니다. 기존 오답뿐 아니라 전체 평가 행의 신규 오답을 포함합니다.",
                  "- 단일 언어만 존재하면 가중 점수는 N/A입니다. 빠진 언어를 0점이나 100점으로 채우지 않습니다.", "",
                  "## 출력 파일", "",
                  "- `summary.json`: 비교 조건, 원본 파일 해시, pooled·fold·언어별 지표와 수정/퇴행 집계",
                  "- `row_changes.csv`: 모든 평가 행의 기준/실험 예측·정답 순위·변화 상태",
                  "- `vessel_changes.csv`: 전체 및 언어별 선박 단위 수정·퇴행·순개선",
                  "- `per_fold_metrics.csv`: fold·언어별 지표, 후보 재현율, 수정/퇴행 수", ""])
    return "\n".join(lines)


def compare_rankers(baseline_dirs, experiment_dirs, output):
    """Write a paired report without retraining, rescoring, or modifying sources."""
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new or empty report directory: {output}")
    baseline, experiment = _load_side(baseline_dirs, "baseline"), _load_side(experiment_dirs, "experiment")
    if baseline.keys() != experiment.keys():
        raise ValueError("Baseline and experiment must contain the same outer folds")
    per_fold, pairs, sources = [], [], []
    for fold in sorted(baseline):
        left, right, rows, verification = _align_pair(baseline[fold], experiment[fold])
        per_fold.append({"fold": fold, **_comparison(left, right, rows)})
        pairs.append((left, right, rows))
        item = {"fold": fold, "cache_verification": verification}
        for side, run in (("baseline", baseline[fold]), ("experiment", experiment[fold])):
            item[side] = {"directory": str(run["directory"]),
                          "training_summary_sha256": run["summary_sha256"],
                          "validation_predictions_sha256": run["predictions_sha256"],
                          "cache_directory": run["summary"].get("cache_directory"),
                          "cache_metadata_sha256": run["summary"].get("cache_metadata_sha256"),
                          "train_load_and_fit_seconds": run["summary"].get("train_load_and_fit_seconds"),
                          "total_seconds": run["summary"].get("total_seconds")}
        sources.append(item)
    left, right, rows = [pd.concat([pair[index] for pair in pairs], ignore_index=True) for index in range(3)]
    first = baseline[min(baseline)]
    dataset = first["dataset"]
    included = sorted(baseline)
    full_folds = included == list(range(dataset["outer_folds"]))
    population = {run["dataset"]["counts"].get("outer_training_rows", 0)
                  + run["dataset"]["counts"].get("outer_validation_rows", 0)
                  for run in baseline.values()}
    if len(population) != 1:
        raise ValueError("Folds disagree about the source training population size")
    full_oof = full_folds and dataset["sampled"] is False
    expected_queries = population.pop()
    if full_oof and len(rows) != expected_queries:
        raise ValueError("Full OOF predictions do not cover the source training population")
    efirst = experiment[min(experiment)]
    bnames, enames = first["features"], efirst["features"]
    derived = any(item["cache_verification"]["kind"] != "same_cache_metadata_and_prediction_alignment"
                  for item in sources)
    summary = {"schema_version": 2 if derived else 1, "complete": True,
               "comparison_type": "verified_derived_feature_experiment" if derived else "cached_feature_ablation",
               "self_comparison": all(baseline[fold]["directory"] == experiment[fold]["directory"] for fold in included),
               "coverage": {"included_folds": included, "outer_folds": dataset["outer_folds"],
                            "missing_folds": [fold for fold in range(dataset["outer_folds"]) if fold not in baseline],
                            "all_folds_present": full_folds, "sampled": dataset["sampled"],
                            "full_oof": full_oof, "queries": len(rows),
                            "source_training_rows": expected_queries},
               "features": {"baseline": bnames, "experiment": enames,
                            "removed": [n for n in bnames if n not in enames],
                            "added": [n for n in enames if n not in bnames]},
               "feature_selection": {
                   "baseline": first["summary"].get("feature_selection", FeatureSelection.from_names(dataset["feature_names"], bnames).to_dict()),
                   "experiment": efirst["summary"].get("feature_selection", FeatureSelection.from_names(efirst["dataset"]["feature_names"], enames).to_dict())},
               "configuration": {"dataset": {key: dataset.get(key) for key in _GLOBAL_DATA_FIELDS},
                                 "experiment_dataset": {key: efirst["dataset"].get(key) for key in _GLOBAL_DATA_FIELDS},
                                 "ranker": first["summary"]["ranker"]},
               "pooled": _comparison(left, right, rows), "per_fold": per_fold,
               "sources": sources, "interpretation": _WARNING}
    vessel_parts = []
    for language, part in (("all", rows), ("hangul", rows[rows.has_hangul]), ("non_hangul", rows[~rows.has_hangul])):
        vessels = _vessels(part)
        vessels.insert(0, "language", language)
        vessel_parts.append(vessels)
    metrics_rows = []
    for comparison in per_fold:
        for language in ("all", "hangul", "non_hangul"):
            b, e = comparison["baseline"][language], comparison["experiment"][language]
            changes = comparison["transitions"][language]
            item = {"fold": comparison["fold"], "language": language, "queries": b["queries"]}
            for field in ("correct_top1", "top1_accuracy", "candidate_recall", "recall_at_5"):
                item["baseline_" + field], item["experiment_" + field] = b[field], e[field]
            item["top1_delta_percentage_points"] = comparison["delta_percentage_points"][language]["top1_accuracy"]
            item.update({key: changes[key] for key in (*_STATUSES, "net_correct_delta")})
            metrics_rows.append(item)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "row_changes.csv", index=False, encoding="utf-8-sig")
    pd.concat(vessel_parts, ignore_index=True).to_csv(output / "vessel_changes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics_rows).to_csv(output / "per_fold_metrics.csv", index=False, encoding="utf-8-sig")
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def add_comparison_parser(subparsers):
    parser = subparsers.add_parser("compare-rankers", help="Compare aligned feature selections or verified derived-cache experiments")
    parser.add_argument("--baseline", required=True, nargs="+", help="Baseline model directories, one per included fold")
    parser.add_argument("--experiment", required=True, nargs="+", help="Experiment model directories for the same folds")
    parser.add_argument("--output", required=True, help="A new or empty comparison report directory")


def run_compare(args):
    summary = compare_rankers(args.baseline, args.experiment, args.output)
    pooled = summary["pooled"]
    print(f"Compared {summary['coverage']['queries']:,} queries across folds {summary['coverage']['included_folds']}", flush=True)
    print(f"Weighted Top-1: {_percent(pooled['baseline']['weighted_top1'])} -> "
          f"{_percent(pooled['experiment']['weighted_top1'])} "
          f"({_points(pooled['delta_percentage_points']['weighted_top1'])})", flush=True)
    print(f"Saved comparison: {Path(args.output).resolve()}", flush=True)
