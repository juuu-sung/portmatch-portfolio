"""Paired reports for fixed-ranker, training-only index-size diagnostics."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .retrieval_comparison import (
    TIMING_COLUMNS, VARIANTS, metric_deltas, paired_changes, summarize_predictions,
)


REFERENCE = "index_80"
EXPERIMENT = "index_100"
STATUSES = ("corrected", "regressed", "still_wrong", "unchanged_correct")
DIAGNOSTIC_FEATURES = (
    "retrieval_name_score", "retrieval_name_rank", "retrieval_phonetic_score",
    "retrieval_phonetic_rank", "retrieval_lookup_support", "retrieval_lookup_purity",
    "flag_probability", "flag_lift_log", "type_probability", "type_lift_log",
    "retrieval_channel_count",
)


def _validate_records(records: pd.DataFrame, variants: tuple[str, ...], coverage: dict) -> pd.DataFrame:
    required = {"model", "variant", "sample_id", "target_vessel_id", "vessel_name",
                "has_hangul", "predicted_vessel_id", "candidate_found", "target_rank",
                "candidate_count", "top_score", "score_margin", "candidate_ids_sha256",
                *TIMING_COLUMNS}
    if (not isinstance(records, pd.DataFrame) or records.empty
            or not records.columns.is_unique or not required.issubset(records.columns)):
        raise ValueError("Nonempty index-scale records with unique required columns are needed")
    if (not variants or len(set(variants)) != len(variants)
            or set(variants) - set(VARIANTS)):
        raise ValueError("Choose unique supported input variants")
    if set(records.model) != {REFERENCE, EXPERIMENT} or set(records.variant) != set(variants):
        raise ValueError("Records must contain exactly both index runners and requested variants")
    data = records.copy(deep=True)
    if data.duplicated(["model", "variant", "sample_id"]).any():
        raise ValueError("Duplicate paired runner/variant/sample IDs")
    for column in ("sample_id", "target_vessel_id", "predicted_vessel_id"):
        if not data[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"Invalid identity column: {column}")
    for column in ("has_hangul", "candidate_found"):
        if not data[column].map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError(f"{column} must contain booleans")
        data[column] = data[column].astype(bool)
    if data.vessel_name.isna().any():
        raise ValueError("vessel_name identity must be present; use an empty string for missing names")
    if (pd.to_numeric(data.candidate_count, errors="raise") <= 0).any():
        raise ValueError("Live rank predictions require a nonempty candidate set")
    values = data.loc[:, TIMING_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Timing values must be finite and nonnegative")
    if not np.allclose(values[:, 3], values[:, :3].sum(axis=1), rtol=1e-7, atol=1e-8):
        raise ValueError("Total timing must equal retrieval, feature and rank time")
    if not np.isfinite(pd.to_numeric(data.top_score, errors="raise")).all():
        raise ValueError("Top scores must be finite")
    for column in data:
        if any(column == f"{side}_{feature}" for side in ("top", "target")
               for feature in DIAGNOSTIC_FEATURES):
            numbers = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
            if np.isinf(numbers).any():
                raise ValueError("Diagnostic feature values must be finite or absent")
    # Check one stable identity per query across both runners AND all variants.
    identities = data[["sample_id", "target_vessel_id", "has_hangul", "vessel_name"]].drop_duplicates()
    if identities.sample_id.duplicated().any():
        raise ValueError("Paired query identity differs across runners or variants")
    ids = set(identities.sample_id)
    for variant in variants:
        for model in (REFERENCE, EXPERIMENT):
            part = data.loc[data.variant.eq(variant) & data.model.eq(model)]
            if len(part) != len(ids) or set(part.sample_id) != ids:
                raise ValueError("All paired runners and variants must cover identical sample IDs")
            # This also validates candidate-found/rank/count consistency.
            summarize_predictions(part)
    correct = data.predicted_vessel_id.eq(data.target_vessel_id)
    if not np.array_equal(correct.to_numpy(), data.target_rank.eq(1).to_numpy()):
        raise ValueError("Predicted identity disagrees with target rank")
    if "queries" in coverage and coverage["queries"] != len(ids):
        raise ValueError("Coverage query count differs from paired records")
    return data


def _error_reason(frame: pd.DataFrame, runner: str) -> np.ndarray:
    correct = frame[f"predicted_vessel_id_{runner}"].eq(frame.target_vessel_id)
    return np.select([correct, ~frame[f"candidate_found_{runner}"]],
                     ["correct", "candidate_missing"], default="ranking_error")


def _changes(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    changes, _ = paired_changes(records, reference=REFERENCE, experiments=(EXPERIMENT,))
    changes.rename(columns={name: name[:-len("_experiment")] + "_" + EXPERIMENT
                            for name in changes if name.endswith("_experiment")}, inplace=True)
    for runner in (REFERENCE, EXPERIMENT):
        changes[f"error_reason_{runner}"] = _error_reason(changes, runner)
    changes["error_transition"] = (changes[f"error_reason_{REFERENCE}"] + "->"
                                   + changes[f"error_reason_{EXPERIMENT}"])
    changes["candidate_gained"] = (~changes[f"candidate_found_{REFERENCE}"]
                                    & changes[f"candidate_found_{EXPERIMENT}"])
    changes["candidate_lost"] = (changes[f"candidate_found_{REFERENCE}"]
                                  & ~changes[f"candidate_found_{EXPERIMENT}"])
    changes["candidate_set_changed"] = changes[f"candidate_ids_sha256_{REFERENCE}"].ne(
        changes[f"candidate_ids_sha256_{EXPERIMENT}"])
    changes["still_wrong_prediction_changed"] = (changes.status.eq("still_wrong")
                                                  & changes.prediction_changed)
    for column in ("top_score", "score_margin", "target_rank", *TIMING_COLUMNS,
                   *(f"{side}_{feature}" for side in ("top", "target")
                     for feature in DIAGNOSTIC_FEATURES)):
        left, right = column + "_" + REFERENCE, column + "_" + EXPERIMENT
        if left in changes and right in changes:
            changes[column + "_delta"] = (pd.to_numeric(changes[right], errors="raise")
                                           - pd.to_numeric(changes[left], errors="raise"))
    changes.sort_values(["variant", "sample_id"], kind="stable", inplace=True)
    changes.reset_index(drop=True, inplace=True)
    # Preserve language: the same vessel may have Hangul AND non-Hangul rows.
    grouped = changes.groupby(["variant", "target_vessel_id", "has_hangul"], sort=True)
    vessels = grouped.agg(
        queries=("sample_id", "size"),
        vessel_names=("vessel_name", lambda values: " | ".join(sorted(set(values)))),
        corrected=("status", lambda values: int(values.eq("corrected").sum())),
        regressed=("status", lambda values: int(values.eq("regressed").sum())),
        still_wrong=("status", lambda values: int(values.eq("still_wrong").sum())),
        unchanged_correct=("status", lambda values: int(values.eq("unchanged_correct").sum())),
        predictions_changed=("prediction_changed", "sum"),
        still_wrong_predictions_changed=("still_wrong_prediction_changed", "sum"),
        candidate_gained=("candidate_gained", "sum"),
        candidate_lost=("candidate_lost", "sum"),
        candidate_sets_changed=("candidate_set_changed", "sum"),
        mean_candidate_delta=("candidate_count_delta", "mean"),
    ).reset_index()
    vessels["net_corrected"] = vessels.corrected - vessels.regressed
    return changes, vessels


def _transitions(frame: pd.DataFrame) -> dict:
    result = {"queries": len(frame)}
    selectors = {status: frame.status.eq(status) for status in STATUSES}
    selectors.update({name: frame[name] for name in (
        "candidate_gained", "candidate_lost", "candidate_set_changed",
        "prediction_changed", "still_wrong_prediction_changed")})
    for name, selected in selectors.items():
        result[name] = int(selected.sum())
        result[name + "_vessels"] = int(frame.loc[selected, "target_vessel_id"].nunique())
    for runner in (REFERENCE, EXPERIMENT):
        result["errors_" + runner] = {
            reason: int(frame[f"error_reason_{runner}"].eq(reason).sum())
            for reason in ("candidate_missing", "ranking_error")}
    result["net_corrected"] = result["corrected"] - result["regressed"]
    return result


def _percent(value) -> str:
    return "—" if value is None else f"{100 * value:.4f}%"


def _report(summary: dict) -> str:
    coverage = summary["coverage"]
    lines = ["# 고정 순위 모델의 훈련 사전 80%·100% 비교", "",
        f"- 외부 fold: {coverage.get('fold', '—')} / {coverage.get('folds', '—')}",
        f"- 평가 행: {coverage.get('queries', '—')} / 전체 검증 {coverage.get('heldout_queries', '—')}",
        f"- 진단 표본 여부: {coverage.get('diagnostic_sample', False)}", "",
        "두 실행은 같은 순위 모델 가중치와 동일한 변형 입력을 사용합니다. "
        "기존 훈련 집합 안의 선박 80%로 만든 사전(index_80)과 전체 훈련 사전(index_100)을 비교합니다. "
        "80%는 선박 수 기준이며 행 수 비율은 다를 수 있습니다. 검증 행과 검증 선박은 두 사전에서 모두 제외합니다.", "",
        "모든 증감·개선·신규 오답은 같은 입력 조건의 **100% − 80%**입니다. "
        "사전에 포함한 별칭·조회 정보·국적/선종 통계를 함께 바꾸므로 후보 집합이 같아야 한다는 요구는 하지 않습니다.", "",
        "## 정확도와 후보 재현율", "",
        "| 입력 조건 | 사전 | 가중 정확도 | 한글 정확도 | 비한글 정확도 | 가중 후보 재현율 | 한글 후보 재현율 | 비한글 후보 재현율 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        for runner in (REFERENCE, EXPERIMENT):
            metric = summary["metrics"][variant][runner]
            values = [metric["weighted_top1"], metric["hangul"]["top1_accuracy"],
                      metric["non_hangul"]["top1_accuracy"], metric["weighted_candidate_recall"],
                      metric["hangul"]["candidate_recall"], metric["non_hangul"]["candidate_recall"]]
            lines.append(f"| {variant} | {runner} | " + " | ".join(map(_percent, values)) + " |")
    lines += ["", "가중 지표는 한글70%·비한글30%입니다. 어느 언어의 행도 없으면 가중 지표를 계산하지 않습니다. "
              "언어별 분자·분모와 recall@5/10/50/100/200은 summary.json에 저장합니다.", "",
        "## 후보 수와 실행 시간", "",
        "| 입력 조건 | 사전 | 후보 평균 | 후보 p95 | 후보 최대 | 검색 초 | 특징 초 | 순위 초 | 합계 초 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        for runner in (REFERENCE, EXPERIMENT):
            metric = summary["metrics"][variant][runner]["all"]
            count = metric["candidate_count"]
            timings = [metric["timing"][key]["sum"] for key in TIMING_COLUMNS]
            lines.append(f"| {variant} | {runner} | {count['mean']:.2f} | {count['p95']:.2f} | "
                         f"{count['max']} | " + " | ".join(f"{value:.3f}" for value in timings) + " |")
    lines += ["", "각 질의의 검색·특징·순위 계산 시간을 측정하며 로드·사전 생성·이름 특징 준비·워밍업은 startup에 분리합니다. "
              "행마다 두 실행의 순서를 번갈아 적용합니다. 캐시·시스템 부하의 영향이 있고 서버 실행 시간을 예측하지 않습니다.", "",
        "## 80% 대비 100%의 개선·신규 오답", "",
        "| 입력 조건 | 정확도 변화 (%p) | 개선 행/척 | 신규 오답 행/척 | 정답 후보 획득 행/척 | 정답 후보 손실 행/척 | 오답 유지 중 예측 변경 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        change = summary["transitions"][variant]
        delta = summary["metric_deltas"][variant]["weighted_top1_pp"]
        delta_text = "—" if delta is None else f"{delta:+.4f}"
        pairs = [f"{change[name]}/{change[name + '_vessels']}" for name in (
            "corrected", "regressed", "candidate_gained", "candidate_lost")]
        lines.append(f"| {variant} | {delta_text} | " + " | ".join(pairs)
                     + f" | {change['still_wrong_prediction_changed']} |")
    lines += ["", "행별 변화는 row_changes.csv, 언어를 구분한 선박별 집계는 vessel_changes.csv에 저장합니다. "
              "candidate_missing은 정답 후보 누락, ranking_error는 후보 안의 순위 오류입니다. "
              "후보 누락·복구와 최종 정답 여부는 별개로 집계합니다. "
              "top_/target_ 진단 열은 각 사전의 예측 1위와 정답 후보 특징이며 정답 후보가 없으면 해당 특징과 차이는 비어 있습니다. "
              "예측이 바뀌면 두 top_ 열은 서로 다른 선박의 특징일 수 있습니다.", "",
        "## 해석 범위", "",
        "이 결과는 훈련에서 제외한 외부 검증의 사전 규모 민감도 진단입니다. "
        "고정 가중치는 원래 전체 훈련 집합에서 학습했으므로 80% 사전은 가상의 축소 환경입니다. "
        "외부 vessel 검증의 정답 선박은 두 사전 모두에서 제외돼 사전 확대가 새로 알게 된 선박에 주는 이득은 측정하지 못합니다. "
        "전체훈련 Public 점수 하락을 직접 재현하거나 원인을 확정하는 실험은 아닙니다. "
        "긴 실행의 완료·원본 예측 일치·가중치 및 입력 보존 확인은 summary.json의 complete와 parity 및 proof를 확인합니다.", ""]
    return "\n".join(lines)


def build_index_scale_report(records: pd.DataFrame, *, coverage: dict, proof: dict,
                             startup: dict, parity: dict, variants=VARIANTS):
    """Return summary, paired rows, vessel/language aggregates and Korean report.

    Completion belongs to the caller after its final provenance checks. The
    reference is always index_80, and every difference means index_100-index_80.
    """
    variants = tuple(variants)
    data = _validate_records(records, variants, coverage)
    changes, vessels = _changes(data)
    metrics, deltas, transitions = {}, {}, {}
    for variant in variants:
        part = data.loc[data.variant.eq(variant)]
        metrics[variant] = {runner: summarize_predictions(part.loc[part.model.eq(runner)])
                            for runner in (REFERENCE, EXPERIMENT)}
        deltas[variant] = metric_deltas(metrics[variant][REFERENCE], metrics[variant][EXPERIMENT])
        changed = changes.loc[changes.variant.eq(variant)]
        groups = {"all": changed, "hangul": changed.loc[changed.has_hangul],
                  "non_hangul": changed.loc[~changed.has_hangul]}
        transitions[variant] = {**_transitions(changed),
                                **{name: _transitions(rows) for name, rows in groups.items()}}
    summary = {"schema_version": 1, "complete": False,
        "purpose": "fixed_ranker_training_index_scale_diagnostic",
        "reference_runner": REFERENCE, "experiment_runner": EXPERIMENT,
        "difference_direction": "index_100 minus index_80 within the same input variant",
        "variants": list(variants), "coverage": copy.deepcopy(coverage),
        "proof": copy.deepcopy(proof), "startup": copy.deepcopy(startup),
        "parity": copy.deepcopy(parity), "metrics": metrics, "metric_deltas": deltas,
        "transitions": transitions,
        "paired_identity_verified": True, "candidate_equality_required": False,
        "files": {"predictions": "predictions.csv", "row_changes": "row_changes.csv",
                  "vessel_changes": "vessel_changes.csv", "report": "report.md"},
        "limitations": [
            "Fixed ranker was trained on all outer-training T; smaller index is an artificial asset-scale diagnostic.",
            "Outer-vessel validation excludes each target vessel from both indexes; newly remembered vessel benefit is unmeasured.",
            "This is not a direct reproduction or causal explanation of the full-training Public score change.",
            "Per-query timings exclude startup/warmup and reflect local repeated execution, not server runtime.",
        ]}
    return summary, changes, vessels, _report(summary)
