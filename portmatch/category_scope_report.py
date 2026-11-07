"""Paired category-scope reports with search fixed at outer-training 100%."""
from __future__ import annotations

import copy

import pandas as pd

from .category_scope_assets import FACTOR_LEVELS, RUNNERS
from .index_factorial_report import (
    _check_same_search_candidates, _format, _pair, _rename_errors, _rename_pair_fields,
)
from .index_scale_report import _changes, _transitions
from .retrieval_comparison import TIMING_COLUMNS, VARIANTS, metric_deltas, summarize_predictions


REFERENCE, EXPERIMENT = RUNNERS
CONTRAST = "category80_minus100_at_search100"


def _report(summary):
    coverage = summary["coverage"]
    lines = ["# 검색100% 고정·국적·선종 통계80% 대100% 비교", "",
        f"- 외부 fold: {coverage.get('fold', '—')} / {coverage.get('folds', '—')}",
        f"- 평가 행: {coverage.get('queries', '—')} / 전체 검증 {coverage.get('heldout_queries', '—')}",
        f"- 진단 표본 여부: {coverage.get('diagnostic_sample', False)}", "",
        "기준 D는 검색100%·범주100%, 실험 C는 검색100%·범주80%입니다. "
        "가중치·86개 특징·검색 사전은 동일하며 국적·선종 통계 모델 전체만 교체합니다. "
        "80%는 외부 훈련T의 선박 수 기준이며 선택된 선박의 원본 행을 모두 사용합니다. "
        "검증 행·선박은 T와80% 부분집합 모두에서 제외합니다.", "",
        "모든 변화·개선·신규 오답은 **C−D, 즉 범주80%−100%**입니다. "
        "기존2×2 보고서의 category_at_search100(D−C)과 부호가 반대입니다.", "",
        "## 정확도와 후보 재현율", "",
        "| 입력 조건 | 조합 | 가중 정확도 | 한글 정확도 | 비한글 정확도 | 가중 후보 재현율 | 한글 후보 재현율 | 비한글 후보 재현율 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        for symbol, runner in zip("DC", RUNNERS):
            metric = summary["metrics"][variant][runner]
            values = [metric["weighted_top1"], metric["hangul"]["top1_accuracy"],
                metric["non_hangul"]["top1_accuracy"], metric["weighted_candidate_recall"],
                metric["hangul"]["candidate_recall"], metric["non_hangul"]["candidate_recall"]]
            lines.append(f"| {variant} | {symbol} | " + " | ".join(
                _format(value, percent=True) for value in values) + " |")
    lines += ["", "가중 지표는 한글70%·비한글30%이며 한 언어가 없으면 계산하지 않습니다. "
        "언어별 정답 수·분모·Recall@5/10/50/100/200은 summary.json에 있습니다. "
        "서로 다른 다섯 입력 조건의 점수를 평균내지 않습니다.", "",
        "## 후보 수와 구간별 시간", "",
        "| 입력 조건 | 조합 | 후보 평균 | 후보 p95 | 후보 최대 | 검색 초 | 특징 초 | 순위 초 | 합계 초 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        for symbol, runner in zip("DC", RUNNERS):
            metric = summary["metrics"][variant][runner]["all"]
            count = metric["candidate_count"]
            values = [_format(metric["timing"][key]["sum"], places=3) for key in TIMING_COLUMNS]
            lines.append(f"| {variant} | {symbol} | {_format(count['mean'], places=2)} | "
                f"{_format(count['p95'], places=2)} | {count['max']} | " + " | ".join(values) + " |")
    lines += ["", "두 실행의 후보 집합·수·정답 포함 여부는 모든 행에서 같아야 합니다. "
        "행별 시간은 검색·특징·순위 계산을 포함하고 초기화·warmup·진단 추출·저장은 제외합니다. "
        "실행 순서를 번갈아도 공유 검색 캐시와 시스템 부하 영향이 남으므로 속도 우위나 서버 시간을 보장하지 않습니다.", "",
        "## C−D 개선·신규 오답", "",
        "| 입력 조건 | 정확도 변화 (%p) | 개선 행/척 | 신규 오답 행/척 | 후보 획득 행/척 | 후보 손실 행/척 | 오답 유지 중 예측 변경 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        transition = summary["transitions"][variant]
        delta = summary["metric_deltas"][variant]["weighted_top1_pp"]
        values = [f"{transition[key]}/{transition[key + '_vessels']}" for key in
            ("corrected", "regressed", "candidate_gained", "candidate_lost")]
        lines.append(f"| {variant} | {_format(delta, signed=True)} | " + " | ".join(values)
            + f" | {transition['still_wrong_prediction_changed']} |")
    lines += ["", "행별·선박별 CSV에는 reference_runner(D)와 experiment_runner(C)를 기록합니다. "
        "candidate_missing은 정답 후보 누락, ranking_error는 후보 안 순위 오류입니다. "
        "동일 원본 행의 다섯 입력 조건을 독립 표본처럼 합산하지 않습니다.", "",
        "## 동일 후보86개 특징과 해석 범위", "",
        "모든 후보의 비범주70개 특징이 동일한지 검사합니다. 정답 후보와 두 실행의1위 후보를 합친 집합에 대해 "
        "실제 모델에 입력한86개 float32 특징·점수·순위를 저장합니다. "
        "후보가 없으면 부재로 남기고 정답을 강제로 추가하지 않습니다. "
        "예측이 바뀐 행의 top_ 열은 서로 다른 후보일 수 있으므로 candidate_features.csv의 동일 선박을 비교합니다.", "",
        "고정 모델은 훈련T 전체로 학습됐고80% 범주 통계는 인위적 축소 환경입니다. "
        "이 실험만으로 전체훈련 재학습 효과나 Public 하락 원인을 확정하지 않습니다. "
        "fold1·3·4는 이전 실험에도 사용한 추가 검증 구간입니다. "
        "summary.json의 complete·parity·proof 및 가중치/자산 보존 검사를 함께 확인합니다.", ""]
    return "\n".join(lines)


def build_category_scope_report(records, *, coverage, proof, startup, parity, variants=VARIANTS):
    """Return incomplete summary, paired rows, vessel aggregates and Markdown.

    Reference is search100_category100 (D); experiment is search100_category80
    (C). Every delta is C-D, separately for each input variant.
    """
    if (not isinstance(records, pd.DataFrame) or records.empty or not records.columns.is_unique
            or not {"model", "variant"}.issubset(records.columns)
            or set(records.model) != set(RUNNERS)):
        raise ValueError("Exactly two category-scope runners with unique columns are required")
    variants = tuple(variants)
    contrast = {"reference": REFERENCE, "experiment": EXPERIMENT}
    data = _pair(records, contrast, variants, coverage)
    _check_same_search_candidates(data)
    changes, vessels = _changes(data)
    metrics, deltas, transitions = {}, {}, {}
    for variant in variants:
        part = records.loc[records.variant.eq(variant)]
        metrics[variant] = {runner: summarize_predictions(part.loc[part.model.eq(runner)])
            for runner in RUNNERS}
        deltas[variant] = metric_deltas(metrics[variant][REFERENCE], metrics[variant][EXPERIMENT])
        changed = changes.loc[changes.variant.eq(variant)]
        groups = {"all": changed, "hangul": changed.loc[changed.has_hangul],
            "non_hangul": changed.loc[~changed.has_hangul]}
        transitions[variant] = _rename_errors({**_transitions(changed),
            **{name: _transitions(rows) for name, rows in groups.items()}})
    changes = _rename_pair_fields(changes).drop(columns="model")
    for frame in (changes, vessels):
        frame.insert(0, "experiment_runner", EXPERIMENT)
        frame.insert(0, "reference_runner", REFERENCE)
        frame.insert(0, "contrast", CONTRAST)
    summary = {"schema_version": 1, "complete": False,
        "purpose": "fixed_ranker_search100_category_scope_diagnostic", "runners": list(RUNNERS),
        "factor_levels": copy.deepcopy(FACTOR_LEVELS), "reference_runner": REFERENCE,
        "experiment_runner": EXPERIMENT, "contrast": CONTRAST,
        "difference_direction": "C-D: category80 minus category100 at fixed search100 within each input variant",
        "variants": list(variants), "coverage": copy.deepcopy(coverage), "proof": copy.deepcopy(proof),
        "startup": copy.deepcopy(startup), "parity": copy.deepcopy(parity), "metrics": metrics,
        "metric_deltas": deltas, "transitions": transitions,
        "paired_identity_verified": True, "same_search_candidate_invariance_verified": True,
        "candidate_equality_required": True,
        "files": {"predictions": "predictions.csv", "row_changes": "row_changes.csv",
            "vessel_changes": "vessel_changes.csv", "report": "report.md"},
        "limitations": [
            "Fixed ranker was trained on all outer-training T; category80 is an artificial diagnostic.",
            "All target vessels are excluded from both category scopes and shared search assets.",
            "These folds were used by earlier experiments; this is additional validation, not a fresh holdout.",
            "This does not directly explain the full-training Public score change or predict retraining effects.",
            "Local per-query timings exclude startup, warmup, diagnostic extraction and output writing; shared caches and load remain.",
            "Input variants are repeated measurements, not independent samples to pool.",
        ]}
    return summary, changes, vessels, _report(summary)
