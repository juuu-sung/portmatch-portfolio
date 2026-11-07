"""Paired reports for retraining without four absolute category-support features."""
from __future__ import annotations

import copy

import pandas as pd

from .index_factorial_report import _format, _pair, _rename_errors, _rename_pair_fields
from .index_scale_report import _changes, _transitions
from .retrieval_comparison import TIMING_COLUMNS, VARIANTS, metric_deltas, summarize_predictions


RUNNERS = ("baseline86", "no_support82")
REFERENCE, EXPERIMENT = RUNNERS
CONTRAST = "no_support82_minus_baseline86"
SUPPORT_FEATURES = (
    "flag_source_support_log", "flag_pair_support_log",
    "type_source_support_log", "type_pair_support_log",
)


def _check_candidates(data):
    fields = ["candidate_count", "candidate_ids_sha256", "candidate_found"]
    keyed = data.set_index(["variant", "sample_id", "model"])
    left = keyed.xs("index_80", level="model").sort_index()
    right = keyed.xs("index_100", level="model").sort_index()
    if not left[fields].equals(right[fields]):
        raise ValueError("Support-feature ablation must preserve candidate sets, counts and target presence")
    if not data.candidate_ids_sha256.map(lambda value: isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)).all():
        raise ValueError("Candidate set signatures must be lowercase SHA256 digests")


def _report(summary):
    coverage = summary["coverage"]
    lines = ["# 지원도 로그4열 제거: 86개 대82개 재학습 비교", "",
        f"- 외부 fold: {coverage.get('fold', '—')} / {coverage.get('folds', '—')}",
        f"- 평가 행: {coverage.get('queries', '—')} / 전체 검증 {coverage.get('heldout_queries', '—')}",
        f"- 진단 표본 여부: {coverage.get('diagnostic_sample', False)}", "",
        "기준 baseline86과 실험 no_support82는 같은 robust 검색100%·국적·선종 통계100%를 사용합니다. "
        "기존 GT 결측 증강 캐시의 후보·라벨·원본 질량0.5 가중치와 나머지82개 특징을 유지하고, "
        "지원도 로그4열만 모델 입력에서 제외해 재학습합니다. 확률·lift·희귀 대응 정보는 유지합니다. "
        "두 모델의 학습된 가중치는 서로 다르므로 고정 가중치 진단이나 통계 규모 실험으로 해석하지 않습니다.", "",
        "제외 열: " + ", ".join(f"`{name}`" for name in SUPPORT_FEATURES) + ".", "",
        "모든 증감·개선·신규 오답은 같은 입력 조건의 **82개−86개**입니다.", "",
        "## 정확도와 후보 재현율", "",
        "| 입력 조건 | 모델 | 가중 정확도 | 한글 정확도 | 비한글 정확도 | 가중 후보 재현율 | 한글 후보 재현율 | 비한글 후보 재현율 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        for runner in RUNNERS:
            metric = summary["metrics"][variant][runner]
            values = [metric["weighted_top1"], metric["hangul"]["top1_accuracy"],
                metric["non_hangul"]["top1_accuracy"], metric["weighted_candidate_recall"],
                metric["hangul"]["candidate_recall"], metric["non_hangul"]["candidate_recall"]]
            lines.append(f"| {variant} | {runner} | " + " | ".join(
                _format(value, percent=True) for value in values) + " |")
    lines += ["", "가중 지표는 한글70%·비한글30%이며 한 언어가 없으면 계산하지 않습니다. "
        "언어별 정답 수·분모·Recall@5/10/50/100/200은 summary.json에 있습니다. "
        "서로 다른 입력 조건의 점수를 평균내지 않습니다.", "",
        "## 후보 수와 구간별 시간", "",
        "| 입력 조건 | 모델 | 후보 평균 | 후보 p95 | 후보 최대 | 검색 초 | 특징 초 | 순위 초 | 합계 초 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        for runner in RUNNERS:
            metric = summary["metrics"][variant][runner]["all"]
            count = metric["candidate_count"]
            times = [_format(metric["timing"][key]["sum"], places=3) for key in TIMING_COLUMNS]
            lines.append(f"| {variant} | {runner} | {_format(count['mean'], places=2)} | "
                f"{_format(count['p95'], places=2)} | {count['max']} | " + " | ".join(times) + " |")
    lines += ["", "두 실행의 후보 집합·수·정답 포함 여부는 모든 행에서 같아야 합니다. "
        "시간은 검색·특징·순위 계산을 포함하고 초기화·warmup·진단 추출·저장은 제외합니다. "
        "공유 캐시와 시스템 부하 영향이 남으므로 속도 우위나 제출 서버 시간을 보장하지 않습니다.", "",
        "## 82개−86개 개선·신규 오답", "",
        "| 입력 조건 | 정확도 변화 (%p) | 개선 행/척 | 신규 오답 행/척 | 후보 획득 행/척 | 후보 손실 행/척 | 오답 유지 중 예측 변경 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        transition = summary["transitions"][variant]
        delta = summary["metric_deltas"][variant]["weighted_top1_pp"]
        values = [f"{transition[key]}/{transition[key + '_vessels']}" for key in
            ("corrected", "regressed", "candidate_gained", "candidate_lost")]
        lines.append(f"| {variant} | {_format(delta, signed=True)} | " + " | ".join(values)
            + f" | {transition['still_wrong_prediction_changed']} |")
    lines += ["", "row_changes.csv와 vessel_changes.csv에는 기준·실험 모델과 변화 방향을 기록합니다. "
        "candidate_missing은 정답 후보 누락, ranking_error는 후보 안 순위 오류입니다. "
        "같은 원본 행의 여러 입력 조건을 독립 표본처럼 합산하지 않습니다.", "",
        "## 실제 입력 특징과 해석 범위", "",
        "실행 중 모든 동일 후보의 공통82개 float32 입력값이 NaN을 포함해 동일한지 검사합니다. "
        "정답 후보와 두 모델의1위 후보를 합친 집합의 실제 입력·점수·순위를 candidate_features.csv에 저장합니다. "
        "82개 모델의 제외4열은 빈칸이며 계산된 NaN이 아니라 모델에 전달하지 않은 열입니다. "
        "feature_count·omitted_features로 이를 구분하며 실제 입력 순서는 feature_schema.json에 있습니다. "
        "정답 후보가 없으면 부재로 남기고 강제로 추가하지 않습니다. "
        "예측이 바뀐 행의 top_ 특징은 서로 다른 후보일 수 있으므로 동일 candidate_vessel_id를 비교해야 합니다.", "",
        "순위 점수와 점수 차이는 재학습 모델 간 보정된 확률이 아닙니다. "
        "이번 결과는 지원도 입력 제거 후 재학습 효과이며 개별 특징의 인과 효과나 Public 점수 개선을 보장하지 않습니다. "
        "이전에 활용한 fold이므로 새 독립 holdout으로 간주하지 않습니다. "
        "summary.json의 complete·parity·proof 및 기존 캐시/모델 보존 검사를 함께 확인합니다.", ""]
    return "\n".join(lines)


def build_support_ablation_report(records, *, coverage, proof, startup, parity, variants=VARIANTS):
    """Return incomplete summary, paired rows, vessel aggregates, and Markdown.

    Every difference is retrained no_support82 minus baseline86, per input variant.
    The caller verifies common input values and asset isolation before completion.
    """
    if (not isinstance(records, pd.DataFrame) or records.empty or not records.columns.is_unique
            or not {"model", "variant"}.issubset(records.columns)
            or set(records.model) != set(RUNNERS)):
        raise ValueError("Exactly two support-ablation runners with unique columns are required")
    variants = tuple(variants)
    data = _pair(records, {"reference": REFERENCE, "experiment": EXPERIMENT}, variants, coverage)
    _check_candidates(data)
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
        "purpose": "retrained_absolute_category_support_ablation", "runners": list(RUNNERS),
        "reference_runner": REFERENCE, "experiment_runner": EXPERIMENT, "contrast": CONTRAST,
        "difference_direction": "no_support82 minus baseline86 within each input variant",
        "feature_counts": {REFERENCE: 86, EXPERIMENT: 82},
        "excluded_features": list(SUPPORT_FEATURES), "fixed_search_fraction": 1.,
        "fixed_category_fraction": 1., "ranker_weights_fixed": False,
        "variants": list(variants), "coverage": copy.deepcopy(coverage), "proof": copy.deepcopy(proof),
        "startup": copy.deepcopy(startup), "parity": copy.deepcopy(parity), "metrics": metrics,
        "metric_deltas": deltas, "transitions": transitions,
        "paired_identity_verified": True, "same_search_candidate_invariance_verified": True,
        "candidate_equality_required": True,
        "files": {"predictions": "predictions.csv", "row_changes": "row_changes.csv",
            "vessel_changes": "vessel_changes.csv", "report": "report.md",
            "candidate_features": "candidate_features.csv",
            "candidate_comparisons": "candidate_comparisons.csv", "feature_schema": "feature_schema.json"},
        "limitations": [
            "Both rankers use outer-training 100% search and category statistics; this is not a statistics-size experiment.",
            "Ranker weights differ after retraining; score deltas are not calibrated probability differences.",
            "Held-out targets must remain excluded from training, search memory and category statistics.",
            "These folds were used by earlier experiments; this is additional validation, not a fresh holdout.",
            "This does not establish the cause of the Public score decrease or guarantee submission improvement.",
            "Local timings exclude startup, warmup, diagnostic extraction and writing; shared caches and load remain.",
            "Input variants are repeated measurements, not independent samples to pool.",
        ]}
    return summary, changes, vessels, _report(summary)
