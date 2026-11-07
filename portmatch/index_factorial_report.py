"""Paired 2x2 reports for independently scoped search and category assets."""
from __future__ import annotations

import copy

import pandas as pd

from .index_factorial_assets import FACTOR_LEVELS, RUNNERS
from .index_scale_report import _changes, _transitions, _validate_records
from .retrieval_comparison import TIMING_COLUMNS, VARIANTS, metric_deltas, summarize_predictions


A, B, C, D = RUNNERS
CONTRASTS = {
    "category_at_search80": {"reference": A, "experiment": B},
    "category_at_search100": {"reference": C, "experiment": D},
    "search_at_category80": {"reference": A, "experiment": C},
    "search_at_category100": {"reference": B, "experiment": D},
    "diagonal": {"reference": A, "experiment": D},
}
CONTRAST_LABELS = {
    "category_at_search80": "검색80 고정·범주80→100",
    "category_at_search100": "검색100 고정·범주80→100",
    "search_at_category80": "범주80 고정·검색80→100",
    "search_at_category100": "범주100 고정·검색80→100",
    "diagonal": "검색·범주 모두80→100",
}
_LANGUAGES = ("all", "hangul", "non_hangul")


def _pair(records, contrast, variants, coverage):
    """Use the validated existing pair machinery without changing its contract."""
    selected = records.loc[records.model.isin(contrast.values())].copy()
    selected["model"] = selected.model.map({contrast["reference"]: "index_80",
                                            contrast["experiment"]: "index_100"})
    return _validate_records(selected, variants, coverage)


def _check_same_search_candidates(pair):
    fields = ["candidate_count", "candidate_ids_sha256", "candidate_found"]
    keyed = pair.set_index(["variant", "sample_id", "model"])
    left = keyed.xs("index_80", level="model").sort_index()
    right = keyed.xs("index_100", level="model").sort_index()
    if not left[fields].equals(right[fields]):
        raise ValueError("Changing only category scope must preserve candidate sets, counts and target presence")
    hashes = pair.candidate_ids_sha256
    if not hashes.map(lambda value: isinstance(value, str) and len(value) == 64
                      and all(char in "0123456789abcdef" for char in value)).all():
        raise ValueError("Candidate set signatures must be lowercase SHA256 digests")


def _rename_pair_fields(frame):
    return frame.rename(columns={name: name[:-len(suffix)] + replacement
        for name in frame.columns
        for suffix, replacement in (("_index_80", "_reference"), ("_index_100", "_experiment"))
        if name.endswith(suffix)})


def _rename_errors(value):
    if not isinstance(value, dict):
        return value
    return {key.replace("errors_index_80", "errors_reference").replace(
                "errors_index_100", "errors_experiment"): _rename_errors(item)
            for key, item in value.items()}


def _interaction(metrics):
    """Differences of effects in additive units, never differences of ratios."""
    def difference(path, scale=1.):
        values = []
        for runner in RUNNERS:
            value = metrics[runner]
            for key in path:
                value = value[key]
            values.append(value)
        if any(value is None for value in values):
            return None
        a, b, c, d = values
        return float(((d - c) - (b - a)) * scale)

    result = {"weighted_top1_pp": difference(("weighted_top1",), 100.),
              "weighted_candidate_recall_pp": difference(("weighted_candidate_recall",), 100.)}
    for language in _LANGUAGES:
        result[language] = {
            "top1_accuracy_pp": difference((language, "top1_accuracy"), 100.),
            "candidate_recall_pp": difference((language, "candidate_recall"), 100.),
            "candidate_count_interaction": {stat: difference((language, "candidate_count", stat))
                for stat in ("mean", "p50", "p95", "max")},
            "timing": {column: {stat + "_seconds_interaction":
                difference((language, "timing", column, stat)) for stat in ("sum", "mean", "p95")}
                for column in TIMING_COLUMNS},
        }
    return result


def _format(value, *, percent=False, signed=False, places=4):
    if value is None:
        return "—"
    if percent:
        return f"{100 * value:.{places}f}%"
    return format(value, ("+" if signed else "") + f".{places}f")


def _report(summary):
    coverage = summary["coverage"]
    lines = ["# 고정 86개 순위 모델의 검색 사전 × 국적·선종 통계 2×2 진단", "",
        f"- 외부 fold: {coverage.get('fold', '—')} / {coverage.get('folds', '—')}",
        f"- 평가 행: {coverage.get('queries', '—')} / 전체 검증 {coverage.get('heldout_queries', '—')}",
        f"- 진단 표본 여부: {coverage.get('diagnostic_sample', False)}", "",
        "모든 조합은 같은 모델 가중치·86개 특징·검증 행과 같은 입력 변형을 사용합니다. "
        "검색 사전은 조회·별칭·TF-IDF를 함께 바꾸고, 범주 통계는 국적·선종 모델 전체를 바꿉니다. "
        "80%는 기존 훈련 선박 수 기준이며 양쪽 범위 모두 외부 검증 행·선박을 제외합니다.", "",
        "| 기호 | 조합 | 검색 사전 | 범주 통계 |", "|---|---|---:|---:|",
        f"| A | {A} | 80% | 80% |", f"| B | {B} | 80% | 100% |",
        f"| C | {C} | 100% | 80% |", f"| D | {D} | 100% | 100% |", "",
        "## 조건별 정확도와 후보 재현율", "",
        "| 입력 조건 | 조합 | 가중 정확도 | 한글 정확도 | 비한글 정확도 | 가중 후보 재현율 | 한글 후보 재현율 | 비한글 후보 재현율 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        for symbol, runner in zip("ABCD", RUNNERS):
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
        for symbol, runner in zip("ABCD", RUNNERS):
            metric = summary["metrics"][variant][runner]["all"]
            count = metric["candidate_count"]
            values = [_format(metric["timing"][key]["sum"], places=3) for key in TIMING_COLUMNS]
            lines.append(f"| {variant} | {symbol} | {_format(count['mean'], places=2)} | "
                f"{_format(count['p95'], places=2)} | {count['max']} | " + " | ".join(values) + " |")
    lines += ["", "행별 시간은 검색·특징·순위 계산을 포함합니다. 초기화·warmup·진단 추출·저장은 분리합니다. "
        "실행 순서를 순환해 배치해도 로컬 캐시·부하 영향은 남으며 서버 시간이나 속도 우위를 보장하지 않습니다.", "",
        "## 다섯 대조의 개선·신규 오답", "",
        "차이는 각 대조의 실험 조합−기준 조합입니다. 범주 효과 B−A / D−C, "
        "검색 효과 C−A / D−B, 기존 대각선 D−A를 분리합니다.", "",
        "| 입력 조건 | 대조 | 정확도 변화 (%p) | 개선 행/척 | 신규 오답 행/척 | 후보 획득 행/척 | 후보 손실 행/척 | 오답 유지 중 예측 변경 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        for name in CONTRASTS:
            transition = summary["transitions"][variant][name]
            delta = summary["metric_deltas"][variant][name]["weighted_top1_pp"]
            values = [f"{transition[key]}/{transition[key + '_vessels']}" for key in
                ("corrected", "regressed", "candidate_gained", "candidate_lost")]
            lines.append(f"| {variant} | {CONTRAST_LABELS[name]} | {_format(delta, signed=True)} | "
                + " | ".join(values) + f" | {transition['still_wrong_prediction_changed']} |")
    lines += ["", "행별·선박별 CSV에는 contrast/reference_runner/experiment_runner를 기록합니다. "
        "같은 원본 행이 여러 대조·입력 변형에 반복되므로 이를 독립 표본처럼 합산하지 않습니다. "
        "후보 누락과 후보 안 순위 오류를 구분하며, 범주만 바꾸는 대조는 후보 집합·수·정답 포함 여부가 같아야 합니다.", "",
        "## 상호작용: (D−C)−(B−A)", "",
        "| 입력 조건 | 가중 정확도 (%p) | 가중 후보 재현율 (%p) | 전체 평균 후보 수 | 합계 시간 (초) |",
        "|---|---:|---:|---:|---:|"]
    for variant in summary["variants"]:
        interaction = summary["factorial_interactions"][variant]
        values = [interaction["weighted_top1_pp"], interaction["weighted_candidate_recall_pp"],
            interaction["all"]["candidate_count_interaction"]["mean"],
            interaction["all"]["timing"]["total_seconds"]["sum_seconds_interaction"]]
        lines.append(f"| {variant} | " + " | ".join(_format(value, signed=True) for value in values) + " |")
    lines += ["", "상호작용은 검색 규모에 따라 범주 규모의 효과가 얼마나 달랐는지 나타내는 산술 차이입니다. "
        "통계적 유의성이나 Public 하락의 원인 증명이 아니며 시간 상호작용은 측정 잡음을 포함할 수 있습니다.", "",
        "## 동일 후보 특징과 해석 범위", "",
        "전체86개 특징 비교는 동일한 후보 선박 ID끼리 확인합니다. 후보가 없는 조합은 부재로 남기며 정답을 강제로 추가하지 않습니다. "
        "예측이 바뀌면 row_changes.csv의 top_ 특징은 서로 다른 후보일 수 있습니다. "
        "고정 모델은 훈련T 전체로 학습됐고80% 사전은 인위적 축소 환경입니다. 외부 검증의 정답 선박은 모든 자산에서 제외돼 "
        "새 선박을 기억하는 이득은 측정하지 못합니다. 실제 전체훈련 재학습 효과와 Public 차이를 이 결과만으로 확정하지 않습니다. "
        "완료·기존 대각선 재현·가중치/자산 보존은 summary.json의 complete/parity/proof와 실행기 검사를 함께 확인합니다.", ""]
    return "\n".join(lines)


def build_index_factorial_report(records, *, coverage, proof, startup, parity, variants=VARIANTS):
    """Return an incomplete summary, five paired comparisons and Korean report.

    Only the caller can mark completion after weight, provenance and feature
    isolation checks. Every contrast is experiment minus reference, within a
    single variant. Interaction is (D-C)-(B-A) in additive units.
    """
    if (not isinstance(records, pd.DataFrame) or records.empty or not records.columns.is_unique
            or not {"model", "variant"}.issubset(records.columns)
            or set(records.model) != set(RUNNERS)):
        raise ValueError("Exactly four nonempty factorial runners with unique columns are required")
    variants = tuple(variants)
    pair_tables = {name: _pair(records, contrast, variants, coverage)
                   for name, contrast in CONTRASTS.items()}
    for name in ("category_at_search80", "category_at_search100"):
        _check_same_search_candidates(pair_tables[name])
    metrics = {variant: {runner: summarize_predictions(records.loc[
        records.variant.eq(variant) & records.model.eq(runner)]) for runner in RUNNERS}
        for variant in variants}
    deltas, transitions = ({variant: {} for variant in variants} for _ in range(2))
    all_changes, all_vessels = [], []
    for name, pair in pair_tables.items():
        changes, vessels = _changes(pair)
        contrast = CONTRASTS[name]
        for variant in variants:
            deltas[variant][name] = metric_deltas(metrics[variant][contrast["reference"]],
                                                 metrics[variant][contrast["experiment"]])
            rows = changes.loc[changes.variant.eq(variant)]
            groups = {"all": rows, "hangul": rows.loc[rows.has_hangul],
                      "non_hangul": rows.loc[~rows.has_hangul]}
            transitions[variant][name] = _rename_errors({**_transitions(rows),
                **{language: _transitions(part) for language, part in groups.items()}})
        changes = _rename_pair_fields(changes).drop(columns="model")
        for frame in (changes, vessels):
            frame.insert(0, "experiment_runner", contrast["experiment"])
            frame.insert(0, "reference_runner", contrast["reference"])
            frame.insert(0, "contrast", name)
        all_changes.append(changes)
        all_vessels.append(vessels)
    summary = {"schema_version": 1, "complete": False,
        "purpose": "fixed_ranker_search_category_2x2_diagnostic", "runners": list(RUNNERS),
        "factor_levels": copy.deepcopy(FACTOR_LEVELS), "contrasts": copy.deepcopy(CONTRASTS),
        "difference_direction": "experiment minus reference within each contrast and input variant",
        "interaction_direction": "(D-C)-(B-A), with A=search80_category80 and D=search100_category100",
        "variants": list(variants), "coverage": copy.deepcopy(coverage), "proof": copy.deepcopy(proof),
        "startup": copy.deepcopy(startup), "parity": copy.deepcopy(parity), "metrics": metrics,
        "metric_deltas": deltas, "transitions": transitions,
        "factorial_interactions": {variant: _interaction(metrics[variant]) for variant in variants},
        "paired_identity_verified": True, "same_search_candidate_invariance_verified": True,
        "candidate_equality_required": "only between category scopes sharing the same search scope",
        "files": {"predictions": "predictions.csv", "row_changes": "row_changes.csv",
                  "vessel_changes": "vessel_changes.csv", "report": "report.md"},
        "limitations": [
            "Fixed ranker was trained on all outer-training T; 80% assets are artificial diagnostics.",
            "All target vessels are excluded from all training assets; newly known vessel benefits are unmeasured.",
            "Interactions are descriptive differences, not statistical significance or a Public-score causal claim.",
            "Local per-query timings exclude startup, warmup, diagnostic extraction and output writing.",
            "Input variants and contrast rows are repeated measurements, not independent samples to pool.",
        ]}
    return (summary, pd.concat(all_changes, ignore_index=True),
            pd.concat(all_vessels, ignore_index=True), _report(summary))
