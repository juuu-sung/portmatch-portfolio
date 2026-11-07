# CH 대체 발음 일관성 — 86개 기준과 93개 실험

> 공개용 정리본: 실제 선박별 입력·정답·예측·점수와 행 식별자는 제외했다. 아래 성능표는 실제 실험의 집계 수치이며, 오류의 원리는 별도의 [가상 예시](../fictional-examples.md)로 설명한다.

[프로젝트 개요](../../README.md) · [전체 학습 안내](../../TRAINING.md) · [86개 기준 결과](token-phonetic-5fold.md)

## 실험 목적

Glovis Chorus의 전체 이름 CH→K 대체 표현은 이미 정답을 지지했지만, 토큰·잔여 모음에는 기본 CH→J 표현이 남았습니다. 기존 86개 특징을 보존하고 CH 대체 표현을 일관되게 사용하는 보조 특징 7개를 추가합니다. 기존 규칙을 덮어쓰거나 특정 선박의 예외 발음을 등록하지 않습니다.

| 구분 | 기준 | 실험 |
|---|---|---|
| 변환 버전 | `token_phonetic_v1` | `token_phonetic_ch_v1` |
| 캐시 열 | 87 | 94 = 기존 87 + 새 7 |
| 학습 특징 | 86 | 93 = 기존 86 + 새 7 |
| 공통 제외 | `retrieval_tonnage_rank` | 동일 |
| 분할·후보·OOF·오답 샘플 | 기존 5-fold | 동일, 다시 검색·샘플링하지 않음 |
| 학습 설정 | 300회, 6스레드, seed 20261022 | 동일 |

## 새 특징과 발음 선택

`ch_variant_features.py`는 기존의 CH→J, CH→K 두 표현을 각 이름에 적용합니다. 이름 한 개 안의 CH는 해당 표현의 규칙을 일관되게 사용하므로, 한 쌍당 최대 네 가지 발음 조합을 비교합니다. 토큰별로 서로 다른 발음을 임의로 섞지 않습니다.

전체 토큰 정렬 점수가 높은 조합을 선택하고, 동점이면 가장 약한 토큰 점수·모음 정렬을 순서대로 비교합니다. 마지막 동점도 문자열에만 의존하는 고정 규칙으로 결정합니다. **선택한 하나의 발음 조합**을 첫/마지막 토큰과 잔여 문맥·모음·존재 여부 계산에 모두 사용합니다. 따라서 잔여 구간을 다른 발음에서 가져오는 일이 없습니다.

| 특징 | 의미 |
|---|---|
| `name_ch_token_context_alignment` | 순서를 유지하는 일대일 토큰 정렬 |
| `name_ch_token_context_weakest` | 정렬에서 가장 약한 토큰의 점수 |
| `name_ch_token_context_first` | 첫 토큰 비교 |
| `name_ch_token_context_last` | 마지막 토큰 비교 |
| `name_ch_token_residual_context_ratio` | 공통 경계 단어를 제외한 나머지 문맥 비교 |
| `name_ch_token_residual_vowel_ratio` | 같은 잔여 구간의 모음 포함 표현 비교 |
| `name_ch_token_residual_present` | 선택한 조합에 잔여 비교 구간이 존재하는지 |

숫자 토큰은 1과 11을 부분 일치로 취급하지 않습니다. 한쪽의 중복 단어가 반대편 단어를 재사용하지 않습니다. CH 대체 표현이 없는 쌍은 기존 대응 7개 특징과 같은 값을 냅니다. 새 열은 이런 쌍에서도 중복 신호를 추가하므로 전체 재학습 결과가 완전히 같다는 뜻은 아닙니다. 특징 입력은 현재 질의 이름과 고정 카탈로그 이름뿐입니다.

## 캐시 보존과 비교

입력은 이미 완료된 `runs/ranker_data_token_phonetic_foldN`입니다. 원본 70열 캐시를 바로 넣거나 CH 변환을 두 번 적용하면 거부합니다. `prepare-ranker`를 다시 실행할 필요가 없습니다.

- 원본 70열 → 이름 87열 → CH 94열의 두 단계 파생 관계를 검사합니다.
- 기존 87열, 후보 ID·순서, 정답·그룹·언어, OOF 통계와 어려운 오답 샘플을 보존합니다.
- 파일 해시와 공통 배열을 확인한 뒤 완료 메타데이터를 기록합니다. 미완료 캐시는 학습할 수 없습니다.
- 비교는 직접 부모인 86개 모델과 새 93개 모델을 연결하며, 다른 분할·후보·선택 특징·학습 설정 변경을 허용하지 않습니다.
- 모델 복원은 원래 특징→이름 특징→CH 특징을 같은 순서로 계산합니다. 실제 예측에는 훈련 캐시가 필요하지 않습니다.

캐시 변환·학습·비교에는 원본 70열과 이름 87열 캐시도 필요합니다. 파일 검증으로 디스크 I/O가 발생하므로 변환·학습 명령이 출력 없이 잠시 검증 중일 수 있습니다. 94열 캐시를 별도 저장할 공간도 확보합니다. 기존 캐시·모델을 삭제하거나 수정하지 않습니다.

## 완료된 fold1·fold2 실행 기록

아래 명령은 사용자가 완료한 재현 기록입니다. 현재 완료된 출력 경로에 다시 실행하지 않습니다. 새 환경에서 해당 출력 경로가 없을 때만 재현에 사용합니다.

```powershell
cd C:\dacon\port

& {
    foreach ($fold in @(1, 2)) {
        .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --transform ch-variants --data "runs/ranker_data_token_phonetic_fold$fold" --output "runs/ranker_data_ch_variants_fold$fold"
        if ($LASTEXITCODE -ne 0) { throw "fold$fold CH 특징 변환 실패" }

        .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_ch_variants_fold$fold" --output "artifacts/ranker_ch_variants_fold$fold" --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
        if ($LASTEXITCODE -ne 0) { throw "fold$fold CH 모델 학습 실패" }

        .\.venv\Scripts\python.exe -m portmatch compare-rankers --baseline "artifacts/ranker_token_phonetic_fold$fold" --experiment "artifacts/ranker_ch_variants_fold$fold" --output "runs/compare_ch_variants_fold$fold"
        if ($LASTEXITCODE -ne 0) { throw "fold$fold CH 모델 비교 실패" }
    }
}
```

변환이 완료됐는데 학습부터 실패했다면 `train-ranker`부터 실행합니다. 학습까지 완료됐으면 `compare-rankers`만 실행합니다. 완료 경로에 전체 블록을 다시 실행하면 `FileExistsError`가 발생합니다. 실패한 출력 폴더는 보존하고 새 이름을 사용하며 이후 명령의 참조 경로도 맞춥니다. `--exclude-features`를 생략하면 94개 학습이 되어 의도한 93개 비교가 아닙니다.

결과는 `runs/compare_ch_variants_fold1/`과 `runs/compare_ch_variants_fold2/`의 `report.md`, `row_changes.csv`, `vessel_changes.csv`입니다. 검토 결과 현재 93개 설정의 나머지 fold 확대는 권하지 않습니다.

## 나머지 fold와 전체 합산 — 현재 미실행·확대 비권고

다음 명령은 처음 계획한 재현 방법으로 보존합니다. 현재 결과에서는 실행하지 않습니다. 향후 같은 설정의 전체 비교를 별도로 진행하기로 결정한다면 위 블록의 `@(1, 2)`를 `@(0, 3, 4)`로 바꿔 실행하고, 다섯 fold 모델이 모두 완료된 뒤 아래 합산을 실행할 수 있습니다.

```powershell
& {
    $baselineModels = @(0..4 | ForEach-Object { "artifacts/ranker_token_phonetic_fold$_" })
    $chModels = @(0..4 | ForEach-Object { "artifacts/ranker_ch_variants_fold$_" })
    .\.venv\Scripts\python.exe -m portmatch compare-rankers --baseline $baselineModels --experiment $chModels --output runs/compare_ch_variants_5fold
    if ($LASTEXITCODE -ne 0) { throw "CH 5-fold 비교 실패" }
}
```

판단에서는 전체 가중 Top-1과 한글·비한글 정답 수, 개선/신규 오답의 행 수·선박 수, fold 편차를 함께 봅니다. Glovis Chorus만 고치고 다른 선박을 잃는지, 기존 개선 38행을 유지하는지 확인합니다. 한글 CH의 J/K 두 표현은 일반 규칙이며 더 많은 이름을 비슷하게 만들 가능성도 있습니다. 모든 관측 fold는 개발용입니다.

## 검증과 구현 범위

전체 144개 테스트 통과. 새 합성 테스트 25개는 발음쌍·잔여 경계 일관성, 빈 값·숫자·중복 토큰, 원본 87열 보존, 부모/원본 변조 거부, 저장·복원과 캐시 점수 일치, 입력 순서·배치 독립성, 캐시 없이 추론을 확인합니다. 새 CH 특징을 모두 제외한 86개 통제 모델은 작은 합성 데이터에서 기존 86개 모델의 예측 점수를 정확히 재현했습니다.

기존 특징 소스 7개는 바이트 단위로 유지됐고, 실제 저장된 86개 모델 5개의 코드·스키마 호환성을 확인했습니다. 이후 사용자가 fold1·fold2 본 학습을 완료했고, 이번 분석은 변경·핵심 사례 13행을 재추론해 저장 결과와 일치를 확인했습니다. 새 교체 대조와 비한글 신뢰도 실험은 미구현입니다. 86개 첫 제출 패키지의 현재 상태는 [제출 준비 상태](../submission-readiness.md)에 있습니다.
