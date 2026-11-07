# 훈련 GT 결측 증강 — robust 검색·86개 특징 고정

> 공개용 정리본: 실제 선박별 입력·정답·예측·점수와 행 식별자는 제외했다. 아래 성능표는 실제 실험의 집계 수치이며, 오류의 원리는 별도의 [가상 예시](../fictional-examples.md)로 설명한다.

후속 상태(2026-10-26): **원본 질량0.75 전체 5-fold도 실행·분석을 완료했으며 미채택**했습니다. 현재 사용자 보고 서버 최고인 **robust·86개·0.5 증강 fold0(0.8424607623)**을 유지합니다. [재가중치 최종 결과](gt-reweighting-5fold.md). 아래 0.5 구현과 10월 25일 분석 수치는 보존하며, 이후 실행 명령은 모두 완료한 재현 기록입니다.

2026-10-25. **구현과 실제 전체5-fold 특징 생성·학습·비교·분석을 완료했습니다.** 아래는 구현 설계와 완료한 재현 기록입니다. 원본98.7385%(기존 대비−0.1682%p)로 기준 교체를 보류하되 변형 개선은 모든 fold에서 확인했습니다. 후속 [원본질량0.75 캐시 재가중치·직접 비교](gt-reweighting.md)는 구현했고, 당시 fold1 결과에서는 채택을 보류하고 나머지 fold 검증을 진행하기로 했습니다. [fold1 당시 결과·완료한 후속 명령](gt-reweighting-fold1.md). [0.5 전체5-fold 분석·판단](gt-missing-5fold.md), [fold1·2 당시 결과와 완료한 명령](gt-missing-fold12.md). 기존86개 기준·첫 ZIP은 유지합니다.

[직전 robust 재학습 분석](robust-retrained-fold0.md)에서 원본 가중 정확도 98.0291%→97.2514%, 국제 GT 결측 98.0291%→91.0159%로 퇴행했습니다. 반면 fold0 훈련 한글 1,011행은 모두 두 GT가 있으며, 전체 훈련에도 한쪽 GT만 없는 입력은 없습니다. 이번에는 검색·이름 특징을 고정하고 **훈련 입력의 결측과 원본별 가중치 관리**를 함께 적용합니다.

## 변경 범위

| 항목 | 이번 설정 |
|---|---|
| 검색 | `robust`, 이름/음차/편집 80/80/40, 문맥/모음/문맥편집 160/160/80, GT 범위 ±10% |
| 특징 | 원본 70열 → token-phonetic 87열 → `retrieval_tonnage_rank` 제외 **86개** |
| 분할 | 외부 선박 5-fold의 fold0, 내부 mixed 3-fold, seed 20261022 |
| 훈련 입력 | 원본 + 두 GT 결측 + 국제 GT만 결측 + 일반 GT만 결측 |
| 샘플링·학습 | 정답 1 + 어려운 오답 최대 96 + 무작위 오답 최대 32, 300회·6스레드 |
| 검증 | 원본 검증 캐시 유지. 별도 비교에서 동일한 다섯 입력 조건 적용 |

`--gt-augmentation missing`은 `--retrieval-profile robust`에서만 허용합니다. 기본값은 계속 `none`이며 기존 캐시·모델의 동작을 유지합니다. 증강 모델 학습에는 정확한 86개 특징을 요구하므로 원본 70열, 선택하지 않은 87열, CH·동점 보정 변환이나 임의 특징 제거는 거부합니다. GT ×1.25는 **평가만** 하며 이번 훈련에는 큰 오차 증강을 넣지 않습니다.

## 원본과 누수 없는 분할

1. 원본 행으로 외부·내부 분할을 먼저 결정합니다. 원본과 모든 증강본은 동일한 `sample_id`와 같은 내부 OOF view를 공유합니다.
2. 조회 사전·별칭·국적·선종 통계는 해당 fitting 부분의 **원본 행만** 사용합니다. 외부 검증 행과 해당 내부 제외 행은 들어가지 않으며, 선박 분리 view에서는 해당 선박 전체를 제외합니다. 내부 row view에는 같은 선박의 다른 훈련 입항 기록이 있을 수 있습니다.
3. 원본은 그대로 처리하고, 각 결측 변형으로 후보·특징·어려운 오답을 새로 계산합니다. 기존 특징 행의 GT 열만 지우지 않습니다. 이름 특징 변환은 GT가 아닌 원본 이름과 후보 이름만 사용하므로 기존 17개 특징 구현을 그대로 씁니다.
4. 이미 결측이라 달라지지 않는 변형은 `skipped_noop`, 같은 source/view에서 앞 변형과 동일한 입력은 `skipped_duplicate`로 제외합니다. 정답이 검색되지 않거나 오답 후보가 없으면 기존 정책대로 해당 그룹을 제외·기록합니다. 정답을 강제로 추가하지 않습니다.
5. 검증은 원본 입력과 전체 후보를 유지하며 증강하지 않습니다. 원본 그룹을 유지한다는 것은 **항상 원본을 시도하고 기존 그룹 제외 정책을 그대로 적용**한다는 뜻입니다.

실제 훈련의 변형 수가 적은 원본, 정답 후보를 찾지 못한 변형, 한 view가 통째로 제외된 원본도 가중치 감사에 포함합니다. 모든 view가 제외된 원본은 학습할 수 없어 `dropped_sources`에 남습니다.

## 원본별 총 가중치

증강 개수가 많은 원본이 더 큰 비중을 차지하지 않도록 **보존된 원본 표본마다 원시 질량 합을 1**로 맞춥니다. 이는 선박별 균등화가 아니라 원본 행별 균등화입니다.

- 남아 있는 OOF view에 균등 배분합니다. mixed의 두 view가 모두 남으면 각각 0.5입니다.
- 각 view 안에서 원본과 변형이 모두 남으면 `--original-mass 0.5`를 원본에, 나머지 0.5를 남은 서로 다른 변형들에 균등 배분합니다.
- 원본만 남으면 해당 view 전부를 원본에, 변형만 남으면 전부를 남은 변형에 배분합니다. 따라서 제외가 있으면 실제 전체 원본 비율이 정확히 50%일 필요는 없습니다.

두 view와 세 변형이 전부 남은 원본 한 건의 예시는 다음과 같습니다.

| 그룹 | vessel view | row view | 합 |
|---|---:|---:|---:|
| 원본 | 1/4 | 1/4 | 1/2 |
| 두 GT 결측 | 1/12 | 1/12 | 1/6 |
| 국제 GT 결측 | 1/12 | 1/12 | 1/6 |
| 일반 GT 결측 | 1/12 | 1/12 | 1/6 |
| 전체 | 1/2 | 1/2 | **1** |

이 원시 질량을 언어 안에서 정규화하고 한글/비한글 0.7/0.3을 적용합니다. 그룹의 후보 수로 나눠 LightGBM 행별 `sample_weight`에 전달하고 최종 행 가중치 평균을 1로 맞춥니다. 한 언어만 있는 작은 테스트는 존재하는 언어로 재정규화합니다. **행별 가중치 합을 관리하는 학습 휴리스틱이며 LambdaRank의 원본별 실제 gradient 기여가 정확히 같다는 의미는 아닙니다.** 증강하지 않는 기존 모델은 기존 가중치 계산 경로를 그대로 사용합니다. 기존 모델과의 차이는 결측 입력 추가 및 이 source/view 질량 관리가 합쳐진 효과입니다.

## 새 캐시·모델의 감사 정보

| 위치 | 기록·검사 |
|---|---|
| `metadata.json`의 `training_augmentation` | 마스크 순서·비율·제외/질량 정책·생성 코드 해시 |
| `training_augmentation_diagnostics` | 변형별 저장/제외 사유 수, 원본·view 보존 수, 원본 총질량 최소/최대, 언어별·변형별 질량 합 |
| `training_augmentation_assets` | `partitions.csv`, `querylog.csv`, 전체 훈련 그룹 메타데이터의 SHA-256 |
| 훈련 NPZ의 `augmentation_variant`, `query_mass` | 그룹별 변형 이름·최종 질량. 검증 NPZ에는 추가하지 않음 |
| `querylog.csv` | 원본 ID·view·내부 fold·변형·후보 수·정답 포함 여부·제외 이유 |
| 모델의 `training_summary.json` → `training_weighting` | 실제 fit에 전달한 질량 합·범위·언어 예산, 그룹·후보쌍 수·가중치 코드 해시 |

캐시 로드는 그룹 전체의 source/view 질량, 원본의 언어·정답 일관성, 합계와 기록을 검증합니다. 70→87열 변환은 새 질량 배열까지 보존·검증합니다. 모델 복원은 증강 설정·가중치 기록·특징 순서를 확인합니다. 기존 완료 캐시와 제출 ZIP은 수정하지 않습니다.

## VS Code PowerShell 실행

**아래 fold0 네 단계는 완료한 실행 기록입니다.** [fold3·4 대조군+증강 비교](gt-missing-fold12.md#vs-code-fold34의-대조군과-증강-모델-비교)도 실행·분석을 완료했습니다. 원본질량0.75의 fold1과 [fold0·2·3·4 명령](gt-reweighting-fold1.md#판단과-다음-실행)도 모두 실행·분석을 완료했습니다. 추가 실행할 fold는 없습니다.

새 경로 네 개는 이번 실험 전용입니다. **각 단계가 완료됐으면 그다음 단계부터 실행**합니다. 중단된 출력 폴더에 자동 이어쓰기를 지원하지 않으므로 같은 경로를 덮어쓰지 않습니다. 별도 재시도는 새 이름을 사용하고 후속 입력 경로도 함께 바꿉니다.

```powershell
cd C:\dacon\port

& {
    .\.venv\Scripts\python.exe -m portmatch prepare-ranker --fold 0 --retrieval-profile robust --gt-augmentation missing --original-mass 0.5 --output runs/ranker_data_robust_gt_missing_fold0
    if ($LASTEXITCODE -ne 0) { throw "GT 결측 증강 OOF 특징 생성 실패" }

    .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --transform token-phonetic --data runs/ranker_data_robust_gt_missing_fold0 --output runs/ranker_data_robust_gt_missing_token_phonetic_fold0
    if ($LASTEXITCODE -ne 0) { throw "토큰·문맥 특징 변환 실패" }

    .\.venv\Scripts\python.exe -m portmatch train-ranker --data runs/ranker_data_robust_gt_missing_token_phonetic_fold0 --output artifacts/ranker_robust_gt_missing_token_phonetic_fold0 --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
    if ($LASTEXITCODE -ne 0) { throw "증강 86개 모델 학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch compare-retrieval --baseline artifacts/ranker_token_phonetic_fold0 --profiles robust --reference-model artifacts/ranker_robust_token_phonetic_fold0 --experiment-model artifacts/ranker_robust_gt_missing_token_phonetic_fold0 --output runs/compare_robust_gt_missing_fold0
    if ($LASTEXITCODE -ne 0) { throw "GT 결측 증강 모델 비교 실패" }
}
```

한 번에 실행하면 특징 생성→변환→학습→비교 순서로 진행하며 실패 시 멈춥니다. 다른 특징 생성·학습과 동시에 비교를 돌리지 않습니다. 새 70열·87열 캐시와 LightGBM 학습에 추가 디스크·메모리가 필요합니다. 현재 fold0 입력 분포에서는 원본 19,540그룹 시도에 변형 최대 54,864그룹 시도가 추가될 수 있습니다. 실제 저장량은 검색 실패·오답 후보 수 등에 따라 줄며 시간 증가율과 같지 않습니다.

## fold0 비교 결과 읽는 법

| 방법 | 검색 | 학습 |
|---|---|---|
| `baseline` | 기존 | 기존 86개 |
| `robust` | robust | 기존 86개 가중치 유지 |
| `reference` | robust | 직전 증강 없는 robust 86개 재학습 |
| `trained` | robust | 이번 GT 결측 증강·질량 관리 86개 학습 |

원본, 두 GT 결측, 국제 GT 결측, 일반 GT 결측, 국제 GT ×1.25를 모든 방법에 적용합니다. fold0 전체는 **2,452행 × 5조건 × 4방법 = 49,040회** 추론입니다. 입력 해시·외부/내부 분할·seed·검색 설정·훈련 사전·범주 통계·특징 순서·학습 설정이 맞아야 합니다. 증강 모델 비교에는 `--reference-model`이 필수입니다. baseline/reference/trained의 원본 예측은 각 저장 검증 결과와 다시 대조합니다.

- `summary.json`, `report.md`: 언어별 정확도·후보 재현율·후보 수·단계별 시간과 개선/신규 오답.
- `row_changes.csv`, `vessel_changes.csv`: 같은 입력의 기존 baseline 대비 변화.
- **`trained_vs_reference_row_changes.csv`, `trained_vs_reference_vessel_changes.csv`**: 이번 증강 모델과 증강 없는 robust 모델의 직접 변화.
- **`summary.json → trained_vs_reference`**: 조건별 정확도 차이·전이·후보 동일성. 두 모델의 후보가 달라지면 실패하므로 검색 변경과 증강 효과를 혼동하지 않습니다.

증강만으로 후보에 없는 정답을 복원할 수는 없습니다. 조건별 평균을 공식 점수로 해석하지 않으며, 이미 반복 사용한 공개 개발 fold의 평가입니다. 실제 추론 시간도 로컬 조건 비교용이며 서버 실행 시간으로 환산하지 않습니다. fold0 당시에는 같은 설정의 fold1·2부터 진행하기로 판단했습니다. 이후 fold3·4도 완료해 변형 개선을 전체 구간에서 확인했습니다. 전체 원본 한글3건 순손실로 당시0.5 설정의 기준 교체는 보류하고 원본 질량0.75 통제 실험을 제안했습니다. 이후 0.75 전체 검증은 미채택으로 종료됐고, 0.5는 사용자 보고 서버 최고 제출로 유지합니다. [최종 판단](gt-reweighting-5fold.md).

## 구현 검증

**전체 회귀 테스트 227개를 통과했습니다.** 작은 합성 데이터로 원본·검증 특징/후보 보존, 분할/훈련 사전 누수 방지, 변형별 새 검색, 제외·중복 처리, 원본별 질량 재분배와 변조 거부, 실제 LightGBM 가중 학습·모델 복원·네 방법 비교를 검증했습니다. 합성 24행 통합 검사에서 네 방법 × 다섯 조건 × 검증 8행의 160개 추론과 직접 비교 산출물을 확인했습니다. 기존 제출 ZIP의 별도 프로세스 추론·재현 테스트도 포함합니다.

구현 당시 실제 기존 `ranker_token_phonetic_fold0`와 `ranker_robust_token_phonetic_fold0`가 각각 86개 특징으로 로드되는 것을 확인했습니다. 보호한 검색·특징 파일 11개의 바이트와 첫 제출 ZIP의 SHA-256도 같았습니다. 로컬 근거는 `work/gt_missing_full_tests.log`, `work/gt_missing_compatibility_audit.json`입니다. 이후 사용자가 실제 fold0 실행을 완료했고 `work/gt_missing_fold0_analysis/`에서 저장 결과·훈련 질량·분할을 감사했습니다. 위 호환성 로그의 출력 폴더 부재 기록은 구현 당시 상태입니다.
