# GT 결측 증강 fold1·2 — 개선 재현, 채택 보류·fold3·4 진행

> 공개용 정리본: 실제 선박별 입력·정답·예측·점수와 행 식별자는 제외했다. 아래 성능표는 실제 실험의 집계 수치이며, 오류의 원리는 별도의 [가상 예시](../fictional-examples.md)로 설명한다.

> 2026-10-25 후속: fold3·4도 완료했고 전체5-fold를 분석했습니다. 원본98.7385%(기존 대비−0.1682%p)로 기준 교체는 보류하고 원본질량0.75 단일 실험을 제안합니다. [최신 결과·계획](gt-missing-5fold.md). 아래는 fold1·2 당시 판단과 완료한 실행 기록입니다.

2026-10-25. **기존 86개 개발 기준과 첫 제출 ZIP은 유지하고, 같은 `original_mass=0.5` 설정으로 fold3·4를 진행합니다.** 결측·톤수 오차에 대한 개선은 두 구간에서도 재현됐습니다. 다만 fold1 원본에는 새로운 선박 오류가 있고 fold0~2 합산 원본 점수는 기존보다 낮아 최종 채택 조건을 충족하지 못했습니다.

분석 대상은 `runs/compare_robust_gt_missing_fold1`과 `runs/compare_robust_gt_missing_fold2`의 **4,856행 × 5입력 × 4방법 = 97,120개 저장 예측**입니다. 추가 학습·전체 재추론은 실행하지 않았습니다. [fold0 결과](gt-missing-fold0.md)까지 합친 보조 분석은 7,308행이며 아직 전체 5-fold 결과가 아닙니다.

## 비교 조건과 검증 범위

| 방법 | 검색 | 모델 |
|---|---|---|
| `baseline` | 기존 legacy | 기존 86개 |
| `robust` | robust | 기존 86개 가중치 고정 |
| `reference` | robust | 증강 없이 재학습한 86개 |
| `trained` | robust | GT 결측 증강·원본별 질량 관리를 적용한 86개 |

증강 전후 차이는 **결측 입력 추가와 원본별 질량 정책을 함께 적용한 효과**입니다. 각각의 효과를 따로 분리한 실험은 아닙니다. 국제 GT ×1.25는 평가 변형이며 훈련 증강에는 포함하지 않았습니다.

- 독립 결과 감사 **2,590/2,590 항목 통과**. 입력·코드·모델·인덱스·캐시 메타데이터 해시, 외부/내부 분할, 사전·범주 통계, 정확한 86개 특징 순서·검색·학습 설정을 확인했습니다.
- 같은 robust 검색의 세 방법은 후보 집합 해시·개수·정답 포함 여부가 같습니다. 두 비교의 합계 **48,560쌍**을 대조했습니다. reference 대비 개선은 후보 추가 효과가 아닌 순위 개선입니다.
- 세 저장 모델의 원본 예측 **14,568행**은 각 저장 검증 결과와 점수까지 같습니다.
- fold0·1·2의 증강 설정, 분할 seed, 300회·6스레드 학습 설정이 일치합니다.

## 정확도: 각 fold와 합산

가중 Top-1은 한글 70% + 비한글 30%입니다. 합산은 언어별 정답 수·전체 수를 모아 계산하며 fold 점수나 입력 조건을 평균하지 않습니다. H/N은 증강 모델의 한글/비한글 정답 수입니다.

### fold1: 2,406행 — 한글 267, 비한글 2,139

| 입력 | 기존 86개 | 검색만 robust | 증강 없는 robust | 증강 | 증강 H / N |
|---|---:|---:|---:|---:|---:|
| 원본 | 97.8746% | 98.9233% | 98.9233% | **97.8746%** | 259 / 2,137 |
| 두 GT 결측 | 78.9841% | 88.9467% | 73.4785% | **93.6658%** | 243 / 2,136 |
| 국제 GT 결측 | 91.5684% | 95.2388% | 89.7332% | **97.3502%** | 257 / 2,137 |
| 일반 GT 결측 | 97.6124% | 98.9233% | 98.9233% | **97.8746%** | 259 / 2,137 |
| 국제 GT ×1.25 | 67.3838% | 74.0924% | 86.8353% | **97.0881%** | 256 / 2,137 |

원본은 기존과 같은 정확도지만 예측은 같지 않습니다. 증강 없는 robust 대비 원본 **−1.0487%p**이며, 한글 4건·비한글 1건 신규 오류가 있습니다.

### fold2: 2,450행 — 한글 229, 비한글 2,221

| 입력 | 기존 86개 | 검색만 robust | 증강 없는 robust | 증강 | 증강 H / N |
|---|---:|---:|---:|---:|---:|
| 원본 | 99.3616% | 99.3616% | 99.3886% | **99.3886%** | 227 / 2,221 |
| 두 GT 결측 | 82.2707% | 92.6638% | 87.7594% | **96.6376%** | 218 / 2,221 |
| 국제 GT 결측 | 93.8594% | 94.7765% | 96.3319% | **99.3886%** | 227 / 2,221 |
| 일반 GT 결측 | 99.3616% | 99.3616% | 98.4716% | **99.3886%** | 227 / 2,221 |
| 국제 GT ×1.25 | 69.1742% | 68.6759% | 90.1373% | **98.4716%** | 224 / 2,221 |

### fold1+2 합산: 한글 496, 비한글 4,360

| 입력 | 기존 86개 | 증강 없는 robust | 증강 | 기존 대비 %p | 증강 전 대비 %p |
|---|---:|---:|---:|---:|---:|
| 원본 | 98.5612% | 99.1395% | **98.5749%** | +0.0138 | −0.5645 |
| 두 GT 결측 | 80.5036% | 80.0733% | **95.0398%** | +14.5363 | +14.9666 |
| 국제 GT 결측 | 92.6269% | 92.7818% | **98.2927%** | +5.6658 | +5.5109 |
| 일반 GT 결측 | 98.4201% | 98.7161% | **98.5749%** | +0.1549 | −0.1411 |
| 국제 GT ×1.25 | 68.1845% | 88.3586% | **97.7282%** | +29.5437 | +9.3696 |

### fold0+1+2 합산: 한글 676, 비한글 6,632

| 입력 | 기존 86개 | 증강 없는 robust | 증강 | 기존 대비 %p | 증강 전 대비 %p |
|---|---:|---:|---:|---:|---:|
| 원본 | 98.4196% | 98.6358% | **98.2261%** | −0.1935 | −0.4097 |
| 두 GT 결측 | 79.1592% | 82.3557% | **93.9669%** | +14.8077 | +11.6112 |
| 국제 GT 결측 | 94.0660% | 92.3101% | **97.9109%** | +3.8449 | +5.6008 |
| 일반 GT 결측 | 98.0054% | 98.4287% | **98.2261%** | +0.2207 | −0.2026 |
| 국제 GT ×1.25 | 68.1921% | 86.7496% | **97.1906%** | +28.9985 | +10.4410 |

## 후보 누락과 추론 비용

증강 전후 검색은 고정입니다. 두 GT 결측의 한글 후보는 **484/496 = 97.5806%**, 비한글은 4,360/4,360입니다. 다른 네 조건은 두 언어 모두 후보 재현율 100%입니다.

다음은 fold1+2 전체 합계 시간과 robust 후보 분포입니다. 비교기의 질의 처리 시간이며 모델·인덱스 초기 로드 및 보고서 저장 시간은 제외합니다.

| 입력 | robust 후보 평균 / p95 / 최대 | 기존 86개 초 | 증강 전 초 | 증강 초 |
|---|---:|---:|---:|---:|
| 원본 | 2,659.7 / 4,714 / 5,884 | 459.26 | 504.79 | 503.65 |
| 두 GT 결측 | 180.1 / 369 / 435 | 234.58 | 277.91 | 278.49 |
| 국제 GT 결측 | 2,634.0 / 4,700 / 5,046 | 456.01 | 499.39 | 497.91 |
| 일반 GT 결측 | 2,636.2 / 4,700 / 5,046 | 456.60 | 500.72 | 499.83 |
| 국제 GT ×1.25 | 4,846.7 / 8,055 / 8,720 | 440.69 | 688.86 | 688.08 |

같은 검색의 증강 전후 시간 차이는 약 **−0.30%~+0.21%**로 비슷합니다. 기존 대비 원본 +9.67%, ×1.25 +56.14%의 비용은 남습니다. 한글 원본만 보면 33.31→77.82초로 약 2.34배여서 전체 평균만으로 비용을 판단하면 안 됩니다. 로컬 비교 기록이며 서버 시간으로 환산하지 않습니다.

## 훈련 원본·질량 보존과 실행 비용

| fold | 원본 수 | 원본 그룹 / 후보쌍 | 증강 전체 그룹 / 후보쌍 | 동일 원본 검증 NPZ |
|---|---:|---:|---:|---:|
| 1 | 9,816 | 19,632 / 2,532,076 | 74,792 / 9,618,394 | 38 |
| 2 | 9,772 | 19,544 / 2,520,480 | 74,374 / 9,565,234 | 39 |

모든 원본의 실제 저장 질량 합은 1이며 원본·view 손실은 없습니다. 원본 그룹의 query·언어·정답·후보 수·샘플 후보 순서·라벨은 증강 전과 같습니다. 원본 특징 X 전체를 이번에 별도 재비교하지는 않았습니다. 분할 파일 동일, 외부 선박/source 중복 0, 내부 fitting/held-out 일치, 원본 검증 NPZ **77개 바이트 동일**을 확인했습니다.

두 GT 마스크에서 정답 후보를 찾지 못한 fold1 40그룹·fold2 46그룹만 제외했고, 국제만/일반만 마스크는 모두 저장했습니다. 원래 두 GT가 없는 616/626개 원본의 중복 마스크는 제외·재분배됩니다. 따라서 `original_mass=0.5`라도 전체 저장 질량 중 원본 비율은 각각 53.1377%/53.2030%이며, 언어 균형까지 적용한 원본 행가중치 비중은 51.0391%/51.0658%입니다. 정책 위반이 아닙니다.

| fold | 대조군 / 증강 특징 생성 | 대조군 / 증강 이름 변환 | 대조군 / 증강 학습 명령 | 네 방법 비교 | 합계 |
|---|---:|---:|---:|---:|---:|
| 1 | 26분48초 / 84분54초 | 2분35초 / 4분49초 | 40초 / 2분18초 | 79분24초 | 3시간21분28초 |
| 2 | 25분42초 / 83분01초 | 2분35초 / 4분46초 | 40초 / 2분18초 | 80분14초 | 3시간19분16초 |

개별 시간은 반올림되며 합계는 원기록 기준입니다. 학습 명령 시간에는 원본 검증도 포함됩니다. 캐시·모델 디스크는 각각 약 1.991GB/1.998GB이고 보고서·RAM은 별도입니다. 다음 두 fold도 장시간 작업이며 이 기록을 시간 보장으로 사용하지 않습니다.

## 판단과 다음 실험

모든 fold는 이미 개발에 사용한 구간입니다. 공개 검증의 변형 실험은 실제 비공개 입력 분포나 Public 향상을 보장하지 않습니다. 제출 모델을 바꾸려면 별도 패키징·행 독립성·실제 실행 시간 검증이 필요합니다.

## VS Code: fold3·4의 대조군과 증강 모델 비교

아래 명령은 사용자가 실행을 완료했고 전체5-fold 분석까지 마친 기록입니다. 같은 경로에 다시 실행하지 않습니다. 최초 안내 시점에는 두 fold의 기존86개 모델만 있고 새 출력14경로가 없는 것을 확인했습니다. 완료한 단계는 건너뛰고 다음 단계부터 실행하며, 중단 폴더 자동 이어쓰기는 지원하지 않습니다. 비교 시간 측정을 위해 다른 학습과 동시에 실행하지 않습니다. 장시간 실행은 사용자가 VS Code에서 완료했습니다.

```powershell
cd C:\dacon\port

foreach ($fold in 3, 4) {
    .\.venv\Scripts\python.exe -m portmatch prepare-ranker --fold $fold --retrieval-profile robust --output "runs/ranker_data_robust_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 대조군 특징 생성 실패" }

    .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --transform token-phonetic --data "runs/ranker_data_robust_fold$fold" --output "runs/ranker_data_robust_token_phonetic_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 대조군 특징 변환 실패" }

    .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_robust_token_phonetic_fold$fold" --output "artifacts/ranker_robust_token_phonetic_fold$fold" --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 대조군 학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch prepare-ranker --fold $fold --retrieval-profile robust --gt-augmentation missing --original-mass 0.5 --output "runs/ranker_data_robust_gt_missing_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 증강 특징 생성 실패" }

    .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --transform token-phonetic --data "runs/ranker_data_robust_gt_missing_fold$fold" --output "runs/ranker_data_robust_gt_missing_token_phonetic_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 증강 학습용 특징 변환 실패" }

    .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_robust_gt_missing_token_phonetic_fold$fold" --output "artifacts/ranker_robust_gt_missing_token_phonetic_fold$fold" --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 증강 학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch compare-retrieval --baseline "artifacts/ranker_token_phonetic_fold$fold" --profiles robust --reference-model "artifacts/ranker_robust_token_phonetic_fold$fold" --experiment-model "artifacts/ranker_robust_gt_missing_token_phonetic_fold$fold" --output "runs/compare_robust_gt_missing_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 네 방법 비교 실패" }
}
```

완료 후 결과는 `runs/compare_robust_gt_missing_fold3`, `runs/compare_robust_gt_missing_fold4`에 저장됩니다. 기존 86개 대비는 `row_changes.csv`·`vessel_changes.csv`, 증강 전 대비는 `trained_vs_reference_row_changes.csv`·`trained_vs_reference_vessel_changes.csv`를 확인합니다.

## 로컬 분석 근거

- `work/gt_missing_fold12_analysis/audit/`: 두 fold 감사 JSON, 언어별 합산 지표·시간·후보 분석.
- `work/gt_missing_fold12_analysis/errors/`: 모든 선박의 개선·신규·잔존 오답, 원본 입력·카탈로그 대조, 중요 선박 추적.
- `work/gt_missing_fold12_analysis/training/`: 원본 샘플 후보·라벨·질량·검증 보존 감사, 실행 시간·디스크.

위 행별 산출물과 원본 데이터·모델·캐시는 GitHub에 올리지 않고, 요약·판단·재현 명령만 문서화합니다. 구현은 변경하지 않았으므로 회귀 테스트를 새로 실행한 결과로 보고하지 않습니다. [증강 구현 당시 227개 테스트 통과 기록](gt-missing-augmentation.md)은 별도입니다.
