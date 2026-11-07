# GT 결측 증강 fold0 — 강건성 개선, 기준 교체 보류

> 공개용 정리본: 실제 선박별 입력·정답·예측·점수와 행 식별자는 제외했다. 아래 성능표는 실제 실험의 집계 수치이며, 오류의 원리는 별도의 [가상 예시](../fictional-examples.md)로 설명한다.

> 2026-10-25 후속: 전체5-fold 분석을 완료했습니다. 원본 한글3건 순손실로 기준 교체는 보류하고 원본질량0.75 단일 실험을 제안합니다. [최신 결과·계획](gt-missing-5fold.md). 아래는 fold0 당시 판단과 완료한 실행 기록입니다.

2026-10-24. **기존 86개 개발 기준·제출 모델 교체는 보류합니다. 증강 방향은 유지하고, 같은 설정으로 fold1·2를 먼저 검증합니다. fold3·4 확대는 그 결과 이후 판단합니다.** 원본 보존 목표는 미달했지만 국제 GT 결측·오차 개선이 여러 선박에서 나타났으며, 증강 없는 robust 모델보다 원본 신규 오답은 늘지 않았습니다.

분석 대상은 `runs/compare_robust_gt_missing_fold0`의 전체 **2,452행 × 5입력 × 4방법 = 49,040개 예측**입니다. 한글 180행, 비한글 2,272행입니다. 이번 작업은 저장 결과와 학습 산출물 분석이며 추가 학습·전체 재추론을 하지 않았습니다.

## 비교 무결성

독립 재계산 **1,328/1,328 항목 통과**, 불일치가 없었습니다.

- 언어별 Top-1·Recall@K·후보 재현율, 후보 분포·시간, 행·선박별 개선/신규 오답을 다시 계산했습니다.
- `robust`, `reference`, `trained`의 후보 ID 집합 해시·개수·정답 포함 여부가 동일합니다. 두 비교의 합계 24,520쌍을 확인했습니다.
- 저장 모델 3개의 원본 예측 7,356행은 각 `validation_predictions.csv`와 점수까지 일치합니다.
- 직전 `compare_robust_retrained_fold0`와 공통 3방법의 36,780행도 예측·후보·점수가 같습니다.
- 데이터·모델·인덱스·메타데이터·비교 코드 해시, 86개 특징 순서, 분할·학습·검색 설정, 훈련 사전·범주 통계를 확인했습니다.
- 실제 사용하는 입력 필드와 모델로 묶은 25,200개 입력 그룹에서 동일 입력의 예측·후보 불일치가 없었습니다. 이 중 10,504그룹은 반복 입력입니다. 이는 저장 실행의 일관성 점검이며 별도의 역순·단독 재추론 실험은 아닙니다.

`baseline`은 기존 검색·86개 모델, `robust`는 기존 가중치에 검색만 변경, `reference`는 증강 없는 robust 재학습, `trained`는 이번 결측 증강·원본별 질량 관리 모델입니다.

## 정확도와 조건별 변화

가중 정확도는 한글 70% + 비한글 30%입니다. 서로 다른 조건의 평균을 공식 점수로 만들지 않습니다.

| 입력 | 기존 86개 | 검색만 robust | 증강 없는 robust | 증강 모델 | 증강 − 기존 (%p) | 증강 − 증강 전 (%p) |
|---|---:|---:|---:|---:|---:|---:|
| 원본 | 98.0291% | 98.0291% | 97.2514% | **97.2646%** | −0.7646 | +0.0132 |
| 두 GT 결측 | 75.4604% | 92.5715% | 88.6562% | **91.0159%** | +15.5556 | +2.3597 |
| 국제 GT 결측 | 98.0291% | 94.9180% | 91.0159% | **96.8625%** | −1.1667 | +5.8465 |
| 일반 GT 결측 | 96.8625% | 96.8625% | 97.6403% | **97.2646%** | +0.4021 | −0.3757 |
| 국제 GT ×1.25 | 68.2478% | 81.8530% | 82.3415% | **95.7090%** | +27.4612 | +13.3675 |

| 입력 | 한글 정답: 기존 / 증강 전 / 증강 | 비한글 정답: 기존 / 증강 전 / 증강 |
|---|---:|---:|
| 원본 | 175 / 173 / **173** | 2,270 / 2,270 / **2,271** |
| 두 GT 결측 | 117 / 151 / **157** | 2,269 / 2,267 / **2,269** |
| 국제 GT 결측 | 175 / 157 / **172** | 2,270 / 2,269 / **2,270** |
| 일반 GT 결측 | 172 / 174 / **173** | 2,270 / 2,270 / **2,271** |
| 국제 GT ×1.25 | 101 / 135 / **169** | 2,194 / 2,260 / **2,271** |

분모는 모든 조건에서 한글 180·비한글 2,272입니다. 증강 전 대비 ×1.25의 한글 순증 34행·비한글 순증 11행이 큰 개선을 설명합니다. 같은 검색의 후보가 같으므로 **정답을 새로 검색한 효과가 아니라 순위 학습 변화**입니다. 단, 결측 입력 추가와 원본별 가중치 관리가 함께 적용됐으므로 각각의 단독 효과까지 분리한 실험은 아닙니다.

| 입력 | 기존 86개 대비 개선 / 신규 (행·척) | 증강 없는 robust 대비 개선 / 신규 (행·척) |
|---|---|---|
| 원본 | 1행·1척 / 2행·1척 | 1행·1척 / 0 |
| 두 GT 결측 | 44행·13척 / 4행·1척 | 15행·5척 / 7행·2척 |
| 국제 GT 결측 | 0 / 3행·1척 | 17행·6척 / 1행·1척 |
| 일반 GT 결측 | 2행·2척 / 0 | 2행·2척 / 2행·1척 |
| 국제 GT ×1.25 | 145행·58척 / 0 | 47행·19척 / 2행·1척 |

동일 선박·원본 행이 여러 조건에 반복되므로 조건별 개선 수를 합해 독립 사례 수로 해석하지 않습니다.

## 후보 수와 실제 추론 시간

| 입력 | 증강 후보 평균 / p95 / 최대 | 기존 86개 초 | 증강 전 robust 초 | 증강 초 |
|---|---:|---:|---:|---:|
| 원본 | 2,656.8 / 4,707.25 / 7,082 | 273.82 | 292.24 | 291.37 |
| 두 GT 결측 | 174.0 / 362 / 421 | 147.09 | 166.18 | 166.84 |
| 국제 GT 결측 | 2,634.5 / 4,698.45 / 5,010 | 281.13 | 298.52 | 297.93 |
| 일반 GT 결측 | 2,632.4 / 4,694 / 5,010 | 273.38 | 290.93 | 290.19 |
| 국제 GT ×1.25 | 4,830.5 / 8,076 / 8,718 | 221.23 | 341.76 | 340.99 |

증강 전 대비 실행 시간 차이는 −0.30%~+0.40%로 비슷하며 빨라졌다고 단정할 수준은 아닙니다. 기존 86개 대비 원본 +6.4%, ×1.25 +54.1%의 비용은 검색 확장에서 발생합니다. 전체 평균 외에 한글 처리 비용도 봐야 합니다. 원본 한글의 기존 검색은 14.72초, 검색만 robust는 31.82초입니다.

전체 비교 5,433.73초(90분 34초)는 **49,040회** 추론과 초기화·출력의 합계입니다. 표의 행 추론 시간은 모델/인덱스 로드·고정 이름 특징 준비·리포트 IO를 제외합니다. 실행 순서를 순환해도 문자열·운영체제 캐시 영향이 남으며 제출 서버의 실행 시간으로 환산하지 않습니다.

## 증강·가중치가 실제로 적용됐는가

저장 훈련 그룹의 메타데이터 배열을 직접 확인했습니다. **원본 9,770건 모두 보존**, 각 원본 질량 합 최소·최대가 모두 1이며 전체 합은 9,770입니다. 외부 선박·입력 분할과 내부 6개 fitting/held-out 기록이 일치하고, 검증은 원본만 남았습니다. 사전·범주 통계도 증강 전과 같습니다.

원본 검증 NPZ 39개는 증강 전후 바이트까지 동일합니다.

| 변형 | 저장 그룹 | 원시 질량 합 |
|---|---:|---:|
| 원본 | 19,540 | 5,198.00 |
| 두 GT 결측 | 18,246 | 1,520.50 |
| 국제 GT 결측 | 18,288 | 1,525.75 |
| 일반 GT 결측 | 18,288 | 1,525.75 |
| 전체 | **74,362** | **9,770.00** |

전체 후보쌍은 2,520,095 → **9,564,947**, 약 3.80배입니다. 원본 그룹·후보쌍 수는 증강 전과 같습니다. 이미 두 GT가 없는 원본 626건의 1,252개 view에서는 모든 마스크가 no-op입니다. 두 GT 마스크 42그룹은 정답 검색 실패로 제외됐습니다(한글 32·비한글 10, 원본 33건·15척). 원본이나 view 전체가 탈락한 사례는 없습니다. 질량 재분배로 전체 원본 비율은 53.2037%이며, 모든 원본이 무조건 50%라는 뜻은 아닙니다. 한글은 원본 질량 505.5/1,011 = 50%입니다.

| 단계 | 증강 없는 robust | 증강 |
|---|---:|---:|
| OOF 특징 생성 | 26분 39초 | 90분 36초 |
| 이름 특징 변환 | 2분 40초 | 5분 02초 |
| 학습 명령 전체 | 45초 | 2분 31초 |

현재 fold0의 원본+파생 캐시+모델은 증강 약 1.23GB, 증강 전 대조군까지 합하면 약 2.00GB입니다. 이것은 디스크 산출물 크기이며 학습 최대 RAM 측정치는 아닙니다. 다른 fold의 시간·메모리를 보장하지 않습니다.

fold0에서 대조군·증강의 생성/변환/학습과 네 방법 비교를 합친 기록은 약 3시간 39분입니다. fold1·2는 두 대조군도 새로 만들어야 하므로 장시간 작업이며, 이 수치를 그대로 실행 시간 보장으로 사용하지 않습니다.

## 채택과 다음 실행

| 사전에 정한 목표 | 실제 | 판단 |
|---|---:|---|
| 원본 한글 ≥175 | 173 | 미달 |
| 원본 비한글 ≥2,270 | 2,271 | 충족 |
| 국제 GT 결측 한글 175 회복 | 172 | 미달 |
| 두 GT 결측 한글 ≥161 | 157 | 미달 |
| ×1.25 한글 ≥135·비한글 ≥2,260 | 169·2,271 | 충족·큰 개선 |

**정식 채택 조건을 통과했다고 보지 않습니다.** 기존 86개 기준과 첫 제출 ZIP은 유지합니다. 다만 증강 없는 robust 모델보다 원본 신규 오답 0, 국제 GT 결측 17행 회복·1행 신규, ×1.25 47행·19척 회복·2행 신규를 확인했으므로 이 방향을 중단할 근거도 부족합니다.

당시 fold1·2에는 기존 86개 모델만 있고 **증강 없는 robust 대조군이 없었습니다**. 따라서 아래 명령은 각 fold의 대조군도 생성합니다. fold0 완료 산출물은 건드리지 않으며, 학습·비교는 사용자가 VS Code에서 실행합니다.

## 완료 기록: fold1·2의 대조군과 증강 모델 비교

새 경로를 사용하는 첫 실행용입니다. 완료한 단계는 건너뛰고 다음 단계부터 실행합니다. 실패하면 즉시 멈추며 중단 폴더 자동 이어쓰기는 지원하지 않습니다. 비교 시간 측정을 위해 다른 학습과 동시에 실행하지 않습니다.

```powershell
cd C:\dacon\port

foreach ($fold in 1, 2) {
    .\.venv\Scripts\python.exe -m portmatch prepare-ranker --fold $fold --retrieval-profile robust --output "runs/ranker_data_robust_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 대조군 특징 생성 실패" }

    .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --transform token-phonetic --data "runs/ranker_data_robust_fold$fold" --output "runs/ranker_data_robust_token_phonetic_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 대조군 특징 변환 실패" }

    .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_robust_token_phonetic_fold$fold" --output "artifacts/ranker_robust_token_phonetic_fold$fold" --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 대조군 학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch prepare-ranker --fold $fold --retrieval-profile robust --gt-augmentation missing --original-mass 0.5 --output "runs/ranker_data_robust_gt_missing_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 증강 특징 생성 실패" }

    .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --transform token-phonetic --data "runs/ranker_data_robust_gt_missing_fold$fold" --output "runs/ranker_data_robust_gt_missing_token_phonetic_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 증강 특징 변환 실패" }

    .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_robust_gt_missing_token_phonetic_fold$fold" --output "artifacts/ranker_robust_gt_missing_token_phonetic_fold$fold" --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 증강 학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch compare-retrieval --baseline "artifacts/ranker_token_phonetic_fold$fold" --profiles robust --reference-model "artifacts/ranker_robust_token_phonetic_fold$fold" --experiment-model "artifacts/ranker_robust_gt_missing_token_phonetic_fold$fold" --output "runs/compare_robust_gt_missing_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold${fold}: 네 방법 비교 실패" }
}
```

결과 폴더는 `runs/compare_robust_gt_missing_fold1`, `runs/compare_robust_gt_missing_fold2`입니다. 사용자가 실행을 완료했고 2026-10-25 분석을 마쳤습니다. 같은 경로에 다시 실행하지 않습니다. 원본/결측 강건성 개발 검증은 Public 개선 보장이 아니며, 전체 fold도 이전 개발에 이미 사용한 구간입니다. 새 모델의 제출 ZIP·서버 시간 검증은 별도 단계입니다.

## 로컬 근거

- `runs/compare_robust_gt_missing_fold0/`: 원본 리포트·49,040예측·baseline 및 reference 상대 변화.
- `work/gt_missing_fold0_analysis/audit/`: 지표/시간/후보·모델/입력 무결성 1,328개 검사와 재계산 스크립트.
- `work/gt_missing_fold0_analysis/errors/`: 전체 선박·언어별 개선/신규/잔존 오류, 동일 입력 일관성 분석.
- `work/gt_missing_fold0_analysis/training/`: 원본별 질량 배열, 내부 분할·증강 제외·시간·디스크·중요도 감사.

학습 구현 시점의 회귀 테스트는 227개 통과했습니다. 이번에는 실행 코드를 바꾸지 않았으며 저장 산출물을 감사했습니다. 원본 데이터·캐시·모델·행별 분석 파일·ZIP은 GitHub에 업로드하지 않습니다.
