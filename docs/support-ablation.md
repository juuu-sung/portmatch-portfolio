# 지원도 로그 4열 제외: 86개 → 82개 재학습 실험

> 공개용 정리본: 실제 선박별 입력·정답·예측·점수와 행 식별자는 제외했다. 아래 성능표는 실제 실험의 집계 수치이며, 오류의 원리는 별도의 [가상 예시](fictional-examples.md)로 설명한다.

2026-10-30. **기존 GT 결측 증강0.5 캐시를 재사용한82개 학습·비교의 fold0·1·3·4를 완료했습니다. 채택은 보류하고 같은 설정으로 남은 fold2를 검증합니다.** 네 fold의 원본 가중 정확도는86개98.5834%→82개98.7323%, 두 GT 결측은92.9464%→93.8924%지만 국제 GT ×1.25는97.7068%→97.5647%로 낮아졌습니다. 현재 최고 제출은 robust·86개·0.5 fold0의 사용자 보고 Public **0.8424607623**이며 새 제출은 하지 않았습니다. [네 fold의 전체 결과·선박별 판단](experiments/support-ablation-fold0134.md).

통계80% 축소는 전체5-fold에서 희귀 선종 대응 손실과 신규 오답을 만들었습니다. 이번에는 검색과 국적·선종 통계를100%로 유지하고, 모델이 절대 지원도에 의존하는 영향을 따로 확인합니다. [직전 전체5-fold 근거](experiments/category-scope-5fold.md).

구현 당시 전체376개 테스트를 통과했습니다(새22개 포함). 합성 데이터의86개/82개 실제 학습·5조건 비교, 양쪽 저장 예측 일치, 전체 후보의 공통82열 일치와 native 점수 재계산, 기존39개 모델 호환성과3개 ZIP 보존을 확인한 기록입니다. 이번 결과 분석에서는 전체 테스트나 학습·후보 검색을 다시 실행하지 않았습니다.

이번 네 fold의9,772행·97,720개 저장 예측을 감사한 **31,163개 검사 모두 통과**했습니다.9,694개 파일 약5.14GB의 무결성을 확인하고 실제 입력98,232개 벡터로 native 점수를 재계산해 최대 오차1.78×10⁻¹⁵를 확인했습니다. 전체 후보 공통82열 불변127,066,783쌍은 실행 당시 기록·코드와 파일 해시로 확인한 범위이며, 이번 사후 감사에서 전체 후보 특징을 다시 생성하지 않았습니다. 기존86개48,860개 조건별 예측도 이전 증강 실험 결과와 일치합니다.

## 변경하는 것과 보존하는 것

기준은 각 fold의 `artifacts/ranker_robust_gt_missing_token_phonetic_foldN`입니다. 해당 모델의 `training_summary.json`에 기록된 기존 캐시 `runs/ranker_data_robust_gt_missing_token_phonetic_foldN`을 그대로 읽습니다.

| 항목 | 기준86개 | 실험82개 |
|---|---|---|
| 원본 캐시 | 동일한 token/phonetic87열 캐시 | 같은 파일을 그대로 재사용 |
| 제외 특징 | `retrieval_tonnage_rank` | 왼쪽1열 + 지원도 로그4열 |
| 후보·특징 값·라벨·그룹·가중치 | 기존0.5 증강 캐시 | 그대로 유지 |
| 검색·국적/선종 통계 | 해당 외부 훈련 부분100% | 동일 |
| 학습 설정 | 300회·6스레드·기존 seed | 동일,82열로 새 가중치 학습 |
| 검증 | 같은 외부 검증 행·선박 | 동일 |

추가로 제외하는4열은 `flag_source_support_log`, `flag_pair_support_log`, `type_source_support_log`, `type_pair_support_log`입니다. 확률·lift·seen/missing의 나머지12개 범주 특징은 유지합니다.87열 원본 캐시에서 총5열을 제외하므로82열이며, 캐시를82열로 다시 저장하거나 원본 파일을 변환하지 않습니다.

여기서100%는 **각 외부 fold의 훈련 집합 전체**입니다. 검증 행·선박을 포함한 공개 데이터 전체가 아닙니다. 기존 mixed 내부 OOF 특징, 외부 선박5-fold와 seed20261022, 원본 질량0.5를 유지합니다. 통계80%를 고르는 부분집합 seed20261027은 이번 실험에서 사용하지 않습니다. 두 모델은 입력 조건이 바뀌어도 각각의 학습된 가중치를 고정하지만,86개와82개 사이의 가중치는 재학습으로 달라집니다.

## VS Code 실행: 남은 fold2

**fold0·1·3·4는 완료했으므로 반복하지 않습니다.** 아래 명령은 남은 fold2만 실행합니다. 해당0.5 증강 캐시와86개 기준 모델이 있으며 새 모델·비교 출력은 없음을 확인했습니다. 후보·특징·질량을 바꾸지 않고 캐시를 재사용하므로 캐시 재생성·특징 변환·재가중치는 필요 없습니다.

```powershell
cd C:\dacon\port

.\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_robust_gt_missing_token_phonetic_fold2" --output "artifacts/ranker_robust_gt_missing_no_support_fold2" --rounds 300 --threads 6 --exclude-features retrieval_tonnage_rank flag_source_support_log flag_pair_support_log type_source_support_log type_pair_support_log
if ($LASTEXITCODE -ne 0) { throw "fold2 지원도 로그 제외82개 학습 실패" }

.\.venv\Scripts\python.exe -m portmatch compare-support-ablation --baseline "artifacts/ranker_robust_gt_missing_token_phonetic_fold2" --experiment "artifacts/ranker_robust_gt_missing_no_support_fold2" --fold 2 --output "runs/compare_support_ablation_fold2"
if ($LASTEXITCODE -ne 0) { throw "fold2 지원도 로그 제외82개 비교 실패" }
```

완료한 네 fold의 비교는 각각 약42~45분이 걸렸습니다. fold2의 실제 시간은 실행 환경에 따라 달라지며 학습·검증과 비교 시간을 별도로 봅니다.

## 완료 기록: fold0·1·3·4

아래는 완료한 네 fold의 재현 명령입니다. 같은 출력 폴더로 다시 실행하지 않습니다.

```powershell
cd C:\dacon\port

foreach ($fold in @(0, 1, 3, 4)) {
    .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_robust_gt_missing_token_phonetic_fold$fold" --output "artifacts/ranker_robust_gt_missing_no_support_fold$fold" --rounds 300 --threads 6 --exclude-features retrieval_tonnage_rank flag_source_support_log flag_pair_support_log type_source_support_log type_pair_support_log
    if ($LASTEXITCODE -ne 0) { throw "fold$fold 지원도 로그 제외82개 학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch compare-support-ablation --baseline "artifacts/ranker_robust_gt_missing_token_phonetic_fold$fold" --experiment "artifacts/ranker_robust_gt_missing_no_support_fold$fold" --fold $fold --output "runs/compare_support_ablation_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold$fold 지원도 로그 제외82개 비교 실패" }
}
```

학습·비교 출력은 새 폴더 또는 빈 폴더여야 합니다. 기존 모델이나 완료 결과에 덮어쓰지 않습니다. 중간에 실패하면 완료한 fold는 반복하지 않고, 남은 fold만 목록에 넣습니다. 학습이 완료되고 비교만 실패한 fold는 학습 명령을 다시 실행하지 말고 비교 명령만 새 출력 폴더로 실행합니다.82개 모델은 이 실험 전용 비교기를 사용하며 기존 `compare-retrieval`의86개 검증 조건을 임의로 해제하지 않습니다.

## 비교 범위와 결과 파일

각 모델을 **원본, 두 GT 결측, 국제 GT 결측, 일반 GT 결측, 양수 국제 GT ×1.25**에서 비교합니다. 정확도는 전체·한글·비한글과 가중 Top-1(한글70%+비한글30%)을 구분하며 입력 조건들을 하나의 점수로 평균내지 않습니다. 변화량은 **82개−86개**입니다.

- `summary.json`과 `report.md`: 조건별 정확도, 후보 재현율·수, 시간, 출처와 완료 상태.
- `predictions.csv`: 행·조건·모델별 예측과 정답 순위, 후보·점수·시간.
- `row_changes.csv`, `vessel_changes.csv`: 개선·신규 오답·오답 유지의 행별·선박별 비교.
- `candidate_features.csv`, `candidate_comparisons.csv`: 정답과 두 모델1위 후보의 합집합에서 실제 모델 입력 특징과 점수·순위 비교.
- `feature_schema.json`: 두 모델의 실제 입력 순서와 제외한4열. 상세 CSV의 `feature_count`, `omitted_features`와 함께 해석합니다.

검색 사전·설정이 같으므로 **전체 후보 집합·수·정답 포함 여부와 공통82개 특징 값이 두 모델에서 정확히 같아야 합니다.** 이 검사는 상세 파일에 선택한 정답/1위 후보뿐 아니라 모든 검색 후보에 적용합니다.86개에서만 사용하는 지원도4열은82개 입력의 결측값이나0으로 취급하지 않고 제외된 열로 구분합니다. 모델 입력에 원래 있던 NaN도0으로 바꾸지 않습니다.

두 모델의 원본 예측은 **각자의 저장 검증 예측**과 일치해야 합니다. 비교기는 모델의 feature schema, 동일 캐시 출처·원본 질량0.5·fold·훈련 자산을 확인하고, 다른 fold나 다른 캐시/증강 설정을 섞은 비교를 거절합니다. 정상 완료한 결과에서 완료 표시, 저장 원본 예측 일치, 전체 후보/공통 특징 일치, 기존 자산 보존을 확인한 뒤 성능을 해석합니다. 중간 산출물은 완료 결과가 아닙니다.

시간은 같은 장비에서 두 모델의 검색·특징·순위 계산 시간을 비교합니다. 초기화·검사·상세 파일 저장을 포함한 전체 경과 시간과 행별 추론 시간을 구분하고, 작은 차이를 서버 속도 개선으로 확정하지 않습니다. 이번 경로는 제출 ZIP이나 서버 실행 시간 검증이 아닙니다.

## 채택 판단

fold2를 포함한 전체5-fold에서 원본 정답 보존, 한글·비한글 손실, 각 톤수 변형의 개선·신규 오답을 다시 합산해 채택을 판단합니다. 비율·seed·특징을 지금 결과에 맞춰 바꾸지 않습니다. 이번 네 fold의 두 GT 결측 후보 누락은38행·17척이며 이전 전체5-fold의42행·18척과 범위가 다릅니다. 이는 검색 과제로 남고 후보가 같은 이번 순위 실험이 회복할 수 있는 오류와 구분합니다. 현재 최고 제출·기존86개 모델·캐시·ZIP은 유지하고 새 제출은 별도로 판단합니다.

두 모델의 점수 차이는 보정된 확률이나 개별 특징의 인과 효과가 아닙니다. 이미 개발 판단에 사용한 fold에서 재검증하는 실험이며 새 독립 검증 집합으로 해석하지 않습니다. 로컬 결과만으로 Public 점수 개선이나 전체훈련 제출의 하락 원인을 확정하지 않습니다.
