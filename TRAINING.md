# 누수 없는 순위 모델 학습

> 공개용 정리본: 실제 선박별 입력·정답·예측·점수와 행 식별자는 제외했다. 아래 성능표는 실제 실험의 집계 수치이며, 오류의 원리는 별도의 [가상 예시](docs/fictional-examples.md)로 설명한다.

> **최종 코드와 근거부터 보기:** [최종 모델·합성 데이터 테스트](docs/final-model.md) · [효과 분리·공개 집계·채택 판단](docs/experiment-evidence.md). 이 문서 아래에는 과거 단계의 명령이 시간순으로 함께 남아 있습니다. 최종 소스와 실험 집계는 공개했으며 대회 데이터·모델·캐시는 로컬 보존물입니다.

> **2026-11-02 프로젝트 종료:** 사용자의 제출기간 종료 알림에 따라 추가 학습·제출 작업을 마감했습니다. 최종 보존 모델은 focused_search fold0(Public0.8515414798, 사용자 보고)입니다. 아래 명령은 과거 실행 및 재현 기록이며 새 실행 지시가 아닙니다. 지원도 제거82개 fold2 등 미완료 연구는 보류 종료로 남깁니다. [목표별 달성 수준과 종료 보고서](docs/project-closeout.md).

> **2026-10-31 최신 제출 결과:** 사용자 보고 Public **0.8515414798**의 `focused_search fold0`를 현재 최고 제출 기준으로 기록합니다. 직전 최고0.8424607623 대비 **+0.9081%p**이며 별도 ZIP의 전체5,293개 예측 일치 검증을 완료했습니다. 기존 모델·제출물은 보존합니다. 아래 과거 실행 기록의 ‘현재 최고’는 당시 기준이며 고정 실험의 `best` 대조군은 변경하지 않습니다. [제출 ZIP·검증·Public 기록](docs/tonnage-repair.md).

> **2026-11-01 후속 실행·분석 완료:** `focused_replay_v1`의3개 보강 모델·25조건·전체5-fold·1,222,200개 예측을 확인했습니다. 이름 오류와×0.70 개선은 각각 확인됐으나 동시 적용×0.55 −3.5307%p, fold0 −5.8994%p로 유지 기준은 세 모델 모두 실패했습니다. 현재 최고 focused_search를 유지하고 기존 모델·ZIP은 보존합니다. 새 실행이나 학습은 필요하지 않습니다. [전체 결과·설계·유지 기준·재현 명령](docs/tonnage-repair.md)의 후속 보강 절에 기록했습니다.

> 저장소 안내: 이 문서는 실험별 실행 기록도 포함합니다. `work/`, `runs/`, `artifacts/`는 GitHub에 포함되지 않는 로컬 데이터·생성 산출물입니다. 아래 완료 실험 명령은 재현 기록이며 같은 출력 폴더에 다시 실행하지 않습니다.
>
> 2026-10-23: 기존 이름 특징 86개 실험의 fold0~4, CH 93개 실험의 fold1·fold2가 완료됐습니다. CH는 개선 0행·신규 오답 1행으로 미채택하고 86개를 유지합니다. [CH 결과](docs/experiments/ch-variants-fold12.md)를 참고하세요. 아래 완료 실험 명령을 같은 출력 경로에 다시 실행하지 않습니다.

> **2026-10-26: 0.75 전체5-fold 검증 완료·미채택.** 원본98.8479%(0.5 대비+0.1094%p), 국제 GT ×1.25 96.4943%(−1.3600%p)로0.5를 유지합니다. 현재 최고 제출은 사용자 보고 **0.8424607623**의 robust·86개·0.5 fold0입니다. 기존86개·첫 ZIP도 보존합니다. [최신 전체 결과와 다음 계획](docs/experiments/gt-reweighting-5fold.md) · [제출 기록](docs/experiments/second-submission.md).

현재 검색 결과를 유지하면서 정답 선박을 선택하는 LightGBM 순위 모델입니다. 채택 기준은 토큰·문맥 음차 17개를 추가한 86개 특징입니다. 이전 69개 모델은 대조군으로 보존합니다. 기본 설정은 CPU 6스레드, LambdaRank, 300회 학습입니다. 기존 특징 캐시에서 열을 선택하거나 버전별 특징을 추가하고 전체 검증 결과를 비교할 수 있습니다.

## 보류 종료 기록: 지원도 로그4열 제외82개 — fold2 미완료

2026-10-30. **fold0·1·3·4의82개 학습·비교와 결과 분석을 완료했고 채택은 보류합니다.** 네 fold의 원본+0.1489%p, 두 GT 결측+0.9459%p이지만 국제 GT ×1.25 −0.1421%p이고 원본 신규 오답3행이 남습니다. 같은 설정으로 fold2를 완료해 전체5-fold를 판단합니다. 검색·통계100%·원본 질량0.5·300회·6스레드·seed20261022를 유지하고87열 캐시에서 기존 톤수 순위1열과 지원도 로그4열만 제외합니다. [네 fold 결과](docs/experiments/support-ablation-fold0134.md) · [설계·산출물](docs/support-ablation.md).

아래는 **아직 실행하지 않은 fold2만** 학습·비교하는 명령입니다. 기존 캐시와86개 기준 모델이 있고 새 출력 폴더는 없음을 확인했습니다. 캐시 생성·변환·재가중치는 필요 없습니다.

```powershell
cd C:\dacon\port

.\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_robust_gt_missing_token_phonetic_fold2" --output "artifacts/ranker_robust_gt_missing_no_support_fold2" --rounds 300 --threads 6 --exclude-features retrieval_tonnage_rank flag_source_support_log flag_pair_support_log type_source_support_log type_pair_support_log
if ($LASTEXITCODE -ne 0) { throw "fold2 지원도 로그 제외82개 학습 실패" }

.\.venv\Scripts\python.exe -m portmatch compare-support-ablation --baseline "artifacts/ranker_robust_gt_missing_token_phonetic_fold2" --experiment "artifacts/ranker_robust_gt_missing_no_support_fold2" --fold 2 --output "runs/compare_support_ablation_fold2"
if ($LASTEXITCODE -ne 0) { throw "fold2 지원도 로그 제외82개 비교 실패" }
```

현재 최고 제출 robust·86개·0.5 fold0(Public0.8424607623)는 유지하며 새 제출은 하지 않았습니다. 같은 설정의 fold2 결과를 포함한 전체5-fold의 원본 보존·톤수 변형 손실을 분석한 뒤82개 채택을 판단합니다. 이번 저장 결과 감사31,163개 검사를 통과했으며376개 테스트는 당시 구현 검증 기록입니다.

## 완료 기록: 지원도 로그4열 제외82개 fold0·1·3·4

아래는 **완료한 네 fold의 재현 명령**입니다. 기존 출력 폴더에 다시 실행하지 않습니다. 실제 비교9,772행·97,720개 예측을 분석했으며 아직 전체5-fold 결과는 아닙니다.

```powershell
cd C:\dacon\port

foreach ($fold in @(0, 1, 3, 4)) {
    .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_robust_gt_missing_token_phonetic_fold$fold" --output "artifacts/ranker_robust_gt_missing_no_support_fold$fold" --rounds 300 --threads 6 --exclude-features retrieval_tonnage_rank flag_source_support_log flag_pair_support_log type_source_support_log type_pair_support_log
    if ($LASTEXITCODE -ne 0) { throw "fold$fold 지원도 로그 제외82개 학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch compare-support-ablation --baseline "artifacts/ranker_robust_gt_missing_token_phonetic_fold$fold" --experiment "artifacts/ranker_robust_gt_missing_no_support_fold$fold" --fold $fold --output "runs/compare_support_ablation_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold$fold 지원도 로그 제외82개 비교 실패" }
}
```

학습·비교 출력은 새 폴더 또는 빈 폴더여야 합니다. 완료한 fold는 반복하지 않으며, 학습 뒤 비교만 실패하면 비교만 새 출력 폴더로 재실행합니다. 원본·네 톤수 변형의 정확도, 후보 재현율·수, 행·선박별 개선/신규 오답과 시간을 비교합니다. 전체 후보와 공통82열은 정확히 같아야 하며 각 모델의 원본 예측은 각자의 저장 검증 예측과 대조합니다. 현재 최고 제출 robust·86개·0.5 fold0(Public0.8424607623)는 유지하고82개 채택이나 새 제출은 결과 확인 후 판단합니다.

## 완료 기록: 검색100% 고정·범주80/100% 전체5-fold

아래는 **완료한 재현 명령**입니다. 각 fold의 기존 robust·86개·GT 결측 증강0.5 모델 가중치, 외부 분할 seed20261022와 부분집합 seed20261027을 고정했습니다. 기존 출력 폴더에 다시 실행하지 않습니다.

```powershell
cd C:\dacon\port

foreach ($fold in @(1, 3, 4)) {
    .\.venv\Scripts\python.exe -m portmatch compare-category-scope --model "artifacts/ranker_robust_gt_missing_token_phonetic_fold$fold" --fold $fold --subset-seed 20261027 --output "runs/compare_category_scope_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold$fold 검색100 고정·범주80/100 비교 실패" }
}
```

출력은 새 폴더 또는 빈 폴더여야 합니다. 기존 비교를 새로 생성하거나 `--reference`를 지정할 필요 없이100% 통계의 원본 예측을 저장 모델의 검증 예측과 대조합니다. 원본·네 톤수 변형의 정확도, 후보 재현율/수, 개선·신규 오답, 시간과 동일 후보의86열을 기록합니다. 전체 후보의 비범주70열·후보 집합은 두 조합 간 같아야 합니다. **변화량은 C−D(통계80%−100%)**로 이전2×2 `category_at_search100`의 D−C와 부호가 반대입니다. [설계·산출물·완료 확인·표본 실행](docs/category-scope-diagnostic.md).

새 결과 감사6,829개 검사가 통과했고 기존 fold0·2 감사는 현재 해시를 대조해 재사용했습니다. 354개 테스트는 당시 구현 기록이며 이 분석에서는 전체 테스트나 긴 추론을 다시 실행하지 않았습니다. 후속으로 검색·통계100%와0.5 캐시를 보존한 지원도 로그4열 제외82개 재학습의 fold0·1·3·4까지 완료했습니다. 나머지12개 범주 특징은 유지하며82개 채택 보류·남은 fold2 명령은 이 문서 상단과 [실험 안내](docs/support-ablation.md)에 있습니다. 새 제출은 하지 않았습니다.

## 완료 기록: 검색 사전·범주 통계 2×2 진단 fold0·2 분석

`compare-index-factorial`의 실제 fold0·2 실행을 완료하고4,902행·98,040개 예측과 동일 후보의86열을 분석했습니다. 검색100%·통계80%는 양쪽100% 대비 원본98.4530%를 유지하고, 두 GT 결측94.1609→94.8455%, 국제GT×1.25 97.2549→97.9395%로 각각4행·1척을 회복했습니다. 신규 오답은0건이지만 효과가 fold0의2척에 집중됐고 fold2의 모든 예측은 같습니다. 기존 최고 fold0는 유지하며 검색100%·통계80%의 채택을 보류합니다. [전체 결과·동일 후보 특징·판단](docs/experiments/index-factorial-fold02.md).

이번 저장 산출물 감사는14,473개 검증 통과·이상0건입니다. 기존 대각선 예측과 D 원본의 저장 예측을 대조하고, 실제 존재하는 동일 후보98,408개의86열로 점수를 재계산했습니다. 새 검색을 재실행한 성능 실험은 아닙니다.

후속으로 subset seed20261027과 검색100%를 고정한 fold1·3·4 C/D 비교를 완료했습니다. 전체5-fold에서 통계 축소의 신규 손실이 나타나80%를 채택하지 않습니다. 기존 `compare-index-factorial`의 fold0·2 경로와 결과는 보존합니다. 다음 지원도 로그4열 제외82개 재학습은 통계100%를 유지한 별도 실험이며 [비교 경로와 명령을 준비했습니다](docs/support-ablation.md). 비율·seed 탐색과 특정 선박 예외는 추가하지 않습니다. 전체333개 테스트·기존 모델/ZIP 검사는 당시2×2 구현 기록입니다.

아래는 **완료한 fold0·2 실행 기록**입니다. 같은 출력 폴더에 재실행하지 않습니다. [설계·파일·완료 기준](docs/index-factorial-diagnostic.md).

```powershell
cd C:\dacon\port

foreach ($fold in @(0, 2)) {
    .\.venv\Scripts\python.exe -m portmatch compare-index-factorial --model "artifacts/ranker_robust_gt_missing_token_phonetic_fold$fold" --reference "runs/compare_index_scale_fold$fold" --fold $fold --subset-seed 20261027 --output "runs/compare_index_factorial_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold$fold 검색·범주 2x2 진단 실패" }
}
```

재현 시 출력은 새 폴더 또는 빈 폴더여야 합니다. 표본 확인은 별도 폴더에서 `--max-queries 64`를 추가하고 본 실행에는 이 옵션을 빼야 합니다. `--reference`는 표본 실행에서도 같은 완료된 전체5조건 비교 결과를 사용합니다. 원본·기존 모델·사전·완료 산출물은 덮어쓰지 않습니다.

2026-10-28: 고정 모델·사전80%/100%의 fold0·2 전체 비교를 분석했고 저장 산출물 감사5,057개를 통과했습니다. 4,902행·49,020개 예측에서 원본 가중 정확도98.1107→98.4530%, 두 GT 결측94.8455→94.1609%, 국제GT×1.25 97.5972→97.2549%입니다. fold2는 모든 예측이 같고, 정답 후보 획득·손실은 두 fold 모두0건입니다. 80% 사전을 채택하지 않고 기존 최고 제출0.8424607623의 fold0를 유지합니다. [전체 결과와 다음2×2 분리 진단 계획](docs/experiments/index-scale-fold02.md).

2026-10-27 사용자 보고0.8324345055는 사용자가 제출 경로를 확인한 전체훈련 ZIP의 결과입니다. 기존 최고보다−1.00262568%p이며 제출번호·서버 시간은 미확인입니다. 사전 규모 진단만으로 이 Public 차이를 설명하지 않습니다. 진단 구현 당시307개 테스트와 기존 모델·ZIP 보존 검사를 통과했으며 이번에는 저장 결과를 분석했습니다. [제출 결과와 진단 배경](docs/full-training.md#public-결과와-다음-진단) · [완료한 진단 설계·명령·산출물](docs/index-scale-diagnostic.md).

`prepare-full-ranker` → `transform-ranker-cache --transform token-phonetic` → `train-full-ranker` → `submission_package --allow-full` → `scripts/verify_full_submission.py`의 실제 실행을 모두 완료했습니다. 공개12,222행·5,851척에서92,964개 그룹·11,955,955개 후보쌍으로 학습했고 원본 단위 탈락은0건입니다. ZIP6조건의 저장 모델 일치와 행별 독립성을 확인했습니다. 원본 전체 실행1,281.59초, ZIP48.11MB입니다. 이는 학습과 겹치는 입력의 기능 검사이며 새 검증 정확도가 아닙니다. [전체 실행 결과와 완료 명령](docs/full-training.md)을 참고하며 같은 출력에 재실행하지 않습니다. 구현 당시281개 테스트 통과 기록은 보존합니다.

`runs/compare_robust_gt_missing_mass075_fold0`부터 `fold4`까지 전체12,222행·305,550예측을 재집계했고 감사20,957개를 통과했습니다. 원본은 기존86개98.9067% / 증강 없는 robust98.6798% / 0.5 98.7385% / 0.75 98.8479%입니다. 0.5 대비 원본4행·1척 회복/5행·4척 신규, 국제 GT ×1.25는1행 회복/27행·11척 신규입니다. [전체 분석·채택 판단](docs/experiments/gt-reweighting-5fold.md).

**원본 질량0.75의 재가중치·학습·비교는 모두 끝났습니다.** [fold1 명령](docs/experiments/gt-reweighting.md)과 [fold0·2·3·4 명령](docs/experiments/gt-reweighting-fold1.md#판단과-다음-실행)은 완료한 재현 기록이며 같은 출력에 재실행하지 않습니다. 기존0.5 캐시·후보·특징·라벨·순서·원본 검증을 보존하고 질량만 바꾼 통제 실험입니다.

현재 다음 제출 개발 설정은 **robust·86개·original_mass=0.5·300회·6스레드**로 유지합니다. 기존86개는 원본 성능 대조군이며,0.5도 기존 원본 보존 목표를 충족한 것은 아닙니다. 질량 비율 추가 탐색은 종료합니다.

후속으로 **전체 공개훈련12,222행의 내부 OOF 특징 생성→단일 모델 학습→공개훈련 전체로 고정한 추론 사전→ZIP 검증**을 완료했습니다. 기존 외부 fold 캐시를 단순 합치지 않고 새 캐시를 생성했습니다. 전체훈련 모델에는 검증 정확도를 붙이지 않으며 ZIP 확인도 학습과 겹치는 입력의 기능·독립성·시간 검사임을 명시합니다.

완료한 `compare-index-scale`은 각 외부 fold의 기존 robust·86개·0.5 모델 가중치를 고정하고 훈련 사전80%와100%를 비교합니다.80%는 **훈련 선박 수의80%를 내림한 선박 집합**이며 해당 선박의 원본 행 전체를 사용합니다. 두 사전의 조회·별칭·TF-IDF·범주 통계를 각각 같은 훈련 범위로 맞추고 외부 검증 선박을 모두 제외합니다. 기본 실행은 해당 fold 검증 전체의 원본·4개 톤수 변형이며, 모든 차이는 같은 조건의100%−80%입니다. 새 학습·특징 캐시 생성·제출 패키징은 필요하지 않습니다. 아래 명령은 **완료한 실행 기록**으로 같은 출력 폴더에 재실행하지 않습니다.

```powershell
cd C:\dacon\port

foreach ($fold in @(0, 2)) {
    .\.venv\Scripts\python.exe -m portmatch compare-index-scale --model "artifacts/ranker_robust_gt_missing_token_phonetic_fold$fold" --fold $fold --subset-seed 20261027 --output "runs/compare_index_scale_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold$fold 사전 범위 진단 실패" }
}
```

출력은 새 폴더 또는 빈 폴더여야 합니다. 완료한 fold를 다시 실행하지 않으며 중단된 폴더는 다른 이름으로 재시도합니다. `summary.json`의 `complete`, 가중치·훈련 자산·보호 파일 보존 플래그,100% 원본의 저장 예측 일치를 확인한 뒤 `report.md`와 행·선박별 CSV를 분석합니다. 작은 표본 진단은 `--max-queries`를 별도 출력 폴더에 사용합니다. [표본 명령·완료 확인·해석 한계](docs/index-scale-diagnostic.md).

아래는 완료한 재학습 비교의 실행 기록입니다. 같은 출력에 다시 실행하지 않습니다.

```powershell
cd C:\dacon\port
.\.venv\Scripts\python.exe -m portmatch compare-retrieval --baseline artifacts/ranker_token_phonetic_fold0 --profiles robust --experiment-model artifacts/ranker_robust_token_phonetic_fold0 --output runs/compare_robust_retrained_fold0
```

아래는 이미 완료한 검색 교체 비교의 재현 기록입니다. 같은 폴더에 다시 실행하지 않습니다.

```powershell
cd C:\dacon\port
.\.venv\Scripts\python.exe -m portmatch compare-retrieval --baseline artifacts/ranker_token_phonetic_fold0 --profiles hangul dual-gt robust --output runs/compare_robust_retrieval_fold0
```

완료된 비교는 2,452행 × 5입력 × 4검색, 총 49,040개 예측입니다. `prepare-ranker --retrieval-profile robust`의 후보와 오답 샘플은 달라져 기존 특징 캐시를 재사용할 수 없습니다. 재학습 후에도 비교는 `compare-retrieval --experiment-model ...`을 사용합니다. 현재 86개 기준과 새 검색 고정 모델, 새 검색 재학습 모델을 구분해 판단합니다. [구현·옵션 안내](docs/experiments/robust-retrieval.md)는 유지합니다.

## 완료한 CH 대체 발음 86→93개 실험

`transform-ranker-cache --transform ch-variants`는 완료된 `runs/ranker_data_token_phonetic_foldN`의 87열을 보존하고 CH 비교 7열을 붙입니다. 새 94열 캐시에서 `retrieval_tonnage_rank`를 제외해 93개로 학습하며, 비교 기준은 `artifacts/ranker_token_phonetic_foldN`의 86개 모델입니다. 원본 70열 캐시를 직접 입력하지 않습니다. 변환·학습·비교에는 원본과 87열 부모 캐시의 무결성 확인도 포함됩니다.

## 현재 개발 기준: 토큰·문맥 음차 86개 특징 채택

**69개 + 이름 특징 17개 = 86개 특징의 전체 5-fold 변환·학습·비교를 완료했습니다.** 표본 제한 없이 같은 선박 분할·후보·오답 샘플·학습 설정으로 전체 12,222행을 비교했습니다.

| 항목 | 이전 69개 기준 | 채택한 86개 모델 |
|---|---:|---:|
| 가중 Top-1 | 97.0368% | **98.9067% (+1.8699%p)** |
| 한글 정답 | 1,141/1,191 (95.8018%) | **1,173/1,191 (98.4887%)** |
| 비한글 정답 | **11,022/11,031 (99.9184%)** | 11,018/11,031 (99.8822%) |
| 전체 오답 | 59행·30척 | **31행·20척** |
| 후보 재현율 | 100% | 100% |

**86개를 개발 기준으로 채택하고 69개는 대조군으로 보존합니다.** 당시 후속 CH→K 대체 표현 실험은 위와 같이 fold1·2 비교 후 미채택했습니다. 첫 Public 결과 이후의 우선순위는 [제출 분석](docs/experiments/first-public-submission.md)에 따릅니다. 비한글의 짧은 음차 키 충돌·원형 정보 보존과 기존 캐시의 특징군 선택 비교는 별도 계획으로 남깁니다. 개별 검증 선박을 위한 예외 규칙은 넣지 않습니다. 전체 fold를 개발에 사용했으므로 98.9067%는 독립 테스트나 리더보드 점수가 아닙니다.

선박별 개선·신규 오답, fold별 지표와 다음 실험 계획은 [86개 특징 전체 비교](docs/experiments/token-phonetic-5fold.md)에 있습니다. 로컬 근거는 `runs/compare_token_phonetic_5fold/`의 `summary.json`, `row_changes.csv`, `vessel_changes.csv`, `per_fold_metrics.csv`입니다. 아래 명령은 이미 완료된 실험의 재현 안내이며, 재학습할 필요가 없습니다.

### 완료 이력: fold2 선행 검토 — 2026-10-23

`runs/compare_token_phonetic_fold2`의 전체 검증 2,450행을 확인했습니다. 표본 제한이 없으며 같은 후보·오답 샘플·학습 설정에서 새 이름 특징 17개를 추가한 비교입니다.

| 항목 | 당시 69개 기준 | 이름 특징 86개 |
|---|---:|---:|
| fold2 가중 Top-1 | 94.4978% | **99.3616% (+4.8638%p)** |
| 한글 정답 / 정확도 | 211/229 · 92.1397% | **227/229 · 99.1266%** |
| 비한글 정답 / 정확도 | **2,221/2,221 · 100%** | 2,219/2,221 · 99.9100% |
| 전체 오답 | 18행·6척 | **4행·2척** |
| 후보 재현율 / Recall@5 | 100% / 100% | 100% / 100% |

fold2 기록상 캐시 변환 151.2초, 캐시 읽기·학습 26.9초, 학습·평가 전체 39.3초입니다. 학습 전 출처/무결성 검사와 비교 리포트 생성 시간은 학습·평가 전체 시간에 포함되지 않으며, 다른 fold나 환경의 실행 시간을 보장하지 않습니다.

근거: 비교 리포트 (`runs/compare_token_phonetic_fold2/report.md`; 로컬 산출물), 최종 판단 (`work/token_phonetic_fold2_analysis/decision.json`; 로컬 산출물), 선박별 진단 (`work/token_phonetic_fold2_analysis/case_findings.md`; 로컬 산출물), 감사 결과 (`work/token_phonetic_fold2_analysis/audit/audit.json`; 로컬 산출물). 이번 결과 분석에서는 새 학습이나 특징 재생성을 실행하지 않았습니다.

### 특징과 캐시 구성

원본 `runs/ranker_data_foldN`의 70개 열과 후보·정답·그룹·순서·오답 샘플을 그대로 복사하고, 새 이름 특징을 뒤에 붙여 87열 캐시를 만듭니다. 학습할 때 반드시 `--exclude-features retrieval_tonnage_rank`를 사용해 이전 69개 대조군과 같은 원래 특징에 이름 17개를 더한 86개를 학습합니다. 미채택 동점 순위 캐시는 이번 입력으로 사용하지 않습니다.

새 특징은 질의의 원래 선박명과 고정 카탈로그 선박명만 비교합니다. 정답 ID·호출부호·별칭·훈련 통계는 새 계산에 사용하지 않습니다. 원래 조회/범주 OOF 값은 복사하므로 전체 특징 생성이나 내부 OOF 인덱스 재구축이 필요하지 않습니다. 캐시 변환도 전체 후보쌍 계산과 압축 파일 저장이 필요하므로 긴 작업은 아래 명령으로 실행합니다.

| 특징군 | 수 | 역할 |
|---|---:|---|
| 토큰 비교 `name_token_*` | 9 | 단어 순서에 따른 대응·약한 토큰·첫/마지막 토큰·공통 경계 단어 이후 잔여 이름 비교 |
| 문맥 음차 `name_context_*` | 3 | 한글 초성/종성과 영문 C/CH 등의 문맥을 반영한 자음 표현 비교 및 보조 변형 |
| 모음 포함 전사 `name_vowel_*` | 2 | 일반적인 전사 표기 차이를 통합한 전체 이름 비교 |
| 접미 보조 표현 `name_suffix_*` | 3 | 원래 이름을 보존하면서 한글 선박 접미 ‘호’ 등을 보조 표현으로 비교 |

기존 `romanized_ratio`도 모음을 포함합니다. 새 특징은 원래 모음 비교를 대체하지 않고 토큰 경계·문맥·전사 변형 정보를 추가합니다. 문맥 음차는 고정된 일반 규칙이며 영어 발음 사전이나 번역기가 아닙니다. 개별 검증 선박 예외 규칙을 넣지 않습니다.

전체 테스트 **119개 통과**: 기존91 + 이름 특징13 + 캐시8 + 통합7입니다. 작은 합성 데이터에서 원본70 배열 보존, 새 특징을 제외한 69개 대조 모델의 예측 bit-exact, 새 86개 모델의 저장·복원·캐시/실시간 추론 일치, 캐시 접근 없이 번들 추론을 확인했습니다. 테스트 기록 (`work/token_phonetic_test.log`; 로컬 산출물), 구현·진단 요약 (`work/token_phonetic_implementation/diagnostic_summary.json`; 로컬 산출물), 31행·65후보쌍의 예시 특징 (`work/token_phonetic_implementation/name_feature_examples.csv`; 로컬 산출물)에 근거가 있습니다. 예시 특징의 구별 신호 자체는 성능 점수가 아닙니다. 이후 본 학습을 완료했고 전체 검증 결과를 위에 별도로 기록했습니다.

### 완료한 fold2 명령 — 재현 기록

아래 fold2 출력과 다음 절의 나머지 fold·전체 합산 출력은 모두 완료됐습니다. 현재 경로로 다시 실행하지 않습니다. 별도로 재현할 때는 새 출력 이름과 후속 입력 경로를 함께 바꿉니다.

```powershell
cd C:\dacon\port

& {
    $fold = 2
    .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --transform token-phonetic --data "runs/ranker_data_fold$fold" --output "runs/ranker_data_token_phonetic_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold$fold 이름 특징 변환 실패" }

    .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_token_phonetic_fold$fold" --output "artifacts/ranker_token_phonetic_fold$fold" --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
    if ($LASTEXITCODE -ne 0) { throw "fold$fold 학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch compare-rankers --baseline "artifacts/ranker_no_tonnage_rank_fold$fold" --experiment "artifacts/ranker_token_phonetic_fold$fold" --output "runs/compare_token_phonetic_fold$fold"
    if ($LASTEXITCODE -ne 0) { throw "fold$fold 비교 실패" }
}
```

완료 결과는 `runs/compare_token_phonetic_fold2/report.md`, `summary.json`, `row_changes.csv`, `vessel_changes.csv`입니다. 모델 요약의 `trained_feature_names`는 86개이고 `retrieval_tonnage_rank`가 제외돼 있습니다. 위 결과 표와 선박별 검토는 이 전체 fold2 출력에 근거합니다.

### 완료한 0·1·3·4 fold 명령 — 재현 기록

fold2 이후 0·1·3·4 fold의 새 이름 캐시·모델·비교도 모두 완료했습니다. 아래는 당시 실행 순서의 기록입니다. 새 환경에서 재현할 경우에만 실행하고, 실패한 폴더는 보존합니다. 중간부터 실행할 때는 이미 완료한 fold·단계를 건너뜁니다.

```powershell
cd C:\dacon\port

& {
    foreach ($fold in 0, 1, 3, 4) {
        .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --transform token-phonetic --data "runs/ranker_data_fold$fold" --output "runs/ranker_data_token_phonetic_fold$fold"
        if ($LASTEXITCODE -ne 0) { throw "fold$fold 이름 특징 변환 실패" }

        .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_token_phonetic_fold$fold" --output "artifacts/ranker_token_phonetic_fold$fold" --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
        if ($LASTEXITCODE -ne 0) { throw "fold$fold 학습 실패" }

        .\.venv\Scripts\python.exe -m portmatch compare-rankers --baseline "artifacts/ranker_no_tonnage_rank_fold$fold" --experiment "artifacts/ranker_token_phonetic_fold$fold" --output "runs/compare_token_phonetic_fold$fold"
        if ($LASTEXITCODE -ne 0) { throw "fold$fold 비교 실패" }
    }
}
```

### 완료한 전체 5-fold 합산 명령 — 재현 기록

5개 fold가 모두 완료된 뒤 아래 비교로 `runs/compare_token_phonetic_5fold`를 생성했습니다. 합산 점수는 언어별 전체 예측을 모아 다시 계산하며 fold 점수의 단순 평균을 사용하지 않습니다.

```powershell
& {
    $baselineModels = @(0..4 | ForEach-Object { "artifacts/ranker_no_tonnage_rank_fold$_" })
    $nameModels = @(0..4 | ForEach-Object { "artifacts/ranker_token_phonetic_fold$_" })
    .\.venv\Scripts\python.exe -m portmatch compare-rankers --baseline $baselineModels --experiment $nameModels --output runs/compare_token_phonetic_5fold
    if ($LASTEXITCODE -ne 0) { throw "이름 특징 5-fold 비교 실패" }
}
```

모든 fold는 이미 개발 판단에 사용했으므로 독립 테스트 점수와 구분합니다. 캐시를 한 번 만들면 신규 특징군을 제외한 별도 모델도 열 선택만으로 학습할 수 있습니다. 전체 17개 추가 효과는 확인했으며, 후속 특징군 비교는 채택한 86개를 기준으로 별도 출력 폴더에서 진행합니다.

### 완료 폴더와 중단 처리

- 변환 완료는 `runs/ranker_data_token_phonetic_foldN/metadata.json`의 `complete=true`, 학습 완료는 모델 폴더의 `training_summary.json`의 `complete=true`, 비교 완료는 비교 폴더의 `summary.json`의 `complete=true`로 확인합니다.
- 변환이 끝났으면 학습 줄부터, 학습도 끝났으면 비교 줄부터 실행합니다. 같은 완료 폴더로 위 전체 블록을 재실행하면 덮어쓰기 방지 오류가 납니다.
- 실패한 부분 폴더는 그대로 두고 새 출력 이름을 정한 뒤 후속 명령의 입력 경로도 맞춥니다. 자동 이어쓰기나 덮어쓰기는 하지 않습니다.
- 원본·파생 캐시는 학습/비교의 출처·파일·공통 배열 검증에 필요합니다. 이 단계가 끝날 때까지 두 폴더를 현재 경로에 보존합니다. 저장 모델의 추론에는 훈련 캐시가 필요하지 않습니다.
- 기존 모델의 핵심 특징 코드와 기존 동점 보정 코드는 유지합니다. 새 코드·특징 순서의 해시가 바뀌면 파생 캐시와 모델을 새 버전으로 다시 준비해야 합니다.

## 이전 개발 기준 이력: 톤수 순위 제거 69개 채택

2026-10-22 전체 선박 5-fold 비교를 완료했고, `artifacts/ranker_no_tonnage_rank_fold0`부터 `fold4`까지의 69개 특징 모델을 다음 실험의 개발 기준으로 채택했습니다. 기준 설정은 `--exclude-features retrieval_tonnage_rank`, 300회·6스레드입니다. CLI에서 선택 옵션을 생략하면 여전히 전체 70개 특징을 사용합니다.

| 항목 | 원래 70개 특징 | 톤수 순위 제외 69개 |
|---|---:|---:|
| 가중 Top-1 | 96.3315% | **97.0368%** |
| 한글 Top-1 | 94.7943% | **95.8018%** |
| 비한글 Top-1 | 99.9184% | 99.9184% |
| 한글 오답 | 62행·24척 | **50행·21척** |

근거: `runs/compare_no_tonnage_rank_5fold/report.md`, `work/no_tonnage_rank_5fold_analysis/decision.json`, `changed_vessels_analysis.csv`, `error_transitions_75.csv`. 기존 모델과 캐시는 그대로 보존합니다.

## 완료 결과: 동점 보정 톤수 순위 미채택 — 2026-10-23

`runs/compare_tonnage_min_5fold`의 5개 fold, 전체 12,222행을 확인했습니다. 표본 제한 없이 같은 후보·훈련 샘플·학습 설정에서 69개 기준 모델과 보정 순위를 추가한 70개 모델을 비교했습니다.

| 항목 | 당시 69개 기준 | 동점 보정 70개 |
|---|---:|---:|
| 가중 Top-1 | **97.0368%** | 96.7429% (−0.2939%p) |
| 한글 Top-1 | **95.8018% (1,141/1,191)** | 95.3820% (1,136/1,191) |
| 비한글 Top-1 | 99.9184% (11,022/11,031) | 동일 |
| 한글 오답 | **50행·21척** | 55행·22척 |
| 전체 오답 | **59행·30척** | 64행·31척 |
| 전체 후보 재현율 | 100% | 100% |
| 전체 Recall@5 | 99.9264% (12,213/12,222) | **99.9591% (12,217/12,222)** |

| Fold | 69개 가중 Top-1 | 보정 70개 가중 Top-1 | 변화(%p) | 개선 / 신규 오답 |
|---|---:|---:|---:|---:|
| 0 | 95.6958% | 96.4736% | +0.7778 | 2 / 0 |
| 1 | 98.6611% | 97.6124% | −1.0487 | 0 / 4 |
| 2 | 94.4978% | 94.4978% | 0.0000 | 0 / 0 |
| 3 | 97.3450% | 96.8227% | −0.5224 | 0 / 2 |
| 4 | 98.2736% | 97.9902% | −0.2834 | 1 / 2 |

최종 예측이 달라진 11행은 모두 한글입니다. 개선 3행·2척, 신규 오답 8행·3척이며, 다른 오답 선박으로 바뀐 사례는 없습니다.

검증 근거: 전체 비교 리포트 (`runs/compare_tonnage_min_5fold/report.md`; 로컬 산출물), 감사 결과 (`work/tonnage_min_5fold_analysis/audit/audit.json`; 로컬 산출물), 변경 행 (`work/tonnage_min_5fold_analysis/audit/changed_rows.csv`; 로컬 산출물), 변경 선박 (`work/tonnage_min_5fold_analysis/audit/changed_vessels.csv`; 로컬 산출물), Top5 변화 (`work/tonnage_min_5fold_analysis/audit/recall5_changes.csv`; 로컬 산출물). 원천 데이터·학습 요약·예측·모델·메타데이터·파생 manifest의 해시와 보고서의 지표·행/선박 변화를 재확인했습니다. 대형 특징 배열 보존은 비교 생성 시의 검증 기록과 현재 해시 연결을 확인했으며, 이번 분석에서 전체 배열을 다시 읽거나 재학습하지 않았습니다.

## 완료한 실험의 방식과 재현 기록: 동점 보정 톤수 순위

`transform-ranker-cache`는 완료된 원본 70개 특징 캐시의 `retrieval_tonnage_rank`를 **`retrieval_tonnage_min_rank`**로 교체해 새 폴더에 저장합니다. 새 모델은 기존 기준 모델의 69개 특징에 이 보정 특징을 더한 70개 특징을 사용합니다. 학습할 때 `--exclude-features retrieval_tonnage_rank`를 붙이지 않습니다. 새 캐시에는 그 이름의 열이 없습니다.

계산식은 `1 + 현재 후보보다 GT 절대 차이가 작은 톤수 검색 후보 수`입니다. 예를 들어 거리 `[0, 0, 10, 10, 20]`의 순위는 `[1, 1, 3, 3, 5]`입니다. 톤수 검색에 포함되지 않은 후보는 0을 유지합니다. 같은 GT 값뿐 아니라 질의의 양쪽에서 같은 거리에 있는 GT도 동점입니다. 마지막 예측 점수의 동점 정렬은 기존의 고정 선박 ID 순서를 유지합니다.

계산에는 **훈련 오답 샘플링 전 전체 톤수 검색 후보**를 사용합니다. 국제총톤수 우선·국내총톤수 보완·최근접 후보 대체 검색도 원래 규칙을 따릅니다. 원본 캐시의 후보·정답·그룹·순서와 나머지 69개 특징값을 유지하므로 이름 검색, 음차 특징, 내부 OOF 사전을 다시 생성할 필요가 없습니다. 새 값에는 질의의 톤수와 고정 카탈로그만 사용하며, 외부 훈련 인덱스의 별칭·국적·선종 통계를 훈련 행에 재적용하지 않습니다.

변환은 압축된 특징 파일을 읽고 새 파일로 저장하므로 실행 시간과 추가 디스크 공간이 필요합니다. 기존 캐시는 덮어쓰지 않습니다. 학습·비교 시 원본 및 새 캐시의 파일 해시와 파생 관계를 확인하므로 **비교 완료까지 원본 캐시를 현재 경로에 보존**합니다. 모델 추론은 저장한 모델 폴더만으로 가능하며 동일 보정 계산을 적용합니다.

### 초기 fold2 결과 — 전체 결과 이전의 잠정 기록, 2026-10-23

아래는 fold2만 완료됐을 때의 진단과 당시 판단입니다. 현재 채택 판단은 위의 전체 5-fold 결과로 대체됐습니다.

`runs/compare_tonnage_min_fold2`의 2,450행을 재집계했습니다. 한글 211/229, 비한글 2,221/2,221로 가중 Top-1은 94.4978%를 유지했습니다. 수정 0건·신규 오답 0건이며 **전체 예측 선박과 정답 순위가 모두 동일**합니다. 후보 재현율과 Recall@5도 양쪽 100%입니다.

새 순위는 모델 분기에 312회 사용됐고 모든 행의 점수도 달라졌습니다. 특징이 적용되지 않은 실험은 아닙니다. 저장 모델 3종으로 남은 18행을 재현했습니다. 새 모델에서 보정 순위의 정답 대 오답 기여 차이는 약 +0.035였고, 이름 및 다른 검색 특징과의 조합을 뒤집기에는 부족했습니다. 기여도는 해당 모델의 설명이며 단일 특징의 인과 효과로 해석하지 않습니다.

**당시 잠정 판단은 채택을 보류하고 나머지 4개 fold를 확인하는 것이었습니다.** fold2에서는 이득도 신규 오답도 없었고, 다른 fold의 기존 개선 유지 여부는 그 시점에 미확인이었습니다. 이후 나머지 fold까지 완료한 결과 최종 미채택으로 결정했습니다. 당시 캐시 변환은 61.2초, 학습·평가는 34.1초였으며, 학습 전 무결성 검사와 비교 검증 시간은 제외된 기록입니다.

상세 근거: `work/tonnage_min_fold2_analysis/vessel_analysis.csv`, `cases.json`, `decision.json`, `audit/audit.json`. 이번 분석에서는 새 모델을 학습하거나 캐시를 다시 생성하지 않았습니다.

### 1. 완료한 fold2 명령 — 재현 기록

아래 fold2 출력과 다음 절의 나머지 fold·전체 비교 출력은 모두 완료됐습니다. 명령은 재현 기록이며 현재 경로로 다시 실행하지 않습니다. 별도로 재현할 때는 모든 새 출력과 후속 입력 경로를 일관되게 변경합니다.

```powershell
cd C:\dacon\port

& {
    .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --data runs/ranker_data_fold2 --output runs/ranker_data_tonnage_min_fold2
    if ($LASTEXITCODE -ne 0) { throw "fold2 캐시 변환 실패" }

    .\.venv\Scripts\python.exe -m portmatch train-ranker --data runs/ranker_data_tonnage_min_fold2 --output artifacts/ranker_tonnage_min_fold2 --rounds 300 --threads 6
    if ($LASTEXITCODE -ne 0) { throw "fold2 학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch compare-rankers --baseline artifacts/ranker_no_tonnage_rank_fold2 --experiment artifacts/ranker_tonnage_min_fold2 --output runs/compare_tonnage_min_fold2
    if ($LASTEXITCODE -ne 0) { throw "fold2 비교 실패" }
}
```

`runs/compare_tonnage_min_fold2/report.md`와 `row_changes.csv`가 생성됐으며 결과는 위 잠정 기록에 정리했습니다. 최종 판단에는 완료된 전체 5-fold 결과를 사용했습니다.

### 2. 완료한 나머지 fold와 전체 합산 명령 — 재현 기록

fold2 이후 1·0·3·4 순서의 캐시 변환·학습과 전체 비교까지 완료한 절차입니다. 현재 추가 실행할 fold는 없습니다.

```powershell
cd C:\dacon\port

& {
    foreach ($fold in 1, 0, 3, 4) {
        .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --data "runs/ranker_data_fold$fold" --output "runs/ranker_data_tonnage_min_fold$fold"
        if ($LASTEXITCODE -ne 0) { throw "fold $fold 캐시 변환 실패" }

        .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_tonnage_min_fold$fold" --output "artifacts/ranker_tonnage_min_fold$fold" --rounds 300 --threads 6
        if ($LASTEXITCODE -ne 0) { throw "fold $fold 학습 실패" }
    }

    $baselineModels = @(0..4 | ForEach-Object { "artifacts/ranker_no_tonnage_rank_fold$_" })
    $experimentModels = @(0..4 | ForEach-Object { "artifacts/ranker_tonnage_min_fold$_" })
    .\.venv\Scripts\python.exe -m portmatch compare-rankers --baseline $baselineModels --experiment $experimentModels --output runs/compare_tonnage_min_5fold
    if ($LASTEXITCODE -ne 0) { throw "5-fold 비교 실패" }
}
```

새 환경에서 재현할 경우에만 두 번째 블록의 반복 대상을 `0..4`로 바꾸어 전체 fold를 실행하고 첫 번째 블록은 생략할 수 있습니다. 출력 폴더는 새 폴더 또는 빈 폴더여야 합니다. 현재 실험 결과를 같은 경로에 덮어쓰지 않습니다.

### 비교 해석과 호환성

실제 특징 변환과 전체 학습은 사용자가 VS Code에서 완료했습니다. 이번 결과 확인에는 기존 산출물을 사용했습니다. 구현 확인용 합성 데이터와 짧은 학습의 결과는 대회 성능으로 해석하지 않습니다.

## 완료한 실험의 재현 방법: 톤수 순위 특징 제거

아래 명령은 절차 기록입니다. 현재 fold0~4 실험과 합산 리포트가 모두 완료됐으므로 같은 출력 경로에 다시 실행하지 마세요. 새로 재현할 경우 출력 폴더를 변경합니다.

현재 `runs/ranker_data_fold0`부터 `runs/ranker_data_fold4`까지 특징 생성과 `artifacts/ranker_fold0`부터 `artifacts/ranker_fold4`까지 기준 학습이 완료돼 있습니다. **이번 실험에는 `prepare-ranker`를 다시 실행하지 않습니다.** 기존 70개 특징 캐시에서 `retrieval_tonnage_rank`만 제외한 69개 특징으로 새 모델을 학습합니다. 후보, 정답, 어려운 오답 샘플, 분할과 학습 설정은 그대로 유지합니다.

### 1. 완료한 fold0 학습·비교 명령 — 재현 기록

VS Code PowerShell에서 완료한 명령입니다. 학습이 성공한 경우에만 비교 리포트를 생성하는 절차를 보존합니다.

```powershell
cd C:\dacon\port

.\.venv\Scripts\python.exe -m portmatch train-ranker --data runs/ranker_data_fold0 --output artifacts/ranker_no_tonnage_rank_fold0 --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe -m portmatch compare-rankers --baseline artifacts/ranker_fold0 --experiment artifacts/ranker_no_tonnage_rank_fold0 --output runs/compare_no_tonnage_rank_fold0
}
```

먼저 `runs/compare_no_tonnage_rank_fold0/report.md`를 확인합니다. 이 결과는 **fold0만의 비교**이며 전체 5-fold 결과가 아닙니다. 특징 제거로 어떤 기존 오답이 개선되고 어떤 정답이 새로 틀렸는지 함께 봅니다. 특징을 기존 모델 입력에서 0으로 바꾸는 실험이 아니라, 69개 특징으로 새로 학습하는 실험입니다.

모델과 리포트 출력은 새 폴더 또는 빈 폴더만 허용합니다. 완료된 결과를 같은 경로에 덮어쓰지 않습니다. 이미 새 실험 모델 학습이 끝났고 리포트만 필요하다면 `compare-rankers` 명령만 실행하세요. 중단된 출력은 보존하고 재실행할 출력 이름을 바꿉니다. 비교 명령에도 바뀐 모델 경로를 지정해야 합니다.

### 2. 완료한 나머지 fold 학습·전체 비교 명령 — 재현 기록

fold0 이후 기존 특징 캐시를 재사용해 나머지 fold를 학습하고 전체 비교까지 완료한 절차입니다. 현재 추가 학습을 요청하는 명령이 아닙니다.

```powershell
cd C:\dacon\port

$experimentOk = $true
foreach ($fold in 1..4) {
    .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_fold$fold" --output "artifacts/ranker_no_tonnage_rank_fold$fold" --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
    if ($LASTEXITCODE -ne 0) {
        $experimentOk = $false
        break
    }
}

if ($experimentOk) {
    $baselineModels = @(0..4 | ForEach-Object { "artifacts/ranker_fold$_" })
    $experimentModels = @(0..4 | ForEach-Object { "artifacts/ranker_no_tonnage_rank_fold$_" })
    .\.venv\Scripts\python.exe -m portmatch compare-rankers --baseline $baselineModels --experiment $experimentModels --output runs/compare_no_tonnage_rank_5fold
}
```

일부 fold만 지정해 비교할 수도 있지만, 기준과 실험 양쪽에 같은 fold를 지정해야 합니다. 보고서는 포함한 fold와 빠진 fold를 표시합니다. 경로 이름이나 인자 순서 대신 저장된 fold 정보와 `sample_id`를 사용해 결과를 연결합니다.

### 특징 선택 옵션과 저장 형식

- `--exclude-features`: 지정한 특징만 제외합니다. 예: `--exclude-features retrieval_tonnage_rank retrieval_channel_count`
- `--include-features`: 지정한 특징만 사용합니다. 예: `--include-features name_ratio romanized_ratio phonetic_ratio international_relative_gap`
- 두 옵션은 동시에 사용할 수 없습니다. 옵션을 생략하면 기존처럼 모든 특징을 사용합니다.
- 이름 또는 와일드카드 패턴을 여러 개 지정할 수 있습니다. PowerShell에서는 `--exclude-features 'flag_*'`처럼 패턴을 따옴표로 감쌉니다. 맞는 특징이 없는 이름·패턴, 선택 결과가 비어 있는 설정은 오류로 처리합니다.
- 입력 옵션에 나열한 순서와 관계없이 캐시의 원래 열 순서를 유지합니다. 원본 캐시 파일은 수정하지 않습니다.
- `training_summary.json`의 `feature_selection`에 `source_feature_names`, `selected_features`, `excluded_features`와 요청 패턴을 저장합니다. `ranker.json`에는 실제 학습한 특징 목록을 저장합니다.
- 검증과 `RankedMatcher`/`predict-ranker` 추론에서도 저장한 동일 특징을 선택합니다. 선택 후 특징 이름·순서가 모델 스키마와 일치하는지 확인합니다.

캐시의 열만 선택하는 실험은 특징을 처음부터 생성할 필요가 없습니다. 특징 계산식이나 검색 범위가 바뀌면 별도 버전의 특징 캐시가 필요합니다. 국적·선종 특징을 제외해도 기존 캐시를 만든 어려운 오답 샘플링에는 해당 정보가 사용돼 있습니다. 따라서 이는 **순위 모델 입력의 특징 효과**를 보는 실험이며, 전체 파이프라인에서 해당 정보를 없애는 실험은 아닙니다.

### 비교 리포트 읽기

| 파일 | 내용 |
|---|---|
| `report.md` | 포함 fold, 언어별·가중 성능 변화, 개선·신규 오답, 선박 단위 변화 요약 |
| `summary.json` | 재계산 지표, 차이, 비교 설정·출처, 검증 범위 |
| `row_changes.csv` | 전체 검증 행의 기준/실험 예측과 정답 여부·순위·변화 구분 |
| `vessel_changes.csv` | 정답 선박별 개선·악화 기록과 순변화 |
| `per_fold_metrics.csv` | fold별·언어별 기준/실험 성능과 차이 |

비교는 저장된 요약 점수만 빼는 방식이 아니라 각 모델의 `validation_predictions.csv` 전체에서 다시 계산합니다. 합산 가중 점수는 **전체 행을 합친 한글 정확도 × 0.7 + 비한글 정확도 × 0.3**이며 fold별 점수의 단순 평균이 아닙니다. 차이는 퍼센트포인트(%p)로 읽습니다. 한 언어가 없는 표본에는 가중 점수를 만들지 않습니다.

같은 캐시·원본 출처·분할·학습 설정을 사용한 특징 선택 비교인지 확인하고, 검증 ID 누락·중복·추가, 정답/언어 불일치, 후보 수·정답 후보 포함 여부의 차이는 거부합니다. 다른 설정까지 바꾼 실험을 특징 하나의 효과로 해석하지 않도록 하기 위한 제한입니다.

기존 한글 오답 62건뿐 아니라 전체 검증 행의 개선과 신규 오답을 함께 평가합니다. 선박 한 척의 반복 기록 여러 개가 개선된 경우와 여러 선박의 개선을 구분합니다. 모든 외부 fold를 이미 진단에 사용했으므로 후속 수치는 개발용 검증이며 독립적인 최종 성능 추정은 아닙니다. 현재 구현 확인용 작은 테스트 결과를 실험 성능으로 사용하지 않습니다.

## 원래 70개 모델 재현 — 새 환경의 fold0

아래는 특징 캐시와 원래 70개 모델이 아직 없는 환경에서 초기 실험을 재현하는 절차입니다. 현재 채택한 86개 모델은 위의 이름 특징 변환·톤수 순위 제외 절차를 사용합니다. **현재 프로젝트의 완료된 fold0~4 폴더에는 다시 실행하지 마세요.**

VS Code의 PowerShell에서 실행하세요. 이 프로젝트의 가상환경에는 LightGBM 4.6.0까지 설치되어 있습니다. 다른 환경에서는 `py -3.11 -m venv .venv` 후 `pip install -r requirements.txt`가 필요합니다.

```powershell
cd C:\dacon\port

.\.venv\Scripts\python.exe -m portmatch prepare-ranker --fold 0 --output runs/ranker_data_fold0
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe -m portmatch train-ranker --data runs/ranker_data_fold0 --output artifacts/ranker_fold0 --rounds 300 --threads 6
}
```

첫 단계는 특징을 생성하고 저장합니다. 두 번째 단계는 저장한 훈련 특징으로 모델을 학습한 뒤, 전체 검증 후보를 평가합니다. 준비 단계가 끝났다면 모델 설정을 바꿀 때 특징을 다시 생성하지 않아도 됩니다. 예를 들어 `--rounds 500`으로 비교하려면 `train-ranker`만 새 출력 폴더로 실행하면 됩니다.

`--max-train-queries`, `--max-valid-queries`는 동작 확인용 표본 제한입니다. 실제 성능 실험에서는 지정하지 마세요. 모든 출력은 새 폴더 또는 빈 폴더를 사용합니다. 중단된 준비 폴더는 학습에 사용할 수 없습니다. 폴더 이름을 바꾸고 다시 실행하세요.

## 누수를 막는 방식

1. **외부 검증 분리:** `--fold 0`이면 기존 선박 5-fold의 fold0을 최종 검증으로 보류합니다. 해당 선박의 정답은 모든 훈련 사전·별칭 인덱스·국적/선종 통계와 순위 모델에서 제외됩니다.
2. **훈련 내부 분할:** 나머지 훈련 데이터에서 다시 3-fold를 만듭니다. 각 질의의 특징은 자신을 제외한 내부 훈련 부분으로 생성합니다.
3. **두 가지 훈련 상황:** 기본 `--inner-split mixed`는 선박 단위 분리와 행 단위 분리를 각각 수행합니다. 선박 분리는 처음 보는 선박을, 행 분리는 훈련에 다른 입항 기록이 있는 선박을 학습합니다. 행 분리에서도 현재 질의와 외부 검증 행은 사전에 포함하지 않습니다.
4. **학습 중 외부 검증 미사용:** 고정 횟수로 학습하고 검증은 학습 완료 후 별도로 수행합니다. 외부 검증 결과를 사용한 조기 종료는 하지 않습니다.

mixed 모드에서는 원본 훈련 질의 하나가 최대 두 개의 순위 학습 그룹이 됩니다. `query_id`에 분리 방식을 붙여 구별하며, 같은 질의의 후보들은 한 그룹으로 유지됩니다. `--inner-split vessel` 또는 `row`로 하나만 생성할 수도 있습니다. 선박 분리만 사용하면 호출부호 조회가 없는 상황만 학습한다는 점에 유의하세요.

외부 검증 후보의 특징은 외부 훈련 전체로 만든 고정 인덱스를 사용합니다. 카탈로그 전체는 모든 단계에서 후보 모집단으로 사용합니다. `sample_id`, 선박 ID, 정답, 그룹 ID는 모델 특징에 넣지 않습니다.

일반적으로 한 fold를 반복해서 보고 설정을 고르면 그 fold는 개발용 검증이 됩니다. 독립 성능 판단에는 설정 선택에 사용하지 않은 평가 데이터가 필요합니다. 현재는 5개 fold를 모두 개발에 사용했으므로, 합산 점수도 개발용 검증 결과입니다.

## 어려운 오답 샘플링

훈련 질의당 기본 최대 **129개 후보**를 저장합니다.

- 정답 후보 1개
- 어려운 오답 최대 96개: 거짓 호출부호·이름 조회 후보를 우선하고, 이름 편집거리·로마자·음차·톤수·국적·선종별 순위를 번갈아 사용
- 나머지에서 무작위 오답 최대 32개

동점과 무작위 선택은 seed 및 질의 ID로 고정됩니다. 질의 처리 순서나 후보 행 순서를 바꿔도 선택되는 선박은 같습니다. 후보가 129개보다 적으면 모두 유지합니다.

검색에서 정답이 누락된 훈련 질의는 정답을 강제로 넣지 않고 기록한 뒤 학습에서 제외합니다. 오답이 없는 그룹도 제외합니다. 제외 수는 `metadata.json`과 `querylog.csv`에 남습니다. **검증은 오답 샘플링도 후보 상한도 적용하지 않습니다.** 정답이 후보에 없으면 최종 정확도와 모든 Recall@K에서 실패로 계산합니다.

후보 전체에 특징을 계산해야 어려운 오답을 고를 수 있으므로, 준비 단계는 이전 검색 전용 검증보다 오래 걸립니다. `mixed`도 원본 훈련 질의를 두 번 처리합니다. 대용량 CSV 대신 64개 질의씩 압축 NPZ로 나누어 저장합니다. 훈련 특징은 샘플링 후 저장하고, 검증 특징은 전체를 저장합니다. 데이터 저장 공간과 학습 중 모델 메모리는 별도로 필요합니다.

## 모델과 지표

LightGBM 4.6.0의 `LGBMRanker(objective="lambdarank")`를 사용합니다. 특징 결측은 NaN으로 유지합니다. 입력 특징의 이름과 순서가 달라지거나 ID·정답 열이 섞이면 학습·예측을 거부합니다. 모델은 native `model.txt`로 저장하고 해시와 특징 스키마를 함께 검증합니다. [LightGBM 공식 문서](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)

훈련 가중치는 같은 언어 안에서 질의별 총량을 같게 하고, 한글/비한글의 전체 행 가중치 합이 70:30이 되도록 구성합니다. 이는 학습 휴리스틱이며 실제 LambdaRank 기울기 기여를 정확히 70:30으로 보장하는 것은 아닙니다. 성능 지표는 별도로 질의별 정답 여부를 집계합니다.

| 파일 | 확인할 내용 |
|---|---|
| `artifacts/ranker_fold0/metrics.json` | 한글·비한글 Top-1 정확도, Recall@5·10·50·100·200, 후보 재현율, `weighted_top1` |
| `artifacts/ranker_fold0/validation_predictions.csv` | 질의별 정답/예측 선박, 정답 순위, 후보 수, 상위 점수와 점수 차이 |
| `artifacts/ranker_fold0/feature_importance.csv` | 특징별 gain/split 중요도 |
| `artifacts/ranker_fold0/training_summary.json` | 학습 설정, 데이터 출처·설정, 시간, 완료 여부 |
| `runs/ranker_data_fold0/partitions.csv` | 외부·내부 fold 배정, 훈련/검증 역할 |
| `runs/ranker_data_fold0/querylog.csv` | 질의별 전체/선택 후보 수, 정답 누락, 제외 사유, 사전 생성에 사용한 행 수 |

`weighted_top1 = 0.7 × 한글 Top-1 정확도 + 0.3 × 비한글 Top-1 정확도`가 핵심 지표입니다. `weighted_candidate_recall`은 후보 재현율이므로 구분하세요. 한 언어가 없는 작은 표본에서는 가중 지표를 계산하지 않습니다. 모델 점수와 점수 차이는 확률이 아닙니다.

## 원래 70개 모델 재현 — 다른 fold와 행 분할

아래는 원래 70개 모델의 선박·행 분할 재현 명령이며 아직 캐시가 없는 환경용입니다. 현재 fold1~4 캐시는 이미 완료돼 있으므로 특징 선택 실험에는 문서 위쪽의 캐시 재사용 명령을 사용하세요. 86개 모델에는 이름 특징 변환과 톤수 순위 제외가 추가로 필요합니다. 새 데이터나 검색·특징 계산으로 캐시를 만드는 경우에는 각 fold마다 별도 특징 생성과 학습이 필요합니다.

```powershell
foreach ($fold in 1..4) {
    .\.venv\Scripts\python.exe -m portmatch prepare-ranker --fold $fold --output "runs/ranker_data_fold$fold"
    if ($LASTEXITCODE -ne 0) { break }
    .\.venv\Scripts\python.exe -m portmatch train-ranker --data "runs/ranker_data_fold$fold" --output "artifacts/ranker_fold$fold" --threads 6
    if ($LASTEXITCODE -ne 0) { break }
}
```

재등장 선박에 대한 조회 효과는 외부 행 분할로 별도 확인할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m portmatch prepare-ranker --split row --fold 0 --output runs/ranker_data_row0
if ($LASTEXITCODE -eq 0) {
    .\.venv\Scripts\python.exe -m portmatch train-ranker --data runs/ranker_data_row0 --output artifacts/ranker_row0
}
```

## 저장한 모델로 예측

모델 폴더에는 native 순위 모델과 해당 외부 훈련 부분으로 만든 `index.joblib`가 함께 들어 있습니다. 이 프로젝트가 만든 신뢰할 수 있는 인덱스만 로드하세요.

```python
from portmatch.ranker_commands import RankedMatcher

matcher = RankedMatcher("artifacts/ranker_token_phonetic_fold0")
vessel_id = matcher.predict(query_dict)
ranked_candidates = matcher.rank(query_dict)
```

실제 질의 파일이 있을 때 CLI로도 실행할 수 있습니다. `YOUR_QUERY_FILE.csv`를 실제 경로로 바꾸세요.

```powershell
.\.venv\Scripts\python.exe -m portmatch predict-ranker --model artifacts/ranker_token_phonetic_fold0 --queries YOUR_QUERY_FILE.csv --output runs/predictions.csv
```

예측 CSV는 `sample_id,target_vessel_id` 두 열입니다. 질의는 한 행씩 고정 인덱스로 처리하고, 평가 행끼리 통계·투표·사전을 공유해 갱신하지 않습니다. 동점은 고정된 선박 ID 순서로 결정합니다.

현재 모델은 해당 fold를 제외한 데이터로 학습한 모델입니다. 고정 86개 fold0 모델의 첫 제출 ZIP을 생성했으며, 전체 데이터로 최종 재학습하는 작업은 별도입니다. 작은 smoke 실험 결과는 성능 추정으로 사용하지 마세요.

## 첫 제출 ZIP 재현

현재 파일은 `submissions/baseline86_fold0/submit.zip`입니다. 아래는 이미 만든 파일을 다시 실행할 필요가 없는 재현용 명령입니다. 빌더는 기존 ZIP을 덮어쓰지 않으며 검증 도구도 새 출력 폴더만 허용합니다.

```powershell
& {
    .\.venv\Scripts\python.exe -m portmatch.submission_package --model artifacts/ranker_token_phonetic_fold0 --output submissions/baseline86_fold0_reproduced/submit.zip
    if ($LASTEXITCODE -ne 0) { throw "제출 ZIP 생성 실패" }

    .\.venv\Scripts\python.exe scripts/verify_submission.py --zip submissions/baseline86_fold0_reproduced/submit.zip --output work/submission_validation/baseline86_fold0_reproduced --reference-model artifacts/ranker_token_phonetic_fold0
    if ($LASTEXITCODE -ne 0) { throw "제출 ZIP 검증 실패" }
}
```

특징 생성·학습을 다시 하지 않습니다. 패키징은 저장 모델과 출처 메타데이터·코드 해시를 확인하며, 검증은 실제 후보 검색·특징 계산까지 수행하므로 수 분 이상 걸릴 수 있습니다. 공개 학습 파일의 fold0 검증 행만 실행 입력으로 만들고 정답 열은 제출 프로세스에 넘기지 않습니다. 출력 형식·행별 독립성·실측 시간과 Linux 검증 여부는 [첫 제출 준비 상태](docs/submission-readiness.md)를 확인하세요.
