# 원본 질량 0.75 재가중치 — 구현과 완료한 실행 기록

> 공개용 정리본: 실제 선박별 입력·정답·예측·점수와 행 식별자는 제외했다. 아래 성능표는 실제 실험의 집계 수치이며, 오류의 원리는 별도의 [가상 예시](../fictional-examples.md)로 설명한다.

후속 상태(2026-10-26): **0.75 재가중치의 전체 5-fold 실행·분석을 완료했고 채택하지 않습니다.** 원본 회복이 일부 선박에 집중되고 0.5 대비 톤수 변형 손실이 확대됐습니다. 현재 사용자 보고 서버 최고인 **0.5 fold0·0.8424607623**을 유지하고 질량 비율 추가 탐색은 종료합니다. [전체 결과와 최종 판단](gt-reweighting-5fold.md). 아래 10월 25일 설계·fold1 결과와 명령은 당시 기록입니다.

2026-10-25. **구현과 실제 fold1 재가중치·학습·비교·분석을 완료**했습니다. 원본98.9092%(0.5 대비+1.0347%p)이나 변형 일부 손실로 당시 채택을 보류하고 같은 설정의 fold0·2·3·4 검증을 권했습니다. [fold1 당시 결과와 완료한 후속 명령](gt-reweighting-fold1.md). 아래는 구현 설계와 완료한 fold1의 실행 기록입니다. [0.5 증강 전체 5-fold 결과](gt-missing-5fold.md)의 원본 손실과 변형 이득을 확인하기 위한 단일 변경입니다. 기존 86개 기준과 0.5 모델·캐시는 보존합니다. 0.75는 사전에 선택한 실험값이며 최적값으로 확정한 값이 아닙니다.

## 바꾸는 것과 보존하는 것

`reweight-ranker-cache`는 완료된 robust + GT 결측 증강 + token-phonetic **87열 캐시**를 새 폴더로 복사하고, 훈련 NPZ의 `query_mass`만 재계산합니다. 검색·특징 계산·오답 샘플링·인덱스 fitting을 다시 하지 않습니다.

- 후보 ID·순서·특징 X의 바이트·라벨·그룹·질의/원본 ID·언어·증강 종류 등 나머지 배열을 그대로 유지합니다.
- 원본 검증 NPZ, 고정 인덱스, 카탈로그 ID, 분할, querylog, 기존 특징 변환 증명을 파일 그대로 복사합니다. 원본 파일에 쓰거나 하드링크를 만들지 않습니다.
- 원본당 총 질량 1과 언어별 질량 합을 유지합니다. 보존된 view에 균등 배분하고, 원본과 마스크가 함께 남은 view에서 원본 0.75·마스크 합계 0.25를 적용합니다. 두 view와 세 마스크가 모두 남으면 원본은 각 view 3/8, 마스크는 각각 1/24입니다. 원본만 남으면 해당 view 질량을 모두 원본에, 원본이 없으면 남은 마스크에 균등 배분합니다. 실제 저장 질량의 원본 비율이 정확히 75%라는 뜻은 아닙니다.
- augmentation 설정·진단·훈련 group digest를 함께 갱신합니다. `training_reweighting`과 `reweighting.json`에는 원본 메타데이터/파일 해시, 변경 파일 해시, 질량 외 배열 증명을 별도로 저장합니다. 기존 특징 변환의 출처도 검증합니다.

동일하거나 중첩된 입력/출력 경로, 비어 있지 않은 출력, 미완료/비증강/다른 특징 캐시, 변경하지 않은 질량, 재가중치 캐시의 연속 재가중치는 거부합니다. 중단된 출력에 이어쓰는 기능은 없습니다. 완료한 단계는 다시 실행하지 않고 다음 단계부터 진행합니다.

학습 전에 원본/파생 파일과 배열 보존 증명을 다시 검사합니다. 모델은 갱신한 질량을 기존 언어 균형·후보 수 보정에 전달하고, 저장 모델의 추론에는 훈련 캐시가 필요하지 않습니다. 검증 자료나 테스트 행을 새 fitting에 사용하지 않습니다.

## VS Code PowerShell 명령

아래 명령은 **이미 완료한 fold1의 재현 기록**입니다. 같은 출력 폴더에 다시 실행하지 않습니다. fold0·2·3·4의 [별도 반복문](gt-reweighting-fold1.md#판단과-다음-실행)도 실행·분석이 완료된 재현 기록입니다. 추가 실행할 fold는 없습니다. 기존 0.5 캐시·모델과 증강 없는 robust 대조군이 완료돼 있으므로 `prepare-ranker`나 이름 특징 변환부터 다시 할 필요가 없습니다. 재가중치도 전체 캐시 읽기·압축·무결성 검사에 시간이 들며, 새 캐시 용량이 추가로 필요합니다.

```powershell
cd C:\dacon\port

& {
    .\.venv\Scripts\python.exe -m portmatch reweight-ranker-cache --data runs/ranker_data_robust_gt_missing_token_phonetic_fold1 --output runs/ranker_data_robust_gt_missing_mass075_token_phonetic_fold1 --original-mass 0.75
    if ($LASTEXITCODE -ne 0) { throw "재가중치 실패" }

    .\.venv\Scripts\python.exe -m portmatch train-ranker --data runs/ranker_data_robust_gt_missing_mass075_token_phonetic_fold1 --output artifacts/ranker_robust_gt_missing_mass075_token_phonetic_fold1 --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
    if ($LASTEXITCODE -ne 0) { throw "학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch compare-retrieval --baseline artifacts/ranker_token_phonetic_fold1 --profiles robust --reference-model artifacts/ranker_robust_token_phonetic_fold1 --control-model artifacts/ranker_robust_gt_missing_token_phonetic_fold1 --experiment-model artifacts/ranker_robust_gt_missing_mass075_token_phonetic_fold1 --output runs/compare_robust_gt_missing_mass075_fold1
    if ($LASTEXITCODE -ne 0) { throw "비교 실패" }
}
```

훈련은 87열 중 `retrieval_tonnage_rank`를 제외한 기존 **86개**를 사용합니다. 0.5 대조군과 같은 300회·6스레드·seed를 유지합니다. 명령 실행 중 출력 폴더를 다시 사용하지 않습니다. 실패 후에는 오류를 확인하고 완료된 앞 단계를 생략합니다.

## 비교 결과 읽기

원본과 두 GT 결측·국제만 결측·일반만 결측·국제 GT ×1.25의 다섯 입력을 평가합니다. 전체 fold1 2,406행 × 다섯 입력 × 다섯 실행 방식 = **60,150개 예측**입니다.

| 실행 이름 | 역할 |
|---|---|
| `baseline` | 기존 검색·86개 모델 |
| `robust` | baseline 가중치를 유지하고 검색만 robust로 교체 |
| `reference` | 증강 없이 재학습한 robust 모델 |
| `control` | 기존 원본 질량 0.5 증강 모델 |
| `trained` | 동일 캐시를 원본 질량 0.75로 바꿔 학습한 모델 |

`--control-model`은 학습 전에 검증한 출처 메타데이터와 대조군의 캐시 해시를 대조하여 바로 그 캐시에서 파생된 모델인지 확인합니다. 비교 단계에서 훈련 캐시 전체를 다시 읽지는 않습니다. 분할·검색·특징·샘플·학습 설정을 고정하며 원본의 저장 예측과 실행 예측도 대조합니다. `reference/control/trained`의 동일 입력 후보 집합·개수·정답 포함 여부는 같아야 합니다.

- `report.md`, `summary.json`: 언어별/가중 정확도, 후보 재현율·후보 수·시간. `trained_vs_control`이 0.5→0.75의 직접 효과입니다.
- `trained_vs_control_row_changes.csv`, `trained_vs_control_vessel_changes.csv`: 0.5 대비 개선·신규 오답.
- `trained_vs_reference_*`: 증강 없는 robust 대비 변화. 질량 변경 단독 효과와 구분합니다.
- `row_changes.csv`, `vessel_changes.csv`: 기존 86개 기준 대비 변화.

## 제출과의 관계

실제0.75 전체5-fold 학습·분석을 완료했고 미채택했습니다. [최종 분석](gt-reweighting-5fold.md). 두 번째 제출은 **5-fold 검증한 robust + 0.5 증강의 고정 fold0 모델**이며 사용자 보고 점수0.8424607623(첫 제출 대비+5.1106%p)입니다. 첫 제출과 같은 fold로 검색·증강 변화의 영향을 확인하는 목적이며 정식 기준 채택이나 전체 데이터 최종 학습과 다릅니다. [두 번째 ZIP의 실제 검증 기록](second-submission.md)을 참고하세요. 현재 패키저는 이 0.5 프로필을 `--allow-robust`로 명시할 때만 허용하며 새 재가중치 모델 패키징은 별도 검증 전까지 거부합니다.

## 구현 검증

합성 데이터로 질량 외 배열의 비트 동일성(NaN 표현·음수 0 포함), 원본 파일 불변, 정답/후보/검증 보존, 제외된 원본·마스크의 재분배, 변조·다른 대조군 거부, 0.75 가중치의 실제 학습 전달과 다섯 방식 비교를 점검했습니다. 전체 회귀249개와 마지막에 추가한 미지원 패키지 설정 거부1개, 총250개를 통과했습니다. 기존 15개 모델의 특징·증강·가중치 메타데이터 호환성과 보호 대상 특징/검색/증강/학습 소스 14개의 기존 커밋 대비 바이트 동일성도 확인했습니다. 이는 구현 시점의 검증 기록입니다. 이후 사용자가 실제 fold1 재가중치·본 학습·비교를 완료했고, 이번 분석에서 실제1,213자산·1,207 NPZ의 보존을 재확인했습니다.

테스트는 쓰기 가능한 로컬 임시 폴더를 지정해 실행합니다. 이 환경의 기본 임시 폴더에서는 기존 출력 파일의 원자적 교체가 거부돼, 아래처럼 임시 경로를 지정한 뒤 회귀 검사를 통과했습니다.

```powershell
New-Item -ItemType Directory -Path work/test_tmp_reweight -Force | Out-Null
$env:TEMP = (Resolve-Path work/test_tmp_reweight).Path
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```
