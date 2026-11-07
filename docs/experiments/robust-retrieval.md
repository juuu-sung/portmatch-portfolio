# 한글 이름·음차 검색과 두 총톤수 검색 결합

2026-10-23 구현. **기존 86개 모델은 대조군으로 유지하며, 새 검색 방식의 채택은 아직 결정하지 않았습니다.** 긴 특징 생성·학습은 사용자가 VS Code에서 실행합니다. 첫 제출 ZIP도 그대로 보존합니다.

**최신 상태 2026-10-24: [robust 재학습 전체 비교](robust-retrained-fold0.md)까지 완료했고 현재 모델은 미채택, fold1~4 확대는 보류합니다.** 원본·국제 GT 결측의 순위 손실이 늘었습니다. 다음은 검색·86개 특징을 고정한 훈련 GT 결측 증강의 별도 구현이며 아직 미구현입니다. [이전 검색 교체 분석](robust-retrieval-fold0.md)과 아래 완료 명령은 재현 기록으로 보존하며 같은 폴더에 재실행하지 않습니다.

[README](../../README.md) · [학습 안내](../../TRAINING.md) · [첫 Public 진단](first-public-submission.md)

## 실험 목적과 구현

첫 제출 Public 점수는 0.7913544371입니다. 공개 입력에 톤수 결측·오차를 준 이전 진단에서는 한글 정답 후보가 누락됐습니다. 실제 비공개 오답 원인은 알 수 없으므로, 이번에는 **원본 성능과 변형 입력의 검색·순위 성능, 후보 수, 비용을 함께 비교**합니다.

| 실행 이름 | 바꾸는 내용 | 유지하는 내용 |
|---|---|---|
| `baseline` | 없음 | 기존 86개 순위 모델·기존 검색 |
| `hangul` | 한글 문맥 음차·모음 표현·편집거리 검색 추가 | 기존 톤수 검색·86개 모델 가중치 |
| `dual-gt` | 일반·국제총톤수의 후보를 각각 찾고 합침 | 기존 이름 검색·86개 모델 가중치 |
| `robust` | 위 두 변경을 함께 적용 | 86개 모델 가중치 |
| `trained` | 새 후보로 내부 OOF 특징·오답 샘플을 다시 만들고 재학습 | 외부 분할·86개 특징 순서·학습 설정 |

`robust_retrieval.py`는 기존 검색 모듈과 분리했습니다. CLI 기본값은 계속 `legacy`이며 기존 특징 계산 파일의 바이트와 완료 캐시·모델은 바꾸지 않습니다.

- **한글 이름 검색:** 카탈로그 이름과 해당 훈련 부분의 관측 별칭을 문맥 음차·CH 대체 표현·모음 보존 표현으로 변환합니다. 접미 `호` 보조 표현도 사용합니다. 문맥 TF-IDF 최대 160척, 모음 TF-IDF 최대 160척, 표현 편집 검색 최대 80척을 기존 후보에 합칩니다. 쿼리의 톤수로 이 경로를 제한하지 않습니다. 특정 정답 선박의 예외 사전은 없습니다.
- **두 톤수 검색:** 서로 다른 유효한 두 GT 각각에 카탈로그 GT 대비 ±10% 범위 검색을 적용합니다. 각 범위가 비면 해당 GT의 최근접 후보를 사용하고 기존 후보와 합칩니다. 한쪽만 유효하거나 두 값이 같으면 기존 톤수 검색과 같습니다. DWT는 GT로 사용하지 않습니다.
- **후보 보존:** 기본값에서는 기존 후보를 모두 남깁니다. 새 이름 검색 결과는 기존 `phonetic` 경로에 최고 유사도로 병합하고 순위를 다시 계산합니다. 특징 개수는 같지만 일부 검색 특징의 의미가 바뀌므로 `retrieval_extension` 버전·코드 해시·전체 설정·저장 인덱스 해시를 기록하고 검증합니다.
- **누수·독립성:** 모든 내부 OOF 및 외부 검증용 인덱스는 해당 훈련 부분만 사용합니다. 평가 입력으로 사전·통계를 갱신하지 않습니다. 비교 시 정답·sample ID는 추론 함수에 넘기지 않고 평가와 행 연결에만 사용합니다.

## 비교 조건과 출력

`compare-retrieval`은 저장된 모델의 외부 분할을 복원하며, 기본적으로 **해당 fold 전체**를 실시간 추론합니다. 후보·특징·점수 캐시를 읽어 시간을 대체하지 않습니다. 원본 대조군의 예측·정답 순위·후보 수가 기존 `validation_predictions.csv`와 일치하지 않으면 실패합니다.

다음 다섯 입력을 모든 방법에 똑같이 적용합니다.

| 변형 | 적용 |
|---|---|
| `original` | 원본 |
| `both_gt_missing` | 두 GT를 모두 결측 처리 |
| `international_gt_missing` | 국제총톤수만 결측 |
| `gross_gt_missing` | 일반총톤수만 결측 |
| `international_gt_x1_25` | 유효한 국제총톤수만 1.25배, 일반총톤수 유지 |

이미 결측인 값은 변형 전후가 같을 수 있어 실제 변경/유지 행 수를 언어별로 기록합니다. 이 입력 변형은 개발용 강건성 점검이며 실제 비공개 입력을 복원한 것이 아닙니다. 비교는 **같은 변형의 baseline 대 실험**으로 합니다.

| 파일 | 내용 |
|---|---|
| `report.md` | 조건별 가중·한글·비한글 Top-1, 한글 후보 재현율, 후보 수·시간·개선/신규 오답 |
| `summary.json` | 완료 여부, 입력·모델·코드 해시, 분할·설정, 실제 변형 행 수, 언어별 Top-1/Recall@K/후보 재현율, 후보 평균·p50·p95·최대, 검색/특징/순위/합계 시간, 초기화 시간, 지표 차이·시간 비율 |
| `predictions.csv` | 모든 조건·방법의 행별 예측, 정답 순위, 후보 포함 여부·수·후보 ID 집합 해시, 단계별 시간 |
| `row_changes.csv` | 같은 변형에서 개선·신규 오답·기존 오답 유지, 후보 포함 여부·수·순위의 변화 |
| `vessel_changes.csv` | 선박별 개선·신규 오답·순개선, 후보 수 변화 |

`candidate_found=false`는 검색 실패, `candidate_found=true`이면서 예측이 정답과 다르면 순위 오류입니다. 후보가 늘면 원래 정답이었던 행이 오답으로 바뀔 수도 있으므로 재현율만으로 채택하지 않습니다.

시간은 실제 검색·특징 계산·순위 계산을 측정하고 모델 로드·인덱스 구축·고정 이름 특징 준비는 `startup`에 분리합니다. 행마다 방법의 실행 순서를 순환하지만 같은 프로세스의 문자열·운영체제 캐시 효과는 남습니다. **이 시간은 제출 ZIP의 서버 실행 시간이나 30분 제한 통과 보증이 아닙니다.** 공정한 비교를 위해 다른 특징 생성·학습과 동시에 돌리지 않습니다.

## 1. 완료한 기존 모델 검색 비교 — 재현 기록

VS Code PowerShell에서 실행합니다. 새 폴더 또는 빈 폴더만 사용하며 완료·중단된 폴더를 덮어쓰거나 자동 이어쓰기하지 않습니다.

아래 **학습 없는 fold0 전체 비교는 완료**했습니다. 2,452행 × 5개 변형 × 4개 방법을 실제 추론했습니다. 별도로 재현할 때는 새 출력 이름을 사용합니다.

```powershell
cd C:\dacon\port

.\.venv\Scripts\python.exe -m portmatch compare-retrieval --baseline artifacts/ranker_token_phonetic_fold0 --profiles hangul dual-gt robust --output runs/compare_robust_retrieval_fold0
```

더 작은 선행 점검을 원하면 아래처럼 한글·비한글 각각 최대 48척에서 한 행씩 선택합니다. 선박·행 선택은 고정 해시로 결정되며, 이전 첫 제출 진단의 96척과 같은 표본은 아닙니다. 표본 결과를 전체 성능으로 해석하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m portmatch compare-retrieval --baseline artifacts/ranker_token_phonetic_fold0 --profiles hangul dual-gt robust --max-vessels-per-language 48 --one-row-per-vessel --selection-seed 20261023 --output runs/compare_robust_retrieval_sample_fold0
```

당시 검색 교체 전체 비교에서 두 보강의 상호 보완성을 확인해 `robust`를 재학습 대상으로 선택했습니다. 이후 재학습 비교까지 완료했으며 최신 결정은 상단과 [재학습 분석](robust-retrained-fold0.md)을 따릅니다.

## 2. 완료한 새 후보 86개 재학습 — 실행 기록

검색이 바뀌면 후보, 검색 특징과 어려운 오답 샘플도 달라집니다. **기존 `runs/ranker_data_fold0` 또는 `ranker_data_token_phonetic_fold0`를 새 검색의 학습 캐시로 재사용할 수 없습니다.** 새로운 70열 OOF 캐시부터 생성하고 토큰·문맥 이름 특징을 추가해 87열로 변환한 뒤 톤수 순위 하나를 제외합니다.

아래는 **완료한 `robust` fold0 재학습·비교 명령의 기록**입니다. 기존 기준과 동일하게 외부 선박 5-fold, 내부 mixed 3-fold, seed 20261022, 300회·6스레드를 사용했습니다. 톤수 결측·오차 **훈련 증강은 이 실험에 포함하지 않았습니다.**

```powershell
cd C:\dacon\port

& {
    .\.venv\Scripts\python.exe -m portmatch prepare-ranker --fold 0 --retrieval-profile robust --output runs/ranker_data_robust_fold0
    if ($LASTEXITCODE -ne 0) { throw "새 검색 OOF 특징 생성 실패" }

    .\.venv\Scripts\python.exe -m portmatch transform-ranker-cache --transform token-phonetic --data runs/ranker_data_robust_fold0 --output runs/ranker_data_robust_token_phonetic_fold0
    if ($LASTEXITCODE -ne 0) { throw "토큰·문맥 이름 특징 변환 실패" }

    .\.venv\Scripts\python.exe -m portmatch train-ranker --data runs/ranker_data_robust_token_phonetic_fold0 --output artifacts/ranker_robust_token_phonetic_fold0 --exclude-features retrieval_tonnage_rank --rounds 300 --threads 6
    if ($LASTEXITCODE -ne 0) { throw "86개 순위 모델 학습 실패" }

    .\.venv\Scripts\python.exe -m portmatch compare-retrieval --baseline artifacts/ranker_token_phonetic_fold0 --profiles robust --experiment-model artifacts/ranker_robust_token_phonetic_fold0 --output runs/compare_robust_retrained_fold0
    if ($LASTEXITCODE -ne 0) { throw "재학습 모델 비교 실패" }
}
```

완료한 단계는 건너뛰고 다음 단계부터 실행합니다. 특징 생성의 완료 여부는 `metadata.json`의 `complete=true`로 확인합니다. 처음부터 블록을 다시 실행하면 기존 출력 보호 오류가 발생합니다.

`trained`는 후보 변화와 재학습 효과를 함께 포함합니다. 동일 가중치의 `robust`와 나눠 봅니다. 기존 `compare-rankers`는 같은 후보·캐시를 전제로 하므로 이 실험에는 사용하지 않습니다. 비교 도구는 동일 분할·입력 해시·훈련 사전·범주 통계·기존 검색 설정·학습 설정·86개 특징 순서를 검사합니다. 설정을 바꾼 모델은 별도의 통제 실험이 필요합니다.

`hangul` 또는 `dual-gt`만 학습하려면 `--retrieval-profile`과 모든 새 출력 이름을 일관되게 바꿉니다. 기본 추가 검색량은 `--context-top-k 160 --vowel-top-k 160 --context-edit-top-k 80`입니다. 검색량 조정은 별도 실험이며 새 캐시가 필요합니다. 이번에 검증한 파생 경로는 `token-phonetic`이며 동점 보정 톤수 순위 캐시와 결합하지 않습니다.

## 구현 확인과 다음 판단

구현 당시 **전체 회귀 테스트 197개를 통과**했습니다. 이후 검색 교체·재학습 비교까지 완료했으며, 비교기의 집합 해시 오판 수정 시점에는 203개 테스트를 통과했습니다. 합성 데이터로 OOF 분리, 캐시 70→87→86 연결, 복원 예측, 행 독립성, 후보 보존, 메타데이터 변경 거부를 검증했습니다. 합성 24행으로 두 모델을 3회씩만 학습한 통합 검사에서는 5개 입력 조건 × 5개 방법 × 검증 8행의 200개 실시간 예측과 재학습 비교 산출물도 확인했습니다. 기존 86개 제출 ZIP의 재생성·별도 프로세스 추론 테스트도 통과했습니다. 아래 8척 결과는 초기 연결 점검의 기록입니다.

실제 fold0에서 한글 4척·비한글 4척, 총 8행에 5개 변형·4개 방법을 적용한 연결 점검을 완료했습니다. 160회 행 추론과 초기화를 포함해 40.33초였고, 원본 baseline 8행은 저장된 예측·후보 수·정답 순위와 모두 일치했습니다. 아래 숫자는 **8행 점검 결과이며 성능 추정이나 채택 근거가 아닙니다.**

| 조건 | 한글 후보 포함 baseline→robust | 한글 정답 baseline→robust | 전체 평균 후보 baseline→robust | 8행 추론 초 baseline→robust |
|---|---:|---:|---:|---:|
| 원본 | 4/4 → 4/4 | 4/4 → 4/4 | 2,627.1 → 2,769.8 | 0.760 → 1.188 |
| 두 GT 결측 | 1/4 → 3/4 | 1/4 → 3/4 | 130.0 → 279.8 | 0.333 → 0.748 |
| 국제 GT 1.25배 | 1/4 → 4/4 | 1/4 → 2/4 | 2,446.5 → 5,079.0 | 0.690 → 1.405 |

비한글 4행은 모든 조건·방법에서 정답이었습니다. 국제 GT 변형의 `robust`는 개선 2행과 신규 오답 1행이 함께 발생했습니다. **정답 후보 회복이 최종 1위 회복을 보장하지 않는다는 점과 후보 비용 증가**를 실제 연결 점검에서도 확인했습니다. 로컬 근거는 `runs/compare_robust_retrieval_smoke_fold0/`입니다.

fold0 검색 교체와 재학습의 원본 신규 오답·선박별 변화, 변형별 후보 회복·순위 오류, 후보 분포·시간을 검토했습니다. [재학습 분석](robust-retrained-fold0.md)에 따라 현재 설정을 미채택하고, 훈련 GT 결측 증강을 fold0에서 먼저 검증한 뒤 확대를 판단합니다. 개발 fold는 이미 반복 사용했으므로 독립 평가로 해석하지 않습니다.

새 검색 모델의 제출 패키징은 별도 실행 검증이 필요합니다. 현재 ZIP 빌더는 이 실험 모델을 거부하며, 검증 완료한 기존 86개 모델의 패키징을 유지합니다. 이번 작업에서 새 ZIP이나 서버 제출은 만들지 않았습니다.
