# 두 번째 서버 진단 — robust·86개·GT 결측 증강 0.5

> 공개용 정리본: 실제 선박별 입력·정답·예측·점수와 행 식별자는 제외했다. 아래 성능표는 실제 실험의 집계 수치이며, 오류의 원리는 별도의 [가상 예시](../fictional-examples.md)로 설명한다.

후속(2026-10-27): [전체 공개훈련 후속 제출 점수0.8324345055](../full-training.md#public-결과와-다음-진단)를 사용자 보고로 기록했고 전체훈련 ZIP 제출 경로도 사용자에게 확인했습니다. 제출번호·서버 시간은 미확인입니다. 아래0.5 fold0의0.8424607623보다−1.00262568%p이므로 현재 최고는 이 fold0 ZIP으로 유지하고 전체훈련 모델로의 교체를 보류합니다. 전체모델의48.11MB ZIP·로컬6조건 검증·12,222행1,281.59초 기록은 보존합니다.

후속 상태(2026-10-26): **0.75 전체 5-fold 분석을 완료했으나 미채택하여, 이 0.5 fold0 ZIP을 현재 사용자 보고 서버 최고 제출로 유지합니다.** 0.75의 새 서버 점수나 제출 검증 결과는 없습니다. [재가중치 전체 결과·최종 판단](gt-reweighting-5fold.md). 아래 10월 25일 점수 보고와 ZIP 검증 수치는 그대로 보존합니다.

2026-10-25. **사용자가 두 번째 제출 점수0.8424607623을 보고했습니다.** 첫 제출0.7913544371 대비 **+0.0511063252(+5.1106%p)**이며 현재 보고된 최고 제출 점수입니다. 직전 제공한 robust·86개·0.5 증강 fold0 ZIP의 후속 기록으로 연결했습니다. 이번 메시지에는 제출번호·서버 시간·화면·파일 해시 확인이 없으므로 사용자 보고와 로컬 ZIP 검증 근거를 구분합니다. 0.75는 당시 별도로 fold1 검증을 완료했으며 이 제출 점수의 모델은 아닙니다. [0.75 fold1 당시 결과·채택 보류 판단](gt-reweighting-fold1.md).

첫 Public 점수는0.7913544371이며 공개 선박5-fold 원본 점수와 큰 차이가 있었습니다. robust 검색·0.5 증강은 전체 다섯 fold에서 두 GT 결측·국제 GT ×1.25에 큰 개선을 보였고, 이번 사용자 보고 점수에서도 개선을 확인했습니다. 이는 검색+증강 조합의 외부 근거이며 각 변경의 개별 효과나 비공개 입력의 결측 원인을 확정하지 않습니다. 높은 점수의 fold를 고르지 않고 첫 제출과 같은 fold0로 준비한 ZIP입니다.

## 제출 파일

`C:\dacon\port\submissions\robust_gt_missing_fold0\submit.zip`

| 항목 | 내용 |
|---|---|
| 모델 | `artifacts/ranker_robust_gt_missing_token_phonetic_fold0` |
| 구성 | robust 검색·token-phonetic 86개 특징·GT 결측 증강·원본 질량0.5 |
| 학습 | fold0 제외 원본9,770행, 고정300회·CPU6스레드 |
| ZIP | 47,848,247바이트, 47.85MB, 33파일 |
| 압축 해제 | 50,318,429바이트, 50.32MB |
| 제출 결과 | 사용자 보고0.8424607623, 제출번호·서버 소요 시간 미확인 |

SHA-256:

```text
f86c8cfc03c9c6538506f3bbed87e134886bae4ad3d65c18e63c059cce46e00b
```

첫 ZIP은 다른 경로에 그대로 보존돼 있으며 SHA-256도 기존 값과 같습니다. 패키지는 `script.py`, `requirements.txt`, `model/` 구조입니다. 입력은 `data/test.csv`·`data/vessel_catalog.csv`, 출력은 `output/submission.csv`의 `sample_id,target_vessel_id` 두 열입니다. 입력 순서를 유지하고 모든 예측 ID는 제공 카탈로그에 속해야 합니다.

`--allow-robust`를 명시한 경우에만 검증한 robust·86특징·0.5 증강·6스레드 프로필을 패키징합니다. 코드·모델·인덱스·증강·가중치 증명을 확인하고 추론에 필요한 고정 정보만 포함합니다. 원본 CSV·훈련 NPZ·검증 정답·로컬 경로는 패키지에 넣지 않습니다. 새로운 0.75 재가중치 모델은 별도 제출 검증 전까지 패키징을 거부합니다.

## 실제 ZIP 실행 검증

각 입력 조건마다 ZIP을 별도 폴더에 풀고 `python -I script.py`를 새 프로세스에서 실행합니다. 로컬 작업 폴더의 모델이나 특징 캐시를 사용하지 않습니다. 실행 중 추론 상태를 갱신하지 않으며 동일 행의 결과가 순서·동반 행·행 수에 따라 바뀌지 않는지 확인합니다.

| 입력 | 행 | 실행 시작~출력 저장 | 저장 모델 일치 |
|---|---:|---:|---|
| 원본 전체 | 2,452 | 295.82초 | 전부 일치 |
| 역순 전체 | 2,452 | 301.27초 | 전부 일치 |
| 한글32+비한글32 부분집합 | 64 | 15.13초 | 전부 일치 |
| 단독 입력 | 1 | 6.06초 | 일치 |

robust 경로를 추가 확인하기 위해 같은 한글32·비한글32행에 두 GT 결측과 국제 GT ×1.25를 적용해 ZIP에서 각각 실행했습니다. **각64행 모두 기존 fold0 비교의 trained 예측과 일치**했고, 실행은 각각11.81초·17.25초였습니다. 이는 변형 표본의 패키지 동작 확인이며 전체 변형 성능 재평가는 아닙니다. 근거는 같은 폴더의 `variant_verification.json`입니다.

전체 실행의 순수 행별 추론은292.01초, 모듈 import를 포함한 스크립트 실행은295.53초입니다. 추론 프로세스 native peak working set은520.39MiB, 샘플링한 전체 자식 프로세스 메모리 합 최대는532.94MiB입니다. Windows·Python3.11.9의 로컬 측정이며 의존성 설치는 포함하지 않았습니다. OS 파일 캐시를 비우지는 않았습니다.

현재 공식 환경은 Ubuntu22.04.5·Python3.11.17·6vCPU·26GB RAM·L4이며 설치10분, 데이터 읽기·전처리·추론·출력 전체30분 제한입니다. ZIP10GB·압축 해제32GB, 하루5회 제한입니다. 비공개 테스트 행 수가 알려지지 않았으므로 로컬 시간이나 첫 제출 화면14분15초에 배율을 곱해 제한 통과를 단정하지 않습니다. [공식 평가 환경](https://dacon.io/competitions/official/236771/overview/evaluation), [제출 규칙](https://dacon.io/competitions/official/236771/overview/rules).

## 실행 기록과 재현

아래 두 명령은 완료한 생성·검증의 재현 기록입니다. 생성된 경로에는 다시 실행하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -m portmatch.submission_package --model artifacts/ranker_robust_gt_missing_token_phonetic_fold0 --output submissions/robust_gt_missing_fold0/submit.zip --allow-robust

.\.venv\Scripts\python.exe scripts/verify_submission.py --zip submissions/robust_gt_missing_fold0/submit.zip --output work/submission_validation/robust_gt_missing_fold0 --reference-model artifacts/ranker_robust_gt_missing_token_phonetic_fold0
```

실제 검증 근거는 `work/submission_validation/robust_gt_missing_fold0/verification.json`과 각 입력 폴더의 실행 로그·출력 CSV에 보존합니다. 코드 검증은 전체 회귀249개와 마지막에 추가한 미지원 패키지 설정 거부1개, 총250개를 통과했습니다. ZIP 안의 실행 소스28개가 현재 파일과 같고 첫 ZIP의 해시가 그대로임도 확인했습니다. 데이터·모델·ZIP은 GitHub에 올리지 않습니다.

제출 메모 예시:

> robust 검색·86개 특징·GT 결측 증강(original_mass=0.5), 고정 fold0. 두 번째 진단 제출.

서버 점수는 위와 같이 기록했고 표시 시간·제출번호는 아직 전달받지 않았습니다. [0.75 fold1 당시 결과](gt-reweighting-fold1.md) 이후 나머지 fold도 완료했으며, [전체5-fold 결과](gt-reweighting-5fold.md)에서 0.75를 미채택했습니다. 현재0.5 제출과 기존 첫 ZIP을 모두 보존합니다.
