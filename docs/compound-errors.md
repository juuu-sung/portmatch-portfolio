# 복합 오류 증강과 교차 학습 오답 선택

기존 robust·86개 특징·GT 결측 증강0.5 설정을 보존하고, 같은 외부 선박5-fold에서 다음 세 가지를 비교한다.

1. `baseline`: 기존 모델·검색 인덱스.
2. `expanded`: 기존 학습 후보쌍에 복합 오류 증강을 추가.
3. `expanded_mined`: expanded와 같은 질의·가중치·후보 수를 쓰고, 추가 증강 그룹의 오답 일부를 교차 학습 모델이 고른 오답으로 교체.

기존 특징·검색·캐시·제출 코드는 수정하지 않는다. 새 모델은 별도 실험 폴더에 저장하며, 기존 최고 제출을 자동 교체하거나 새 제출을 실행하지 않는다.

**2026-10-30 전체5-fold 실행·분석 완료.** 증강 확대는 모든 부가정보 결측+3.1368%p·문자 삭제+2.2033%p·미학습 순서 교환+2.7350%p를 개선했지만, 순서 교환+두 GT×0.70은−3.5830%p였다. 어려운 오답 선택의 추가 효과는 조건별로 엇갈렸다. 기존 최고 제출을 유지한다. [전체 결과와 판단](experiments/compound-errors-5fold.md).

## 고정 검증

`compound_protocol.py`의 `compound_errors_v1`은 원본, 두GT 결측, 국제GT 결측, 일반GT 결측, 국제GT×1.25, 두GT×1.25, 국적·선종 결측, 모든 부가정보 결측, 이름 문자 삭제, 이름 삭제+두GT 결측, 문자 순서 교환, 문자 순서 교환+두GT×0.70의12조건을 정의한다.

각 fold의 전체 검증 행에 같은 변형을 적용하고, 세 모델에 같은 후보·86열 특징 배열을 전달한다. 원본 baseline의 전체 행은 저장된 예측·정답 순위·후보 수·최상위 점수와 일치해야 한다. 다른 테스트 행이나 테스트 분포로 보정하지 않는다. 입력6필드가 같은 경우 결정적 계산 결과만 재사용한다. sample_id와 arrival_year는 기존 특징에서 사용하지 않으며, 원본 전행 일치와 단독·역순 실행을 검증한다.

조건별 한글0.7·비한글0.3 가중Top1, 언어별 정답/분모, 후보 재현율, Recall@K, 개선/신규오답 행·선박 수를 보고한다. 서로 다른 오류 조건의 점수를 임의로 평균하지 않는다. 기존5-fold는 이미 개발에 사용했으므로 독립 테스트나 Public 성능으로 해석하지 않는다. 문자 순서 교환과GT×1.25/×0.70은 추가 학습 증강에 넣지 않는다.

## 추가 증강

외부 훈련의 모든 원본 행은 기존 내부 vessel/row 두OOF 관점에서 추가 변형 하나씩을 받을 수 있다. 원본 입력값·관점·고정seed20261030의 해시로 다음6개 중 하나를 결정한다.

- 문자 하나 삭제
- 문자 사이 띄어쓰기 제거
- 국적·선종 결측
- 두GT를0.85배 또는1.15배로 변형
- 문자 삭제+두GT 결측
- 두GT·국적·선종·호출부호 결측

변형에는 정답이나 sample_id를 사용하지 않는다. 숫자 시퀀스와 한글/비한글 구분을 보존하고, 문자 삭제·교환은 로마 숫자 토큰을 보호한다. 바뀌지 않은 변형과 정답 후보가 없는 변형은 별도 기록하고 건너뛴다. 정답 후보를 강제로 추가하지 않는다.

추가 그룹이 있는 원본 행의 총 학습 질량은 원본0.5·기존GT마스크0.25·추가변형0.25이다. 해당 범주 안에서는 기존 질량 비례 또는 추가 그룹 균등으로 분배한다. 기존GT마스크가 없으면 그 질량은 원본으로 이동한다. 추가변형이 모두 탈락하면 그 원본의 기존 질량을 그대로 쓴다. 각 원본의 질량 합1과 두 실험의 동일 질량을 검사한다.

## 어려운 오답을 고르는 보조 모델

공유 `teacher_pool`은 **훈련 행0개**로 만든 카탈로그 전용 검색에서 생성한다. 같은 선박의 대표 공개 입력1개를 선택하고 원본·두GT 결측의 후보쌍을 만든다. 이 풀에는 모델이 없으며, 외부 검증 데이터가 보조 모델의 fitting으로 들어가지 않는다.

각 외부 fold마다 내부 선박3-fold의 보조 모델3개를100회 학습한다. 해당 보조 모델의 제외 선박과 외부 검증 선박은 fitting에서 제외한다. 보조 모델은 원본 질의와 고정 카탈로그만으로 계산되는54열만 사용한다. 조회·별칭·검색 점수·범주 통계 특징은 모두 제외하며, 후보 생성 자체도 훈련 별칭·범주 통계와 독립적이다.

각 추가 증강 질의를 학습하지 않은 보조 모델로 전체 후보를 점수화한다. 정답을 보존하고 최대64개 최고 점수 오답을 고른 뒤 기존 휴리스틱 오답으로 원래 후보 수를 채운다. 기존 캐시 그룹의 후보·특징·라벨은 바꾸지 않는다. 보조 모델의 fitting/excluded 선박, 추가 그룹의 teacher fold, 후보 교체 여부를 기록한다.

최종 두 모델은 기존과 동일한86열·300회·6스레드 설정으로 학습한다. 추가 증강과 오답 선택의 효과를 분리하기 위해 검색 인덱스·특징·모델 설정을 유지한다.

## 실행

저장소 루트에서 새 출력 폴더를 지정한다. 기존 폴더를 덮어쓰지 않는다.

**VS Code의 PowerShell 터미널 실행 명령:**

```powershell
Set-Location C:\dacon\port
.\.venv\Scripts\python.exe scripts\run_compound_study.py --output runs\compound_v1_20261030 --workers 3
```

2026-10-30 사용자 요청에 따라 Codex의 장시간 실행을 중단한 뒤, 사용자가 위 명령으로 전체 실행을 완료했다. 보조 자료5,851척·11,686그룹을 재사용했고, 중단했던 fold 자료는 `interrupted_20261030_025043_435267/`에 보존됐다. **현재 fold0~4 학습·12조건 비교·저장 결과 감사가 모두 완료됐다.** 같은 명령을 다시 실행하면 학습을 건너뛰고 집계·감사만 다시 실행한다. 이미 완료된 결과를 읽기 위해 재실행할 필요는 없다.

위 명령은 완료한 보조 자료와 완료한 fold를 건너뛴다. 중단된 fold의 폴더·로그는 같은 실험 폴더의 `interrupted_날짜_시간/`으로 보존하고 그 fold를 처음부터 다시 실행한다. 중단된 fold의 일부 증강 조각을 이어 붙이는 방식은 아니다. 최대3개 fold를 병행하되 큰 행렬 학습은 한 번에 하나로 제한한다. 학습·12조건 비교·전체 집계·저장 결과 감사를 한 번에 수행하며 새 제출은 만들지 않는다. 같은 명령의 중복 실행은 파일 잠금으로 막는다. `Ctrl+C`로 중단할 수 있고 다음 실행에서도 완료 단계는 재사용한다.

학습 없이 진행 상태와 보존 여부만 확인하려면:

```powershell
.\.venv\Scripts\python.exe scripts\run_compound_study.py --output runs\compound_v1_20261030 --status
```

완료 결과는 `runs/compound_v1_20261030/metrics.csv`, `comparison.json`, `confidence_intervals.json`, `analysis_audit.json`이다. 터미널에는 단계별 진행 상황이 표시되고 같은 내용이 fold별 로그에도 저장된다.

개별 단계의 실행 기록은 아래와 같다. 처음부터 별도 재현할 때에는 모든 `--output`에 아직 존재하지 않는 새 실험 폴더를 지정한다. 기존 완료 폴더에 `freeze`를 실행하면 보존을 위해 거부한다.

```powershell
.venv\Scripts\python.exe -m portmatch.compound_experiment freeze --output runs/compound_v1_20261030
.venv\Scripts\python.exe -m portmatch.compound_experiment prepare-teachers --output runs/compound_v1_20261030
.venv\Scripts\python.exe -m portmatch.compound_experiment run --output runs/compound_v1_20261030 --fold 0
# 같은 방식으로 fold1~4 실행
.venv\Scripts\python.exe -m portmatch.compound_experiment report --output runs/compound_v1_20261030
.venv\Scripts\python.exe scripts/analyze_compound.py --root runs/compound_v1_20261030
```

freeze는 프로토콜·공식 데이터·구현·기존43개 모델 번들 및 제출 디렉터리의 해시를 기록한다. report는 보호 파일을 다시 읽어 같은지 확인한다. 서로 다른 fold의 실행은 독립적이며, 큰 훈련 행렬 생성과 학습은 파일 잠금으로 한 번에 하나만 진행한다.

모델은 `foldN/expanded/`, `foldN/expanded_mined/`에 저장된다. `CompoundMatcher`는 해시가 고정된 기존 fold 인덱스에 연결해 새 모델을 복원한다. 따라서 실험 모델 폴더만으로 제출ZIP을 구성할 수 없으며, 제출 패키징은 별도 작업이다.

`comparison.json`과 `paired_changes.csv`는 완료한 fold만 합산하고 전체5-fold 여부를 명시한다. `foldN/predictions.csv`는 조건별·모델별 전행 예측, `augmentation_log.csv`는 증강 탈락 내역, `inner_partitions.json`과 `teachers/*/crossfit.json`은 분할 근거이다. 비교는 검색·특징을 공유하므로 합산 비교 시간은 단일 제출의 실행 시간으로 해석하지 않는다.

`analyze_compound.py`는 저장 예측의 검증 행 전체 포함·중복, 보조 모델의 선박 분리, 추가 후보쌍의 정답 라벨·후보 수·해시를 검사한다. `metrics.csv`, `changed_rows.csv`, `vessel_changes.csv`, `confidence_intervals.json`, `analysis_audit.json`을 저장한다. 변화량 구간은 선박을 단위로 같은 재표집 가중치를 두 모델에 적용하는 Poisson bootstrap 5,000회(seed20261030)로 계산한다. 같은 선박의 여러 행을 독립 표본으로 취급하지 않으며, 이 구간은 개발 자료 내 불확실성이다. 반복 모델 선택과 비공개 데이터의 분포 변화까지 보정하지 않는다.

## 검증

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p test_compound_experiment.py -v
.venv\Scripts\python.exe -m unittest discover -s tests
```

합성 통합 검증은 기존 캐시 준비부터 세 모델 비교·보조 모델 분리·원본 예측 일치·새 모델 복원·순서 독립성·보호 파일 동일성까지 실행한다. 합성 데이터의 정확도는 대회 성능으로 사용하지 않는다.

2026-10-30 구현 검증에서 전체384개 테스트가 통과했다. 저장 결과 감사 절차를 통합한 뒤 새8개 테스트도 다시 통과했다. Windows 기본 임시 폴더에서 기존 패키징 테스트의 파일 교체 권한 문제가 발생한 환경에서는 `TEMP`와 `TMP`를 작업 공간 내 `work/test_tmp`로 지정해 실행했다.
