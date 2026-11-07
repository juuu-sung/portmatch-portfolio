# 검색·기본 특징 사용법

> 공개용 정리본: 실제 선박별 입력·정답·예측·점수와 행 식별자는 제외했다. 아래 성능표는 실제 실험의 집계 수치이며, 오류의 원리는 별도의 [가상 예시](fictional-examples.md)로 설명한다.

검색 경로, 수치 처리와 후보 재현율을 확인하는 상세 안내입니다. 전체 구조는 [README](../README.md), 모델 학습은 [TRAINING](../TRAINING.md), 누수 방지 설계는 [아키텍처](architecture.md)를 참고합니다.

아래는 보존한 기본 검색(`legacy`)의 설명입니다. 선택적으로 한글 문맥·모음·편집 검색(`hangul`), 일반·국제 GT 후보 합집합(`dual-gt`), 두 변경 합산(`robust`)을 구현했습니다. 기존 86개 모델의 가중치를 고정해 원본·톤수 변형을 비교하는 `compare-retrieval`과 새 OOF 학습 명령은 [검색 강건성 실험](experiments/robust-retrieval.md)에 있습니다. 새 검색은 아직 채택하지 않았으며 기본값도 바뀌지 않았습니다.

## 구현 내용

| 기능 | 동작 |
|---|---|
| 호출부호 조회 | 훈련 부분의 호출부호 → 카탈로그 선박 후보. 익명 호출부호는 정확히 일치할 때만 사용 |
| 정규화 이름 조회 | 유니코드·대소문자·공백·구두점 정규화 후 조회. 충돌하면 여러 후보와 각각의 지지도를 보존 |
| 이름 검색 | 카탈로그 이름 + 해당 훈련 부분의 관측 이름을 문자 TF-IDF로 검색. 비한글 이름은 편집거리 검색도 별도로 사용 |
| 한글 음차 검색 | 일반 로마자 변환과 자음 중심 음차 키를 함께 검색. 외부 선박 사전이나 생성한 학습 예시는 사용하지 않음 |
| 톤수 검색 | 국제총톤수 우선, 카탈로그 GT 기준 상대오차 10% 이내 전체 후보. 국제총톤수가 없거나 범위가 비면 국내 총톤수로 보완 |
| 비교 특징 | 국적·선종의 조건부 확률/지지도, 문자열·음차 유사도, 선박명 숫자 충돌, GT/국제GT 차이와 결측 표시, 검색 경로·순위 |
| 특징 선택 학습 | 기존 캐시의 열만 포함/제외해 학습. 저장한 선택 목록을 검증·추론에도 적용 |
| 동점 보정 순위 | 샘플링 전 전체 톤수 후보에서 같은 GT 거리에 같은 최소 순위 부여. 원본 캐시를 보존하고 순위 열만 새 캐시로 변환 |
| 토큰·문맥 음차 | 현재 질의 이름·고정 카탈로그 이름만으로 17개 특징 추가. 원본70 캐시·OOF·후보·샘플을 보존하고 학습에는 톤수 순위를 뺀 86개 사용 |
| 모델 비교 | 같은 캐시 또는 검증된 특징 변환 파생 캐시의 설정·후보·검증 ID를 확인하고 전체 행의 개선/신규 오답을 비교 |

음차 변환은 거친 비교 수단이며 한글 이름을 정확한 영어 선명으로 번역하는 기능은 아닙니다. 수치 비교는 GT끼리만 수행하며 DWT를 GT와 직접 비교하지 않습니다. 특징의 수치 상대오차는 대칭적인 `abs(query - catalog) / max(query, catalog)`를 사용합니다. 수치 결측은 NaN과 별도 유효성 표시로 유지하므로 후속 모델에서 결측 지원 또는 훈련 부분으로만 맞춘 결측 대체가 필요합니다.

## 검색 기반을 처음부터 점검하는 명령

아래는 검색·특징 생성 단계의 사용 예시입니다. 현재 완료된 실험을 반복할 필요는 없으며, 순위 모델의 특징 제거 실험은 `TRAINING.md`의 캐시 재사용 명령을 사용합니다.

PowerShell 터미널을 사용합니다. 로컬 개발 환경 기준 예시입니다. 새 복제본에서는 먼저 가상환경을 만들고 별도로 받은 대회 데이터를 `work/data`에 배치해야 합니다.

```powershell
cd C:\dacon\port
```

다른 환경에서 새로 설치할 때만 다음을 실행합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 1. 먼저 미관측 선박 검증 1개 fold

```powershell
.\.venv\Scripts\python.exe -m portmatch evaluate --split vessel --folds 5 --fold 0 --output runs/vessel_fold0
```

같은 선박의 모든 입항 기록을 같은 fold에 둡니다. 따라서 검증 선박은 호출부호·이름 사전의 훈련 정답에 등장하지 않습니다. 카탈로그 전체는 후보 모집단으로 사용할 수 있습니다. 이 결과로 처음 보는 선박의 후보 검색 성능을 확인합니다.

### 2. 시간 여유가 있으면 전체 5-fold 및 행 분할 비교

```powershell
.\.venv\Scripts\python.exe -m portmatch evaluate --split vessel --folds 5 --output runs/vessel_5fold
.\.venv\Scripts\python.exe -m portmatch evaluate --split row --folds 5 --output runs/row_5fold
```

행 분할은 이미 본 선박의 재입항이 포함되므로 조회 사전의 효과를 확인하는 보조 실험입니다. 행 분할 성능만으로 미관측 선박의 성능을 판단하면 안 됩니다. 두 방식 모두 fold마다 사전, 별칭 검색 인덱스, 국적·선종 통계를 훈련 부분에서 새로 만듭니다. 검증 정답은 검색 입력에서 제거됩니다.

### 3. 후보별 특징을 실제 파일로 확인

```powershell
.\.venv\Scripts\python.exe -m portmatch evaluate --split vessel --folds 5 --fold 0 --max-queries 50 --save-pairs --output runs/features_50
```

`pairs_fold_0.csv`에는 표본 질의의 전체 후보, ID, 정답 여부와 70개 특징을 저장합니다. ID와 `label`은 관리용 열이며 모델 특징에 포함하지 않습니다. 정답 후보를 강제로 추가하지 않아 검색 실패도 그대로 남습니다. 넓은 톤수 범위를 유지하므로 전체 데이터에 `--save-pairs`를 켜면 파일이 상당히 커질 수 있습니다. 먼저 작은 표본으로 확인하세요.

`--max-queries`는 fold당 표본 수입니다. 가능하면 두 언어가 모두 포함되도록 표본을 뽑으므로, 특히 작은 표본 수치는 성능 추정에 사용하지 마세요. 이 옵션을 빼면 선택한 fold의 모든 검증 행을 평가합니다. 기본 평가는 검색을 전 행에서 실행하고, 비교 특징은 fold당 앞 10개 질의에서 생성·점검합니다. `--save-pairs`는 평가하는 모든 질의의 특징을 생성합니다.

### 4. 검증 후 전체 훈련 데이터로 고정 인덱스 생성

```powershell
.\.venv\Scripts\python.exe -m portmatch build --output artifacts/full_index
```

`index.joblib`에는 훈련 사전, 검색 인덱스, 범주 통계와 특징 생성기가 들어 있습니다. **학습된 순위 모델은 아닙니다.** 이 전체 훈련 인덱스로 같은 훈련 데이터의 검증 점수를 측정하면 누수가 생깁니다. 검증에는 반드시 `evaluate`를 사용하세요.

출력은 새 폴더 또는 빈 폴더만 허용합니다. 재실행할 때 `--output`의 마지막 이름을 바꾸면 이전 실험과 섞이지 않습니다.

## 결과 읽는 방법

- `summary.json`: 전체 및 한글/비한글 후보 재현율, 각 검색 경로의 재현율, 후보 수의 중앙값/95백분위/최댓값, 시간, 설정과 원본 파일 해시.
- `queries_fold_N.csv`: 각 질의에서 정답이 후보에 있었는지, 어느 경로로 찾았는지, 후보 수, 훈련에 같은 선박이 있었는지.
- `splits.csv`: 모든 학습 행의 fold 배정. 같은 seed와 입력으로 재현할 수 있습니다.
- `summary_fold_N.json`: 해당 fold 결과. 긴 실행 도중 완료된 fold부터 확인할 수 있습니다.
- `pairs_fold_N.csv`: `--save-pairs` 지정 시 후보별 특징과 정답 여부.

`weighted_candidate_recall`은 `0.7 × 한글 후보 재현율 + 0.3 × 비한글 후보 재현율`입니다. **정답이 후보 안에 들어온 비율이며, 최종 정확도나 대회 점수가 아닙니다.** `recall_call`, `recall_edit`, `recall_phonetic`, `recall_tonnage` 등은 각 경로에서 찾은 비율이며 서로 겹칠 수 있습니다.

`--gt-window`, `--name-top-k`, `--phonetic-top-k`, `--edit-top-k`로 검색 범위를 조절할 수 있습니다. 선택적 `--max-candidates`는 조회/정확 일치 후보를 보호하면서 나머지를 줄이지만 정답이 탈락할 수 있습니다. 우선 기본 설정의 미관측 한글 재현율과 후보 수를 확인한 뒤 조절하세요. 최종 성능을 보고할 별도 검증 부분은 튜닝에서 제외해야 합니다.

## 코드에서 연결

```python
import joblib

# 이 프로젝트에서 직접 만든 신뢰할 수 있는 로컬 자산만 로드합니다.
pipeline = joblib.load("artifacts/full_index/index.joblib")
# 설명용 가상 입력이며 실제 대회 행이 아닙니다.
query = {
    "sample_id": "DEMO-QUERY", "vessel_name": "DEMO LUMEN 7",
    "call_sign": "", "gross_tonnage": 14000,
    "international_gross_tonnage": 14000,
    "flag_name": "", "vessel_type": "", "arrival_year": 2025,
}
candidates = pipeline.retrieve(query)
X = pipeline.features(query, candidates)
vessel_ids = pipeline.catalog.iloc[candidates.catalog_index].vessel_id.to_numpy()
# X의 각 행과 vessel_ids가 같은 순서입니다.
```

각 질의는 고정 인덱스를 읽기만 하며 다른 평가 행을 이용한 사전 갱신, 빈도 집계, 투표를 하지 않습니다. `sample_id`, `vessel_id`, 정답 ID는 특징에 포함하지 않습니다. 순위 모델을 학습할 때도 자신을 포함한 사전/범주 통계로 훈련 특징을 만들지 않도록 fold 밖 특징을 구성해야 합니다. 최종 평가 fold의 정답은 순위 모델 학습 및 통계 생성에서도 제외해야 합니다.

## 짧은 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

사전 누수 방지, 조회 충돌 보존, 넓은 톤수 후보 유지, 편집거리·음차 검색, 결측값, 숫자 충돌, 국적·선종 통계, 행 순서 독립성을 확인합니다.

검색·특징·누수 방지·오답 샘플링·순위 모델 저장과 복원, 특징 선택과 비교 리포트의 정합성을 테스트합니다. 전체 데이터 학습은 `TRAINING.md`의 명령어로 실행할 수 있습니다.
