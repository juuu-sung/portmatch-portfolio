# 최종 모델 코드와 재현 안내

[프로젝트 개요](../README.md) · [실험 근거](experiment-evidence.md) · [학습 기록](../TRAINING.md)

최종 제출은 **focused_search fold0 단일 모델**, Public **0.8515414798**이다. 원본 9,770행이 속한 fold0 훈련 부분과 증강 예제로 학습했으며 전체훈련 모델이나 5-fold 앙상블이 아니다. 아래 소스는 완료한 실험·제출에 사용한 구현이다. 포트폴리오 게시를 위해 모델을 다시 학습하거나 가중치를 변경하지 않았다.

## 구현을 읽는 순서

| 단계 | 소스 | 역할 |
|---|---|---|
| 최종 추론 | [focused_runtime.py](../portmatch/focused_runtime.py) | 해시·스키마 확인, 기존 후보 특징 보존, 추가 후보 점수 계산, 고정 규칙으로 ID 선택 |
| 한글 후보 추가 | [hangul_expansion.py](../portmatch/hangul_expansion.py) | 카탈로그에서 만든 고정 검색 자산으로 한글 질의에 최대 256개 후보 추가 |
| 순위 학습 | [ranking.py](../portmatch/ranking.py) | 86개 특징의 LightGBM LambdaRank, 300회·CPU 6스레드 |
| 최종 학습 실험 | [실험 구현](../portmatch/tonnage_repair_experiment.py) · [프로토콜](../portmatch/tonnage_repair_protocol.py) | 넓은 이름·톤수 증강과 추가 검색 오답 선택의 효과 비교 |
| 후속 효과 분리 | [실험 구현](../portmatch/focused_replay_experiment.py) · [프로토콜](../portmatch/focused_replay_protocol.py) | 이름·중간 톤수·동시 보강과 원본·기존 오류 유지 비교 |
| 제출물 생성 | [focused_submission_package.py](../portmatch/focused_submission_package.py) | 고정 인덱스·가중치·런타임 소스의 독립 ZIP, 기존 파일 덮어쓰기 거부 |
| 제출 실행 | [submission_focused/script.py](../submission_focused/script.py) | 패키지 무결성·카탈로그 확인, 행별 추론과 제출 형식 출력 |
| 제출 검증 | [verify_focused_submission.py](../scripts/verify_focused_submission.py) | 별도 프로세스 복원, 저장 예측 일치, 입력 순서·부분·단일 실행 확인 |

추론용 ZIP에는 실험 모듈·정답·훈련 캐시가 필요하지 않다. 반면 **패키징 프로그램**은 학습 출처를 확인하기 위해 원래 실험 산출물을 읽는다. 이 두 경로의 의존성을 구분한다.

## 공식 데이터 없이 확인할 수 있는 것

Python 3.11과 [고정 의존성](../requirements.txt)을 설치한 뒤 저장소 루트의 VS Code PowerShell에서 실행한다. 합성 데이터로 작은 학습·후보 검색·패키징까지 확인하며 대회 모델은 재학습하지 않는다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force work | Out-Null
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_focused_submission.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_focused_replay.py -v
```

[제출 테스트](../tests/test_focused_submission.py)는 모든 오류 레시피의 점수 일치, ID·연도·라벨 독립성, 고립된 프로세스의 복원, 카탈로그 재정렬, 변조 거부와 ZIP 덮어쓰기 방지를 검사한다. [후속 실험 테스트](../tests/test_focused_replay.py)는 보강별 학습 비중·동일 후보·저장 모델 복원·완료 fold 재사용 등을 확인한다. 합성 결과는 실제 Public 성능의 재측정이 아니다.

## 실제 데이터 실험의 선행 단계

| 순서 | 필요한 산출물 | 실행·설계 안내 |
|---|---|---|
| 1 | 공식 학습 데이터·카탈로그, robust 검색·86개 특징·톤수 결측 증강의 fold0~4 모델과 캐시 | [TRAINING.md](../TRAINING.md) |
| 2 | 복합 오류 고정 검증·확대 증강·교차 학습 오답 선택 | [설계](compound-errors.md) · [실행기](../scripts/run_compound_study.py) |
| 3 | 이름·톤수 조합 증강 × 한글 검색의 효과 분리 | [설계](joint-search.md) · [실행기](../scripts/run_joint_search_study.py) |
| 4 | 넓은 톤수 보강·추가 오답 학습 → 최종 focused_search | [설계](tonnage-repair.md) · [실행기](../scripts/run_tonnage_repair_study.py) |
| 5 | focused_search 기준 이름·중간 톤수·동시 보강 비교 | [설계와 결과](tonnage-repair.md) · [실행기](../scripts/run_focused_replay_study.py) |

각 실험은 부모 산출물·설정·코드 해시를 고정한다. 공개 소스만 내려받은 환경에서는 선행 모델과 캐시가 없으므로 최종 실행기 하나로 과거 결과 전체를 복원할 수 없다. 공식 데이터를 준비해 앞 단계부터 생성해야 한다. 기존 실험 경로를 옮겨 재사용할 때도 절대 경로와 출처 검증을 고려해야 하며, 기록된 해시를 임의로 수정해 검사를 통과시키지 않는다.

## 보존된 자산으로 제출 ZIP을 다시 검증하기

아래 명령은 **공식 데이터와 완료된 로컬 실험·최종 ZIP을 이미 보유한 환경**용이다. 전체 추론은 수 분 이상 걸리므로 필요할 때 VS Code에서 실행한다. 기존 결과와 겹치지 않는 새 출력 경로를 사용한다.

```powershell
.\.venv\Scripts\python.exe scripts\verify_focused_submission.py --zip submissions\focused_search_fold0\submit.zip --study runs\tonnage_repair_v1_20261031 --output runs\verify_focused_search_fold0_reproduced --mode full
```

완료된 전체 검증은 [집계 기록](evidence/submission-verification.json)에 있다. 5,293개 예측이 저장된 실험 결과와 일치했고, 원본 2,452행의 로컬 Windows 실행은 약 295초·최대 프로세스 트리 메모리 약 567 MiB였다. 공식 서버 전체 입력의 처리 시간이나 Linux 실행 성능을 의미하지 않는다.

원본 데이터·모델·제출물은 로컬에 보존한다. 코드의 바이트를 해시로 검증하므로 [.gitattributes](../.gitattributes)의 Python 줄바꿈 보존 설정도 함께 유지한다.
