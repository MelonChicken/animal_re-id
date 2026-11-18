# EXP-XXX: <실험 제목>

<!--
작성 지침
- 이 파일은 `docs/templates/experiment_report.md` 템플릿이다. 보고서를 작성할 때
  `docs/reports/experiments/EXP-XXX-<description>.md`로 복사한 뒤 채운다.
- `<...>` 자리표시자를 실제 값으로 바꾼다. 확인할 수 없는 값은 지우지 말고
  `Not verified` 또는 `Not available`로 남긴다.
- 사실(측정값·로그·코드에서 직접 확인)과 해석(원인 추정·판단)을 명확히 구분한다.
- 큰 로그, 캐시, 데이터셋, 체크포인트는 복사하지 않고 상대 경로로 참조한다.
- 수치는 결과 JSON/CSV의 원본 값과 일치해야 한다. 반올림 자릿수를 표에 명시한다.
-->

| 항목 | 값 |
|---|---|
| Report ID | `EXP-XXX` |
| Experiment ID | `<config의 experiment 값 또는 run 이름>` |
| 작성일 | `YYYY-MM-DD` |
| 실행일 | `YYYY-MM-DD` / `Not available` |
| 상태 | `Completed` / `Partial` / `Failed` / `Not verified` |
| 작성자 | `<이름 또는 에이전트>` |
| 사람 검토 | `Not verified` / `<검토자, 날짜>` |
| 관련 문서 | `<docs/... 상대 경로>` |
| Git Commit | `<hash>` (`dirty` 여부 포함) |

---

## 1. Overview

### 1.1 Objective

<!-- 이 실험이 무엇을 확인하려 했는지 1~3문장으로 쓴다. -->

### 1.2 Research Question

<!-- 답하려는 질문을 검증 가능한 형태로 쓴다. -->

### 1.3 Hypothesis

<!-- 사전에 등록된 가설만 쓴다. 없으면 "사전 등록된 가설 없음"이라고 쓴다.
     사후 해석은 5.3절로 분리한다. -->

### 1.4 핵심 결과 요약

<!-- 가장 중요한 수치 1~3개와 상태를 먼저 쓴다. 과장하지 않는다. -->

| 항목 | 결과 | 비고 |
|---|---|---|
| `<핵심 지표>` | `<값 ± 표준편차>` | `<비교 조건, 단위>` |
| 상태 | `<Completed / Partial>` | `<중단·실패 사유>` |

---

## 2. Experimental Setup

### 2.1 Dataset 및 데이터 버전

| 항목 | 값 | 출처 |
|---|---|---|
| Dataset | `<이름>` | `<경로 또는 URL>` |
| Version / Release | `<버전, 릴리스 ID, 파일 크기>` | `<출처>` |
| 규모 | `<프레임/시퀀스/개체 수>` | `<검증 명령 또는 로그>` |
| 전처리 전 원본 | `<해상도, 포맷>` | `<출처>` |

### 2.2 Train / Validation / Test Split

| 항목 | 값 |
|---|---|
| Split 프로토콜 | `<공식 fold / 재추첨 / 기타>` |
| Split 파일 | `<상대 경로>` |
| Split signature | `<hash>` |
| Fold별 크기 | `<fold 1: n, ...>` |
| Seed | `<값 / 해당 없음>` |
| 재사용 여부 | `<기존 split 재사용 / 새로 생성>` |

### 2.3 Query / Gallery Protocol

| 항목 | 값 |
|---|---|
| Query | `<정의>` |
| Gallery | `<정의, gallery scope>` |
| Self-match | `<제외 여부>` |
| Positive 정의 | `<정의, query당 평균 개수>` |
| Negative 정의 | `<정의>` |
| Closed-set 여부 | `<true/false, 검사 방법>` |
| 거리 / 유사도 | `<cosine similarity 등>` |

### 2.4 Evaluation Unit

<!-- image / sequence / video / tracklet 중 무엇인지, 집계 단위를 쓴다. -->

- 평가 단위: `<Image / Sequence / Video>`
- 단위당 입력: `<예: 시퀀스당 대표 프레임 1장>`
- 집계 방식: `<fold별 계산 후 평균 등>`

### 2.5 Model 및 Pretrained Checkpoint

| 항목 | 값 |
|---|---|
| Model | `<이름>` |
| 아키텍처 | `<예: SwinTransformer>` |
| 파라미터 수 | `<값>` |
| Embedding 차원 | `<값>` |
| Checkpoint | `<HF Hub ID / 로컬 경로 / revision hash>` |
| 학습 여부 | `<Frozen / Fine-tuned / Trained from scratch>` |
| 라이브러리 | `<timm, torch 버전>` |

### 2.6 Preprocessing / Frame Sampling

<!-- 리사이즈, 정규화, 프레임 선택 규칙, 클립 길이 등을 실제 코드와 일치하게 쓴다. -->

```text
<transform 또는 전처리 요약>
```

- 프레임 선택: `<규칙, 예: 시퀀스 중앙값에 가장 가까운 프레임>`
- 입력 크기: `<C × H × W>`
- 정규화: `<mean / std>`

### 2.7 Hyperparameters / Seed

| 항목 | 값 |
|---|---|
| Batch size | `<값>` |
| Learning rate | `<값 / 해당 없음>` |
| Epochs | `<값 / 해당 없음>` |
| Optimizer | `<값 / 해당 없음>` |
| Seed | `<값 / 해당 없음>` |
| 기타 | `<num_workers, mixed precision 등>` |

### 2.8 Hardware / Software Environment

| 항목 | 값 |
|---|---|
| OS | `<platform>` |
| Python | `<버전>` |
| GPU | `<모델명, VRAM>` |
| CUDA / torch | `<버전>` |
| 주요 패키지 | `<timm, torchvision, numpy, pandas 등>` |

---

## 3. Implementation & Validation

### 3.1 주요 구현 사항

<!-- 실험을 실제로 수행한 코드 경로와 핵심 로직을 요약한다. -->

### 3.2 관련 코드

| 파일 | 역할 |
|---|---|
| `<상대 경로>` | `<역할>` |
| `<상대 경로>` | `<역할>` |

### 3.3 테스트 결과

| 명령 | 실행 시점 | 결과 |
|---|---|---|
| `<예: uv run pytest -q>` | `<YYYY-MM-DD>` | `<n passed, n skipped>` |
| `<실험 재현 명령>` | `<YYYY-MM-DD>` | `<성공/실패>` |

<!-- 실험 실행 시점의 테스트가 아니라 보고서 작성 중 재실행한 테스트라면 그 사실을 쓴다.
     실행하지 않은 테스트를 통과했다고 쓰지 않는다. -->

### 3.4 데이터 무결성 및 누출 검사

| 검사 | 방법 | 결과 |
|---|---|---|
| Sequence key 유일성 | `<검사 함수/명령>` | `<통과/실패>` |
| Split 합계 일치 | `<검사 방법>` | `<통과/실패>` |
| Query/Gallery 교집합 | `<검사 방법>` | `<통과/실패, self-match 제외>` |
| Positive 없는 query | `<검사 방법>` | `<0건 등>` |
| 데이터 누출 | `<검토 내용>` | `<확인/미확인>` |

### 3.5 평가 지표 검산

<!-- 원본 per-query CSV에서 지표를 다시 계산했다면 방법과 결과를 쓴다.
     fold 평균과 pooled 평균이 다를 수 있으므로 집계 방식을 명시한다. -->

| 지표 | 원본 JSON | 재계산 | 차이 / 설명 |
|---|---|---|---|
| Rank-1 | `<값>` | `<값>` | `<집계 방식 차이 등>` |
| Rank-3 | `<값>` | `<값>` | `<...>` |
| mAP | `<값>` | `<값>` | `<...>` |

---

## 4. Results

### 4.1 주요 지표

<!-- 단위별 표와 집계 표를 분리한다. 평균 ± 표준편차와 표준편차 종류(ddof)를 명시한다. -->

| 단위 | Rank-1 | Rank-3 | mAP | Query 수 | Gallery 수 |
|---|---|---|---|---|---|
| `<fold 1>` | `<값>` | `<값>` | `<값>` | `<n>` | `<n>` |
| ... | ... | ... | ... | ... | ... |
| **평균 ± 표준편차** | `<값 ± 값>` | `<값 ± 값>` | `<값 ± 값>` | `<n>` | `<n>` |

### 4.2 Baseline 비교

<!-- 같은 split·전처리·평가 단위에서 측정한 baseline만 비교한다. -->

| 방법 | Rank-1 | Rank-3 | mAP | 동일 조건 여부 | 비고 |
|---|---|---|---|---|---|
| `<이번 실험>` | `<값>` | `<값>` | `<값>` | `Yes` | `<...>` |
| `<baseline>` | `<값>` | `<값>` | `<값>` | `<Yes/No>` | `<차이>` |

### 4.3 논문 결과와의 비교 가능성

| 항목 | 논문 | 이번 실험 | 비교 가능 여부 |
|---|---|---|---|
| Backbone | `<...>` | `<...>` | `<Yes/No>` |
| 학습 방식 | `<...>` | `<...>` | `<Yes/No>` |
| 평가 단위 | `<...>` | `<...>` | `<Yes/No>` |
| Gallery 범위 | `<...>` | `<...>` | `<Yes/No>` |
| 보고 수치 | `<...>` | `<...>` | `<직접 비교 금지 등>` |

<!-- 프로토콜이 다르면 수치를 나란히 두고 "재현했다"고 쓰지 않는다. -->

### 4.4 실행 시간 / GPU 사용량 / Cache

| 항목 | 값 | 출처 |
|---|---|---|
| 전체 실행 시간 | `<초>` | `<결과 JSON>` |
| Feature extraction | `<초>` | `<결과 JSON>` |
| Cache 사용 | `<true/false>` | `<결과 JSON>` |
| Cache 경로 / key | `<상대 경로 / hash>` | `<결과 JSON>` |
| GPU 메모리 | `<GB>` | `<환경 기록>` |

### 4.5 반복 실험 통계

<!-- 동일 조건 반복 실행이 있으면 평균±표준편차와 반복 수를 쓴다.
     없으면 "반복 실행 없음"과 그 이유를 쓴다. -->

| 조건 | 반복 수 | Rank-1 | Rank-3 | mAP |
|---|---|---|---|---|
| `<조건>` | `<n>` | `<값 ± 값>` | `<값 ± 값>` | `<값 ± 값>` |

---

## 5. Analysis

### 5.1 확인된 실험 결과

<!-- 측정값에서 직접 말할 수 있는 것만 쓴다. -->

### 5.2 가능한 원인과 근거

<!-- 원인 추정과 그 근거를 분리한다. 확정하지 못한 것은 "추정"이라고 쓴다. -->

### 5.3 가설의 지지 여부

| 가설 | 결과 | 판단 |
|---|---|---|
| `<가설>` | `<관측>` | `지지 / 부분 지지 / 기각 / 판단 불가` |

### 5.4 Failure Cases

| 사례 | 관측 | 추정 원인 | 검증 상태 |
|---|---|---|---|
| `<sequence/ID>` | `<rank, AP 등>` | `<...>` | `<확인/미확인>` |

### 5.5 예상과 다른 결과 및 부정적 결과

<!-- 성능 저하, 실패, 재현 불일치, 유의미하지 않은 차이를 숨기지 않는다. -->

---

## 6. Limitations

### 6.1 Dataset Leakage

<!-- 학습/사전학습 데이터와 평가 데이터의 중복 가능성. -->

### 6.2 Protocol Mismatch

<!-- 논문/벤치마크와 다른 split, gallery, 평가 단위, 프레임 선택. -->

### 6.3 사전학습 데이터 노출

<!-- Foundation model의 사전학습 데이터에 평가셋이 포함되었는지. -->

### 6.4 통계적 불확실성

<!-- fold 수, 개체 수, 표준편차, 신뢰구간, 반복 실행 부재. -->

### 6.5 재현성 한계

<!-- Git dirty 여부, checkpoint revision, 하드웨어/라이브러리 의존성, 캐시. -->

### 6.6 확인되지 않은 조건

<!-- 실행하지 않은 변형, 측정하지 않은 민감도. -->

---

## 7. Conclusion & Next Steps

### 7.1 결론

<!-- 연구 질문에 대한 답을 확인된 범위 안에서 쓴다. -->

### 7.2 후속 실험

| 우선순위 | 실험 | 목적 | 선행 조건 |
|---|---|---|---|
| `<P1>` | `<...>` | `<...>` | `<...>` |

### 7.3 남은 연구 질문

- `<질문>`

---

## Artifacts

<!-- 대용량 파일은 복사하지 않고 상대 경로만 적는다. 존재 여부를 확인한다. -->

| 종류 | 경로 / ID | 비고 |
|---|---|---|
| Config | `<config/...>` | `<...>` |
| Metrics | `<experiments/results/...json>` | `<...>` |
| Per-query | `<experiments/results/...csv>` | `<...>` |
| Runtime log | `<경로 / Not available>` | `<...>` |
| Split | `<experiments/splits/...json>` | `<hash>` |
| Feature cache | `<experiments/cache/...npz>` | `<key>` |
| Model checkpoint | `<HF Hub ID / 경로 / Not available>` | `<revision>` |
| Git Commit | `<hash>` | `<dirty 여부>` |
| Run ID | `<created_at 또는 run 이름>` | `<...>` |
