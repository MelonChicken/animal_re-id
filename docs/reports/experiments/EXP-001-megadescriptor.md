# EXP-001: Frozen MegaDescriptor-L-384 PolarBearVidID 5-fold Retrieval Baseline

| 항목 | 값 |
|---|---|
| Report ID | `EXP-001` |
| Experiment ID | `megadescriptor_frozen_polarbear_5fold` |
| 작성일 | 2026-11-18 |
| 실행일 | 2026-11-17 (UTC, 결과 JSON `created_at` 기준) |
| 상태 | `Completed` (재현성 제한 있음) |
| 작성자 | OpenCode agent |
| 사람 검토 | `Not verified` |
| 관련 문서 | `docs/benchmark_protocols.md` |
| Git Commit | `6dd705c6d66fcb4f2858a50676ab9b1b34895bc1` (`dirty`) |

**증거 수준 표기**

- **확인**: 결과 JSON/CSV, 코드, 테스트 출력에서 직접 확인한 값.
- **추정**: 기록된 값에 근거한 해석이며, 원인을 실험적으로 분리하지는 못한 것.
- **미확인**: 실행하거나 검증하지 않은 것.

---

## 1. Overview

### 1.1 Objective

PolarBearVidID의 공식 5-fold image-level 프로토콜에서 **학습하지 않은(frozen)**
MegaDescriptor-L-384 특징의 closed-set retrieval 성능을 측정하고, gallery 범위
(`fold` vs `all`)가 지표에 미치는 영향을 기록한다.

### 1.2 Research Question

- **RQ1**: Frozen MegaDescriptor-L-384 feature만으로 공식 fold에서 어느 수준의
  Rank-1 / Rank-3 / mAP를 얻는가?
- **RQ2**: Gallery를 해당 test fold로 제한할 때와 전체 1,431 시퀀스로 둘 때
  지표는 어떻게 달라지는가?

### 1.3 Hypothesis

사전 등록된 가설은 **없다**. 이 실험은 프로토콜 검증과 베이스라인 측정을 목적으로
수행되었고, 본 보고서는 결과를 사후에 정리한 것이다. 따라서 가설 지지 여부는
5.3절에서 탐색적으로만 다룬다.

### 1.4 핵심 결과 요약

| 항목 | 결과 | 비고 |
|---|---|---|
| Rank-1 (gallery=`fold`, 주 결과) | **0.7771 ± 0.0376** | 5-fold 평균 ± 표본표준편차 |
| Rank-3 (gallery=`fold`) | **0.8826 ± 0.0332** | 5-fold 평균 ± 표본표준편차 |
| mAP (gallery=`fold`) | **0.3969 ± 0.0299** | 5-fold 평균 ± 표본표준편차 |
| Rank-1 (gallery=`all`) | 0.7827 ± 0.0219 | 동일 feature, gallery만 전체로 확장 |
| mAP (gallery=`all`) | 0.2715 ± 0.0160 | fold 대비 **0.1254 감소** |
| 상태 | `Completed` | zero-shot 아님, 논문 재현 아님 |

> **주의**: 이 수치는 MegaDescriptor의 사전학습 데이터와 PolarBearVidID가 겹칠
> 가능성이 있는 조건에서 측정되었다(6.1절). **Zero-shot 결과가 아니며 SOTA로
> 표기하지 않는다.** 또한 Zuerl et al. (2023)의 수치를 재현한 것도 아니다.

---

## 2. Experimental Setup

### 2.1 Dataset 및 데이터 버전

| 항목 | 값 | 출처 |
|---|---|---|
| Dataset | PolarBearVidID | `config/polarbear.yaml` |
| Release | Zenodo `7564529` / `PolarBearVidID.zip` (1,573,948,301 bytes) | `docs/benchmark_protocols.md` §1 |
| 로컬 경로 | `data/raw/PolarBearVidID` | 결과 JSON `dataset.root` |
| 규모 | 138,363 프레임 / 1,431 시퀀스(트랙릿) / 13 개체 / 6 동물원 | 결과 JSON + `docs/benchmark_protocols.md` §1 |
| 원본 해상도 | 256 × 128 | `docs/benchmark_protocols.md` §1 |
| 카메라 | 3대 (`C01`, `C02`, `C03`) | `docs/benchmark_protocols.md` §1 |
| 프레임/트랙릿 번호 | 0부터 시작 | `docs/benchmark_protocols.md` §1 |
| Git 추적 | 제외 (`.gitignore`의 `/data/`) | `.gitignore` |

데이터 경로는 상대 경로로만 참조하며, 데이터셋 자체는 보고서에 복사하지 않는다.

### 2.2 Train / Validation / Test Split

| 항목 | 값 |
|---|---|
| Split 프로토콜 | `five_fold` (릴리스에 포함된 공식 fold 파일 사용) |
| Split 파일 | `experiments/splits/polarbear_five_fold_seed0.json` |
| Split signature | `c66a81ab11b8fdd987c5fc42109cfab341fbcfe9eba9451b8ed0e0708ec069a7` |
| Dataset signature | `ad9d020a60955c9b54871599c92a4518991407b519c03d10a86e2aa9c50d038b` |
| Fold별 시퀀스 수 | Fold 1: 287, Fold 2: 286, Fold 3: 286, Fold 4: 286, Fold 5: 286 |
| Seed | 0 (공식 fold는 고정 배정이므로 재추첨에 사용되지 않음) |
| Source | `official release` |
| 재사용 여부 | 기존 split 파일 재사용 (결과 JSON `split.path`와 동일) |

**중요**: 이 실험은 모델을 학습하지 않는다. 5-fold는 학습용 train/validation/test가
아니라, **5개의 query/gallery 평가 단위**로만 사용되었다. Fold 배정은 릴리스의
`track_fold_info_1.csv` … `track_fold_info_5.csv`를 그대로 읽는다.

### 2.3 Query / Gallery Protocol

| 항목 | 값 |
|---|---|
| Query | 해당 fold의 시퀀스별 대표 프레임 1장 (287 또는 286장) |
| Gallery (`fold`, 주 설정) | 같은 fold의 대표 프레임 전부 (287 또는 286장), self-match 제외 |
| Gallery (`all`, 비교 설정) | 전체 1,431 시퀀스의 대표 프레임, self-match 제외 |
| Self-match | `path`가 동일한 gallery 항목을 순위에서 제외 |
| Positive | 같은 identity의 다른 시퀀스 |
| Query당 eligible positive 수 | **21.1개** (fold) / **109.5개** (all) — 보고서 재계산 |
| Negative | 다른 identity 전부 |
| Closed-set | `true`; 5개 fold 모두 `n_queries_without_positives = 0` |
| 거리 / 유사도 | cosine similarity |
| 지표 | Rank-1, Rank-3, mAP |
| 집계 | fold별 계산 후 평균 ± 표본표준편차 (`ddof=1`) |

**Positive 수 재계산에 관한 주의**: `docs/benchmark_protocols.md` §2.2와 §2.4에는
각각 "fold 내 평균 22개", "positive 수가 22개 → 122개"로 적혀 있다. 그러나 현재
split과 `build_retrieval_task()`로 다시 계산하면 self-match를 제외한 eligible
positive는 fold 평균 **21.1개**, all-gallery 평균 **109.5개**다. 문서의 22개는
self-match를 포함한 동일 identity 수(평균 22.1개)로 보이며(추정), 122개 역시
재계산값 109.5개와 일치하지 않는다. 이 보고서는 재계산값을 사용하며, 원 문서는
이번 작업에서 수정하지 않았다.

### 2.4 Evaluation Unit

- 평가 단위: **Image**
- 단위당 입력: 시퀀스(트랙릿)당 대표 프레임 1장
- 총 query 수: 1,431 (5개 fold 합계, 각 시퀀스가 정확히 한 번 query가 됨)
- 집계: fold별 지표를 계산한 뒤 5개 fold의 평균과 표준편차를 보고

### 2.5 Model 및 Pretrained Checkpoint

| 항목 | 값 |
|---|---|
| Model | `hf-hub:BVRA/MegaDescriptor-L-384` |
| 아키텍처 | SwinTransformer |
| 파라미터 수 | 195,198,516 |
| Embedding 차원 | 1,536 |
| Checkpoint | Hugging Face Hub `BVRA/MegaDescriptor-L-384` |
| Checkpoint revision hash | `Not available` (결과 JSON에는 Hub ID만 기록됨) |
| 학습 여부 | **Frozen** (추가 학습 없음) |
| 라이브러리 | `timm` 1.0.31, `torch` 2.11.0+cu128 |

### 2.6 Preprocessing / Frame Sampling

```text
Compose(
    Resize(size=(384, 384), interpolation=bicubic, antialias=True)
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
```

- 프레임 선택: `middle` — 시퀀스 중앙값에 가장 가까운 프레임, 동률이면 낮은 번호
  (`docs/benchmark_protocols.md` §2.2)
- 입력 크기: 3 × 384 × 384
- 정규화: ImageNet mean/std
- 클립/프레임 샘플링: 없음 (시퀀스당 1장)

### 2.7 Hyperparameters / Seed

| 항목 | 값 |
|---|---|
| Batch size | 32 |
| `num_workers` | 0 |
| Learning rate | 해당 없음 (학습하지 않음) |
| Epochs | 해당 없음 |
| Optimizer | 해당 없음 |
| Seed | 0 (split 메타데이터) |
| Device | `cuda` |

### 2.8 Hardware / Software Environment

| 항목 | 값 |
|---|---|
| OS | Windows-11-10.0.26200-SP0 |
| Python | 3.12.10 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8.59 GB |
| CUDA / torch | torch 2.11.0+cu128 |
| torchvision | 0.26.0 |
| timm | 1.0.31 |
| wildlife-datasets | 1.0.9 |
| wildlife-tools | 0.0.9 |
| numpy / pandas | 2.5.5 / 3.1.1 |

---

## 3. Implementation & Validation

### 3.1 주요 구현 사항

`scripts/evaluate.py`가 config를 읽고, 데이터 카탈로그를 로드하고, 공식 fold를
재사용하거나 생성한 뒤, 시퀀스별 대표 프레임을 frozen MegaDescriptor로 임베딩하고,
cosine similarity 기반 closed-set retrieval을 계산한다. 결과 JSON에는 config,
환경, dataset/split/model 메타데이터, fold별 지표, 집계, runtime이 함께 저장된다.

### 3.2 관련 코드

| 파일 | 역할 |
|---|---|
| `scripts/evaluate.py` | 전체 평가 파이프라인, 결과 JSON/CSV 저장 |
| `src/animal_re_id/data/wildlife.py` | PolarBearVidID 카탈로그 로딩 |
| `src/animal_re_id/data/splits.py` | 공식 fold, RoVF split, 대표 프레임, retrieval task 구성 |
| `src/animal_re_id/evaluation/retrieval.py` | cosine similarity, Rank-1/3, mAP 계산 |
| `src/animal_re_id/models/megadescriptor.py` | 모델 로딩, transform, feature extraction |
| `scripts/fold_sensitivity.py` | 공식 fold vs 무작위 치환 fold 민감도 |
| `config/polarbear.yaml` | 이 실험의 config |

### 3.3 테스트 결과

| 명령 | 실행 시점 | 결과 |
|---|---|---|
| `uv run pytest -q` | 2026-11-18 (보고서 작성 중 2회) | **49 passed, 4 skipped**, 29 warnings (35.77 s / 81.73 s) |
| `uv run python scripts/evaluate.py --config config/polarbear.yaml` | 2026-11-17 (실험 실행) | 결과 JSON 존재. 실제 셸 출력/종료 코드 로그는 `Not available` |
| `RUN_MODEL_TESTS=1` integration tests | 실행하지 않음 | 4 skipped. 실제 checkpoint 통합 테스트는 `Not verified` |

- 4개의 skipped 테스트는 `tests/test_megadescriptor.py`의 integration 테스트로,
  `RUN_MODEL_TESTS=1`이 없으면 의도적으로 skip된다.
- 2026-11-18 테스트 실행은 **보고서 검증을 위해 재실행한 것**이며, 원 실험 실행
  시점의 테스트 결과가 아니다.
- 원 실험은 per-run 로그 디렉터리(`run.log`, `metadata.json`, `metrics.json`)를
  생성하지 않았다. `experiments/results/`의 JSON/CSV만 남아 있다. 현재 저장소의
  `src/animal_re_id/utils/logging.py`는 파일 타임스탬프상 이후 추가된 유틸리티이며,
  이 실험 결과에 소급 적용되지 않았다.
- 원 실험의 stdout은 저장되어 있지 않다. 결과 JSON의 `runtime_seconds`와
  `created_at`만 확인 가능하다.

### 3.4 데이터 무결성 및 누출 검사

| 검사 | 방법 | 결과 |
|---|---|---|
| Sequence key 유일성 | `check_assignments()` — `(identity, video)` 복합 키 | 통과 (코드 경로 확인) |
| 공식 fold 합계 | 릴리스 fold 파일 합집합 = `track_info.csv` 1,431 시퀀스 / 138,363 프레임 | 통과 (`docs/benchmark_protocols.md` §2.1) |
| Split 합계 | 결과 JSON `split.counts` = 1,431 시퀀스, 13 identity, 138,363 프레임 | 통과 |
| Query/Gallery closed-set | 5개 fold 모두 `n_queries_without_positives = 0` | 통과 |
| Self-match 제외 | `path` 동일 항목을 ranking에서 제외 | 코드 확인 |
| 데이터 누출 | 사전학습 데이터와 평가 데이터의 중복 가능성 검토 | **조건부 — 6.1절 참조** |
| 이미지 단위 near-duplicate | 같은 시퀀스 프레임을 다중 사용하지 않음 (대표 1장) | 설계상 회피, 별도 검사는 `Not verified` |

### 3.5 평가 지표 검산

`*_per_query.csv`의 `first_positive_rank`와 `average_precision`에서 지표를 다시
계산했다. JSON의 aggregate는 **fold별 지표의 산술평균(macro)**, 재계산값은
**전체 query를 합친 pooled 평균**이므로 fold 크기 차이(287 vs 286)만큼 미세하게
다르다.

| 지표 | 원본 JSON (macro) | Per-query 재계산 (pooled) | 차이 | 설명 |
|---|---|---|---|---|
| Rank-1 | 0.777103 | 0.777079 | -0.000024 | fold 크기 287/286 차이 |
| Rank-3 | 0.882598 | 0.882600 | +0.000002 | 반올림 수준 |
| mAP | 0.396873 | 0.396892 | +0.000019 | fold 크기 차이 |

- 1,431개 query 전부 `scored=True`, `n_queries_without_positives=0`.
- Rank-1 miss: 319 / 1,431 (22.3%), Rank-3 miss: 168 / 1,431 (11.7%).
- `average_precision = 0`인 query는 0건.
- 위 재계산은 **보고서 작성 시점의 검산**이며, 원 실험을 다시 실행한 것이 아니다.

---

## 4. Results

### 4.1 주요 지표

**주 결과: gallery = `fold` (테스트 fold 내 leave-one-out)**

| 단위 | Rank-1 | Rank-3 | mAP | Query 수 | Gallery 수 |
|---|---|---|---|---|---|
| Fold 1 | 0.7422 | 0.8850 | 0.4234 | 287 | 287 |
| Fold 2 | 0.7308 | 0.8252 | 0.3513 | 286 | 286 |
| Fold 3 | 0.8112 | 0.8951 | 0.4237 | 286 | 286 |
| Fold 4 | 0.8007 | 0.8986 | 0.3973 | 286 | 286 |
| Fold 5 | 0.8007 | 0.9091 | 0.3886 | 286 | 286 |
| **평균 ± 표준편차** | **0.7771 ± 0.0376** | **0.8826 ± 0.0332** | **0.3969 ± 0.0299** | 1,431 | 287/286 |

**비교 결과: gallery = `all` (전체 1,431 시퀀스)**

| 단위 | Rank-1 | Rank-3 | mAP | Query 수 | Gallery 수 |
|---|---|---|---|---|---|
| Fold 1 | 0.7735 | 0.8920 | 0.2684 | 287 | 1,431 |
| Fold 2 | 0.7692 | 0.8601 | 0.2573 | 286 | 1,431 |
| Fold 3 | 0.8077 | 0.9161 | 0.2940 | 286 | 1,431 |
| Fold 4 | 0.8042 | 0.9231 | 0.2809 | 286 | 1,431 |
| Fold 5 | 0.7587 | 0.8951 | 0.2569 | 286 | 1,431 |
| **평균 ± 표준편차** | 0.7827 ± 0.0219 | 0.8973 ± 0.0247 | 0.2715 ± 0.0160 | 1,431 | 1,431 |

값은 소수점 넷째 자리에서 반올림했다. 표준편차는 5개 fold 값의 표본표준편차
(`ddof=1`)다. 원본 전체 정밀도는 결과 JSON에 있다.

### 4.2 Baseline 비교

동일한 split·전처리·평가 단위에서 측정된 **다른 모델 baseline은 이 저장소에 없다**
(`Not available`). 따라서 아래 표는 baseline이 아니라, 같은 실험 안에서 조건만
바꾼 비교와 참고용 별도 프로토콜을 구분해 제시한다.

| 방법 | Rank-1 | Rank-3 | mAP | 동일 조건 여부 | 비고 |
|---|---|---|---|---|---|
| Frozen MegaDescriptor-L-384, gallery=`fold` | 0.7771 | 0.8826 | 0.3969 | `Yes` (주 결과) | 공식 5-fold |
| Frozen MegaDescriptor-L-384, gallery=`all` | 0.7827 | 0.8973 | 0.2715 | 모델·split 동일, gallery만 다름 | 조건 비교 |
| Frozen MegaDescriptor-L-384, RoVF split | 0.6607 | 0.8143 | 0.2908 | `No` (split·query 구성 다름) | 단일 test split, n=280 |
| Frozen MegaDescriptor-L-384, stratified redraw s0 | 0.6307 | 0.7906 | 0.2782 | `No` (fold 재추첨) | split 민감도 |
| Frozen MegaDescriptor-L-384, stratified redraw s1 | 0.6231 | 0.7855 | 0.2863 | `No` (fold 재추첨) | split 민감도 |
| Frozen MegaDescriptor-L-384, stratified redraw s2 | 0.6256 | 0.7904 | 0.2875 | `No` (fold 재추첨) | split 민감도 |

RoVF 수치는 image-level로 평가한 것이며, RoVF 논문의 video-level 수치와 직접
비교할 수 없다(`docs/benchmark_protocols.md` §3.3).

### 4.3 논문 결과와의 비교 가능성

| 항목 | Zuerl et al. (2023) | 이번 실험 | 비교 가능 여부 |
|---|---|---|---|
| Backbone | 논문 자체 image-based 모델 | Frozen MegaDescriptor-L-384 | `No` |
| 학습 | 5-fold CV (4 fold 학습 / 1 fold 평가) | 학습 없음 (frozen) | `No` |
| 대표 프레임 | 이미지 단위, 선택 규칙 미공개 | 중앙 프레임 1장 (문서화된 규칙) | `No` |
| Gallery 범위 | 미공개 | `fold`와 `all` 모두 보고 | `No` (논문 쪽 미확인) |
| 평가 단위 | Image (정의에 모호성 있음) | Image (시퀀스당 1장) | 부분 일치 |
| 보고 수치 (image-based) | Rank-1 0.853 ± 0.011, mAP 0.454 ± 0.024 | 0.7771, 0.3969 (gallery=`fold`) | **직접 비교 금지** |
| 보고 수치 (video-based) | Rank-1 0.966 ± 0.016, mAP 0.882 ± 0.090 | 해당 없음 (image-level만 수행) | `No` |

논문의 image-based 수치와 이번 결과는 **수치만 나란히 놓을 수 있을 뿐**,
backbone·학습 방식·프레임 선택·gallery 범위가 다르므로 "재현" 또는 "동등 비교"로
해석하지 않는다. 특히 이번 모델은 사전학습 데이터 중복 가능성이 있어(6.1절),
논문보다 불리하거나 유리한 어느 방향으로도 단정할 수 없다.

### 4.4 실행 시간 / GPU 사용량 / Cache

| 항목 | 값 | 출처 |
|---|---|---|
| 전체 실행 시간 (gallery=`fold`) | 167.95 s | `..._foldgallery.json` |
| Feature extraction (gallery=`fold`) | 125.53 s, 1,431 images, cache miss | `..._foldgallery.json` |
| 전체 실행 시간 (gallery=`all`) | 29.66 s, cache hit | `..._allgallery.json` |
| 전체 실행 시간 (RoVF split) | 36.04 s, cache hit | `..._rovf_foldgallery.json` |
| 전체 실행 시간 (stratified s0/s1/s2) | 37.29 / 37.42 / 32.06 s, cache hit | 각 결과 JSON |
| Feature cache | `experiments/cache/features_cd47ec852b70ce62.npz` | 결과 JSON `model.cache_path` |
| Cache key | `cd47ec852b70ce62da87d5e51c4dc2d57c924513836e8ad337af4bca53ff5e66` | 결과 JSON `model.cache_key` |
| GPU | RTX 4060 Laptop, 8.59 GB | 결과 JSON `environment` |

실행 시간은 노트북 GPU에서 측정된 단일 실행 값이며, 성능 벤치마크가 아니다.
Cache hit 시에는 feature extraction 시간이 0으로 기록된다.

### 4.5 반복 실험 통계

| 조건 | 반복 수 | Rank-1 | Rank-3 | mAP |
|---|---|---|---|---|
| 공식 fold, gallery=`fold` | 5 fold (동일 모델, 1회 실행) | 0.7771 ± 0.0376 | 0.8826 ± 0.0332 | 0.3969 ± 0.0299 |
| 공식 fold, gallery=`all` | 5 fold (동일 모델, 1회 실행) | 0.7827 ± 0.0219 | 0.8973 ± 0.0247 | 0.2715 ± 0.0160 |
| 공식 fold 라벨 무작위 치환 | 5회 | 0.6240 ± 0.0164 | — | — |
| Stratified 재추첨 | 3 seeds | 0.6265 ± 0.0039 (계산값) | — | — |

- 위 표의 ±는 **fold 간 변동**이지 run-to-run 변동이 아니다.
- 동일 조건 반복 실행(다른 난수 없이 같은 입력으로 재실행)은 수행하지 않았다.
  결정론적 추론이 기대되지만 **검증하지 않았다**.
- Stratified 3개 값(0.6307, 0.6231, 0.6256)의 평균·표준편차는 이 보고서에서
  계산한 값이며, 원본 `fold_sensitivity.json`에는 없다.

---

## 5. Analysis

### 5.1 확인된 실험 결과

- 공식 fold, gallery=`fold`에서 **Rank-1 0.7771 ± 0.0376, Rank-3 0.8826 ± 0.0332,
  mAP 0.3969 ± 0.0299**를 얻었다.
- Gallery를 전체 1,431장으로 확장하면 **Rank-1은 0.7827로 거의 같고(+0.0056)**,
  **mAP는 0.2715로 0.1254 감소**한다.
- mAP 감소는 **5개 fold 모두에서** 나타났다(0.4234→0.2684, 0.3513→0.2573,
  0.4237→0.2940, 0.3973→0.2809, 0.3886→0.2569).
- Rank-1의 gallery scope 변화는 fold마다 방향이 혼재한다(3개 fold 상승, 2개 fold 하락).
- Fold 간 변동 폭은 Rank-1 0.7308–0.8112, mAP 0.3513–0.4237이다.
- 1,431개 query 전부 positive를 가지며 scored되었다.

### 5.2 가능한 원인과 근거

- **mAP 하락 (추정)**: all-gallery에서는 query당 eligible positive가 21.1개에서
  109.5개로 늘고 gallery도 287/286에서 1,431로 커진다. AP는 상위 rank의 precision에
  민감하므로, positive가 많아질수록 상위권에 모든 positive를 모으기 어려워진다.
  방향은 5개 fold 모두 일관되지만, gallery 크기와 positive 수 중 어느 쪽이 얼마나
  기여했는지는 분리 실험하지 않았다.
- **Rank-1이 거의 변하지 않은 이유 (추정)**: Rank-1은 최상위 1개만 보므로 gallery
  확장의 영향을 상대적으로 덜 받는다. 다만 이것은 사후 해석이며, 별도 ablation으로
  확인하지 않았다.
- **Fold 2의 낮은 성능 (미확인)**: Fold 2는 Rank-1 0.7308, mAP 0.3513으로 가장
  낮다. Fold별 identity 구성이나 촬영 조건 차이를 분석하지 않았으므로 원인은
  **미확인**이다.
- **공식 fold와 재추첨 fold의 큰 격차 (문서 기록)**: 공식 fold 0.7771 vs 치환 평균
  0.6240, 차이 +0.1531이다. `docs/benchmark_protocols.md` §6은 저자가 트랙릿을
  세션/촬영 조건 기준으로 묶었을 가능성을 제시하지만, 릴리스 메타데이터만으로는
  확정할 수 없다고 기록한다. 이 보고서도 그 이상을 주장하지 않는다.

### 5.3 가설의 지지 여부

사전 등록 가설이 없으므로 RQ1은 기술(descriptive) 결과로만 답한다.

| 질문/가설 | 관측 | 판단 |
|---|---|---|
| RQ1: frozen feature의 공식 fold 성능 | Rank-1 0.7771, mAP 0.3969 | 기술적 답변 (가설 검정 아님) |
| RQ2 (사후): gallery 확장 시 mAP가 달라질 것 | mAP 0.3969 → 0.2715, 5/5 fold 감소 | **부분 지지** (방향 일관, 기여도 미분리) |
| RQ2 (사후): gallery 확장 시 Rank-1도 일관되게 변할 것 | 3개 fold 상승, 2개 fold 하락 | **판단 불가** |

### 5.4 Failure Cases

Per-query CSV에서 확인한 **Rank-1 실패 상위 사례**다. 원인 분석(이미지 검토, 촬영
조건 확인)은 하지 않았으므로 원인은 `Not verified`다.

| Sequence | Identity | First positive rank | AP | Fold |
|---|---|---|---|---|
| `Nana\|80` | Nana | 88 | 0.0595 | 2 |
| `Nora\|102` | Nora | 87 | 0.0586 | 5 |
| `Charlotte\|13` | Charlotte | 64 | 0.0629 | 3 |
| `Finja\|4` | Finja | 61 | 0.0603 | 1 |
| `Hertha\|51` | Hertha | 53 | 0.0704 | 1 |

**AP 하위 사례**는 Rank-1 실패와 순서가 다르다. 예: `Nanuq\|17` (rank 18, AP 0.0478,
fold 4), `Sprinter\|85` (rank 27, AP 0.0519, fold 2), `Charlotte\|14` (rank 16,
AP 0.0585, fold 5). 이는 첫 positive가 늦는 것과 positive 전체가 상위에 몰리는
것이 다른 실패 양상임을 보여준다.

Per-identity 재집계(per-query CSV 기준, fold 혼합):

| Identity | Query 수 | Rank-1 | mAP |
|---|---|---|---|
| Charlotte | 112 | 0.6518 | 0.3241 |
| Finja | 107 | 0.7290 | 0.3009 |
| Nana | 119 | 0.7227 | 0.2659 |
| Vitus | 104 | 0.9135 | 0.5895 |
| Larissa | 107 | 0.9252 | 0.5229 |

전체 13개 identity 중 Charlotte가 가장 낮고 Vitus/Larissa가 가장 높다. 다만
identity별 query 수가 100–123개로 작고 fold 구성이 섞여 있으므로, 이 차이는
기술적 관찰이며 통계적 유의성 검정을 하지 않았다.

### 5.5 예상과 다른 결과 및 부정적 결과

- **All-gallery mAP의 큰 하락**: gallery를 넓히면 더 많은 positive를 찾을 수 있다는
  직관만 보면 성능이 좋아질 것 같지만, mAP는 0.1254 하락했다. mAP가 상위 rank의
  precision에 민감하기 때문에 지표 정의상 설명 가능한 방향이며, gallery scope 없이
  mAP를 보고하면 오해를 부를 수 있음을 보여준다.
- **공식 fold의 높은 성능**: 공식 fold 0.7771은 무작위 치환 평균 0.6240보다 0.1531
  높다. 이는 모델 성능이라기보다 **split 난이도 차이**로 해석된다(추정). 공식 fold
  수치를 다른 촬영 조건으로의 일반화 성능으로 해석하면 안 된다.
- **논문 수치와의 차이**: 논문 image-based Rank-1 0.853 / mAP 0.454보다 낮지만,
  backbone·학습·프레임 선택이 달라 직접 비교할 수 없다. 이 차이를 모델 우열의
  근거로 쓰지 않는다.
- **긴 꼬리 실패**: 평균 Rank-1이 0.7771이어도 319개 query(22.3%)는 Rank-1에
  실패하며, worst case는 rank 88까지 내려간다.
- **검증 공백**: checkpoint integration test 4개는 skip되었고, 원 실험의 stdout/
  종료 코드는 저장되지 않았다.

---

## 6. Limitations

### 6.1 Dataset Leakage

- PolarBearVidID는 `wildlife-datasets` 컬렉션에 포함되어 있고, MegaDescriptor 계열은
  같은 컬렉션의 공개 재식별 데이터셋으로 사전학습되었다(HF 모델 카드 기준).
- PolarBearVidID는 WildlifeReID-10k의 구성 데이터이기도 하며, WildlifeReID-10k
  논문은 이 중복 때문에 MegaDescriptor-L을 비교 대상에서 제외했다.
- 따라서 이 결과는 **zero-shot이 아니며**, 학습 데이터와 평가 데이터의 개체가
  겹칠 가능성이 높은 조건(transductive/leaky)이다.
- 체크포인트가 실제로 본 이미지 목록은 공개되어 있지 않아 **개별 이미지 단위의
  중복은 검증할 수 없다**. 위 서술은 데이터셋 구성과 모델 카드에 근거한 것이다.
- 결론: 이 수치를 SOTA로 표기하거나, 누출이 없는 벤치마크처럼 다른 방법과
  나란히 비교하지 않는다.

### 6.2 Protocol Mismatch

- 논문은 image-level gallery를 test fold로 제한했는지 전체로 두었는지 명시하지
  않는다. 이 실험은 두 변형을 모두 보고하지만, 어느 쪽이 논문과 같은지는
  **미확인**이다.
- 논문의 대표 프레임 선택 규칙은 공개되지 않았다. 이 실험의 중앙 프레임 규칙은
  문서화된 자체 규칙이며, 논문과 같다는 보장이 없다.
- "image-level"은 (a) 시퀀스당 대표 1장, (b) 전체 138,363 프레임의 두 해석이
  가능하다. 이 실험은 (a)만 수행했고 (b)는 실행하지 않았다.
- 논문은 자체 모델을 4 fold로 학습하지만, 이 실험은 학습하지 않는다.
- Rank-1/Rank-3/mAP의 정의는 맞추었지만, 위 차이 때문에 논문 수치와의 직접
  비교는 금지된다.

### 6.3 사전학습 데이터 노출

- 6.1절과 동일하다. MegaDescriptor-L-384의 사전학습 데이터에 PolarBearVidID가
  포함되었을 가능성이 있다.
- 결과 JSON에는 Hub ID만 있고 **checkpoint revision hash가 없어**, 어떤 가중치가
  사용되었는지 고정할 수 없다.
- 사전학습 데이터 노출을 통제한 비교(예: DINOv2 frozen)는 이번에 수행하지 않았다.

### 6.4 통계적 불확실성

- 평가 개체는 13마리뿐이며, fold당 query는 286–287개다.
- 보고된 ±는 5개 fold 값의 표본표준편차(`ddof=1`)이며, 신뢰구간이나 부트스트랩
  추정은 없다.
- 같은 identity의 query들이 fold 안에서 서로 상관되어 있어, query를 독립 표본으로
  가정한 오차 추정은 낙관적일 수 있다.
- 동일 조건 반복 실행이 없어 run-to-run 변동을 측정하지 못했다.
- Per-identity 비교(5.4절)는 기술적 관찰이며 유의성 검정을 하지 않았다.

### 6.5 재현성 한계

- 결과 JSON의 `git_commit`은 `6dd705c...`이지만 `git_dirty = true`다. 실행 시점에
  작업 트리에 커밋되지 않은 변경이 있었다.
- 보고서 작성 시점 기준으로 `scripts/`, `src/animal_re_id/data/`, `evaluation/`,
  `models/`, `tests/`, `config/`, `docs/`의 상당 부분이 아직 Git에 추가되지 않은
  상태다. 따라서 기록된 commit만으로 실험 코드를 재구성할 수 **없다**.
- Split signature(`c66a81...`), dataset signature(`ad9d02...`), feature cache key는
  기록되어 있어, 같은 산출물을 사용했는지는 확인할 수 있다.
- Checkpoint revision hash가 없고, CUDA 커널·timm 버전에 따라 마지막 자리 수치가
  달라질 수 있다.
- 결과 JSON과 per-query CSV는 남아 있지만, 원 실험의 stdout/로그는 없다.
- 보고서 작성 중 저장소의 `scripts/evaluate.py` 등 기존 코드가 변경되는 것이
  관찰되었다. 이번 작업은 해당 코드를 수정하지 않았지만, 현재 코드와 실험 시점
  코드가 동일하다는 보장은 없다.

### 6.6 확인되지 않은 조건

- 전체 138,363 프레임 image-level 평가: 실행하지 않음.
- 프레임 선택 민감도(중앙 vs 첫/마지막 vs 다중 프레임 평균): 측정하지 않음.
- Zoo별 성능 분해: 실행하지 않음.
- Fine-tuning 또는 linear probing: 실행하지 않음.
- 다른 backbone(DINOv2 등)과의 동일 조건 비교: 실행하지 않음.
- Video-level / RoVF 프로토콜: image-level 결과만 있으며, RoVF의 clip·video 집계는
  재현하지 않음.
- Run-to-run 반복 및 결정론 검증: 하지 않음.
- 원 실험 시점의 전체 테스트 스위트: `Not verified` (2026-11-18 재실행 결과만 있음).
- 개별 이미지의 사전학습 노출 여부: `Not verified`.

---

## 7. Conclusion & Next Steps

### 7.1 결론

공식 5-fold PolarBearVidID image-level 프로토콜에서 frozen MegaDescriptor-L-384는
gallery=`fold` 기준 **Rank-1 0.7771 ± 0.0376, Rank-3 0.8826 ± 0.0332,
mAP 0.3969 ± 0.0299**를 기록했다. 같은 feature로 gallery를 전체 1,431장으로
확장하면 Rank-1은 0.7827로 거의 유지되지만 mAP는 0.2715로 크게 낮아진다.

이 결과는 **공식 fold라는 특정 조건의 closed-set 베이스라인**이며, 다음 이유로
일반화 성능이나 논문 재현으로 해석하지 않는다.

1. MegaDescriptor의 사전학습 데이터와 평가 데이터가 겹칠 가능성이 있다.
2. 공식 fold는 무작위 재추첨보다 약 0.153 Rank-1 높아, split 난이도 효과가 크다.
3. 논문과 backbone·학습 방식·프레임 선택·gallery 범위가 다르다.
4. 실행 시점 코드가 Git에 완전히 커밋되지 않아 완전한 재현이 어렵다.

### 7.2 후속 실험

| 우선순위 | 실험 | 목적 | 선행 조건 |
|---|---|---|---|
| P1 | DINOv2 frozen, 동일 split·대표 프레임 | 누출 가능성이 낮은 공정 기준선 | 모델 로더 추가 |
| P1 | 프레임 선택 민감도 (중앙 vs 첫/마지막 vs 다중 프레임) | 대표 프레임 규칙 의존성 측정 | 없음 |
| P1 | 코드 커밋 및 checkpoint revision 고정 후 재실행 | 재현성 확인 | Git 정리, Hub revision 기록 |
| P2 | Zoo별 성능 분해 | 난이도 차이의 출처 탐색 | fold 메타데이터 유지 |
| P2 | Gallery 크기/positive 수 분리 ablation | mAP 하락의 기여도 분해 | 동일 feature 재사용 |
| P3 | RoVF video-level 프로토콜 구현 | video 단위 비교 | 20-frame clip, co-occurrence negative, video 집계 구현 |
| P3 | 전체 프레임 image-level 평가 | "image-level" 정의 (b) 검증 | 계산 비용·near-duplicate positive 처리 방침 확정 |

### 7.3 남은 연구 질문

- 논문이 사용한 gallery 범위는 `fold`와 `all` 중 어느 쪽인가?
- 공식 fold가 무작위 fold보다 쉬운 이유는 무엇인가(세션/촬영 조건/트랙릿 묶음)?
- 대표 프레임 선택을 바꾸면 순위가 얼마나 달라지는가?
- 사전학습 데이터 노출을 통제한 모델에서도 같은 순위 경향이 나오는가?
- Query당 failure의 원인은 개체 외형, 자세, 조명, 카메라, occlusion 중 무엇인가?
- 동일 조건 반복 실행 시 수치가 완전히 결정론적인가?

---

## Artifacts

| 종류 | 경로 / ID | 비고 |
|---|---|---|
| Config | `config/polarbear.yaml` | `five_fold`, gallery=`fold` |
| Metrics (주 결과) | `experiments/results/megadescriptor_frozen_polarbear_5fold_foldgallery.json` | 2026-11-17T17:20:56Z |
| Per-query (주 결과) | `experiments/results/megadescriptor_frozen_polarbear_5fold_foldgallery_per_query.csv` | 1,431 rows |
| Metrics (all gallery) | `experiments/results/megadescriptor_frozen_polarbear_5fold_allgallery.json` | 2026-11-17T17:21:45Z |
| Per-query (all gallery) | `experiments/results/megadescriptor_frozen_polarbear_5fold_allgallery_per_query.csv` | 1,431 rows |
| Metrics (RoVF split) | `experiments/results/megadescriptor_frozen_polarbear_rovf_foldgallery.json` | 2026-11-17T17:22:53Z, n=280 |
| Metrics (stratified s0/s1/s2) | `experiments/results/megadescriptor_frozen_polarbear_5fold_stratified_s{0,1,2}_foldgallery.json` | split 민감도 |
| Fold sensitivity | `experiments/results/fold_sensitivity.json` | 5 permutations |
| Split | `experiments/splits/polarbear_five_fold_seed0.json` | signature `c66a81...` |
| Feature cache | `experiments/cache/features_cd47ec852b70ce62.npz` | key `cd47ec85...` |
| Runtime log | `Not available` | 원 실험은 per-run `run.log` 미생성; 결과 JSON에 runtime만 기록 |
| Model checkpoint | HF Hub `BVRA/MegaDescriptor-L-384` | revision `Not available` |
| Git Commit | `6dd705c6d66fcb4f2858a50676ab9b1b34895bc1` | `dirty`; 코드 전체가 커밋되지 않음 |
| Run ID | `2026-11-17T17:20:56.309799+00:00` | 결과 JSON `created_at` |
| Protocol document | `docs/benchmark_protocols.md` | 데이터·프로토콜 검증 기록 |
| Evaluation code | `scripts/evaluate.py` | 결과 JSON/CSV 생성 |
