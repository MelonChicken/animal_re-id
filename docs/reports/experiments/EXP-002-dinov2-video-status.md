# EXP-002 — DINOv2 비디오 평가 및 프로젝트 현황

| 항목 | 값 |
|---|---|
| Report ID | `EXP-002` |
| Experiment ID | `dinov2_s_mean_video`, `dinov2_s_mean_video_rovf_preproc` |
| 작성일 / 실행일 | 2026-11-18 / 2026-11-18 (KST) |
| 상태 | Completed (보존된 unmasked RoVF split 실행 2건) |
| 작성자 / 사람 검토 | Codex / Not verified |
| 관련 문서 | `docs/dinov2_video_baseline.md`, `docs/benchmark_protocols.md`, `docs/reports/experiments/EXP-001-megadescriptor.md` |
| Git Commit | `6dd705c6d66fcb4f2858a50676ab9b1b34895bc1` (두 실행 및 현 점검 모두 dirty) |

## 1. Overview

### 1.1 Objective

PolarBearVidID에서 frozen DINOv2-S를 사용한 비디오 기준선과 현재 저장소의 재현 가능 범위를 점검했다. 가장 최근 실험인 DINOv2 실행 2건을 본 보고서의 대상으로 삼았다.

### 1.2 Research Question

RoVF test split에서 입력 프레임 수, 공간 풀링, 정규화에 따라 pair 검색과 closed-set 시퀀스 검색 지표가 어떻게 달라지는가?

### 1.3 Hypothesis

사전 등록된 가설은 Not available.

### 1.4 핵심 결과 요약

| 항목 | 결과 | 비고 |
|---|---|---|
| RoVF pair Top-1 / Top-3 | 26.5% / 54.5% | 기본 설정, ImageNet 정규화, patch 평균, 20프레임 |
| Closed-set Rank-1 / Rank-3 / mAP | 0.836 / 0.932 / 0.407 | 기본 설정, test 280개 |
| 상태 | Completed | 두 실행의 완료 로그, 결과 JSON 및 per-query CSV 확인 |

## 2. Experimental Setup

### 2.1 Dataset 및 데이터 버전

| 항목 | 값 | 출처 |
|---|---|---|
| Dataset | PolarBearVidID | `data/raw/PolarBearVidID/` |
| Version / Release | 로컬 아카이브 해시 Not verified | [원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10000026/) |
| 규모 | JPEG 프레임 138,363장, 시퀀스 1,431개, 개체 13마리 | 두 실행의 메타데이터 |
| 전처리 전 원본 | 시퀀스별 프레임 JPEG, 전체 원본 해상도 분포 Not verified | 로컬 데이터 |

### 2.2 Train / Validation / Test Split

| 항목 | 값 |
|---|---|
| Split 프로토콜 | RoVF 70/10/20 tracklet split |
| Split 파일 | `experiments/splits/polarbear_rovf_split_seed0.json` |
| Split signature | `6728cdbb34a73b4b14c729670799bb5a0a362bf670511a862231a07a24ccd2ed` |
| 분할 크기 | train 1,006 / val 145 / test 280 |
| Seed | 메타데이터 값 0 |
| 재사용 여부 | 두 실행의 경로와 signature 일치. 최초 작성 시점 Not verified |

### 2.3 Query / Gallery Protocol

| 항목 | Pair 평가 | Closed-set |
|---|---|---|
| Query | 100개 positive pair를 양방향으로 평가해 200 trial | test 시퀀스 280개 |
| Gallery | trial당 후보 11개에서 query를 제외한 10개 | test 시퀀스 280개에서 해당 query 제외 |
| Positive / Negative | 각 query당 지정된 같은 개체 1개 / 다른 개체 9개 | 개체 ID가 같은 나머지 시퀀스 / 다른 개체의 시퀀스 |
| Scoring | Top-1/Top-3, 공식 tie 처리 방식 사용 | Rank-1/Rank-3/mAP, 280개 query 전부 채점 |
| 거리 | Euclidean | Euclidean |

출처는 `data/raw/rovf/Precomputed_test_examples_polarbear.csv`와 [RoVF 참조 구현](https://github.com/Strong-AI-Lab/RoVF-Meerkat-Reidentification)이다.

### 2.4 Evaluation Unit

시퀀스마다 샘플링한 프레임 임베딩을 평균해 시퀀스 임베딩을 만든다. Pair 평가는 trial 평균, closed-set은 test query 평균을 사용한다.

### 2.5 Model 및 Pretrained Checkpoint

| 항목 | 값 |
|---|---|
| Model / 아키텍처 | `facebook/dinov2-small` / Dinov2Model |
| 파라미터 수 / 임베딩 차원 | 22,056,576 / 384 |
| Checkpoint revision | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| 학습 여부 | Frozen |
| 라이브러리 | transformers 5.23.0, torch 2.11.0+cu128 |

### 2.6 Preprocessing / Frame Sampling

시퀀스마다 `np.linspace(0, n_frames - 1, 20, dtype=int)`로 20프레임을 뽑은 뒤, 5·10·20프레임으로 균등 재샘플링한다. 입력은 `3 × 224 × 224`이며 bicubic resize를 적용한다. ImageNet 모드는 픽셀을 `/255`로 스케일한 뒤 mean `(0.485, 0.456, 0.406)`과 std `(0.229, 0.224, 0.225)`로 정규화한다. `rovf` 모드는 `/255`만 적용한다.

### 2.7 Hyperparameters / Seed

Batch size는 32, num_workers는 0이며 모델 학습용 learning rate·epochs·optimizer는 해당 없음이다. 기본 결과는 `patches` 20프레임이다.

### 2.8 Hardware / Software Environment

Windows 11, Python 3.12.10, RTX 4060 Laptop GPU, torch의 CUDA 12.8 빌드, torchvision 0.26.0, NumPy 2.5.5, pandas 3.1.1. GPU 용량은 실행 메타데이터에 8.59 GB로 기록되어 있다.

## 3. Implementation & Validation

### 3.1 주요 구현 사항

PolarBearVidID 로더, 공식 5-fold와 RoVF split, frozen 특징 추출, 캐시, 검색 지표, 실험 로그 기록이 구현되어 있다. DINOv2는 `patches`, `all`, `cls`를 한 번의 프레임 추출에서 계산하며 프레임 수별로 시간 평균을 적용한다.

### 3.2 관련 코드와 파일 구성

| 경로 | 역할 |
|---|---|
| `src/animal_re_id/data/` | `wildlife.py` 데이터 로딩, `splits.py` 분할, `video.py` 프레임 샘플링 및 집계 |
| `src/animal_re_id/models/` | MegaDescriptor-L-384 및 DINOv2-S 특징 추출 |
| `src/animal_re_id/evaluation/` | closed-set 검색 지표와 RoVF pair 평가 |
| `src/animal_re_id/utils/` | 실행 환경 및 로그 기록 |
| `scripts/` | `evaluate.py`, `evaluate_video.py`, `fold_sensitivity.py` |
| `config/` | `polarbear.yaml`, `dinov2_video.yaml` |
| `tests/`, `docs/` | 테스트, 프로토콜 문서, 보고서 |
| `data/`, `experiments/` | 로컬 원본, split, 캐시, 결과와 실행 로그. Git ignore 확인 |

실행 진입점은 `uv run python scripts/evaluate.py --config config/polarbear.yaml` 및 `uv run python scripts/evaluate_video.py --config config/dinov2_video.yaml`이다. 보조 실행의 전처리와 실험명은 `--preprocessing rovf --experiment dinov2_s_mean_video_rovf_preproc`으로 지정할 수 있다. 실제 전체 평가 명령은 이번 점검 중 재실행하지 않았다.

### 3.3 테스트 결과

| 명령 | 실행 시점 | 결과 |
|---|---|---|
| `.\.venv\Scripts\python.exe -m pytest -q` | 2026-11-18 점검 | 123 passed, 9 skipped, 39 warnings. 경고는 `wildlife-datasets` 내부의 pandas `inplace` 사용에서 발생 |
| `RUN_MODEL_TESTS=1`, `HF_HUB_OFFLINE=1`로 `.\.venv\Scripts\python.exe -m pytest -q -m integration` | 2026-11-18 점검 | 12 passed, 120 deselected, 5 warnings. 로컬 캐시의 체크포인트 사용 |
| `.\.venv\Scripts\python.exe scripts/evaluate.py --help` 및 `.\.venv\Scripts\python.exe scripts/evaluate_video.py --help` | 2026-11-18 점검 | 두 스크립트가 인수 도움말을 출력하고 종료 코드 0으로 완료 |
| `.\.venv\Scripts\animal-re-id.exe --help` | 2026-11-18 점검 | ImportError. `pyproject.toml`이 지정한 `animal_re_id:main`이 빈 `__init__.py`에 없음 |

### 3.4 데이터 무결성 및 누출 검사

| 검사 | 방법 | 결과 |
|---|---|---|
| Sequence key 유일성 | 로컬 split JSON | 1,431개 행의 sequence key 모두 고유 |
| Split 합계 일치 | 로컬 split JSON 재집계 | 1,006 + 145 + 280 = 1,431 |
| Query/Gallery 교집합 | 평가 구현 및 per-query CSV를 test 시퀀스와 대조 | 기록된 query 및 pair positive는 모두 test 소속. Closed-set은 test 집합을 양쪽에 사용하며 self-match를 제외. Pair는 방향마다 현재 query를 제외 |
| Positive 없는 query | per-query CSV와 결과 JSON 확인 | closed-set에서 280/280 query 채점 |
| 사전학습 노출 | 모델 및 벤치마크 문서 검토 | DINOv2 중복 Not verified. MegaDescriptor는 개체 중복 가능성이 높으며 이미지 단위 중복 Not verified |

### 3.5 평가 지표 검산

| 지표 | 원본 JSON | CSV 재계산 |
|---|---|---|
| Pair Top-1 / Top-3 | 기본 설정 0.265 / 0.545 | 각각 성공 trial 수를 200으로 나눠 일치 확인 |
| Closed-set Rank-1 / Rank-3 | 기본 설정 0.835714 / 0.932143 | 첫 positive 순위가 1, 3 이내인 query 비율과 일치 |
| Closed-set mAP | 기본 설정 0.406524 | 채점 query별 AP 평균과 일치 |

각 DINOv2 run의 4,320개 per-query 행에서 18개 평가 조합의 지표를 검산했다. MegaDescriptor 주 결과도 fold별 CSV 지표를 계산해 JSON의 평균과 표본 표준편차를 확인했다.

## 4. Results

### 4.1 주요 지표

대표 조합을 표에 기록했다. Top-k는 퍼센트, closed-set 지표는 소수 셋째 자리까지 표시한다. 모든 행의 pair 평가는 200 trial, closed-set은 query/gallery 각각 280개로 구성했다.

| 전처리 / 공간 풀링 / 프레임 | Pair Top-1 / Top-3 | Closed-set Rank-1 / Rank-3 / mAP |
|---|---|---|
| ImageNet / patches / 20 (기본) | 26.5% / 54.5% | 0.836 / 0.932 / 0.407 |
| `/255` / patches / 20 | 25.5% / 53.5% | 0.818 / 0.929 / 0.390 |
| ImageNet / patches / 5 | 28.5% / 51.0% | 0.850 / 0.932 / 0.412 |
| `/255` / patches / 5 | 29.5% / 55.5% | 0.829 / 0.918 / 0.396 |
| ImageNet / CLS / 5 | 36.0% / 63.0% | 0.896 / 0.954 / 0.463 |

### 4.2 Baseline 비교

P0 MegaDescriptor는 공식 5-fold 배정, 시퀀스당 중앙 프레임 1장, fold 내부 gallery를 사용했다. Fold 평균 ± 표본 표준편차는 Rank-1 0.7771 ± 0.0376, Rank-3 0.8826 ± 0.0332, mAP 0.3969 ± 0.0299다. 전체 gallery, stratified seed 0~2 및 RoVF split 변형의 결과도 보존되어 있다. P1과는 split, 입력 프레임 수, 평가 조건이 달라 모델 간 우열을 판단할 수 없다.

### 4.3 논문 결과와의 비교 가능성

| 항목 | RoVF 논문 | 이번 실험 | 일치 여부 |
|---|---|---|---|
| Backbone / 학습 방식 | frozen DINOv2-S 평균 풀링 기준선 | frozen DINOv2-S | 대응 구성 있음 |
| 평가 단위 / Gallery | 시퀀스, 공식 pair 후보 | 같은 형식의 pair 평가와 별도 closed-set 평가 | Pair 평가에서 대응 |
| 전처리 / 입력 | 마스크 적용, 참조 코드에서 bilinear | 마스크 미적용, bicubic | 다름 |
| 보고 수치 | masked Table 3 | unmasked pair 및 closed-set | 직접 비교 불가 |

[RoVF 논문](https://link.springer.com/article/10.1007/s11263-025-02709-8)

### 4.4 실행 시간 / GPU 사용량 / Cache

| 항목 | ImageNet | `/255` | 출처 |
|---|---|---|---|
| 전체 run 시간 | 11.77초 | 11.51초 | 두 run의 metadata.json |
| 콜드 특징 추출 시간 | Not available | Not available | 두 run 모두 cache hit |
| Cache | `dinov2_frames_b0bcec0fec7d9487.npz` | `dinov2_frames_abf024c284ae5b3a.npz` | `experiments/cache/` |
| GPU peak memory | 0.09 GB | 0.09 GB | cache hit 시 메타데이터 |

### 4.5 반복 실험 통계

전처리마다 보존된 실행 1건이다. 같은 조건 반복 실행 및 분할 간 분산은 Not available.

## 5. Analysis

### 5.1 확인된 실험 결과

ImageNet 5프레임에서 CLS는 patch 평균보다 pair Top-1이 7.5%p, closed-set Rank-1이 약 4.6%p 높았다. 분석 대상 두 실행은 종료 상태가 completed이고 기록된 런타임 실패가 0건이다.

### 5.2 가능한 원인과 근거

CLS가 개체 구분에 필요한 정보를 더 보존했을 가능성은 추정이다. 동일 5프레임 조건에서의 지표 차이는 확인했으며 특징의 차이가 발생한 원인은 Not verified.

### 5.3 가설의 지지 여부

사전 등록 가설이 없어 판단 불가.

### 5.4 Failure Cases

ImageNet 기본 구성의 pair Top-1은 53/200건이다. Per-query CSV에서는 예를 들어 `Vera|109`가 `Vera|118`을 찾는 ImageNet patch 5프레임 trial의 positive 순위가 5위였다. 원인은 Not verified.

### 5.5 예상과 다른 결과 및 부정적 결과

ImageNet patch 평균에서 프레임 수가 5에서 20으로 늘어날 때 pair Top-1은 28.5%에서 26.5%로 낮아졌고 Top-3은 51.0%에서 54.5%로 높아졌다.

## 6. Limitations

### 6.1 Dataset Leakage

MegaDescriptor 학습용 데이터와 PolarBearVidID 평가 개체의 중복 가능성은 `EXP-001`에 기록되어 있다. RoVF 분할은 tracklet 기준이며, novel-identity 일반화 성능은 Not verified.

### 6.2 Protocol Mismatch

RoVF 논문은 masked 평가를 보고한다. 이번 결과에는 마스크를 적용하지 않았고 리사이즈 보간법도 다르다. 논문의 프레임별 순위 majority vote와 max pooling은 현재 결과의 범위 밖이다.

### 6.3 사전학습 데이터 노출

DINOv2 사전학습 코퍼스와 PolarBearVidID의 데이터 중복은 Not verified.

### 6.4 통계적 불확실성

RoVF test split 하나에서 측정했으며 신뢰구간은 Not available. 결과 JSON의 단일 관측치 표준편차 0은 반복 실행이나 분할 변동의 측정값으로 해석할 수 없다.

### 6.5 재현성 한계

`main`과 로컬 `origin/main`은 `6dd705c`를 가리킨다. 코드 주요 부분, 설정, 문서, 테스트와 `uv.lock`은 현재 미추적이고, 스테이지에는 `.idea/` 파일 6개가 있다. 두 실행의 dirty 상태는 커밋 해시만으로 재구성할 수 없다. 모델 코드가 직접 import하는 `transformers`는 `pyproject.toml`의 직접 의존성 목록에 없고 현재 lock에 전이 의존성으로 들어 있다.

### 6.6 확인되지 않은 조건

공식 5-fold DINOv2 비디오 측정, MegaDescriptor와 동일 대표 프레임 조건의 DINOv2 측정, masked 평가와 recurrent head 결과는 Not available. `config/dinov2_video.yaml`의 주석은 105쌍으로 표기하지만 로컬 CSV와 실행 결과는 100쌍이다.

## 7. Conclusion & Next Steps

### 7.1 결론

P0 MegaDescriptor 공식 5-fold 평가와 P1 DINOv2 unmasked RoVF 비디오 평가까지 진행됐다. DINOv2의 두 실행 결과는 기존 CSV에서 검산됐다. 동일 조건의 모델 비교와 커밋 기준 재현성 확보가 남았다.

### 7.2 후속 실험

| 우선순위 | 실험 | 목적 | 선행 조건 |
|---|---|---|---|
| P1 | DINOv2 공식 5-fold, 중앙 프레임 및 비디오 구성 | P0 대응 조건과 프레임 집계 효과 평가 | 동일 split, gallery, 지표와 프레임 샘플링을 명시 |
| P2 | masked RoVF 평가 | 논문 조건과 맞춘 비교 | 마스크 및 리사이즈 조건 확보 |

### 7.3 남은 연구 질문

- 동일 fold와 대표 프레임에서 두 frozen 모델의 차이는 어느 정도인가?
- 동일 후보 집합에서 마스킹이 pair 순위에 어떤 영향을 주는가?

## Artifacts

| 종류 | 경로 / ID | 비고 |
|---|---|---|
| Config | `config/dinov2_video.yaml` | 각 run 디렉터리에 실제 적용 설정 스냅샷 보존 |
| Metrics | `experiments/results/dinov2_s_mean_video_20261117T191336Z-03da46.json` 및 `experiments/results/dinov2_s_mean_video_rovf_preproc_20261117T191358Z-62df4f.json` | 각 run 디렉터리의 `metrics.json`도 존재 |
| Per-query | `experiments/results/dinov2_s_mean_video_20261117T191336Z-03da46_per_query.csv` 및 `experiments/results/dinov2_s_mean_video_rovf_preproc_20261117T191358Z-62df4f_per_query.csv` | 지표 검산 |
| Runtime log | `experiments/polarbear/dinov2_s_mean_video/20261117T191336Z-03da46/run.log` 및 `experiments/polarbear/dinov2_s_mean_video_rovf_preproc/20261117T191358Z-62df4f/run.log` | 동일 디렉터리에 `metadata.json` 보존 |
| Split | `experiments/splits/polarbear_rovf_split_seed0.json` | §2.2 signature 참조 |
| Feature cache | `experiments/cache/dinov2_frames_b0bcec0fec7d9487.npz`, `experiments/cache/dinov2_frames_abf024c284ae5b3a.npz` | 각각 ImageNet, `/255` |
| Model checkpoint | `facebook/dinov2-small` | §2.5 revision 참조 |
| Git Commit | `6dd705c6d66fcb4f2858a50676ab9b1b34895bc1` | dirty |
| Run ID | `20261117T191336Z-03da46`, `20261117T191358Z-62df4f` | UTC 기준 ID |
| P0 주 결과 | `experiments/results/megadescriptor_frozen_polarbear_5fold_foldgallery.json`, `experiments/results/megadescriptor_frozen_polarbear_5fold_foldgallery_per_query.csv` | 공식 5-fold, fold gallery |
