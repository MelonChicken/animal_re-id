# Frozen DINOv2-S 비디오 베이스라인 (P1)

이 문서는 Frozen DINOv2-S + 공간/시간 평균 풀링 비디오 베이스라인의 구현, 평가 조건,
측정값, 그리고 RoVF 논문(IJCV 2026, Table 3)과의 비교 가능성 여부를 기록한다.
**논문 수치를 재현했다고 주장하지 않는다.**

- 관련 코드: `src/animal_re_id/models/dinov2.py`, `src/animal_re_id/data/video.py`,
  `src/animal_re_id/evaluation/video_retrieval.py`, `scripts/evaluate_video.py`
- 설정: `config/dinov2_video.yaml`
- 실행 로그: `experiments/polarbear/<experiment>/<run_id>/`
- 결과: `experiments/results/dinov2_s_mean_video_*.json`, `*_summary.csv`, `*_per_query.csv`
- 공식 프로토콜 문서: `docs/benchmark_protocols.md`

---

## 1. 요약

1. 공식 RoVF 저장소의 평가 경로를 소스에서 직접 재구성했다. 핵심은 **클립 20프레임
   균등 샘플링, 평가 시 5/10프레임 재샘플링, `/255`만 적용, 11개 후보(positive 2 + negative 9),
   100세트 × 2방향 = 200 trial, Euclidean 거리, Top-1/Top-3** 이다.
2. 본 구현의 주 pooling은 **patch token 평균**(CLS 제외)이다. 논문 산문은 "patch embeddings
   평균"이라고 쓰지만 참조 코드의 `forward_strat="average"`는 **CLS를 포함한 257개 토큰 평균**이다.
   두 방식을 모두 추출해 차이를 측정했고(`patches` vs `all`), Top-1 차이는 최대 0.5%p였다.
3. 공식 pair 프로토콜(200 trial)에서 patch 평균 5프레임은 **Top-1 28.5% / Top-3 51.0%**
   (ImageNet 정규화), **29.5% / 55.5%** (`/255`만, 논문 전처리)다.
   논문의 masked DINOv2-S avg 5프레임은 24.0% / 41.0%다.
4. **직접 비교는 성립하지 않는다.** 논문은 masked 클립으로 평가했지만 마스크가 공개되지
   않아 본 작업은 unmasked로 실행했다. 이 차이는 수치 차이의 방향조차 알 수 없게 만든다.
5. 동일 split에서 closed-set 시퀀스 검색도 함께 보고한다(Rank-1 0.850 / Rank-3 0.932 /
   mAP 0.412, patch 평균 5프레임). 이 지표는 RoVF 프로토콜에 없으므로 별도로 표기한다.

---

## 2. 공식 평가 경로 재구성 (소스 확인 결과)

확인한 파일(모두 `Strong-AI-Lab/RoVF-Meerkat-Reidentification`, `primary` 브랜치):

| 파일 | 확인한 내용 |
|---|---|
| `Dataset/convert_polar_bear_dataset.py` | 클립당 20프레임 `np.linspace(0, len-1, 20, dtype=int)`, `cv2.resize(..., (224, 224))`, BGR→RGB 저장 |
| `dataloaders/ReID.py` | 평가 시 `np.linspace(0, 19, num_frames, dtype=int)` 재샘플링, `torch.from_numpy(uint8)/255.0`, `transformations=None`이면 resize·crop·정규화 없음 |
| `models/dinov2_wrapper.py` | `average`: 프레임별 **257개 토큰 전체 평균** 후 프레임 평균. `cls`: CLS만 프레임 평균. 출력 L2 정규화 없음 |
| `evaluation/get_metrics.py` | 거리 = **Euclidean**, `indices_of_smallest`로 서로 다른 거리값 3개 선택, Top-1/Top-3만 계산 (mAP 없음) |
| `Dataset/polarbears_h5files/Precomputed_test_examples_polarbear.csv` | 100행 × 11열. `Query` 열도 gallery 후보에 포함됨 |
| 논문 3.5.2절 | "100 unique sets ... positive examples were alternated as the query clip ... resulting in 200 sets" |

### 2.1 프로토콜의 실제 형태

- 파일 1행 = positive pair 2개 + negative 9개 = 후보 11개.
- `Query`라는 이름의 첫 열도 후보에 포함된다. 즉 "query 1 + gallery 10"이 아니라
  **11개 후보 안에서 자기 자신만 제외**하는 순위 문제다.
- 각 행에서 두 positive가 번갈아 query가 되므로 **200 trial**이다. 논문 서술과 일치한다.
- 11개 후보 중 query를 뺀 10개가 gallery이므로 chance Top-1은 10%, Top-3은 30%다.

### 2.2 prose와 코드의 충돌

| 항목 | 논문 산문 | 참조 코드 | 본 구현 |
|---|---|---|---|
| 공간 풀링 | patch embeddings 평균 | CLS 포함 257 토큰 평균 | `patches`(주) / `all`(코드 충실) / `cls` 모두 보고 |
| 전처리 | 명시 없음 | `/255`만 (정규화 없음) | `imagenet`(주) / `rovf`(논문 충실) 모두 실행 |
| 거리 | Euclidean | Euclidean | Euclidean |

`patches`와 `all`의 실측 차이는 작다. CLS 토큰 하나가 257개 평균에서 차지하는 비중이
1/257이고, DINOv2 토큰들이 공통 방향 성분을 크게 공유하기 때문이다. 즉 **"CLS 포함 여부"는
이 베이스라인에서 결론을 바꾸는 요인이 아니다.** 반면 `cls` 단독 사용은 뚜렷하게 다르다.

---

## 3. 구현

| 모듈 | 역할 |
|---|---|
| `data/video.py` | 20프레임 샘플링(`sample_frame_indices`), 2단계 샘플링 합성(`nested_frame_indices`), 프레임 평균(`temporal_mean`, `pool_sequence`), 시퀀스 임베딩 조립(`sequence_embeddings`) |
| `models/dinov2.py` | `facebook/dinov2-small` 로드(frozen, eval), 전처리 2종, `spatial_pool`(patches/all/cls), `SpatialPooler`로 한 번의 forward에서 3종 동시 추출, NaN/Inf/shape 검증, L2 정규화, 체크포인트 리비전 기록 |
| `evaluation/video_retrieval.py` | 공식 pair 파일 로드, clip id → sequence key 매핑, positive-pair 검증, Euclidean 거리, **공식 tie 규칙을 그대로 옮긴** `official_ranking`, Top-1/Top-3 |
| `utils/logging.py` | run 디렉터리(`config.yaml`, `metadata.json`, `metrics.json`, `run.log`, `failures.csv`), 단계별 타이밍, 실패 기록, 예외 시 traceback 보존. `metadata.json`에 실행 config 사본도 함께 남긴다 |
| `utils/runtime.py` | 환경/GPU/패키지/git 기록, dataset signature, feature cache, split cache (P0에서 분리) |
| `scripts/evaluate_video.py` | 위 요소를 연결한 단일 실행 경로 |

### 3.1 추출 방식

- 28,620 프레임(=1,431 시퀀스 × 20)을 **한 번만** forward하고, 3개 spatial reduction을
  한 행에 이어 붙여 `(frames, 3, 384)`로 저장한다. 이후 5/10/20프레임 변형은 캐시에서
  읽어 재조합하므로 forward를 반복하지 않는다. 콜드 추출은 5,000프레임 청크로 나누어
  `Extraction progress: done/total`을 로그에 남긴다.
- Feature cache key는 모델 설명·전처리·해상도·reduction 목록·프레임 경로 목록의 해시다.
  **split/protocol/seed는 key에서 제외**한다(임베딩은 split의 함수가 아니며, split signature는
  결과에 따로 기록된다). `cache_hit` 여부와 key를 결과 JSON에 남긴다.
- 체크포인트 리비전은 `transformers` 5.x가 `config._commit_hash`를 채우지 않아 HF 캐시의
  `refs/main`에서 읽는다(오프라인 조회). 기록된 리비전: `ed25f3a31f01632728cabb09d1542f84ab7b0056`.

### 3.2 재사용한 것과 새로 만든 것

- 재사용: `load_polar_bear_vid_id`, `add_sequence_columns`, `rovf_assignment`,
  `build_retrieval_task`, `rank_retrieval`, `cosine_similarity`, 기존 split/cache 작성기.
- 신규: 위 표의 비디오 전용 모듈. P0(MegaDescriptor 이미지 단위) 코드 경로는 변경하지 않았고,
  `scripts/evaluate.py`는 공통 helper를 `utils/runtime.py`에서 import하도록만 정리했다
  (기존 결과 수치는 재실행 없이 그대로 유효하다).

---

## 4. 실험 조건 (실측)

| 항목 | 값 |
|---|---|
| 체크포인트 | `facebook/dinov2-small` (revision `ed25f3a…`) |
| 백본 | frozen, eval mode, 22,056,576 파라미터, hidden 384, patch 14 |
| 입력 | 224 × 224 (원본 256 × 128을 정사각 리사이즈 = 가로 압축) |
| 토큰 | 1 CLS + 256 patch = 257 (논문 각주와 일치) |
| 클립 | 시퀀스당 20프레임, `np.linspace(0, T-1, 20, dtype=int)` |
| 평가 프레임 | 5 / 10 / 20 |
| split | `rovf_split` (train 1,006 / val 145 / test 280), signature `6728cdbb34a7…` |
| query/gallery | 공식 CSV 100세트 × 2 = 200 trial, 후보 11개 |
| 거리 | Euclidean, 임베딩 L2 정규화 없음 |
| 하드웨어 | RTX 4060 Laptop 8.6 GB, torch 2.11.0+cu128 |
| 실행 | `uv run python scripts/evaluate_video.py --config config/dinov2_video.yaml` |
| 실행 시 git | `6dd705c6d6`, **dirty = true** (작업 트리 미커밋 상태에서 측정) |

보존된 run은 feature cache를 재사용했으므로 `cache_hit = true`이고, forward pass 없이
약 1분(66.5초 / 68.1초)이 걸렸다. 대부분은 데이터 로드(21초)와 시퀀스 풀링(구성당 약 2초)이며
디스크·시스템 상태에 따라 변동한다. 같은 이유로 같은 실행도 11초까지 짧아질 수 있다.
콜드 추출 시간은 보존된 산출물에 기록되어 있지 않다(Not available).

---

## 5. 결과

### 5.1 공식 pair 프로토콜 (200 trial, unmasked)

Top-1 / Top-3 (%). 괄호 안은 논문의 해당 구성.

| 전처리 | 공간 풀링 | 5프레임 | 10프레임 | 20프레임 |
|---|---|---|---|---|
| ImageNet 정규화 | patches | 28.5 / 51.0 | 28.0 / 51.5 | 26.5 / 54.5 |
| ImageNet 정규화 | all (CLS 포함) | 28.5 / 51.5 | 27.5 / 51.5 | 26.5 / 54.5 |
| ImageNet 정규화 | cls | 36.0 / 63.0 | 36.5 / 59.0 | 39.0 / 57.5 |
| `/255`만 (논문 전처리) | patches | 29.5 / 55.5 | 26.5 / 53.5 | 25.5 / 53.5 |
| `/255`만 (논문 전처리) | all (CLS 포함) | 29.5 / 55.5 | 26.5 / 53.5 | 25.5 / 53.5 |
| `/255`만 (논문 전처리) | cls | 35.0 / 64.0 | 34.0 / 61.0 | 32.5 / 61.0 |

### 5.2 Closed-set 시퀀스 검색 (280 query, gallery = test 280)

Rank-1 / Rank-3 / mAP. 이 지표는 RoVF 프로토콜에 없다.

| 전처리 | 공간 풀링 | 5프레임 | 10프레임 | 20프레임 |
|---|---|---|---|---|
| ImageNet | patches | 0.850 / 0.932 / 0.412 | 0.836 / 0.929 / 0.408 | 0.836 / 0.932 / 0.407 |
| ImageNet | all | 0.846 / 0.932 / 0.412 | 0.836 / 0.929 / 0.408 | 0.836 / 0.932 / 0.407 |
| ImageNet | cls | 0.896 / 0.954 / 0.463 | 0.893 / 0.971 / 0.461 | 0.889 / 0.964 / 0.459 |
| `/255`만 | patches | 0.829 / 0.918 / 0.396 | 0.818 / 0.925 / 0.393 | 0.818 / 0.929 / 0.390 |
| `/255`만 | all | 0.829 / 0.918 / 0.396 | 0.818 / 0.925 / 0.393 | 0.818 / 0.929 / 0.390 |
| `/255`만 | cls | 0.861 / 0.943 / 0.437 | 0.857 / 0.932 / 0.434 | 0.857 / 0.925 / 0.433 |

### 5.3 관찰

- `patches`와 `all`은 pair 프로토콜에서 최대 0.5%p, closed-set에서 최대 0.4%p 차이다.
  257개 토큰 평균에서 CLS 하나의 영향이 작기 때문이며, **"CLS vs patch" 논쟁은 이 셋업에서
  실질적 차이가 아니다.**
- `cls` 단독은 pair Top-1이 +7.5%p, closed-set Rank-1이 +4.6%p 높다(5프레임, ImageNet).
  평균 풀링이 patch 토큰의 공통 성분에 묻히는 반면 CLS는 장면 단위 표현이기 때문으로 **추정**한다.
  인과 검증은 하지 않았다.
- 프레임 수를 5 → 20으로 늘려도 좋아지지 않는다. pair Top-1은 patch 평균에서 28.5 → 26.5로
  오히려 낮아진다. 시퀀스가 길어질수록 평균이 희석되는 것으로 보이나 **추정**이다.
- 전처리(`imagenet` vs `/255`)는 closed-set에서 ImageNet 쪽이 일관되게 +2%p 정도 높다.
  pair 프로토콜에서는 차이가 방향이 일정하지 않다.
- `patches`와 `all`의 순위가 사실상 동일하므로, 아래 논문 비교는 `all`을 기준으로 읽어도 같다.

---

## 6. RoVF 논문과의 비교

### 6.1 논문이 보고한 값 (IJCV 2026, Table 3, **masked**, polar bears)

| 모델 | 프레임 | Top-1 (%) | Top-3 (%) |
|---|---|---|---|
| Random chance | - | 10.0 | 30.0 |
| DINOv2-S avg (zero-shot, video) | 5 | 24.0 | 41.0 |
| DINOv2-S max (zero-shot, video) | 10 | 26.5 | 48.0 |
| DINOv2-S (zero-shot, frame-based) | 10 | 26.0 | 45.5 |
| RoVF-st mask (제안 모델, epoch 3) | 5 | 55.0 | 82.0 |

### 6.2 대응 관계

| 논문 행 | 본 작업의 대응 | 논문 | 본 작업 |
|---|---|---|---|
| DINOv2-S avg 5f | `all`, 5f, `/255` | 24.0 / 41.0 | 29.5 / 55.5 |
| DINOv2-S avg 5f | `all`, 5f, ImageNet | 24.0 / 41.0 | 28.5 / 51.5 |
| DINOv2-S max 10f | **미구현** (mean pooling만 범위) | 26.5 / 48.0 | Not available |
| DINOv2-S frame-based 10f | `cls` 10f (평균 풀링, 프로토콜 다름) | 26.0 / 45.5 | 36.5 / 59.0 |
| RoVF-st mask 5f | **범위 밖** (recurrent head 미구현) | 55.0 / 82.0 | Not available |

### 6.3 프로토콜 일치 여부

| 항목 | 논문/참조 코드 | 본 작업 | 일치 |
|---|---|---|---|
| 데이터·분할 | 1,431 시퀀스, 70/10/20, test 280 | 동일 함수, test 280 확인 | 예 |
| Query/gallery 세트 | 공식 CSV 100세트 × 2 | 동일 파일, 277개 clip 전부 test split에 존재 | 예 |
| 후보 수 | 11 (`Query` 열 포함) | 11 (`Query` 열 포함) | 예 |
| 클립 샘플링 | `linspace(0, T-1, 20)` | 동일 | 예 |
| 평가 재샘플링 | `linspace(0, 19, n)` | 동일 | 예 |
| 리사이즈 | `cv2.resize` (bilinear) | `torchvision` bicubic | **아니오 (보간 커널만)** |
| 전처리 | `/255`만 | `rovf` 모드에서 동일 | 예 (보조 run) |
| 공간 풀링 | 257 토큰 평균 | `all` 모드 | 예 |
| 시간 풀링 | 프레임 평균 | 동일 | 예 |
| 거리 | Euclidean | Euclidean | 예 |
| Top-k 규칙 | distinct distance 3개, tie 시 제거 | `official_ranking`으로 그대로 이식 | 예 |
| 마스크 | masked 클립 | **unmasked** | **아니오** |
| 프레임 기반 기준선 | 프레임별 순위 majority vote | 평균 풀링 | **아니오** |

### 6.4 직접 비교가 불가능한 이유

1. **마스크가 공개되지 않았다.** 저장소에는 `boxes.json`, `boxes_YOLO.json`만 있고
   `PB_masks.pkl`은 없다. 논문의 모든 Table 3 수치는 masked test set 기준이므로, unmasked
   수치와의 차이는 "모델 차이"인지 "배경 정보 차이"인지 분리할 수 없다. 방향도 알 수 없다
   (배경이 단서를 더할 수도, 잡음을 더할 수도 있다).
2. **프레임 기반 기준선의 집계가 다르다.** 논문은 프레임별 Top-k를 majority vote로 집계하지만
   본 작업의 `cls`는 CLS를 프레임 평균한다. 따라서 36.5 / 59.0과 26.0 / 45.5는 같은 문제의
   두 답이 아니다.
3. **`max` 풀링 미구현.** 논문의 DINOv2-S 최고 video 구성은 `max` 10프레임이다. 본 작업 범위는
   공간/시간 **평균**이므로 구현하지 않았다.
4. 보간 커널과 라이브러리 버전 차이(사소하지만 0.1%p 수준의 차이는 만들 수 있음).

따라서 본 결과는 **"같은 split·같은 query/gallery 세트·같은 거리에서 측정한 unmasked 기준선"**
으로만 읽어야 한다. 논문 수치보다 높게 나온 것은 배경 단서 때문일 수 있으나 검증하지 않았다.

---

## 7. 사전학습 데이터 누출 검토

- DINOv2는 레이블 없는 큐레이션 이미지(LVD-142M)로 자기지도 학습된 범용 백본이다.
- PolarBearVidID가 그 코퍼스에 포함되었다는 근거는 확인하지 못했다. 포함되지 않았다는
  근거도 확인하지 못했다. → **`Not verified`**
- MegaDescriptor와 달리 "재식별 데이터셋으로 지도학습되었다"는 구조적 누출은 없지만,
  **zero-shot이라고 단정하지 않는다.**
- 어떤 경우에도 **SOTA라는 표현을 쓰지 않는다.** 본 수치는 frozen 특징의 상한에 가깝다.

---

## 8. 한계

1. 마스크 미적용(6.4절).
2. `max` 풀링, RoVF recurrent head, fine-tuning 모두 범위 밖.
3. ImageNet 정규화가 주 설정이다. 논문 재현용 `/255` 결과도 함께 보고하지만, 논문의
   interpolation까지 맞추지는 않았다.
4. 20프레임은 논문의 grid(5/10)에 없지만 저장 클립 길이와 같아 참고용으로 포함했다.
5. RoVF test split은 단일 분할이므로 표준편차가 없다. closed-set 수치의 신뢰구간은 보고하지 않는다.
6. 실행 시 git이 dirty였다. 코드 상태는 이 문서와 테스트로 고정되지만 commit hash만으로는
   재현되지 않는다.
7. 콜드 추출 시간은 보존된 run에 기록되어 있지 않다.

---

## 9. 검증

- 기본 스위트: `uv run pytest -q` → **135 passed, 9 skipped**
- 체크포인트 포함: `$env:RUN_MODEL_TESTS="1"; uv run pytest -q` → **144 passed**
- 신규 테스트가 고정하는 것:
  - `tests/test_video.py`: `linspace` 샘플링 일치, 첫/마지막 프레임 포함, 결정성, 짧은 시퀀스
    패딩, 2단계 샘플링 합성, 프레임 순서 보존, 시간 평균, 시퀀스 임베딩 정렬
  - `tests/test_dinov2.py`: `patches`는 CLS 제외, `all`은 포함, `cls`는 CLS만, 세 결과가 서로
    다름, 전처리 2종, NaN/Inf/shape 거부, L2 정규화
  - `tests/test_video_retrieval.py`: Euclidean 계산, 공식 CSV 파싱, clip id 매핑, positive-pair
    검증, **공식 tie 규칙**, Top-1과 Top-3의 구분, self-match 제외
  - `tests/test_logging.py`: run 디렉터리 파일 생성, config 보존, status 전이, metrics 직렬화,
    실패 CSV, 예외 시 traceback과 로그 보존, metadata의 config 사본
  - `tests/test_runtime.py`: feature cache key의 결정성·민감도(split 열은 무시), cache
    round-trip, key/행 수 불일치 시 miss, **손상된 cache 파일을 miss로 처리**, 원자적 저장
  - `tests/test_dinov2_integration.py`: 실제 체크포인트에서 257 토큰·384 차원, 세 reduction의
    실제 차이, 결정성, 체크포인트 리비전 기록
- 합성 데이터로 로직을 먼저 검증한 뒤 실제 데이터로 실행했다.
- 실제 데이터 실행에서 공식 pair 파일의 clip id가 프로젝트 sequence key와 다르다는 점을
  발견해 매핑 계층(`sequence_clip_ids`)을 추가했다. 277개 clip 전부가 test split에 존재함을
  확인했다.

---

## 10. 다음 단계

1. 마스크 확보 또는 재현(공개된 bbox로 SAM2 파이프라인 실행) 후 masked 조건 재측정.
2. `max` 풀링 추가 → 논문 DINOv2-S 최고 구성과 직접 비교.
3. 프레임 기반 majority-vote 프로토콜 구현 → 논문 frame-based 행과 비교.
4. 공식 5-fold에서 동일 베이스라인 측정 → P0 MegaDescriptor와 같은 표에 놓기
   (**이미지 단위 수치와 비디오 단위 수치를 같은 표에 넣지 않는다**).
5. RoVF split의 단일 분할 한계를 보완할 반복 분할 민감도 측정.
