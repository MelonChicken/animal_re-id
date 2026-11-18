# PolarBearVidID 벤치마크 프로토콜 검증

이 문서는 Frozen MegaDescriptor(P0) 베이스라인을 어떤 조건에서 평가했는지, 그 조건이
공식 프로토콜과 어떻게 같고 다른지를 기록한다. **논문 재현과 자체 실험을 구분**하는 것이
목적이며, 확인하지 못한 것은 확인하지 못했다고 적는다.

- 검증 기준 릴리스: Zenodo `7564529` / `PolarBearVidID.zip` (1,573,948,301 bytes)
- 관련 코드: `src/animal_re_id/data/splits.py`, `scripts/evaluate.py`
- 결과 파일: `experiments/results/*.json`, `experiments/results/*_per_query.csv`

---

## 1. 데이터셋 검증 결과 (실측)

`uv run python src/animal_re_id/data/wildlife.py data/raw/PolarBearVidID` 로 확인한 값이다.

| 항목 | 값 |
|---|---|
| 프레임 수 | 138,363 |
| 트랙릿(=시퀀스) 수 | 1,431 |
| 개체 수 | 13 |
| 동물원 수 | 6 |
| 해상도 | 256 × 128 |
| 카메라 | 3대 (`C01`, `C02`, `C03`) |
| 파일명 규칙 | `{id:03d}/{id:03d}C{cam:02d}T{tracklet:03d}F{frame:03d}.jpg` |
| 프레임/트랙릿 번호 | **0부터 시작** (예: `F000`) |
| 시퀀스 길이 | 평균 96.7, 중앙값 100, 최소 28, 최대 100 |

### 1.1 시퀀스 키는 반드시 `(identity, video)` 복합 키

WildlifeDatasets 로더에서 `video = file[7:10]`, 즉 3자리 트랙릿 번호다.
트랙릿 번호는 개체별로만 유일하고 개체 간에 반복된다(예: 13개 개체 모두 `video=0`을 갖는다).
따라서 `video` 단독 키는 사용할 수 없다. 코드에서는 `sequence = "{identity}|{video}"`로
직렬화하고, `(identity, video)` 쌍이 유일함을 `check_assignments()`에서 검사한다.

추가 확인: `(identity, tracklet)` 하나에 카메라가 정확히 1대이므로, fold 파일의
`(id, cam, tracklet)`을 `(identity, video)`로 안전하게 축약할 수 있다.

### 1.2 동물원별 개체 분포

| 동물원 | 개체 | 이름 |
|---|---|---|
| Tiergarten Nürnberg | 2 | Vera, Nanuq |
| Tierpark Berlin | 2 | Tonja, Hertha |
| Tiergarten Schönbrunn | 2 | Nora, Finja |
| Tierpark Neumünster | 2 | Larissa, Vitus |
| Zoo Karlsruhe | 2 | Lloyd, Charlotte |
| Zoo Hannover | 3 | Nana, Milana, Sprinter |

- 동일 개체가 여러 동물원에 존재하지 않는다(개체 ↔ 동물원 1:1).
- 동물원당 개체가 2~3마리뿐이다.

### 1.3 개체별 시퀀스/프레임 수

| ID | 이름 | 시퀀스 | 프레임 |
|---|---|---|---|
| 0 | Vera | 123 | 11,604 |
| 1 | Nanuq | 105 | 10,220 |
| 2 | Tonja | 108 | 10,365 |
| 3 | Hertha | 118 | 11,275 |
| 4 | Nora | 104 | 10,243 |
| 5 | Finja | 107 | 10,368 |
| 6 | Larissa | 107 | 10,546 |
| 7 | Vitus | 104 | 10,272 |
| 8 | Lloyd | 120 | 11,352 |
| 9 | Charlotte | 112 | 11,005 |
| 10 | Nana | 119 | 11,210 |
| 11 | Milana | 104 | 10,056 |
| 12 | Sprinter | 100 | 9,847 |
| | **합계** | **1,431** | **138,363** |

---

## 2. 프로토콜 A — PolarBearVidID 공식 5-fold (주 프로토콜)

### 2.1 공식 fold 배정이 공개되어 있다

당초 "논문은 Table A1에 개체별 fold 개수만 공개하고 실제 시퀀스 배정은 비공개"라고
판단했으나, **이는 틀렸다.** 릴리스 압축 파일 안에 다음 파일이 포함되어 있다.

```
track_fold_info_1.csv   (287행)
track_fold_info_2.csv   (286행)
track_fold_info_3.csv   (286행)
track_fold_info_4.csv   (286행)
track_fold_info_5.csv   (286행)
```

각 파일은 `id, cam, tracklet, start, end` 컬럼을 가지며, 5개 파일의 합집합은
`track_info.csv`의 1,431 트랙릿과 정확히 일치하고 중복이 없다. 프레임 합도 138,363으로
전체와 일치한다.

결정적으로, 이 파일들로 계산한 개체별 fold 분포가 **논문 Table A1과 완전히 일치**한다.

| ID | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 |
|---|---|---|---|---|---|
| 0 Vera | 25 | 25 | 25 | 24 | 24 |
| 1 Nanuq | 21 | 21 | 21 | 21 | 21 |
| 2 Tonja | 22 | 21 | 21 | 22 | 22 |
| 3 Hertha | 23 | 24 | 24 | 24 | 23 |
| … | … | … | … | … | … |
| 12 Sprinter | 20 | 20 | 20 | 20 | 20 |

따라서 본 프로젝트의 `five_fold` 프로토콜은 **재생성(regenerated)이 아니라 릴리스에 포함된
공식 배정을 그대로 읽는다.** 남는 가정은 "`track_fold_info_*.csv`가 논문 Table 3의 fold
배정과 동일하다"는 것 하나이며, Table A1과의 정확한 일치가 그 근거다.

### 2.2 평가 단위

| 항목 | 값 |
|---|---|
| 평가 단위 | **이미지** (시퀀스당 대표 프레임 1장) |
| 프레임 선택 | 시퀀스 중앙값에 가장 가까운 프레임, 동률이면 낮은 번호 |
| Query | 해당 fold의 대표 프레임 전부 (287 또는 286장) |
| Gallery | **기본값: 같은 fold의 대표 프레임 전부** (leave-one-out) |
| Self-match | `path`가 동일한 항목을 순위에서 제외 |
| Positive | 같은 identity의 다른 시퀀스 (fold 내 평균 22개) |
| Negative | 다른 identity 전부 |
| Closed-set | query의 모든 identity가 gallery에 존재해야 함 (검사함) |
| 거리 | cosine similarity |
| 지표 | Rank-1, Rank-3, mAP |
| 집계 | fold별 계산 후 평균 ± 표준편차 (5 fold) |

선택한 대표 프레임의 상대 위치는 평균 0.495 (표준편차 0.002)로, 거의 정확히 중앙이다.

### 2.3 mAP 정의

query마다 self-match를 제거한 뒤 유사도 내림차순으로 정렬하고,

```
AP = (1 / #positives) * Σ_{k: rank k가 positive} (#{positive at rank ≤ k} / k)
```

동률(tie)은 gallery 순서로 결정론적으로 깨진다. 5 fold 평균을 mAP로 보고한다.

### 2.4 Gallery 범위의 모호성 — 두 변형을 모두 보고

논문은 image-level 평가에서 gallery를 "테스트 fold로 제한"했는지 "전체 데이터"로 두었는지
명시하지 않는다. 임의로 추정하지 않기 위해 두 변형을 모두 실행했다.
`evaluation.gallery_scope`로 선택한다.

| 변형 | Rank-1 | Rank-3 | mAP |
|---|---|---|---|
| `fold` (기본, 테스트 fold 내 leave-one-out) | 0.7771 ± 0.0376 | 0.8826 ± 0.0332 | **0.3969 ± 0.0299** |
| `all` (전체 1,431장 기준) | 0.7827 ± 0.0219 | 0.8973 ± 0.0247 | 0.2715 ± 0.0160 |

Rank-1은 거의 같지만 mAP는 크게 다르다. positive 수가 22개 → 122개로 늘면 상위 순위의
precision이 희석되기 때문이다. 논문의 image-level mAP(0.454)에 근접한 쪽은 `fold`이며,
이것이 논문이 테스트 fold 내에서 평가했을 가능성을 뒷받침하는 **간접 증거**다.
단정하지 않고 기본값 `fold`를 쓰되 두 수치를 모두 공개한다.

### 2.5 논문과의 차이

| 항목 | 논문 (Zuerl et al., 2023) | 본 작업 |
|---|---|---|
| Fold 배정 | 공개 fold 파일 | **동일** (릴리스 파일 그대로) |
| 백본 | 논문 자체 image-based 모델 | **Frozen MegaDescriptor-L-384** (학습 없음) |
| 학습 | 5-fold CV 구조상 4 fold 학습 / 1 fold 평가 | **학습하지 않음** (frozen) |
| 대표 프레임 | 이미지 단위 사용, 선택 규칙 미공개 | 중앙 프레임 1장 (문서화된 규칙) |
| Gallery 범위 | 미공개 | `fold` / `all` 두 변형 모두 보고 |
| 지표 | Rank-1, mAP | Rank-1, Rank-3, mAP |
| 표준편차 | 5 fold 표준편차 | 동일 |

즉 **프로토콜(분할·평가 단위·지표)은 논문과 일치시키고, 모델만 바꾼 비교**다.
논문 수치와 직접 비교하려면 백본 차이를 감안해야 한다. 본 작업은 논문 수치를 재현했다고
주장하지 않는다.

### 2.6 이미지 단위 정의의 남은 모호성

"image-level"은 두 가지로 해석될 수 있다.

1. 시퀀스당 대표 프레임 1장 → **본 작업이 채택**
2. 모든 프레임(138,363장)을 각각 query/gallery로 사용

(2)는 같은 시퀀스의 인접 프레임이 near-duplicate positive가 되어 지표가 크게 올라간다.
논문의 mAP 0.454는 (2)에서는 나오기 어려운 수준이므로 (1)이 합리적이라고 판단했지만,
**확인한 것은 아니다.** (2)는 실행하지 않았다.

### 2.7 실행되지 않은 변형

- 전체 프레임(138,363장) 이미지 평가 — 실행하지 않음 (계산량 및 위 모호성)
- Zoo-disjoint 평가 — 아래 4절 참조, 주 프로토콜로 부적합

---

## 3. 프로토콜 B — RoVF 70/10/20 tracklet split

공식 저장소 `Strong-AI-Lab/RoVF-Meerkat-Reidentification`의
`Dataset/convert_polar_bear_dataset.py`를 직접 읽고 확인했다.

### 3.1 원본 로직

- `track_info.csv`를 **파일 순서대로** 순회하며 개체별 카운터를 증가시킨다.
- `pos >= n*0.8` → test, `pos >= n*0.7` → val, 그 외 → train (개체별).
- 클립당 20 프레임을 균등 샘플링하고 224×224로 저장, 채널을 BGR→RGB로 뒤집는다.
- Query/Gallery는 **비디오(클립) 단위**이며, co-occurrence(같은 시점에 등장한 다른 개체)를
  negative로 쓰는 별도 구성이 있다.

### 3.2 본 작업의 재현

`rovf_assignment()`는 `track_info.csv`의 **행 순서를 그대로 사용**한다(정렬을 재유도하지
않음). 공개 파일은 `(id, tracklet)` 오름차순이지만, 순서를 다시 계산하면 파일이 재정렬될 때
분할이 조용히 달라질 수 있기 때문이다. 카탈로그와 `track_info.csv`의 시퀀스 집합이
일치하는지도 검사한다.

실측 분할 (1,431 시퀀스):

| split | 시퀀스 |
|---|---|
| train | 1,006 |
| val | 145 |
| test | 280 |

### 3.3 본 작업이 재현하지 **않은** 부분

- 20 프레임 클립 구성과 224×224 리사이즈 (P0는 이미지 단위이므로)
- 비디오 단위 집계, co-occurrence negative 구성
- RoVF 학습 및 추론

따라서 **RoVF test 분할을 image-level 지표로 평가한 결과(3.4)를 RoVF 논문의 video-level
수치와 직접 비교하면 안 된다.** 분할만 공유하는 별도 실험이다.

### 3.4 결과 (참고용, n=280)

| Rank-1 | Rank-3 | mAP |
|---|---|---|
| 0.6607 | 0.8143 | 0.2908 |

단일 test split이므로 표준편차는 보고하지 않는다. 공식 fold보다 어려운데, 뒤 20% 트랙릿에
test가 몰려 있어 fold보다 조건이 다르기 때문으로 보이나 원인 분석은 하지 않았다.

---

## 4. 채택하지 않은 프로토콜

### 4.1 Zoo-disjoint (주 프로토콜로 부적합)

동물원이 6곳, 동물원당 개체가 2~3마리다. Zoo-disjoint로 나누면 zoo당 1~2개체만 남아
identity classification 기준 chance level이 1/2~1/3이 되고, retrieval negative도
사실상 사라진다. **평가 조건이 성립하지 않으므로 구현하지 않았다.**

대신 필요하면 zoo별 성능 분해(per-zoo breakdown)를 보고하는 편이 낫다. 다만 Zoo Hannover만
3개체이고 fold당 query 수가 적어 분산이 크다. 본 작업에서는 보고하지 않았다.

### 4.2 Identity-disjoint / Open-set

이 데이터셋은 13개체를 5-fold로 나누는 **closed-set** 설계다. Train/Test 개체를 분리하면
(test 개체를 gallery에서 제거) positive pair 자체가 없어져 retrieval이 성립하지 않는다.
WildlifeDatasets의 `DisjointSetSplit`/`OpenSetSplit`은 분류용이며, 본 retrieval 평가에는
적용하지 않는다.

---

## 5. 사전학습 데이터 누출 — Zero-shot이 아니다

이 항목은 결과 해석에 직접 영향을 준다.

- PolarBearVidID는 `wildlife-datasets` 컬렉션에 포함되어 있다.
- MegaDescriptor 계열은 **`wildlife-datasets` 컬렉션의 공개 재식별 데이터셋으로 사전학습**된
  모델이다(HF 모델 카드: "Superwisely pre-trained on animal re-identification datasets").
- PolarBearVidID는 WildlifeReID-10k의 구성 데이터이기도 하다(1,391 이미지 / 13 개체,
  100프레임 간격 샘플링).
- WildlifeReID-10k 논문은 이 중복 때문에 MegaDescriptor-L을 비교 대상에서 제외했다.

따라서:

- 본 결과는 **zero-shot이 아니다.**
- 학습 데이터와 평가 데이터의 개체가 겹칠 가능성이 높다(transductive/leaky 조건).
- **SOTA로 표기하지 않는다.** 다른 벤치마크 수치와 나란히 비교하지 않는다.
- 이 조건에서 측정하는 것은 "도메인 적응된 특징의 상한"에 가깝다.

체크포인트가 실제로 어떤 이미지를 봤는지는 공개된 목록이 없어 **개별 이미지 단위로는
검증할 수 없다.** 위 서술은 데이터셋 구성과 모델 카드에 근거한 것이며, 그 이상은 주장하지 않는다.

---

## 6. Fold 배정 민감도 (실측) — 공식 fold를 써야 하는 이유

공식 fold는 **하나의 고정된 추첨**이다. 그 추첨이 지표에 얼마나 영향을 주는지 측정했다
(`scripts/fold_sensitivity.py`). 방법은 공식 fold 라벨을 **개체 내에서 무작위로 치환**하는
것이다. 개체별 fold 크기는 정확히 보존되고, 저자가 사용한 묶음 구조만 파괴된다.

| 배정 | Rank-1 | Rank-3 | mAP |
|---|---|---|---|
| 공식 fold | **0.7771 ± 0.0376** | 0.8826 | 0.3969 |
| 치환 #0 | 0.6506 ± 0.0232 | 0.8134 | 0.2892 |
| 치환 #1 | 0.6107 ± 0.0319 | 0.7855 | 0.2814 |
| 치환 #2 | 0.6268 ± 0.0316 | 0.8015 | 0.2833 |
| 치환 #3 | 0.6108 ± 0.0171 | 0.7981 | 0.2811 |
| 치환 #4 | 0.6213 ± 0.0211 | 0.8036 | 0.2843 |
| 치환 평균 | 0.6240 ± 0.0164 | — | 0.2839 |

별도로 구현한 stratified 재추첨(seed 0/1/2)도 0.6307 / 0.6231 / 0.6256으로 같은 구간에
들어온다.

**관찰:**

- 격차는 +0.153 Rank-1로 매우 크고, 5개 fold 전부에서, 13개 개체 대부분에서 일관되게 나타난다.
- query당 positive 수는 공식 22.1개, 재추첨 22.3개로 사실상 같다. gallery 크기나 positive
  수의 차이로 설명되지 않는다.
- 따라서 **공식 fold는 임의의 균형 분할과 교환 가능하지 않다.** 저자는 트랙릿을 어떤
  기준으로 묶었고, 그 묶음이 같은 조건(같은 세션/촬영 환경)의 프레임을 한 fold에 모아
  retrieval을 더 쉽게 만든 것으로 보인다.
- 원인을 확정할 수는 없다. 릴리스 메타데이터(`track_info.csv`)에는 시각 정보가 없고,
  트랙릿 번호 순서와 fold는 무관해 보인다(연속 런 수가 무작위 추첨 기대값과 같다).

**결론:**

1. **공식 fold 파일을 반드시 사용한다.** 재생성한 5-fold 수치는 공식 수치보다 약 15점 낮게
   나오므로, "5-fold를 재현했다"고 쓰면 안 된다.
2. 공식 fold 기반 수치는 "같은 세션 내 재식별"에 가까운 조건이며, 다른 촬영 조건으로의
   일반화 성능을 과대평가할 수 있다. 이 점을 결과와 함께 밝힌다.
3. 이 격차 자체가 유용한 진단이다. 이후 모델 비교는 모두 동일한 공식 fold에서 해야 한다.

---

## 7. 논문이 보고한 수치 (참고용, 본 작업의 결과가 아님)

Zuerl et al., *Animals* 2023, Table 3 (5-fold 평균 ± 표준편차):

| 방법 | Rank-1 | mAP |
|---|---|---|
| Image-based | 0.853 ± 0.011 | 0.454 ± 0.024 |
| Video-based | 0.966 ± 0.016 | 0.882 ± 0.090 |

Table 4는 zoo별 video-based 결과를 보고한다(Nuremberg 0.991, Berlin 0.920, Vienna 0.948,
Neumünster 0.995, Karlsruhe 0.983, Hannover 0.972 Rank-1).

이 수치들은 **논문 자체 모델의 결과**이며 본 작업의 frozen MegaDescriptor 결과와 같지 않다.
본 작업은 이 수치를 재현했다고 주장하지 않는다.

---

## 8. 재현성 한계

1. **Fold 배정**: 공식 파일을 그대로 사용하므로 재현 가능. 단 6절의 이유로 "임의의 5-fold"와
   교환 불가.
2. **대표 프레임 선택**: 중앙 프레임은 문서화된 규칙이지만 논문의 규칙은 아니다. 프레임
   선택을 바꾸면 수치가 달라질 수 있으며, 이 민감도는 측정하지 않았다.
3. **Gallery 범위**: 논문 미공개. 두 변형을 모두 보고했으나, 어느 쪽이 논문과 같은지는
   확정하지 못했다.
4. **사전학습 누출**: 5절. 통제 불가.
5. **하드웨어 의존성**: CUDA 커널과 timm 버전에 따라 마지막 자리 수치가 달라질 수 있다.
   결과 JSON에 git commit, 패키지 버전, GPU, 체크포인트 ID, 전처리, split signature,
   feature cache key를 모두 기록한다.
6. **단일 이미지 대표성**: 시퀀스당 1프레임만 쓰므로 시퀀스 내 변동을 반영하지 못한다.
   이는 P0(이미지 베이스라인)의 의도된 한계이며, P3/P4에서 다룰 문제다.

---

## 9. 다음 단계에 필요한 작업

### P1 — DINOv2 (Frozen) — 비디오 베이스라인 구현 완료

- 구현: `models/dinov2.py`, `data/video.py`, `evaluation/video_retrieval.py`,
  `scripts/evaluate_video.py`, `config/dinov2_video.yaml`.
- 체크포인트는 `transformers`로 로드하는 `facebook/dinov2-small`(frozen)이며, 224 입력과
  ImageNet 정규화를 기본으로 하되 논문 충실용 `/255` 모드도 함께 지원한다.
- **계획과 달라진 점**: 당초 "같은 공식 fold, 같은 대표 프레임"에서 이미지 단위로 비교할
  예정이었으나, 실제로는 RoVF의 비디오 프로토콜(20프레임, 시퀀스 단위)로 구현했다.
  따라서 P0 MegaDescriptor의 이미지 단위 수치와 **같은 표에 넣지 않는다.**
- 공식 5-fold에서의 비디오 단위 DINOv2 측정은 아직 하지 않았다(다음 단계).
- 결과와 논문 비교 가능성: `docs/dinov2_video_baseline.md`.
- 누출 관점에서 DINOv2는 범용 사전학습이므로 MegaDescriptor보다 공정한 기준선이 된다.

### P4 — RoVF (Video)

- 3.3절의 미재현 부분 구현이 선행 조건: 20프레임 클립, 224×224, co-occurrence negative,
  비디오 단위 집계.
- 지표를 video 단위로 다시 정의하고, image 단위 수치와 **같은 표에 넣지 않는다.**
- RoVF test split은 `rovf_assignment()`를 그대로 재사용할 수 있다.

### 공통

- 프레임 선택 민감도(중앙 vs 첫/마지막 vs 다중 프레임 평균) 측정
- zoo별 성능 분해
- 전체 프레임 이미지 평가(2.6절 변형 2)는 계산 비용과 positive 구성 문제를 먼저 정리한 뒤 결정
