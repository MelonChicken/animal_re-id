# EXP-005: Temporal Consistency 음의 Δ 및 Mask Quality Audit

| 항목 | 값 |
|---|---|
| Report ID | `EXP-005` |
| Experiment ID | 기존 run `20261124T120950Z-d2ad62`의 audit `20261124T121830Z-c8fef7` |
| 작성일 / 실행일 | 2026-11-24 / 2026-11-24 |
| 상태 | Partial — AI visual screening 완료, 독립 GT 검증 미완료 |
| 작성자 | Codex |
| 사람 검토 | Not verified |
| 관련 문서 | `docs/benchmark_protocols.md`, `notebooks/temporal_consistency_failure_audit.ipynb` |
| Git Commit | audit base `b79f957`, dirty; 원 feature 추출 base `c4143e9` |

## 1. Overview



### 1.1 Objective

음의 Δ 18개 sequence를 우선 검토하고, target 움직임·crop 내부 배경·생성 mask 오류를 구분한다. 기존 feature에서 계산한 metric CSV를 분석하며 모델 추론/학습은 재실행하지 않는다.

### 1.2 Research Question

음의 Δ를 큰 동물 움직임으로 설명할 수 있는가? 일부 frame의 생성 mask가 Target/Non-target 비교를 지지하는가? uncropped 원본 영상이 확보되어 있는가?

### 1.3 Hypothesis

이번 failure audit에 사전 등록된 가설 없음. 음의 Δ를 확인한 뒤 수행한 사후 탐색 분석이며 원인 추정과 측정 결과를 구분한다.

### 1.4 핵심 결과 요약

음의 Δ 18개 전부와 양의 Δ 대조군 3개, 영역 결측 사례 2개를 검토했다. 23 clip/184개 sampled frame의 contact sheet를 살펴보았고, 최종 QC 선택은 104 frame이다. 이는 AI 시각 screening이며 independent human GT 검증은 **Not verified**이다.

선택한 fur pixel 7개 중 생성 mask가 target을 누락한 pixel은 5개였다. 오류를 의심한 위치를 고른 표본이므로 5/7을 데이터셋 오류율로 해석하지 않는다. Δ 양수인 대조군에도 몸통 누락이 있다. 원본 배포 ZIP에는 JPG/CSV만 있고 `data/` 아래 uncropped video는 확인되지 않았다.

## 2. Experimental Setup



### 2.1 Dataset 및 데이터 버전

PolarBearVidID Zenodo v1, 기존 3개체 Test 분석의 identity 1/2/11과 62 clip/496 frame을 기준으로 한다. 이번 직접 시각 검토는 23 clip/184 frame이다. 로컬 ZIP은 JPG 138,363개/CSV 7개이며 sampled 원본 JPG는 width 256 × height 128, 분석 H5는 224×224 RGB다.

[공식 배포](https://zenodo.org/records/7564529)는 하나의 ZIP을 제공하고, [데이터셋 논문](https://www.mdpi.com/2076-2615/13/5/801)은 검출/추적한 개체를 원 영상에서 crop하는 과정을 설명한다. 배포 crop은 enclosure 전체 영상이 아니다.

### 2.2 Train / Validation / Test Split

RoVF converter의 identity 내 70/10/20 Test 분할을 재사용했다. 공식 5-fold와 다르다. 실패 18개는 raw Δ<0으로 모두 선택, 양수 대조군은 identity마다 1개를 seed=20261124로 선택했다(`001_087`, `002_107`, `011_088`). 결측 사례는 `001_095`, `001_096`이다. 새 train/test split은 만들지 않았다. 원본 manifest SHA256은 audit metadata의 sources에 기록했다.

### 2.3 Query / Gallery Protocol

해당 없음: retrieval/query/gallery를 구성하지 않았다. 동일 격자 및 matched-support patch cosine을 진단한 실험의 결과를 분석한다. Rank-1/Rank-3/mAP는 Not available (실행하지 않음).

### 2.4 Evaluation Unit

pair → sequence → identity. 그룹 비교 표는 sequence별 수치의 중앙값이다. 각 영역 C는 두 영역 모두 유효한 동일 pair 집합에서 계산했다. pixel 진단은 독립적인 선택 지점 확인이며 전체 segmentation IoU가 아니다.

### 2.5 Model 및 Pretrained Checkpoint

기존 frozen `facebook/dinov2-small` patch feature, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. DINOv2-S/14, patch 차원 384. 파라미터 수는 Not verified. CLS는 본 영역 분석에 사용하지 않는다. 이번 audit에서 backbone은 로드하지 않았다.

### 2.6 Preprocessing / Frame Sampling

기존 저장 frame index `[0,2,5,8,10,13,16,19]`, unmasked 224×224 RGB/ImageNet normalization, L2-normalized patch token을 재사용했다. QC 선택은 첫/중간/끝(t=0/4/7), 최소 raw Δ pair 양 끝, 영역 결측 pair 양 끝의 합집합이다. t는 0-based 8-frame index이다. 저장 frame 0/10/19는 mask generator prompt 위치와 겹쳐 prompt가 과대표집된다. contact sheet는 모든 8개 시점을 포함한다.

### 2.7 Hyperparameters / Seed

추론 batch size 8/seed 20261124는 기존 추출 기록이다. 이번 audit는 학습률/optimizer/epoch 해당 없음. threshold 0.9의 raw Δ로 실패를 선정했다. pixel MAD는 RGB 절대차를 채널 평균하고 255로 나눈 뒤 mask intersection 영역에서 평균한다. optical flow/pose는 도입하지 않았다.

### 2.8 Hardware / Software Environment

기존 추출: Windows, Python 3.12.10, RTX 4060 Laptop GPU, torch 2.11.0+cu128, transformers 4.46.3, numpy 2.5.4/pandas 3.1.1/h5py 3.16.0/matplotlib 3.11.2 (기존 metadata). 이번 audit는 CPU로 기존 RoVF interpreter 및 프로젝트 interpreter에서 CSV/그림을 처리했다. audit peak memory/runtime은 Not verified.

## 3. Implementation & Validation



### 3.1 주요 구현 사항

`scripts/audit_temporal_consistency.py`는 입력을 읽기만 하고 고유 audit 디렉터리에 접촉 시트, pair별 RGB MAD, 시퀀스별 진단, QC 선택과 provenance를 저장한다. 독립 GT를 자동 생성하거나 마스크를 수정하지 않는다. 재실행 시 시각 관찰 필드는 pending으로 시작한다. 시각 판정과 fur pixel 검토는 이번 audit에만 수동으로 기록했다.

### 3.2 관련 코드

| 경로 | 역할 |
|---|---|
| `notebooks/temporal_consistency_pilot.ipynb` | 원 추출·지표 정의 |
| `scripts/audit_temporal_consistency.py` | 모델 추론 없는 audit 산출물 생성 |
| `notebooks/temporal_consistency_failure_audit.ipynb` | 저장된 audit 결과 확인 |
| `experiments/reference/rovf/segmentation/gen_pb_masks.py` | DINOv2/LDA/SAM2 생성 mask 출처; prompt `[0,10,19]` |

### 3.3 테스트 결과

이번 audit의 검증 명령과 실제 결과는 `validation.json`에 기록한다.

| 실제 실행 명령 | 실행일 | 결과 |
|---|---|---|
| `experiments/reference/rovf/.venv/Scripts/python.exe scripts/audit_temporal_consistency.py` | 2026-11-24 | 23 clip contact sheet/진단 생성 완료 |
| `.venv/Scripts/python.exe scripts/run_temporal_consistency_notebook.py --notebook notebooks/temporal_consistency_failure_audit.ipynb` | 2026-11-24 | 4개 코드 셀 오류 없이 완료 |
| `.venv/Scripts/python.exe experiments/verify_temporal_failure_audit.py` | 2026-11-24 | 아래 무결성 검사 수행; 상세 결과는 validation.json |
| `.venv/Scripts/python.exe experiments/verify_fresh_temporal_audit.py` | 2026-11-24 | 새 audit의 지표/7개 probe/104개 QC 선택 재현 및 검토 상태 pending 보존 확인 |

 source CSV SHA256 일치, 음의 Δ 집합 18개 일치, 23개 contact sheet 존재, 184개 frame/QC 표 키 유일성, paired C 차이와 Δ 일치, notebook schema/code syntax를 확인한다. 전체 pytest는 실행하지 않았다. 원 추출 테스트를 이번 테스트로 다시 주장하지 않는다.

### 3.4 데이터 무결성 및 누출 검사

원 62 clip manifest를 재사용하고 audit 입력 checksum을 보관한다. 원본 experiment의 mask/feature/metrics/기존 mask_qc.csv는 수정하지 않는다. failure-selected subset과 결과를 본 뒤의 pixel 선택에는 선택 편향이 있다. 마스크 생성기의 DINOv2/LDA와 SAM2 propagation은 독립 annotation이 아니며 feature consistency와 시간 smoothing의 순환성이 남는다.

### 3.5 평가 지표 검산

각 sequence에서 양 영역 모두 유효한 pair의 target/non-target C를 평균한 차이가 기존 sequence Δ와 일치함을 확인했다(max abs diff <1e-15). pixel MAD는 cosine과 다른 진단이며 두 지표의 대소관계를 같은 의미로 해석하지 않는다. Rank-1/Rank-3/mAP 검산은 해당 없음.

## 4. Results



### 4.1 주요 지표

| 그룹 | 시퀀스 수 | Target C 중앙값 | Non-target C 중앙값 | Motion proxy 중앙값 | Shape proxy 중앙값 |
|---|---:|---:|---:|---:|---:|
| 음의 Δ | 18 | 0.743149 | 0.818437 | 0.001754 | 0.223151 |
| 양의 Δ | 44 | 0.726434 | 0.650890 | 0.002128 | 0.327786 |

음의 그룹의 motion/shape proxy 중앙값은 양의 그룹보다 크지 않다. 음의 그룹은 Non-target C 중앙값이 더 높고 Target C 중앙값도 더 높다. 이는 사후 그룹 기술 통계이며 인과 검정이 아니다. seed 또는 identity 간 불확실성을 새로 추정하지 않았다.

### 4.2 Baseline 비교

대조군 3개는 양의 Δ 사례에서 고른 시각 QC 비교용이며 독립 baseline 성능 비교가 아니다. `002_107`에는 큰 몸통 누락이 있고 `001_087`, `011_088`에도 경계 문제가 보인다. Mask 오류가 음의 Δ에서만 나타난다고 주장할 수 없다.

### 4.3 논문 결과와의 비교 가능성

TCViT의 attention이나 Re-ID 성능을 재현하지 않는다. PolarBearVidID zoo crop과 camera-trap uncropped 영상은 다른 관측 조건이다. 논문 retrieval 수치를 본 audit의 cosine과 직접 비교하지 않는다.

### 4.4 실행 시간 / GPU 사용량 / Cache

원 cache `experiments/polarbear/temporal_consistency_identity_test/20261124T120950Z-d2ad62/patch_features.npz`에서 계산한 metric CSV를 재사용했다. NPZ의 feature 재계산이나 모델 추출/학습 횟수는 이번 audit에서 0이다. audit runtime/GPU memory는 Not verified. 결과는 `experiments/polarbear/temporal_consistency_identity_test/20261124T120950Z-d2ad62/audits/20261124T121830Z-c8fef7/`에 저장했다.

### 4.5 반복 실험 통계

반복 추출 없음. 이번 작업은 고정된 실패 subset에 대한 사후 diagnostic audit 한 번이다. 반복 수·seed 분산을 측정하지 않았다.

## 5. Analysis



### 5.1 확인된 실험 결과

`001_092`, `001_103`, `002_096`에는 큰 몸 전체 움직임이 관찰되지 않지만 Δ는 음수다. `002_101`, `002_103`, `011_084`, `011_096` 등은 보행·방향 변화 또는 crop 이탈을 보인다. `002_103`의 물은 동적이므로 모든 Non-target을 정적 배경으로 간주할 수 없다. 개별 관찰은 아래 표와 contact sheet에 대응한다.

### 5.2 가능한 원인과 근거

추정: 정적인 rock/foliage 등 높은 consistency의 Non-target, 동물의 자세·조명/그림자 변화, crop 변환, mask support 선택 오류가 함께 작용할 수 있다. 하나로 원인을 배정할 수 없다. `001_092`에서는 RGB MAD는 target이 더 작지만 feature C는 Non-target이 더 높아 pixel 변화량과 representation consistency가 다름을 보인다.

저장 frame 0/10/19의 거친 경계/직사각형 누락이 반복된다. 이 위치가 generation prompt와 같다는 사실은 코드로 확인했지만, prompt 처리의 인과 효과는 Not verified이다. GT가 없어 post-hoc mask 수정이나 corrected Δ는 만들지 않았다.

### 5.3 가설의 지지 여부

| 진술 | 판단 |
|---|---|
| 음의 Δ는 모두 큰 동물 움직임 때문이다 | 이 표본의 관찰 및 motion proxy로 지지되지 않음 |
| 생성 mask면 Target/Non-target 비교가 검증된다 | 판단 불가: 명백한 누락과 독립 GT 부재 |
| uncropped camera-trap 배경/occluder 가정을 검증했다 | 판단 불가: zoo crop과 분리되지 않은 Non-target만 있음 |
| 마스크 오류가 음의 Δ의 유일한 원인이다 | 판단 불가: 양의 대조군에도 오류가 있음 |

### 5.4 Failure Cases

| Sequence | Raw Δ | Centroid motion percentile* | 관찰된 동작 | Mask 관찰 (AI screening) |
|---|---:|---:|---|---|
| `001_092` | -0.007217 | 4.8 | 몸 전체 변화가 작음 | 다리/외곽 일부 누락; 큰 몸통 구멍은 뚜렷하지 않음 |
| `001_094` | -0.000227 | 38.7 | 보행·머리 회전 | 저장 0/10/19의 블록형 경계; 정확한 윤곽은 미검증 |
| `001_097` | -0.026898 | 22.6 | 머리 숙임·크기 변화 | 저장 19의 아래쪽 머리/다리 범위 누락 의심 |
| `001_103` | -0.038209 | 3.2 | 몸 전체 변화가 작음 | 아래 윤곽에서 돌 포함/다리 누락 가능성 |
| `002_087` | -0.005067 | 66.1 | 몸 회전·머리 숙임 | 저장 0/10/19 몸통 누락; 저장 0의 fur pixel 누락 확인 |
| `002_093` | -0.055944 | 25.8 | 몸 회전·머리 숙임 | 저장 10의 직사각형 몸통 누락 |
| `002_094` | -0.044494 | 37.1 | 머리 회전·작은 몸통 이동 | 저장 10의 다리/몸통 누락과 윤곽 침식 |
| `002_096` | -0.020307 | 6.5 | 몸 전체 변화가 작음 | 저장 0/10/19의 좁고 거친 다리/몸통 경계 |
| `002_097` | -0.003338 | 27.4 | 머리 회전·작은 몸통 이동 | 저장 0/10/19의 발/다리 경계 누락 |
| `002_101` | -0.069685 | 74.2 | 보행·머리 회전·조명 변화 | 저장 10의 큰 몸통/다리 누락, 저장 19 경계 잘림 |
| `002_103` | -0.017329 | 80.6 | 물가를 따라 멀어짐 | 저장 8/10 부근 머리/외곽 누락; 물은 정적 배경이 아님 |
| `002_104` | -0.001402 | 71.0 | 몸 회전·머리 들기 | 저장 0/10/19 몸통 내부 누락; 저장 0 fur probe 확인 |
| `002_105` | -0.051724 | 9.7 | 머리 회전·작은 몸통 이동 | 저장 10의 등 내부 구멍, 저장 0/19 경계 침식 |
| `011_084` | -0.032659 | 46.8 | 멀어지는 보행·회전 | 저장 10의 등 구멍; fur probe 누락 확인 |
| `011_087` | -0.036292 | 35.5 | 몸 회전·보행 | prompt 위치 다리/경계 불일치; 큰 몸통 구멍은 뚜렷하지 않음 |
| `011_093` | -0.079872 | 14.5 | 보행·회전·그림자 변화 | 저장 10 옆구리 누락; 저해상도/blur로 GT 판단 제한 |
| `011_096` | -0.078037 | 53.2 | 보행·crop 밖 이탈 | 저장 19에서 동물이 오른쪽 crop 경계에 잘림; mask 경계도 거침 |
| `011_103` | -0.074652 | 41.9 | 보행·회전·그림자 변화 | 저장 0/10/19 상하 윤곽이 거침; blur로 정밀 판단 제한 |

*62개 Test sequence 내 순위 백분율이며 물리적 속도 또는 성공 기준이 아니다. Δ는 소수점 6자리 반올림. 관찰 용어와 상세 frame 근거는 `sequence_review.csv`/`frame_qc.csv`에 있다.

### 5.5 예상과 다른 결과 및 부정적 결과

가장 큰 양의 Δ 사례 `001_096`에서도 stored 8 mask가 심하게 파편화되고 머리/앞몸통을 놓친다. 반면 `001_095`는 foreground log에 의한 실제 가림과 common-support 축소가 함께 보여 결측을 모두 segmentation 실패로 볼 수 없다. Fur pixel probes는 7개 중 5개 누락/2개 일치이며, 오류가 의심되는 영역을 고른 검증이라 정확도 추정에 사용할 수 없다. frame의 mask=0이 양 시점 모두의 pure Non-target patch가 됐다는 뜻도 아니다. 한쪽에서만 누락되면 patch exclusion/coverage 선택에 영향을 준다.

## 6. Limitations



### 6.1 Dataset Leakage

Test 결과로 실패 subset과 후속 QC 위치를 선택한 탐색 단계이다. 이 subset에서 방법을 조정한 뒤 같은 Test 결과를 독립 validation으로 보고하면 안 된다.

### 6.2 Protocol Mismatch

원본 전체 영상·camera coordinate가 아닌 tracklet crop 및 square resize 좌표를 본다. crop 내부 landmark가 안정적이어도 실제 카메라가 고정됐다고 확정하지 않는다. occluder label은 분리되어 있지 않다.

### 6.3 사전학습 데이터 노출

DINOv2 pretraining overlap은 Not verified. 생성 mask의 DINOv2 의존성과 SAM2 temporal propagation은 독립 GT를 대체하지 않는다.

### 6.4 통계적 불확실성

3개체 및 결과로 선택된 실패 subset이다. 부호/central tendency 기술 통계만 보고하며 p-value, generalization 또는 데이터셋 전체 mask error rate를 추정하지 않는다.

### 6.5 재현성 한계

audit base Git `b79f957`, dirty 작업 트리에서 수행. 원 feature 추출 metadata의 Git은 `c4143e9`이며 이번 audit commit과 다르다. source CSV hash 및 각 pixel probe 좌표는 보관했다. AI 시각 screening은 사람 검토와 같지 않다.

### 6.6 확인되지 않은 조건

Independent human segmentation GT, GT IoU/Dice, physical speed, full-frame camera motion, 별도 occluder metrics, 원본 uncropped 공개 다운로드는 Not verified/Not available. 해당 없는 값을 0으로 채우지 않는다.

## 7. Conclusion & Next Steps



### 7.1 결론

이번 crop/생성 mask 결과만으로 TCA 가정을 확정하거나 큰 움직임만을 실패 원인으로 볼 수 없다. 우선 torso 누락·파편화와 crop 이탈을 포함한 작은 독립 mask subset을 검증해야 한다. 현재 배포와 로컬 자료에서는 uncropped camera 영상을 확보하지 못했다. 확인한 공식 자료에 원본 다운로드 링크가 없다는 것이 원본 자체가 존재하지 않는다는 뜻은 아니다.

### 7.2 후속 실험

| 순서 | 작업 | 선행 조건 |
|---|---|---|
| 1 | 5쌍·10 frame의 독립 target mask 검토 | 사람 또는 별도 GT; prompt/propagated frame 모두 포함 |
| 2 | 고정 mask/지원 집합에서 Δ 민감도 재측정 | 수정 기준 사전 고정; 기존 Test 결과와 분리 |
| 3 | 원본 영상 제공 가능성 확인 | 전체 영상/프레임, camera ID/시간, crop-to-original box/transform |
| 4 | full-background 및 occluder 가설 검증 | uncropped 환경과 독립 target/occluder 구분; camera-trap 여부 확인 |

권장하는 최소 독립 mask 표본은 `002_104` 저장 index 0/2, `002_087` 0/2, `011_084` 8/10, `001_096` 5/8, `002_107` 0/2다. 음의 Δ 3 clip과 양의 Δ 2 clip에서 오류 프레임과 인접 프레임을 포함한다. 이 표본은 아직 독립 annotation하지 않았으며, 향후 pair 수준의 mask 민감도만 확인할 수 있다. 18개 전체 시퀀스의 교정된 Δ나 일반적인 mask 정확도를 추정할 수 없다.

저자에게 문의를 보내거나 대용량 원본을 다운로드하지 않았다. 해당 후속 행동은 아직 수행하지 않은 작업이다.

### 7.3 남은 연구 질문

독립 mask에서도 음의 Δ가 유지되는가? crop/조명 변화와 실제 non-rigid 변형을 분리할 수 있는가? zoo 원 영상에서도 관찰이 유지되는가? 별도 camera-trap 도메인과 occluder label이 확보되는가?

## Artifacts

모든 대용량 산출물은 `experiments/polarbear/temporal_consistency_identity_test/20261124T120950Z-d2ad62/audits/20261124T121830Z-c8fef7/` 아래에 있다.

| 파일 | 내용 |
|---|---|
| `sequence_review.csv` | 23개 시퀀스 관찰과 기존 지표 |
| `frame_qc.csv`, `qc_selection.csv` | 184 frame overview 및 104 frame 선택 |
| `mask_pixel_probes.csv`, `mask_pixel_probes.png` | 7개 fur pixel의 좌표와 mask 값 |
| `all_sequence_diagnostics.csv` | 62개 sequence의 paired C/Δ/motion |
| `negative_positive_group_medians.csv` | 사후 그룹 기술 통계 |
| `pair_pixel_diagnostics.csv` | mask intersection RGB MAD |
| `*_all_frames.png` | 23개 8-frame RGB/mask contact sheet |
| `audit_metadata.json` | 입력 hash, 선정/검토 범위 |
| `raw_video_availability.json` | 로컬/공식 배포 원본 확보 판단 |
| `validation.json` | 실제 검증 결과 |
| `proposed_independent_mask_subset.csv` | 다음 독립 검토용 5쌍·10 frame 경로; annotation은 아직 없음 |

Original metrics/cache/manifest는 상위 source run에서 재사용했으며 수정하지 않았다.

재생성 검증용 audit는 같은 상위 디렉터리의 `20261124T123055Z-f9da37/`에 있다. 수치와 mask probe 값이 일치하지만 수동 시각 판정은 복사하지 않아 pending 상태다. 이는 새 모델 실험이나 추가 mask 검증 결과가 아니다.

확인 출처: [Zenodo 공식 release](https://zenodo.org/records/7564529), [Zuerl et al. 데이터셋 논문](https://www.mdpi.com/2076-2615/13/5/801). 확인일 2026-11-24.
