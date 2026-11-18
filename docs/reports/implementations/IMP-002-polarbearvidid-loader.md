# IMP-002 PolarBearVidID 데이터 로더와 검증 CLI

| 항목 | 값 |
|---|---|
| Report ID | `IMP-002` |
| Task | PolarBearVidID 로더 구현 및 공식 릴리스 검증 |
| 작성일 | 2026-11-18 |
| 상태 | Completed |
| 작성자 | Codex |
| 관련 Git Commit | `6dd705c` 기준 미커밋 작업 트리 |
| 관련 실험 | `EXP-001`에서 로더 사용 |

## 1. Overview

### 1.1 구현 목적

로컬 PolarBearVidID의 프레임 경로, 개체 라벨, 시퀀스 번호를 후속 Re-ID 실험에서 사용할 수 있도록 읽고, 원본 파일과 메타데이터의 일관성을 확인한다.

### 1.2 범위

| 구분 | 내용 |
|---|---|
| 포함 | 메타데이터 로딩, 파일 검사, 시퀀스별 프레임 수 대조, 표본 디코딩 |
| 제외 | 데이터 분할, 특징 추출, 모델 학습, 검색 성능 평가 |

## 2. Changes

### 2.1 변경 파일

| 상태 | 파일 | 변경 내용 |
|---|---|---|
| Added | `src/animal_re_id/data/__init__.py` | 데이터 패키지 초기화 |
| Added | `src/animal_re_id/data/wildlife.py` | 카탈로그 로더와 JSON 검증 CLI |

두 코드는 점검 시점에 Git에 커밋되지 않은 상태였다.

### 2.2 주요 구현 내용

`load_polar_bear_vid_id(root)`는 WildlifeDatasets 카탈로그의 개체 라벨과 시퀀스 번호를 유지하고, 이미지 경로를 절대 경로로 변환해 존재 여부를 검사한다. CLI는 개체별 프레임·시퀀스 수, 중복 경로, 누락 또는 잘못된 라벨을 집계한다. `track_info.csv`가 있으면 시작·끝 프레임을 포함한 구간 길이와 실제 프레임 수를 시퀀스별로 대조한다.

### 2.3 기존 코드와의 연결

`config/polarbear.yaml`의 데이터 경로는 `data/raw/PolarBearVidID/`다. 현재 `scripts/evaluate.py`와 `scripts/evaluate_video.py`에서 로더를 호출한다. 데이터 규약과 후속 실험은 [벤치마크 프로토콜](../../benchmark_protocols.md)과 [EXP-001](../experiments/EXP-001-megadescriptor.md)에 기록돼 있다.

## 3. Technical Decisions

### 3.1 아키텍처 및 알고리즘 선택

WildlifeDatasets가 제공하는 `video`를 유지한다. 123개 번호 중 120개가 여러 개체에서 반복되므로 시퀀스 집계에는 `(identity, video)`를 쓴다.

### 3.2 기존 라이브러리 활용 방식

로더 작업에서 새 의존성을 추가하지 않았다. 점검 환경에서 확인한 버전은 다음과 같다.

| 라이브러리 | 사용 목적 | 버전 |
|---|---|---|
| WildlifeDatasets | PolarBearVidID 카탈로그 생성 | 1.0.9 |
| pandas | 메타데이터 처리 및 집계 | 3.1.1 |
| Pillow | JPEG 표본 디코딩 | 13.0.0 |

### 3.3 대안 및 Trade-off

파일 경로와 트랙릿 수는 전체 검사하고 JPEG는 5장만 디코딩했다. 전체 이미지의 픽셀을 읽는 비용을 줄였으며 디코딩 결과는 표본에 한정된다.

## 4. Testing & Validation

### 4.1 실행한 테스트와 결과

2026-11-18에 `.venv/Scripts/python.exe -B -m animal_re_id.data.wildlife $datasetRoot --decode-samples 5`를 공식 ZIP의 임시 추출본에서 다시 실행했다. 종료 코드는 0이었다.

### 4.2 실제 데이터 검증 여부

| 검증 | 방법 | 결과 |
|---|---|---|
| 데이터 집계 | 카탈로그 및 복합 시퀀스 키 집계 | 138,363장, 13개체, 1,431개 시퀀스 |
| 경로와 라벨 | 전체 카탈로그 검사 | 누락 경로, 중복 경로, 잘못되거나 비어 있는 라벨, 누락 시퀀스 번호 각각 0건 |
| 트랙릿별 프레임 수 | `track_info.csv`와 대조 | 누락·초과 프레임 0장 |
| 이미지 파일 | 최초 검증에서 공식 ZIP의 MD5와 카탈로그 경로 대조 | 체크섬 일치, ZIP과 카탈로그 사이 경로 차이 0건 |
| JPEG 디코딩 | Pillow로 5장 읽기 | 5장 성공, 모두 256 × 128 |

공식 배포 자료는 [Zenodo 릴리스 7564529](https://zenodo.org/records/7564529)다.

### 4.3 확인된 버그 및 해결 상태

| 문제 | 영향 | 상태 |
|---|---|---|
| 최초 다운로드에서 Zenodo 호스트 이름 조회 실패 | 기본 실행 환경에서 ZIP 다운로드 중단 | 호스트 환경에서 범위 요청으로 내려받고 공식 MD5 확인 |
| Windows CP949 콘솔에서 독일어 지명 출력 중 오류 | ZIP의 메타데이터를 출력하는 임시 추출 명령 중단 | 출력 설정을 바꿔 추출·검증 완료 |
| 점검 세션의 `LOCALAPPDATA`가 샌드박스 임시 경로로 지정됨 | 앞서 안내한 환경 변수 기반 경로로 추출본 조회 실패 | 실제 추출본 경로를 지정해 CLI 재실행 성공 |

## 5. Limitations

### 5.1 구현 범위

검증 지표는 데이터와 메타데이터의 일관성을 대상으로 한다. 모델 성능 및 사전학습 데이터 중복 여부는 후속 실험 조건에 해당한다.

### 5.2 알려진 제약

`track_info.csv`가 없는 데이터 루트에서는 파일 검사와 카탈로그 집계가 실행되며, CLI의 `track_manifest_checked`는 `false`다.

### 5.3 검증되지 않은 동작

전체 JPEG 디코딩과 현재 작업 트리의 전체 테스트 스위트는 이번 점검에서 실행하지 않았다.

## 6. Next Steps

| 구분 | 후속 작업 |
|---|---|
| 구현 | 해당 없음 |
| 테스트 | 해당 없음 |
