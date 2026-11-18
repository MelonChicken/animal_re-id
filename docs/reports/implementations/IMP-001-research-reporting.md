# IMP-001: Standardized Research Reporting System

| 항목 | 값 |
|---|---|
| Report ID | `IMP-001` |
| Task | 연구 보고서 템플릿 및 `AGENTS.md` 보고 규칙 추가 |
| 작성일 | 2026-11-18 |
| 상태 | `Completed` |
| 작성자 | OpenCode agent |
| 관련 Git Commit | `6dd705c6d66fcb4f2858a50676ab9b1b34895bc1` (`dirty`, 이번 작업은 미커밋) |
| 관련 실험 | `EXP-001` |

---

## 1. Overview

### 1.1 구현 목적

실험 결과와 코드 구현 내용을 매번 다른 형식으로 정리하면 수치·조건·한계가 누락되기
쉽다. 이 작업은 **Experiment Report**와 **Implementation Report**의 고정 템플릿을
추가하고, `AGENTS.md`에 트리거·증거 수집·작성 원칙·저장 규칙을 정의해 보고서
작성 절차를 표준화한다.

### 1.2 범위

| 구분 | 내용 |
|---|---|
| 포함 | `AGENTS.md` 보고 규칙, 두 Markdown 템플릿, `docs/reports/` 디렉터리, 예시 보고서 2개 |
| 제외 | 모델·학습·평가 코드 수정, 실험 재실행, 외부 로깅 서비스, OpenCode Skill, 자동 생성 스크립트/CI |

---

## 2. Changes

### 2.1 변경 파일

| 상태 | 파일 | 변경 내용 |
|---|---|---|
| `Added` | `AGENTS.md` | Reporting 트리거, 분류, 증거 수집, 원칙, 워크플로, 저장·명명 규칙, 검증 체크리스트 |
| `Added` | `docs/templates/experiment_report.md` | Experiment Report 7개 섹션 + Artifacts 템플릿 |
| `Added` | `docs/templates/implementation_report.md` | Implementation Report 6개 섹션 템플릿 |
| `Added` | `docs/reports/experiments/EXP-001-megadescriptor.md` | 기존 MegaDescriptor 실험 산출물 기반 예시 보고서 |
| `Added` | `docs/reports/implementations/IMP-001-research-reporting.md` | 이 작업의 Implementation Report (본 문서) |
| `Modified` | 없음 | 기존 코드·설정·실험 결과를 수정하지 않음 |
| `Deleted` | 없음 | — |

### 2.2 주요 구현 내용

- **템플릿**: 두 보고서 모두 메타데이터 표, 필수 섹션, 표 기반 증거 기록, `Not verified`
  / `Not available` 표기, 사실·해석 구분을 포함한다. 작성 지침은 HTML 주석으로 넣어
  렌더링된 보고서에는 보이지 않게 했다.
- **AGENTS.md**: "리포트 작성해줘" 등 요청을 Experiment/Implementation으로 분류하는
  규칙, 확인해야 할 7가지 증거, 과장·누출·프로토콜 비교 금지 원칙, 8단계 워크플로,
  `EXP-XXX` / `IMP-XXX` 명명 및 번호 충돌 방지 규칙을 정의했다.
- **예시 보고서**: `EXP-001`은 실제 결과 JSON/CSV만 사용해 작성했고, `IMP-001`은
  이번 작업 자체를 기록한다.

### 2.3 기존 코드와의 연결

```text
사용자 요청
  -> AGENTS.md (트리거/분류/절차)
  -> docs/templates/*.md (형식)
  -> 기존 산출물 읽기 (config, experiments/results, experiments/splits, src, tests)
  -> docs/reports/{experiments,implementations}/EXP-XXX|IMP-XXX-*.md
```

기존 `docs/benchmark_protocols.md`의 데이터·프로토콜 검증 내용을 재사용했으며,
새로운 런타임 코드나 의존성은 추가하지 않았다.

---

## 3. Technical Decisions

### 3.1 아키텍처 및 알고리즘 선택

문서 기반 규약을 선택했다. 에이전트가 항상 읽는 `AGENTS.md`에 절차를 두고, 템플릿은
`docs/templates/`에 분리해 사람이 직접 복사·작성할 수도 있게 했다. 보고서 저장소는
`docs/reports/`로 두어 Git 추적이 가능하게 했다(`experiments/`는 `.gitignore` 대상).

### 3.2 기존 라이브러리 활용 방식

새 라이브러리나 서비스를 추가하지 않았다. Markdown, Git, 기존 pytest만 사용한다.

| 라이브러리 | 사용 목적 | 버전 |
|---|---|---|
| Markdown | 템플릿·보고서 형식 | 해당 없음 (파일 형식) |
| Git | 변경 추적 및 commit 기록 | 기존 저장소 |
| pytest | 기존 테스트 회귀 확인 | `pyproject.toml`의 `pytest>=9.2.0` |

### 3.3 대안 및 Trade-off

| 대안 | 장점 | 단점 | 채택 여부 |
|---|---|---|---|
| `AGENTS.md` + 템플릿 | 의존성 없음, 사람이 검토·수정 가능, 제약 준수 | 자동 강제 없음 | **Yes** |
| OpenCode 전용 Skill | 트리거 자동화 가능 | "Skill을 만들지 않는다" 제약 위반 | No |
| 보고서 생성 스크립트/CI | 형식·수치 자동 검증 | 초기 범위에 비해 과한 자동화 | No |
| 외부 로깅 서비스 | 중앙 집계 | 새 의존성·비밀 관리 필요 | No |
| 단일 통합 템플릿 | 관리 파일 수 감소 | 실험과 구현의 필수 항목이 달라 혼재 | No |

Trade-off: 자동 검증이 없으므로 보고서 품질은 에이전트의 증거 확인과 사람 검토에
의존한다. 대신 이번 작업의 범위(문서 체계)를 넘는 자동화를 추가하지 않았다.

---

## 4. Testing & Validation

### 4.1 실행한 테스트와 결과

| 명령 | 실행 시점 | 결과 | 비고 |
|---|---|---|---|
| `uv run pytest -q` | 2026-11-18 | **49 passed, 4 skipped**, 29 warnings (2회: 35.77 s, 81.73 s) | 코드 변경 없음의 회귀 확인 |
| 문서 구조 검사 | 2026-11-18 | 통과 | 두 템플릿의 필수 섹션 존재 확인 |
| 수치 일치 검사 | 2026-11-18 | 통과 | `EXP-001`의 주요 수치를 결과 JSON과 대조 |
| Per-query 재계산 | 2026-11-18 | 통과 | pooled Rank-1/mAP가 JSON macro 평균과 소수점 4자리에서 일치 |

- 4개 skipped는 `RUN_MODEL_TESTS=1`이 필요한 MegaDescriptor integration 테스트다.
  이번 작업에서는 실행하지 않았으므로 checkpoint 통합 동작은 `Not verified`다.
- 테스트는 기존 코드에 대한 회귀이며, 새 보고서 시스템 전용 테스트는 없다.

### 4.2 실제 데이터 검증 여부

| 검증 | 방법 | 결과 |
|---|---|---|
| 실험 산출물 존재 | `experiments/results/`, `experiments/splits/` 파일 확인 | 통과 |
| `EXP-001` 수치 | 결과 JSON의 aggregate와 표 대조 | 통과 |
| `EXP-001` 재계산 | `*_per_query.csv`에서 Rank-1/Rank-3/mAP 재계산 | 통과 (macro vs pooled 차이 설명) |
| 참조 경로 | 보고서의 상대 경로 존재 확인 | 통과 |
| 파일명·ID | `EXP-001`, `IMP-001`, kebab-case description | 통과 |
| 기존 파일 덮어쓰기 | 생성 전 `docs/reports/` 확인 | 해당 없음 (신규) |
| 코드 변경 여부 | 이번 작업에서 `src/`, `scripts/`, `tests/`, `config/`, `experiments/`에 쓰기 없음 | 통과 |

### 4.3 확인된 버그 및 해결 상태

| 버그 | 영향 | 상태 | 관련 파일 |
|---|---|---|---|
| 기존 문서의 positive 수 서술 불일치: "평균 22개", "22 → 122" vs 재계산 21.1 / 109.5 | 보고서 수치 해석 | `Open` (이번 범위에서 문서 수정하지 않음) | `docs/benchmark_protocols.md` §2.2, §2.4 |
| 원 실험 stdout/종료 코드 미저장 | 실행 재현성 | `Open` (기존 실험 산출물의 한계) | `experiments/results/*.json` |
| 실행 시점 코드 미커밋 (`git_dirty=true`) | 코드 버전 재구성 | `Open` (기존 상태) | Git history |

문서 불일치는 `EXP-001` 2.3절과 3.5절에 재계산값과 함께 명시했다. 원 문서를
임의로 고치면 기존 연구 기록이 변경되므로, 별도 검토 작업으로 남긴다.

---

## 5. Limitations

### 5.1 구현되지 않은 기능

- 보고서 자동 생성 스크립트, ID 자동 할당, 수치 자동 대조 CI는 만들지 않았다.
- `docs/reports/README.md` 같은 인덱스 파일은 없다. ID 충돌은 디렉터리 스캔으로
  확인한다.
- 보고서 언어는 한국어 기본이며, 영어 템플릿이나 다국어 지원은 없다.

### 5.2 알려진 문제

- 템플릿 섹션 누락이나 수치 불일치를 기계적으로 막을 수 없다.
- ID 배정이 수동이므로 동시에 여러 보고서를 작성하면 충돌할 수 있다.
- `experiments/`와 `data/`는 Git에 없으므로, 다른 checkout에서는 보고서가 참조하는
  산출물이 존재하지 않을 수 있다.
- 기존 문서의 positive 수 불일치가 남아 있다.
- 보고서 작성 중 저장소에 외부 변경이 관찰되었다(`src/animal_re_id/utils/logging.py`,
  `runtime.py` 추가, `scripts/evaluate.py`의 import 변경 등). 이번 작업은 해당 파일을
  수정하지 않았으며, 테스트는 변경 전후 두 번 실행해 모두 통과했다.
- 이번 작업 자체는 커밋되지 않아 `IMP-001`의 Git Commit은 실제 변경 내용을
  가리키지 않는다.

### 5.3 검증되지 않은 동작

- 새로운 세션에서 "리포트 작성해줘" 트리거가 실제로 `AGENTS.md` 규칙을 따라
  Experiment/Implementation을 올바르게 분류하는지 end-to-end로 검증하지 않았다.
  (이번 세션에서는 규칙에 따라 문서를 직접 작성했다.)
- 사람이 템플릿을 읽고 작성하는 워크플로는 테스트하지 않았다.
- checkpoints integration test 4개는 skip 상태다.
- Markdown 렌더러별 표/주석 표시는 확인하지 않았다.

---

## 6. Next Steps

### 6.1 후속 구현

| 우선순위 | 작업 | 이유 |
|---|---|---|
| P1 | P1 DINOv2 실험에 `EXP-002` 템플릿 적용 | 실제 두 번째 실험에서 형식 검증 |
| P1 | 원 실험 코드 커밋 및 checkpoint revision 기록 | `EXP-001` 재현성 한계 해소 |
| P2 | `docs/benchmark_protocols.md` positive 수 서술 검토·수정 | 문서-코드 불일치 해소 |
| P2 | 보고서 ID 인덱스 또는 간단한 링크 검사 | 수동 ID 충돌·broken link 방지 |
| P3 | 필수 섹션 존재 여부를 확인하는 경량 검사 | 템플릿 준수 자동화 (선택) |

### 6.2 후속 테스트

| 테스트 | 목적 | 선행 조건 |
|---|---|---|
| 새 세션에서 트리거 분류 확인 | `AGENTS.md` 규칙의 실제 동작 검증 | 보고서 시스템 커밋 |
| 템플릿 렌더링 확인 | 표·주석·링크 표시 검증 | Markdown 뷰어 |
| `RUN_MODEL_TESTS=1` integration test | checkpoint 로딩 경로 검증 | 모델 다운로드 가능 환경 |
| `EXP-001` 수치 재현 | 코드 커밋 후 동일 결과 확인 | Git clean 상태, 동일 cache |
