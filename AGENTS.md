# AGENTS.md

## Project Context

Animal Re-ID 연구 프로젝트다. PolarBearVidID를 대상으로 frozen MegaDescriptor
베이스라인을 평가하며, 재현 가능한 split·평가·기록을 우선한다.

- 주요 코드: `src/animal_re_id/`, `scripts/`, `tests/`
- 설정: `config/`
- 실험 산출물: `experiments/results/`, `experiments/splits/`, `experiments/cache/`
- 프로토콜 문서: `docs/benchmark_protocols.md`
- 데이터: `data/raw/PolarBearVidID/` (Git 추적 제외)

`experiments/`와 `data/`는 `.gitignore` 대상이다. 보고서에는 이 경로를 상대 경로로
참조하고, 산출물 자체를 복사하지 않는다.

---

## Research Reporting

이 프로젝트는 두 종류의 표준 보고서를 사용한다.

| 종류 | 템플릿 | 저장 위치 | ID |
|---|---|---|---|
| Experiment Report | `docs/templates/experiment_report.md` | `docs/reports/experiments/` | `EXP-XXX` |
| Implementation Report | `docs/templates/implementation_report.md` | `docs/reports/implementations/` | `IMP-XXX` |

### Trigger Rules

다음 요청을 받으면 이 절의 절차를 수행한다.

- "리포트 작성해줘"
- "실험 보고서 작성해줘"
- "구현 보고서 작성해줘"
- "현재 작업 정리해줘"
- 위와 동등한 의미의 요청

요청이 명확하면 해당 템플릿을 사용한다.

일반적인 "리포트 작성해줘" 요청은 최근 완료된 작업을 확인해 다음 기준으로 분류한다.

- 모델 실행·성능 평가 → **Experiment Report**
- 코드 구현·수정·리팩터링 → **Implementation Report**
- 구현과 실험이 모두 포함된 경우 → **Experiment Report**를 기본으로 하고
  Implementation 내용을 포함한다.
- 분류가 불명확하면 임의로 정하지 말고 사용자에게 확인한다.

사용자가 특정 실험·작업·템플릿을 지정하면 그 지정이 위 기본 분류보다 우선한다.

요청 예시:

- "실험 보고서 작성해줘: `megadescriptor_frozen_polarbear_5fold`" → Experiment Report
- "구현 보고서 작성해줘: dataset loader 변경" → Implementation Report
- "리포트 작성해줘: 방금 끝난 5-fold 평가" → Experiment Report
- "현재 작업 정리해줘" → 최근 변경을 확인해 Experiment/Implementation으로 분류

### Evidence Collection

보고서 작성 전에 다음 정보를 실제로 확인한다.

1. 관련 Git 변경 사항 및 Commit (`git log`, `git status`, `git diff`)
2. 실제 구현된 코드 (`src/`, `scripts/`, `tests/`)
3. Config와 실행 명령 (`config/`, 스크립트 docstring)
4. 실험 로그와 Metrics (`experiments/results/*.json`, `*_per_query.csv`;
   per-run 로그가 있으면 `experiments/polarbear/<experiment>/<run_id>/run.log`,
   `metadata.json`, `metrics.json`)
5. 테스트 실행 결과 (실행한 경우 명령과 출력)
6. 모델 Checkpoint 및 데이터 분할 (`experiments/splits/`, 결과 JSON의 model/split)
7. 관련 논문 및 벤치마크 프로토콜 (`docs/benchmark_protocols.md`)

실험 로그가 없는 경우 결과를 임의로 생성하지 않는다. 확인할 수 없는 값은
`Not verified` 또는 `Not available`로 표시한다.

### Reporting Principles

- 실제 확인된 사실과 해석을 구분한다. 해석에는 "추정"임을 명시한다.
- 성공 사례뿐 아니라 실패·오류·성능 저하도 기록한다.
- 모델의 성능을 과장하지 않는다. `SOTA`, "재현했다" 같은 표현은 근거가 있을 때만 쓴다.
- 평가 프로토콜이 다른 논문 수치를 직접 비교하지 않는다. 비교표에는 조건 차이를 함께 쓴다.
- 데이터 누출 가능성을 반드시 검토한다. 특히 사전학습 데이터와 평가 데이터의 중복.
- 실행하지 않은 테스트를 통과했다고 기록하지 않는다.
- 정보가 없으면 `Not verified` 또는 `Not available`로 표시한다.
- 기존 보고서에 작성된 결과를 새로운 실험 결과로 오인하지 않는다.
- 보고서 작성만을 위해 고비용 학습이나 전체 데이터 평가를 재실행하지 않는다.
- 보고서 내용은 기본적으로 한국어로 작성하되 모델·지표·방법론의 공식 명칭은 유지한다.
- 큰 로그, 캐시, 데이터셋, 체크포인트는 보고서에 복사하지 않고 상대 경로로 참조한다.
- 수치·경로·출처가 원본 산출물과 일치하는지 저장 전에 검사한다.

### Report Generation Workflow

1. 보고 대상 작업 또는 실험 식별
2. 적절한 템플릿 선택
3. 관련 코드와 실험 아티팩트 확인
4. 실험 조건·결과·검증 상태 추출
5. 템플릿에 따라 보고서 작성
6. 수치·경로·출처의 일치 여부 검사
7. `docs/reports/`에 저장
8. 사용자에게 저장 경로와 핵심 내용을 요약

기존 작업과 무관한 파일은 수정하지 않는다.

### Storage and Naming

```text
docs/
├── templates/
│   ├── experiment_report.md
│   └── implementation_report.md
└── reports/
    ├── experiments/
    │   └── EXP-001-<description>.md
    └── implementations/
        └── IMP-001-<description>.md
```

- Experiment 파일명: `EXP-XXX-<description>.md`
- Implementation 파일명: `IMP-XXX-<description>.md`
- `<description>`은 소문자 kebab-case ASCII로 쓴다. 예: `megadescriptor`, `dataset-loader`.
- 기존 파일을 덮어쓰지 않는다.
- 새 ID는 해당 디렉터리의 기존 보고서를 확인한 뒤 `최대 번호 + 1`로 배정한다.
  기존 보고서가 없으면 `001`부터 시작한다. 결번이 생겨도 기존 ID를 재사용하지 않는다.
- 같은 ID가 이미 존재하면 파일명을 바꾸지 말고 다음 번호를 사용한다.
- 보고서는 Git에서 추적되도록 `docs/reports/`에 저장한다. `experiments/`의 대용량
  산출물은 Git에 추가하지 않는다.

### Report Validation Checklist

저장 전에 다음을 확인한다.

- [ ] 템플릿의 필수 섹션이 모두 있는가?
- [ ] 모든 수치가 결과 JSON/CSV와 일치하는가?
- [ ] 참조한 경로가 실제로 존재하는가?
- [ ] 확인할 수 없는 값이 `Not verified` / `Not available`로 표시되었는가?
- [ ] 사실과 해석이 구분되어 있는가?
- [ ] 데이터 누출과 프로토콜 차이를 검토했는가?
- [ ] 실행하지 않은 테스트를 통과로 쓰지 않았는가?
- [ ] 기존 파일을 덮어쓰지 않았는가?

### Scope Guardrails

- 새로운 외부 Logging 서비스나 의존성을 추가하지 않는다.
- 기존 평가·학습 파이프라인을 보고서 작성을 위해 수정하지 않는다.
- OpenCode 전용 Skill을 별도로 만들지 않는다. 이 문서와 템플릿만 사용한다.
- 이 `AGENTS.md`의 기존 내용을 삭제하거나 덮어쓰지 않는다. Reporting 규칙 변경은
  해당 섹션만 수정한다.
- 필요하지 않은 대규모 자동화 시스템을 구축하지 않는다. 템플릿 복사와 수동 검증으로
  충분하다.
