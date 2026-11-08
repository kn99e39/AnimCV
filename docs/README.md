# AnimCV 검토 문서

> **상위 제품 목표:** [PRODUCT_VISION.md](../PRODUCT_VISION.md). AnimCV의 장기 목표는 영상 동작의 단순 복원을 넘어 **편집 가능한 리그 애니메이션, 스타일 재현/전이, 사용자 주도 애니메이션 제작**이다. 아래 초기 보고서와 게이트는 이 제품 비전의 일부 단계만 다룬다.

## 현재 기준 문서·진행상황 (2026-11-08)

> **2026-11-09 갱신 (W126):** 최신 완료 Worklog는 [126 — Commit-Time World-Pin Gate Integrity](126_WORKLOG_COMMIT_TIME_WORLD_PIN_GATE_INTEGRITY.md), **동결 CASE 규칙상 CASE C**. 사유: 선언된 음성 대조 N12 한 건의 1차 거부 분류 예측이 틀렸다(선언 자체의 우선순위와 모순); 그 대조도 게시 없이 거부되었고 나머지 대조·실제 replay 게이트는 모두 통과. EXPERIMENTAL opt-in `experimental-video-action commit --decision accept`는 더 이상 `proposal.json`의 gate 결정을 신뢰하지 않는다: 원본 요청 바이트, 검증된 head readback, 불변 baseline에서 재유도한 binding으로 동결 W116/W118 gate를 재계산(저장 readback 1회 + 새 Blender readback 1회)하고 봉인된 제안과 일치할 때만 게시하며, 불일치는 `REFUSED_SEMANTIC_AUTHORIZATION`(head 불변)이다. W124/W125 미봉인 proposal은 accept 불가(재제안 필요). W125 우회(UNREACHABLE 후보의 gate 결정 수동 ACCEPT)는 시작 커밋 코드에서 재현되었고(합성 136 mm, S1 저장 산출물 557 mm 게시), W126 코드에서는 합성 BaseRig와 S1·S3 저장 산출물 replay에서 거부되었다 ([사용법](EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md)). 증거 수준: **Agent-run**; 봉인은 비키 SHA(일관성 검사이지 인증이 아님), 실행 전체를 일관되게 재작성하는 공격은 범위 밖. 애니메이션 품질·FBX(여전히 FAIL)·Production 기본 이행 주장이 아니다.

> **2026-11-09 갱신 (W125):** 최신 완료 Worklog는 [125 — Crash-Consistent Editable Action Transactions](125_WORKLOG_CRASH_CONSISTENT_EDITABLE_ACTION_TRANSACTIONS.md), **CASE A (트랜잭션 능력 한정)**. EXPERIMENTAL opt-in 경로 `experimental-video-action`의 accept/reject/revert가 OS 잠금 + 봉인된 journal + 단일 원자적 `refs.json` 게시 지점으로 바뀌었고 `recover`/`adopt` 명령이 추가되었다 ([사용법](EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md)). 증거 수준: **Agent-run**, Windows 11/NTFS의 프로세스 수준 장애(예외, `os._exit`, 강제 종료)와 경쟁 프로세스만 시험; 정전·OS 크래시는 시험·주장하지 않음. `proposal.json`은 봉인되지 않아 수동 변조된 gate 결정은 막지 못한다(W124부터 존재). 애니메이션 품질·FBX(여전히 FAIL)·Production 기본 이행 주장이 아니다.

> **2026-11-09 갱신 (W124):** 최신 완료 Worklog는 [124 — Opt-in Video → Editable Action + World Pin](124_WORKLOG_OPT_IN_VIDEO_TO_EDITABLE_ACTION_WORLD_PIN.md), **CASE B**. 첫 **EXPERIMENTAL opt-in** 사용자 경로 `motion-tool experimental-video-action` ([사용법](EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md))이 추가되었다: 인증된 30 fps 실제 영상(W121/W122 S1·S3 engineering replay) → frozen M1+S0 live 인식 → W105 BaseRig 편집 가능 `.blend` Action → left-wrist WORLD_POSITION_PIN 제안/수락/거부/정확한 되돌리기. `.blend` 경로는 Agent-run 검증됨; FBX round-trip은 FAIL(W107–W112 계열 결함 재현). 기본 CLI/GUI 경로·기본 정책은 변경되지 않았으며 Production 기본 이행이 아니다. W123은 Worklog 123으로 완료(CASE B)되었다.

- **브랜치:** `arch/single_frame_first` (사용자 확인). `On_Work`가 활성이라는 오래된 `AGENTS.md` 문구는 수정되었다.
- **제품·방향:** [PRODUCT_VISION.md](../PRODUCT_VISION.md) §4.1 — Reference → Pose Hypothesis + Learned Uncertainty → Reliability-Aware Base → 미래 Conditional Generator → Semantic Constraints + Rig Evaluator.
- **Perception 계약:** [Architecture_v3_FramePose.md](../Architecture_v3_FramePose.md); v3가 대체하지 않은 MotionGraph/Blender 계약은 [Architecture_v2.md](../Architecture_v2.md).
- **마스터 플랜 및 실구현 상태:** [00_마스터_플랜.md](00_마스터_플랜.md), [01_프레임워크_현황.md](01_프레임워크_현황.md). 둘의 과거 2D→Temporal-only 계획은 역사적으로 보존되며, 2026-11-08 최신 상태 절이 우선한다.
- **최신 완료 Worklog:** [122 (Real-time World Pin Guide Utility)](122_WORKLOG_REALTIME_WORLD_PIN_GUIDE_UTILITY.md), CASE B (제한적); [121](121_WORKLOG_INDEPENDENT_REAL_VIDEO_WORLD_PIN_TRANSFER.md) CASE D; [120](120_WORKLOG_MATCHED_GUIDE_FLEX_FLOOR_ATTRIBUTION.md) CASE B.
- **최신 누적 코드 감사:** [W116–W120 audit](AUDIT_W116_W120_2026_11_06.md). W121–W122는 다음 5배수 감사 주기 대상이며, 여기서 독립 재실행했다고 주장하지 않는다.
- **발행되었지만 결과가 확인되지 않은 Batch:** [W123 — Learned 3D Reliability Assessor](next_batch.md). W123은 frozen 모델 + sidecar calibration/decision-utility 가설만 검증하며 Generator/Production 통합 승인과 다르다.
- **Pose Completion:** [POSE_COMPLETION_PRODUCT_PLAN.md](POSE_COMPLETION_PRODUCT_PLAN.md) — 장기 제품 필수 요구, 현재 일부 메커니즘 실패가 요구 자체의 폐기를 뜻하지 않는다.

아래 표는 **초기 00–22번 문서의 당대 판정 목차**다. '최신 판정' 열은 프로젝트 전체의 현행 상태가 아닌 해당 문서 작성 단계의 판정으로 읽는다.

이 디렉터리는 사람이 읽는 한국어 중간 보고서의 기준 위치다. JSON은
재현 가능한 기계 판정·파이프라인 입출력으로 계속 유지하며, 이 문서는
그 결과의 목적·판정·한계를 설명한다.

| 문서 | 내용 | 최신 판정 |
| --- | --- | --- |
| [00_마스터_플랜](00_마스터_플랜.md) | 현재 방향(상단 최신화) + 초기 선형 계획 보존 | W123 발행, Production 통합 미완료 |
| [01_프레임워크_현황](01_프레임워크_현황.md) | 실제 코드·실험·제품 통합 상태 구분 | 최신 FramePose/Editing은 부분 구현 및 Scratch 검증 |
| [02_품질_보강_보고서](02_품질_보강_보고서.md) | R1–R3 품질 게이트와 수치 | R1–R3 통과 |
| [03_카메라_보정_보고서](03_카메라_보정_보고서.md) | 캘리브레이션·재투영 검증 방법 | 명시적 보정 지원, 데모 자동 추정은 거부 |
| [04_MPI-INF-3DHP_보정_평가](04_MPI_INF_3DHP_보정_평가.md) | 외부 GT 기반 2D/3D/yaw 평가 | 3D·yaw 실패, 제품 채택 불가 |
| [05_자체_학습_전환_계획](05_자체_학습_전환_계획.md) | licensed 데이터 기반 학습·holdout 운영 | baseline 구현 중 |
| [06_SERVER_AI_AGENT_TRAINING_RUNBOOK](06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md) | 서버 Agent용 데이터 검증·학습·판정 실행 지침 | 실행 준비 완료 |
| [07_WEB_DATASET_TRAINING_PLAN](07_WEB_DATASET_TRAINING_PLAN.md) | 웹 원천 데이터셋 도입·license·adapter 계획 | C1–C6 선행 구현 필요 |
| [08_RESEARCH_DATASET_ASSESSMENT](08_RESEARCH_DATASET_ASSESSMENT.md) | 공개 dataset 조사, 코드 현황, 구현 배치 | 설계 완료·구현 승인 대기 |
| [15_WORKLOG_A12_MAGNITUDE_DIRECTION_ATTRIBUTION](15_WORKLOG_A12_MAGNITUDE_DIRECTION_ATTRIBUTION.md) | A12 Cartesian torso residual magnitude/direction 진단 및 A13 판정 | **NO-GO: A13 미실행** |
| [16_WORKLOG_SOURCE_TAIL_AGGREGATION](16_WORKLOG_SOURCE_TAIL_AGGREGATION.md) | A12 global hard-tail source reweighting 및 3DPW coverage 진단 | **Case B: coverage 우선, 후보 학습 중단** |
| [17_WORKLOG_3DPW_GENERALIZATION_SUPPORT](17_WORKLOG_3DPW_GENERALIZATION_SUPPORT.md) | 3DPW GT target/input support, hard-case overlap, ambiguity 진단 | **Mixed: validation input shift + test monocular ambiguity** |
| [18_WORKLOG_A14_BILATERAL_FORWARD_DEPTH](18_WORKLOG_A14_BILATERAL_FORWARD_DEPTH.md) | all-frame bilateral forward-depth(+Y) 명시 supervision 학습·평가 | **Case B(악화 동반): hard set 개선하지만 yaw는 악화, 거절** |
| [19_WORKLOG_TRAINING_THROUGHPUT_DIAGNOSIS](19_WORKLOG_TRAINING_THROUGHPUT_DIAGNOSIS.md) | 학습 step stage-by-stage 처리량 진단(GPU utilization 10-20% 증상) | **Case D: 커널 launch-overhead 지배, 이번 batch fix 없음** |
| [20_WORKLOG_TORCH_COMPILE_CANDIDATE](20_WORKLOG_TORCH_COMPILE_CANDIDATE.md) | 커널 단위 fragmentation 증거 + torch.compile 후보 검증 | **ACCEPT: +54.7% 처리량, opt-in `compile_training_graph` 추가** |
| [21_WORKLOG_CORRECTED_BILATERAL_FORWARD_DEPTH](21_WORKLOG_CORRECTED_BILATERAL_FORWARD_DEPTH.md) | A11/A12 진단 repair + denominator 수정한 A14 clean A/B(compiled A9 control) | **Case C+E: 수정해도 자기 target도 개선 못함, historical A14보다 강한 negative** |
| [22_WORKLOG_A16_GENERALIZATION_GAP](22_WORKLOG_A16_GENERALIZATION_GAP.md) | A11 selector 완전 isolation + A16이 train-domain에서만 되고 3DPW에서 실패하는 이유 | **Case D(coherence+temporal branch persistence), frozen probe P1 → 다음은 structured torso head** |

## 원칙

- FBX를 생성할 수 있다는 사실만으로 게임 제작 품질을 주장하지 않는다.
- `examples/production_demo/*.json`은 수치를 재검증하는 원본 증거다.
- 문서의 **통과**는 해당 단계의 제한된 게이트 통과를 뜻한다. 이후 단계의
  품질, 특히 IK/FK 리타깃·발 접지·루트 이동을 자동으로 보장하지 않는다.

## 관련 기계 산출물

- 최종 R3 3D 품질: `examples/production_demo/bend_stabilized_quality_report.json`
- R2 2D 보존 프록시: `examples/production_demo/kinematic_reprojection_report.json`
- 자동 카메라 추정: `examples/production_demo/auto_camera_calibration_report.json`
- 보정 카메라 입력 예시: `examples/camera_calibration.example.json`
