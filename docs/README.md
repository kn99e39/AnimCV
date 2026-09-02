# AnimCV 검토 문서

이 디렉터리는 사람이 읽는 한국어 중간 보고서의 기준 위치다. JSON은
재현 가능한 기계 판정·파이프라인 입출력으로 계속 유지하며, 이 문서는
그 결과의 목적·판정·한계를 설명한다.

| 문서 | 내용 | 최신 판정 |
| --- | --- | --- |
| [00_마스터_플랜](00_마스터_플랜.md) | 유일한 선형 실행 계획 | 3D estimator/자체 학습 단계 진행 중 |
| [01_프레임워크_현황](01_프레임워크_현황.md) | 현재 구조, 완료 범위, 남은 작업 | 3D 타깃 단계까지 구현, 최종 리타깃은 미완료 |
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
