# AnimCV 검토 문서

이 디렉터리는 사람이 읽는 한국어 중간 보고서의 기준 위치다. JSON은
재현 가능한 기계 판정·파이프라인 입출력으로 계속 유지하며, 이 문서는
그 결과의 목적·판정·한계를 설명한다.

| 문서 | 내용 | 최신 판정 |
| --- | --- | --- |
| [01_프레임워크_현황](01_프레임워크_현황.md) | 현재 구조, 완료 범위, 남은 작업 | 3D 타깃 단계까지 구현, 최종 리타깃은 미완료 |
| [02_품질_보강_보고서](02_품질_보강_보고서.md) | R1–R3 품질 게이트와 수치 | R1–R3 통과 |
| [03_카메라_보정_보고서](03_카메라_보정_보고서.md) | 캘리브레이션·재투영 검증 방법 | 명시적 보정 지원, 데모 자동 추정은 거부 |
| [04_MPI-INF-3DHP_보정_평가](04_MPI_INF_3DHP_보정_평가.md) | 외부 GT 기반 2D/3D/yaw 평가 | 3D·yaw 실패, 제품 채택 불가 |

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
