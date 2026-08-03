# MPI-INF-3DHP 보정 평가 보고서

## 목적

기존 `production_demo`는 내부 일관성(R1~R6)만 점검했다. 이 평가는 공식 MPI-INF-3DHP의 동기화 RGB, 2D/3D GT, intrinsic calibration으로 실제 오차를 측정한다. 원본은 연구용 라이선스이므로 프로젝트 밖 로컬 임시 캐시에만 저장했다.

## 실행 조건

| 항목 | 값 |
| --- | --- |
| 시퀀스 | S1/Seq1, camera 0 |
| 해상도 / FPS | 2048×2048 / 25fps |
| 회전 구간 | source frame 3212~3331, 120프레임 |
| GT yaw 변화 폭 | 315.33° |
| 2D / 3D 모델 | RTMPose-tiny COCO / VideoPose3D 81-frame H36M |
| 2D 조건 | GT bbox 제공(검출기 제외, 회귀만 평가) |

GT는 공식 28관절을 canonical 16관절로 변환하고, camera axis `(+X right, +Y down, +Z forward)`를 AnimCV axis `(+X right, +Y forward, +Z up)`으로 바꿨다. trim 영상에 맞춰 frame index는 0부터 다시 부여한다.

## 정량 결과

| 단계 | 지표 | 결과 | 판정 |
| --- | --- | ---: | --- |
| 기본 2D 경로 | visible landmark frame | 0 / 120 | 실패 |
| GT-box 2D | PCK@0.2 | 0.9443 (1,920 joints) | 조건부 통과 |
| GT-box 2D | mean / median error | 56.14 / 36.37 px | 참고 |
| 3D lift | root-relative MPJPE | 356.82 mm | 실패 |
| 3D lift | PA-MPJPE | 179.25 mm | 실패 |
| root orientation | yaw MAE / P95 | 112.32° / 167.82° | 실패 |

## 확인된 원인

1. **기본 제품 경로에 사람 검출기가 없다.** `estimate-pose`의 기본 top-down 호출은 전 프레임 전체를 하나의 사람 crop처럼 처리한다. 이 2048² 원거리 피사체에서는 120/120 프레임이 low confidence였다.
2. **2D와 3D의 confidence gate가 분리돼 있다.** 2D `visibility_threshold=0.1`을 줘도 lift 단계의 `min_observation_confidence=0.3`이 torso observation을 다시 무효화해 root yaw가 중단됐다. 두 값을 모두 0.1로 맞춰야 진단을 끝까지 수행할 수 있었다. 이는 품질 개선이 아니라 실패를 측정하기 위한 조건 통일이다.
3. **H36M 학습 VideoPose3D가 이 데이터/동작에서 3D 구조와 방향을 복원하지 못한다.** GT bbox라는 유리한 조건에서도 PA-MPJPE와 yaw 오차가 크다. 이전 yaw smoothing/bend-plane 통과는 이상치 억제에 한정된다.

## 결론과 다음 우선순위

현재 프레임워크는 FBX를 만들 수 있지만 독립 GT 평가에서 3D와 root yaw가 실패했으므로 **게임 제작 파이프라인에 채택할 수준으로 증명되지 않았다.**

1. 제품 경로에 detector/tracker를 필수 연결하고 full-frame top-down fallback을 실패로 명시한다.
2. 단일 `observation_confidence` 정책을 2D→3D→root에 전달하고 hold/reject를 보고한다.
3. MPI-INF-3DHP와 도메인이 맞는 3D estimator 또는 camera-aware/multi-view 보정 방법을 도입한 뒤, 같은 회전 구간에서 MPJPE·PA-MPJPE·yaw MAE를 재측정한다.
4. 세 지표의 사전 gate를 정한 뒤에만 constraint rig와 FBX를 재검증한다.

## 보완 1단계: detector/tracker 필수화 — 통과

기본 `estimate-pose`는 이제 RTMDet-tiny COCO person detector를 자동 준비하고,
단일 대상 tracker가 선택한 bbox에서만 RTMPose를 실행한다. 이전의 full-frame
top-down fallback은 `process_frame` API에서도 제거했다. CLI는 기본적으로
`*_tracking_report.json`을 함께 작성하며, 추적 성공률 95%를 입구 게이트로 둔다.

같은 회전 구간(120프레임)에서 RTMDet-tiny의 결과는 다음과 같다.

| 지표 | 결과 | 게이트 | 판정 |
| --- | ---: | ---: | --- |
| tracked frame | 120 / 120 | ≥ 95% | 통과 |
| tracking success rate | 1.000 | ≥ 0.950 | 통과 |
| no-detection frame | 0 | 0 권장 | 통과 |
| mean person candidates | 1.058 | 참고 | 단일 대상 안정 |
| 2D PCK@0.2 (처음 30프레임) | 1.000 (480 joints) | ≥ 0.900 | 통과 |

이로써 **검출/추적 입구 문제는 해당 구간에서 해소**됐다. 단, 기존 3D/yaw 실패
수치는 이 변경으로 아직 재평가되지 않았으므로, 다음 단계는 confidence 정책 통합이다.

## 보완 2단계: confidence 정책 통합 — 통과

`PoseSequence`는 이제 2D의 `observation_confidence_threshold`를 JSON에 기록한다.
`lift-pose3d`는 별도 `--min-observation-confidence`가 없으면 이 값을 상속하고,
`LiftedPoseSequence`와 `RootMotionSequence`도 같은 값을 보존한다. 따라서 한 단계에서
유효였던 관절이 다음 단계의 다른 기본 threshold 때문에 무효가 되는 이중 게이트를 없앴다.

| 검증 | 결과 | 판정 |
| --- | --- | --- |
| 입력 pose 기록 threshold | 0.1 | 기록됨 |
| lift 상속 threshold | 0.1 | 일치 |
| root 상속 threshold | 0.1 | 일치 |
| 회전 구간 root 처리 | 120 / 120 frames | 완주 |

이전 산출물처럼 정책 메타데이터가 없는 JSON은 재현성을 깨지 않도록 legacy fallback
0.3을 사용한다. 이 단계는 낮은 신뢰도 데이터를 신뢰할 수 있게 만든다는 뜻이 아니며,
정책을 명시하고 hold/reject 판단을 일관되게 만드는 작업이다.

## 보완 3단계: 3D estimator 후보 조사 — 라이선스 결정 대기

| 후보 | 3DHP 적합성 | 통합 판단 |
| --- | --- | --- |
| MMPose VideoPose3D / MotionBERT | 제공 checkpoint가 H36M 전용 | 현재 실패 기준선. MMPose도 일반화 한계를 명시한다. |
| ManiPose | MPI-INF-3DHP checkpoint 제공 | 코드가 AGPL-3.0이라 상용 제품 경로 통합 불가 |
| MeTRAbs | MPI-INF-3DHP skeleton·calibration 지원 | 코드 MIT이나 공개 모델은 학습 데이터 라이선스로 비상업용 전용 |

따라서 공개 checkpoint를 그대로 채택하는 것은 게임 제작이라는 최종 목표와 충돌한다.
다음 중 하나의 제품 결정을 받은 후에만 3D 단계 구현을 확정한다.

1. 상업적 사용권이 명시된 외부 3D pose SDK/model을 도입한다.
2. 사용권을 확보한 데이터로 3D estimator를 자체 학습·배포한다.

어느 경우든 이 문서의 동일 120프레임 회전 구간에서 MPJPE, PA-MPJPE, yaw MAE를
재측정해 기존 실패 수치보다 개선됐음을 확인해야 한다.
