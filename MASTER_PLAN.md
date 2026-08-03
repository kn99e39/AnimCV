# AnimCV 마스터 플랜

이 문서는 AnimCV를 **단일 영상에서 게임 제작에 사용할 수 있는 FBX
애니메이션**으로 발전시키기 위한 선형 기준 계획이다. 각 단계는 이전
산출물과 검증 게이트를 통과해야 다음 단계로 진행한다. JSON 보고서는
기계 증거, `docs/`는 한국어 검토 문서다.

## 최종 성공 기준

1. 입력 영상의 인물이 3D 타깃에서 해부학적으로 일관된 자세를 유지한다.
2. 리그는 올바른 root/torso 방향, hinge 방향, 팔·다리 end-effector를
   표현한다.
3. FBX는 정확한 FPS·프레임 범위·키를 포함하며 Blender 재임포트에서
   시각적으로 원본 동작과 일치한다.
4. 결과는 단일 성공 사례가 아니라 회전·보행·가림·측면·다인물 입력에서
   품질 게이트와 회귀 테스트를 통과한다.

## 1. 입력과 재현성 — 완료

- 영상, 추출 프레임, 추적 2D pose, rig, mapping, JSON 보고서를 보존한다.
- deterministic demo와 실패 사례를 분리한다.
- **게이트:** 동일 입력에서 동일 중간 산출물과 감사 수치를 재생성한다.

## 2. 좌표·리그 계약 — 완료

- 이미지·카메라·캐릭터·rig-local·Blender quaternion/부모 변환을 명시한다.
- FBX key writer가 parent-relative local rotation과 frame range를 보존한다.
- **게이트:** identity, ±90° 회전, 부모-자식, FPS/frame range round trip.

## 3. 시간적 3D 타깃 재구성 — 완료(품질 보강 지속)

- VideoPose3D로 pelvis-relative 3D를 생성한다.
- R1 관측 유효성, R2 고정 길이, R3 bend plane, R4 yaw provenance,
  R5 uncertainty sidecar, R6 시각/회귀 검증을 적용한다.
- **현재 게이트:** limb CV·bend flip·yaw hold·uncertainty·시각 skeleton 검사.

## 4. 카메라·전역 방향 복원 — 진행 중, 현재 최우선

MPI-INF-3DHP의 실제 회전 구간에서 현 H36M VideoPose3D/root-yaw 경로가 실패했음이
확인됐다. detector/tracker 연결과 estimator 교체/보정 검증이, 이후 리그 polish보다
앞선다.

### 4-0. detector/tracker 입구 게이트 — 완료

- RTMDet-tiny detector와 단일 대상 tracker를 기본 pose 경로에 연결했다.
- S1/Seq1 camera 0 회전 구간 120프레임에서 tracking success rate 1.000,
  no-detection 0으로 0.95 게이트를 통과했다.
- 다음 작업: confidence 정책을 2D→3D→root 단계에 통합하고, 동일 구간의
  3D/yaw 오차를 재측정한다.

### 4-1. 관측 confidence 정책 통합 — 완료

- 2D pose artifact가 threshold를 기록하고 3D lift와 root motion이 이를 기본 상속한다.
- 0.1 정책의 회전 구간 120프레임이 root 단계까지 완주했다.
- 다음 작업: MPI-INF-3DHP 회전 구간의 3D/yaw 오차를 줄일 수 있는 estimator를 선정하고
  동등 조건에서 재측정한다.

### 4-2. 상용 사용 가능한 3D estimator 확보 — 제품 결정 대기

- 공개 3DHP checkpoint 후보를 조사했으나, ManiPose는 AGPL-3.0이고 MeTRAbs 공개 모델은
  비상업용 전용이다. 현재 목표에는 그대로 채택하지 않는다.
- 필요한 결정: 상용 SDK/model 라이선스 도입 또는 사용권 확보 데이터로 자체 학습.
- 결정 후 게이트: 동일 회전 구간 MPJPE·PA-MPJPE·yaw MAE 재측정 및 사전 기준 통과.

- 원본 영상의 dataset/sequence/camera ID를 식별한다.
- 확인 가능한 공개 calibration은 intrinsics·distortion·extrinsics까지
  `animcv_camera_calibration_v1`으로 기록한다.
- camera-space 3D, 2D pose, 2D 실루엣/폭, 발 진행 방향을 결합해 root yaw를
  추정한다.
- **게이트:** calibrated reprojection, 공개 GT 또는 수동 기준 yaw 대비 오차,
  측면 전환 시 root/torso가 실제로 회전하는 Blender 시각 검사.
- **차단 조건:** 출처/카메라 ID가 확정되지 않은 웹 보정값은 적용하지 않는다.

## 5. 제약 기반 리그 adapter — 진행 중

- R3/R5의 end-effector·pole target을 리그별 rest basis와 hinge axis로
  변환한다.
- 현재의 안전한 3D direction FK bake를 기준선으로 유지한다.
- 그 위에 two-bone IK, pole vector, knee/elbow limit, torso/root track,
  feet target을 추가한다.
- **게이트:** constraint target의 unsafe limb hold, joint-limit 위반 0,
  side-facing clip의 root/torso 방향 일치.

## 6. 전역 이동·발 접지

- calibrated camera 또는 접지 제약으로 root translation을 복원한다.
- 발 contact, plantar lock, foot sliding 보정을 구현한다.
- **게이트:** 접지 구간 foot velocity/slide 한도, root trajectory 검증.

## 7. FBX 생성·재임포트 검증

- dense clip을 허용 오차 내에서 key reduction 후 Blender/FBX로 export한다.
- `.blend`와 exported FBX를 모두 재오픈해 frame range, FPS, FCurve,
  animated bone, rendered pose를 검사한다.
- **게이트:** 지정 clip range/FPS, 각 mapped bone 키 존재, target/rig render
  비교에서 축·부모·root 회전 오류 없음.

## 8. 확장 검증과 제품화

- 정면·측면·회전·보행·가림·다인물·이동 카메라 fixture를 추가한다.
- CLI/GUI에 calibration input, uncertainty, audit render, failure report를
  노출한다.
- **게이트:** 지원 입력군별 합격/거부 기준, 재현 가능한 docs와 regression suite.

## 현재 위치와 다음 순서

현재는 4단계와 5단계의 경계다. 3D target 자체의 관절 반전은 해결했지만,
production clip에서 root yaw가 실제 측면 회전을 놓쳐 BaseRig FBX의 torso가
정면에 남는다. 따라서 다음 실행 순서는 다음과 같다.

`4-A 출처·camera ID 식별 → 4-B 보정값 적용/검증 → 4-C yaw 정확도 게이트
→ 5-A root/torso adapter 수정 → 5-B rig-space IK/pole/limit → 6 foot/root
→ 7 FBX visual acceptance → 8 확장 검증`

이 순서를 건너뛰고 IK나 key reduction만 정교하게 만드는 것은 잘못된 root
방향을 더 정교하게 bake할 뿐이므로 금지한다.
