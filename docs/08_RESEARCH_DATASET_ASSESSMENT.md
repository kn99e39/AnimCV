# Corrected Project Assumptions

**[Confirmed]** AnimCV는 프로젝트 소유자가 개인적으로 쓰는 단일 RGB video→3D pose→FBX
도구다. 직접 촬영, 산업용 카메라, performer 모집, 대규모 자체 데이터셋 제작은 기본 계획이
아니다.

**[Recommendation]** 기본 경로는 `MMPose/RTMPose 2D → pretrained VideoPose3D 또는
research-dataset-trained AnimCV lifter → 기존 보정·retarget → FBX`로 둔다. 공개
VideoPose3D는 실패 baseline으로 유지하며, 연구 데이터셋을 이용한 재학습 후보와 비교한다.

**[Confirmed]** 기존 MPI-INF-3DHP 회전 구간에서 H36M VideoPose3D baseline은
PA-MPJPE 179.25 mm, root yaw MAE 112.32°로 목표를 통과하지 못했다.

# Existing Repository Support

| 범위 | 상태 | 근거 |
| --- | --- | --- |
| MMPose/RTMPose 2D front end | **[Confirmed]** 구현됨 | `estimate-pose`와 tracker 경로 |
| Pretrained VideoPose3D | **[Confirmed]** 구현됨 | `lift-pose3d`, H36M 17-joint 입력/출력 |
| MPI-INF-3DHP import | **[Confirmed]** 구현됨 | GT import와 `import-mpi3dhp-supervised-dataset`, raw 28→canonical 17 변환 |
| MPI-INF-3DHP 2D/3D audit | **[Confirmed]** 구현됨 | `audit-mpi3dhp-2d`, `audit-mpi3dhp-3d`; MPJPE/PA-MPJPE/yaw report |
| Small from-scratch temporal lifter | **[Confirmed]** 구현됨 | mask·clip-safe combine·build/train/evaluate/infer CLI와 smoke test |
| Human3.6M adapter | **[Confirmed]** 없음 | 저장소 검색상 importer·CLI·test 없음 |
| AMASS adapter/virtual camera projection | **[Confirmed]** 없음 | 저장소 검색상 SMPL/AMASS renderer·mapping 없음 |
| 3DPW adapter/audit | **[Confirmed]** 없음 | 저장소 검색상 importer·CLI·test 없음 |

**[Confirmed]** 기존 MPI importer는 dataset의 camera-space mm 좌표를 AnimCV camera axes
`+X right,+Y forward,+Z up`의 root-relative metre로 바꾸며, pelvis 기준을 뺀다. 이 변환은
새 source adapter의 기준 구현으로 재사용할 수 있다.

# Dataset and License Matrix

아래는 법률 판단이 아니라 2026-08-03에 공식 페이지에서 확인한 접근 조건 기록이다.

| Dataset | 공식 원천/형식 | 연구·개인 비상업 사용 | 학습/재배포 상태 | 판정 |
| --- | --- | --- | --- | --- |
| Human3.6M | [공식 EULA](https://vision.imar.ro/human3.6m/eula.php), multi-camera RGB·2D/3D | 무료 접근은 academic address의 academic use only | 상용은 별도 문의, 데이터 재배포 금지 | **[Blocked]** 소유자의 academic eligibility와 checkpoint 공개 조건 확인 필요 |
| MPI-INF-3DHP | [공식 페이지](https://vcai.mpi-inf.mpg.de/3dhp-dataset/), indoor/outdoor RGB·2D/3D | 공식 페이지에 download는 있으나 명시적 terms가 보이지 않음 | training/test subset 사용·재배포 조건 미확인 | **[Blocked]** official terms 또는 담당자 확인 필요 |
| AMASS | [공식 license](https://amass.is.tue.mpg.de/license.html), mocap/SMPL motion | non-commercial scientific research, education, artistic project | dataset 공유 금지; RGB pair는 직접 virtual projection 필요 | **[Likely]** 개인 사용 목적과 sub-dataset 조건 확인 후 synthetic training 후보 |
| 3DPW | [공식 license](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html), moving-camera RGB·2D/3D | non-commercial scientific research만 명시 | 데이터 수정·제3자 제공 제한; challenge는 training 금지 | **[Likely]** external evaluation 우선, training 제외 권고 |

**[Recommendation]** dataset 원본·가공 annotation·dataset-derived checkpoint는 Git 밖에 둔다.
사용 조건이 모호하면 사용을 법적으로 단정하지 않고 `owner review required`로 기록한다.

# Recommended Dataset Roles

| 역할 | 우선 source | 이유와 제한 |
| --- | --- | --- |
| Baseline supervised training | Human3.6M | **[Likely]** 2D/3D lifting의 표준 비교 축. indoor bias와 access 조건이 존재한다. |
| 도메인 보강·평가 | MPI-INF-3DHP | **[Confirmed]** 이미 adapter/audit가 있어 회전·측면·가림 평가를 즉시 재사용 가능하다. training subset은 조건 확인 뒤 결정한다. |
| motion 다양성 synthetic augmentation | AMASS | **[Likely]** 40시간 이상/300명 이상 mocap archive지만 RGB가 아니다. virtual camera·noise augmentation 구현이 선행된다. |
| 외부 generalization holdout | 3DPW | **[Recommendation]** moving-camera outdoor RGB/2D/3D라서 final holdout에 적합하며, 공식 challenge도 training을 금지한다. |
| 실제 사용 visual regression | 소유자 video | **[Recommendation]** 정량 GT 없이 FBX visual acceptance만 수행한다. |

# Skeleton and Coordinate Contracts

**[Confirmed]** AnimCV temporal target은 H36M-style 17 joint다:
`pelvis, hips/knees/ankles, spine, thorax, neck, head, shoulders/elbows/wrists`.
2D canonical `neck`은 target의 `thorax`에도 복제된다.

**[Recommendation]** adapter는 source별로 다음 manifest를 보존한다: source joint name/index,
AnimCV joint name, direct/derived 여부, derivation 식, validity rule. source에 없는 joint를
임의 위치로 합성하지 않는다. 필요한 경우 해당 joint를 invalid로 두고 mask 정책으로 처리한다.

**[Confirmed]** ingestion boundary에서만 source coordinate convention을 AnimCV의
camera-root-relative metre axes로 변환한다. source별로 world/camera frame, handedness,
axis, units, `world_to_camera`/`camera_to_world`, root subtraction 시점을 manifest에 명시한다.

# Training and Evaluation Strategy

**[Recommendation]** 다음 후보를 같은 independent protocol로 비교한다.

| Candidate | 입력→3D | 목적 |
| --- | --- | --- |
| A | MMPose 2D → pretrained VideoPose3D → current post-processing | 현재 실패 baseline |
| B | dataset GT 2D → pretrained VideoPose3D | detector noise와 lifter 한계 분리 |
| C | MMPose 2D → H36M-trained AnimCV lifter | source-trained baseline |
| D | MMPose 2D → H36M + permitted 3DHP + AMASS synthetic lifter | domain/motion diversity ablation |
| E | D + optional depth cue | depth ablation |

**[Confirmed]** `animcv_supervised_3d_lifter_dataset_v2`는 `target_valid` mask와 `sequences`
목록을 보존한다. trainer/evaluator는 invalid joint를 loss·MPJPE에서 제외하고, combined
dataset의 temporal window를 sequence 안에만 구성한다. MPI-INF-3DHP direct importer와
dataset-neutral 3D evaluator가 있다. H36M/AMASS/3DPW adapter는 아직 없다.

**[Confirmed]** 현재 own-data `evaluate`는 MPJPE와 joint-error P95만 낸다. 반면
MPI-specific audit에는 MPJPE, PA-MPJPE, root yaw MAE/P95가 있으나 representative gate를
판정하지 않는다. hinge flip, per-action/view/occlusion metrics, velocity/acceleration,
inference time, VRAM은 없다.

**[Recommendation]** GT 2D와 MMPose-estimated 2D 실험을 별도 report로 유지하고, 실제 사용과
가까운 지표는 estimated 2D 입력으로만 주장한다. split은 source subject·원본 sequence·action
단위이며, AMASS는 원본 mocap sequence가 같으면 virtual camera가 달라도 같은 split에 둔다.

# Depth Estimator Role

**[Confirmed]** Depth Anything V2는 optional depth sampling으로 구현돼 있으나, temporal
lifter input/GT 생성에는 사용되지 않는다.

**[Recommendation]** depth는 GT 대체물이 아니라 front/back ordering, root yaw 보조,
occlusion/uncertainty, predicted 3D depth-order consistency에 한정해 Candidate D/E ablation으로
판정한다. 기본 학습 경로의 필수 의존성으로 올리지 않는다.

# Repository Gaps

1. **[Confirmed]** dataset intake manifest와 source adapter 공통 contract가 없다.
2. **[Confirmed]** Human3.6M/AMASS/3DPW adapter가 없다.
3. **[Confirmed]** invalid-joint mask와 sequence-aware multi-clip sampler는 temporal lifter v2에 구현됐다. source-level validity/provenance audit은 없다.
4. **[Confirmed]** dataset-neutral PA-MPJPE/yaw/flip/per-slice evaluator가 없다.
5. **[Likely]** AMASS virtual projection에는 SMPL→17 mapping, camera sampler, detector-like
   noise/confidence/occlusion augmentation이 별도 필요하다.

# Proposed Implementation Batches

1. **Dataset intake and canonical adapter contract** — manifest, coordinate/skeleton provenance,
   H36M adapter부터 test fixture로 검증한다.
2. **Dataset-neutral evaluation** — PA-MPJPE, yaw, flip, per-sequence/action/view/occlusion 및
   latency/VRAM report를 구현하고 MPI audit를 공통 contract로 옮긴다.
3. **Research-source adapters and baseline experiments** — H36M, MPI training subset(조건 확인 후),
   3DPW evaluation adapter를 추가하고 A/B/C를 재현한다.
4. **AMASS synthetic branch and depth ablation** — mapping 검증 후 virtual projection·augmentation을
   추가하고 D/E를 비교한다.
5. **Retarget acceptance** — 통과 후보만 실제 video와 Blender FBX visual regression에 연결한다.

# Documentation Changes

**[Recommendation]** Stage 4-3은 다음으로 재정의한다:

`Research-dataset-based supervised 2D→3D baseline and domain evaluation`

기존 own-data/multiview 코드는 삭제하지 않고 optional future path로 보류한다. Stage 4-2의
상용 estimator·제품화 표현은 research baseline comparison으로 교체한다. 이 보고서와
`05`, `06` 문서는 web-dataset ingestion을 기본 경로로 삼는다.

# Decisions Required From Owner

1. Human3.6M의 academic access 신청 가능 여부.
2. MPI-INF-3DHP의 공식 terms 확인 또는 담당자 문의 진행 여부.
3. AMASS/3DPW의 개인 사용 목적이 각 공식 비상업 연구 조건에 맞는지 소유자가 확인할지 여부.
4. 첫 구현 범위를 Batch 1–3(공통 기반)까지만 승인할지, H36M adapter와 A/B/C 실험까지
   포함할지 여부.
