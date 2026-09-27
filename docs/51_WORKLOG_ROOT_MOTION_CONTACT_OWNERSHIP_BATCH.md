# Worklog — Root Motion / Foot Contact ownership 배치 (2026-09-27)

## 범위와 시작 상태

`arch/single_frame_first`에서 Frame Pose reconstruction 배치(C1/C2/C3 temporal
refiner, VLM reliability, nonlinear occlusion reconstruction, Sign Advisor,
Pose Reconciliation) 이후 첫 Animation Semantics 배치. 목표는 완성된 애니메이션
솔버가 아니라 **architecture decision**: Root Motion과 Foot Contact 중 어떤
부분이 현재 Frame Pose 표현으로부터 실제로 복원 가능한지, 어떤 부분이 UNKNOWN으로
남아야 하는지 결정하는 것.

- branch: `arch/single_frame_first`
- start HEAD: `8c7bddcdb3e1128bcd868e1ccd78542f2ab52ea6` ("Bound VLM reliability
  and nonlinear reconstruction")
- start remote: origin과 동일, worktree clean
- 이번 세션 commit: `b686e2cb595a711f8bf9620c8cde9d421b4eb65c` ("Define Root
  Motion / Foot Contact ownership and bounded diagnostics")

위 배치들(temporal refiner, VLM reliability, nonlinear reconstruction, Sign
Advisor, Pose Reconciliation)은 재개하지 않았고 그 historical 결과(docs/44-50)도
수정하지 않았다.

## 보존한 것들

- `src/pose/root_motion.py`는 **전혀 수정하지 않았다**. 아래 "Historical
  root_motion.py 가정" 절에서 그 알고리즘을 문서화만 했다.
- Frozen FramePose/H0, SignState, Pose Reconciliation, 기존 reliability
  metadata(`observation_valid`, `in_frame_mask`, `ObservationProvenance`),
  canonical coordinates — 전부 untouched. 이번 배치가 새로 추가한 6개 모듈은
  전부 `src/pose/` 아래 독립 파일이며 기존 파일을 import는 하지만 수정하지는
  않는다.
- FramePose, VLM, reconciliation, temporal refiner, occlusion reconstruction —
  튜닝하지 않았다.

## Historical root_motion.py 가정 (섹션 2)

`root_motion.estimate_root_motion`은 다음을 가정한다:

1. **Bilateral torso yaw** — `left_shoulder`/`right_shoulder`와
   `left_hip`/`right_hip`의 2D(x,y) 축 벡터를 각각의
   `length × observation_confidence × validity`로 가중한 원형 평균으로 yaw를
   추정한다 (root_motion.py:144-180).
2. **Confidence weighting** — 최종 confidence는 yaw 자체가 아니라 각 source의
   detector confidence를 weight로 가중평균한 값이다. Shoulder/hip 간
   disagreement는 별도 신호(`yaw_agreement_degrees`)로 분리되어 있고, 걷기/torso
   twist에서 자연스럽게 발생하므로 confidence 저하로 취급하지 않는다
   (root_motion.py:175-179의 명시적 주석).
3. **Median smoothing** — unwrap 이후 sliding-window median (기본 window=5).
4. **Max-yaw-step hold** — 이전 출력 대비 `max_yaw_step_degrees`(기본 20°)를
   넘는 변화는 "monocular lifter axis flip"으로 간주해 이전 값을 유지하고
   confidence를 25%로 깎는다(root_motion.py:202-219).
5. **`root_translation=None`** — 모든 프레임에서 예외 없이 `None`으로
   고정되어 있다(root_motion.py:109). 이를 우회하는 코드 경로는 없다.
   `RootMotionSequence.translation_observable`도 기본값 `False`를 그대로
   유지한다.

## Ownership 모델과 observability audit (섹션 3, 4)

`src/pose/motion_ownership.py`에 6개 quantity를 코드로 고정했다 (재사용
가능한 data, 알고리즘 아님):

| Quantity | Owner | Observability |
|---|---|---|
| local_articulation | Frame Pose | directly_observable |
| body_heading_yaw | Root Orientation | weakly_inferable |
| global_horizontal_translation | Root Motion | **not_observable** |
| ground_height_placement | Ground/Contact | **not_observable** |
| foot_planted_or_moving | Contact | only_temporally_inferable |
| observation_reliability | Reliability metadata | directly_observable |

핵심 이유:

- **global_horizontal_translation이 not_observable인 이유**: root-relative
  표현은 pelvis가 매 프레임 원점에 고정되므로 그 자체로 global translation을
  내포할 수 없고, monocular RGB에는 camera-motion 추정치가 없다. Planted-foot
  같은 외부 제약이 있을 때만 조건부로 복원 가능(섹션 8/9 실험 참고).
- **ground_height_placement이 not_observable인 이유**: 3DPW/MPI-INF-3DHP/
  AMASS 중 어느 것도 Frame Pose 좌표계에서 ground plane/floor height를
  노출하지 않는다 (adapter 3개 전수 조사, 아래 참고). Ankle height만으로
  ground를 추정하지 말 것.
- **foot_planted_or_moving이 only_temporally_inferable인 이유**: 단일 프레임의
  ankle 위치는 아무 정보가 없고, 짧은 window의 속도/높이 일관성만 의미가 있다.

## Root Orientation 진단 (섹션 5)

`src/pose/root_orientation_diagnostic.py`는 `root_motion.py`의 private
helper를 import하지 않고 canonical_pose.YAW_PAIRS만으로 독립적으로 raw yaw를
재계산한다. `observation_valid`를 weight에서 빼지 않아(=production과 다르게)
unreliable 프레임에서도 yaw를 계산하고, reliability는 별도 필드로만 기록한다
— 그래야 "reliability-conditioned failure rate"를 측정할 수 있다(production
방식으로는 unreliable 프레임이 그냥 UNKNOWN이 되어 측정 자체가 불가능해짐을
draft 단계에서 확인하고 설계를 수정함).

**3DPW validation (12 sequences, 16 actors, 10,858 frames, `oracle_geometry`
regime)** 실측 결과:

```
unknown_rate: 0.0
total_flips (>90deg raw yaw jump): 0
mean_flip_rate_among_unreliable_frames: 0.0
mean_flip_rate_among_reliable_frames: 0.0
mean_historical_root_motion_yaw_held_rate (production estimate_root_motion): 0.0
```

**해석 — 과대 해석 금지**: 3DPW의 `jointPositions`는 SMPL GT이므로 이 배치는
전부 `oracle_geometry` regime이다. GT 위치는 애초에 monocular lifter의 axis
flip이나 detector 노이즈를 포함하지 않으므로, `max_yaw_step_degrees` hold가
0%로 나온 것은 "hold heuristic이 필요 없다"는 증거가 **아니라** "이 오라클
데이터에는 hold가 다루려는 실패 모드 자체가 존재하지 않는다"는 것이다. 이
heuristic이 `real_animcv_observation`(MMPose+RTMDet) regime에서도 여전히
필요한지는 이번 배치 범위 밖이며, 그 데이터로 다시 측정해야 한다. 이는 완료
조건에서 요구하는 "결과를 실제보다 강하게 주장하지 않는다"는 원칙을 지키기
위해 명시한다.

Bilateral disagreement(shoulders vs hips)는 평균 6.5~7.9°로 계속 존재하지만
confidence와는 별개로 잘 분리되어 기록된다 (root_motion.py가 이미 그렇게
설계됨을 재확인).

## Contact candidate 대 reference (섹션 6, 7)

**Reference 재고 조사 (섹션 7)**: 3DPW, MPI-INF-3DHP, AMASS 세 adapter 전부
foot-contact GT 필드를 파싱하지 않는다(전수 코드 조사 완료). 따라서 이번
배치가 만든 모든 reference는 `KINEMATIC_PROXY`이며 `EXACT_LABEL`이 아니다.
`src/pose/contact_reference.py`가 이 구분을 코드/문서로 강제한다.

**Contact threshold — TRAIN에서만 적합 (섹션 6)**: 3DPW 공식 train split(24
sequences)의 world-frame ankle 속도 분포에서 fit:

```
left:  contact_speed=0.0629 m/s, moving_speed=0.3072 m/s, height_std=0.00137 m
right: contact_speed=0.0604 m/s, moving_speed=0.2933 m/s, height_std=0.00163 m
reference(proxy) speed threshold (world-frame, 20th pct of TRAIN): 0.1693 m/s
```

**3DPW validation 전체 집계 (16 actors, 10,842 frames after alignment)**:

```
candidate counts (left):  CONTACT=1260  MOVING=5815  UNKNOWN=3783 (34.9%)
candidate counts (right): CONTACT=1158  MOVING=5883  UNKNOWN=3817 (35.2%)
aggregate confusion vs. KINEMATIC_PROXY reference:
  left:  precision=0.684  recall=0.880  specificity=0.935  balanced_accuracy=0.908
  right: precision=0.667  recall=0.859  specificity=0.937  balanced_accuracy=0.898
```

이 숫자는 **"candidate(root-relative kinematics) vs. world-frame proxy 간의
internal-consistency"**이지 진짜 contact accuracy가 아니다(스크립트가 매 결과에
이 caveat 문자열을 함께 기록한다). Balanced accuracy가 높게 나온 것은 candidate
신호가 world-frame 사실과 상당히 잘 상관된다는 뜻이지, "진짜 contact"을 측정했다는
뜻이 아니다. Precision(~0.67-0.68) < recall(~0.86-0.88)은 candidate가 reference보다
CONTACT을 더 자주 부른다는 뜻 — pelvis 대비 상대속도가 낮아도 몸 전체가 이동 중이면
world 기준으로는 이미 MOVING인 경우가 있기 때문으로 해석된다. 이는 정확히
motion_ownership.py가 예견한 문제(contact 판정이 root motion을 이미 알아야
완전해진다는 순환성)의 실측 증거다.

UNKNOWN 비율이 35% 전후로 높은 것은 height_std threshold가 TRAIN 분포 40th
percentile 기준 밀리미터 단위(1.4~1.6mm)로 매우 엄격하게 fit되었기 때문이다.
Coverage보다 안전(false CONTACT 방지)을 우선한 결과이며, 이는 섹션 10의
"증거 불충분 시 UNKNOWN" 요구와 일치한다.

## Root Translation control 실험 (섹션 8, 9)

`src/pose/oracle_world_reference.py` + `root_translation_control.py`가 두
조건을 비교한다: **world_oracle**(GT world-frame ankle-pelvis 관계, camera
motion에 면역) vs. **naive_camera_frame**(실제 Frame Pose가 출력하는 것과 같은
camera-anchored root-relative ankle 위치에 동일한 identity를 순진하게 적용).

**3DPW (moving camera, 실제 handheld 카메라 rotation 포함)**:

```
recoverable segments (both-frame CONTACT by proxy): left 1,901 / 10,842 (17.5%)
                                                     right 1,822 / 10,842 (16.8%)
world_oracle_mean_error:  left 0.0034 m   right 0.0032 m
naive_camera_mean_error:  left 0.0074 m   right 0.0054 m   (2.2x / 1.7x world_oracle)
mean correlation(naive_error, camera_rotation_delta_degrees): left 0.566  right 0.503
```

**MPI-INF-3DHP (static camera contrast, S1-S4, Seq1, camera 0, 1000 frames
each)**:

```
naive_camera_mean_error == world_oracle_mean_error, 비트 단위로 동일
(S1: 0.00228 m, S2: 0.00154-0.00214 m, S3: 0.00293-0.00297 m, S4: 0.00195-0.00200 m)
```

**해석**: static camera 조건에서는 naive와 oracle이 정확히 일치한다 — 카메라가
움직이지 않으면 camera-anchored root-relative 표현이 곧 world-frame 표현과
같은 identity이기 때문에 당연한 결과이며, 동시에 identity 자체의 구현이
정확하다는 검증이기도 하다. Moving camera(3DPW) 조건에서는 naive 오차가
oracle 대비 1.7~2.2배로 커지고, 그 초과 오차가 camera rotation 크기와 중간
정도 양의 상관(0.50~0.57)을 보인다 — **camera가 회전하면 순진한 방법은
측정 가능하게 나빠진다**는 것을 smoothing으로 가리지 않고 직접 수치로
보여준다(섹션 9 요구사항).

Residual world_oracle error(~0.002-0.004m, 0이 아님)는 KINEMATIC_PROXY
reference의 속도 threshold가 정확히 0이 아니라서 "CONTACT"으로 분류된
프레임에도 실제로는 약간의 world 이동이 남아있기 때문이다 — 실제 보행에서
지지발도 완전히 고정되지 않는 것과 일치하며, 버그가 아니라 proxy의 본질적
한계다.

**핵심 답**: planted-foot 기하학은 **작동한다** — 하지만 오직 "양쪽 프레임 모두
proxy가 CONTACT으로 판정한" 프레임의 ~17-18%에서만, 그리고 카메라가 정지해
있거나 회전을 별도로 알고 있을 때만. 나머지 ~82%는 "두 발 다 스윙 중"이거나
"신뢰 가능한 contact 증거가 없는" 구간이며 이번 배치는 그 구간의 global
translation을 UNKNOWN으로 남긴다 — 발명하지 않는다.

## Camera motion 한계 (섹션 9, 명시)

현재 AnimCV 파이프라인에는 camera pose 추정이 전혀 없다. 위 실험은 3DPW의
GT `cam_poses`를 빌려 camera rotation 크기를 사후에 측정했을 뿐이며, 실제
production Frame Pose는 이 정보에 접근할 수 없다. 따라서 naive_camera_frame
조건이 이번 배치가 측정한 것보다 **더 나쁠 수 있다** — 3DPW가 상대적으로
완만한 handheld 흔들림이기 때문이다. 이것이 이번 배치가 global translation을
"not_observable"로 분류한 근거이지, "관찰했지만 나빴다"는 것이 아니다.

## Reliability (섹션 10)

이번 배치는 `observation_valid`(3DPW의 `campose_valid`)와 in-frame 구조적
validity만 사용했다. Qwen VLM reliability(`reliability_advisor.py`,
`sign_advisor.py`)는 docs/50에서 이미 "D — BOTH INSUFFICIENT" 판정을 받았고
이번 배치에서도 다시 사용하지 않았다. Contact candidate와 root translation
control 둘 다 증거 부족 시 UNKNOWN을 반환하도록 설계·테스트되어 있다
(`tests/test_contact.py::test_unreliable_observation_refuses_to_unknown...`,
`tests/test_root_translation_control.py::test_unplanted_or_invalid_frames_do_not_invent_a_translation`).

## Qualitative review package (섹션 13)

`~/animcv-output/root_motion_contact_batch/review/`에 7개 정지 프레임 +
`manifest.json`. 카테고리는 hand-pick이 아니라 이번 배치가 만든 진단 수치로
선택했다(스크립트가 근거를 `notes`에 기록):

| Category | Sequence:actor#frame | 근거 |
|---|---|---|
| standing | courtyard_drinking_00:actor0#344 | 최저 평균 ankle-relative speed (0.235 m/s) |
| walking | courtyard_drinking_00:actor1#344 | 최다 CONTACT↔MOVING alternation (18회) |
| turning | outdoors_parcours_01:actor0#253 | 최대 reliable frame-to-frame yaw 변화 (66.5°) |
| tracking_loss | courtyard_rangeOfMotions_01:actor0#351 | 최장 camera-pose-invalid run (105 frames) |
| moving_camera_failure | downtown_walkDownhill_00:actor0#229 | 최대 naive 오차(0.079m) @ rotation delta 0.46° |
| ankle_short_dropout | courtyard_basketball_01:actor0#498 | 짧은(1-frame) validity dropout |
| foot_lift | — | **찾지 못함** (아래 한계 참고) |

**한계 — 솔직히 기록**: (1) "foot_lift"(CONTACT 직후 바로 MOVING으로 전환되는
프레임)는 스캔한 16 actor 어디에서도 조건을 만족하는 프레임이 없었다 —
`_suppress_short_runs`가 짧은 전환 프레임을 UNKNOWN으로 흡수하기 때문에 직접
인접 전환이 드문 것으로 보인다. (2) "sitting"은 현재 kinematic 신호만으로는
"standing"과 구분할 방법이 없어 별도 카테고리로 선택하지 않았고, 대신
ankle_short_dropout으로 대체했다 — 없는 신호를 있는 것처럼 꾸미지 않기 위함.

## 변경 파일 (섹션 16)

신규 파일만 추가, 기존 파일은 0건 수정:

```
src/pose/motion_ownership.py
src/pose/contact.py
src/pose/contact_reference.py
src/pose/root_orientation_diagnostic.py
src/pose/oracle_world_reference.py
src/pose/root_translation_control.py
scripts/run_root_motion_contact_diagnostics.py
tests/test_motion_ownership.py
tests/test_contact.py
tests/test_contact_reference.py
tests/test_root_orientation_diagnostic.py
tests/test_root_translation_control.py
docs/51_WORKLOG_ROOT_MOTION_CONTACT_OWNERSHIP_BATCH.md (this file)
```

## 테스트 결과

Focused tests만 실행(공유 계약을 수정하지 않았으므로 섹션 15 규칙에 따라 full
regression은 생략):

```
$ pytest tests/test_motion_ownership.py tests/test_contact.py \
  tests/test_contact_reference.py tests/test_root_orientation_diagnostic.py \
  tests/test_root_translation_control.py -q
30 passed
```

로컬(Mac)과 `animcv-framepose:cuda118` 컨테이너(LabServer63) 양쪽에서 동일하게
30 passed 확인. 실데이터 실행은 `animcv-framepose:cuda118`에서 3DPW raw
(`/home/nd/animcv-data/3dpw/raw/DATASET_Motion`) + MPI-INF-3DHP
(`/home/nd/AnimCV/datasets/mpi_inf_3dhp`, repo에 내장된 annotation)로 수행,
report는 `~/animcv-output/root_motion_contact_batch/report.json`.

## 최종 architecture 결론 (완료 조건에 대한 답)

> "Given AnimCV's current Frame Pose representation, which parts of Root
> Motion and Foot Contact are actually recoverable from available evidence,
> and which must remain UNKNOWN or wait for a later source of information?"

1. **Local articulation** — 이미 Frame Pose Core(Layer A)가 owner. 이번
   배치는 건드리지 않았다.
2. **Body heading (yaw)** — 현재 bilateral shoulder/hip evidence로
   `oracle_geometry` regime에서는 완전히 안정적(flip 0, hold 0%)이다. 하지만
   이는 GT 데이터의 특성이지 production 검증이 아니다 — `root_motion.py`의
   hold heuristic이 `real_animcv_observation`에서도 필요한지는 **다음 배치로
   유보**해야 한다.
3. **Global horizontal translation (Root Motion)** — **조건부로만
   recoverable**: 두 프레임 모두 발이 planted라고 판단될 때만(전체 프레임의
   ~17-18%), 그리고 camera가 정지해 있거나 그 rotation을 알 때만. 나머지는
   현재 증거로는 **not_observable**이며 UNKNOWN으로 남아야 한다 — 발명하지
   않는다.
4. **Ground height placement** — 어떤 wired dataset도 floor reference를
   주지 않으므로 **완전히 not_observable**. 다음 정보원(캘리브레이션된 ground
   plane 등)을 기다려야 한다.
5. **Foot contact (CONTACT/MOVING/UNKNOWN)** — root-relative kinematics만으로
   candidate 신호를 만들 수 있고 world-frame proxy와 balanced accuracy
   ~0.90-0.91로 상관되지만, 이것은 "stance/swing phase" 신호에 가깝고 진짜
   ground-planted state는 아니다(그 자체가 아직 모르는 root motion에
   의존하는 순환성이 있음). Contact는 **only_temporally_inferable**로
   유지한다.
6. **Reliability metadata** — 이미 owner가 명확(existing `observation_valid`
   등). VLM reliability는 여전히 채택하지 않는다.

이 배치는 architecture decision이며 완성된 solver가 아니다. Foot locking,
IK correction, retargeting으로 자동 진행하지 않는다.
