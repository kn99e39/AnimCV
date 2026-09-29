# Worklog — Pre-retargeting Animation Semantics 계약 (2026-09-29)

## 범위와 시작 상태

docs/51-52가 남긴 증거를 **하나의 명시적 pre-retargeting 표현**으로 닫는
contract-closure 배치. Pose 정확도 최적화, contact ablation, yaw 튜닝, global
translation 복원은 하지 않았다. IK / retargeting / foot locking / FBX / camera
motion 이전에서 멈춘다.

- branch: `arch/single_frame_first`
- start HEAD: `e4bf99a` (docs/52 worklog), origin과 동일, worktree clean
  (LabServer63 checkout도 동일 HEAD)
- 코드 commit: `a2efc4e` ("Add pre-retargeting AnimationSemantics sidecar and
  FramePose H0 bridge")
- regime: 전부 `benchmark_detector_observation` (3DPW shipped detector 2D →
  frozen H0)

## 보존

FrameBank, frozen H0, Frame Pose Core, SignState, Pose Reconciliation, docs/47-52,
`src/pose/root_motion.py`, docs/51 contact 구현(`contact.py`), docs/52
`framepose_bridge.py`, `MotionGraph` / `MotionGraphBuilder` / `motion_io.py` —
**기존 파일 수정 0건**. 신규 파일 4개 + 이 worklog만 추가.

## 기존 MotionGraph audit

- `MotionGraph`(Architecture_v2 §5)는 `PoseSequence`(2D landmark, 선택적 depth
  sample)에서 `MotionGraphBuilder`로 만든 **landmark track 컨테이너**다. frame당
  `points{name → position_2d, position_3d, confidence, visible}`, `importance`,
  `locked`, 시퀀스 수준 `fps`, `source_metadata`(free dict)만 갖는다.
- schema/version 필드가 **없다**. `from_dict`는 고정 key를 직접 읽는다.
- 직접 소비자: `retarget/solver.py`, `ik_solver.py`, `fk_solver.py`,
  `temporal_filter.py`, `quality.py`, `app/cli.py`, `ui/gui_app.py`,
  `pose/pose_lifter.py`, `common/coordinates.py`.
- FramePose/H0 경로는 MotionGraph를 **만들지 않는다**. MotionGraph는 FramePose
  이전 시대의 2D 경로 산물이다.

## 선택한 표현: B — AnimationSemantics sidecar

MotionGraph 확장(A)을 택하지 않은 이유:

1. version 필드가 없는 공유 계약을 5개 이상의 retarget 소비자가 직접 읽는다.
   필드를 추가하면 버전 없이 계약이 바뀌고, 과거 `.motion.json`은 새 필드를
   가질 수 없으므로 결국 "없음"을 기본값으로 **발명**해야 한다.
2. MotionGraph의 frame 단위는 2D video frame이고, semantics의 단위는 FrameBank의
   (stride-sampled) sequence row다. 한 구조 안에 넣으면 두 frame 집합의 정렬을
   구조 자체가 떠안게 된다.
3. MotionGraph가 이미 소유한 것(landmark observation track)과 새 semantics는
   겹치지 않는다. 분리하면 owner가 겹치지 않는다.

Sidecar의 결합 방식: `(sequence_id, frame_index, timestamp)`. MotionGraph에는
`motion_graph_reference(semantics)`가 만든 **포인터**(schema, sequence_id,
content_digest, link key)만 `source_metadata`에 넣을 수 있고 어떤 semantic 값도
복사하지 않는다. `check_motion_graph_alignment`가 semantics의 모든 frame이 같은
timestamp의 MotionGraph frame에 대응하는지 검사한다(semantics는 graph의 부분집합일
수 있음). MotionGraph 직렬화는 바이트 수준에서 그대로다.

## Quantities와 owners

`src/motion/animation_semantics.py`, schema `animcv_animation_semantics_v1`.
Owner 표는 `pose/motion_ownership.OWNERSHIP_MODEL`(docs/51)에서 **파생**되어
직렬화되고, load 시 불일치하면 거부한다(두 번째 owner 정의를 만들지 않음).

| 필드 (frame당) | docs/51 quantity | Owner | 표현 |
|---|---|---|---|
| `articulation` | local_articulation | frame_pose | H0 17관절, `camera_root_relative`, m |
| `root_orientation` | body_heading_yaw | root_orientation | `known`/`unknown`, yaw, historical_confidence, yaw_held |
| `foot_motion` | foot_planted_or_moving | contact | 좌/우 `ContactState` contact/moving/unknown |
| `reliability` | observation_reliability | reliability | 관절별 observation_valid, 관절별 in_frame |
| `root_translation` | global_horizontal_translation | root_motion | 항상 `{"status":"unavailable"}` |
| `ground_height` | ground_height_placement | ground_contact | 항상 `{"status":"unavailable"}` |

단일 confidence scalar는 없다. Root orientation의 `historical_confidence`는
estimator 자신의 출력(root orientation 소유)이고, reliability는 구조적 validity만
담는다.

**UNKNOWN과 UNAVAILABLE과 0의 구분**:

- `unknown` — owner와 estimator가 있지만 이 frame에 증거가 없음 (yaw 값 전부 None).
  `known=True, yaw=0.0`과 직렬화에서 구분된다.
- `unavailable` — 현재 어떤 증거원도 그 양을 만들 수 없음 (docs/51 not_observable).
  값 자리가 아예 없다.
- load는 root_translation/ground_height 자리에 **어떤 숫자도**(영벡터 `[0,0,0]`,
  `{"status":"observed","value":[0,0,0]}`, `0.0`, `null` 포함) 거부한다. v1 schema에는
  그 값을 만드는 producer가 없기 때문이다.

## Root Orientation 정책

새 estimator 없음. `pose.root_motion.estimate_root_motion`을 **무수정, 기본값**
(median window 5, `max_yaw_step_degrees` 20)으로 호출한다. provenance에 기록:
policy `historical_root_motion_estimate_segmented_v1`, hold가 historical이라는 것,
docs/52에서 H0에 실질적으로 작동한다고 측정된 것, **20°가 최적이라고 증명되지
않았다**는 것(`max_yaw_step_status: historical_default_not_proven_optimal`),
yaw가 카메라 기준 heading이지 world heading이 아니라는 것.

한 가지 wrapper 규칙: `estimate_root_motion`은 bilateral torso pair가 유효하지 않은
frame이 **하나라도** 있으면 시퀀스 전체에 대해 raise한다. 이를 수정하지 않고,
yaw를 계산할 수 있는 frame의 maximal run마다 독립적으로 호출하고 그 밖의 frame은
`unknown`으로 둔다. 공백이 없으면 전체 시퀀스에 대한 단일 호출과 동일하다(테스트로
고정). 실제 H0 train/validation에는 계산 불가 frame이 0개라 이번 replay에서는 항상
단일 호출이었다.

### docs/52 해석에 대한 정정 — hold에 release가 없다

`_limit_yaw_velocity`는 새 값을 **hold된 출력값**과 비교한다. 그래서 smoothed yaw가
고정된 값에서 20° 넘게 멀어지면, yaw가 우연히 20° 안으로 돌아올 때까지 이후
frame이 전부 hold된다. 이번 replay에서 측정한 결과:

```
held frames 998 / 3,407   (per-sequence mean hold rate 0.2731 — docs/52와 동일)
held runs   23            longest 337 rows (outdoors_parcours_00:actor0, 544 rows 중; ~34 s)
courtyard_dancing_00:actor0  115 rows 중 99 held (단일 run, -168.8°에 고정)
```

docs/52는 hold된 frame 998개 중 978개(~98%)가 "oracle 기준 20° 이하 전이 = 겉보기
flip 방지"라고 해석했다. 이 분류는 **frame 단위 전이 oracle delta**만 봐서 latch
구간의 **오래된 고정값**을 볼 수 없다. 긴 latch 안에서는 oracle 전이가 작아도 hold
값 자체는 실제 heading에서 멀어질 수 있다. 따라서 "98% 정확한 억제"는 이 동작에
대해 실제보다 강한 서술이다. 이번 배치는 정책을 바꾸거나 튜닝하지 않았다. 대신 계약
provenance notes에 명시했다: *held frame은 마지막으로 수용된 yaw이며, 관측된
heading이 아니라 degraded heading으로 취급할 것*. 다음 Root Orientation 작업의 첫
입력으로 남긴다. latched yaw와 oracle 사이의 오차는 이번 배치 범위(target 미사용)
밖이라 측정하지 않았다.

## Contact 보정 정책

- 규칙: docs/51 `classify_foot_contact` 그대로, `rule_version =
  docs51_contact_candidate_v1`, percentile 15/60/40, `min_reliable_run` 2 (상수로 고정,
  sweep 없음).
- 보정: docs/52 T1 절차 그대로 — **H0 TRAIN 34 sequences / 11,334 rows**에서 한 번
  fit, `contact_calibration.json`으로 persist, 이후 모든 시퀀스가 **읽기만** 한다.
  per-video fit이 아니라 model/domain calibration이다. `ContactCalibration`은
  `source.split != "train"`을 생성·load 단계에서 거부한다.
- 재현성: 이번 fit 결과가 docs/52 report의 T1 threshold와 **float 단위로 정확히
  일치**했다(`contact_calibration_reproduces_docs52_t1: true`). 스크립트는 불일치하면
  중단하도록 되어 있다.

```
left:  contact 0.19830 m/s  moving 0.72790 m/s  height_std 0.005933 m
right: contact 0.20539 m/s  moving 0.72653 m/s  height_std 0.006572 m
provenance: bank 75519e63…, H0 train sha256 43a5d53b…, regime benchmark_detector_observation
```

- 의미: 직렬화된 `interpretation`은 "contact-like stance/swing evidence; 물리적
  foot-ground contact으로 증명되지 않음; KINEMATIC_PROXY로만 검증"이다.

### 새로 드러난 한계 — sampling domain 불일치 (수정하지 않음)

FrameBank는 split마다 stride가 다르다: **train rows = 원본 2 frame 간격, validation =
3 frame 간격**(둘 다 `fps=30`). `classify_foot_contact`는 timestamp를 쓰지 않고
**인접 row를 `1/source_fps` 간격으로 가정**해 속도를 계산한다. 그래서 T1 threshold는
stride 2 row에서 fit되었고, docs/52의 T1 validation 수치(balanced accuracy
0.771/0.840)는 stride 3 row에서 측정되었다. 이번 배치는 규칙이나 threshold를 바꾸지
않았다(튜닝 금지). 대신 계약에 기록한다:

- calibration `sampling = {rule_time_basis: row_neighbours_at_source_fps, source_fps:
  30, median_row_stride_frames: 2}`
- 각 시퀀스 provenance에 자신의 stride와 `sampling_matches_calibration`
- 결과: validation 16개 시퀀스 모두 `sampling_matches_calibration = false`

Contact state 값은 docs/52와 같은 조건에서 나온 것이며, 이 flag가 downstream에게
"보정 domain과 row 간격이 다르다"는 사실을 숨기지 않는다. 해결(timestamp 기반 속도,
또는 stride를 맞춘 재보정)은 새 contact 실험이므로 범위 밖이다.

## Unavailable quantities

`root_translation`, `ground_height`: 모든 frame에서 `{"status": "unavailable"}`.
영벡터 없음, pelvis-relative pseudo-translation 없음, ankle에서 유도한 ground height
없음, smoothing fallback 없음. 이유 문자열은 sequence 수준
`unavailable_quantities`에 한 번 직렬화된다.

## Reliability

`joint_observation_valid` = `bank.input_valid AND finite(H0)` — 모든 semantic 계산을
gate한 바로 그 validity. `joint_in_frame` = 기존
`framepose.nonlinear_reconstruction.in_frame_mask(input_2d, input_valid)`(구조적,
occlusion 추론 아님). Provenance에는 sample의 `ObservationProvenance`와 regime을
그대로 담는다. `learned_reliability: null`이다 — Qwen VLM reliability(docs/50 기각)는
싣지 않는다. 테스트로 확인: head/wrist validity를 바꾸면 reliability만 바뀌고 root
orientation / foot motion은 바뀌지 않는다.

## FramePose → AnimationSemantics bridge

`src/motion/animation_semantics_bridge.py`:

- `calibrate_contact_from_h0_train(bank, h0, identity)` — TRAIN sequence만 사용.
- `build_animation_semantics(bank, h0, identity, sequence_id, calibration)` — docs/52
  `build_h0_lifted_sequence`(input_2d / input_valid / H0만 읽음) 위에서 한 시퀀스의
  row만 사용한다. frame_index와 timestamp는 bank sample 값을 그대로 옮긴다.
- `target_3d` / `target_valid`는 읽지 않는다. target을 오염시켜도 출력이 동일함을
  테스트로 확인했다(bank content digest는 target을 포함하므로 identity는 고정해서
  비교).
- 결정론: 같은 입력을 두 번 build하면 content digest가 동일하다(실데이터 16/16).

## 실데이터 replay (frozen H0 validation, `benchmark_detector_observation`)

`scripts/run_animation_semantics_replay.py`, LabServer63 `animcv-framepose:cuda118`,
commit `a2efc4e` checkout. Artifact: `~/animcv-output/animation_semantics_h0_replay/`
(`report.json` sha256 `130bdc77…e057`, `contact_calibration.json`,
`sequences/*.semantics.json` 16개). H0 identity 확인: bank `75519e63…`, H0 train
`43a5d53b…`, validation `88a007fc…` (docs/52와 동일).

```
sequences 16, frames 3,407
root orientation   known 3,407 / unknown 0      (held 998, 23 runs, longest 337)
left foot          CONTACT 164   MOVING 2,113   UNKNOWN 1,130
right foot         CONTACT 167   MOVING 2,017   UNKNOWN 1,223
observation        frames with ≥1 invalid joint 1,310; invalid joints 2,761; out-of-frame joints 2,802
root_translation   unavailable 3,407 / 3,407
ground_height      unavailable 3,407 / 3,407
deterministic replay 16/16, serialization round-trip identity 16/16
sampling_matches_calibration 0/16  (validation stride 3 vs calibration stride 2)
```

CONTACT이 전체의 ~5%뿐인 것은 이 subset(parcours, crosscountry, jumpBench 등
동적 시퀀스 비중이 큼)과 엄격한 T1 기준 때문이다. 이 수치는 계약 accounting이지
contact 품질 지표가 아니다.

## Owner-case replay

docs/52 review manifest의 frame을 그대로 쓰고, walking만 T1 semantics에서 다시
선택했다(docs/52의 walking은 T0 상태 기준, 교대 1회였음).

| Case | sequence#frame | Root Orientation | L / R | valid / in-frame | 비고 |
|---|---|---|---|---|---|
| known-good pose | courtyard_dancing_00:a0#424 | -168.8° **held** | contact / contact | 17 / 17 | 자세는 가장 정확하지만 heading은 115 rows 중 99 rows가 latch |
| turning | outdoors_crosscountry_00:a0#202 | -6.5° **held** | moving / unknown | 12 / 12 | 가장 큰 raw H0 yaw 변화(84.6°)가 hold에 흡수됨; 참 회전인지 flip인지는 이 계약이 판단하지 않음 |
| ankle failure | outdoors_parcours_00:a0#384 | 43.0° | moving / moving | 15 / 15 | 최대 ankle 오차 frame에서도 contact은 발명되지 않음 (시퀀스 전체 CONTACT 0) |
| tracking loss | courtyard_hug_00:a1#312 | 142.0° | unknown / unknown | 15 / 15 | invalid run 안에서 foot state가 UNKNOWN으로 거부됨 |
| walking (docs/52 T0 선택) | courtyard_rangeOfMotions_01:a0#234 | 146.2° | unknown / unknown | 17 / 17 | T1에서 시퀀스 교대 L2 / R2 |
| walking (T1 선택, 최다 교대) | courtyard_drinking_00:a1#6 | -164.4° | unknown / moving | 17 / 17 | 직전 2 rows contact/contact → moving; 교대 L8 / R6 |

모든 case에서 root_translation / ground_height는 `unavailable`이다.

## MotionGraph 호환성

MotionGraph, builder, `motion_io`는 수정하지 않았다. 과거 MotionGraph dict가 그대로
load·재직렬화되고(테스트), semantics 포인터를 `source_metadata`에 넣어도 top-level
key 집합이 `{fps, source_metadata, frames, tracks}`로 유지된다. 공유 직렬화 계약을
바꾸지 않았으므로 full regression은 규칙에 따라 생략했다.

## 변경 파일

```
src/motion/animation_semantics.py          (신규: 데이터 모델, 직렬화, MotionGraph 링크)
src/motion/animation_semantics_bridge.py   (신규: H0 → semantics, TRAIN-only 보정)
scripts/run_animation_semantics_replay.py  (신규: 실데이터 계약 replay)
tests/test_animation_semantics.py          (신규: 25 tests)
docs/53_WORKLOG_ANIMATION_SEMANTICS_CONTRACT.md
```

## 테스트

```
pytest tests/test_animation_semantics.py tests/test_motion_graph.py tests/test_motion_builder.py \
  tests/test_framepose_bridge.py tests/test_contact_failure_attribution.py tests/test_h0_replay_contracts.py \
  tests/test_motion_ownership.py tests/test_contact.py tests/test_contact_reference.py \
  tests/test_root_orientation_diagnostic.py tests/test_root_translation_control.py -q
83 passed   (Mac local, 및 LabServer63 animcv-framepose:cuda118에서 동일)
```

신규 25개가 확인하는 것: target leakage 없음, sequence-local, frame/timestamp 보존,
articulation = H0 무변경, 결정론, historical estimator 동일성과 provenance, torso
증거 없는 frame은 UNKNOWN(run 분할), UNKNOWN과 yaw 0의 구분, contact = docs/51 규칙과
보정의 정확한 결과, ContactState 값 집합, TRAIN-only 보정과 provenance(validation을
바꿔도 threshold 불변), non-train/다른 rule version 거부, sampling 불일치 기록,
unavailable 두 양의 전 frame 직렬화, 숫자/영벡터 load 거부(5 variants × 2 quantities),
reliability 분리, ownership 표 = docs/51 강제, round trip identity, frame 순서 강제,
MotionGraph 포인터 링크 / 정렬 검사 / 과거 graph load.

## 완료 조건에 대한 답

> "Does AnimCV now have one explicit, reproducible, pre-retargeting Animation
> Semantics representation containing every semantic quantity that current
> evidence supports, while leaving unsupported world-motion quantities
> explicitly UNKNOWN?"

**예 — 단, 아래 두 caveat가 계약 안에 명시된 상태로.** `AnimationSemantics` v1은
local articulation, camera 기준 root yaw(known/unknown + hold 상태), 좌/우 contact-like
stance/swing 상태, 구조적 reliability를 owner별로 분리해 담는다. Global root
translation과 ground height는 모든 frame에서 `unavailable`이며 숫자로 load될 수
없다. H0 validation 16개 시퀀스에서 결정론적이고 round trip이 정확하다.

Caveat(해결하지 않고 기록만):

1. **Historical yaw hold는 release가 없다.** Held frame(29%)은 오래된 값일 수 있다.
   계약은 이를 `yaw_held`와 provenance notes로 드러낸다. docs/52의 "98%" 해석을
   이 worklog에서 정정한다.
2. **Contact 보정의 row stride(2)와 validation row stride(3)가 다르다.** 모든
   validation 시퀀스가 `sampling_matches_calibration=false`로 표시된다.

Root Motion을 해결했다고 주장하지 않는다. 실제 ground contact을 해결했다고
주장하지 않는다. IK, retargeting, foot locking, camera-motion 복원으로 진행하지
않았다.
