# Worklog — Root Orientation/Contact 신호가 frozen FramePose H0에서도 살아남는가 (2026-09-27)

## 범위와 시작 상태

docs/51이 `oracle_geometry`(3DPW GT joint position)로 확립한 Root Orientation/
Contact ownership 모델과 진단 수치가, AnimCV의 실제 frozen FramePose 출력
(`benchmark_detector_observation` regime, H0)을 통과해도 살아남는지 검증하는
production-evidence replay 배치. **새 알고리즘을 설계하지 않았다** — docs/51의
`root_orientation_diagnostic.py`, `pose.root_motion.estimate_root_motion`,
`contact.py`, `contact_reference.py`, `root_translation_control.py`를 전부
그대로 import해서 입력만 오라클 GT 대신 H0로 바꿔 재실행했다.

- branch: `arch/single_frame_first`
- start HEAD: `e22bd03c7181dce80f5e77356937086ff09da81b` (docs/51 worklog)
- 이번 세션 commit: `09f4e59eb69bd4c84b8b9d7beb8493be50e1e39c`,
  `750c2f2...`(R1_T1 보강)
- git diff 확인: 이번 세션에서 수정된 기존 파일은 **0건**, 전부 신규 파일
  추가(`git status --short` / `git diff --stat` 둘 다 세션 시작·종료 시점에
  확인).

## 보존 확인 (섹션 1)

FrameBank, frozen H0/O_BILATERAL lineage, Frame Pose Core, SignState, Pose
Reconciliation, docs/47-51 결과, historical `src/pose/root_motion.py`,
docs/51 ownership 모델, ContactState(CONTACT/MOVING/UNKNOWN) — 전부 코드
1바이트도 수정하지 않았다. 신규 파일만 추가했다:

```
src/pose/framepose_bridge.py            (FrameBank+H0 -> LiftedPoseSequence, 이전에 존재하지 않던 다리)
src/pose/contact_failure_attribution.py (O-vs-H0 불일치 원인 5분류)
scripts/run_root_motion_contact_h0_replay.py (orchestrator)
tests/test_framepose_bridge.py
tests/test_contact_failure_attribution.py
tests/test_h0_replay_contracts.py
docs/52_WORKLOG_ROOT_MOTION_CONTACT_H0_PRODUCTION_REPLAY.md (this file)
```

**Target leakage 방지**: `framepose_bridge.py`는 `bank.arrays["input_2d"]`,
`["input_valid"]`, 그리고 호출자가 넘긴 H0 배열만 읽고 `["target_3d"]`,
`["target_valid"]`는 절대 읽지 않는다. `tests/test_framepose_bridge.py::
test_build_h0_lifted_sequence_never_reflects_target_arrays`가 target_valid를
input_valid의 **반대값**으로, target_3d를 999999.0 sentinel로 오염시킨 뒤 이를
검증한다.

## 정확한 frozen H0 identity (섹션 2, 13)

```
bank:            /home/nd/animcv-output/framepose/bank_3dpw_paired_v2.json (+.npz)
bank_content_digest: 75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536
regime:          benchmark_detector_observation (3DPW 자체 OpenPose-format 2D 검출)
split counts:    train 11,334 / validation 3,407 / test 7,076
checkpoint:      sign_attr_v2/O_BILATERAL/checkpoint.pt (candidate name
                 "O_BILATERAL_oracle_forward_depth_only")
H0 artifact:     /home/nd/animcv-output/framepose/context_h0_v2/
  prediction_train.npy       sha256 43a5d53bee99eed6159b2a83107235632add91912cb985a9eae0900db65c6af6
  prediction_validation.npy  sha256 88a007fca42eee08833ec8ece453aa3160a26232738be01cd849cf5016524937
  prediction_test.npy        sha256 6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5
```

`prediction_test.npy`의 SHA는 `scripts/run_context_refiner.py`의
`FIXED_H0_TEST_SHA256`와 **바이트 단위로 일치** — 이 배치가 사용한 H0는 이미
frozen으로 검증된 바로 그 lineage이며, 새로 추론을 돌리지 않았다
(`--expect-bank-content-digest`로 매 실행 시 재확인).

**Bank는 3DPW 전체 프레임이 아니라 stride-sampled RGB-paired subset**이다
(docs/51의 12개 sequence, 16 actor는 bank validation split에 전부 있지만
frame 수는 3,407로 docs/51의 10,858보다 적다). Oracle과 H0을 공정하게
비교하기 위해, 오라클 쪽도 **bank가 실제로 샘플링한 frame_index 집합에만
맞춰 재구성**했다(`_matched_oracle_sequence`) — 그래야 신호 차이가 H0 품질
때문인지 샘플링 간격 때문인지 뒤섞이지 않는다.

## Root Orientation replay (섹션 3)

`observe_yaw`/`summarize`를 오라클(matched subset)과 H0 양쪽에 **변경 없이**
적용:

```
                         oracle      H0
unknown_rate             0.0         0.0
total_flips (>90°)       1           3
total frames             3,407       3,407
mean_yaw_error_degrees (H0 vs oracle, same-frame)   6.47°
total_yaw_sign_flip_count (>90° H0-vs-oracle delta)  1
```

**해석**: H0은 bilateral shoulder/hip evidence를 오라클과 거의 동일한 빈도로
안정적으로 만들어낸다. Flip count와 unknown rate 모두 낮고 거의 같으며, 평균
yaw 오차 6.5°는 몸통 방향 추정치로는 작다. **Body-heading evidence는 H0에서도
usable 하게 살아남는다.**

## Historical hold를 H0에 diagnostic으로만 재생 (섹션 4)

`smoothing_window`, `max_yaw_step_degrees` **수정하지 않고** 기본값 그대로
`estimate_root_motion`을 H0에 적용:

```
mean_h0_historical_hold_rate:            27.3%   (docs/51 oracle: 0.0%)
hold-triggered frames classified against the SAME-transition oracle delta:
  suppressed a true rotation (oracle delta > 20°): 20   (~2%)
  prevented an apparent flip (oracle delta <= 20°): 978  (~98%)
```

**해석 — 이게 이번 배치의 핵심 발견 중 하나**: docs/51은 oracle GT에서 hold가
0% 발동한다고 보고했고, 그때 이미 "이건 hold가 불필요하다는 증거가 아니라
오라클 데이터에 hold가 다루는 실패 모드 자체가 없다는 뜻"이라고 명시적으로
경고했다. 이번 배치가 정확히 그 유보를 확인했다: **실제 H0에서는 hold가
27.3%의 프레임에서 발동**하고, 그 발동 중 98%는 오라클 기준 실제로 작은 회전
구간(스퓨리어스 H0 플립)을 정확히 잡아낸 것이었고, 참 회전을 억누른 경우는
2%뿐이다. `max_yaw_step_degrees=20°` heuristic은 **production 조건에서 결코
no-op이 아니며, 그 기본값은 지금 측정 기준으로 유용하게 작동하고 있다.**

## Contact replay: T0 대 T1 (섹션 5, 6)

**T0 — docs/51 오라클/TRAIN 문턱값을 H0에 무변경 적용**:

```
                    left                right
H0_T0 recall        0.0                 0.0
H0_T0 candidate CONTACT count: 2        0     (reference CONTACT count: 132 / 160)
```

**T0 transfer는 사실상 완전히 실패한다** — H0에 오라클 기준 문턱값을 그대로
적용하면 CONTACT을 거의 전혀 예측하지 못한다(3,407프레임 중 2회, 0회). H0의
관절 위치가 오라클보다 훨씬 더 "빠르게" (root-relative 속도가 노이즈로 인해
체계적으로 높게) 보이기 때문에, 오라클 기준 "느림=CONTACT" 문턱값이 H0에서는
거의 항상 넘겨지지 않는다.

**T1 — 동일한 규칙(`fit_contact_thresholds`, percentile 15/60/40 **무변경**)을
H0 TRAIN에 재적합**:

```
                              left              right
T0 (oracle) contact_speed     0.0629 m/s        0.0604 m/s
T1 (H0)     contact_speed     0.1983 m/s        0.2054 m/s   (~3.2x)
T0 moving_speed                0.3072            0.2933
T1 moving_speed                0.7279            0.7265       (~2.4x)

balanced_accuracy vs reference:
  oracle candidate (T0 on oracle): 0.736 (L) / 0.765 (R)
  H0 candidate, T0 (transfer):     0.500 (L) / 0.500 (R)   <- chance level
  H0 candidate, T1 (recalibrated): 0.771 (L) / 0.840 (R)   <- meets/exceeds oracle
```

**답 — 섹션 6이 명시적으로 요구한 질문**: "T0는 transfer를 답한다, T1은 단순
runtime-domain calibration으로 충분한지를 답한다." **T0 transfer는 실패한다.
T1(같은 규칙, 다른 데이터로 재적합)은 충분하다** — balanced accuracy가
오라클 수준과 같거나 그 이상으로 회복된다. Contact 신호 자체(discriminative
signal)는 H0에 남아있지만, **오라클에서 얻은 문턱값을 그대로 재사용할 수는
없다.**

## Contact 실패 원인 분류 (섹션 7)

TRAIN에서만 적합한 position-error 문턱값(90th percentile = 0.0555m)으로
oracle-candidate 대 H0-candidate(T0) 불일치를 5개 범주로 귀속:

```
                                   left    right
body_root_relative_motion_ambiguity  456    412   (~65-71%)
observation_invalidity                108    144   (~17-23%)
ankle_specific_failure                 20     36   (~3-6%)
pose_position_error                    19     21   (~3%)
temporal_jitter                        40     26   (~4-6%)
total_disagreements                   643    639
```

**해석**: O-vs-H0 불일치의 **대다수(65-71%)는 H0의 잘못이 아니라 docs/51이
이미 지적한 root-relative/world 모호성**이다(오라클 자신도 그 프레임에서
reference와 이미 불일치했음). 진짜 H0-도입 오차는 observation invalidity(H0가
스스로 관절을 invalid로 표시, 17-23%)가 가장 크고, ankle-specific 실패는
작지만 실재한다(3-6%).

## R0/R1/R1_T1 root-translation control (섹션 8)

```
                                          mean naive error   recoverable segments (of 3,407-3,391)
O  (oracle contact + oracle geometry)     0.0054-0.0056 m    78 (L) / 108 (R)
R0 (oracle contact + H0 geometry)         0.0149-0.0166 m    78 (L) / 108 (R)   (~2.7-3.0x O)
R1 (H0-T0 contact + H0 geometry)          0.020 m (L only, 1 segment) / N/A (R, 0 segments)
R1_T1 (H0-T1 contact + H0 geometry)       0.032-0.035 m      117 (L) / 118 (R)  (~5.8-6.5x O)
```

**해석**: R0가 순수하게 geometry 품질만 격리한다(같은 오라클 contact
선택, ankle 위치만 H0) — 여기서 벌써 오차가 ~3배 커진다. R1(T0)은 섹션 6에서
이미 확인한 recall≈0 문제 때문에 recoverable segment가 사실상 사라져 숫자
자체가 무의미하다. **R1_T1(제대로 재적합한 contact 선택)은 segment coverage를
오라클 수준으로 회복시키지만, 그 상태에서도 오차는 오라클 대비 ~6배 더
크다** — geometry 오차(R0)와 contact-selection 관대화(T1이 더 넓은 CONTACT
정의를 씀)가 누적된다. **결론: planted-foot 기하학은 H0에서도 완전히
붕괴하지는 않지만, 오차가 실질적으로 커지고, 이 성능은 contact 문턱값을
H0 도메인에 맞게 재적합했을 때만 나온다.**

## Camera motion (섹션 9)

이번 배치에서 camera pose 추정, optical-flow 보정, SLAM, ground-plane
추정을 추가하지 않았다. Docs/51의 3DPW(moving) vs MPI-INF-3DHP(static)
결과는 그대로 참조로만 인용한다 — MPI-INF-3DHP용 FrameBank/H0가 존재하지
않으므로 이번 배치는 H0 조건에서 static-camera 대조군을 재현하지 않았다
(재현하려면 새 FrameBank를 만들어야 하는데, 이는 "새 H0 lineage를 만들지
말라"는 제약과 충돌할 소지가 있어 범위 밖으로 남겼다). Moving-camera
contamination이 여전히 지배적인 장벽이라는 docs/51의 결론은 이번 배치가
바꾸지 않는다 — 오히려 R0/R1의 오차 확대가 그 위에 추가로 쌓인다.

## Qualitative owner replay (섹션 10)

`~/animcv-output/root_motion_contact_h0_replay/review/` 5개 정지 프레임 +
`manifest.json`:

| Category | Sequence:actor#frame | 근거 |
|---|---|---|
| known_good_pose | courtyard_dancing_00:actor0#424 | 최저 H0-vs-oracle 평균 관절 오차 (0.0131 m) |
| ankle_or_foot_loss | outdoors_parcours_00:actor0#384 | 최대 ankle 오차 (0.9258 m) — 로프에 매달려 다리를 뻗은 실제로 어려운 자세, 육안으로도 확인됨 |
| turning | outdoors_crosscountry_00:actor0#202 | 최대 reliable H0 yaw 변화 (84.6°) |
| tracking_loss | courtyard_hug_00:actor1#312 | 최장 H0 관절-invalid run (106 bank rows) |
| walking | courtyard_rangeOfMotions_01:actor0#234 | 최다 H0(T0) CONTACT↔MOVING 교대 (**단 1회** — T0 recall≈0 문제의 직접적 증상) |

`ankle_or_foot_loss` 프레임은 실제로 봤을 때도 한쪽 발이 로프에, 다른 쪽
다리는 뒤로 뻗어 기둥을 짚은 확연히 어려운 monocular 자세였다 —
수치상 최대 오차가 시각적으로도 타당한 실패와 대응한다는 것을 확인했다.
`walking`의 빈약한 결과(1회 교대)는 그 자체로 T0 recall 붕괴를 다시 보여주는
증거다.

## 변경 파일

```
src/pose/framepose_bridge.py
src/pose/contact_failure_attribution.py
scripts/run_root_motion_contact_h0_replay.py
tests/test_framepose_bridge.py
tests/test_contact_failure_attribution.py
tests/test_h0_replay_contracts.py
docs/52_WORKLOG_ROOT_MOTION_CONTACT_H0_PRODUCTION_REPLAY.md (this file)
```

기존 파일 수정 0건.

## 테스트 결과 (섹션 12)

```
$ pytest tests/test_framepose_bridge.py tests/test_contact_failure_attribution.py \
  tests/test_h0_replay_contracts.py tests/test_motion_ownership.py tests/test_contact.py \
  tests/test_contact_reference.py tests/test_root_orientation_diagnostic.py \
  tests/test_root_translation_control.py -q
51 passed
```

로컬(Mac)과 `animcv-framepose:cuda118` 컨테이너(LabServer63) 양쪽에서 동일하게
51 passed. Full regression은 실행하지 않았다(공유 계약 미수정, 섹션 12 규칙).
섹션 12가 요구한 항목 커버리지:

- oracle/H0 regime 분리 → report.json의 모든 필드가 `_oracle`/`_h0` 접미로
  분리, 절대 혼합 집계 없음(스크립트 구조로 보장).
- target leakage 없음 → `test_build_h0_lifted_sequence_never_reflects_target_arrays`.
- fixed percentile-rule identity → `test_fit_contact_thresholds_percentile_rule_defaults_are_unchanged`.
- TRAIN-only H0 threshold fitting → `test_sequence_ids_in_split_never_returns_a_validation_id_for_train`
  + 스크립트의 `_fit_h0_train_thresholds`가 `sequence_ids_in_split(bank, "train")`만 사용.
- 변경 없는 docs/51 contact semantics → `test_contact_state_semantics_are_exactly_contact_moving_unknown`.
- historical root_motion.py 무변경 → `git diff --stat` 확인(0 변경).
- deterministic replay → `test_build_h0_lifted_sequence_is_deterministic_on_replay`
  + docs/51 자체의 결정론 테스트들(재사용, 무변경).
- UNKNOWN 보존 → `test_build_h0_lifted_sequence_marks_non_finite_h0_as_invalid`
  + docs/51의 UNKNOWN 관련 테스트 전부 무변경.

## 최종 architecture verdict

> "Are the Root Orientation and Contact signals established under oracle
> geometry still usable when driven by AnimCV's actual frozen FramePose
> output?"

**E — MIXED / INSUFFICIENT** (아래에 정확히 어느 부분이 왜 그런지 명시)

- **Root Orientation: 명확히 SURVIVES.** Yaw 오차 6.5°, flip/sign-flip 드묾,
  historical hold heuristic이 production 조건에서 실제로 유용하게 작동
  (98% 정확한 억제, 2% 오억제). 추가 조치 없이 다음 단계 입력으로 사용
  가능.
- **Contact: 조건부로만 SURVIVES.** Discriminative signal 자체는 살아있다
  (T1 재적합 시 balanced accuracy가 오라클과 동등하거나 상회). 하지만
  **오라클에서 얻은 문턱값(T0)을 그대로 재사용하면 완전히 붕괴한다**
  (recall→0). "그냥 survive한다"고 말할 수 없고, "도메인별 재적합이라는
  조건 하에서 survive한다"고 말해야 정확하다 — 이래서 A(둘 다 그냥
  survive)도 B(root orientation만 survive)도 아니고 E가 맞다.
- **Root translation control(참고용, verdict 문자에는 포함 안 됨)**: 순수
  geometry 열화만으로 오차 ~3배, 여기에 재적합된 contact 선택까지 더하면
  ~6배. 붕괴하지는 않지만 실사용을 논하기엔 아직 이르다.

이 배치는 "완성된 Root Motion solver"를 선언하지 않는다. Foot locking, IK,
retargeting, 새 contact 모델, camera 추정으로 자동 진행하지 않는다. 다음
단계로 넘어가려면 최소한 **contact 문턱값의 도메인별(H0) 재적합 절차를
production 경로에 넣는 결정**이 먼저 필요하다 — 그 결정 자체는 이번 배치의
범위 밖에 남겨둔다.
