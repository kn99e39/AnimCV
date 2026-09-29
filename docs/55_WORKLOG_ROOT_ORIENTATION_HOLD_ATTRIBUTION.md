# Worklog — Historical yaw hold attribution: 안정화인가, stale heading의 원인인가 (2026-09-29)

## 범위와 시작 상태

docs/53이 발견하고 docs/54가 시각적으로 확인한 "release 없는 yaw hold"가 frozen
FramePose H0에서 **heading을 돕는지**, 아니면 **stale heading의 주원인인지**를
귀속(attribution)하는 진단 배치. 새 estimator, threshold, release 규칙은 만들지 않았다.

- branch `arch/single_frame_first`, 시작 HEAD `382f177` (다른 세션의
  `docs/POSE_COMPLETION_PRODUCT_PLAN.md` 추가 commit을 pull한 상태)
- 코드 commit: `65bf295` (accounting + 측정 + 리뷰 렌더러), `ecd58d5` (렌더러 표시 수정)
- regime: H0 신호는 `benchmark_detector_observation`, oracle은 3DPW GT geometry
  (평가 전용)

## 보존

`src/pose/root_motion.py`, AnimationSemantics schema와 bridge, contact calibration과
contact state, reliability, frozen H0, FrameBank, docs/51-54 — 수정 0건. LEGACY-HOLD는
docs/53이 persist한 semantics 파일을 **읽기만** 했다(파일 content digest를 docs/53
report와 대조한 뒤 사용).

## Control experiment 설계

같은 H0, 같은 fused bilateral heading 수식, 같은 oracle matching, 같은 validation
3,407 rows. 바꾼 것은 **평가하는 yaw 신호** 하나뿐이다.

- **H0-CURRENT**: `root_orientation_diagnostic.observe_yaw(H0)` — frame별 shoulder+hip
  fused heading
- **LEGACY-HOLD**: AnimationSemantics yaw = `estimate_root_motion` (median 5 + 20° hold)
- **oracle**: docs/52 `_matched_oracle_sequence` → 같은 `observe_yaw`. 3,407 rows 모두
  oracle pair가 valid였다.

같은 frame에서 두 신호의 yaw convention이 일치함을 테스트로 고정했다(unwrap된 legacy
yaw도 원형 차이로 비교). 20°(root_motion 기본 step), 90°(diagnostic flip 정의), 45°는
**보고용 구간**일 뿐 어떤 신호에도 들어가지 않는다.

H0 identity: bank `75519e63…`, H0 validation `88a007fc…` (docs/52-53과 동일).
Artifact: `~/animcv-output/root_orientation_hold_attribution/` (`report.json` sha256
`c7f6ef95…28b8`, `rows.jsonl`, `review/`). 로컬 복사본은 `user_QE/animcv-output/`에서
볼 수 있다.

## Held-run accounting (target 없음)

```
held frames 998 / 3,407, held runs 23
run length: median 24, p90 70.6, max 337
run-length histogram: 1 row 0 | 2-5 rows 3 | 6-20 rows 6 | >20 rows 14
held frames by run age: first 23 | 2-5 88 | 6-20 239 | >20 648
```

held frame의 **65%가 run의 21번째 row 이후**에 있다. hold는 짧은 flip 제거보다 긴 latch로
주로 쓰이고 있다.

## Held-vs-current gap (target 없음)

held yaw와 그 frame의 H0-CURRENT fused yaw 사이 각도:

```
                 n    median   p90    p95    max    >20°   >45°   >90°
all held        998   48.9    150.6  166.6  179.6   83.6%  53.7%  30.7%
age first        23   27.0     41.4   47.4   85.2  100%     8.7%   0%
age 2-5          88   48.1    102.7  119.1  148.4  100%    63.6%  12.5%
age 6-20        239   60.8    154.8  169.0  178.8   98.7%  69.5%  38.5%
age >20         648   43.7    151.8  167.2  179.6   75.2%  48.1%  31.3%
non-held (참고) 2409    0.3      5.1    8.3   69.7    1.1%   0.3%   0%
```

gap은 첫 held row(median 27°)에서 2-5 rows(48°), 6-20 rows(61°)로 **latch 기간에 따라
커진다**. >20 rows 구간 median(44°)이 약간 낮은 것은 오래 걸린 latch 도중 몸이 우연히
원래 방향으로 돌아오는 구간이 섞이기 때문이다(p90/p95는 계속 150° 이상). target 없이도
held heading이 현재 관측에서 체계적으로 멀어진다.

## Oracle 오차: H0-CURRENT vs LEGACY-HOLD

```
subset           signal       n     mean  median  p90    p95    max    >45°    >90°
all              H0-CURRENT  3407    6.4    4.6   13.2   16.8  101.1   0.6%    0.03% (1 frame)
all              LEGACY      3407   23.6    6.3   78.9  133.8  179.9  15.9%    8.7%  (297 frames)
non-held         H0-CURRENT  2409    6.0    4.6   12.5   15.8   81.7   0.5%    0%
non-held         LEGACY      2409    5.4    4.3   11.5   14.3   45.2   0%      0%
held             H0-CURRENT   998    7.3    4.8   14.7   22.3  101.1   1.0%    0.1%
held             LEGACY       998   67.5   48.6  153.0  166.6  179.9  54.1%   29.8%
held age first   H0-CURRENT    23   10.2   10.0   21.2   25.4   33.9   0%      0%
held age first   LEGACY        23   24.1   23.1   37.8   40.1   62.9   4.3%    0%
held age 2-5     H0-CURRENT    88   10.3    6.3   22.1   30.5   74.8   1.1%    0%
held age 2-5     LEGACY        88   53.1   46.8   89.7   99.3  130.0  55.7%   10.2%
held age 6-20    H0-CURRENT   239    7.5    4.2   14.2   20.3  101.1   2.5%    0.4%
held age 6-20    LEGACY       239   81.0   65.5  153.2  171.4  179.9  76.2%   38.5%
held age >20     H0-CURRENT   648    6.7    4.8   14.0   19.8   50.0   0.5%    0%
held age >20     LEGACY       648   66.1   41.5  157.8  167.0  179.7  47.5%   30.2%
```

- held가 아닌 frame에서는 LEGACY가 H0-CURRENT보다 약간 낫다(mean 5.4 vs 6.0).
  hold가 아닌 **median smoothing**의 효과다.
- held frame에서는 LEGACY 오차가 모든 run age에서 H0-CURRENT의 5-10배다. **첫 held
  row부터** LEGACY가 더 나쁘다(24.1 vs 10.2).
- H0-CURRENT가 oracle과 90° 넘게 다른 frame은 3,407개 중 **1개**다. hold가 제거할
  H0 flip이 이 데이터에는 거의 없다.
- held frame 998개 중 LEGACY가 H0-CURRENT보다 oracle에 가까운 frame은 54개(5.4%)다.

## 실제 큰 회전 vs 가짜 flip (run 단위 누적 오차)

**Run 단위 귀속** — run 전체의 oracle 오차를 누적해서 비교했다. 단일 전이로 판단하지
않는다.

```
runs 23:  hold_worse_from_start 18 (held frames 971)
          stale_after_reasonable_start 3 (19)
          hold_better_than_current 2 (8)
oracle heading이 run 시작 직전 accepted row 대비 20° 넘게 움직인 run: 22 / 23
```

긴 run 상위 10개는 전부 `hold_worse_from_start`다. 이 run들에서 oracle heading은 anchor
대비 최대 105-180° 움직였고, 몸이 실제로 방향을 바꾸는 동안 hold가 이전 heading을
유지했다. 예: parcours_00 337 rows, LEGACY mean 51° vs CURRENT 6°; crosscountry_00 65
rows, 153° vs 9°.

**Row 단위 큰 변화** — 연속 row 사이 H0-CURRENT 변화가 20°를 넘은 event:

```
                         n    held  held 중 LEGACY가 더 나은 event
oracle도 >20° 변화       17    16      1      ← 실제 빠른 회전 16/17을 hold가 억제
oracle은 ≤20° 변화      141    80      7      ← H0 가짜 변화; held인 80개 중 hold가 이긴 것 7개
H0-CURRENT 변화 >90°      3     3      0
```

H0가 가짜로 튄 event에서도 held인 경우 대부분 이미 진행 중인 latch 안에 있어서
LEGACY 쪽이 더 틀렸다(mean 44° vs 18°).

## docs/52 해석의 유효성

docs/52는 hold frame의 "98%(978/998)가 oracle 전이 ≤20°에서 발동 → 겉보기 flip 방지"라고
해석했다. **이 해석은 run 단위 오차 accounting에서 유지되지 않는다.** 같은 998 frame 중
971개가 `hold_worse_from_start` run에 속하고, held frame의 LEGACY 오차 mean은 67.5°다.
oracle 전이가 작다는 사실은 "hold가 옳았다"는 뜻이 아니었다. 그 frame들은 hold가 이미
틀린 heading에 latch된 **이후**의 작은 전이였을 뿐이다. docs/53의 1차 정정("98%는 과대
서술")을 이번 측정이 정량적으로 확정한다.

## Owner case replay

`scripts/render_root_orientation_hold_review.py`,
`~/animcv-output/root_orientation_hold_attribution/review/` (로컬 H.264 사본):
RGB + H0 skeleton, TOP view에 H0-CURRENT(청록) / LEGACY(held면 빨강) / oracle(초록)
세 heading, run age, 오차 곡선 timeline.

| Case | frame | run age | H0-CURRENT err | LEGACY err | gap | 보이는 것 |
|---|---|---|---|---|---|---|
| docs/54 turning | crosscountry_00#202 | 4 | 74.8 | 60.8 | 135.6 | 드물게 H0-CURRENT도 크게 틀린 frame(두 신호 모두 나쁨) |
| docs/54 known-good held | dancing_00#424 | 88 | 2.4 | 3.1 | 5.5 | latch가 우연히 맞는 구간. 같은 run의 앞부분은 크게 틀림(run mean 71° vs 10°) |
| longest held run | parcours_00#1138 (337 rows) | 169 | 46.6 | 6.1 | 40.6 | 이 frame만 보면 hold가 이김. timeline을 보면 run 전체에서 LEGACY가 180°까지 튀는 구간이 여러 번 있음 |
| largest held-current gap | crosscountry_00#462 | 38 | 5.0 | 174.6 | 179.6 | 카메라를 향해 걸어오는 사람을 hold는 반대 방향으로 고정 |
| fastest oracle turn (98°/row) | crosscountry_00#343 | 1 | 22.3 | 62.9 | 85.2 | 실제 빠른 회전의 첫 row에서 hold가 발동해 옛 heading 유지 |

한 frame만 골라 보면 어느 쪽 사례든 찾을 수 있다(longest-run frame, docs/54 turning).
이번 배치가 run 단위 누적 오차를 기준으로 삼은 이유다.

표시 한계: crosscountry_00의 일부 clip에서 docs/46 `_load_camera_state`의 GT root 값이
depth 음수인 상수로 나와 H0 skeleton을 RGB에 투영할 수 없었다. 렌더러는 이제 그 사실을
화면에 표시한다(`ecd58d5`). 이 문제는 **표시용 anchoring 경로에만** 해당한다.
oracle yaw 평가는 matched oracle의 root-relative geometry를 쓰고 3,407 rows 모두
valid다.

## Verdict: **B — DEMOTE LEGACY HOLD**

- H0-CURRENT가 전체적으로 낫다: 전체 mean 6.4° vs 23.6°, p95 16.8° vs 133.8°, >90°
  1 frame vs 297 frames.
- hold가 심각한 stale heading을 만든다: held frame LEGACY mean 67.5°, >45° 54%.
- **C가 아닌 이유**: C는 "초기 rejection은 유용하다"가 전제다. 이번 측정에서는 첫 held
  row부터 LEGACY가 더 나쁘고(24 vs 10), run 23개 중 hold가 이긴 것은 2개다. 실제 빠른
  회전 17개 중 16개를 억제했다. H0 flip(>90° 오차)은 1 frame뿐이라 rejection이 해결할
  대상 자체가 거의 없다. release를 설계해도 이득을 볼 초기 rejection이 없다.
- 해석 (측정한 것이 아니라 설명 가설): `root_motion.py`의 hold는 VideoPose3D temporal
  lifter의 한 frame axis flip(50 fps 기준)을 겨냥해 만들어졌다. frozen H0는 single-frame
  모델이고, validation bank row는 3 frame 간격(0.1 s)이어서 "row당 20°"는 200°/s에
  해당한다. 실제 사람의 회전이 이 값을 자주 넘는다.

**다음 구현 단계 명세 (이번 배치에서 구현하지 않음)**:
FramePose-era AnimationSemantics는 `root_motion.py`의 hold가 아닌 **별도의
current-heading policy**를 써야 한다. 출발점은 frame별 fused bilateral heading
(`observe_yaw`와 같은 정의)이다. `root_motion.py`는 historical baseline으로 무수정
유지한다. 정책 교체는 AnimationSemantics bridge의 root-orientation provenance/policy
버전 변경으로 드러나야 한다(v1 파일 재해석 금지). non-held frame에서 median
smoothing이 약간(0.6° mean) 도움이 되었다는 관찰은 기록만 한다. smoothing을 유지할지는
이 배치에서 정하지 않는다(튜닝 금지).

## 한계

- 3DPW validation 16 sequences / 3,407 rows, `benchmark_detector_observation` 한
  regime뿐. `real_animcv_observation`(MMPose)에서는 H0-CURRENT flip 빈도가 다를 수 있다.
- oracle heading도 같은 bilateral 정의라서 "몸통 heading" 기준의 오차다.
- row stride는 validation 3 frame이다. 다른 stride에서 hold가 발동하는 빈도는 다를 수
  있다(측정하지 않음).

## 변경 파일

```
src/pose/root_orientation_hold_attribution.py      (신규: accounting)
scripts/run_root_orientation_hold_attribution.py   (신규: control experiment)
scripts/render_root_orientation_hold_review.py     (신규: owner 리뷰)
tests/test_root_orientation_hold_attribution.py    (신규: 16 tests)
scripts/render_animation_semantics_review.py       (수정: anchoring 불가 frame 표시, docs/54 리뷰 스크립트)
docs/55_WORKLOG_ROOT_ORIENTATION_HOLD_ATTRIBUTION.md
```

## 테스트

```
pytest tests/test_root_orientation_hold_attribution.py tests/test_animation_semantics.py -q
41 passed
```

신규 16개: 같은 frame에서 current/legacy yaw convention 일치, 원형 차이(7 cases +
unwrap), held-run age와 구간, run 경계, current/legacy/oracle row 정렬과 mismatch 거부,
oracle validity, run 단위 분류 3종, 큰 변화 event의 oracle context, 분포 구간,
report 생성 결정론. production/shared 계약을 바꾸지 않았으므로 full regression은
생략했다.
