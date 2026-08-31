# Worklog — A14 Bilateral Forward-Depth Supervision (2026-08-31)

> 이번 batch는 3DPW generalization 진단(docs/17)이 지목한 monocular ambiguity
> 가설을 직접 시험했다: 어깨/엉덩이 bilateral forward-depth(`q = (y_right -
> y_left) / sqrt(2)`, canonical `+Y`)를 모든 valid 프레임에 대해 base
> coordinate loss와 동일한 reduction으로 명시 supervise하면, 결정론적
> temporal lifter가 이 mode를 회복하는지 시험했다. yaw loss 추가, hard-tail
> mining, 수동 auxiliary weight는 모두 사용하지 않았다. 정량 수치와
> evaluator/gate 정의의 단일 출처는 `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`다.

## 1. Coordinate-contract verification

`src/pose/pose_lifter.py`의 `_to_lifted_points`에 명시된 canonical camera
frame은 `+X` right, `+Y` forward/depth, `+Z` up이다 (H36M `(x,y,z)` →
`(x,z,y)` 재배열 후 X/Z 부호 반전). 따라서 position array의 index `1`이
forward/depth다 — 이전 진단(docs/17)의 `signed_forward_y`도 동일 index를
사용했다. `FORWARD_DEPTH_AXIS = 1`을 `src/training/temporal_lifter.py`에
상수로 고정하고, 이전 canonical Z 축 기반 관습(`d = z_right - z_left`)과
혼동하지 않도록 주석으로 근거를 남겼다.

## 2. Exact relative-depth loss definition

`_bilateral_forward_depth_grid`가 기존 `TORSO_INDICES`(shoulder-then-hip,
`right - left`) 페어 컨벤션과 validity contract를 그대로 재사용해
`q_shoulder`, `q_hip`을 계산한다:

```
q = (prediction[right, FORWARD_DEPTH_AXIS] - prediction[left, FORWARD_DEPTH_AXIS]) / sqrt(2)
```

`_bilateral_forward_depth_residual_sum`이 base coordinate loss와 동일한
smooth-L1 family(default beta)로 `(pred_q, target_q)` 잔차의 `(sum, count)`를
반환한다.

## 3. Coordinate-equivalent normalization derivation

`_supervision_loss`는 `bilateral_forward_depth_supervision=True`일 때 이
`(sum, count)`를 base coordinate loss 자신의 `(sum, count)`에 **더한다** —
별도 평균을 내 tunable lambda를 곱하는 대신, "하나의 relational scalar
잔차가 정확히 하나의 추가 scalar coordinate처럼 기여한다"는 계약을
그대로 구현했다. `1/sqrt(2)`는 `common=(y_R+y_L)/sqrt(2)`,
`q=(y_R-y_L)/sqrt(2)`가 `[y_R, y_L]`의 orthonormal 변환이 되게 하는
basis normalization이며 튜닝 대상이 아니다.

## 4. Synthetic contract results

`tests/test_bilateral_forward_depth_loss.py` 17개 전부 PASS:
identical→zero, uniform forward translation 불변, pure common-depth error
불변, anti-symmetric error(`right+a, left-a`)의 반대 부호 gradient(quadratic
region에서 정확히 `+-a`), 부호만 반전된 큰 오차의 큰 residual, GT bilateral
depth가 0이어도 finite, shoulder/hip 독립, 관련 없는 joint·축에는 gradient
0, base coordinate loss와 동일한 smooth-L1 beta transition, coordinate-
equivalent pooling 계약, pair validity masking, deterministic replay.

## 5. Fixed-batch gradient comparison

`scripts/diagnose_bilateral_forward_depth_gradients.py`로 A9/A11/A12가 봤던
동일 첫 epoch 10개 batch(seed `1337`, source-balanced permutation,
augmentation)를 init과 A9 최종 checkpoint에 replay했다 (학습 없음).

| 모델 상태 | mean candidate/base ratio | max ratio | mean cosine |
| --- | ---: | ---: | ---: |
| init | 0.375 | 0.461 | 0.121 |
| A9 최종 checkpoint | 0.458 | 0.628 | 0.208 |

historical angular yaw-tail의 A9 상태 ratio는 `3,169.92`, A12 Cartesian
torso-tail은 `5.28`이었다 — A14 후보는 그보다도 낮아 A11식 gradient-scale
붕괴를 재현하지 않는다. cosine은 두 상태 모두 양수(방향 정렬).

## 6. Source-wise diagnostic behavior

A9 상태 fixed-batch replay 10개 합산, source별 raw residual(all-frame,
tail 아님):

| Source | valid pair count | raw residual mean |
| --- | ---: | ---: |
| 3DPW | 824 | 0.000116 |
| AMASS | 877 | 0.000125 |
| MPI-INF-3DHP | 836 | 0.000209 |

세 source 모두 동일 order of magnitude다 — A11/A12의 tail selector가
보였던 AMASS 45~92% 편중과 달리, all-frame supervision은 source를
극단적으로 편중하지 않는다. source-isolated gradient/base ratio도
`0.83~1.61` 범위였다.

## 7. GO / NO-GO decision

6개 조건 전부 충족 → **GO**.

1. canonical `+Y` 검증 완료 (Section 1)
2. synthetic 계약 17개 전부 PASS (Section 4)
3. coordinate-equivalent normalization 도출·테스트 완료 (Section 3)
4. A11식 gradient-scale pathology 없음(Section 5, ratio `<1`)
5. source-wise pathology 없음(Section 6)
6. gradient가 실제로 shoulder/hip forward-depth에만 도달함을
   `_endpoint_gradient` 계약으로 확인(`tests/test_diagnose_bilateral_forward_depth_gradients.py`)

## 8. Controlled training result

정확히 하나의 A9 조건 + `--bilateral-forward-depth-supervision`만 추가한
학습을 실행했다(`ablation_a14_bilateral_forward_depth_10e_v2`, seed
`1337`, 10 epoch, `direct_mix`). dataset fingerprint 6개(mpi_train,
three_dpw_train, three_dpw_holdout, amass_train, amass_holdout,
validation) 전부 A9와 SHA-256 완전 일치 확인. yaw_loss/yaw_tail_loss/
hinge_flip_loss/end_effector_loss/cartesian_torso_tail_loss는 모두 0으로
유지했다.

주의: 첫 시도는 `--amass-holdout /data/amass/prepared_aug_v1/holdout.json`
(31,910 frame)을 써서 A9의 실제 baseline holdout(`/data/amass/prepared/
holdout.json`, 10,792 frame, docs/10 A8 참고)과 fingerprint가 달랐다 —
그 run은 폐기하고 올바른 경로로 재실행했다.

## 9. A9 vs candidate metric table

| Holdout | 지표 | A9 | A14 | 변화 |
| --- | --- | ---: | ---: | --- |
| 3DPW test | PA-MPJPE mm | 75.31 | 77.34 | +2.03 (악화) |
| 3DPW test | yaw MAE ° | 14.90 (통과) | **15.15 (신규 실패)** | 악화 |
| 3DPW test | yaw P95 ° | 34.77 (기존 실패) | 38.18 (더 실패) | 악화 |
| 3DPW test | hinge flip | 2.36% | 2.79% | 악화(진단용) |
| AMASS internal | PA-MPJPE mm | 69.19 (통과) | 73.87 (통과) | +4.68 (악화, 게이트는 유지) |
| AMASS internal | yaw MAE ° | 8.77 (통과) | 7.71 (통과) | 개선 |
| AMASS internal | yaw P95 ° | 22.37 (통과) | 21.19 (통과) | 개선 |
| training MPJPE mm | — | 40.19 | 40.13 | 사실상 동일 |

training MPJPE가 사실상 그대로라는 점은 Section 5의 gradient 진단이
예측한 대로 A11식 붕괴가 없음을 확인해준다. 그러나 3DPW test의 yaw MAE는
기존에 통과하던 게이트를 새로 실패했고, yaw P95는 A12(38.45)와 비슷한
수준으로 더 악화됐다. AMASS는 3개 게이트 모두 유지했지만 PA-MPJPE가
소폭 악화됐다.

## 10. Forward-depth residual/sign attribution

`scripts/attribute_bilateral_forward_depth.py`로 3DPW 공식 test holdout
(35,310 frame) 전체에 A9 evaluator 자신의 per-frame root-yaw 순위로 hard
top-5%/top-1%를 고정한 뒤(candidate 관찰 후 재정의하지 않음), A9/A14
예측에 대해 `_bilateral_forward_depth_diagnostics`를 동일 subset에 적용했다.

| Subset (n) | 지표 | A9 | A14 |
| --- | --- | ---: | ---: |
| all eligible (35,310) | shoulder abs residual m | 0.0879 | 0.0862 |
| | hip abs residual m | 0.0326 | 0.0318 |
| | shoulder sign disagreement | 14.59% | 15.59% |
| | hip sign disagreement | 16.45% | 15.21% |
| A9 hard top-5% (1,766, cutoff 34.77°) | shoulder abs residual m | 0.2312 | **0.1814** |
| | hip abs residual m | 0.0907 | **0.0672** |
| | shoulder sign disagreement | 48.45% | **41.49%** |
| | hip sign disagreement | 50.28% | **39.07%** |
| A9 hard top-1% (354, cutoff 50.09°) | shoulder abs residual m | 0.2949 | **0.2197** |
| | hip abs residual m | 0.1283 | **0.0887** |
| | shoulder sign disagreement | 68.75% | 67.65% |
| | hip sign disagreement | 61.58% | 59.04% |
| non-hard (33,544) | shoulder abs residual m | 0.0810 | 0.0817 |
| | hip abs residual m | 0.0295 | 0.0300 |
| | shoulder sign disagreement | 12.96% | 14.34% |
| | hip sign disagreement | 14.67% | 13.96% |

hard top-5%/top-1%에서 forward-depth abs residual과 sign disagreement가
모두 **의미 있게 감소**했다 — 직접 supervise한 양은 실제로 학습됐다.
그러나 non-hard subset에서는 residual/sign disagreement가 오히려 소폭
**악화**됐고(shoulder sign disagreement 12.96%→14.34%), top-1%의 sign
disagreement는 여전히 60% 안팎(거의 동전 던지기 수준)에서 크게 움직이지
않았다.

## 11. Epoch telemetry

512-window 고정 subset(train-domain) 기준, `bilateral_forward_depth_raw`가
epoch 0의 `0.000259`에서 epoch 9의 `0.000130`로 매끄럽게 줄었고
`coordinate`(`0.00260→0.00105`)와 같이 수렴했다 — A11처럼 발산하지 않음을
학습 중에도 확인했다. `diagnostic_shoulder_forward_depth_sign_disagreement`도
`0.0234→0.0117`대로, `diagnostic_hip_forward_depth_sign_disagreement`도
`0.0391→0.0273`대로 줄었다(train-domain 기준이라 절대값이 test holdout보다
훨씬 낮다). 이 diagnostic 필드는 optimizer에 연결되지 않는다.

## 12. Matched qualitative review

docs/10 A9 tail 진단에서 이미 고정된 4개 시퀀스(worst-P95
`downtown_stairs_00:actor0`, 최장-run `downtown_walking_00:actor1`, P95
근접 `downtown_bus_00:actor1`, 대조군 `downtown_bar_00:actor0`)에 대해
A9/A14 예측을 재생했다(candidate 관찰 후 시퀀스를 새로 고르지 않음).

| Sequence | A9 yaw mean/P95 ° | A14 yaw mean/P95 ° | A9→A14 worst-frame yaw ° | 대조군 sign disagreement (shoulder) |
| --- | --- | --- | --- | --- |
| downtown_stairs_00:actor0 | 24.40 / 59.15 | 21.33 / **64.72** | 91.20 → 66.70 | — |
| downtown_walking_00:actor1 | 21.78 / 47.42 | 21.61 / **56.58** | 55.13 → 35.35 | — |
| downtown_bus_00:actor1 | 11.09 / 32.26 | 11.71 / 31.07 | 67.71 → **99.30** | — |
| downtown_bar_00:actor0 (대조군) | 9.18 / 16.97 | 10.71 / **27.80** | 25.20 → 22.51 | 6.99% → **15.98%** |

결과는 시퀀스별로 방향이 엇갈린다: worst-frame yaw가 줄어든 경우(stairs,
walking)도 있지만 같은 시퀀스의 P95는 오히려 악화됐고, bus 시퀀스의
worst frame은 67.7°에서 99.3°로 더 나빠졌다. 특히 이전까지 orientation
문제가 거의 없던 **대조군 시퀀스(bar:actor0)의 shoulder sign disagreement가
6.99%에서 15.98%로 새로 악화**됐다 — A14가 hard-set 평균 residual은
줄이지만, 이전에 문제없던 영역에 새 혼동을 들여왔다는 증거다.

## 13. Validation-vs-test interpretation

이번 batch는 validation을 별도로 평가하지 않았다 — docs/17이 이미
validation hard set의 주원인을 input-domain shift로 결론지었고, 이번
가설(monocular ambiguity)은 official test를 겨냥한 것이었다. `run_lifter_
experiments.py`가 항상 실행하는 validation 지표(3DPW+AMASS 통합, gate
없음)만 리포트에 남아 있으며, crop/projection/confidence augmentation은
이번 batch에서 변경하지 않았다.

## 14. Portability assessment

- canonical joint contract에만 의존하는가: 예 (`TORSO_INDICES`, H36M 17-joint)
- dataset-specific camera label이 아닌 canonical forward/depth 축을
  쓰는가: 예 (`FORWARD_DEPTH_AXIS`)
- hard-example threshold를 피하는가: 예 (all-frame, tail selection 없음)
- source-specific weight를 피하는가: 예 (source group 없음)
- normalization이 base coordinate-loss 계약에서 도출되는가: 예 (pooled
  sum/count, Section 3)
- residual이 0으로 가면 loss도 자연히 0으로 가는가: 예 (synthetic
  contract 1)
- 여러 source에서 일관되게 동작하는가: gradient-scale 관점에서는 그렇다
  (Section 6). 그러나 실제 평가 결과(Section 9-12)는 대조군 시퀀스에
  새 오류를 들여오는 등 일관되게 이롭지 않다.
- 미래 commercial dataset이 canonical 3D 좌표 계약에만 매핑하면 이
  loss를 그대로 쓸 수 있는가: 형식적으로는 예 — 그러나 이번 결과가
  "그대로 써도 이득"이라는 근거는 아니다.

## 15. Architecture verdict

**Case B (그러나 이상화된 Case B보다 나쁨) — depth mode는 hard-set
평균에서 개선되지만 yaw는 개선되지 않고 오히려 악화됨.**

- training MPJPE는 그대로였다(Section 9) — A11식 붕괴는 없다.
- A9 hard top-5%/top-1%에서 forward-depth abs residual과 sign
  disagreement는 실질적으로 줄었다(Section 10) — 직접 supervise한 신호는
  실제로 학습됐다.
- 그러나 3DPW test의 yaw MAE/P95는 둘 다 악화됐고 yaw MAE는 새로운 게이트
  실패를 만들었다(Section 9). AMASS PA-MPJPE도 소폭 악화됐다(게이트는
  유지).
- non-hard subset과 이전까지 깨끗했던 대조군 시퀀스에서는 sign
  disagreement가 오히려 악화됐다(Section 10, 12) — Case D(hard-case sign
  ambiguity 잔존)의 조짐도 함께 있다: hard top-1%의 sign disagreement는
  거의 그대로다(68.75%→67.65%, 61.58%→59.04%).

지시에 따라 이 결과에서 **weight를 조정하지 않는다. sign classification을
추가하지 않는다. 이 loss family를 이번 batch에서 계속 튜닝하지 않는다.**
bilateral forward-depth 자체는 yaw failure와 관련이 있지만(Section 10),
evaluator의 orientation state를 결정하기에 충분하지 않다는 결론이며,
non-hard 영역에 새 오류를 만든다는 점에서 A9 대비 순이익이 없다. 다음
아키텍처 질문은 (a) 더 풍부한 torso relational/local-frame 표현, 또는
(b) temporal/latent orientation-state supervision·multi-hypothesis
modeling(Case D 방향)이다 — 둘 다 이번 batch에서 구현하지 않는다.

## 16. Exact files changed

- `src/training/temporal_lifter.py` — `bilateral_forward_depth_supervision`
  config, `_bilateral_forward_depth_grid`/`_residual_sum`/`_diagnostics`,
  `_supervision_loss`/telemetry/report 연결
- `scripts/run_lifter_experiments.py` — `--bilateral-forward-depth-supervision`
- `scripts/diagnose_bilateral_forward_depth_gradients.py` (신규)
- `scripts/attribute_bilateral_forward_depth.py` (신규)
- `tests/test_bilateral_forward_depth_loss.py` (신규)
- `tests/test_diagnose_bilateral_forward_depth_gradients.py` (신규)
- `tests/test_attribute_bilateral_forward_depth.py` (신규)
- `tests/test_supervised_temporal_lifter.py` — telemetry key 계약 갱신
- `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`
- `docs/README.md`
- `docs/18_WORKLOG_A14_BILATERAL_FORWARD_DEPTH.md` (본 문서)

A9–A12 checkpoint/fingerprint/report, production config 기본값, gate,
`.vscode/`는 변경하지 않았다.

## 17. Focused/full tests executed

- A14 synthetic contract: `17 passed`
- A14 gradient-diagnosis 단위: `5 passed`
- A14 attribution 단위: `4 passed`
- telemetry 계약 포함 전체 로컬 회귀: `400 passed`
- `py_compile`: PASS
- LabServer63 GPU 진단/학습/평가 재현: PASS (fingerprint 6개 일치 확인)

## 18. Commit hashes / push / server synchronization state

- `3dd34ac` — feat: A14 all-frame bilateral forward-depth supervision candidate
- `34064ec` — fix: A14 gradient/sign-region test fixtures and telemetry key contract
- `db710e1` — diagnostic: A14 fixed-batch gradient replay (init and A9 checkpoint)
- `b8f3542` — diagnostic: A9-vs-A14 forward-depth attribution on 3DPW test hard set
- 문서 커밋은 본 worklog와 함께 별도로 기록한다.

`origin/On_Work`와 `LabServer63:/home/nd/AnimCV`는 `b8f3542`까지
fast-forward 상태였다(문서 커밋 이전). 서버의 기존 미추적 `.DS_Store`,
`.animcv_sync_stage/`, `docker/`, 로컬의 `.vscode/`는 건드리지 않았다.

학습 checkpoint/report는 git에 커밋하지 않고 다음 서버 경로에 남아 있다.

- `/home/nd/animcv-output/experiments/a14_bilateral_forward_depth_diagnosis/gradient_diagnosis.json`
- `/home/nd/animcv-output/experiments/a14_bilateral_forward_depth_diagnosis/test_attribution.json`
- `/home/nd/animcv-output/experiments/ablation_a14_bilateral_forward_depth_10e_v2/` (checkpoint, report, experiment_matrix)

폐기된 첫 시도(`ablation_a14_bilateral_forward_depth_10e`, 잘못된
`amass-holdout` fingerprint)는 서버에 남아 있으나 어떤 결론에도 근거로
사용하지 않았다.
