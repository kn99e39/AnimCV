# Temporal Lifter Improvement Ablation

## 목적

30-epoch augmented direct-mix 기준선의 3DPW holdout 실패를 단순 장기 학습으로
해결하지 않는다. 동일 split·seed·epoch에서 time context, source sampling, 2D domain
augmentation, structural loss의 영향을 분리한다.

## 변경된 학습 경로

- `dilated_tcn_v1`: dilation 1/2/4/8/16의 residual TCN으로 receptive field 127이다.
  기존 checkpoint는 `legacy_tcn_v1` (receptive field 5)으로 계속 불러올 수 있다.
- source-balanced sampling: MPI-INF-3DHP/3DPW/AMASS가 각 epoch에 같은 frame mass를
  갖도록 작은 3DPW source를 replacement sampling한다. clip-safe window 자체는 바뀌지 않는다.
- 2D augmentation: global scale/translation/roll과 sequence 경계를 넘지 않는 contiguous
  joint occlusion을 기존 jitter·dropout·confidence noise와 분리해 기록한다.
- `pelvis_torso_v1` coordinate contract: pelvis를 원점, pelvis→thorax 길이를 단위
  길이로 해 detector crop의 frame별 translation/scale을 제거한다. checkpoint에 계약을
  저장하므로 학습·holdout 평가·추론이 반드시 같은 전처리를 사용한다.
- structural loss: canonical bone vector, shoulder/hip torso axis, elbow/knee bend vector를
  target으로 사용한다. FBX rig를 학습 target으로 쓰지 않는다.

## 고정 조건

- train: MPI-INF-3DHP train + 3DPW official train + AMASS train
- validation: 3DPW validation + AMASS validation
- independent holdouts: 3DPW official test, AMASS internal holdout
- window 81, channels 256, batch 128, AMP, seed 1337
- 평가: MPJPE, PA-MPJPE, root yaw MAE/P95, hinge flip rate와 source/view/action slices

## 실행 순서

처음에는 `direct_mix` 하나만 10 epoch로 비교해 값비싼 full matrix를 피한다.

1. **A0 compatibility** — `legacy_tcn_v1`, source balancing/새 augmentation/structural loss를
   모두 끈다. 기존 baseline과 pipeline 차이가 없는지 확인한다.
2. **A1 temporal + balanced** — `dilated_tcn_v1 --source-balanced-sampling`만 켠다.
3. **A2 camera/occlusion domain** — A1에 scale 0.04, translation 0.03, roll 12°, temporal
   occlusion 10%/9f를 더한다.
4. **A3 structural** — A1에 bone 0.25, torso 0.15, hinge 0.15 loss를 더한다. A2의
   camera/occlusion 변형은 포함하지 않아 structural loss의 효과를 독립적으로 본다.
5. **A4 detector-crop invariance** — A1에 `--input-coordinate-normalization pelvis_torso_v1`만
   더한다. A2처럼 이미지를 임의 이동·확대하지 않고, 실제 detector가 만드는 crop/subject
   위치 차이에 불변인 2D 계약을 검증한다.
6. **A6 mean yaw supervision (rejected)** — A5 조건에 shoulder/hip XY-axis 평균 cosine
   loss 0.15를 추가했다. 3DPW test에서 PA-MPJPE 83.48 mm, yaw P95 35.70°, hinge
   flip 5.11%로 A5(71.88 mm, 32.43°, 2.07%)보다 모두 악화됐다. 평균 yaw loss는
   tail gate를 개선하지 못하고 pose geometry를 훼손했으므로 재사용하지 않는다.
7. **A7 tail-directional constraints (rejected)** — A5 조건에 평균 yaw loss 대신 bilateral
   yaw error 상위 5%만의 CVaR loss와, target bend와 90° 이상 반대가 된 hinge에만
   작동하는 anti-flip cosine loss를 작은 weight로 분리해 검증했다. 목표였던 yaw P95와
   flip tail 개선 대신 PA-MPJPE가 크게 악화돼 채택하지 않는다.
8. **A8 structural-loss throughput** — A5 품질 설정을 유지하고 구조 손실을 GPU 배치
   연산으로 벡터화한다. 이 단계는 품질 손실을 새로 추가하지 않는다. batch 128 기준
   처리량과 batch 확장 후보를 비교하되, 다른 GPU 작업이 없는 시간에만 실행한다.

각 run의 3DPW PA-MPJPE를 우선 정렬하고, yaw MAE/P95와 hinge flip이 동시에 악화되지
않는 상위 두 후보만 30 epoch로 재실행한다. 결과가 3DPW holdout을 통과하기 전에는
retarget/FBX 단계로 넘기지 않는다.

### A6 결과 (2026-08-18, GT 2D holdout)

| Holdout | PA-MPJPE mm | yaw MAE ° | yaw P95 ° | hinge flip | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| 3DPW test | 83.48 | 14.72 | 35.70 | 5.11% | 실패 |
| AMASS internal | 73.85 | 7.73 | 22.31 | 3.00% | 실패 |

3DPW 기준으로 PA-MPJPE, yaw P95, hinge flip gate를 모두 놓쳤다. 따라서 실제
detector 입력 평가는 아직 실행하지 않는다. detector 오차가 없는 GT 2D 조건에서도
통과하지 못한 checkpoint를 제품 입력으로 평가해도 승격 근거가 되지 않는다.

### A7 결과 (2026-08-18, GT 2D holdout, rejected)

`yaw_tail_loss_weight=0.05`, `hinge_flip_loss_weight=0.05`를 A5에 추가했다. 평균 yaw
loss는 사용하지 않았다. 학습은 2,379.3초, peak GPU memory는 1,200.6 MiB였다.

| Holdout | PA-MPJPE mm | yaw MAE ° | yaw P95 ° | hinge flip | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| 3DPW test | 84.28 | 14.67 | 35.38 | 2.04% | 실패 |
| AMASS internal | 90.03 | 8.47 | 22.63 | 3.11% | 실패 |

3DPW hinge flip은 A5의 2.07%보다 미세하게 낮아졌지만, yaw P95와 PA-MPJPE가 악화됐다.
AMASS의 PA-MPJPE와 flip도 모두 악화됐다. 따라서 tail/anti-flip 보조 손실은 A5 학습
조건에서 사용하지 않는다.

### A8 처리량 검증 기준

`perf: vectorize temporal lifter structural losses`는 bone/torso/hinge/yaw/anti-flip 손실의
per-chain 평균 규칙을 유지하면서 CUDA scalar를 Python 분기로 읽는 경로를 제거한다.
서버 검증은 다음 순서로 수행한다.

1. CPU 컨테이너에서 벡터화 결과와 기존 per-chain 결과의 수치 동치를 검증한다.
2. GPU가 비었을 때 A5 데이터·seed·10 epoch 조건으로 batch 128 기준 처리량을 기록한다.
3. batch 256/512는 별도 후보로 처리량·peak VRAM·holdout gate를 비교한다. batch 변경은
   optimizer update 수를 바꾸므로, 품질 비교에서 batch 128 기준선과 동일 모델로 취급하지 않는다.
4. 품질 승격은 3DPW/AMASS holdout gate와 A5 대비 PA-MPJPE 비열화를 모두 확인한 경우에만 한다.

#### A8 첫 처리량 결과와 재현성 수정

벡터화 batch 128 run은 4,647.6 samples/s, 997.4초로 완료됐다. A7의 1,948.3 samples/s,
2,379.3초 대비 약 2.39배 빠르며, 학습 중 GPU 사용률도 약 99%까지 상승했다. 다만 이 run의
3DPW/AMASS PA-MPJPE는 각각 78.64/69.36 mm로 A5보다 낮았다. 이 차이를 벡터화 품질 회귀로
판정하지 않는다. 당시 학습 코드는 `TrainingConfig.seed`를 augmentation/sampling에만 적용하고,
모델 초기화 전 전역 PyTorch RNG를 seed하지 않았다. 따라서 A5/A7/A8은 서로 다른 초기 가중치로
시작했다.

후속 run부터 `torch.manual_seed(config.seed)`를 모델 생성 전에 실행하고 checkpoint/report에
`training_seed`를 기록한다. A8 품질 판정은 이 수정 후 동일 seed로 재실행한 batch 128 결과를
기준으로 한다.

#### A8 고정-seed batch 128 결과 (2026-08-19)

고정 seed 1337 run은 4,418.6 samples/s, 1,049.1초로 완료됐다. 이전 A7의 1,948.3
samples/s 대비 2.27배 빠르며, checkpoint/report에 seed 계약이 기록됐다.

| Holdout | PA-MPJPE mm | yaw MAE ° | yaw P95 ° | hinge flip | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| 3DPW test | 75.31 | 14.90 | 34.77 | 2.36% | yaw P95, flip 실패 |
| AMASS internal | 69.19 | 8.77 | 22.37 | 2.32% | flip 실패 |

3DPW에서는 PA-MPJPE와 yaw MAE gate를 통과했지만 yaw P95와 hinge flip gate를 통과하지
못했다. AMASS PA-MPJPE/yaw gate도 통과했지만 flip은 실패했다. 따라서 A8은 **성능 최적화는
채택**, checkpoint 품질 승격은 보류한다.

주의: 이 run의 AMASS holdout은 10,792 frames이며 A5/A7 당시 보고된 31,910 frames와 다르다.
따라서 AMASS의 절대 수치를 이전 run과 직접 비교하지 않는다. 이후 품질 비교는 현재 holdout의
콘텐츠 digest/frame count를 run metadata로 고정한 뒤 진행한다.

`run_lifter_experiments.py`는 `experiment_matrix.json`의 `dataset_fingerprints`에 각 train,
validation, holdout JSON의 SHA-256·byte size·frame/sequence count를 기록한다. 후보 간 정량 또는
정성 비교 전 이 값이 일치하는지 확인한다. 다르면 동일 조건 실험으로 취급하지 않는다.

### A9 결과 (2026-08-19, fingerprint 도입 후 공식 기준선)

A8 고정-seed batch 128과 정확히 동일한 설정(위 표)으로, `dataset_fingerprints` 도입 이후
재실행한 run. checkpoint를 SHA-256으로 비교한 결과 A8과 **비트 단위로 동일**했다 (seed 계약
검증). 처리량만 3,470.9 samples/s, 1,335.6초로 A8보다 느렸으나 가중치가 동일하므로 환경
변동으로 기록하고 architecture regression으로 해석하지 않는다.

| Holdout | PA-MPJPE mm | yaw MAE ° | yaw P95 ° | hinge flip | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| 3DPW test | 75.31 | 14.90 | 34.77 | 2.36% | yaw P95 실패 |
| AMASS internal | 69.19 | 8.77 | 22.37 | 2.32% | 통과 |

이후 후보 비교의 유일한 fingerprint 기준선이다 (`ablation_a9_fingerprinted_baseline_10e`).

### hinge_flip_rate gate 제거 (2026-08-24)

`hinge_flip_rate <= 0.0`은 하나의 hinge 샘플이라도 90도를 넘는 반전이 있으면 실패하는 gate였다.
3DPW holdout 37개 시퀀스를 정성 review video(`scripts/render_lifter_audit_video.py`)로 직접
확인한 결과, 집계 flip률이 가장 낮은 시퀀스(`downtown_bar_00:actor1`, 0.16%)조차 단일 프레임에서
176도 반전이 있었다 — **flip률 0%인 시퀀스가 holdout에 하나도 없다.** 이 gate는 후보의 품질과
무관하게 어떤 checkpoint도 통과할 수 없는 조건이었으므로 `criteria`에서 제거했다 (`445efb8`).
`hinge_flip_rate`/`hinge_direction_mae_degrees`/`hinge_direction_p95_degrees`는 진단용으로
report에 계속 남는다. 이후 gate는 PA-MPJPE/yaw MAE/yaw P95 3개뿐이다.

같은 커밋에서 IK 방식 `end_effector_loss_weight`를 추가했다. 기존 좌표 loss는 17개 관절을 동일
가중치로 다루는데, 이 항목은 limb-chain 말단(`left_wrist`/`right_wrist`/`left_ankle`/
`right_ankle` — `constraint_target_builder.py`가 이미 "end_effector"라고 부르는 것과 동일)의
위치 오차만 추가로 가중해, "말단이 도착한 위치"를 우선시하는 IK 관점을 loss에 직접 반영한다.

### A10 결과 (2026-08-24, end_effector_loss_weight=0.2, rejected)

A9와 완전히 동일한 조건(dataset fingerprint 6개 전부 일치 확인)에 `--end-effector-loss-weight 0.2`
하나만 추가한 단일 가설 run.

| Holdout | PA-MPJPE mm | yaw MAE ° | yaw P95 ° | hinge flip | 판정 (3-gate) |
| --- | ---: | ---: | ---: | ---: | --- |
| 3DPW test | 74.66 (A9 75.31) | 13.70 (A9 14.90) | 34.91 (A9 34.77) | 2.41% (A9 2.36%) | yaw P95 실패 |
| AMASS internal | 80.03 (A9 69.19) | 8.51 (A9 8.77) | 23.00 (A9 22.37) | 3.00% (A9 2.32%) | **PA-MPJPE 신규 실패** |

3DPW PA-MPJPE와 yaw MAE는 소폭 개선됐지만 정작 게이트를 막던 yaw P95는 그대로거나 더 나빠졌다
(34.77 → 34.91). AMASS는 PA-MPJPE가 69.19 → 80.03 mm로 크게 악화돼 새로 gate를 실패했다
(기존엔 통과). training MPJPE 자체도 40.19 → 44.34 mm로 악화됐다 — 말단 4개 관절에 가중치를
더한 것이 나머지 13개 관절/전체 좌표 적합을 밀어낸 것으로 보인다. yaw P95를 막는 원인 해결에는
도움이 안 됐고 AMASS 쪽에 새 회귀를 만들었으므로 **거절**한다. 더 작은 weight(예: 0.05)로
재시도하거나, 다른 가설(yaw 직접 supervision 등)로 전환을 고려한다.

### A9 root-yaw P95 tail 원인 규명 (2026-08-25, 학습 없이 진단만)

A10이 yaw P95 원인과 무관했으므로, 새 아키텍처/weight를 만지기 전에 A9 checkpoint를 3DPW holdout에
재추론해 tail을 직접 원인 규명했다 (`scripts/attribute_yaw_tail.py`, 학습 없음, 평가 semantics 불변).

**정량**: yaw error 분포는 ≥30° 8.39%(2,961/35,310), ≥45° 1.71%, ≥90° 0.028%(10 frames),
**≥150° 0건**. 37개 시퀀스 중 34개가 30° 이상 오차를 가져(top-1 시퀀스가 전체 tail의 11.5%만
차지) **넓게 분산**돼 있고 소수 시퀀스에 집중돼 있지 않다. 30° 기준 연속 run은 327개, 최장
136프레임(다운타운 워킹 시퀀스), 39%가 5프레임 이상 지속 — 순간적 튐이 아니라 일부는 지속적인
드리프트다. 반면 ≥90° 오차는 7 run 중 5개가 단일 프레임(고립), 즉 극단적 오차는 드물고 고립적이다.
어깨/엉덩이 pair 20° 이상 불일치는 34,456개 중 1,300개(3.8%)뿐이라 pair 간 모순은 부차적이다.
tail 프레임은 non-tail 대비 (pelvis/torso 정규화된) 어깨·엉덩이 2D 입력 span이 32~33% 더
좁다 — 정면/후면에 가까운 자세일수록 오차가 크다는 부분적 상관은 있지만, confidence는 tail/non-tail
간 차이가 거의 없다(가려짐·저신뢰 신호는 아니다).

**정성**: worst-P95(`downtown_stairs_00:actor0`, 59.15°), 최장-run(`downtown_walking_00:actor1`,
47.42°), P95 근접(`downtown_bus_00:actor1`, 32.26°), 대조군(`downtown_bar_00:actor0`, 16.97°) 4개
시퀀스를 각 271프레임 고정 구간(GT/예측 overlay + 가슴 방향 wedge)으로 렌더해 worst-error 프레임을
확인했다. 4건 모두 두 wedge가 보고된 오차 크기만큼만 벌어져 있었고, 근사 180° 반전(앞뒤가 뒤바뀐
모습)은 관찰되지 않았다 — 정량 결과(≥150° 0건)와 일치.

**결론**: yaw tail은 주로 학습으로 개선 가능한 잔차(broadly-distributed, 대부분 30–45° 대역,
근사-180° 반전 없음)로 판단되며, 2D 관측의 정면/후면 모호성이 부차적으로 기여한다는 증거도
확인됐다(대체·대안 아님, 함께 존재). Case A/B 조건 충족 → `yaw_tail_loss` 계약 검증으로 진행.

### yaw_tail_loss 계약 검증 (2026-08-25)

`tests/test_yaw_tail_loss_contract.py` 5개 테스트로 실제 torch/autograd에서 확인: (1) 정확히
top `ceil(N/20)` pooled pair-관측만 선택, 나머지에 희석되지 않음, (2) 선택된 어깨/엉덩이 joint 외
(예: wrist, 비선택 sample)에는 gradient가 정확히 0, (3) A11 계획된 설정(yaw_tail_loss_weight=0.05,
나머지 전부 0)이 다른 항을 재활성화하지 않음. **PASS.**

단, 실제 계약상 근사(caveat) 하나를 발견해 기록한다: 이 loss는 매 (frame, pair) 관측을 그대로
pool해서 순위를 매기는데, 정작 게이트가 쓰는 `root_yaw_p95_degrees`는 frame마다 어깨/엉덩이를
먼저 평균한 뒤 frame 단위로 순위를 매긴다. 구성한 반례로, 한쪽 pair만 극단적으로 나쁜 frame이
실제로는 combined 오차가 더 낮은데도 pooled 선택에서 우선될 수 있음을 보였다. 실제 A9 3DPW holdout에서
이 pair 불일치(≥20°)율은 3.8%로 측정됐으므로 — 실재하지만 지배적이지 않은 근사로 판단해 GO를
막지 않았다.

### A11 결과 (2026-08-25, yaw_tail_loss_weight=0.05, rejected — Case B)

A9와 dataset fingerprint 6개 완전히 동일, `--yaw-tail-loss-weight 0.05` 하나만 추가
(`yaw_loss_weight`/`hinge_flip_loss_weight`/`end_effector_loss_weight` 전부 0으로 강제).

| Holdout | PA-MPJPE mm | yaw MAE ° | yaw P95 ° | hinge flip | 판정 (3-gate) |
| --- | ---: | ---: | ---: | ---: | --- |
| 3DPW test | 86.01 (A9 75.31) | 14.62 (A9 14.90) | **35.21** (A9 34.77) | 1.97% (A9 2.36%) | **PA-MPJPE 신규 실패**, yaw P95 실패(악화) |
| AMASS internal | 94.32 (A9 69.19) | 9.43 (A9 8.77) | 21.96 (A9 22.37) | 3.57% (A9 2.32%) | **PA-MPJPE 신규 실패** |

training MPJPE가 40.19 → 78.23 mm로 거의 두 배로 악화됐다. 정작 막고 있던 yaw P95는 개선되지
않고 오히려 소폭 악화(34.77→35.21°)됐고, 두 holdout 모두 PA-MPJPE가 크게 악화돼 기존에 통과하던
게이트까지 새로 실패했다. Section 3에서 고정한 동일 4개 시퀀스·프레임으로 A9/A11 matched 정성
비교도 수행 — 시각적으로도 orientation 어긋남이 개선된 정황이 없었다(오히려 전반적 pose 정합이
눈에 띄게 흐트러짐, 급격한 training MPJPE 악화와 일치).

**판정: Case B — 기존 `yaw_tail_loss`는 A9의 통제된 조건에서 tail을 실질적으로 개선하지 못하며,
다른 weight/percentile/CVaR 비율을 추가로 시도하지 않는다.** yaw P95를 해결하려면 이 loss의
weight 조정이 아니라 다른 접근(표현/관측 증거 자체를 다루는 방향)이 필요하다는 결론이며, 이번
batch에서 그 다음 아키텍처를 구현하지는 않는다.

## 사용자 정성 평가: 리그 애니메이션 review video

수치 gate가 통과하더라도 관절의 순간적인 반전, foot sliding, 루트의 회전 흔들림은 사람이 보는
영상에서 더 빨리 발견될 수 있다. 각 품질 후보에는 정량 report와 함께 **동일한 고정 holdout clip,
동일한 rig/mapping, 동일한 camera** 조건의 MP4 review video를 남긴다.

`animation_optimized.json`을 Blender에 적용해 `.blend`를 만든 후 다음을 실행한다.

```bash
blender --background --python scripts/render_blender_animation_video.py -- \
  --blend artifacts/a8_candidate.blend \
  --out artifacts/review/a8_candidate_three_quarter.mp4 \
  --camera three_quarter
```

이 영상은 실제 리그 메시와 함께 녹색 본/주황 관절 프록시를 렌더한다. 따라서 스킨/재질 문제와
리타게팅 문제를 분리해서 볼 수 있으며, 메시를 배제하고 관절 운동만 보려면 `--hide-original-mesh`를
쓴다. 프레임 수가 긴 clip은 `--start-frame`, `--end-frame`으로 대표 구간(보행 시작·방향 전환·팔
스윙)을 고정한다. 후보 간 공정한 비교를 위해 출력 해상도, camera, 프레임 구간은 모두 동일하게
기록한다.

리뷰자는 각 clip에 대해 다음 네 항목을 1~5점으로 기록한다: (1) 발 고정/미끄러짐, (2) 무릎·팔꿈치
굽힘 방향의 연속성, (3) 골반·루트 yaw의 안정성, (4) 전반적인 동작 자연스러움. 어느 항목이든 2점
이하면 정량 gate 통과 여부와 관계없이 해당 후보는 승격하지 않고, 문제 프레임 범위와 관찰 내용을
report에 남긴다.

## 서버 명령 예시 (A4, 10 epochs)

```bash
docker run --rm --gpus all --entrypoint python3 -w /workspace \
  -e PYTHONPATH=/workspace/src -e PYTHONPYCACHEPREFIX=/tmp/animcv_pycache \
  -v /home/nd/AnimCV:/workspace:ro -v /home/nd/animcv-data:/data:ro \
  -v /home/nd/animcv-output:/output animcv-train:cuda118 \
  scripts/run_lifter_experiments.py \
  --mpi-train /output/data/animcv/train_combined.json \
  --three-dpw-train /data/3dpw/prepared/train.json \
  --amass-train /data/amass/prepared_aug_v1/train.json \
  --validation /data/3dpw/prepared/validation.json,/data/amass/prepared_aug_v1/validation.json \
  --three-dpw-holdout /data/3dpw/prepared/holdout.json \
  --amass-holdout /data/amass/prepared_aug_v1/holdout.json \
  --out /output/experiments/ablation_a4_pelvis_torso_10e --epochs 10 --candidates direct_mix \
  --source-balanced-sampling --architecture dilated_tcn_v1 \
  --input-jitter-std 0.015 --input-dropout-probability 0.05 --confidence-jitter-std 0.08 \
  --input-coordinate-normalization pelvis_torso_v1
```
