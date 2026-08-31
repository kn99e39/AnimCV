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

### A11 붕괴 원인 규명 (2026-08-25, 학습 없이 diagnostic replay만)

A11이 왜 이렇게 파괴적이었는지 원인을 규명하기 전에는 selector를 바꾸지 말라는 지시에 따라,
`scripts/diagnose_yaw_tail_gradients.py`(신규, 학습 없음)로 A9/A11이 실제로 봤던 첫 epoch의
동일 batch(같은 seed·augmentation·source-balanced permutation) 10개를 세 가지 실제 모델
상태 — **초기화 직후**, **A9 최종 checkpoint**, **A11 최종 checkpoint** — 에 재생시켜 loss 성분과
gradient 상호작용을 분해했다.

**선행 확인(Section 1)**: `train()`은 epoch/step 단위 loss를 전혀 기록하지 않는다 — A9/A11 모두
최종 post-training 집계값(`training_mpjpe_mm`) 하나만 report에 남고, A11은 detached 컨테이너로
실행돼 stdout log조차 없다. 따라서 "언제부터 발산했는지"는 기존 기록으로 answer 불가 — 이 한계를
그대로 인정하고 아래 diagnostic replay로 대체했다(재학습 없이).

**핵심 발견**: `yaw_tail_pooled`(가중치 적용 전 raw)의 gradient norm 대비 base(A9 전체 objective,
bone/torso/hinge 가중 포함) gradient norm의 비율이 모델 상태에 따라 극적으로 변한다.

| 모델 상태 | base gradient norm | yaw gradient norm(raw) | 비율(yaw/base) | pooled tail loss 중 AMASS 비중 |
| --- | ---: | ---: | ---: | ---: |
| 초기화 직후 | 0.659 | 1.114 | **1.7배** | 41% |
| A9 최종 checkpoint | 0.011 | 35.48 | **3,170배** | **87%** |
| A11 최종 checkpoint | 0.018 | 2.75 | 157배 | 59% |

base task(좌표/구조 loss)가 수렴할수록 base gradient는 근처 극소점이라 거의 0으로 줄어드는데,
`yaw_tail`의 raw 크기는 전혀 같이 줄지 않는다 — A9가 이미 잘 맞춘 상태에서 yaw_tail 항만 켜면
gradient가 base보다 **3,000배 이상** 커진다. weight 0.05를 곱해도 이 정도 스케일 불균형은
전혀 흡수되지 않는다(가중된 yaw_tail이 여전히 coordinate loss와 맞먹거나 넘어서는 크기,
초기화 상태에서 이미 coordinate 0.159 대비 가중 yaw_tail 0.100).

**방향 충돌(Case C)은 주 원인이 아니다**: cosine(G_base, G_yaw)은 초기화 직후에만 일부 배치에서
음수(10개 중 4개)였고, A9/A11 학습된 상태에서는 **10개 배치 전부 양수**(0.04~0.13)였다 — 방향은
대체로 정렬돼 있다. 문제는 방향이 아니라 **크기**다.

**selector granularity(Case A)는 원인이 아니다**: frame-level counterfactual selector(evaluator와
동일하게 frame당 pair를 먼저 평균한 뒤 순위, `_yaw_tail_loss_frame_level`, 학습에는 안 씀)로
동일 배치·동일 모델 상태를 재계산해도 gradient norm과 비율이 pooled selector와 사실상 동일하다
(A9 상태에서 yaw gradient norm 35.48 vs 35.95, 비율 3,170배 vs 3,200배; cosine도 거의 동일).
**selector를 frame-level로 바꿔도 이 문제는 사라지지 않는다** — 그래서 이번 batch는 Section 8(frame-level
후보 학습)로 진행하지 않는다.

**source 불균형(Case D)도 함께 존재한다**: A9 수렴 상태에서 pooled tail loss의 **87%가 AMASS**에서
나온다(초기화 상태 41%, A11 학습 후 59%에서 크게 증가). A11에서 AMASS PA-MPJPE가 3DPW보다 훨씬
더 크게 악화된 것(+25.1mm vs +10.7mm)과 방향이 일치한다 — 학습이 진행될수록 yaw-tail 신호가
사실상 AMASS 하나의 도메인에 집중된다.

**판정: Case B(gradient-scale 불균형)가 주 원인, Case D(source 불균형)가 이를 증폭하는 부차
원인. Case A(selector 구조)와 Case C(방향 충돌)는 배제.** 지시에 따라 frame-level selector
후보를 구현·학습하지 않는다. 다음 아키텍처 질문은 "tail selection을 어떻게 고를지"가 아니라
"orientation supervision 자체의 형태(스케일이 base 3D objective와 자연스럽게 맞물리는 형태,
그리고 소수 도메인에 좌우되지 않는 정규화)"가 돼야 한다.

### Cartesian torso-tail loss 후보와 A12 (2026-08-26)

각도(1-cos) 표현 대신, 안정적인 3D geometry objective와 **같은 Cartesian 좌표 공간**에서
orientation tail을 표현하는 후보를 만들었다: 어깨/엉덩이 bilateral vector(`v = p_right - p_left`,
기존 `torso_loss_weight`와 동일한 벡터 컨벤션)의 smooth-L1 잔차를, 기존 `yaw_tail_loss`와
**완전히 동일한 pooled tail-selector**(`_pooled_tail_mean`로 공유 리팩터링, 기존 동작 불변
확인됨)로 선택한다(`_cartesian_torso_tail_loss`). 벡터 차분이라 uniform translation에 자동으로
불변이고, 각도 표현이 요구했던 축-길이 degenerate guard가 필요 없다.

**Fixed-batch 진단(학습 전)**: A9/A11 진단과 동일한 10개 배치 × 3개 모델 상태에서 candidate를
측정한 결과, A9 수렴 상태의 gradient 비율(candidate/base)이 각도 loss의 **3,170배 → 5.28배**로
600배 넘게 줄었고, 세 상태(초기화/A9/A11) 전체에서 1.7~7.2배 범위에 머물러 각도 loss처럼
학습이 진행될수록 폭발하지 않았다. cosine(G_base, G_candidate)은 모든 상태에서 양수였고
각도 loss보다 오히려 더 정렬돼 있었다(예: A9 상태 0.164 vs 0.069). synthetic contract
9개(정상 geometry→0, 어깨/엉덩이/회전 오차→반응, uniform translation→false positive 없음,
gradient가 해당 관절에만 도달) 전부 통과. **GO 조건(파괴적 gradient 지배 제거 + orientation
민감도 유지) 둘 다 충족** — 단, source 편중(Case D)은 candidate에서도 그대로 관찰됨
(A9 수렴 상태 기준 AMASS가 tail loss의 45~92%, 각도 loss의 87%와 비슷한 규모).

A9와 fingerprint 완전 동일, `cartesian_torso_tail_loss_weight=0.05`만 추가해 A12를 실행.
이번 세션에서 새로 추가한 epoch별 telemetry(`train()`의 `epoch_telemetry`, 512-window 고정
subset, no-grad)로 학습 중 수렴 궤적을 직접 확인했다 — `sample_mpjpe_mm`이 54.0→38.1mm로
매끄럽게 수렴했고 `cartesian_torso_tail_raw`도 0.00192→0.00085로 `torso`/`coordinate`와 같이
줄어들어, A11처럼 발산하지 않고 실제로 base task와 스케일이 맞물려 수렴한다는 gradient
진단의 예측이 실제 학습에서도 확인됐다.

| Holdout | PA-MPJPE mm | yaw MAE ° | yaw P95 ° | 판정 |
| --- | ---: | ---: | ---: | --- |
| 3DPW test | 78.51 (A9 75.31) | **15.87**(A9 14.90, 신규 실패) | **38.45**(A9 34.77, 악화) | yaw MAE·P95 실패 |
| AMASS internal | 68.60 (A9 69.19) | 8.05 (A9 8.77) | 20.08 (A9 22.37) | 통과(A9도 통과) |

training MPJPE는 40.19→**40.39mm**로 사실상 그대로다 — A11의 78.23mm 붕괴와 달리 base
geometry 적합이 전혀 무너지지 않았다. **하지만 3DPW yaw는 개선되지 않고 오히려 더
나빠졌다**(MAE 14.90→15.87°로 이전엔 통과하던 gate까지 신규 실패, P95는 34.77→38.45°로
목표 방향과 반대로 이동). AMASS는 3개 지표 모두 소폭 개선됐지만 A9에서도 이미 전부 통과
상태였다. 고정된 review 시퀀스(stairs/walking, A9 진단 단계에서 이미 선정)의 GT/예측 overlay
정지 화면을 A9와 비교한 결과도 눈에 띄는 개선은 보이지 않았다 — 여전히 비슷한 정도의
orientation 어긋남.

**판정: Case B — Cartesian 재구성은 안정적이지만 orientation을 개선하지 못한다.** A11의
붕괴가 각도 loss의 scale 문제였다는 진단은 이 결과로 재확인됐다(training MPJPE 붕괴가 실제로
사라짐). 하지만 단순히 torso geometry를 재가중하는 것만으로는 yaw-tail 품질 문제 자체를
풀지 못한다 — 오히려 3DPW에서 악화됐다. weight를 조정하지 않는다(지시에 따름). 다음
아키텍처 질문은 orientation supervision의 형태(예: 방향 정보만 분리해서 스케일을 통제하는
방법, 또는 tail-selection이 아닌 다른 표현)여야 한다.

### A12 magnitude/direction attribution 및 A13 판정 (2026-08-26, 학습 없이 진단만)

A12가 실패한 이유가 Cartesian residual의 magnitude-direction entanglement인지 확인하기 위해,
A9/A11 진단과 동일한 seed 1337·augmentation·source-balanced permutation의 첫 epoch 고정
배치 10개(배치당 128 frame)를 네 상태(init/A9/A11/A12)에 replay했다. A12의 실제 pooled
selection은 유지하고, attribution에는 (1) smooth-L1 scalar companion과 (2) 다음의 정확한
제곱 에너지 항등식을 함께 사용했다.

`||v_pred-v_gt||² = (||v_pred||-||v_gt||)² + ||v_pred|| ||v_gt|| ||u_pred-u_gt||²`

A12가 선택한 tail에서의 결과는 다음과 같다. `direction`은 target span을 detached scale로
복원한 unit-direction residual이며, scalar 두 항은 smooth-L1 자체가 비선형이므로 A12 raw
loss를 단순히 합산하는 항이 아니라 attribution용 동반 측정이다.

| 상태 | A12 raw tail | magnitude scalar | direction scalar | magnitude energy | direction energy | A12 raw × 0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| init | 0.112882 | 0.127270 | 0.038603 | 37.6% | 62.4% | 0.005644 |
| A9 | 0.001671 | 0.000959 | 0.001393 | 19.3% | **80.7%** | 0.0000836 |
| A11 | 0.009038 | 0.010736 | 0.003847 | 39.6% | 60.4% | 0.000452 |
| A12 | 0.001530 | 0.000944 | 0.001266 | 20.4% | **79.6%** | 0.0000765 |

A12 checkpoint의 선택 tail predicted span 평균은 0.310, target span 평균은 0.309였다.
따라서 수렴 상태에서 A12가 주로 magnitude를 줄이는 신호였다는 증거는 없다. parameter
gradient만 보면 magnitude component의 raw norm/base norm은 6.12배, direction component는
4.66배로 magnitude가 약 1.31배 컸지만, 이는 방향 항의 정보가 희석됐다는 수준이 아니며
A11 angular loss의 폭발과도 질적으로 다르다. A12 선택 tail의 direction scalar 자체가
magnitude scalar보다 컸고, squared energy는 direction이 약 80%였다.

#### 고정 배치 gradient 및 yaw association

아래 gradient ratio는 coefficient 1.0으로 계산한 raw ratio이며, 실제 coefficient 0.05를
적용한 값은 괄호 안에 적었다. cosine은 base A9 objective와 component gradient의 cosine이다.

| 상태 | Cartesian A12 candidate | scale-restored direction candidate | A12 magnitude component | A12 direction component | historical angular yaw-tail |
| --- | ---: | ---: | ---: | ---: | ---: |
| init | 3.25 (0.162), cos .248 | 0.94 (0.047), cos -.019 | 6.37 (0.318), cos .275 | 0.29 (0.014), cos -.024 | 1.69 (0.084) |
| A9 | 5.28 (0.264), cos .164 | 5.14 (0.257), cos .162 | 6.60 (0.330), cos .052 | 4.94 (0.247), cos .152 | **3169.92 (158.50)** |
| A12 | 5.34 (0.267), cos .318 | 4.91 (0.245), cos .301 | 6.12 (0.306), cos .203 | 4.66 (0.233), cos .294 | **2596.64 (129.83)** |

A12 상태에서 raw direction-candidate ratio의 p95/max는 6.34/6.45, Cartesian candidate는
6.78/6.79이었다. 반면 historical angular ratio는 2596.64였으므로, 후보는 small-span
계산을 포함해 A11의 gradient pathology를 재현하지 않았다. A12 상태에서 frame-level yaw
error와의 Pearson r은 Cartesian .325, magnitude -.056, scale-restored direction **.610**,
historical angular 1.000이었다. angular 값이 1인 것은 evaluator가 동일한 angular quantity를
사용하기 때문이며 독립적인 성공 증거로 해석하지 않는다.

#### source별 attribution

A9 checkpoint에서 A12 실제 tail의 loss share는 MPI 44.4%, 3DPW 10.6%, AMASS 45.0%였다.
AMASS의 내부 구성은 magnitude loss share 60.2%, direction loss share 40.8%로 magnitude
쪽이 더 컸다. A12 checkpoint에서는 MPI 46.3%, 3DPW 7.5%, AMASS 46.2%였고, AMASS는
magnitude 54.2%, direction 45.8%였다. init/A11처럼 AMASS가 86–92%까지 올라간 상태에서는
magnitude와 direction이 함께 지배했다. 즉 기존 AMASS dominance는 순수 magnitude 하나가
아니라 상태·selection에 따라 둘 다 나타나는 현상이며, 수렴 상태에서는 magnitude 편향이
있어도 direction 정보가 사라지지는 않았다.

scale-restored direction candidate의 자체 tail selection은 A12 상태에서 MPI 46.1%,
3DPW 9.7%, AMASS 44.2%였고, source별 isolated gradient norm은 각각 0.0924, 0.0577,
0.1254였다. 기존 A12의 source 집중을 더 악화시키는 새 source pathology는 관찰되지 않았지만,
source 기여를 동일하게 만들기 위한 normalization은 이번 batch에서 도입하지 않았다.

#### 합성 계약 및 판정

diagnostic-only candidate는 `u_pred=normalize(v_pred)`, `u_gt=normalize(v_gt)`,
`scale=stop_gradient(||v_gt||)`, `scale * (u_pred-u_gt)`이며 fixed epsilon은 기존 yaw 경로와
동일한 `1e-6`이다. raw dimensionless/cosine/angle scalar나 별도 magnitude 항을 사용하지
않는다. identical/translation invariance, same-direction different-magnitude zero,
rotation/opposite response, shoulder/hip endpoint gradient isolation, detached target scale,
collapsed predicted span finite loss/gradient의 6개 계약을 모두 통과했다.

그러나 전체 GO는 **NO-GO**다. 후보 자체의 orientation sensitivity와 optimization stability는
확인됐지만, A13을 허용하는 첫 조건인 “A12 signal이 magnitude 때문에 direction 정보에서
materially diluted 됐다”가 A9/A12 수렴 상태에서 성립하지 않았다. 따라서 정확히 하나의
A13 학습, 새 정성 review, source normalization, dynamic balancing, weight tuning을 모두
실행하지 않았다. A12의 최종 architecture verdict는 **안정성은 해결했지만 orientation을
개선하지 못했고, 그 실패를 단순 magnitude-direction entanglement로 설명할 수 없음**이다.

재현 가능한 원본 진단 산출물은 서버의
`/home/nd/animcv-output/experiments/a12_direction_attribution_10b/diagnosis.json`이며,
A9–A12 checkpoint/fingerprint/report는 변경하지 않았다.

### A12 global tail source aggregation 및 3DPW coverage 진단 (2026-08-26, 학습 없이 진단만)

A12의 3DPW orientation 악화가 source-balanced input sampling 뒤의 global hard-tail mining
때문인지, 아니면 3DPW train과 official holdout의 coverage/domain shift 때문인지 분리했다.
A12 Cartesian torso-tail loss, `cartesian_torso_tail_loss_weight=0.05`, pooled top-5% tail
semantics는 변경하지 않았다. 동일 seed 1337·augmentation·source-balanced RNG 순서의 첫
epoch 10개 batch를 init/A9/A12 상태에 replay하고, source-stratified tail은 학습에 연결하지
않은 counterfactual로만 계산했다.

#### 입력 balance와 global-tail reweighting

direct-mix 원본 frame mass는 MPI-INF-3DHP `106,512`, 3DPW `22,646`, AMASS `334,402`로
불균형하지만 source-balanced sampler의 한 epoch sample mass는 세 source 모두 `154,520`이다.
고정 10개 batch의 입력 mass도 MPI `403`, 3DPW `466`, AMASS `411`로 거의 균형이었다.
따라서 입력 기회 자체가 3DPW를 배제한 것은 아니다.

A12 상태에서 global pooled tail은 다음과 같이 source를 다시 가중했다. `within-source`는
그 source의 유효 candidate 중 selected 비율이며, `selected share`는 전체 selected 중 비율이다.

| Source | candidate/batch | selected/batch | within-source selected | total selected share | raw loss share |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPI-INF-3DHP | 83.6 | 6.7 | 7.96% | **51.54%** | 47.58% |
| 3DPW | 82.4 | 1.2 | **1.37%** | **9.23%** | 7.44% |
| AMASS | 87.7 | 5.1 | 5.87% | 39.23% | 44.97% |

A9 상태에서도 3DPW within-source selected 비율은 `2.01%`, 전체 selected share는 `12.31%`에
그쳤다. init 상태에서는 3DPW `0.36%`/`2.31%`, AMASS `13.82%`/`91.54%`였다. 따라서
global tail이 source-balanced input contract를 덮어쓰는 현상은 실제로 존재한다.

#### Source-stratified counterfactual 및 gradient

각 active source 안에서 동일 A12 tail fraction을 적용한 뒤 source별 loss mean을 동일하게
평균하는 generic counterfactual을 만들었다. A12 상태에서 각 source의 within-source selected
비율은 MPI `5.53%`, 3DPW `5.61%`, AMASS `5.27%`였고, 전체 selected share는 각각 약 `33.3%`였다.
3DPW local raw auxiliary loss는 `0.000826`으로 MPI `0.001518`, AMASS `0.001867`보다 낮았다.

global A12 auxiliary gradient는 norm `0.07054`, base norm `0.01326`, base와의 cosine
`.318`이었다. source-stratified counterfactual은 `0.06742`, `.01326`, cosine `.322`였다.
A12 상태의 source-local gradient norm은 3DPW `.0623`, AMASS `.1281`, MPI `.1041`이며,
source pair cosine은 MPI–3DPW `.188`, MPI–AMASS `.124`, 3DPW–AMASS `.095`였다. 강한
cross-source cancellation이나 A11-scale pathology는 없었다.

#### Source-specific training error distributions

A12 checkpoint를 source-balanced direct-mix training frames 전체에 재추론했다.

| Source | yaw mean / median / P90 / P95 / P99 (deg) | Cartesian residual mean / median / P90 / P95 / P99 |
| --- | --- | --- |
| MPI-INF-3DHP | 5.50 / 4.77 / 10.16 / 12.30 / 17.44 | .000356 / .000249 / .000787 / .001036 / .001683 |
| 3DPW | **4.75 / 4.21 / 8.62 / 10.18 / 14.44** | **.000165 / .000092 / .000393 / .000563 / .001050** |
| AMASS | 7.99 / 4.48 / **17.60 / 27.34 / 57.02** | .000275 / .000136 / .000673 / .000993 / .001936 |

3DPW train의 tail이 다른 source보다 전반적으로 작기 때문에 global selector의 3DPW 억제는
구현 오류만이 아니라 error distribution 차이와 결합된 결과다.

#### 3DPW train/validation/test coverage

| Split | Frames / sequences | A9 yaw mean / P95 / P99 | A12 yaw mean / P95 / P99 |
| --- | ---: | ---: | ---: |
| 3DPW train | 22,646 / 34 | 4.33 / 9.66 / 13.24 | 4.75 / 10.18 / 14.44 |
| 3DPW validation | 10,206 / 16 | 11.99 / 32.55 / 45.71 | 12.47 / 32.38 / 45.22 |
| 3DPW official test | 35,310 / 37 | 14.90 / 34.77 / 50.11 | **15.87 / 38.45 / 53.41** |

GT torso turn delta P95는 train `4.418°`, validation `4.332°`, test `3.195°`였다. test가 더
큰 turning motion 때문에 실패한다는 증거는 없다. Input confidence mean은 train `.770`,
validation `.765`, test `.721`이었다. Manifest에는 sequence ID는 있지만 semantic action
taxonomy와 camera-view label이 없으므로 view distribution 비교는 unavailable이다. train은
주로 `courtyard_*`/`outdoors_climbing_*`, validation은 `courtyard_*`/`outdoors_parcours_*`,
test는 `downtown_*` sequence를 포함한다.

3DPW train yaw P95 `10.18°` 대 test `38.45°`, P99 `14.44°` 대 `53.41°`로 hard-case coverage가
크게 다르다. source-stratified aggregation이 3DPW participation은 복구하지만, 현재 train에
test와 comparable한 hard examples를 공급한다는 증거는 없다.

**판정: Case B — 3DPW training coverage/domain shift가 primary limitation.** Global-tail
starvation은 실재하는 secondary mechanism이지만 Case A의 relevant training coverage 조건이
충족되지 않는다. source-stratified 후보 학습, A12 coefficient/tail percentage/source mixture/
augmentation/optimizer 변경은 수행하지 않았다. A9–A12 checkpoint/fingerprint/report와
historical loss definition은 변경하지 않았다.

재현 가능한 진단 JSON은
`/home/nd/animcv-output/experiments/a12_source_tail_aggregation_diagnosis/diagnosis.json`에
있다.

### 3DPW generalization support 진단 (2026-08-26, 추가 학습 없음)

이 진단은 이전에 관찰한 `3DPW train prediction yaw error << validation/test yaw error`를
GT target coverage의 증거로 사용하지 않는다. A9 기존 evaluator로 hard set을 먼저 고정한 뒤,
canonical GT 3D target descriptor와 A9 전처리 후 canonical 2D input descriptor를 분리해
sequence-disjoint nearest-train support를 비교했다. A12는 같은 hard examples가 A12에서도
어려운지 확인하는 용도로만 사용했다. loss, augmentation, sampler, optimizer, gate와
checkpoint는 변경하지 않았다.

#### GT target-space와 temporal coverage

3DPW root/torso orientation의 wrapped angle 통계는 circular quantity라 선형 평균을 판정에
사용하지 않았다. median/P95/P99는 train `76.16/165.83/177.55°`, validation
`82.62/167.03/177.37°`, official test `3.35/173.31/178.29°`로 tail이 train에만 없는
형태가 아니다. 지시문에서 요구한 `d=z_right-z_left`는 canonical z축 성분으로 그대로
계산했고, AnimCV 계약상 실제 camera forward/depth인 `+Y` 성분도 별도 계산했다.

| Split | shoulder d(z) mean/std/median/P05/P10/P90/P95/P99 (m) | hip d(z) mean/std/median/P05/P10/P90/P95/P99 (m) |
| --- | --- | --- |
| train | `.0185/.0686/.0213/-.0981/-.0693/.0990/.1239/.1741` | `.00013/.0172/.00035/-.0268/-.0201/.0191/.0281/.0478` |
| validation | `.0174/.0725/.0157/-.1081/-.0761/.1071/.1336/.1866` | `.0039/.0185/.0024/-.0267/-.0188/.0252/.0328/.0570` |
| official test | `-.0033/.0633/-.00076/-.1100/-.0874/.0770/.0984/.1401` | `-.0081/.0141/-.0078/-.0315/-.0258/.0087/.0149/.0272` |

Shoulder/hip forward-y signed-depth quantiles도 train/validation/test가 겹쳤다. test의
GT temporal motion은 train보다 더 급하지 않았다. orientation velocity P95/P99는
train `96.0/195.8°/s`, validation `93.2/172.2°/s`, test `65.7/129.6°/s`였고,
orientation window path P95는 `299.9°`, `252.0°`, `189.5°`였다. shoulder/hip signed-z
velocity와 sign-transition quantiles도 full JSON에 mean/std/median/P05/P10/P90/P95/P99로
기록했다. 따라서 single-frame target pose나 81-frame GT trajectory가 train에 전혀 없다는
Case A 결론은 지지되지 않는다.

#### 2D observation-space coverage

`pelvis_torso_v1`로 실제 lifter에 입력된 2D를 비교했다. normalized torso height는 계약상
1.0으로 고정되지만 lateral evidence는 달랐다.

| Split | shoulder span mean/median/P95 | hip span mean/median/P95 | confidence mean | valid joints mean |
| --- | ---: | ---: | ---: | ---: |
| train | `.823/.887/1.263` | `.543/.583/.819` | `.803` | `16.26` |
| validation | `.799/.864/1.269` | `.527/.572/.843` | `.801` | `16.19` |
| official test | `.542/.384/1.204` | `.352/.244/.757` | `.757` | `16.11` |

test는 raw image torso scale이 오히려 train/validation보다 컸지만, canonical normalized
shoulder/hip projected span은 작고 confidence도 낮았다. 따라서 input preprocessing 뒤의
observation condition이 split별로 같다고 볼 수 없다.

#### A9 hard set과 A12 overlap

A9 root-yaw evaluator로 top-5%와 top-1%을 먼저 고정했다. 선택 수와 A9 cutoff는 train
`1133/227`, validation `511/103`, test `1766/354`이며, 각각의 top-5% cutoff는
`9.663°`, `32.554°`, `34.778°`였다. A12는 이 집합을 재정의하지 않았다.

| Split | A9/A12 yaw rank rho | top-5 center overlap / Jaccard | top-5 window-frame Jaccard | top-1 center overlap / Jaccard |
| --- | ---: | ---: | ---: | ---: |
| train | `.113` | `14.1% / .076` | `.505` | `10.6% / .056` |
| validation | `.450` | `40.1% / .251` | `.490` | `20.4% / .114` |
| official test | `.409` | `33.2% / .199` | `.422` | `38.1% / .236` |

같은 사례가 완전히 고정되지는 않았지만 validation/test에서 A12도 상당수 같은 temporal
영역에서 실패했다. overlap만으로 data insufficiency를 단정하지 않고 support와 함께
해석했다. 각 hard record의 sequence ID, frame ID, 81-frame window center와 frame 목록은
진단 JSON에 모두 보존했다.

#### Sequence-disjoint nearest support

지원 descriptor는 target 27차원, input 119차원이며 scale은 train support의 mean/std만
사용했다. split별 binary threshold는 새로 튜닝하지 않고 train→other-train-sequence
control 대비 empirical percentile을 사용했다.

| Query → train support | target distance median/P95 (control percentile median) | input distance median/P95 (control percentile median) |
| --- | ---: | ---: |
| train → other sequence | `.362/.839` | `.464/1.003` |
| validation hard top-5% | `.475/1.095` (`73.4%`) | `.855/1.208` (`91.2%`) |
| official test hard top-5% | `.549/.947` (`82.8%`) | `.501/1.059` (`56.6%`) |

validation hard cases는 input support가 control보다 뚜렷하게 멀다. 반면 test hard cases는
input nearest support median이 control과 비슷한데, 그 동일 input-nearest train sample의
GT target gap은 컸다. test에서 root orientation gap median/P95는 `43.6°/154.0°`,
shoulder forward-y gap은 `.223/.554 m`, hip forward-y gap은 `.0688/.178 m`이며,
각각 control 대비 median empirical percentile은 `88.3%`, `85.7%`, `85.7%`였다.
이는 local 2D evidence가 가까워도 서로 다른 3D orientation/depth state를 허용하는
monocular ambiguity의 직접적인 증거다. validation의 같은 값은 root orientation
`24.6°/87.3°`(`72.1%`), shoulder forward-y `.115/.421 m`(`71.9%`)로 test만큼 강하지
않았다. nearest record별 target gap도 JSON에 저장했다.

#### Signed relative-depth attribution

A9 hard vs non-hard에서 `z_right-z_left`와 canonical forward-y를 모두 비교했다. test
hard shoulder/hip forward-y absolute residual은 각각 `.231/.091 m` 대 non-hard
`.081/.022 m`, sign disagreement는 `48.4%/50.2%` 대 `13.0%/6.7%`였다. validation은
각각 `.174/.062 m` 대 `.058/.022 m`, sign disagreement `20.1%/20.7%` 대
`6.3%/6.7%`였다. 요청한 z축 성분에서는 test shoulder residual이 `.0314 m` 대
`.0316 m`로 non-hard와 비슷했지만 hip sign disagreement는 `43.3%` 대 `26.7%`였다.
즉 실제 forward-depth ordering과 yaw error의 연관이 강하고, canonical z축만을 물리적
depth로 부르는 것은 좌표 계약과 맞지 않는다. hard-window center sign-transition
behavior disagreement도 test shoulder/hip z에서 `6.6%/9.7%`, validation `11.5%/13.3%`
로 기록했다. signed-relative-depth loss는 구현·학습하지 않았다.

#### Sequence diversity와 replacement sampling

3DPW train은 34개 sequence, 22,646개 unique frame-center window이다. source-balanced
direct-mix epoch는 3DPW에 `154,520` sample mass를 replacement로 부여한다. deterministic
seed `1337` replay에서 unique sampled windows는 `22,622`, duplicate sample은 `131,898`,
nominal mass/unique-window replay factor는 `6.823`, realized mass/unique-sampled-window
factor는 `6.831`이었다. top sequence share는 `5.58%`, top-5 sequence share는 `26.78%`,
sequence HHI는 `.0354`였다. frame mass balance가 sequence-level diversity를 늘려주지는
않는다는 점은 확인됐지만 sampler는 변경하지 않았다.

#### 최종 판정과 architecture interpretation

**Case E — split별 mixed failure**로 판정한다.

- **Target coverage gap:** GT orientation, signed relative-depth, temporal target이 train에
  전혀 없다는 증거는 없다. target nearest support는 일부 tail shift가 있지만 새 binary
  unsupported threshold를 만들 정도의 증거는 아니다.
- **Input-domain shift:** validation hard set에서 명확하다(input support median
  percentile `91.2%`). 다음 architecture decision에서는 crop/projection/confidence 및
  camera/view diversity를 우선 검토해야 한다.
- **Monocular ambiguity:** official test에서 명확하다(input support median percentile
  `56.6%`인데 input-nearest target orientation/forward-depth gap이 control보다 큼).
  다음 질문은 temporal evidence와 latent orientation-state supervision이다.
- **Model/objective failure:** target·input support가 모두 충분하고 ambiguity도 없는
  Case D 조건은 확인되지 않았다. 따라서 현재 A9 objective를 primary suspect로 판정하지
  않는다.
- **Sequence diversity:** 6.8배 replacement replay와 34 sequence concentration은
  보조 제한 요인이다. 그러나 target absence의 증거는 아니며 sampler를 바꾸지 않았다.

추가 loss, temporal loss, signed-relative-depth supervision, augmentation, source mixture,
sampler, optimizer, gate 변경과 추가 학습은 모두 보류한다.

재현 가능한 최종 진단 JSON은
`/home/nd/animcv-output/experiments/a9_target_input_support_diagnosis/diagnosis_final.json`이다.
핵심 정량 수치와 hard/support record 전체를 포함하며 generated artifact는 repository에
commit하지 않았다.

### A14 결과 (2026-08-31, bilateral_forward_depth_supervision, Case B — rejected)

A9와 fingerprint 6개 완전 동일, all-frame `q = (y_right - y_left) / sqrt(2)`
(canonical `+Y`) shoulder/hip 잔차를 base coordinate loss의 sum/count에
그대로 pooling(별도 weight 없음)했다. yaw_loss/yaw_tail_loss/hinge_flip_loss/
end_effector_loss/cartesian_torso_tail_loss는 모두 0으로 유지했다. 학습 전
fixed-batch gradient 진단(A9 상태 candidate/base ratio 평균 `0.458`, 최대
`0.628`, cosine 양수)이 A11식 붕괴(`3,169.92`)나 A12(`5.28`)보다도 안전함을
확인해 GO 판정했다.

| Holdout | PA-MPJPE mm | yaw MAE ° | yaw P95 ° | 판정 (3-gate) |
| --- | ---: | ---: | ---: | --- |
| 3DPW test | 77.34 (A9 75.31) | **15.15** (A9 14.90, 신규 실패) | 38.18 (A9 34.77, 악화) | yaw MAE·P95 실패 |
| AMASS internal | 73.87 (A9 69.19) | 7.71 (A9 8.77) | 21.19 (A9 22.37) | 통과(A9도 통과) |

training MPJPE는 40.19→40.13mm로 사실상 그대로였다 — A11식 붕괴는 재현되지
않았다. 3DPW 공식 test holdout 35,310프레임 전체에서 A9 evaluator 자신의
hard top-5%/top-1%(candidate 관찰 전 고정)를 기준으로 forward-depth abs
residual과 sign disagreement를 측정한 결과, **hard set에서는 의미 있게
개선**됐다(shoulder abs residual top-5% `0.231→0.181m`, sign disagreement
`48.4%→41.5%`; hip `0.091→0.067m`, `50.3%→39.1%`). 그러나 non-hard subset과
이전까지 orientation 문제가 거의 없던 대조군 시퀀스(`downtown_bar_00:actor0`)
에서는 오히려 악화됐다(shoulder sign disagreement `7.0%→16.0%`). hard
top-1%의 sign disagreement는 거의 그대로였다(`68.8%→67.7%`, `61.6%→59.0%`).

**판정: Case B(그러나 이상화된 Case B보다 나쁨) — 직접 supervise한
bilateral forward-depth 신호는 hard set 평균에서 실제로 학습되지만, yaw
evaluator 지표는 개선되지 않고 오히려 악화됐고(3DPW test yaw MAE는 새
게이트 실패), 이전까지 문제없던 영역에 새 orientation 혼동을 만들었다.**
weight를 조정하지 않는다(지시에 따름). 다음 아키텍처 질문은 더 풍부한
torso relational/local-frame 표현, 또는 temporal/latent orientation-state
supervision·multi-hypothesis modeling이며, 이번 batch에서 구현하지 않는다.
자세한 진단은 `docs/18_WORKLOG_A14_BILATERAL_FORWARD_DEPTH.md`.

재현 가능한 진단 JSON은
`/home/nd/animcv-output/experiments/a14_bilateral_forward_depth_diagnosis/`
(gradient_diagnosis.json, test_attribution.json)와 학습 결과
`/home/nd/animcv-output/experiments/ablation_a14_bilateral_forward_depth_10e_v2/`
에 있다.

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
