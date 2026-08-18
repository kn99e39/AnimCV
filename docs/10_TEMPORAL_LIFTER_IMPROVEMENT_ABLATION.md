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
