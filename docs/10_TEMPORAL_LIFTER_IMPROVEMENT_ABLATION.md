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

각 run의 3DPW PA-MPJPE를 우선 정렬하고, yaw MAE/P95와 hinge flip이 동시에 악화되지
않는 상위 두 후보만 30 epoch로 재실행한다. 결과가 3DPW holdout을 통과하기 전에는
retarget/FBX 단계로 넘기지 않는다.

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
