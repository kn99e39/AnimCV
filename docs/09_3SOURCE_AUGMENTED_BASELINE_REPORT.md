# 3-Source Augmented Baseline Report

## 목적

AMASS의 단일 정면 가상 카메라와 무노이즈 2D 입력으로 인한 domain gap을 줄이고,
MPI-INF-3DHP·3DPW·AMASS를 함께 쓰는 temporal lifter의 실제 30 epoch 기준선을 만든다.
이 문서는 품질 통과 선언이 아니라, 재현 가능한 실패/개선 기준을 남긴다.

## 구현한 보강

- AMASS source마다 세 개의 명시적 virtual camera를 생성했다.
  - yaw 0°, pitch 0°, distance 4.5m, focal 1500
  - yaw -45°, pitch 10°, distance 5.0m, focal 1300
  - yaw +45°, pitch -8°, distance 4.0m, focal 1750
- 학습 epoch마다 normalized 2D input에 coordinate jitter 0.015, observed-joint
  dropout 0.05, confidence jitter 0.08을 적용했다.
- augmentation은 3D target이나 validation/holdout 입력을 바꾸지 않는다.
- AMASS pretrain checkpoint를 compatible temporal lifter에 초기화해 fine-tune할 수 있게 했다.
- `run_lifter_experiments.py`는 MPI-only, MPI+3DPW, direct mix, AMASS pretrain,
  pretrain→fine-tune을 같은 validation·holdout에서 비교한다.

## Augmented Corpus Integrity

| Split | Source motions | Camera views | Sequences | Frames | 검사 |
| --- | ---: | ---: | ---: | ---: | --- |
| AMASS train | 1,000 | 3 | 3,000 | 334,402 | ID 충돌 0, NaN/Inf 0 |
| AMASS validation | 72 | 3 | 216 | 24,380 | ID 충돌 0, NaN/Inf 0 |
| AMASS holdout | 100 | 3 | 300 | 31,910 | ID 충돌 0, NaN/Inf 0 |
| direct-mix train | MPI 12 + 3DPW 34 + AMASS 3,000 | — | 3,046 | 463,560 | ID 충돌 0 |

3DPW official test는 train과 validation에서 제외했다. AMASS internal holdout도 train과
validation에서 제외했다.

## 1-Epoch Candidate Smoke

모든 candidate에 같은 online input augmentation을 적용한 실행 경로 검증 결과다. 한 epoch
수치이므로 후보 선택의 근거가 아니라 pipeline smoke 기준이다.

| Candidate | Validation MPJPE mm | 3DPW test MPJPE mm | AMASS holdout MPJPE mm |
| --- | ---: | ---: | ---: |
| MPI only | 309.91 | 272.40 | 404.63 |
| MPI + 3DPW | 231.95 | 162.52 | 343.46 |
| Direct mix | 129.79 | 163.21 | 123.45 |
| AMASS pretrain | 261.10 | 340.66 | 121.98 |
| AMASS pretrain → MPI+3DPW fine-tune | 248.43 | 161.19 | 387.13 |

Direct mix만 양쪽 holdout에 비교적 균형적이어서 다음 30 epoch baseline 대상으로 골랐다.

## 30-Epoch Augmented Direct-Mix Result

학습 설정은 window 81, channels 256, batch 128, learning rate 0.001, CUDA AMP다.

| 측정 | 결과 |
| --- | ---: |
| train MPJPE | 63.53 mm |
| validation MPJPE / P95 | 90.78 / 261.02 mm |
| 3DPW test MPJPE / P95 | 137.54 / 358.41 mm |
| AMASS holdout MPJPE / P95 | 81.59 / 246.69 mm |
| 학습 시간 | 929.66 s |
| 처리량 | 14,959 samples/s |
| peak allocated VRAM | 1,125.86 MiB |

RTX 3080 Ti 12GB에서 자원 문제는 없었다. 하지만 3DPW test 137.54mm는 실제 이동 카메라
도메인에서 아직 부족한 결과다. 이 checkpoint를 game-production 품질로 선언하거나 FBX
기본 경로로 승격하지 않는다.

## 다음 품질 단계

공통 evaluator는 이제 dataset-neutral하게 PA-MPJPE, bilateral shoulder/hip root-yaw
MAE/P95, elbow/knee bend-direction MAE/P95·flip rate, source/view/action slice를 보고한다.
게이트는 PA-MPJPE ≤80mm, yaw MAE ≤15°, yaw P95 ≤30°, unambiguous hinge flip rate 0%다. 기존 30-epoch
report는 v1 evaluator로 산출됐으므로, 이 새 항목의 값은 아직 없다. checkpoint를 변경하지
않고 holdout 평가만 재실행해 v2 report를 남겨야 한다.

1. GT 2D가 아닌 실제 MMPose/RTMPose 입력으로 같은 3DPW/MPI protocol을 재실행한다.
2. augmentation 강도를 calibration하고 연속 temporal occlusion을 추가한다.
3. direct mix와 pretrain→fine-tune을 충분한 epoch·동일 seed 조건으로 재비교한다.
4. holdout을 통과한 후보만 constraint retarget 및 다중 FBX visual regression으로 넘긴다.
