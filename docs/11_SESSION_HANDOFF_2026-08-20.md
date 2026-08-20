# Session Handoff — Temporal Lifter 품질 개선

> 작성일: 2026-08-20 (KST)  
> 목적: 다음 Agent가 코드, LabServer63, 데이터 계약과 품질 실험 상태를 재조사하지 않고 이어서 작업한다.

## 1. 현재 결론

AnimCV의 canonical 17-joint temporal lifter는 CUDA 학습, 독립 holdout 평가, retarget/Blender
export까지 연결돼 있다. 다만 현재 최고 비교 기준선은 3DPW의 **root-yaw P95**와 **hinge flip**
gate를 통과하지 못했으므로 제품/리깅 단계로 승격하면 안 된다.

현재 작업의 목표는 단순히 epoch를 늘리는 것이 아니라, 동일 데이터·seed·입력 계약에서 yaw/hinge
tail 품질을 개선하고 수치 및 사람이 보는 리그 애니메이션 영상으로 검증하는 것이다.

## 2. 저장소와 작업 원칙

| 항목 | 값 |
| --- | --- |
| 로컬 저장소 | `/Users/nadan/Projects/AnimCV` |
| 브랜치 | `On_Work` |
| 최신 커밋 | `a4a24f2 feat: fingerprint lifter experiment inputs` |
| 서버 저장소 | `LabServer63:/home/nd/AnimCV` |
| 서버 최신 상태 | `a4a24f2`까지 fast-forward 완료 |
| 로컬 작업 트리 | 사용자 소유의 `?? .vscode/`만 존재. 수정·추적·삭제하지 말 것. |

사용자는 이후 **commit과 push를 Agent가 직접 수행**하도록 승인했다. 단, data/output 대형 파일은
git에 넣지 않는다. 서버의 다른 사용자의 GPU 작업 또는 process는 종료/변경하지 않는다.

GPU 상태는 불필요하게 자주 polling하지 않는다. 사용자가 별도로 요청하지 않는 한 **10분보다 자주
조회하지 않는다.**

## 3. 서버 환경과 안전한 확인

| 항목 | 값 |
| --- | --- |
| 접속 | `ssh LabServer63` (credential을 문서화하지 않는다) |
| GPU | RTX 3080 Ti, 12 GB |
| 컨테이너 | `animcv-train:cuda118` (PyTorch 2.1.2+cu118) |
| 데이터 root | `/home/nd/animcv-data` (`/data` read-only mount) |
| 출력 root | `/home/nd/animcv-output` (`/output` mount) |

재개 첫 명령은 읽기 전용으로 한다.

```bash
ssh LabServer63 'nvidia-smi; screen -ls; cd /home/nd/AnimCV && git status --short && git log -1 --oneline'
```

2026-08-20 13:02 KST 확인 당시 GPU는 유휴였고 A9 학습은 완료됐다.

## 4. 현재 코드 변경과 계약

### Reproducibility / input provenance

- `src/training/temporal_lifter.py`: 모델 생성 전에 `torch.manual_seed(config.seed)`를 호출하고
  checkpoint/report에 `training_seed`를 기록한다 (`0f6c1ca`).
- `scripts/run_lifter_experiments.py`: `experiment_matrix.json` v2에 `dataset_fingerprints`를
  기록한다 (`a4a24f2`). 각 원본 train/holdout 및 materialized validation JSON에 대해 path,
  SHA-256, byte size, frame count, sequence count를 저장한다.
- 새 실험의 정량/정성 비교는 fingerprint가 일치할 때만 동일 조건으로 취급한다.

### Throughput

- structural loss의 bone/torso/hinge/yaw/anti-flip 계산은 GPU batch 연산으로 벡터화됐다
  (`81246ed`, compatibility fix `bc2e64d`).
- CPU container에서 이전 per-chain 구현과 수치 동치가 확인됐다.
- `peak_gpu_memory_mb`는 PyTorch allocated 값이며, `nvidia-smi` reserved VRAM과 직접 같지
  않을 수 있다.

### 정성 평가 영상

- `scripts/render_blender_animation_video.py` (`08f2b2a`)가 animated `.blend`를 MP4로
  headless Blender 렌더한다. 실제 메시와 녹색 bone/주황 joint 프록시를 함께 그려 메시
  문제와 관절 추출 문제를 분리한다.
- 기본 사용 예:

```bash
blender --background --python scripts/render_blender_animation_video.py -- \
  --blend artifacts/candidate.blend \
  --out artifacts/review/candidate_three_quarter.mp4 \
  --camera three_quarter
```

- 모든 후보에 동일 holdout clip, rig/mapping, camera, 해상도, frame range를 쓴다. 발
  미끄러짐, hinge 연속성, root yaw 안정성, 자연스러움을 각각 1–5점으로 기록한다. 어느
  항목이든 2점 이하면 승격하지 않는다.

## 5. 데이터 계약

| 입력 | 컨테이너 경로 | A9 fingerprint |
| --- | --- | --- |
| MPI train | `/output/data/animcv/train_combined.json` | 106,512 f / 12 seq / `f0b760…90952` |
| 3DPW train | `/data/3dpw/prepared/train.json` | 22,646 f / 34 seq / `cc81be…e9742` |
| AMASS train | `/data/amass/prepared_aug_v1/train.json` | 334,402 f / 3,000 seq / `56a88d…36449` |
| 3DPW holdout | `/data/3dpw/prepared/holdout.json` | 35,310 f / 37 seq / `fff2f3…06971` |
| AMASS holdout | `/data/amass/prepared/holdout.json` | 10,792 f / 100 seq / `cab22d…35cc` |
| validation materialization | A9 output `datasets/validation.json` | 34,586 f / 232 seq / `80bed9…3434a` |

중요: 과거 A5/A7은 AMASS augmented holdout(31,910 frames)을 썼고 A8/A9는 original
AMASS holdout(10,792 frames)을 썼다. 따라서 과거와 AMASS 절대 수치를 직접 비교하지 않는다.
A9 fingerprint가 이후 후보의 유일한 비교 기준이다.

## 6. 확정된 실험 결과

### A6/A7 (과거, 현재 data contract와 직접 비교 금지)

- A6: mean yaw supervision 0.15은 3DPW PA 83.48 mm / yaw P95 35.70° / flip 5.11%로
  악화되어 거절됐다.
- A7: yaw-tail/anti-flip 각각 0.05는 PA 84.28 mm / yaw P95 35.38° / flip 2.04%였다.
  tail gate 개선 없이 PA가 악화되어 거절됐다.

### A8 (fixed seed, fingerprint 이전)

- A8 seeded batch 128: 4,418.6 samples/s, 1,049.1 sec.
- 3DPW PA 75.31 mm, yaw MAE 14.90°, yaw P95 34.77°, flip 2.36% → P95/flip 실패.
- AMASS PA 69.19 mm, yaw MAE 8.77°, yaw P95 22.37°, flip 2.32% → flip 실패.

### A9 — 현재 공식 기준선

출력: `/home/nd/animcv-output/experiments/ablation_a9_fingerprinted_baseline_10e/`

- 설정: direct mix, 10 epochs, window 81, channels 256, batch 128, seed 1337, CUDA AMP,
  `dilated_tcn_v1`, source-balanced, pelvis-torso v1, online 2D augmentation, bone/torso/hinge
  loss `0.25/0.15/0.15`, yaw 및 anti-flip loss 0.
- 결과는 A8 seeded와 **정확히 동일**하여 seed contract가 검증됐다.
- 3DPW: PA 75.307 mm (통과), yaw MAE 14.903° (통과), yaw P95 34.772° (실패),
  flip 2.361% (실패).
- AMASS: PA 69.189 mm, yaw MAE 8.774°, yaw P95 22.370° (통과), flip 2.324% (실패).
- 처리량: 3,470.9 samples/s, 1,335.6 sec. A8보다 느리지만 품질 값은 동일하므로 환경
  변동으로만 기록하며 architecture regression으로 해석하지 않는다.

## 7. 다음 Agent의 실행 순서

1. 위 3절의 read-only 상태 확인 후 GPU가 유휴일 때만 run을 시작한다.
2. A9의 `experiment_matrix.json`에서 dataset fingerprints를 복사해 후보 결과가 정확히
   일치하는지 확인한다.
3. 새 품질 후보는 한 번에 하나의 원인만 바꾼 10-epoch direct-mix run으로 실행한다.
   기존 A7의 0.05 tail/anti-flip은 **다른 data/reproducibility contract에서 거절됐으므로
   그대로 재실행하지 말고**, A9 baseline에 대해 작고 분리된 가설을 먼저 설계한다.
4. gate 판단은 3DPW를 우선한다. PA ≤80 mm, yaw MAE ≤15°, yaw P95 ≤30°, flip = 0%를
   동시에 만족해야 승격 후보가 된다. AMASS는 같은 fingerprint에서 보조 확인한다.
5. 수치 상위 후보만 fixed holdout clip을 retarget → `apply_motion.py` → Blender review MP4
   경로로 보내 정성 점수를 기록한다. 아직 품질 gate를 통과한 checkpoint는 없다.
6. 결과, 정확한 command, fingerprint, 판정을 `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`
   에 추가하고 test → commit → push한다.

## 8. 주요 파일

- 실험 runner: `scripts/run_lifter_experiments.py`
- 학습/evaluator: `src/training/temporal_lifter.py`
- ablation 기록 및 명령 예: `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`
- 서버 runbook: `docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md`
- Blender 적용: `scripts/apply_motion.py`
- review MP4 렌더: `scripts/render_blender_animation_video.py`

## 9. 검증 현황

최근 로컬 변경별 검증:

- MP4 renderer: `PYTHONPATH=src pytest -q tests/test_render_blender_animation_video.py tests/test_apply_motion_script.py tests/test_blender_executor.py` → 17 passed.
- fingerprint: `PYTHONPATH=src pytest -q tests/test_run_lifter_experiments.py tests/test_supervised_temporal_lifter.py tests/test_lifter_evaluation_metrics.py` → 3 passed, 1 skipped (로컬 PyTorch 없음).
- 각 변경 시 `python -m compileall` 및 `git diff --check` 통과.

