# Worklog — Training Throughput Diagnosis (2026-09-02)

> 이 batch는 quality experiment가 아니다. A9의 이미 확립된 benchmark 설정을
> 그대로 얼려 사용했고, 새 architecture/loss 실험을 만들지 않았다. 관찰된
> 증상은 `nvidia-smi` GPU utilization 10~20%, 메모리 사용량 약 1.4GB였다.
> GPU utilization 수치 자체를 최적화 목표로 삼지 않고, end-to-end 처리량을
> 실제로 제한하는 단계를 측정으로 규명했다.

## 1. Exact baseline hardware/training configuration

- GPU: LabServer63의 NVIDIA RTX 3080 Ti (12,288MiB), driver `nvidia-smi`로 확인.
- Docker image `animcv-train:cuda118`, `torch==2.1.2+cu118`.
- Benchmark config는 A9의 frozen 설정을 그대로 재사용했다(docs/10, docs/18
  Section 1): `window=81`, `channels=256`, `batch_size=128`,
  `learning_rate=1e-3`, `mixed_precision=True`(이미 기존 기본값, 이번에 새로
  켠 것 아님), `architecture=dilated_tcn_v1`, `source_balanced_sampling=True`,
  `input_coordinate_normalization=pelvis_torso_v1`, augmentation
  (`input_jitter_std=0.015` 등 A9와 동일), `bone/torso/hinge_loss_weight=
  0.25/0.15/0.15`, seed `1337`.
- Dataset: A9가 이미 materialize한
  `/output/experiments/ablation_a9_fingerprinted_baseline_10e/datasets/direct_mix_train.json`
  (463,560 window, 3,622 step/epoch @ batch 128)를 그대로 재사용했다 — 새
  dataset을 만들지 않았다.
- 모델 파라미터 수: 2,021,171.

## 2. Baseline throughput

`scripts/profile_temporal_lifter_training.py --warmup-steps 50 --measure-steps
300 --seed 1337`(diagnostic-only, `train()`을 수정하거나 import하지 않음):

| 지표 | 값 |
| --- | ---: |
| samples/sec (측정 창) | **3,939.85** |
| windows/sec | 3,939.85 |
| mean step wall time | 32.49 ms |
| median step wall time | 32.58 ms |
| P95 step wall time | 35.11 ms |
| epoch wall time 추정치 | 117.67 초 |

과거 A9 공식 run의 처리량(`3,470.9 samples/s`, 1,335.6초, docs/10)과 order가
일치한다 — A9 수치는 telemetry snapshot·DDP context·checkpoint 저장 등 전체
run 오버헤드를 포함하므로 완전히 같지는 않지만 cross-check로 충분하다.

## 3. Step-stage timing breakdown

같은 seed로 재생한 별도 attribution pass(step마다 CUDA event +
`torch.cuda.synchronize()`를 추가한 진단 전용 pass, 이 pass 자체의 wall
time은 진단용이며 처리량으로 보고하지 않음):

| Stage | mean | step 대비 비중 |
| --- | ---: | ---: |
| batch construction (gather) | 0.66 ms | 2.0% |
| forward | 5.33 ms | 16.4% |
| loss computation (coordinate+bone+torso+hinge) | 7.55 ms | 23.3% |
| backward | 14.54 ms | 44.8% |
| optimizer.step | 2.73 ms | 8.4% |
| scaler.update (AMP inf-check) | 0.14 ms | 0.4% |
| **합계** | **30.96 ms** | **95.3%** |
| (throughput pass의 실측 mean step time과 차이) | 1.53 ms | 4.7% (Python 오버헤드 등, 큰 유휴 구간 아님) |

stage 합계가 실제 throughput-pass의 mean step time(32.49ms)의 95.3%를
설명한다 — 즉 batch 사이에 GPU가 크게 노는 구간은 없다. 그런데도 같은 측정
구간의 `nvidia-smi` 평균 utilization은 20.8%(P95 24%, max 37%)에 불과했다.
이 두 사실을 동시에 만족하는 유일한 설명은: forward/loss/backward가 각각
"오래 걸리는" 이유가 GPU 연산 자체가 크기 때문이 아니라, **작은 커널을
순차적으로 아주 많이 launch**하기 때문이라는 것이다(5개 residual dilated
block × conv1d 2개 + activation, coordinate/bone/torso/hinge 4개의 개별
`smooth_l1_loss` 호출 등). 각 커널의 실제 SM 점유 시간은 매우 짧고,
`nvidia-smi`가 측정하는 "커널이 실행 중이던 시간"은 낮지만, 커널을 순차
launch하는 데 걸리는 host-side dispatch latency의 총합이 step wall time을
지배한다.

## 4. DataLoader/input-pipeline findings

`src/training/temporal_lifter.py`의 `train()`은 **DataLoader를 전혀 사용하지
않는다.** `_arrays()`가 학습 시작 전 딱 한 번 NumPy 배열을 만들고, 즉시
`torch.as_tensor(..., device=device)`로 GPU 상주 텐서(`x`, `y`,
`valid_tensor`, `offset_tensor`, `source_tensor`)를 만든다. epoch마다
`_augment_inputs()`가 GPU 텐서 연산으로 한 번에 augmentation을 적용하고,
매 step의 `windows = epoch_inputs[offset_tensor[batch]]`는 GPU 상의 advanced
indexing(gather) 하나다. 따라서:

- `num_workers`/`persistent_workers`/`prefetch_factor`/`pin_memory` 같은
  DataLoader 파라미터가 **존재하지 않는다** — 이 파이프라인에는 최적화할
  worker 설정 자체가 없다.
- step마다 host→device 전송이 **없다**(1회 초기 업로드 이후 전부 GPU 상주).
- source-balanced mixing은 `_source_balanced_permutation`이 epoch당 한 번
  (매 step이 아님) 3개 source에 대해서만 반복하는 매우 작은 Python 루프다 —
  step 경로에 없다.
- disk에서 sample을 다시 만드는 일이 없다(1회 JSON parse, 8.02초, 3,622
  step/epoch × 10 epoch에 분산돼 무시할 수준).
- `windows` 텐서는 gather의 결과라 항상 contiguous다.

**결론: Case A(DataLoader/CPU starvation)와 Case B(H2D bottleneck)는 이
파이프라인 구조상 해당하지 않는다** — 애초에 그 메커니즘 자체가 없다.

## 5. Python/synchronization findings

production hot path(`train()`의 batch 루프)를 검사한 결과 매 step마다
`.item()`/`.cpu()`/`.numpy()`/명시적 `torch.cuda.synchronize()` 호출이
**없다.** `_epoch_telemetry_snapshot`은 epoch마다 딱 한 번(3,622 step 중
1번) 호출되며 그 안의 15개 남짓한 `.item()` 호출은 step당으로 분산하면
무시할 수준이다. AMP `GradScaler.update()`가 내부적으로 수행하는 유일한
per-step 암묵적 host 동기화(inf/nan 체크)는 진단에서 `scaler_update_ms`로
직접 측정했고 평균 0.14ms(0.4%)로 무의미한 크기다. `optimizer.zero_grad()`는
인자 없이 호출되지만 이 torch 버전(2.1.2)의 기본값이 이미
`set_to_none=True`라 추가 조치가 필요 없다.

**결론: Case C(강한 의미의 명시적 동기화 정체)는 해당하지 않는다.** 다만
"Python이 아주 많은 작은 커널을 순차 발행하는 데 걸리는 dispatch latency"는
넓은 의미로는 Python/host-side 오버헤드의 일종이며, Section 3의 결과와
결합해 Case D의 구체적 메커니즘으로 재분류했다(Section 6 참고).

## 6. GPU compute-size findings

- batch 128, channels 256, window 81, 5-block dilated TCN(2,021,171
  parameter)은 RTX 3080 Ti 기준으로 작은 모델이다.
- forward(5.33ms)보다 loss computation(7.55ms)이 더 오래 걸린다 —
  coordinate/bone/torso/hinge 4개 항을 각각 별도 `smooth_l1_loss` 호출 +
  여러 개의 advanced-indexing/reduction 커널로 순차 계산하기 때문에, 전체
  텐서 크기는 작지만 커널 개수는 많다.
- stage 합계(30.96ms)가 실측 step time(32.49ms)의 95.3%를 설명 — GPU가
  batch 사이에 노는 큰 유휴 구간은 없다(Section 3).
- 그런데도 `nvidia-smi` 측 평균 utilization은 20.8%다.

이 조합은 "GPU가 데이터를 기다리는 것"이 아니라 "GPU에 일이 끊임없이
공급되지만, 개별 커널이 GPU를 포화시키기엔 너무 작아 launch/dispatch
overhead가 wall time을 지배하는 것"이라는 **Section 9의 CASE D 정의와
정확히 일치**한다. VRAM 사용량(peak allocated 913.5MB, reserved 1038MB —
사용자가 관찰한 "약 1.4GB"와 일치)만으로 이를 추론하지 않고, stage
timing breakdown과 utilization 수치를 함께 사용해 확인했다.

## 7. Primary bottleneck verdict

**CASE D — GPU workload too small (kernel-launch/dispatch-latency
dominated), Case A/B는 구조적으로 해당 없음, Case C(강한 동기화 정체)도
아님.**

## 8. Exact fixes implemented and causal justification for each

**없음.** Section 4–7의 증거는 다음을 보여준다:

- DataLoader 병렬화, pinned memory, non-blocking H2D, worker prefetch — 이
  파이프라인에 DataLoader 자체가 없으므로 해당 없음(Section 4).
- per-step 동기화 제거 — 이미 없음(Section 5).
- 정적 index/mask precompute — `BONE_INDICES`/`TORSO_INDICES`/
  `HINGE_INDICES`는 이미 모듈 레벨 상수다. `_vector_loss`가 매 step
  `first, second = zip(*pairs)`를 다시 계산하는 것은 사실이지만 16/2개
  원소 tuple의 Python-level zip은 마이크로초 단위이며, 32.49ms의 step
  time에서 측정 가능한 신호로 나타나지 않는다 — "관찰된 병목에 대응하지
  않는 변경은 하지 않는다"는 지시에 따라 손대지 않았다.
- 남은 유일한 실제 병목(작은 커널의 순차 launch overhead)을 이번 batch의
  허용된 fix 목록(Section 10-13)으로 해결할 방법이 없다: 그 문제의 실제
  해법은 커널 fusion(`torch.compile`) 또는 CUDA graph capture인데, 둘 다
  이번 batch에서 명시적으로 금지됐다(Section 13). loss 항 4개를 하나의
  배치 연산으로 수동 재구성하는 것도 가능은 하지만, loss 산출 순서를
  바꾸는 실질적 리팩터이고 부동소수점 reduction 순서가 바뀔 위험이 있어
  "loss design 변경 금지" 지시와 정면으로 충돌한다. 이번 batch에서
  시도하지 않았다.

**따라서 production `train()`/`_supervision_loss`/`run_lifter_experiments.py`
는 전혀 수정하지 않았다.** 유일한 신규 산출물은 diagnostic-only
`scripts/profile_temporal_lifter_training.py`다.

## 9. Baseline vs optimized throughput table

해당 없음 — 이번 batch는 optimized 버전을 생성하지 않았다(Section 8).
baseline 수치만 존재한다(Section 2-3).

## 10. GPU/CPU/VRAM utilization comparison

해당 없음(Section 9와 동일한 이유). baseline 단독 수치:

- GPU utilization: mean 20.8%, P95 24%, max 37%
- GPU memory: allocated 549.6MB, reserved 1038MB, peak allocated 913.5MB
- Host RSS: 10,779.7MB(JSON 463,560-frame 데이터셋을 process 메모리에 올린
  상태 포함)
- Process CPU utilization: wall 대비 189.8%(멀티스레드 텐서 연산 + Python
  루프가 GPU 커널을 순차 launch하는 동안 다중 코어를 사용한다는 뜻이며,
  DataLoader worker가 없으므로 이 수치는 Python/torch 자체의 CPU 스레드
  사용량이다)

## 11. Determinism / semantic-equivalence result

프로덕션 코드를 변경하지 않았으므로 학습 semantics는 그대로다. 진단
스크립트 자체의 신뢰성은 focused test로 검증했다:
`tests/test_profile_temporal_lifter_training.py`에서 `_setup()`이 동일 seed로
반복 호출돼도 `epoch_inputs`, batch 순서, 모델 초기 파라미터가 완전히
동일함을 확인했고(두 pass가 공정한 비교인지의 전제조건), throughput pass와
attribution pass가 동일 seeded 상태에서 첫 step의 loss 값이 일치함을
확인했다(진단용 CUDA event/sync 삽입이 계산값 자체를 바꾸지 않음).

## 12. Short-training/checkpoint equivalence result

해당 없음 — production 코드에 어떤 변경도 없으므로 checkpoint identity
질문 자체가 발생하지 않는다.

## 13. Remaining bottleneck after optimization

변경이 없었으므로 baseline과 동일: 커널 launch/dispatch latency가
지배적이다(Section 3, 6).

## 14. Whether further GPU-compute optimization is justified

예, 그러나 이번 batch의 허용 범위 밖의 후속 가설로만 남긴다. 지시에 따라
자동 적용하지 않았다.

- **커널 fusion (`torch.compile`)**: loss stage가 4개의 개별
  `smooth_l1_loss` + 여러 reduction 커널로 나뉘어 있어 fusion 이득이 클
  가능성이 높다. 수치적으로 완전히 동등한지 검증이 필요하며 이번 batch
  범위 밖이다.
- **CUDA graph capture**: 매 step의 텐서 shape(batch 128, window 81)가
  고정이라 이상적인 후보다. 이 역시 다음 batch 범위다.
- **batch size 확장**: launch overhead를 더 많은 sample에 분산시켜
  throughput을 올릴 수 있지만, gradient noise/optimizer update 빈도가
  바뀌므로 quality experiment로 별도 평가해야 한다(Section 12 지시에 따라
  이번 batch에서 자동 적용하지 않음).
- 위 세 후보 모두 **이번 batch에서 구현하지 않았다.**

## 15. Portability to future commercial datasets

- MPI/3DPW/AMASS 이름에 의존하는가: 아니다. 진단 스크립트와 현재
  파이프라인 모두 `dataset["sequences"]`/`source` 필드의 generic schema만
  사용한다(`_arrays`, `_frame_metadata`와 동일 계약).
- DataLoader 동작이 generic source adapter에 적용되는가: 해당 없음 —
  DataLoader 자체가 없다. 이는 이 파이프라인의 근본 설계이며 미래
  commercial dataset도 동일하게 "전체를 GPU에 올린 뒤 GPU-side
  augmentation/윈도잉"하는 방식이 유지되는 한 그대로 적용된다.
- prefetch가 bounded인가: 해당 없음(같은 이유).
- 메모리 사용이 예측 가능한가: 예 — 전체 dataset을 GPU/host 메모리에 한
  번에 올리는 구조이므로, dataset 크기가 커지면 메모리 사용량이 선형으로
  커진다는 것이 예측 가능한 한계다. 훨씬 큰 commercial dataset에서는 이
  "전부 GPU에 상주"라는 설계 자체가 다음 성능/메모리 병목이 될 수
  있다(이번 batch에서 관찰된 병목은 아니다 — 별도로 기록해 둔다).
- worker RNG semantics가 deterministic한가: 해당 없음(worker 없음);
  전체 학습 자체는 seed 하나로 deterministic함이 이미 기존 테스트로
  보장돼 있다.
- 성능 telemetry/bottleneck 진단 도구가 재사용 가능한가: 예 —
  `scripts/profile_temporal_lifter_training.py`는 `--train-dataset` 하나만
  받아 어떤 combined dataset에도 동일하게 동작한다.

## 16. Exact files changed

- `scripts/profile_temporal_lifter_training.py` (신규, diagnostic-only)
- `tests/test_profile_temporal_lifter_training.py` (신규)
- `docs/19_WORKLOG_TRAINING_THROUGHPUT_DIAGNOSIS.md` (본 문서)
- `docs/README.md`

production training code(`src/training/temporal_lifter.py`,
`scripts/run_lifter_experiments.py`), A9–A14 checkpoint/fingerprint/report,
`.vscode/`는 전혀 변경하지 않았다.

## 17. Tests and benchmarks executed

- 프로파일러 focused test: `5 passed`
- 전체 로컬 회귀(production 코드 무변경이므로 참고용): `405 passed`
- `py_compile`: PASS
- LabServer63 GPU baseline profile 실행: PASS(`baseline_profile.json`)

## 18. Commit hashes / push / server synchronization state

- `d88e5ff` — diagnostic: bounded stage-by-stage training throughput profiler
- 본 worklog 커밋은 이후 별도로 기록한다.

`origin/On_Work`와 `LabServer63:/home/nd/AnimCV`는 `d88e5ff`까지
fast-forward 상태였다(문서 커밋 이전). 서버의 기존 미추적 `.DS_Store`,
`.animcv_sync_stage/`, `docker/`, 로컬의 `.vscode/`는 건드리지 않았다.
진단 산출물은 git에 커밋하지 않고 다음 서버 경로에 남아 있다.

`/home/nd/animcv-output/experiments/a15_training_throughput_diagnosis/baseline_profile.json`
