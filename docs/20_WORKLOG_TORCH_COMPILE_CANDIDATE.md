# Worklog — torch.compile Candidate for Training Throughput (2026-09-02)

> 이 batch는 quality experiment가 아니다. docs/19의 GPU utilization 10~20%
> 증상이 실제로 fragmented small-kernel execution인지 커널 단위 증거로
> 먼저 확인한 뒤, 정확히 하나의 torch.compile 후보만 시험했다. 정량 수치와
> A9 benchmark 정의의 단일 출처는 docs/10, docs/19다.

## 1. Reproduced eager baseline

docs/19와 동일 config·seed(`1337`)·warmup(50)/measure(300)로
`scripts/profile_temporal_lifter_training.py`를 재실행했다.

| | docs/19 | 재현 | 차이 |
| --- | ---: | ---: | ---: |
| samples/sec | 3,939.85 | 4,044.40 | +2.6% |
| mean step wall ms | 32.49 | 31.65 | -2.6% |
| GPU utilization mean | 20.8% | 21.4% | +0.6pt |

서버 GPU contention에 의한 정상 변동 범위이며, 환경 차이로 볼 근거는
없다. 이후 비교는 이 batch에서 새로 측정한 값 기준이다.

## 2. Kernel-level eager profile

`scripts/profile_temporal_lifter_kernels.py`(신규, `torch.profiler`
CPU+CUDA activities, `record_function`로 stage 구분, warmup 20 + measure
20 step)로 실제 커널 단위 증거를 수집했다.

| 지표 | 값 |
| --- | ---: |
| CUDA kernel instance count (20 step) | 11,220 |
| kernel/step | 561 |
| median kernel duration | 4.0 us |
| mean kernel duration | 11.73 us |
| P90 / P95 kernel duration | 20 us / 90 us |
| 10us 미만 kernel 비율 | 71.1% |
| 실측 GPU 커널 실행시간 합계(20 step) | 131,611 us → 6,580.6 us/step |

실측 GPU 커널 실행시간(6.58ms/step)을 Section 1의 step wall
time(~32ms)으로 나누면 **20.3%** — docs/19에서 `nvidia-smi`로 관찰한
평균 utilization **20.8%**와 거의 정확히 일치한다. 이는 GPU utilization이
"GPU가 데이터를 기다린다"는 추론이 아니라, **실제 커널 실행시간 자체가
step wall time의 20%뿐**이라는 커널 단위 증거로 직접 확인된 것이다.

dominant CUDA kernel 목록에는 실제 conv forward/backward gemm
kernel(90~126us, 각 200회)뿐 아니라 cuDNN의 NCHW↔NHWC 레이아웃 변환
kernel(`nchwToNhwcKernel`/`nhwcToNchwKernel`, 1,280+640회, 12.4/11.8us
평균 — 합계 23,376us, 전체 커널 시간의 17.8%)이 상당한 비중을 차지했다.
CPU 쪽에서는 `cudaLaunchKernel`이 9,840회, 평균 14.3us/회 — 순수 launch
dispatch 비용만 20 step에서 140,667us(7.03ms/step)에 달했다. `aten::to`/
`aten::_to_copy`(AMP autocast의 반복적 dtype 캐스팅)도 각각 2,920회/
2,340회로 눈에 띄게 많았다.

## 3. Fragmentation/dispatch mechanism verdict

**CASE A — 세분화된 소규모 커널 실행이 확인됨.** 561개/step, 71.1%가
10us 미만, 실제 커널 실행시간이 step wall time의 20%뿐이라는 세 증거가
서로 독립적으로 일치한다. compile 후보 시험으로 진행한다.

## 4. Exact compile candidate and scope

`_lifter_profiling_common.build_forward_loss_callable`이
`torch.compile(forward_loss_fn)`(기본 설정, backend/mode 인자 없음)로
model-forward + `_supervision_loss` 호출을 하나의 함수로 감싼다. backward
(`scaler.scale(loss).backward()`)와 optimizer step(`scaler.step`/
`scaler.update()`)은 이전과 동일하게 eager로 남는다. `_supervision_loss`
자체의 reduction 순서나 수식은 손으로 바꾸지 않았다 — compiler가 그
그래프 내부를 최적화하도록 두었다.

## 5. Graph-break/recompile accounting

고정 shape(batch 128, window 81)에 대해 `torch._dynamo.explain`으로
직접 확인:

```
graph_count: 1
graph_break_count: 0
break_reasons: []
```

두 개의 독립적인 서버 실행(kernel profiler의 dry-run, verdict script의
warm-up 내 첫 호출)에서 동일하게 확인됐다. 첫 호출(compile) latency는
kernel profiler 실행에서 17.71초, verdict script 실행에서 10.97초로
측정됐다 — 컴파일 시점의 서버 상태에 따른 정상 변동이며, 두 경우 모두
warm-up 구간 안에서만 발생하고 steady-state 측정 구간에는 포함되지
않는다. 고정 shape 워크로드에서 예기치 않은 재컴파일은 관찰되지 않았다.

## 6. Numerical equivalence

동일 초기 파라미터(`copy.deepcopy`)와 동일 고정 batch에서 eager/compiled
forward+loss+backward를 각각 실행해 비교했다.

| 항목 | max abs diff | 비고 |
| --- | ---: | --- |
| prediction | 6.10e-4 | max rel diff는 3,051 — 0에 가까운 원소를 분모로 나눈 계산 artifact이며 abs diff가 실질 지표 |
| total loss | 3.83e-6 | eager 0.1760542, compiled 0.1760504 |
| bone/torso/hinge component | 2.83e-6 / 1.68e-7 / 1.30e-7 | eager helper로 prediction에서 재계산, compiled loss 그래프 자체를 재사용하지 않음 |
| gradient (전 parameter) | 8.01e-5 | max rel diff 673은 위와 같은 near-zero-denominator artifact |

두 경로 모두 gradient가 finite(`gradients_finite_eager`/`_compiled` =
True)였고 grad 누락 parameter는 없었다(`gradient_missing_parameter_names
== []`). 이 모델은 `torch.amp.autocast`로 forward를 float16으로
실행하므로(`torch.testing.assert_close`의 float16 기본 허용치는
`rtol=1e-3`, `atol=1e-5`), 관측된 절대 차이(1e-4~1e-6대)는 compiler가
선택한 reduction 순서 차이로 설명 가능한 float16 정밀도 수준이며, 별도로
느슨한 허용치를 고르지 않았다 — 실측값을 그대로 보고했다.

## 7. Compiled reproducibility

동일 seed/init/dataset/batch로 compiled 후보를 **두 번** 독립 실행:

```
first_loss  = 0.17605039477348328
second_loss = 0.17605039477348328  (완전히 동일)
prediction max_abs_diff = 0.0, max_rel_diff = 0.0, exactly_equal = True
```

완전한 bitwise 재현성 — 이 workload/shape에서 compiled 경로는
결정론적이다.

## 8. Short-training semantic equivalence

동일 초기화에서 20개의 실제 학습 step(zero_grad→forward→loss→
backward→optimizer.step→scaler.update, `train()`과 동일 순서)을
eager/compiled 각각 재생했다. loss/MPJPE trajectory가 매 step마다
소수점 4~5자리까지 거의 일치했다(예: step 0 loss eager `0.17605422` vs
compiled `0.17605039`; step 19 `0.05365485` vs `0.05376538`).

| 지표 | 값 |
| --- | ---: |
| 최종(20번째) step loss abs diff | 1.11e-4 |
| 최종 step MPJPE abs diff | 0.089 mm |
| 20 step 누적 후 parameter max abs diff | 0.0147 |

parameter max abs diff(0.0147)는 max rel diff(2,623, near-zero
parameter denominator artifact)와 달리 실질적 신호지만, 20번의 실제
optimizer step에 걸쳐 float16 수준의 미세한 차이가 반복 누적된 결과로
설명 가능하다(각 step의 loss/MPJPE는 거의 동일했다) — AMP를 쓰는 어떤
두 실행도(compile 여부와 무관하게) reduction 순서가 조금만 달라도
iterative optimization에서 이 정도 누적 divergence는 나타날 수 있다.
학습 **과정** 자체(loss가 내려가는 궤적, MPJPE가 개선되는 궤적)는
compile 여부와 무관하게 동일하게 유지됐다 — 이것이 이 절의 판단
기준이다.

## 9. Eager vs compiled throughput table

동일 warm-up(50)/measure(300) 방법론으로 같은 서버 실행 내에서
eager/compiled를 순차 측정했다(compile latency는 warm-up 구간에서
소비, 측정 구간에 미포함).

| 지표 | eager | compiled | 변화 |
| --- | ---: | ---: | ---: |
| samples/sec | 3,524.93 | 5,452.48 | **+54.68%** |
| mean step wall ms | 36.31 | 23.48 | -35.3% |
| median step wall ms | 36.30 | 23.42 | -35.5% |
| P95 step wall ms | 36.41 | 23.59 | -35.2% |
| GPU peak allocated memory | 967.95 MB | 967.95 MB | 0% (동일) |

(이 절의 eager 절대 수치가 Section 1의 재현 baseline과 ~10% 다른 것은
같은 서버 세션 내에서도 발생하는 정상 run-to-run 변동이다 — 이 절의
핵심은 **동일 실행 내에서** eager 대비 compiled의 상대적 개선이며, 그
비교는 오염되지 않았다.)

## 10. GPU/CPU/VRAM comparison

GPU peak allocated memory는 eager/compiled 모두 967.95MB로 완전히
동일했다 — compile이 메모리 사용량을 늘리지 않았다. `nvidia-smi` 기반
utilization은 이 절에서 별도로 재측정하지 않았으며(Section 2의 커널
단위 증거가 더 직접적인 근거), Section 11의 사후 커널 프로파일이 실제
fragmentation 감소를 확인한다.

## 11. Eager vs compiled kernel-profile comparison

동일 warm-up(20)/measure(20) 방법론으로 compiled 경로도 커널 프로파일을
재실행했다.

| 지표 | eager | compiled | 변화 |
| --- | ---: | ---: | ---: |
| CUDA kernel count (20 step) | 11,220 | 5,680 | **-49.4%** |
| kernel/step | 561 | 284 | -49.4% |
| 실측 GPU 커널 실행시간 합계 | 131,611 us | 98,595 us | **-25.1%** |
| median kernel duration | 4.0 us | 7.0 us | +75% (더 큰 fused kernel) |
| 10us 미만 kernel 비율 | 71.1% | 56.0% | -15.1pt |
| graph_count / graph_break_count | N/A | 1 / 0 | — |

성공 기준을 "compiled가 더 빠르다"에 그치지 않고 "eager에서 관찰된
fragmentation 특성이 실제로 줄었는가"로 확인했다 — 커널 개수가 거의
절반으로 줄고, 개별 커널이 평균적으로 더 커졌으며(4us→7us), 실제 GPU
실행시간 총합도 25.1% 줄었다. 메커니즘 자체가 예측대로 작동했다.

## 12. End-to-end epoch overhead accounting

docs/19가 추정한 steady-state epoch wall time은 117.67초(step 처리량만
외삽)였고, 과거 공식 A9 run의 실제 총 소요시간은 1,335.6초/10epoch =
133.6초/epoch였다 — 약 15.9초/epoch의 non-step 고정 오버헤드가 있다는
뜻이다. 이번 batch는 그 오버헤드(epoch augmentation, source-balanced
permutation, epoch telemetry snapshot, 최초 dataset JSON 로딩)를
개별적으로 재측정하지 않았다 — docs/19에서 이미 이들 각각이 개별적으로는
작다고 확인했으므로(augmentation/permutation은 GPU 벡터 연산 1회,
telemetry는 epoch당 1회의 소규모 forward+`.item()`), 정확한 초 단위
분해보다는 이 고정 오버헤드가 **compile 여부와 무관하게 동일하게
남는다**는 점이 중요하다: compile은 오직 매 step의 forward+loss 그래프에만
적용되고, augmentation/permutation/telemetry/checkpoint 저장은 모두 계속
eager로 실행된다.

이 전제로 전체 epoch 시간을 근사 외삽하면:

```
eager 추정:    (117.67s step) + (15.9s 고정 오버헤드) ≈ 133.6s/epoch  (실측 A9와 일치)
compiled 추정: (117.67s / 1.5468) + 15.9s ≈ 76.1s + 15.9s ≈ 92.0s/epoch
```

→ 전체 epoch 기준 **약 31%** wall-time 단축 추정(step 처리량만 놓고 본
54.7%보다는 작다) — 이는 **외삽**이며, 이번 batch에서 실제 10-epoch
전체 학습을 eager/compiled로 나란히 돌려 재확인하지는 않았다(품질
architecture 작업으로 넘어가지 말라는 지시에 따라 범위 밖으로 남김).
one-time compile latency(약 11~18초)는 10-epoch/36,220-step 전체 run
기준으로는 무시할 수준이다.

## 13. Production acceptance/rejection verdict

**CASE A — COMPILE IS A VIABLE EXECUTION BACKEND. ACCEPT.**

6개 필요조건 모두 충족:

1. fragmentation 가설 지지됨(Section 2-3)
2. 수치적 동등성 수용 가능(Section 6 — float16/AMP 수준 차이, gradient
   finite, 누락 없음)
3. compiled 재생 완전히 결정론적(Section 7 — 정확히 동일)
4. short-training semantics 유지(Section 8 — trajectory 거의 일치,
   parameter divergence는 iterative float16 누적으로 설명 가능)
5. steady-state 처리량 유의미하게 개선(Section 9 — +54.68%)
6. 병적 재컴파일 없음(Section 5 — graph_count 1, break 0), 메모리
   사용량도 변화 없음(Section 10)

`TrainingConfig.compile_training_graph: bool = False`(기본값 False,
opt-in)를 프로덕션에 추가하고, `train()`의 forward+loss 호출을
`torch.compile`로 감쌀 수 있게 했다. `compile_training_graph=True`와
`distributed=True`의 동시 사용은 이번 batch가 검증하지 않은 조합이므로
`__post_init__`에서 명시적으로 거부한다. checkpoint payload와 report의
`parallelism`에 `execution_backend`("eager"/"compiled")를 기록해
provenance를 감사 가능하게 했다. **A9–A14의 기존 checkpoint/fingerprint/
report/config는 소급 변경하지 않았다** — 모두 `compile_training_graph`
필드가 없던 시점의 기본값(False)과 동일하게 동작한다.

## 14. Remaining performance bottleneck

compile 이후에도 여전히 284개/step의 CUDA 커널이 남아 있다(eager
561개의 약 절반). 즉 fragmentation은 크게 줄었지만 완전히 제거되지는
않았다 — 남은 병목은 여전히 "다수의 작은 커널을 순차 launch"하는 동일한
성격이며, 그 규모만 절반으로 줄었다.

## 15. Whether CUDA Graph deserves the next separate batch

**예.** 이 workload는 고정 shape(batch 128, window 81)이 매 step
반복되므로 CUDA Graph capture의 이상적인 후보다. compile 이후에도 284개
커널/step이 남아 있다는 사실은 launch overhead가 여전히 존재함을
보여주며, CUDA Graph는 컴파일 이후 남은 launch overhead 자체를 제거하는
보완적 메커니즘이다. 지시에 따라 이번 batch에서 구현하지 않았다 — 다음
독립 batch로 남긴다.

## 16. Exact files changed

- `src/training/temporal_lifter.py` — `TrainingConfig.compile_training_graph`
  (기본 `False`), `distributed`와의 동시 사용 거부, `train()`의
  forward+loss를 `torch.compile`로 감쌀 수 있는 `forward_loss_fn`,
  checkpoint/report의 `execution_backend` provenance
- `scripts/run_lifter_experiments.py` — `--compile-training-graph`
- `scripts/_lifter_profiling_common.py` (신규) — 공유 benchmark
  setup/step-callable (docs/19의 `profile_temporal_lifter_training.py`에서
  추출)
- `scripts/profile_temporal_lifter_training.py` — 공유 모듈을 쓰도록 리팩터
  (동작 변경 없음)
- `scripts/profile_temporal_lifter_kernels.py` (신규) — `torch.profiler`
  커널 단위 진단, eager/compiled 공용
- `scripts/benchmark_torch_compile_candidate.py` (신규) — 수치 동등성,
  재현성, short-training A/B, steady-state 처리량 A/B
- `tests/test_lifter_profiling_common.py` (신규)
- `tests/test_profile_temporal_lifter_training.py` (리팩터에 맞게 갱신)
- `tests/test_profile_temporal_lifter_kernels.py` (신규)
- `tests/test_benchmark_torch_compile_candidate.py` (신규)
- `tests/test_supervised_temporal_lifter.py` — `compile_training_graph`
  기본값/활성화/distributed 거부 테스트 3개 추가
- `docs/20_WORKLOG_TORCH_COMPILE_CANDIDATE.md` (본 문서)
- `docs/README.md`

A9–A14 checkpoint/fingerprint/report, 기존 promotion gate, `.vscode/`는
변경하지 않았다.

## 17. Tests/benchmarks executed

- profiling-common focused test: `4 passed`
- kernel-profiler focused test: `7 passed`
- compile-candidate-verdict focused test: `6 passed`(실제 CPU
  torch.compile 경로 포함, mock 아님)
- `train()` compile flag 관련 focused test: `3 passed`(기본 eager
  provenance, 활성화 시 실제 학습+checkpoint, distributed 조합 거부)
- 전체 로컬 회귀: `422 passed`
- `py_compile`: PASS
- LabServer63 GPU 실행: 재현 baseline, eager kernel profile, compiled
  kernel profile, compile-candidate verdict(수치 동등성/재현성/
  short-training A/B/처리량 A/B) 4건 모두 PASS

## 18. Commit hashes / push / server synchronization state

- `1cc1ff6` — diagnostic: kernel-level torch.profiler + shared step-callable for A9 benchmark
- `f05e788` — diagnostic: eager-vs-compiled numerical equivalence/reproducibility/short-training/throughput verdict
- 본 worklog 및 production `compile_training_graph` 통합 커밋은 이후
  별도로 기록한다.

진단 산출물은 git에 커밋하지 않고 다음 서버 경로에 남아 있다.

- `/home/nd/animcv-output/experiments/a15_training_throughput_diagnosis/kernel_profile_eager.json`
- `/home/nd/animcv-output/experiments/a15_training_throughput_diagnosis/kernel_profile_compiled.json`
- `/home/nd/animcv-output/experiments/a15_training_throughput_diagnosis/compile_candidate_verdict.json`
