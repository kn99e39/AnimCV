# Worklog — A12 Magnitude/Direction Attribution (2026-08-26)

> 정량 수치와 gate 정의는 `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`에 기록한다.
> 이 문서는 이번 세션의 진단 과정과 판정만 남긴다. 서버 실행 환경은
> `docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md`를 따른다.

## 1. A12 magnitude-vs-direction attribution

A12의 실제 Cartesian torso residual을 다음 bilateral vector로 분해했다.

`v_pred = p_right_pred - p_left_pred`, `v_gt = p_right_gt - p_left_gt`

측정 항목은 predicted/target span, span residual, unit-direction chord, smooth-L1 scalar
companion, 그리고 squared Cartesian energy의 정확한 magnitude/direction 분해다. A12가 선택한
tail은 그대로 유지했으며 A12 training behavior는 변경하지 않았다.

A12 checkpoint 고정 배치 결과:

- A12 raw tail loss: `0.00152985`
- magnitude scalar: `0.00094354`
- scale-restored direction scalar: `0.00126633`
- magnitude energy: `20.4%`
- direction energy: `79.6%`
- selected predicted span mean: `0.309996`
- selected target span mean: `0.309247`

따라서 A12가 수렴 후 magnitude error에 의해 direction supervision을 잃었다는 가설은
지지되지 않았다. magnitude gradient가 direction component보다 약 1.31배 컸다는 약한
optimization preference는 있었지만, direction signal은 scalar/energy 모두 더 컸고
A11 angular-loss 수준의 폭발은 없었다.

## 2. Source-wise decomposition

A9 checkpoint의 A12 actual tail은 MPI 44.4%, 3DPW 10.6%, AMASS 45.0%였다. 이때 AMASS의
magnitude share는 60.2%, direction share는 40.8%였다. A12 checkpoint에서는 MPI 46.3%,
3DPW 7.5%, AMASS 46.2%였고 AMASS 내부 magnitude/direction share는 54.2%/45.8%였다.

init/A11처럼 AMASS가 86–92%를 차지한 상태에서는 magnitude와 direction이 함께 지배했다.
그러므로 과거에 관찰한 AMASS dominance는 magnitude-only가 아니라 both이며, source
normalization은 이번 세션에서 도입하지 않았다. 방향 후보 자체의 A12-state tail share는
MPI 46.1%, 3DPW 9.7%, AMASS 44.2%였다.

## 3. Proposed representation

진단-only counterfactual은 다음 하나만 검토했다.

```text
u_pred = normalize(v_pred)
u_gt   = normalize(v_gt)
scale  = stop_gradient(||v_gt||)
residual = scale * (u_pred - u_gt)
```

fixed epsilon `1e-6`을 사용하고, raw dimensionless loss·cosine loss·angle scalar·별도
magnitude 항은 사용하지 않는다. translation invariant이며 target span은 gradient를 받지
않는다. 생산 학습 설정에는 연결하지 않았다.

## 4. Synthetic contracts

6개 계약을 모두 PASS했다.

- identical vector와 uniform translation: zero
- 같은 방향, 다른 magnitude: zero
- rotation/opposite direction: 양의 반응 및 opposite가 더 큼
- 선택된 shoulder/hip endpoint에만 gradient
- target scale detach
- predicted span collapse에서도 finite loss/gradient

## 5. Fixed-batch gradient comparison

기존 A9/A11 진단과 동일한 seed 1337, augmentation, source-balanced permutation의 첫 epoch
10개 batch를 init/A9/A11/A12 네 상태에 replay했다. A12 상태의 raw ratio는 다음과 같다
(실제 coefficient 0.05 적용 시 20분의 1):

| 항 | raw ratio mean | p95 | max | cosine |
| --- | ---: | ---: | ---: | ---: |
| A12 Cartesian candidate | 5.34 | 6.78 | 6.79 | 0.318 |
| scale-restored direction candidate | 4.91 | 6.34 | 6.45 | 0.301 |
| A12 magnitude component | 6.12 | 9.64 | 10.21 | 0.203 |
| A12 direction component | 4.66 | 6.27 | 6.32 | 0.294 |
| historical angular yaw-tail | 2596.64 | 5962.44 | 7101.92 | 0.058 |

A9 상태에서도 Cartesian 5.28, direction 5.14, historical angular 3169.92였다. 방향
후보는 안정적이지만 A12 failure의 원인을 입증할 만큼 “magnitude가 direction을 희석했다”는
패턴은 아니었다.

## 6. Small-span behavior

합성 collapsed-span 계약에서 loss, endpoint gradient 모두 finite였다. fixed epsilon은 기존
angular yaw convention과 동일한 수치 guard이며 dataset-specific threshold가 아니다.

## 7. Real-data yaw association

A12 checkpoint에서 frame-level evaluator root-yaw error와의 Pearson r은 Cartesian `.325`,
magnitude `-.056`, scale-restored direction `.610`, historical angular `1.000`이었다.
따라서 방향 후보가 magnitude보다 orientation에 선택적으로 반응하는 성질은 확인했다.
historical angular의 1.000은 evaluator와 동일 quantity를 비교한 결과라 독립 성공 증거는
아니다. A9 checkpoint에서는 Cartesian `.346`, magnitude `.160`, direction `.338`이었다.

## 8. Source contribution and portability

direction 후보는 canonical shoulder/hip pair만 사용하므로 commercial dataset에도 구조적으로
이식 가능하다. translation invariance, detached target scale, fixed numerical guard는
dataset-specific 의미론에 의존하지 않는다. 다만 AMASS selection dominance는 상태에 따라
남아 있고, 후보가 이를 해결했다고 주장하지 않는다. source normalization/dynamic balancing은
이번 batch 범위를 벗어나므로 실행하지 않았다.

## 9. GO / NO-GO decision

**NO-GO.** 후보의 계약·gradient 안정성·yaw 선택성은 양호했지만, A12 수렴 상태에서 direction
energy가 약 80%이고 direction scalar도 magnitude보다 컸다. 따라서 A13 실행의 선행조건인
material direction dilution이 충족되지 않았다.

## 10. A13 metrics and matched qualitative review

A13은 실행하지 않았다. 그러므로 A13 metric, training telemetry, 새 qualitative review는
해당 없음이다. 기존 A9/A12의 고정 review 결과(동일 stairs/walking 및 4개 고정 3DPW
sequence)는 변경하지 않았고, A13에 유리한 새 프레임을 선택하지 않았다.

## 11. Architecture verdict

A12는 A11의 optimization-scale 붕괴를 해결했지만 3DPW orientation gate를 개선하지 못했다.
이번 attribution은 그 실패를 단순한 Cartesian magnitude-direction entanglement로 설명하지
못했다. 이 세션에서는 추가 orientation architecture를 구현·학습하지 않는다.

## 12. Exact files, tests, commits, server state

변경 파일:

- `src/training/temporal_lifter.py` — 진단-only scale-restored direction geometry helper
- `scripts/diagnose_yaw_tail_gradients.py` — A12 attribution, source/gradient/yaw replay
- `tests/test_scale_restored_direction.py` — 6 synthetic contracts
- `tests/test_diagnose_yaw_tail_gradients.py` — squared-energy attribution contract
- `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md` — 정량 결과 추가
- `docs/README.md` — worklog index 추가
- `docs/15_WORKLOG_A12_MAGNITUDE_DIRECTION_ATTRIBUTION.md` — 본 worklog

생성된 진단 JSON은 commit하지 않고 서버 output에만 보관한다:

`/home/nd/animcv-output/experiments/a12_direction_attribution_10b/diagnosis.json`

테스트/검증:

- focused pytest: `24 passed`
- `py_compile`: PASS
- full regression: major documentation/code closure에서 실행 예정

커밋:

- `0c5d6bd` — diagnose A12 torso residual geometry
- `5e56b1c` — fix diagnostic gradient concentration
- 다음 문서/테스트 closure commit 예정

`origin/On_Work`와 `LabServer63:/home/nd/AnimCV`는 위 코드 커밋까지 fast-forward 상태다.
`.vscode/`와 서버의 기존 미추적 `.DS_Store`, `.animcv_sync_stage/`, `docker/`는 건드리지
않았다. A9–A12 checkpoint, fingerprint, report, metric은 변경하지 않았다.
