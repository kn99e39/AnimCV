# Worklog — A12 Source-Tail Aggregation and 3DPW Coverage (2026-08-26)

> 이번 batch는 orientation-loss representation을 중단하고 A12 Cartesian hard-tail의
> multi-source aggregation semantics만 진단했다. 정량 수치와 evaluator/gate 정의의 단일
> 출처는 `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`다.

## 1. Input source balance

직전 A9/A12와 동일한 seed `1337`, augmentation, coordinate normalization, window `81`,
batch `128`, source-balanced sampler replay를 사용했다. 원본 direct-mix frame mass는
MPI-INF-3DHP `106,512`, 3DPW `22,646`, AMASS `334,402`지만 deterministic source-balanced
epoch sample mass는 각각 `154,520`으로 동일했다. 대표 10개 batch도 MPI `403`, 3DPW `466`,
AMASS `411`이었다.

## 2. Global-tail source reweighting

A12 상태의 global pooled top-5% tail:

| Source | candidate/batch | selected/batch | within-source selection | total selected share | raw loss share |
| --- | ---: | ---: | ---: | ---: | ---: |
| MPI-INF-3DHP | 83.6 | 6.7 | 7.96% | 51.54% | 47.58% |
| 3DPW | 82.4 | 1.2 | **1.37%** | **9.23%** | 7.44% |
| AMASS | 87.7 | 5.1 | 5.87% | 39.23% | 44.97% |

입력 기회는 균형이지만 global hard mining은 3DPW를 source 내 1.37%만 선택했다. 이 현상은
실재하지만, source별 error distribution의 차이를 함께 확인해야 한다.

## 3. Per-source orientation/error distributions

A12 checkpoint direct-mix training frames의 분포:

| Source | yaw mean / median / P90 / P95 / P99 (deg) | Cartesian residual mean / median / P90 / P95 / P99 |
| --- | --- | --- |
| MPI-INF-3DHP | 5.50 / 4.77 / 10.16 / 12.30 / 17.44 | .000356 / .000249 / .000787 / .001036 / .001683 |
| 3DPW | **4.75 / 4.21 / 8.62 / 10.18 / 14.44** | **.000165 / .000092 / .000393 / .000563 / .001050** |
| AMASS | 7.99 / 4.48 / 17.60 / 27.34 / 57.02 | .000275 / .000136 / .000673 / .000993 / .001936 |

3DPW train은 global tail이 선택할 hard examples가 적은 source다.

## 4. 3DPW train/validation/test coverage comparison

| Split | Frames / sequences | A9 yaw mean / P95 / P99 | A12 yaw mean / P95 / P99 |
| --- | ---: | ---: | ---: |
| train | 22,646 / 34 | 4.33 / 9.66 / 13.24 | 4.75 / 10.18 / 14.44 |
| validation | 10,206 / 16 | 11.99 / 32.55 / 45.71 | 12.47 / 32.38 / 45.22 |
| official test | 35,310 / 37 | 14.90 / 34.77 / 50.11 | **15.87 / 38.45 / 53.41** |

GT torso turn delta P95는 train `4.418°`, validation `4.332°`, test `3.195°`였으므로 test의
turning motion이 더 크다는 설명은 지지되지 않는다. Input confidence mean은 train `.770`,
validation `.765`, test `.721`이었다. Sequence ID는 있지만 semantic action taxonomy와
camera-view label은 없어 view 비교는 unavailable이다. train은 `courtyard_*`/
`outdoors_climbing_*`, validation은 `courtyard_*`/`outdoors_parcours_*`, test는 `downtown_*`
sequence를 포함한다.

train yaw P95 `10.18°` 대 test `38.45°`, P99 `14.44°` 대 `53.41°`로 test hard-case coverage가
크게 다르다. 3DPW train에 test와 comparable한 hard orientation cases가 충분히 포함됐다고
보기 어렵다.

## 5. Per-source gradient norms

A12 fixed replay source-local gradient norm:

- MPI-INF-3DHP: `.1041`
- 3DPW: `.0623`
- AMASS: `.1281`

global A12 auxiliary gradient는 `.07054`, base gradient는 `.01326`, base와의 cosine은 `.318`이었다.

## 6. Pairwise cross-source gradient cosine

A12 상태 source-local gradient pair cosine:

- MPI-INF-3DHP–3DPW: `.188`
- MPI-INF-3DHP–AMASS: `.124`
- 3DPW–AMASS: `.095`

강한 cross-source gradient incompatibility나 cancellation은 관찰되지 않았다.

## 7. Diagnostic source-stratified counterfactual

각 source 자체에서 동일 A12 tail fraction을 고른 뒤 active source mean을 동일하게 aggregate했다.
hard-coded source weight는 없다. A12 상태에서 source별 within-source selected 비율은 MPI
`5.53%`, 3DPW `5.61%`, AMASS `5.27%`, total selected share는 각각 약 `33.3%`였다.

source-stratified raw local loss는 MPI `.001518`, 3DPW `.000826`, AMASS `.001867`였고,
aggregate gradient는 norm `.06742`, base cosine `.322`였다. global A12의 `.07054`/`.318`과
비슷해 A11식 gradient pathology는 재현되지 않았다.

## 8. Mechanism verdict

**Case B: 3DPW train coverage/domain shift가 primary limitation.**

Global selection starvation은 secondary mechanism으로 확인됐다. 그러나 source-stratified
aggregation이 복구하는 것은 현재 3DPW train의 낮은 tail을 더 자주 반영하는 것이며, official
test의 30–50° hard cases를 학습에 공급한다는 증거는 아니다. Case A의 relevant training
coverage 조건이 없으므로 source-stratified 후보를 학습하지 않는다. Case C의 강한 gradient
incompatibility도 아니다.

## 9. Conditional training result

Case A가 아니므로 source-stratified 후보 학습은 실행하지 않았다. 새 checkpoint와 holdout
metric은 없다.

## 10. A9/A12/candidate comparison

A9/A12의 기존 metric은 변경하지 않았다. source-stratified candidate는 diagnostic-only이며
metric 비교 대상이 아니다. A12의 기존 3DPW yaw MAE `15.87°`, P95 `38.45°`도 그대로다.

## 11. Per-epoch source telemetry

새 학습을 하지 않았으므로 per-epoch source telemetry는 해당 없음이다. fixed replay에는 source
input mass, candidate count, selected count, within-source fraction, raw/weighted auxiliary
mass, source-local gradient norm을 기록했다. 기존 A9–A12 telemetry는 수정하지 않았다.

## 12. Portability assessment

counterfactual은 generic source IDs/groups와 동일 canonical A12 loss만 사용한다. source가
하나면 local tail mean으로 fallback하고, N개 source면 active source mean으로 aggregate한다.
수동 source weight 없이 global hard mining이 sampling policy를 덮어쓰는 현상과 source별
telemetry를 노출한다. 다만 genericity가 metric 개선을 보장하지 않으며, commercial corpus에도
source/group coverage 검증이 먼저 필요하다.

## 13. Exact files changed

- `scripts/diagnose_source_tail_aggregation.py`
- `tests/test_source_tail_aggregation.py`
- `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`
- `docs/README.md`
- `docs/16_WORKLOG_SOURCE_TAIL_AGGREGATION.md`

이번 batch에서 production training loss/config, A12 coefficient, tail fraction, dataset mixture,
augmentation, optimizer, yaw gate는 변경하지 않았다. `.vscode/`와 A9–A12 checkpoint/fingerprint/
report도 변경하지 않았다.

## 14. Tests executed

- source-tail focused tests: `4 passed`
- A12 Cartesian/source-balance focused tests: `10 passed`
- `py_compile`: PASS
- LabServer63 Docker replay: PASS

## 15. Commit hashes and synchronization state

- `1eaa64d` — diagnose source stratified A12 tail
- `13a365f` — match source replay RNG order
- documentation closure commit 예정

진단 JSON은 commit하지 않고 다음 서버 output에 보관한다.

`/home/nd/animcv-output/experiments/a12_source_tail_aggregation_diagnosis/diagnosis.json`

`origin/On_Work`와 `LabServer63:/home/nd/AnimCV`는 `13a365f`까지 fast-forward 상태다.
서버의 기존 미추적 `.DS_Store`, `.animcv_sync_stage/`, `docker/`도 건드리지 않았다.
