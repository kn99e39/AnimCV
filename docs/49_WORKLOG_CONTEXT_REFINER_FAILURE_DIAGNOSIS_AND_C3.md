# Worklog — Context Refiner Failure Diagnosis and C3 Controlled Test (2026-09-22)

## 범위와 보존 원칙

이번 세션의 목적은 C2가 C1 parameter-free interpolation보다 약했던 이유를
진단하고, F1/F2가 실제로 입증될 때만 supervision-domain만 바꾼 C3를 한 번
검증하는 것이었다. C0/C1/C2의 architecture, gate, threshold, checkpoint,
previous temporal diagnostic 및 review 결과는 덮어쓰지 않았다.

모든 수치는 `benchmark_detector_observation` regime이다. `oracle_geometry`나
`real_animcv_observation`과 비교하지 않는다.

시작 시점의 repository identity:

- branch: `arch/single_frame_first`
- `git fetch origin` 완료 후 local/remote head 동일
- HEAD: `f01ef1aa153c3769be0e8b6e1328b9dac808bc99`
- 시작 status의 uncommitted files: `docs/48_WORKLOG_CONTEXT_ASSISTED_FRAME_REFINER.md`,
  `scripts/export_context_refiner_review.py`, `scripts/prepare_context_h0.py`,
  `scripts/run_context_refiner.py`, `src/framepose/context_refiner.py`,
  `tests/test_context_refiner.py`
- 시작 시 핵심 파일 SHA-256:

  - `src/framepose/context_refiner.py`: `c57f6f2ec12dbd2160158270c0e9f9db711b638302c0b284c719e09f57209a25`
  - `scripts/run_context_refiner.py`: `9fa4d02acf2a20a37a86af6f3b2e466540b8c233985fcdeec06745053fc4baf5`
  - C2 checkpoint: `039675c8c1bce26a2f8d77381294ca8d5913d75997de8d72174e256ddd118199`
  - C2 `experiment_report.json`: `c4995b5e13380613031dd3d82ef6229cb41c12fd4dcdddaabadaa9d82c870cf7`
  - frozen C2 test prediction: `1bbbd35fd189585d6ac79b4d94735908cf6d9fc57df3267c8f2f5c5eeaa8fa49`

현재 추가된 것은 진단/C3 실행기와 focused tests뿐이다. 현재 worktree에는
commit이나 push가 없으며, 핵심 C2 파일은 그대로다.

## C2 실제 training/runtime contract

고정 checkpoint를 재생하고 코드의 loss implementation을 직접 대조했다.

| 항목 | 확인 결과 |
|---|---|
| 학습 rows | train 11,334 frame 모두 입력 |
| residual 출력 | 17개 관절 모두에 대해 예측 |
| runtime/training gate | residual을 적용할 때 gate를 적용; gate-off 출력은 exact H0 |
| coordinate loss | target-valid gate-off 관절도 전체 coordinate objective에 기여 |
| structural loss | bone/torso/hinge가 gate-on/off 관절을 연결쌍·체인으로 coupling |
| validation 선택 | gate-conditioned가 아니라 whole-frame validation MPJPE |
| objective | `baseline_geometry_v1`: coordinate 1.0, bone .25, torso .15, hinge .15 |

실제 gate activation은 다음과 같다.

| split | frames | 전체 joint rows | gate-on rows | gate-on rate |
|---|---:|---:|---:|---:|
| train | 11,334 | 192,678 | 3,137 | 1.628% |
| validation | 3,407 | 57,919 | 2,290 | 3.954% |
| test | 7,076 | 120,292 | 4,390 | 3.649% |

### C1/C2 population parity

C1과 C2를 동일한 frozen runtime gate에 통과시켜 적용 여부를 exact array
비교했다. `C1-only=0`, `C2-only=0`이므로 population mismatch는 원인이 아니다.

| split | both | neither | C1-only | C2-only | gate mismatch |
|---|---:|---:|---:|---:|---:|
| train | 3,137 | 189,541 | 0 | 0 | 0 / 0 |
| validation | 2,290 | 55,629 | 0 | 0 | 0 / 0 |
| test | 4,390 | 115,902 | 0 | 0 | 0 / 0 |

저장된 C2 test prediction은 all-bank replay와 bitwise 동일했다.

## Residual attribution

각 gate-on target row에서 `R* = target - H0`, `R1 = C1 - H0`,
`R2 = C2 - H0`를 동일 population에 대해 비교했다. cosine, signed projection,
orthogonal magnitude, axis 성분은 scalar composite로 합치지 않았다. 전체
분포(mean/median/P05/P50/P95), X/Y/Z 및 모든 지정 cohort의 원자료는 다음에
있다.

`LabServer63:/home/nd/animcv-output/framepose/context_refiner_diagnosis_v1_retry/residual_attribution.json`

### ALL GATE-ON split 결과

아래 delta는 `H0 error - candidate error`라서 양수가 개선이다.

| split / candidate | target residual norm mean mm | learned norm mean mm | norm ratio mean | cosine mean | signed projection mean mm | orthogonal mean mm | error delta mean / median / P05 / P95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| train / C1 | 41.335 | 40.563 | 1.246 | .622 | 27.236 | 19.710 | 10.091 / 7.888 / -36.626 / 58.255 |
| train / C2 | 41.335 | 19.134 | .602 | .411 | 10.593 | 11.039 | 6.645 / 3.624 / -13.395 / 35.266 |
| validation / C1 | 89.281 | 55.550 | .913 | .410 | 24.999 | 31.072 | 8.588 / 6.166 / -60.000 / 83.335 |
| validation / C2 | 89.281 | 20.091 | .370 | .131 | 2.667 | 13.125 | .154 / .351 / -30.391 / 28.460 |
| test / C1 | 94.547 | 55.441 | .891 | .341 | 19.825 | 32.728 | 4.578 / 4.228 / -68.418 / 78.104 |
| test / C2 | 94.547 | 17.843 | .318 | .046 | 1.097 | 11.823 | -.746 / -.701 / -26.312 / 24.516 |

C2는 train에서는 개선을 학습하지만 test에서는 target 방향 projection이 거의
사라지고 cosine이 `.046`까지 낮아진다. 따라서 단순히 “residual이 0으로
축소됐다”거나 “항상 반대 방향이다”라고 단정할 수 없다. train/validation/test
분리에서 alignment/generalization failure가 직접 관찰된다.

Test ALL GATE-ON의 평균 axis 성분(mm)은 다음과 같다.

| candidate | target R* X/Y/Z | learned R X/Y/Z |
|---|---|---|
| C1 | -6.053 / -16.020 / 3.791 | .104 / -.818 / .544 |
| C2 | -6.053 / -16.020 / 3.791 | -3.863 / -.334 / -1.121 |

### 지정 cohort 결과

아래는 test gate-on row에서의 mean 요약이다. 완전한 p05/p50/p95 분포는
위 JSON에 보존했다.

| cohort / candidate | target norm mm | learned norm mm | norm ratio | cosine | orthogonal mm | delta mm | improved / worsened |
|---|---:|---:|---:|---:|---:|---:|---:|
| stable-2D/H0-jitter / C1 | 76.966 | 69.526 | 1.315 | .492 | 36.170 | 8.278 | 290 / 226 |
| stable-2D/H0-jitter / C2 | 76.966 | 14.386 | .349 | .070 | 9.670 | .071 | 247 / 269 |
| distal ankle / C1 | 362.020 | 107.591 | .313 | .538 | 44.179 | 57.742 | 46 / 14 |
| distal ankle / C2 | 362.020 | 30.091 | .086 | .202 | 14.044 | 5.081 | 35 / 25 |
| articulation mismatch / C1 | 130.262 | 69.273 | .811 | .445 | 35.329 | 10.507 | 37 / 24 |
| articulation mismatch / C2 | 130.262 | 20.162 | .234 | -.006 | 12.390 | -2.334 | 31 / 30 |

해석은 cohort마다 다르다. C2의 작은 learned norm은 ankle에서 일부 magnitude를
줄이지만 방향 정보는 약하고, articulation mismatch에서는 평균 projection이
음수로 바뀐다. stable jitter에서도 C1의 직접 evidence gain을 C2가 재현하지
못한다.

## Loss contribution accounting

실제 C2 output에 대해 동일한 canonical reduction의 분모를 유지하고 기여분만
분해했다. coordinate는 관절 단위, bone/torso/hinge는 “하나라도 gate-on이면
gate-on-or-coupled, 모두 off면 gate-off-only”로 표시했다. 이는 objective를
바꾼 masking이 아니다.

| split | coordinate off-only | bone off-only | torso off-only | hinge off-only |
|---|---:|---:|---:|---:|
| train | 97.690% | 95.356% | 94.340% | 93.165% |
| validation | 93.632% | 88.532% | 88.769% | 83.262% |
| test | 94.225% | 90.203% | 90.647% | 87.646% |

특히 train gate-on이 1.628%인 상황에서 gate-aligned rows가 objective의 대부분을
차지하지 않는다. 이것이 F1/F2가 “가능성”이 아니라 material contract 문제임을
보인다.

## C1 information-control 결과

C1은 promotion candidate가 아니라 정보 통제다. 같은 test prediction replay에서
standard frame-mean 평가와 joint-row slice 평가를 구분했다.

| candidate | standard MPJPE mm | standard PA-MPJPE mm | joint-row ALL TEST MPJPE mm |
|---|---:|---:|---:|
| C0 | 79.229 | 56.425 | 79.057 |
| C1 | 79.026 | 56.580 | 78.881 |
| C2 | 79.258 | 56.486 | 79.086 |
| C3 | 79.239 | 56.485 | 79.066 |

C1 gate-on에서는 `+4.578 mm`, 2,412 improved / 1,978 worsened였지만 damage-tail
P95는 `94.616 mm`였다. per-joint standard mean error의 C0 대비 개선값
(positive = C1 better)은 다음과 같다.

| joint | pelvis | L hip | L knee | L ankle | R hip | R knee | R ankle | spine | thorax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0-C1 mm | .064 | .079 | .010 | -.041 | .044 | -.007 | .494 | .073 | .065 |

| joint | neck | head | L shoulder | L elbow | L wrist | R shoulder | R elbow | R wrist |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0-C1 mm | .167 | .069 | .156 | .325 | .356 | .320 | .321 | .592 |

MPJPE는 절대 joint 위치 오차라서 sparse interpolation이 gate-on row에서
개선되면 내려갈 수 있다. PA-MPJPE는 frame별 similarity alignment 뒤 남는
relative geometry를 보므로, 관절별 interpolation이 pose 내부 관계를 일관되게
개선하지 못하면 반대로 약간 올라갈 수 있다. 이번 C1의 `79.229 -> 79.026`
대 `56.425 -> 56.580`은 그 divergence를 보여주는 정보 통제 결과이며, C1을
승격했다는 뜻은 아니다.

## 원인 분류

- **F1 training-domain dilution: 확인.** gate-on이 train joint rows의 1.628%이고,
  coordinate loss의 97.690%가 off-only였다. 구조 항도 93.165% 이상이 off-only였다.
- **F2 loss/runtime contract mismatch: 확인.** gate-off는 output에서 H0로
  고정되지만 coordinate objective에는 남고, 구조 항은 pair/chain coupling으로
  on/off가 분리되지 않는다. validation selection도 whole-frame이다.
- **F3 residual representation insufficient: 주원인으로는 미확인.** C2는 train에서
  개선되고, coordinate-only C3에서도 output이 완전히 무너지지는 않았다.
  capacity만으로 설명할 증거는 부족하다.
- **F4 generalization failure: 확인.** 같은 gate-on rows에서 C2 delta가
  train `+6.645`, validation `+0.154`, test `-0.746 mm`이고 cosine이
  `.411 -> .131 -> .046`으로 감소했다.
- **F5 gate/support mismatch: 주원인 아님.** C1/C2 applied population이 exact
  동일하고 mismatch count가 모든 split에서 0이다.
- **F6 mixed/other: 보조 분류.** F1/F2와 F4의 혼합으로 충분히 설명되며,
  별도의 다른 원인을 추가할 근거는 없다.

## C3 controlled test

F1/F2가 명확했으므로 C3를 한 번만 실행했다.

보존한 항목은 C2의 small temporal MLP(780-128-128-51, 123,059 params), fixed
runtime gate/threshold, AdamW, lr `1e-3`, wd `1e-4`, epochs 80, batch 256,
seed 1337, cosine schedule, mixed precision, whole-frame validation MPJPE
selection이다. 유일한 intervention은 기존 canonical `coordinate_only_v1`를
`target_valid AND gate_on` joint rows에만 적용한 것이다. pair/chain structural
loss는 의미를 바꾸는 임의 masking을 피하기 위해 제외했다. gate-off output은
계속 exact H0다.

### C3 test 결과

| slice | C0 | C1 | C2 | C3 | C3 gate-on delta |
|---|---:|---:|---:|---:|---:|
| ALL TEST joint-row MPJPE | 79.057 | 78.881 | 79.086 | 79.066 | -0.229 mm |
| ALL GATE-ON MPJPE | 94.547 | 89.969 | 95.294 | 94.777 | -0.229 mm |
| STABLE CONTROL | 40.200 | 40.200 | 40.200 | 40.200 | 0 gate rows |
| STABLE-2D/H0-JITTER | 80.404 | 77.590 | 80.380 | 80.080 | +0.952 mm |
| HIGH 2D INSTABILITY | 83.996 | 83.996 | 83.996 | 83.996 | 0 gate rows |
| OBSERVATION LOSS | 79.860 | 79.860 | 79.860 | 79.860 | 0 gate rows |
| DISTAL ANKLE | 340.467 | 337.808 | 340.233 | 340.457 | +0.223 mm |
| ARTICULATION-MISMATCH | 101.494 | 101.033 | 101.596 | 101.728 | -5.346 mm |

C3 gate-on test rows는 4,390개이며 2,110 improved / 2,280 worsened,
gate-off changed rows는 0개였다. C3는 C2보다 조금 덜 나빴지만 C1의 gain을
회복하지 못했고, stable jitter와 articulation mismatch를 동시에 해결하지도
못했다. 따라서 F1/F2 contract repair만으로는 충분하지 않으며 F4가 남는다.

C3 artifacts:

- directory: `LabServer63:/home/nd/animcv-output/framepose/context_refiner_c3_gate_aligned_v1/`
- checkpoint SHA-256: `b61e58326a4bdf7bcebbb1ae37fa15d3828be89d72828f00c96e413b1e687028`
- C3 test prediction SHA-256: `3c9f4528478d4bf2fe56bcdee43a36c7e3af19d547ea883a172e65015b7874da`
- `training_report.json`: `03b4e3eb9e0f7f33e789bc3de31038001a3cdb04eabc2d060b94b6ee90869fbc`
- `experiment_report.json`: `34b4b5d3962226a12733fe606a81392b1f5161667c8e5ecc2f2dc9cbd0426fb5`
- complete slice audit: `c3_audit_slice_metrics.json`
- complete gate audit: `c3_audit_gate_accounting.json`

Diagnosis artifacts:

- directory: `LabServer63:/home/nd/animcv-output/framepose/context_refiner_diagnosis_v1_retry/`
- summary: `ab5f160235f933074dd7ff304adee3a9bdbc768143833163f5ca7a7aa4d1feea`
- runtime contract: `40287ce70ea1ae17a1b43ceaea2a3e24fe1a340866cf8df101d37797b6201123`
- parity: `aa6d7285b171fa8aeefef99a34bfcd3f48d4e5aae2b7f18239fcff211b8cff85`
- residual attribution: `944f80973fbbd535f3f9b8c67a36410b89b350341fbe2d7ba09387372fa15a61`
- loss accounting: `112abe3aba221972ce2bbdc23312a912b5f095ae06343bdf2869eea9ea5a96a0`

## Verification and final decision

로컬 focused suite는 `12 passed, 2 skipped`였다(torch 미설치로 skip). LabServer63
CUDA container에서 context diagnostic, C3 contract, temporal evidence,
Frame Pose contract/evaluation/dependency isolation 및 canonical parity를 포함한
focused suite가 `49 passed`였다.

변경하지 않은 것: Frame Pose Core, H0, FrameBank, Pose Reconciliation,
SignState, canonical pose mathematics, historical temporal diagnostic, C0/C1/C2
artifacts 및 review outputs. 현재 결론은 **C3도 승격하지 않음**이다. 추가적인
temporal model, loss-weight tuning, gate tuning, architecture widening 또는
retargeting 작업은 이 결과만으로 시작하지 않는다.
