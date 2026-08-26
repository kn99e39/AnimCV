# Worklog — 3DPW Generalization Support Diagnosis (2026-08-26)

> 이 batch는 추가 모델 학습 없이 3DPW generalization failure의 남은 ambiguity를 닫았다.
> `train prediction error tail`을 `GT target coverage gap`으로 해석하지 않고, A9 hard set,
> GT target-space support, canonical 2D input-space support, monocular ambiguity,
> temporal target coverage, sequence diversity를 분리했다. 정량 source of truth는
> `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`와 LabServer63의 최종 JSON이다.

## 1. GT target-space split comparison

3DPW split은 train `22,646 frames / 34 sequences`, validation `10,206 / 16`, official test
`35,310 / 37`이다. 모든 split은 FPS `30.0`이다. Root/torso orientation은 wrapped angle이므로
선형 평균보다 quantile과 geometry descriptor를 우선 해석했다.

| Split | torso orientation median/P95/P99 (wrapped degrees) | shoulder d(z) median/P05/P95/P99 (m) | hip d(z) median/P05/P95/P99 (m) |
| --- | ---: | ---: | ---: |
| train | `76.16 / 165.83 / 177.55` | `.0213 / -.0981 / .1239 / .1741` | `.00035 / -.0268 / .0281 / .0478` |
| validation | `82.62 / 167.03 / 177.37` | `.0157 / -.1081 / .1336 / .1866` | `.00244 / -.0267 / .0328 / .0570` |
| test | `3.35 / 173.31 / 178.29` | `-.00076 / -.1100 / .0984 / .1401` | `-.00784 / -.0315 / .0149 / .0272` |

여기서 지시문이 요구한 `d=z_right-z_left`는 canonical z축 성분으로 보고했다. AnimCV
좌표계는 `+X right, +Y forward/depth, +Z up`이므로 실제 forward-depth signed component인
`y_right-y_left`도 함께 계산했다. shoulder/hip forward-y의 train/validation/test
quantile은 서로 겹친다. GT target orientation/relative-depth state가 train에서 부재라는
Case A 증거는 없다.

## 2. Temporal target-space comparison

81-frame window에서 sequence 경계를 넘지 않게 FPS-aware finite difference를 계산했다.
orientation velocity P95/P99는 train `96.0/195.8°/s`, validation `93.2/172.2°/s`, test
`65.7/129.6°/s`; orientation acceleration P95/P99는 `517.9/1607.7`, `674.7/1780.1`,
`574.6/1710.1°/s²`이다. orientation window-path P95는 `299.9°`, `252.0°`, `189.5°`로
test가 train보다 더 큰 sustained turning trajectory를 보이지 않았다. signed-z velocity,
window net, sign transitions, longest signed orientation run의 전체 mean/std/median/P05/
P10/P90/P95/P99는 JSON의 `gt_target_space[*].temporal`에 있다.

## 3. 2D input-space split comparison

A9 checkpoint의 `pelvis_torso_v1` preprocessing 결과를 사용했다.

| Split | normalized shoulder span mean/median/P95 | normalized hip span mean/median/P95 | confidence mean | valid-joint mean |
| --- | ---: | ---: | ---: | ---: |
| train | `.823/.887/1.263` | `.543/.583/.819` | `.803` | `16.26` |
| validation | `.799/.864/1.269` | `.527/.572/.843` | `.801` | `16.19` |
| test | `.542/.384/1.204` | `.352/.244/.757` | `.757` | `16.11` |

normalized torso height는 계약상 `1.0`이다. test는 lateral projected evidence가 작고
confidence가 낮아 input domain이 validation/train과 동일하지 않다. raw image torso scale,
bilateral validity, projected left/right ordering, temporal 2D speed도 JSON에 보존했다.

## 4. A9/A12 hard-case overlap

A9 기존 root-yaw evaluator를 사용해 downstream 결과를 본 뒤 threshold를 바꾸지 않았다.

| Split | A9 eligible / top-5% / top-1% | A9 top-5 cutoff | yaw rank rho A9↔A12 | top-5 center overlap / Jaccard | window-frame Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | `22646 / 1133 / 227` | `9.663°` | `.113` | `14.1% / .076` | `.505` |
| validation | `10206 / 511 / 103` | `32.554°` | `.450` | `40.1% / .251` | `.490` |
| test | `35310 / 1766 / 354` | `34.778°` | `.409` | `33.2% / .199` | `.422` |

Top-1 center overlap은 train `10.6%`, validation `20.4%`, test `38.1%`이다. A12에서도 일부
동일 temporal areas가 hard하지만 overlap만으로 data insufficiency를 주장하지 않았다.
모든 top-5/top-1 record에는 sequence ID, frame ID, local index, window center/start/end와
81-frame ID 목록이 있다.

## 5. Target-space nearest-support results

Target descriptor는 canonical GT geometry, bilateral orientation, requested z/forward signed
components, 81-frame temporal changes를 포함한 27차원이다. train support mean/std만으로
표준화했고 split-specific threshold는 튜닝하지 않았다. same sequence support는 제외했다.

| Query → train | median / P95 target distance | median empirical percentile vs train control |
| --- | ---: | ---: |
| train → other train sequence | `.362 / .839` | control |
| validation hard top-5% | `.475 / 1.095` | `73.4%` |
| official test hard top-5% | `.549 / .947` | `82.8%` |

test target support가 control보다 tail에 있지만, raw GT orientation/relative-depth가 train에
없다는 분리 증거는 아니다. target-nearest support에서 test root orientation gap median/P95는
`8.30°/31.94°`이고 control 대비 median percentile `55.9%`다.

## 6. Input-space nearest-support results

Input descriptor는 A9 canonical 2D pose geometry, confidence/validity pattern, window temporal
2D motion을 포함한 119차원이며 target 3D 정보는 넣지 않았다.

| Query → train | median / P95 input distance | median empirical percentile vs train control |
| --- | ---: | ---: |
| train → other train sequence | `.464 / 1.003` | control |
| validation hard top-5% | `.855 / 1.208` | `91.2%` |
| official test hard top-5% | `.501 / 1.059` | `56.6%` |

validation은 input-domain shift가 강하다. test는 hard yaw가 커도 2D nearest support는
control 범위에 있어, test 실패를 단순 input absence로 설명할 수 없다.

## 7. Signed-relative-depth hard-case attribution

signed-relative-depth loss는 학습하지 않았다. A9 hard vs non-hard의 주요 값:

| Split/subset | shoulder forward-y abs residual / sign disagreement | hip forward-y abs residual / sign disagreement |
| --- | ---: | ---: |
| validation hard | `.174 m / 20.1%` | `.0617 m / 20.7%` |
| validation non-hard | `.0577 m / 6.3%` | `.0216 m / 6.7%` |
| test hard | `.231 m / 48.4%` | `.0907 m / 50.2%` |
| test non-hard | `.0810 m / 13.0%` | `.0216 m / 6.7%` |

지시문상의 z축 성분에서 test shoulder absolute residual은 hard/non-hard `.0314/.0316 m`로
비슷하지만, hip sign disagreement는 `43.3%/26.7%`였다. 따라서 실제 canonical forward-y
ordering과 hard yaw가 더 직접적으로 연결된다. center-edge temporal sign-transition
disagreement는 test shoulder/hip z `6.6%/9.7%`, validation `11.5%/13.3%`였다. A9/A12
각 state, shoulder/hip, z/forward-y의 전체 분포는 JSON의
`signed_relative_depth_attribution`에 있다.

## 8. 3DPW sequence diversity / replacement accounting

3DPW train은 unique sequence `34`, unique frame-center window `22,646`이다. direct-mix
source-balanced epoch에서 3DPW sample mass는 `154,520`이다.

- deterministic seed `1337` unique sampled windows: `22,622`
- duplicate sample count: `131,898`
- nominal replay factor: `6.823`
- realized replay factor: `6.831`
- top sequence share / top-5 share: `5.58% / 26.78%`
- sequence HHI: `.0354`

frame mass equality가 sequence-level diversity를 해결하지는 않지만 sampler는 변경하지
않았다.

## 9. Failure-class verdict

최종 판정은 **Case E — split별 mixed failure**다.

- Target coverage gap: primary 아님. GT pose/relative-depth/temporal target이 train에 없다는
  증거 없음.
- Input-domain shift: validation hard set의 primary. input support percentile `91.2%`.
- Monocular ambiguity: official test hard set의 primary. input percentile `56.6%`이지만
  input-nearest target orientation/forward-depth gap percentile이 약 `86–88%`.
- Model/objective failure: Case D 조건이 아니므로 primary로 판정하지 않음.
- Mixed: validation과 official test가 다른 failure class를 보여 global 단일 원인을 강제하지
  않음.

## 10. Exact limitations from unavailable metadata

3DPW prepared manifest에는 sequence ID와 frame provenance는 있지만 camera-view label과
semantic action taxonomy가 없다. 따라서 view/action overlap 및 causal action attribution은
불가하다. sequence ID 문자열(`courtyard_*`, `outdoors_*`, `downtown_*`)은 diversity/concentration
표시로만 사용했다. Wrapped orientation angle의 linear mean도 causal evidence로 쓰지 않았다.

## 11. Architecture interpretation

추가 mechanism을 자동 구현하지 않는다. 향후 의사결정은 validation에는 crop/projection/
confidence 및 camera/view diversity, official test에는 temporal evidence와 latent
orientation-state supervision/ambiguity-aware modeling을 우선 질문으로 삼는다. signed-
relative-depth supervision, temporal loss, sampler/mixture/augmentation/optimizer/loss
weight/gate 변경은 다음 별도 승인 batch로 미룬다.

## 12. Exact files changed

- `scripts/diagnose_3dpw_generalization_support.py`
- `tests/test_3dpw_generalization_support.py`
- `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`
- `docs/README.md`
- `docs/17_WORKLOG_3DPW_GENERALIZATION_SUPPORT.md`

Production training loss/config, A9–A12 checkpoints/fingerprints/reports/evaluator semantics,
existing gates, `.vscode/`는 변경하지 않았다. 진단 JSON은 generated artifact라 commit하지
않았다.

## 13. Tests executed

- focused diagnostic tests: `6 passed`
- `python3 -m py_compile scripts/diagnose_3dpw_generalization_support.py`: PASS
- `git diff --check`: PASS
- LabServer63 Docker diagnostic replay: PASS; final output `train 22646`, `validation 10206`,
  `test 35310`

Full pytest는 production shared code가 변경되지 않아 반복 실행하지 않았다.

## 14. Commit hashes / synchronization state

- `b7ab3a5` — input-nearest target-gap diagnostic
- `a950d1c` — temporal diagnostic unit names and missing-value handling
- `43857f8` — documentation closure

최종 진단 JSON:

`/home/nd/animcv-output/experiments/a9_target_input_support_diagnosis/diagnosis_final.json`

최종 JSON은 LabServer63에서 A9/A12 fixed inference와 sequence-disjoint support search를
수행해 생성했다. A13, source-stratified A12, signed-relative-depth loss, temporal loss는
실행하지 않았다.
