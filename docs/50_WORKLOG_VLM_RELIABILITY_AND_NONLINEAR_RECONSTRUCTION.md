# Worklog — VLM Reliability Advisor와 bounded nonlinear reconstruction (2026-09-22)

## 범위와 시작 상태

이번 세션은 `arch/single_frame_first`에서 VLM visual reliability advisor와
결정론적 nonlinear gap reconstruction을 별도 research module로 검증하는
bounded architecture batch였다. 기존 Frame Pose production path와 닫힌
C0/C1/C2/C3 temporal refiner lineage는 재개하지 않았다.

시작 전에 `git fetch origin`을 실행했고 local HEAD와 원격 기준은 동일했다.

- branch: `arch/single_frame_first`
- start HEAD: `e91c4a504f9a18f6b9ebd890bc9bf579679fe5fc`
- start remote: `origin/arch/single_frame_first` 동일
- start worktree: clean
- fixed bank: `bank_3dpw_paired_v2`, content digest
  `75519e6394a764e3749ddaa30555b58b73a01db582ccee14f661374b9a0ed536`
- fixed H0 test SHA-256:
  `6df04c2031b2bde6a4f0af37fdae0a7f0228e9fc23401e15e1aa2b3bd35695a5`

모든 pose 수치는 `benchmark_detector_observation` regime이다. Track A의
RGB 판단은 별도 visual evidence이며, detector confidence를 VLM 입력이나
occlusion ground truth로 사용하지 않았다.

## 보존한 계약

FrameBank, frozen H0, Frame Pose Core, Geometry Observation, SignState, Pose
Reconciliation, canonical pose mathematics 및 과거 temporal evidence는
변경하지 않았다. 임시 canonical representation은 다음 세 항목뿐이다.

- parent joint position
- child direction
- trusted local bone length와 direction에서의 deterministic swing quaternion

full target-rig transform, roll/twist, wrist orientation, local rig axis는
표현하지 않았다.

## Track A — VLM observation reliability

`src/framepose/reliability_advisor.py`에 canonical 17-joint 단일 응답 계약을
추가했다. 허용 상태는 `RELIABLE`, `WEAK`, `OCCLUDED`, `OUT_OF_FRAME`,
`UNKNOWN`이며, parser는 정확히 하나의 JSON object 또는 하나의 outer
Markdown JSON fence만 허용한다. 누락/추가 key, 중첩 값, prose, 숫자 confidence는
거절하고 전 joint를 `UNKNOWN`으로 둔다.

`scripts/diagnose_vlm_reliability.py`는 고정 Qwen/Qwen2-VL-2B-Instruct,
고정 448 crop/prompt, real RGB 대 deterministic shuffled RGB control을
실행한다. prompt에는 2D 좌표, detector confidence, depth, correction,
interpolation을 넣지 않았다.

고정 validation/test subset 64프레임에는 이전 owner replay sample을 포함했다.
결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| real valid response | 15 / 64 |
| shuffled valid response | 15 / 64 |
| real UNKNOWN 비율 | 76.5625% |
| real/shuffled 모두 valid인 프레임 | 6 |
| both-valid 상태 변경 | 0 |
| 관찰된 상태 | `RELIABLE`, `UNKNOWN`만 |
| `WEAK`/`OCCLUDED`/`OUT_OF_FRAME` 출력 | 0 |

따라서 shuffle change rate 28.125%는 image-grounded state change가 아니라
대부분 response validity 차이이며, both-valid에서는 실제 상태 변화가 없었다.
현재 local VLM은 reliability owner로 충분한 신호를 주지 못했다.

label inventory는 명시적으로 분리했다.

- `OUT_OF_FRAME`: finite valid normalized coordinate가 `[0,1]` 밖인 exact geometry
- `detector_missing`: 기존 `input_valid=false` contract 사실
- `structural_invalid` proxy: 위 두 가지의 합집합; occlusion label이 아님
- detector confidence: artifact에 기록만 하고 reliability ground truth로 승격하지 않음
- true `OCCLUDED` GT와 human annotation: 이번 batch에는 없음

산출물:

- `LabServer63:/home/nd/animcv-output/framepose/vlm_reliability_a_v1/diagnostic.json`
  SHA-256 `64e95e29e8455e28b68d6bf604ccde8b6493a2c30455a08e616f9da1461a62f7`
- `diagnostic_records.json` SHA-256
  `b1e386571a88bdeb44633ac7535bd7ca86f71da6e2442230563f3338e180ed94`
- `predictions.npz` SHA-256
  `e4ade5f9a0b8783fe1712966a821cb43866d9d71877feeba7b5d32ff6f9411a4`

## Track B — structural reliability control

`src/framepose/nonlinear_reconstruction.py`와
`scripts/run_nonlinear_reconstruction.py`는 VLM 없이 exact structural
reliability를 사용한 oracle/structural control이다. anchor는
`input_valid AND finite normalized coordinate inside [0,1]`일 때만 trusted로
간주하며 sequence 밖으로 support를 찾지 않는다.

복원 규칙은 평가 전에 고정했다.

- position: timestamp-aware cubic Hermite 한 segment
- tangent: 가능한 경우 `A-2,A-1,A`와 `B,B+1,B+2`의 finite-difference derivative
- direction: canonical bone direction → deterministic swing quaternion →
  hemisphere continuity → SQUAD
- child: reconstructed parent + interpolated direction × trusted length
- trusted length: `A-2,A-1,A,B,B+1,B+2` 중 finite positive length가 2개 이상일
  때의 local median
- 양쪽 anchor나 필요한 support가 없으면 `UNRESOLVED`; extrapolation과
  production fallback 없음
- linear interpolation은 historical quantitative control로만 저장

고정 test H0에서의 accounting:

| 항목 | 수 |
|---|---:|
| structurally usable joint rows | 113,894 |
| structurally unusable joint rows | 6,398 |
| contiguous gaps | 2,187 |
| reconstructable gaps | 2,092 |
| unresolved gaps | 95 |
| changed joint frames | 6,039 |
| reconstructable gap 중 target-valid score rows | 104 |

`benchmark_detector_observation`의 reconstructable target-valid rows에서
positive delta는 nonlinear error가 H0보다 커졌다는 뜻이다.

| population | H0 MPJPE | nonlinear MPJPE | delta (nonlinear − H0) | improved / worsened |
|---|---:|---:|---:|---:|
| reconstructable | 81.465 mm | 90.159 mm | **+8.694 mm** | 48 / 56 |
| gap 1–2 | 57.395 mm | 51.509 mm | −5.885 mm | 15 / 10 |
| gap 3–5 | 142.035 mm | 156.673 mm | +14.638 mm | 10 / 17 |
| gap >5 | 61.588 mm | 74.205 mm | +12.617 mm | 23 / 29 |

reconstructable population의 median delta는 `+2.686 mm`, p95 damage는
`+85.729 mm`였다. linear control은 `97.101 mm`로 더 나빴다. 즉 nonlinear
후보가 단순 linear보다 낫다는 사실만으로 H0보다 안전하다고 볼 수 없고,
실제 target-frame pose는 전체적으로 악화됐다.

계약 검증은 통과했다.

- endpoint position error: 0.0 mm
- trusted bone-length error: numerical zero 수준
- reconstruction 밖 joint: exact no-change
- velocity jump: median `665.322 mm/s`, p95 `3393.363 mm/s`

산출물:

- `LabServer63:/home/nd/animcv-output/framepose/nonlinear_reconstruction_b_v2/reconstruction.json`
  SHA-256 `64a4d69a8597005b3b25318cd3f47310426d11457d260b67fb899196f4fe4b99`
- `reconstruction.npz` SHA-256
  `c92a885520de6bcbf359c3c1100940ac9990fffb2224b4dce9c0d5a2ccd75ced`

## Owner replay / qualitative package

이전 owner qualitative examples를 다음 범주로 replay했다: tracking loss,
out-of-observation/subject absence, ankle/foot loss, stable jitter, good case,
implausible articulation, deterministic observation loss/degradation. 이는
현재의 human GT나 VLM correctness label로 재사용하지 않았다.

RGB sequence, structural state, sampled VLM state, H0, nonlinear output,
oracle target, two-sided anchor를 포함하는 8-event compact package를 만들었다.

- `LabServer63:/home/nd/animcv-output/framepose/reliability_reconstruction_review_v3/`
- `review_manifest.json` SHA-256
  `0dd1df2583c3d24dc5f6c7d562399d578005f2ea9fd0b003cd41d777653ff886`

## Track C와 최종 architecture verdict

Track C는 Track A가 meaningful image-grounded signal을 보여줄 때만 실행하도록
제한했다. 이번 Track A는 valid coverage가 23.4375%이고 both-valid shuffle
state change가 0이므로 조건을 충족하지 못했다. 따라서 VLM-gated reconstruction은
실행하지 않았고, VLM 결과를 structural anchor에 섞지 않았다.

최종 verdict는 **D — BOTH INSUFFICIENT**이다.

- reliability: parser contract와 real/shuffled control은 구현·측정됐지만
  현재 VLM이 reliability owner가 될 신호가 부족하다.
- reconstruction: structural oracle control은 계약적으로 bounded였지만
  eligible target-frame pose에서 H0보다 `+8.694 mm` 악화됐다.
- production 승격, root/contact/retargeting, 추가 temporal model은 이 batch에서
  시작하지 않는다.

## 변경 파일과 검증

추가 파일은 다음과 같다. production Frame Pose 파일은 수정하지 않았다.

- `src/framepose/reliability_advisor.py`
- `src/framepose/nonlinear_reconstruction.py`
- `scripts/diagnose_vlm_reliability.py`
- `scripts/run_nonlinear_reconstruction.py`
- `scripts/export_reliability_reconstruction_review.py`
- `tests/test_reliability_advisor.py`
- `tests/test_nonlinear_reconstruction.py`
- 본 worklog

검증 결과:

- focused reliability/nonlinear suite: `8 passed`
- repository suite: `PYTHONPATH=.:src pytest -q` → `594 passed, 42 skipped`
- parser fence/schema, same-sequence/no-cross, insufficient-support unresolved,
  Hermite endpoint/derivative/timestamp, quaternion hemisphere/SQUAD endpoint,
  trusted bone length, exact no-change outside gap을 focused tests로 고정했다.

최종 local commit 후 worktree는 clean 상태로 종료한다. LabServer63의 기존
untracked `.animcv_sync_stage/`, `docker/`, 이전 연구 파일은 건드리지 않았다.
