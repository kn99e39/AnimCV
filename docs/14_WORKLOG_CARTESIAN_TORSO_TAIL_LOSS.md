# Worklog — Cartesian Torso-Tail Loss (2026-08-25)

> 형식은 `docs/11_WORKLOG_REVIEW_TOOLING_AND_END_EFFECTOR_LOSS.md`가 정한 관례를 따른다:
> 세션 서사만 여기 남기고 파일명은 그 세션의 작업을 관통하는 이름으로 짓는다(날짜는 제목
> 안에만). 실험 수치·데이터 계약·gate 정의는 `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`가
> 단일 출처다. 서버 환경은 `docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md`를 그대로 참조한다.

## 시작 상태

직전 세션의 A11 gradient 진단 결론: yaw_tail의 붕괴 원인은 selector granularity(Case A, 배제)나
방향 충돌(Case C, 배제)이 아니라 **gradient-scale 불균형(Case B)** — base task가 수렴할수록
base gradient는 거의 0으로 줄어드는데 (1-cos) 각도 loss의 raw 크기는 전혀 같이 줄지 않아
A9 수렴 상태에서 비율이 3,170배까지 벌어진다. AMASS 쪽 source 편중(Case D, 87%)도 부차 원인.

## 사용자 지시(요약)

각도(angular) yaw-tail objective를 weight tuning/gradient clipping/dynamic balancing/
selector 변경으로 구제하지 말 것. 대신 orientation error를 안정적인 3D geometry objective와
**같은 Cartesian 좌표 공간**에서 표현하는 새 후보(tail-selected torso-vector Cartesian residual)
하나를 정의하고, fixed-batch gradient 진단으로 먼저 검증한 뒤에만(그리고 GO 조건 충족 시에만)
정확히 하나의 통제 학습(A12)을 실행한다.

## 한 것

1. **Cartesian torso-tail loss 후보 구현**(`_cartesian_torso_tail_loss`, `src/training/temporal_lifter.py`)
   — 어깨/엉덩이 bilateral vector의 smooth-L1 잔차를, 기존 torso 구조 loss와 동일한 벡터
   컨벤션(`v = p_right - p_left`)으로 계산하고, `_yaw_tail_loss`와 완전히 같은 pooled
   tail-selector로 선택. 공유 selector 로직은 `_pooled_tail_mean`으로 리팩터링(기존
   `_yaw_tail_loss` 동작은 불변 — 35개 기존 테스트로 확인). `TrainingConfig.cartesian_torso_tail_loss_weight`
   추가(기본 0.0). synthetic contract 9개 테스트로 정상 geometry/오차 반응/translation
   불변/gradient 격리를 서버 torch로 검증.
2. **Epoch별 telemetry 추가**(`train()`의 `epoch_telemetry`) — A11 진단 때 "epoch 단위 기록이
   없어서 재구성 못함" 문제를 반복하지 않기 위해, 매 epoch 끝에 고정 512-window subset에서
   no-grad로 loss 성분(coordinate/bone/torso/hinge/yaw_tail_raw/cartesian_torso_tail_raw)과
   sample MPJPE를 기록. gradient/optimizer에 전혀 관여하지 않음 — seed 재현성(bit-identical
   checkpoint) 유지 확인.
3. **Fixed-batch 진단 확장 및 실행** — 기존 A11 진단 도구(`diagnose_yaw_tail_gradients.py`)에
   candidate의 loss/gradient/selection 계측을 추가해 동일 10개 배치 × 3개 모델 상태에서
   측정. 상세 수치는 `docs/10`의 "Cartesian torso-tail loss 후보와 A12" 절 참고.
4. **A12 실행** — GO 조건(gradient 지배 제거 + orientation 민감도 유지) 둘 다 충족돼 진행.
   A9와 fingerprint 완전 동일, `cartesian_torso_tail_loss_weight=0.05`만 추가.
5. **정성 비교** — A9 진단 단계에서 이미 고정한 stairs/walking 시퀀스로 A9 vs A12 GT/예측
   overlay 정지 화면 비교. 눈에 띄는 개선 없음 — 정량 결과와 일치.

**핵심 결론**: Cartesian 재구성은 **안정성 문제는 실제로 고쳤다**(training MPJPE 40.19→40.39mm,
A11의 78.23mm 붕괴와 달리 거의 그대로 — epoch telemetry로 수렴 궤적 직접 확인). 하지만
**orientation 품질은 개선되지 않고 3DPW에서 오히려 악화**됐다(yaw MAE 14.90→15.87°, 신규
gate 실패; yaw P95 34.77→38.45°, 더 나빠짐). AMASS는 3개 지표 모두 소폭 개선됐지만 A9에서
이미 전부 통과 상태였다. source 편중(Case D)도 candidate에서 그대로 관찰됨(AMASS가 tail loss의
45~92%).

**결정**: Case B(불안정성은 해결, orientation 품질 미해결)로 판정하고 weight를 조정하지
않는다. 다음 아키텍처는 orientation supervision의 형태(scale 통제와 방향 정보 분리) 자체를
다뤄야 한다.

## 이식성 평가

이 candidate는 canonical joint contract(어깨/엉덩이)만 쓰고 dataset-specific threshold나
yaw semantics가 전혀 없어서, 그 자체로는 상용 데이터셋에 이식하기 쉽다. gradient 스케일도
학습이 진행될수록 base task와 같이 자연스럽게 줄어드는 것을 epoch telemetry로 직접
확인했다 — 이 부분은 앞으로 orientation loss를 설계할 때 유지해야 할 성질이다. 다만
source 편중(AMASS가 tail의 45~92%)은 candidate를 바꿔도 그대로 남아 있어 **loss-form
자체의 이식성과 별개로, source 집계 품질은 여전히 미해결**이다 — 이번 batch에서 손대지
않았다.

## 다음 세션이 이어받을 지점

yaw P95는 여전히 미해결(A9: 34.77°, gate 30°). "loss weight 튜닝", "selector granularity
변경", "각도→Cartesian 표현 전환"까지 세 가지 방향이 이번 세 세션에 걸쳐 순서대로 배제됐다.
다음 가설은 두 가지 성질(gradient scale이 base task와 같이 수렴 / uniform translation 불변)을
유지하면서도 순수 magnitude가 아니라 **방향 정보를 실제로 겨냥하는** 표현이어야 한다 —
예를 들어 unit-normalize된 벡터의 차이를 쓰되 그 자체의 dimensionless scale 문제를 어떻게
피할지가 핵심 질문. 구체적 설계는 아직 없음 — 사용자와 논의 필요.

## 확정 커밋 (이 세션)

```
1b281ad feat: add Cartesian torso-tail loss candidate (Section 2-4)
9eae1ed feat: extend gradient diagnosis to the Cartesian torso-tail candidate
71bbe3e feat: record per-epoch loss telemetry so future runs stay auditable
f542678 feat: expose --cartesian-torso-tail-loss-weight on the experiment runner
<이 문서 커밋 예정>
```

## 서 있는 작업 합의 (계속 유효)

`docs/11`~`docs/13`과 동일 — data/output 대형 파일 미포함, 타 GPU 프로세스 불간섭, GPU polling
10분 이상 간격, commit/push는 Agent가 직접 수행, `.vscode/` 불간섭. A9/A10/A11 기록된 결과는
수정하지 않는다.
