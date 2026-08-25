# Worklog — A11 Gradient Diagnosis (2026-08-25)

> 형식은 `docs/11_WORKLOG_REVIEW_TOOLING_AND_END_EFFECTOR_LOSS.md`가 정한 관례를 따른다:
> 세션 서사만 여기 남기고 파일명은 그 세션의 작업을 관통하는 이름으로 짓는다(날짜는 제목
> 안에만). 실험 수치·데이터 계약·gate 정의는 `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`가
> 단일 출처다. 서버 환경은 `docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md`를 그대로 참조한다.

## 시작 상태

A11(`yaw_tail_loss_weight=0.05`)이 직전 세션에서 Case B로 거절됨: yaw P95는 개선되지 않고
(34.77→35.21°), 양쪽 holdout PA-MPJPE가 크게 악화(3DPW 75.31→86.01mm, AMASS 69.19→94.32mm),
training MPJPE는 거의 두 배(40.19→78.23mm) 악화됐다. 왜 이렇게 파괴적이었는지는 아직 규명하지
않은 상태였다.

## 사용자 지시(요약)

`yaw_tail_loss`의 pooled (frame,pair) selector를 frame-level selector로 바로 교체하지 말 것.
먼저 A11의 붕괴가 (A) selector granularity, (B) gradient 크기 불균형, (C) yaw supervision과
기존 좌표/구조 objective 간 gradient 방향 충돌, (D) source/domain 불균형 중 무엇 때문인지
diagnostic으로 규명한다. Case A(selector 문제)가 강하게 뒷받침될 때만 frame-level selector
후보를 구현·학습한다.

## 한 것

1. **Section 1 (기록 재구성)** — `train()`은 epoch/step 단위 loss를 전혀 기록하지 않음을 확인
   (A9/A11 모두 post-training 집계 `training_mpjpe_mm` 하나뿐, A11은 detached 컨테이너라
   stdout log조차 없음). 이 한계를 그대로 보고하고, 재학습 대신 diagnostic replay로 대체.
2. **`scripts/diagnose_yaw_tail_gradients.py` 작성**(신규, 학습 없음) — A9/A11이 실제로 학습에
   썼던 direct_mix train 데이터셋과 정확히 같은 seed·augmentation·source-balanced permutation으로
   첫 epoch 배치를 재구성하고, 세 모델 상태(초기화 직후 / A9 최종 checkpoint / A11 최종
   checkpoint)에서 loss 성분(coordinate/bone/torso/hinge/yaw_tail raw+weighted)과 gradient
   상호작용(G_base, G_yaw norm·cosine, source별 isolate)을 계산.
   - frame-combined counterfactual selector(`_yaw_tail_loss_frame_level`)도 함께 구현 —
     evaluator와 동일하게 frame당 pair를 먼저 평균한 뒤 순위를 매기지만, 학습에는 쓰지 않고
     비교용으로만 사용.
   - 5개 유닛 테스트로 실제 torch/autograd에서 선택 시맨틱스·source 격리를 검증(server).
3. **10개 배치 × 3개 모델 상태 전체 실행** — GPU에서 실행, 결과 다운로드 후 집계·분석.
   상세 수치와 판정은 `docs/10`의 "A11 붕괴 원인 규명" 절 참고.

**핵심 결론**: gradient-scale 불균형(Case B)이 주 원인 — base task가 수렴할수록 base
gradient는 근처 극소점이라 작아지는데 `yaw_tail`의 raw 크기는 전혀 같이 줄지 않아, A9 수렴
상태에서 yaw/base gradient norm 비율이 **3,170배**까지 벌어진다. 방향 충돌(Case C)은 학습된
상태에서는 없었다(cosine 항상 양수). selector를 frame-level로 바꿔도(Case A) 이 비율은 거의
그대로였다(3,170배 vs 3,200배) — **selector 구조는 원인이 아니다.** source 불균형(Case D)도
동시에 존재: A9 수렴 상태에서 pooled tail loss의 87%가 AMASS 하나에서 나온다.

**결정**: Case A가 뒷받침되지 않으므로 Section 8(frame-level selector 후보 학습)로 진행하지
않는다. frame-level selector를 구현·학습하지 않았고, weight도 다시 만지지 않았다.

## 이식성 평가(향후 상용 데이터셋 재학습 관점)

frame-level selector로 전환하지 않기로 했으므로 새 프로덕션 코드는 없지만, 진단에서 나온
사실 자체가 이식성에 중요하다: `yaw_tail`류 objective를 이후에라도 쓰려면 그 raw 크기가
base 3D objective의 수렴 단계에 따라 상대적으로 폭발적으로 커진다는 것을 전제해야 한다 —
이건 3DPW/AMASS 특유의 아티팩트가 아니라 (1-cos) 형태의 각도 loss와 smooth-L1 좌표 loss의
근본적인 스케일 차이에서 오는 문제라서, 상용 데이터셋으로 바꿔도 그대로 재현될 가능성이 높다.
따라서 다음 orientation supervision 설계는 (a) base loss가 수렴해도 스케일이 같이 줄어들도록
정규화하거나, (b) 특정 데이터셋 하나에 편중되지 않도록 source별 정규화를 두는 방향이어야
한다 — 이번 batch에서 구현하지는 않는다.

## 다음 세션이 이어받을 지점

yaw P95는 여전히 미해결(A9: 34.77°, gate 30°). 이번 진단으로 "loss weight를 만지는 방향"과
"selector granularity를 바꾸는 방향" 둘 다 배제됐으므로, 다음 가설은 orientation supervision의
**형태 자체**(스케일이 자연히 수렴하는 정규화, 또는 소수 도메인에 좌우되지 않는 표현)를 겨냥해야
한다. 구체적 설계는 아직 없음 — 사용자와 논의 필요.

## 확정 커밋 (이 세션)

```
ddc7fb2 feat: add A11 gradient-diagnosis instrumentation (diagnostic only)
<이 문서 커밋 예정>
```

## 서 있는 작업 합의 (계속 유효)

`docs/11`/`docs/12`와 동일 — data/output 대형 파일 미포함, 타 GPU 프로세스 불간섭, GPU polling
10분 이상 간격, commit/push는 Agent가 직접 수행, `.vscode/` 불간섭. A9/A10/A11 기록된 결과는
수정하지 않는다.
