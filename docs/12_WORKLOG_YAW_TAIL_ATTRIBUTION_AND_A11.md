# Worklog — Yaw Tail Attribution and A11 (2026-08-25)

> 형식은 `docs/11_WORKLOG_REVIEW_TOOLING_AND_END_EFFECTOR_LOSS.md`가 정한 관례를 따른다:
> 세션 서사만 여기 남기고 파일명은 그 세션의 작업을 관통하는 이름으로 짓는다(날짜는 제목
> 안에만). 실험 수치·데이터 계약·gate 정의는 `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`가
> 단일 출처다. 서버 환경은 `docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md`를 그대로 참조한다.

## 시작 상태

A10(`end_effector_loss_weight=0.2`)이 거절된 직후. A9는 여전히 3DPW `root_yaw_p95_degrees`
gate(34.77° > 30°)만 실패 중이었고, 다음 가설을 정하지 않은 상태였다.

## 사용자 지시(요약)

새 아키텍처나 weight 튜닝에 들어가기 전에, 남은 yaw tail이 (A) 기존 `yaw_tail_loss`로 해결
가능한 학습 가능한 잔차인지, 아니면 (B) 2D 관측/orientation 표현 자체의 근본적 모호성인지부터
규명할 것. 두 조건(원인 규명이 학습 가능한 잔차를 가리킴 + `yaw_tail_loss` 계약 검증 통과)을
**모두** 만족할 때만 정확히 하나의 통제 실험(A9 + `yaw_tail_loss_weight=0.05`, A7에서 이미 썼던
값 재사용, 새 sweep 아님)을 실행. 결과가 부정적이어도 batch는 "성공"이며, 실패 시 다른 weight로
재시도하지 않는다.

## 한 것

1. **A9 통제군 재동결** — checkpoint SHA-256, config, dataset fingerprint 6개를 디스크에서
   다시 확인. 지난 세션 기록과 완전히 일치, 재학습 불필요.
2. **정량 원인 규명** — `scripts/attribute_yaw_tail.py`(신규, 학습 없음) 작성 후 A9 checkpoint로
   3DPW holdout 재추론. 공식 evaluator의 `_angle_delta`/`YAW_INDICES`/`_root_yaw_error_degrees`를
   그대로 재사용해 gate 수치와 일치함을 보장. 첫 실행에서 `_pair_disagreement`의
   "missing_shoulder_pair" 카테고리가 "missing_hip_pair"와 동일한 조건을 계산하는 복붙 버그를
   발견(4개 카테고리 합이 전체 프레임 수와 안 맞아서 드러남) — 수정하고, 기존 테스트가 우연히
   통과했던 이유(두 카테고리 크기가 우연히 같아서)까지 재현한 회귀 테스트로 교체(`2c407e4`).
   결과 요약은 `docs/10`의 "A9 root-yaw P95 tail 원인 규명" 절 참고.
3. **정성 원인 규명** — 원인 규명 단계에서 고정한 4개 시퀀스(worst-P95: stairs, 최장-run:
   walking, P95 근접: bus, 대조군: bar)를 각 271프레임 GT/예측 overlay + 가슴 wedge 영상·정지
   화면으로 렌더. 4건 모두 근사 180° 반전 없이 오차 크기만큼만 벌어짐을 확인.
4. **`yaw_tail_loss` 계약 검증** — `tests/test_yaw_tail_loss_contract.py`(신규) 5개 테스트로
   실제 torch/autograd 검증: tail 선택 개수 정확성, gradient 격리, config 격리. 전부 PASS.
   pooled (frame,pair) 순위와 gate의 frame-결합 순위 사이의 실제 근사(caveat)를 반례로 입증하고
   실 데이터 비율(3.8% 불일치)로 정량화(`30a0a3a`).
5. **GO 판정** — 두 조건 모두 충족(원인 규명이 학습 가능한 잔차를 가리킴, 계약 검증 PASS) →
   A11 실행.
6. **A11 실행 및 거절** — A9와 fingerprint 완전 동일, `yaw_tail_loss_weight=0.05`만 추가.
   yaw P95 개선 없음(오히려 34.77→35.21° 악화), 양쪽 holdout PA-MPJPE가 크게 악화돼 기존에
   통과하던 게이트까지 새로 실패(training MPJPE도 40.19→78.23mm로 거의 2배 악화). Section 3에서
   고정한 동일 4개 시퀀스로 A9/A11 matched 정성 비교도 수행 — 시각적 개선 없음. **Case B로 결론**:
   다른 weight/percentile로 재시도하지 않음. 상세 수치는 `docs/10`의 "A11 결과" 절 참고.

## 확정 커밋 (이 세션)

```
dfe654f feat: add quantitative yaw-tail attribution tool (diagnostic only)
2c407e4 fix: correct missing-shoulder-pair category in yaw-tail attribution
30a0a3a test: verify yaw_tail_loss contract before using it in A11
<이 문서 커밋 예정>
```

모두 로컬 → GitHub `On_Work` → `LabServer63:/home/nd/AnimCV` fast-forward 동기화 완료
(문서 커밋 제외, 아래에서 마무리).

## 아키텍처 해석

**yaw P95 실패는 loss-weight 튜닝으로 풀리지 않는다는 것이 이번 batch의 결론이다.** 원인 규명
자체는 tail이 (근사-180° 반전이 아닌) 학습 가능한 잔차 쪽에 가깝다는 증거를 보여줬지만, 유일하게
존재하는 학습 메커니즘(`yaw_tail_loss`)이 그 잔차를 실제로 줄이지 못했고 오히려 전반적 3D 적합을
크게 해쳤다. 다음 아키텍처 결정은 이 세션에서 구현하지 않지만, 후보 방향은:

- `yaw_tail_loss`의 pooled (frame,pair) 선택을 frame-결합 선택으로 바꾸는 재설계 (계약 검증에서
  드러난 근사를 없애는 방향 — 이것도 "다른 weight 재시도"는 아니고 손실 정의 자체의 수정이므로
  다음 batch의 별도 결정 사항)
- 표현/관측 증거 쪽 접근 — tail 프레임이 non-tail 대비 2D 어깨/엉덩이 span이 32~33% 좁다는
  부분적 상관은, 정면/후면에 가까운 자세의 관측 모호성이 일부 기여함을 시사

## 서 있는 작업 합의 (계속 유효)

`docs/11`과 동일 — data/output 대형 파일 미포함, 타 GPU 프로세스 불간섭, GPU polling 10분
이상 간격, commit/push는 Agent가 직접 수행, `.vscode/` 불간섭.
