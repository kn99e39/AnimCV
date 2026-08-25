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

(이 절은 진행하면서 채운다 — 진단 결과가 나오는 대로 아래에 기록)

## 확정 커밋 (이 세션)

(진행하면서 채운다)

## 서 있는 작업 합의 (계속 유효)

`docs/11`/`docs/12`와 동일 — data/output 대형 파일 미포함, 타 GPU 프로세스 불간섭, GPU polling
10분 이상 간격, commit/push는 Agent가 직접 수행, `.vscode/` 불간섭. A9/A10/A11 기록된 결과는
수정하지 않는다.
