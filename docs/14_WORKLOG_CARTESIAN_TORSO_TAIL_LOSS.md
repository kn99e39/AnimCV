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

(진행하면서 채운다)

## 확정 커밋 (이 세션)

(진행하면서 채운다)

## 서 있는 작업 합의 (계속 유효)

`docs/11`~`docs/13`과 동일 — data/output 대형 파일 미포함, 타 GPU 프로세스 불간섭, GPU polling
10분 이상 간격, commit/push는 Agent가 직접 수행, `.vscode/` 불간섭. A9/A10/A11 기록된 결과는
수정하지 않는다.
