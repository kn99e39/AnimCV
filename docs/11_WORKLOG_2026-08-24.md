# Worklog — 2026-08-24

> 이 문서부터 "다음 Agent를 위한 handoff"가 아니라 **세션 단위 작업 기록**으로 남긴다.
> 매 세션 새 번호로 추가하고(다음은 `12_WORKLOG_<date>.md`), 이전 파일을 지우거나 덮어쓰지
> 않는다. 실험 수치·데이터 계약처럼 계속 갱신되는 사실은 여전히
> `docs/10_TEMPORAL_LIFTER_IMPROVEMENT_ABLATION.md`가 단일 출처(source of truth)이고, 이
> 문서는 "그 세션에 무엇을 왜 했는지"만 기록한다. 서버 환경/컨테이너 세부사항은
> `docs/06_SERVER_AI_AGENT_TRAINING_RUNBOOK.md`가 이미 다루므로 여기서 반복하지 않는다.

## 시작 상태

이전 세션이 `docs/11_SESSION_HANDOFF_2026-08-20.md`(이번에 삭제)에 남긴 결론에서 시작:
A9가 fingerprint 기준선이고 3DPW yaw P95(34.77°)와 hinge flip gate(2.36%)를 통과하지 못해
승격 보류 상태였다.

## 한 것

1. **A9 재검증** — 핸드오프에 적힌 수치를 그대로 믿지 않고, config/dataset fingerprint 6개를
   디스크의 실제 파일에서 SHA-256 재계산, checkpoint를 A8과 바이트 단위로 비교, 그리고 A9
   checkpoint로 두 holdout을 실제로 재추론해서 report 수치와 일치하는지 확인했다. 전부 일치.
2. **정성 리뷰 도구 3종 신규 제작** (`scripts/`):
   - `export_lifter_audit_frames.py` + 테스트 — checkpoint의 GT/예측을 정지 프레임 단위로
     `render_3d_audit.py`가 읽는 스키마로 export.
   - `export_lifter_audit_sequence.py` / `render_lifter_audit_video.py` + 테스트 — 연속 구간을
     GT/예측 골격이 겹쳐진 MP4로 렌더 (rig/mesh/retarget 불필요). 사용자 요청으로 이후
     두 가지를 추가:
     - 가슴(thorax)에 어깨선×척추축 외적으로 계산한 방향 쐐기(cone) — 본/구체 프록시엔 앞뒤
       구분이 없어서 root-yaw flip을 눈으로 못 알아보는 문제 해결.
     - flip된 hinge 관절(팔꿈치/무릎)을 예측 골격에서만 빨간색으로 강조 — 공식 평가와 동일한
       `_hinge_errors`를 재사용해서 report의 `hinge_flip_rate`가 세는 것과 항상 일치하도록 함.
   - 진행 중 Blender 5.1의 `image_settings.file_format`이 `media_type="VIDEO"`를 먼저 설정하지
     않으면 `FFMPEG`을 거부하는 걸 발견 — 기존에 이미 병합돼 있던
     `render_blender_animation_video.py`도 같은 버그가 있어서 같이 고쳤다(`fda0f59`).
   - 결과물은 Artifact로 공개해 사용자가 직접 보고 판단하도록 함
     (`https://claude.ai/code/artifact/6da086b9-ea2c-458d-97cd-165abe784689`).
3. **hinge_flip_rate gate 제거** — 리뷰 영상으로 3DPW holdout 37개 시퀀스를 직접 확인한 결과,
   집계 flip률이 가장 낮은 시퀀스조차 단일 프레임에 176° 반전이 있어 **flip률 0%인 시퀀스가
   하나도 없음**을 확인. 어떤 후보도 통과 불가능한 조건이었으므로 `criteria`에서 제거
   (`445efb8`). 지표 자체는 report에 계속 남긴다.
4. **IK 스타일 end-effector 위치 loss 추가** — `TrainingConfig.end_effector_loss_weight`
   (같은 커밋). limb-chain 말단(`left/right_wrist`, `left/right_ankle` —
   `constraint_target_builder.py`가 이미 "end_effector"라 부르는 것과 동일)의 위치 오차만
   추가로 가중. 실제 torch/autograd로 가중치 격리(말단 오차만 증폭, 동일 크기의 비말단 오차는
   그대로)와 `train()`+`evaluate()` 풀 라운드트립을 서버에서 검증.
5. **A10 실행 및 거절** — A9와 fingerprint 완전히 동일한 조건에 `--end-effector-loss-weight 0.2`
   하나만 추가. 3DPW yaw P95는 그대로/악화(34.77→34.91°), AMASS PA-MPJPE는 통과권에서 실패권으로
   악화(69.19→80.03mm). 상세 수치와 판정 근거는 `docs/10`의 A10 절 참고. 거절.

## 확정 커밋 (이 세션)

```
debab4f feat: add static-frame lifter audit export
68a7552 feat: add overlaid GT/predicted skeleton review video
fda0f59 fix: gate FFMPEG output on media_type for Blender 5.x
ec040d3 feat: add chest-forward direction wedge to skeleton proxy video
e5cf1c0 feat: color flipped hinge joints red in the audit review video
445efb8 feat: drop unreachable hinge-flip gate, add IK-style end-effector loss
c2910d7 feat: expose --end-effector-loss-weight on run_lifter_experiments.py
34019d7 docs: record A9 baseline, gate removal, and A10 (rejected)
```

모두 로컬 → GitHub `On_Work` → `LabServer63:/home/nd/AnimCV` 순으로 fast-forward 동기화 완료.

## 지금 gate (변경됨)

이제 3개뿐이다: `pa_mpjpe_mm ≤ 80mm`, `root_yaw_mae_degrees ≤ 15°`, `root_yaw_p95_degrees ≤ 30°`.
`hinge_flip_rate`는 report에는 남지만 통과 여부에 더는 영향을 주지 않는다.

## 다음 세션이 이어받을 지점

A10이 막 거절됐고, 다음 가설을 아직 사용자와 정하지 않았다. 후보로 논의된 것:

1. `end_effector_loss_weight`를 더 작게(예: 0.05) 재시도
2. 애초에 막고 있던 yaw P95를 직접 겨냥 — 이미 코드에 있는 `yaw_tail_loss_weight`(A9는 0으로
   꺼둠)를 켜보는 쪽이 원인에 더 가까울 수 있음
3. 다른 접근

시작 전에 `docs/10`의 A9 fingerprint와 항상 재확인하고, 한 번에 하나의 원인만 바꿔서 실행할 것.

## 서 있는 작업 합의 (계속 유효)

- data/output 대형 파일은 git에 넣지 않는다.
- 서버의 다른 사용자 GPU 작업/process는 건드리지 않는다.
- GPU 상태는 사용자가 요청하지 않는 한 10분보다 자주 polling하지 않는다.
- 사용자는 Agent가 commit/push를 직접 수행하는 것을 승인했다.
- 로컬 작업 트리의 `?? .vscode/`는 사용자 소유 — 수정·추적·삭제하지 않는다.
