# Worklog — AnimationSemantics owner-case bone-overlay 리뷰 (2026-09-29)

## 범위

docs/53의 owner-case replay는 JSON 수치와 skeleton 없는 원본 frame뿐이어서, 사람이
보고 무엇을 판단해야 할지 알 수 없었다. 이 세션은 docs/53이 persist한
`AnimationSemantics` 파일을 **읽기만 해서** bone overlay 리뷰 영상을 만들었다. 계약,
bridge, calibration, H0 중 어느 것도 수정·재계산하지 않았다.

- 시작 HEAD `d1ba2b6` (docs/53), 코드 commit `11bafe6`
- regime: `benchmark_detector_observation`

## 만든 것

`scripts/render_animation_semantics_review.py` — docs/53 owner case 6개 각각에 대해
±20 bank rows MP4(10 fps, 보간 없음)와 case frame PNG:

- RGB: 2D detector 입력 skeleton, 그리고 H0 skeleton 투영. 투영 anchoring은 docs/46과
  같은 `research_oracle_absolute_root_placement`(3DPW GT pelvis + intrinsics,
  `replay_pose_reconciliation._load_camera_state` 재사용)이며 **표시 전용**이다.
  AnimationSemantics는 root translation을 갖지 않는다.
- TOP view: semantic heading 화살표와 **그 frame의 어깨 heading**을 함께 그리고 각도
  gap을 적는다. hold latch가 오래된 값인지 눈으로 판단할 수 있다.
- FRONT view, 발목 상태 marker, row별 yaw-held / L·R foot / validity timeline.

Artifact: 서버 `~/animcv-output/animation_semantics_h0_replay/review/`(OpenCV `mp4v`),
로컬 `~/animcv-output/animation_semantics_h0_replay/review/`(macOS `avconvert` H.264
변환본 + 한국어 `보는법.md`). 레포에서는 gitignore된 `user_QE/animcv-output` symlink로
접근한다.

## 관찰 (정성적, case frame 기준)

| Case | 관찰 |
|---|---|
| turning (crosscountry_00:a0#202) | yaw −6.5° HELD, 현재 어깨 heading과 **gap 131°**. docs/53이 지적한 no-release latch로 heading이 **실제로 오래된 값**이 되는 것을 시각적으로 확인 |
| known_good_pose (dancing_00:a0#424) | clip 전체 HELD지만 gap 2° — latch가 우연히 맞는 경우. 서 있는 동안 양발 CONTACT-like, 2D·3D 일치 양호 |
| ankle_failure (parcours_00:a0#384) | H0 발목 투영이 실제 발과 크게 어긋남(docs/52 최대 ankle 오차). 발 상태는 MOVING/UNKNOWN만 나오고 CONTACT를 만들지 않음 |
| tracking_loss (hug_00:a1#312) | 앉은 자세, 왼팔 invalid. 발이 바닥에 있는데도 대부분 UNKNOWN — 보수적 거부 |
| walking T1-selected (drinking_00:a1#6) | "교대 최다" 선택 기준이 **서 있는 사람**을 골랐다. 초반 CONTACT 뒤 MOVING으로 바뀌지만 발은 육안상 움직이지 않음 → **false MOVING 의심** |

해석의 한계: 정지 frame 몇 장과 clip 6개에 근거한 정성 관찰이다. 비율 주장은 하지
않는다. false MOVING의 원인은 H0 jitter와 docs/53의 calibration stride 불일치(2 vs 3)가
유력하지만, 이번 세션에서는 분리 측정하지 않았다. 3DPW validation subset에 실제 걷기
장면이 부족해서, 교대 횟수로 "walking"을 고르는 방식이 걷기 장면을 보장하지 않는다는
점도 확인했다.

## 다음 단계로 넘길 것

1. Root Orientation: HELD frame의 heading과 현재 관측 heading 사이 gap을 연속
   지표로 측정 (target 없이 계산 가능). latch 문제의 정량화.
2. Contact: 정지 자세에서 나오는 MOVING의 비율을 stride를 맞춘 조건과 비교 —
   새 contact 실험이므로 별도 배치로.

## 테스트

렌더러는 계약 코드를 바꾸지 않는 리뷰 스크립트다. 기존 focused test(docs/53의 83개)에는
영향이 없고, 새 테스트는 추가하지 않았다. 대신 서버 렌더에 사용한 스크립트의 sha256이
commit된 파일과 일치함을 확인했다.
