# Web Dataset Training Plan

## 방향 정정

AnimCV의 학습 원천은 직접 촬영한 다중 카메라 데이터가 아니라, 웹에서 확보한 RGB/2D 관측과
3D pose GT가 짝지어진 데이터셋이다. 여기서 “자체 학습 데이터셋”은 원천
데이터를 AnimCV canonical schema로 변환·검증·분할해 만든 **파생 학습 세트**를 뜻한다.

기존 triangulation 코드는 현 학습 계획에서 사용하지 않는다.

## 원천 데이터 기록 원칙

- 웹에서 내려받을 수 있다는 사실만으로 RGB/2D/3D paired annotation이 보장되지는 않는다.
- 원천 데이터, RGB 영상, annotation, 모델 학습의 사용 조건과 version을 dataset별로 기록한다.
- 공개 motion archive는 RGB-2D→3D lifter의 짝 데이터가 아닐 수 있으므로 source format을
  먼저 확인한다.

## Codex 작업: 저장소 결합이 필요한 구현

| 우선순위 | 작업 | 완료 조건 | 신뢰 가능한 학습 차단 |
| --- | --- | --- | --- |
| C1 | dataset intake manifest | source URL/version/access conditions/sha256/사용 목적을 machine-readable로 기록한다. | 예 |
| C2 | source adapter 계약 | 원천 RGB/2D/3D/camera annotation을 canonical `PoseSequence`과 root-relative `LiftedPoseSequence`로 변환하는 공통 interface를 만든다. | 예 |
| C3 | 유효 supervision mask | invalid/occluded/missing 3D 관절을 loss·metric에서 제외하고, pelvis 무효 frame은 거부한다. | 예 |
| C4 | sequence-aware dataset | 복수 clip/take를 manifest로 결합하고 temporal window가 경계를 넘지 않게 한다. | 예 |
| C5 | own-data evaluator | PA-MPJPE, root yaw MAE/P95, knee/elbow flip, per-source/per-sequence 결과와 gate를 추가한다. | 예 |
| C6 | source-specific QA | image size, FPS, coordinate axes, unit, camera frame, joint mapping, train/holdout leakage를 검증한다. | 예 |
| C7 | end-to-end acceptance | adapter→training→holdout→Blender FBX 회귀 검증을 추가한다. | 신뢰 가능한 출력 |

실행 순서는 `C1 → C2 → C3 → C4 → C5 → C6 → C7`이다. C1–C6 전에는 의미 있는
학습·holdout 결론을 내리지 않는다.

## External GPT 작업: 웹 조사와 데이터 선정

외부 GPT는 최신 dataset availability·access conditions·형식·사용 조건을 조사한다. 결과는 코드가
아닌 evidence-backed source shortlist와 intake manifest 초안이어야 한다.

| ID | 작업 | 필수 산출물 |
| --- | --- | --- |
| G1 | 후보 데이터셋 조사 | RGB/2D/3D GT 유무, camera 정보, 규모, 포맷, URL, 버전, 라이선스 원문 링크 |
| G2 | 사용 조건 분류 | dataset별 접근·재배포·학습 조건을 원문 근거와 함께 정리 |
| G3 | adapter 난이도 분석 | AnimCV 17-joint, metre, camera-root-relative 출력으로 변환할 때 필요한 mapping/좌표 변환/결측 처리 |
| G4 | 추천 조합 | 최소 한 개의 권리 확인 가능 데이터와 독립 holdout 후보를 제시하고, 없으면 구매·계약 경로를 명시 |
| G5 | 다운로드·검증 계획 | 공식 다운로드 절차, 예상 용량, hash/version, license artifact 보관 방법, leakage 방지 split |

GPT는 사용 조건이 불명확한 dataset을 추정으로 단정하지 말고 `conditions unclear`로 표시한다.

## 수집 후 학습 경로

`공식 원천 dataset + license evidence → source adapter → canonical paired dataset → source-level holdout → supervised lifter training → independent quality gates → FBX visual acceptance`
