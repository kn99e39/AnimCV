# README_EXEC — AnimCV 실행 가이드

이 문서는 `Architecture_v2.md`에 정의된 7개 마일스톤이 모두 구현된 현재 상태 기준으로, **실제로 이 프로그램을 어떻게 설치하고 실행하는지**를 정리한 실행 가이드입니다. 설계 배경은 `Architecture_v2.md`, 마일스톤별 구현 상세와 검증 내역은 `result/result_mil1.txt` ~ `result_mil7.txt`, 실제로 발견되어 고쳐진 버그 목록은 `result/result_occurred_errors.txt`를 참고하세요.

## 0. 현재 상태 요약

- 섹션 11의 마일스톤 1~7 전부 구현 완료.
- CLI 명령 8개(`extract-frames`, `estimate-pose`, `parse-rig`, `create-mapping`, `build-motion`, `retarget`, `optimize`, `export-blender`) 전부 스텁이 아니라 실제로 동작함.
- 유닛/통합 테스트 130개 전부 통과 (`pytest`).
- **Windows뿐 아니라 macOS에서도 실행 가능하도록 Blender 실행파일 자동탐지 로직을 OS별로 분기 처리함** (Windows/macOS/Linux 각각의 표준 설치 경로를 확인). macOS 실제 기기로는 검증하지 못했지만, 경로 탐색 로직 자체는 가짜 파일시스템으로 단위 테스트됨 (아래 "Mac 지원" 항목 참고).
- 합성 데이터로 **전체 파이프라인(영상 → 프레임 → 포즈 → 모션그래프 → 리타겟 → 키프레임 최적화 → Blender 출력)을 처음부터 끝까지 실제로 실행해서 결과 `.blend`/`.fbx` 파일에 키프레임이 정확히 들어가는 것까지 확인함**.
- 단, 아래 두 가지 외부 의존성은 이 개발 환경 자체의 제약으로 실제 모델/라이브러리로는 검증하지 못했음 (코드 자체는 준비되어 있고, 로직은 가짜(mock) 객체로 단위 테스트됨):
  - **MMPose**: 실제 포즈 추정 모델 체크포인트로 `estimate-pose`를 돌려본 적은 없음.
  - **Assimp(pyassimp)**: 네이티브 assimp 공유 라이브러리가 이 환경에 설치되지 않아 `parse-rig`/`create-mapping`/`retarget`이 실제 FBX/리그 파일을 파싱하는 것은 검증하지 못함.
- **Blender 연동은 실제 로컬 Blender 4.5 LTS / 5.1 두 버전 모두로 완전히 검증됨.**

## 1. 설치

### 1.1 필수 요구사항

- Python 3.11 이상 (개발 시 3.14로 테스트됨)
- [Blender](https://www.blender.org/) (4.5 이상 권장) — `export-blender` 명령에서만 필요하며, 시스템에 설치만 되어 있으면 됨 (pip 설치 아님)

### 1.2 Python 패키지 설치

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # macOS / Linux
pip install -e ".[dev]"
```

이 기본 설치만으로 `extract-frames`, `parse-rig`, `create-mapping`, `build-motion`, `retarget`, `optimize`, `export-blender`는 (외부 라이브러리 문제가 없다면) 전부 실행 가능합니다. `numpy`, `opencv-python`, `pyyaml`이 기본 의존성으로 함께 설치됩니다.

### 1.3 선택적 의존성

**포즈 추정(estimate-pose)을 실제로 쓰려면:**

```bash
pip install -e ".[pose]"   # mmpose, mmcv, mmengine, mmdet (매우 무거움 — PyTorch 포함)
```

추가로 MMPose 모델 설정 파일(config)과 체크포인트(checkpoint)가 필요합니다. `third_party/mmpose/configs`에서 원하는 모델 config를 찾고, 해당 체크포인트를 MMPose 모델 zoo에서 받아와야 합니다.

**리그 파싱(parse-rig / create-mapping / retarget)을 실제로 쓰려면:**

```bash
pip install pyassimp
```

`pip install`만으로는 부족합니다. **네이티브 assimp 공유 라이브러리(assimp.dll/libassimp.so 등)를 시스템에 직접 설치**해야 `pyassimp`가 정상 동작합니다 (이 문서를 작성한 개발 환경에는 이 네이티브 라이브러리가 없어서 실제 FBX 파일로는 테스트하지 못했습니다). 라이브러리가 없으면 `import pyassimp` 자체가 실패하며, CLI는 이를 잡아서 친절한 에러 메시지로 알려줍니다 (트레이스백이 아니라).

**Blender 연동(export-blender):**

별도 pip 설치 없이, 시스템에 Blender만 설치되어 있으면 됩니다. 실행 시 다음 순서로 Blender 실행 파일을 자동으로 찾습니다:

1. `--blender-executable` 옵션으로 직접 지정
2. `BLENDER_EXECUTABLE` 환경변수
3. `PATH`에 등록된 `blender`
4. OS별 기본 설치 경로 (아래 "Mac 지원" 항목 참고)

어느 것도 못 찾으면 `FileNotFoundError`로 명확히 알려줍니다.

### 1.4 Mac 지원

이 프레임워크의 핵심 로직(순수 Python + `pathlib`)과 외부 의존성(OpenCV, MMPose, pyassimp, Blender)은 전부 macOS에서도 동작합니다. Blender 실행파일 자동탐지도 OS별로 분기되어 있어 별도 설정 없이 Mac 표준 설치 위치를 찾습니다:

- `/Applications/Blender*.app/Contents/MacOS/Blender`
- `~/Applications/Blender*.app/Contents/MacOS/Blender` (사용자별 설치)

Mac에서 필요한 추가 설정:

```bash
brew install assimp     # parse-rig / create-mapping / retarget에 필요한 네이티브 라이브러리
pip install pyassimp
```

MMPose(`estimate-pose`)는 Mac에서도 `pip install -e ".[pose]"`로 설치되지만, NVIDIA CUDA가 없으므로 `--device cpu`로 실행해야 합니다 (Apple Silicon의 MPS 백엔드 지원 여부는 MMPose 버전에 따라 다르므로 사전 확인 필요).

**주의**: 이 macOS 관련 코드(`_default_blender_search_paths`의 Darwin 분기)는 가짜 파일시스템 레이아웃으로 단위 테스트되어 있지만(`tests/test_cli.py`), 실제 Mac 기기에서 실행해본 적은 없습니다. 실제 Mac에서 문제가 발견되면 알려주세요.

## 2. 전체 파이프라인 흐름

```
영상(.mp4) 또는 이미지 시퀀스
      │  extract-frames
      ▼
프레임 이미지 시퀀스 (PNG + metadata.json)
      │  estimate-pose  (MMPose 필요)
      ▼
pose.json  (PoseSequence)
      │  build-motion
      ▼
motion_graph.json  (MotionGraph — 중간 표현, 리그와 무관)

리그 파일(.fbx/.blend)
      │  parse-rig  (Assimp 필요)
      ▼
rig_profile.json  (RigProfile — 본 계층 구조)
      │  create-mapping  (사용자가 대화형으로 본 ↔ 관절 매핑)
      ▼
mapping.json  (BoneMappingProfile)

motion_graph.json + rig_profile.json(또는 원본 리그 파일) + mapping.json
      │  retarget
      ▼
animation.json  (AnimationClip — dense, 프레임마다 키프레임)
      │  optimize
      ▼
animation_optimized.json  (AnimationClip — sparse, 편집 가능한 키프레임)
      │  export-blender  (Blender 필요)
      ▼
result.blend (+ 선택적으로 result.fbx)
```

## 3. 명령어별 사용법

모든 명령은 `python -m app.cli <서브커맨드> [옵션...]` 형태입니다. (`pyproject.toml`에 `motion-tool` 콘솔 스크립트도 등록되어 있어, 패키지를 설치했다면 `motion-tool <서브커맨드>`로도 실행 가능합니다.)

### 3.1 `extract-frames` — 영상을 프레임 이미지로 분해

```bash
python -m app.cli extract-frames --video input.mp4 --out cache/frames --fps 24
```

- `--video`: 입력 영상 경로 (필수)
- `--out`: 프레임 PNG들과 `metadata.json`을 저장할 디렉터리 (필수)
- `--fps`: 원본보다 낮은 fps로 다운샘플링하고 싶을 때 (선택, 기본값은 원본 fps 유지)

### 3.2 `estimate-pose` — 프레임에서 2D 포즈 랜드마크 추출

```bash
python -m app.cli estimate-pose \
  --frames cache/frames \
  --out cache/pose.json \
  --pose-config third_party/mmpose/configs/.../some_config.py \
  --pose-checkpoint /path/to/checkpoint.pth \
  --device cpu \
  --visibility-threshold 0.3
```

- `--frames`: `extract-frames`로 만든 프레임 디렉터리 (필수)
- `--pose-config` / `--pose-checkpoint`: MMPose 모델 설정/체크포인트 (필수)
- `--device`: `cpu` 또는 `cuda` (기본값 `cpu`)
- `--visibility-threshold`: 이 신뢰도 미만인 랜드마크는 "안 보임" 처리 (기본값 0.3)

MMPose가 설치되어 있지 않으면 트레이스백 대신 설치 안내 메시지가 출력됩니다.

### 3.3 `parse-rig` — 리그 파일에서 본 계층 구조 추출

```bash
python -m app.cli parse-rig --rig character.fbx --out cache/rig_profile.json
```

- 이 명령은 `Architecture_v2.md`의 CLI 목록에는 없지만, 마일스톤 3의 수용 기준("rig_profile.json 산출물 생성")을 만족시키기 위해 추가한 명령입니다.
- Assimp 기반이라 `.fbx`뿐 아니라 Assimp가 읽을 수 있는 대부분의 포맷을 지원합니다.
- **주의**: 리그의 모든 씬 노드를 "본 후보"로 취급합니다 (사람이 아닌 임의 스켈레톤 지원을 위해). 실제로 애니메이션이 적용되는 건 다음 단계에서 사용자가 매핑한 본뿐입니다.

### 3.4 `create-mapping` — 본과 비디오 관절을 대화형으로 매핑

```bash
python -m app.cli create-mapping --rig character.fbx --frame cache/frames/00000.png --out profiles/mapping.json
```

실행하면 리그의 각 본에 대해 한 줄씩 순서대로 물어봅니다:

```
upper_arm.L> (landmark <name> | direction <a> <b> | custom_point <id> | skip) direction left_shoulder left_elbow
forearm.L> (landmark <name> | direction <a> <b> | custom_point <id> | skip) direction left_elbow left_wrist
hips> (landmark <name> | direction <a> <b> | custom_point <id> | skip) landmark pelvis
```

입력 가능한 명령:

| 명령 | 의미 |
|---|---|
| `direction <a> <b> [축힌트]` | 두 랜드마크(a→b) 사이의 방향 변화를 이 본의 회전으로 매핑 |
| `landmark <이름> [축힌트]` | 랜드마크 하나의 위치 변화를 이 본의 이동(translation)으로 매핑 |
| `custom_point <id> [축힌트]` | 사용자 지정 추적점을 이 본에 매핑 (아직 실제 추적 파이프라인은 없음) |
| `skip` 또는 그냥 엔터 | 이 본은 매핑하지 않고 건너뜀 |
| `done` | 여기서 중단하고 지금까지 답한 것만 저장 |

축 힌트는 `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z` 중 하나(생략 가능, 기본은 `+Z`).

여러 본을 이어지는 체인(예: 팔 전체)으로 매핑하고 싶으면, 특별한 문법 없이 그냥 연속된 본마다 `direction` 답을 이어서 입력하면 됩니다 (예: `spine_01`은 `direction pelvis chest`, `spine_02`는 `direction chest neck`).

랜드마크 이름은 다음 16개(가나다순 아님, `pose/schemas.py`의 `CANONICAL_LANDMARKS`)를 사용합니다:

```
pelvis, spine, neck, head,
left_shoulder, left_elbow, left_wrist,
right_shoulder, right_elbow, right_wrist,
left_hip, left_knee, left_ankle,
right_hip, right_knee, right_ankle
```

### 3.5 `build-motion` — 포즈 시퀀스를 Motion Graph로 변환

```bash
python -m app.cli build-motion --pose cache/pose.json --out cache/motion_graph.json
```

리그와 무관한 중간 표현(Motion Graph)을 만듭니다. 이 파일 하나로 나중에 다른 리그에 재사용(리타겟)할 수 있습니다.

### 3.6 `retarget` — Motion Graph를 리그에 적용해 애니메이션 곡선 생성

```bash
python -m app.cli retarget \
  --motion cache/motion_graph.json \
  --rig character.fbx \
  --mapping profiles/mapping.json \
  --out cache/animation.json
```

- `direction` 매핑 본: 첫 프레임 대비 각도 변화를 쿼터니언 회전으로 변환 (2D 이미지 평면 기준 — 깊이 추정 없음).
- `landmark`/`custom_point` 매핑 본: 첫 프레임 대비 위치 변화를 이동값으로 변환.
- 매핑 프로필에 있지만 리그에 없는 본, 또는 알 수 없는 매핑 모드는 에러 없이 조용히 건너뜁니다 (일부만 매핑해도 정상 동작).
- **이 결과물은 완벽한 3D 모션캡처가 아니라, 수동 보정을 전제로 한 "쓸만한 초안"입니다.**

### 3.7 `optimize` — 촘촘한 키프레임을 편집 가능한 수준으로 압축

```bash
python -m app.cli optimize \
  --animation cache/animation.json \
  --collapse medium \
  --out cache/animation_optimized.json
```

- `--collapse` 값: `none`(압축 안 함) / `light` / `medium` / `aggressive` / `custom`
- `custom`을 쓰려면 `--threshold <숫자>`를 반드시 같이 지정해야 합니다.
- 회전 오차는 도(degree) 단위, 이동 오차는 픽셀 단위 임계값입니다 (`light`=1°/2px, `medium`=3°/8px, `aggressive`=8°/20px).
- 각 본 트랙마다 원본/압축 후 키 개수와 최대 오차가 콘솔에 출력됩니다.
- **프레임 잠금(locked frame) 기능은 현재 Python API(`optimize/collapse.py`)에만 있고, CLI 옵션으로는 노출되어 있지 않습니다.**

### 3.8 `export-blender` — Blender에 키프레임 적용 후 저장/내보내기

```bash
python -m app.cli export-blender \
  --animation cache/animation_optimized.json \
  --rig character.fbx \
  --out result.blend \
  --fbx-out result.fbx
```

- `--fbx-out`은 선택사항이며, 지정하지 않으면 `.blend`만 만들어집니다.
- 내부적으로 실제 `blender.exe`를 `--background --python scripts/apply_motion.py -- ...` 형태로 서브프로세스 실행합니다 (이 프로젝트의 venv 파이썬은 `bpy`를 import할 수 없기 때문).
- 회전 매핑 본은 `rotation_quaternion`, 이동 매핑 본은 `location` 채널에 키프레임이 들어가며, 모든 키프레임의 보간(interpolation)은 `BEZIER`로 설정되어 Blender 그래프 에디터/도프시트에서 바로 편집 가능합니다.
- Blender를 못 찾으면 `--blender-executable`로 직접 경로를 지정하거나 `BLENDER_EXECUTABLE` 환경변수를 설정하세요.

## 4. 빠른 동작 확인 (합성 데이터로 전체 파이프라인 돌려보기)

실제 영상/리그 없이 프레임워크가 제대로 동작하는지만 빠르게 확인하고 싶다면, 테스트 스위트를 돌려보는 게 가장 빠릅니다:

```bash
pytest
```

130개 테스트가 전부 통과해야 정상입니다. 이 중 `tests/test_blender_executor.py`, `tests/test_apply_motion_script.py`는 가짜(fake) `bpy` 모듈로 Blender 없이도 동작 검증을 하고, `tests/test_rig_parser.py`, `tests/test_mmpose_adapter.py`는 가짜 노드/결과 객체로 assimp/mmpose 없이도 핵심 로직을 검증합니다.

**실제 Blender까지 포함한 검증**을 하고 싶다면 Blender가 설치되어 있어야 하며, `export-blender`를 실행한 뒤 생성된 `.blend` 파일을 Blender에서 직접 열어 그래프 에디터에 키프레임이 보이는지 확인하면 됩니다.

## 5. 알려진 제약사항

- 다중 캐릭터 미지원, 물리 시뮬레이션 없음, 완전 자동 3D 모션캡처 아님 — 이건 의도된 설계입니다 (`Architecture_v2.md` 섹션 1.3 "Excluded From v2" 참고).
- Depth Anything V2, SAM2, DWPose 등은 이번 버전(v2) 범위에서 의도적으로 제외되었습니다.
- 리그의 임의 스켈레톤(사람이 아닌 크리처, 소품 등)을 지원하지만, 자동으로 어떤 본이 "진짜 관절"인지 판단하지는 않습니다 — 사용자가 `create-mapping`으로 직접 지정해야 합니다.
- 회전 리타게팅은 2D 이미지 평면 각도 변화만 사용하며(깊이 정보 없음), 완벽한 3D 재구성이 아닙니다.

## 6. 참고 문서

- `Architecture_v2.md` — 전체 설계 문서 (원본 스펙)
- `README.md` — 마일스톤별 구현 상태와 기술적 디테일 (영어, 개발자용)
- `result/result_mil1.txt` ~ `result_mil7.txt` — 마일스톤별 실제 구현/검증 내역 (로컬 전용, git 추적 안 함)
- `result/result_occurred_errors.txt` — 실제로 발견되고 고쳐진 버그 목록 (로컬 전용, git 추적 안 함)
