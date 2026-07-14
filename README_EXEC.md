# README_EXEC — AnimCV 실행 가이드

이 문서는 `Architecture_v2.md`에 정의된 7개 마일스톤 + 그 이후 추가된 품질 개선(깊이 기반 3D 리타게팅, 2-bone IK, rest-pose 축 보정)까지 반영된 현재 상태 기준으로, **실제로 이 프로그램을 어떻게 설치하고 실행하는지**를 정리한 실행 가이드입니다. 설계 배경은 `Architecture_v2.md`, 마일스톤별 구현 상세와 검증 내역은 `result/result_mil1.txt` ~ `result_mil8.txt`, 실제로 발견되어 고쳐진 버그 목록은 `result/result_occurred_errors.txt`를 참고하세요.

## 0. 현재 상태 요약

- 섹션 11의 마일스톤 1~7 전부 구현 완료.
- CLI 명령 8개(`extract-frames`, `estimate-pose`, `parse-rig`, `create-mapping`, `build-motion`, `retarget`, `optimize`, `export-blender`) 전부 스텁이 아니라 실제로 동작함.
- **v2 문서 범위를 벗어나는 추가 기능** (사용자 명시적 요청으로 구현, `Architecture_v2.md` 섹션 1.3/14.1이 v2에서 명시적으로 제외한 Depth Anything V2를 포함하므로 문서와 의도적으로 다름 — `result/result_mil8.txt` 참고):
  - **깊이(Depth Anything V2) 기반 3D 리타게팅**: 랜드마크에 상대 깊이값을 샘플링해 2D 평면 근사 대신 실제 3D 회전/이동을 계산 (깊이 없으면 자동으로 2D 방식으로 폴백)
  - **2-bone IK**: 어깨-팔꿈치-손목, 골반-무릎-발목 같은 체인을 코사인 법칙 기반으로 풀어서 end-effector가 실제 추적점을 더 정확히 따라가게 함
  - **Rest-pose 축 보정**: 리그의 rest pose 방향을 반영해서 본의 로컬 좌표계 기준으로 회전/이동을 재해석 (이전엔 리그 정보를 받기만 하고 실제로 안 썼음)
- 유닛/통합 테스트 220개 전부 통과 (`pytest`).
- **Windows뿐 아니라 macOS에서도 실행 가능하도록 Blender 실행파일 자동탐지 로직을 OS별로 분기 처리함** (Windows/macOS/Linux 각각의 표준 설치 경로를 확인).
- 합성 데이터로 **전체 파이프라인(영상 → 프레임 → 포즈 → 모션그래프 → 리타겟 → 키프레임 최적화 → Blender 출력)을 처음부터 끝까지 실제로 실행해서 결과 `.blend`/`.fbx` 파일에 키프레임이 정확히 들어가는 것까지 확인함**. IK 체인과 rest-pose 보정도 별도의 실제 Blender 픽스처로 재검증함.
- **macOS(Apple Silicon)에서 이 전체 파이프라인을 실제 의존성으로 처음부터 끝까지 검증함** (`mac/setup_mac.command`, 아래 "Mac 지원" 항목 참고): 합성 영상 → 실제 다운로드한 RTMPose-tiny 체크포인트로 `estimate-pose` → `build-motion` → 실제 Blender로 생성한 FBX 리그를 실제 assimp로 `parse-rig` → `create-mapping` → `retarget` → `optimize` → 실제 Blender 5.1로 `export-blender` → 결과 `.blend`를 다시 열어 fcurve/keyframe/interpolation이 정확히 일치하는 것까지 확인. 이 과정에서 실제 버그 3개를 발견/수정함 (아래 "Mac 지원" 항목 참고).
- **Blender 연동은 실제 로컬 Blender 4.5 LTS / 5.1 / 5.1.2 세 버전 모두로 완전히 검증됨** (기본 리타게팅 + rest-pose 보정 + IK 체인 각각 별도 픽스처로).
- **Depth Anything V2**는 이번 macOS 검증 범위에는 포함되지 않음 — 이전과 마찬가지로 실제 모델 클래스 forward pass까지만 확인했고(미학습 가중치), 실제 학습된 체크포인트로는 검증하지 못함.
- **터미널 CLI뿐 아니라 GUI(`python -m app.gui`, Tkinter)도 있음** — 8개 CLI 명령을 탭으로 감싸고, Mapping 탭은 `Architecture_v2.md` 섹션 6.2가 원래 설계한 대로 이미지 위 랜드마크를 직접 클릭해서 본을 매핑합니다 (아래 3.9절 참고). GUI를 통해 프로그래밍적으로 클릭을 재현해 합성 데이터로 전체 파이프라인을 끝까지 돌렸고, 결과가 CLI와 정확히 일치하는 것까지 확인함.

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

**macOS에서는 `pip install -e ".[pose]"` 한 줄로 끝나지 않습니다.** OpenMMLab은 mmcv의 macOS용 prebuilt wheel을 아예 배포하지 않아 항상 소스 빌드가 필요하고, 그 소스 빌드가 최신 setuptools/pip 툴체인과 충돌합니다(아래 "Mac 지원" 항목의 `mac/setup_mac.command` 참고). Linux/Windows에서 prebuilt wheel을 쓸 수 있는 환경이라면 이 한 줄로 충분합니다.

**리그 파싱(parse-rig / create-mapping / retarget)을 실제로 쓰려면:**

```bash
pip install pyassimp
```

`pip install`만으로는 부족합니다. **네이티브 assimp 공유 라이브러리(assimp.dll/libassimp.so 등)를 시스템에 직접 설치**해야 `pyassimp`가 정상 동작합니다. 라이브러리가 없으면 `import pyassimp` 자체가 실패하며, CLI는 이를 잡아서 친절한 에러 메시지로 알려줍니다 (트레이스백이 아니라).

**깊이 기반 3D 리타게팅(estimate-pose --depth-checkpoint)을 쓰려면:**

```bash
pip install -e ".[depth]"   # torch, opencv-python, matplotlib
```

추가로 [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) 체크포인트가 필요합니다 (`vits` 사이즈만 Apache-2.0, 나머지는 CC-BY-NC-4.0). `--depth-device`는 기본값 `auto`를 그대로 두세요 — Depth Anything V2 자체의 `infer_image`가 입력 텐서를 cuda/mps/cpu 중 시스템에서 사용 가능한 것으로 무조건 배치하고 이를 바꿀 방법이 없기 때문에, 명시적으로 다른 장치를 지정하면 모델 forward pass 도중 장치 불일치 에러가 나기 쉽습니다(`result/result_mil8.txt`에서 실제로 재현/확인함). `auto`를 벗어난 값을 넣으면 크래시 대신 명확한 에러 메시지가 먼저 나옵니다.

**Blender 연동(export-blender):**

별도 pip 설치 없이, 시스템에 Blender만 설치되어 있으면 됩니다. 실행 시 다음 순서로 Blender 실행 파일을 자동으로 찾습니다:

1. `--blender-executable` 옵션으로 직접 지정
2. `BLENDER_EXECUTABLE` 환경변수
3. `PATH`에 등록된 `blender`
4. OS별 기본 설치 경로 (아래 "Mac 지원" 항목 참고)

어느 것도 못 찾으면 `FileNotFoundError`로 명확히 알려줍니다.

### 1.4 Mac 지원

이 프레임워크의 핵심 로직(순수 Python + `pathlib`)과 외부 의존성(OpenCV, MMPose, pyassimp, Blender)은 전부 macOS에서 실제로 검증되었습니다 (Apple Silicon, Python 3.11, Blender 5.1.2). Blender 실행파일 자동탐지도 OS별로 분기되어 있어 별도 설정 없이 Mac 표준 설치 위치를 찾습니다:

- `/Applications/Blender*.app/Contents/MacOS/Blender`
- `~/Applications/Blender*.app/Contents/MacOS/Blender` (사용자별 설치)

**한 번만 실행하면 되는 설치 스크립트**: `bash mac/setup_mac.command` — venv 생성부터 pose/depth extras, native assimp, Blender 존재 확인까지 한 번에 끝납니다. `mac/build_gui_mac.command`(PyInstaller로 exe 하나에 통째로 번들, 수 GB, 재빌드도 느림)와는 다른 용도이니 혼동하지 마세요. `mac/setup_mac.command`는 그냥 평소처럼 `python -m app.cli ...`로 실행할 수 있는 venv만 준비합니다.

`estimate-pose`(mmpose 스택)를 macOS에서 설치하는 건 `pip install -e ".[pose]"` 한 줄로 안 끝나서 `mac/setup_mac.command`가 특정 순서로 여러 단계를 거칩니다 — 이유는 스크립트 안 주석에 적어뒀지만 요약하면:

- OpenMMLab이 mmcv의 macOS prebuilt wheel을 아예 배포하지 않아 항상 소스 빌드가 필요함.
- `mmdet`을 먼저 설치하면 그게 딸려오는 `mmcv`를 pip의 격리된 빌드 환경에서 빌드하는데, 그 환경엔 이미 설치된 torch가 안 보여서 컴파일된 연산(ops) 없이 조용히 "lite" 버전으로 떨어짐 — 나중에 `mmcv`를 다시 설치해도 이미 만족되는 버전이 있으면 pip가 아무것도 안 하므로 이 문제가 눈에 안 띄게 남습니다. `mmengine`을 먼저 깔고 `mmcv`를 명시적으로 빌드한 다음에 `mmdet`을 설치해야 합니다.
- mmcv/`chumpy`(mmpose 의존성)의 `setup.py`가 최신 setuptools/pip 툴체인에서 깨지는 레거시 패턴(`import pkg_resources`, `import pip`)을 쓰고 있어서 `setuptools<81` 고정 + `--no-build-isolation`이 필요함.
- `mmdet`은 `mmcv<2.2.0`을 요구하고, mmpose의 `xtcocotools`는 numpy 2.0 이전 C ABI로 컴파일되어 있어서 `numpy`/`opencv-python`도 같이 낮은 버전으로 고정해야 함.

Mac에서 필요한 네이티브 라이브러리 설치(`mac/setup_mac.command`가 자동으로 처리):

```bash
brew install assimp     # parse-rig / create-mapping / retarget에 필요한 네이티브 라이브러리
pip install pyassimp
```

**실제로 Mac에서 검증하며 발견하고 고친 버그 3개** (합성 영상 + 실제 다운로드한 RTMPose-tiny 체크포인트 + 실제 assimp로 만든 FBX + 실제 Blender로 전체 파이프라인을 끝까지 돌려봄으로써 발견됨 — 코드 리뷰만으로는 안 보였을 문제들):

1. **`pyassimp`가 Apple Silicon Homebrew의 assimp를 못 찾음**: pyassimp 자체 라이브러리 검색 로직(`pyassimp/helper.py`)이 `/usr/local/lib`(Intel Homebrew 기본 경로)만 확인하고 `/opt/homebrew/lib`(Apple Silicon Homebrew 기본 경로)은 확인하지 않습니다. `brew install assimp`로 라이브러리가 정상 설치되어 있어도 `import pyassimp`가 "assimp library not found"로 실패합니다. `src/rig/rig_parser.py`의 `_ensure_assimp_library_path()`가 macOS에서 `import pyassimp` 전에 `LD_LIBRARY_PATH`에 `/opt/homebrew/lib`를 추가해서 해결합니다.
2. **PyTorch 2.6+의 `weights_only=True` 기본값 변경과 mmengine의 충돌**: mmengine(0.10.7, mmdet의 `mmcv<2.2.0` 제약과 호환되는 최신 버전)의 체크포인트 로더가 `torch.load()`를 `weights_only` 지정 없이 호출하는데, OpenMMLab 체크포인트는 순수 텐서 이상을 담고 있어 PyTorch 2.6+의 새 기본값에서 로드가 깨집니다. `src/pose/mmpose_adapter.py`의 `_mmengine_checkpoint_compat()`가 모델 초기화 호출 동안만 `torch.load`의 기본값을 되돌리는 방식으로 해결합니다 (프로세스 전역이 아니라 그 호출 범위로 한정).
3. **`pyproject.toml`의 `pose` extra가 실제로 호환되는 조합을 명시하지 않았음**: `mmcv` 버전 제약이 없어 `mmdet`이 요구하는 `mmcv<2.2.0`과 어긋날 수 있었고, `numpy`/`opencv-python` 버전도 mmpose의 `xtcocotools` 네이티브 확장 ABI와 안 맞을 수 있었습니다. 지금은 `mmcv>=2.0.0rc4,<2.2.0`, `numpy<2`, `opencv-python<4.10`으로 명시되어 있습니다.

`estimate-pose --device cpu`로 실행하세요 (NVIDIA CUDA 없음; Apple Silicon의 MPS 백엔드 지원 여부는 MMPose 버전에 따라 다르므로 사전 확인 필요).

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
python -m app.cli extract-frames --video input.mp4 --out cache/frames --fps 24 \
  --start-frame 120 --end-frame 360
```

- `--video`: 입력 영상 경로 (필수)
- `--out`: 프레임 PNG들과 `metadata.json`을 저장할 디렉터리 (필수)
- `--fps`: 원본보다 낮은 fps로 다운샘플링하고 싶을 때 (선택, 기본값은 원본 fps 유지)
- `--start-frame` / `--end-frame`: 영상 전체가 아니라 특정 구간만 레퍼런스로 쓰고 싶을 때 (`Architecture_v2.md` 섹션 1.1의 "Start frame / End frame" 선택 입력 — 둘 다 원본 영상 기준 프레임 인덱스이고 양 끝 포함, 각각 독립적으로 선택 가능, 기본값은 영상 처음/끝). `--fps`와 같이 쓰면 잘라낸 구간 안에서 다운샘플링이 적용됩니다. `--start-frame`이 `--end-frame`보다 뒤면 에러가 납니다.

### 3.2 `estimate-pose` — 프레임에서 2D 포즈 랜드마크 추출

```bash
python -m app.cli estimate-pose \
  --frames cache/frames \
  --out cache/pose.json \
  --device cpu \
  --visibility-threshold 0.3 \
  --depth-checkpoint /path/to/depth_anything_v2_vits.pth \
  --depth-encoder vits \
  --depth-device auto
```

- `--frames`: `extract-frames`로 만든 프레임 디렉터리 (필수)
- `--pose-config` / `--pose-checkpoint`: MMPose 모델 설정/체크포인트 (**선택사항** — 생략하면 RTMPose-tiny 기본 모델을 씁니다: config는 설치된 `mmpose` 패키지에 이미 들어있는 파일을 그대로 쓰고, 체크포인트는 OpenMMLab 공식 model zoo에서 최초 1회 `~/.cache/animcv/models`로 내려받아 캐싱합니다(~13MB). 직접 받은 다른 모델을 쓰려면 두 옵션을 그대로 지정하면 됩니다 — `pose/default_model.py` 참고). 예: `--pose-config third_party/mmpose/configs/.../some_config.py --pose-checkpoint /path/to/checkpoint.pth`
- `--device`: `cpu` 또는 `cuda` (기본값 `cpu`)
- `--visibility-threshold`: 이 신뢰도 미만인 랜드마크는 "안 보임" 처리 (기본값 0.3)
- `--depth-checkpoint`: **선택사항**. 지정하면 Depth Anything V2로 프레임마다 깊이를 추정하고, 각 랜드마크 픽셀 위치에서 깊이값을 샘플링해 `pose.json`에 함께 저장합니다. 이후 `retarget`이 이 깊이 정보를 자동으로 감지해서 2D 평면 근사 대신 실제 3D 회전을 계산합니다.
- `--depth-encoder`: `vits`(기본, Apache-2.0) / `vitb` / `vitl` / `vitg` (셋 다 CC-BY-NC-4.0)
- `--depth-device`: 기본값 `auto` 유지 권장 (위 1.3절 참고)

MMPose/Depth Anything V2가 설치되어 있지 않으면 트레이스백 대신 설치 안내 메시지가 출력됩니다.

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

본 매핑을 다 마치면(또는 `done`으로 바로 넘어가면), **IK 체인**을 정의하는 두 번째 입력 루프가 시작됩니다 (`Architecture_v2.md`에는 없는 기능 — 아래 3.6절 참고):

```
ik-chain> (<root_bone> <mid_bone> <end_bone> <root_source> <mid_source> <end_source> [root_axis] [mid_axis] | done) upper_arm.L forearm.L hand.L left_shoulder left_elbow left_wrist
ik-chain> (... | done) done
```

한 줄에 `root_bone mid_bone end_bone root_source mid_source end_source`(필수) + `root_axis mid_axis`(선택)를 공백으로 구분해서 입력하면 됩니다. 빈 줄이나 `done`으로 종료합니다.

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

- `direction` 매핑 본: 첫 프레임 대비 각도 변화를 쿼터니언 회전으로 변환. `estimate-pose --depth-checkpoint`로 깊이가 샘플링되어 있으면 자동으로 실제 3D 회전(두 3D 벡터 사이의 최단 회전)을 계산하고, 깊이가 없는 프레임/랜드마크는 자동으로 2D 이미지 평면 근사로 폴백합니다.
- `landmark`/`custom_point` 매핑 본: 첫 프레임 대비 위치 변화를 이동값으로 변환 (마찬가지로 깊이 있으면 3D 오프셋, 없으면 2D).
- **IK 체인** (`create-mapping`에서 정의): 어깨-팔꿈치-손목처럼 2개 본으로 된 체인은 코사인 법칙 기반 2-bone IK로 풀어서, 손목(end effector)이 실제 추적된 위치를 더 정확히 따라가도록 root/mid 본의 회전을 계산합니다. 본 길이는 리그의 rest pose가 아니라 **첫 프레임에서 관찰된 랜드마크 간 거리로 자체 보정**합니다 (카메라 캘리브레이션이 없어서 리그 단위와 픽셀 단위를 직접 연결할 방법이 없기 때문).
- **Rest-pose 축 보정**: 리그 파일에 본의 rest pose 행렬 정보가 있으면, 위에서 계산한 회전/이동을 월드/이미지 좌표계가 아니라 그 본의 로컬 좌표계 기준으로 재해석합니다. 리그 정보가 없으면 이 단계는 그냥 생략됩니다(이전 동작과 동일).
- 매핑 프로필에 있지만 리그에 없는 본, 또는 알 수 없는 매핑 모드는 에러 없이 조용히 건너뜁니다 (일부만 매핑해도 정상 동작).
- **이 결과물은 완벽한 3D 모션캡처가 아니라, 수동 보정을 전제로 한 "쓸만한 초안"입니다.** (깊이/IK/rest-pose 보정으로 품질이 올라가지만 카메라 캘리브레이션이나 물리 기반 재구성은 여전히 없습니다.)

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

### 3.9 GUI — 터미널 명령어 대신 클릭 기반으로 전체 파이프라인 실행

```bash
python -m app.gui
```

`src/ui/gui_app.py`, Tkinter 기반(표준 라이브러리만 사용, CLI 쪽에 새 의존성 추가 없음 — 다만 파이썬 인터프리터 자체에 Tcl/Tk가 있어야 함, macOS는 `mac/setup_mac.command`가 `python-tk` 설치까지 자동으로 처리). 위 3.1~3.8절 8개 CLI 명령을 탭으로 그대로 감싸고 있고, 내부적으로는 CLI와 완전히 동일한 라이브러리 함수를 직접 호출합니다(`VideoLoader`, `PoseEstimator`, `RigParser`, `MotionGraphBuilder`, `RetargetSolver`, `collapse_animation_clip`). `export-blender`만은 `app.cli.run_export_blender`를 그대로 재사용합니다 (Blender 실행파일 자동탐지 + 종료코드 처리 로직이 이미 `tests/test_cli.py`로 검증되어 있어서 중복 구현하지 않음).

#### 탭별 사용법

창을 열면 위쪽에 8개 탭이 있고, 아래쪽엔 항상 상태 표시줄 + 로그 창이 있습니다. 각 탭의 결과 경로는 다음 탭의 입력 필드에 자동으로 채워지므로(예: Frames 탭에서 추출한 디렉터리가 Pose 탭의 "Frames directory"에 자동 입력), 보통은 순서대로 진행하면서 필요한 값만 채우면 됩니다. 모든 경로 입력칸 옆의 `Browse...` 버튼은 파일/디렉터리 선택 대화상자를 엽니다.

1. **1. Frames**: `Video file`에서 영상 파일 선택 → (선택) 구간을 눈으로 보고 정하고 싶으면 **Load Preview**를 눌러 미리보기/스크러버를 엽니다. `Start frame`/`End frame` 슬라이더를 드래그하면 그 프레임이 위 미리보기에 실시간으로 표시되고(영상 편집 프로그램처럼), 옆의 숫자 입력칸과 양방향으로 동기화됩니다 — 숫자를 직접 입력해도 슬라이더가 따라가고, 슬라이더를 움직여도 숫자가 갱신됩니다. 슬라이더는 `Start <= End`를 자동으로 유지합니다(끌고 있는 쪽이 우선). Load Preview 없이 숫자 입력칸만 써도 되고, 비워두면 영상 전체가 대상입니다(양 끝 포함, 원본 영상 기준 인덱스) → (선택) `Target FPS` 입력 → `Output frames dir` 지정 → **Extract Frames** 클릭. 완료되면 프레임 개수가 표시됩니다.
2. **2. Pose**: `Frames directory`(자동 채워짐), `MMPose config`(.py)와 `MMPose checkpoint`(.pth)를 직접 갖고 있으면 선택하고, 없으면 **Use Default Model (RTMPose-tiny)** 버튼 한 번으로 채울 수 있습니다(체크포인트는 최초 1회만 `~/.cache/animcv/models`에 다운로드되고, 이후엔 즉시 재사용됨) → `Device`는 CUDA 없으면 `cpu` → (선택) 깊이 기반 3D 리타게팅을 쓰려면 `Depth checkpoint` 지정 → `Output pose.json` 지정 → **Run Pose Estimation** 클릭. 시간이 걸리는 작업이라 창이 멈추지 않고 아래 상태 표시줄에 "Running pose estimation..."이 뜹니다.
3. **3. Rig**: `Rig file`(.fbx 등) 선택 → `Output rig_profile.json` 지정 → **Parse Rig** 클릭. 아래쪽 리스트에 파싱된 본 목록이 나타납니다.
4. **4. Mapping** (클릭 기반 매핑, 자세한 내용은 아래 참고): `Rig file`/`Reference frame image`/`Pose JSON`을 확인(자동 채워짐)한 뒤 **Load Frame + Landmarks** 클릭 — 캔버스에 프레임 이미지와 랜드마크 점이 표시됩니다. 왼쪽 본 목록에서 본을 하나 선택하고, 매핑 모드(`Landmark`/`Direction`/`Custom point`)를 고른 뒤 캔버스를 클릭해서 매핑합니다. 필요하면 IK 체인도 추가한 뒤 `Save mapping to:` 경로를 지정하고 **Save Mapping** 클릭.
5. **5. Motion**: `Pose JSON`(자동 채워짐), `Output motion_graph.json` 지정 → **Build Motion Graph** 클릭.
6. **6. Retarget**: `Motion graph JSON`/`Rig file`/`Mapping JSON`(모두 자동 채워짐), `Output animation.json` 지정 → **Retarget** 클릭.
7. **7. Optimize**: `Animation JSON`(자동 채워짐), `Collapse preset`(`none`/`light`/`medium`/`aggressive`/`custom`) 선택(`custom`이면 `Custom threshold`도 입력) → `Output` 지정 → **Optimize** 클릭. 아래 텍스트 창에 본별 압축 전/후 키 개수가 표시됩니다.
8. **8. Export**: `Optimized animation JSON`/`Rig file`(자동 채워짐), `Output .blend` 지정, (선택) `.fbx`도 함께 원하면 `Output .fbx`, Blender를 자동으로 못 찾으면 `Blender executable`에 직접 지정 → **Export to Blender** 클릭.

#### Mapping 탭 자세히

**Mapping 탭이 CLI 대비 실질적으로 달라지는 부분**: `create-mapping`의 텍스트 프롬프트(`landmark <name>`, `direction <a> <b>` 타이핑) 대신, 기준 프레임 이미지 위에 검출된 랜드마크를 클릭 가능한 점으로 표시하고 실제로 클릭해서 매핑합니다 — `Architecture_v2.md` 섹션 6.2가 원래 설계했고 `ui/mapping_ui.py`의 docstring이 "future work"로 미뤄뒀던 바로 그 워크플로입니다. `custom_point`만은 여전히 텍스트 입력입니다 — 이 모드는 파이프라인 자체에 아직 실제 추적 데이터가 없어서(3.4절 참고) 캔버스에 클릭할 대상 자체가 없기 때문에, 프론트엔드를 뭘 쓰든 마찬가지입니다.

포즈 추정/Blender 내보내기처럼 시간이 걸리는 단계는 백그라운드 스레드에서 실행해 창이 멈추지 않습니다. 스레드에서 메인 스레드로 결과를 넘길 때 `root.after()`를 워커 스레드에서 직접 호출하는 방식은 macOS의 Tk 백엔드에서 안전하지 않아 실제로 앱이 멈추는 걸 확인했고, 대신 순수 `queue.Queue`로 결과만 전달하고 메인 스레드가 주기적으로 큐를 확인하는 방식으로 구현되어 있습니다.

**실제 macOS에서 검증**: 합성 영상 → `extract-frames` → 실제 RTMPose-tiny 체크포인트로 `estimate-pose` → 실제 assimp로 `parse-rig` → Mapping 탭에서 실제 좌표를 클릭해 `forearm.L`/`upper_arm.L`(direction)·`hips`(landmark) 매핑 → `build-motion` → `retarget` → `optimize` → 실제 Blender로 `export-blender`까지 GUI를 통해 프로그래밍적으로 클릭을 재현해 끝까지 실행했고, 결과 `.blend`의 fcurve/키프레임 개수가 동일한 입력으로 CLI를 돌렸을 때와 정확히 일치하는 것까지 확인했습니다. 이 과정에서 실제 버그 1개를 발견/수정함: 매핑을 하나 저장할 때마다 본 목록(Treeview)을 지우고 다시 그리는데, 이게 현재 선택을 지워버려서(그리고 지연된 `<<TreeviewSelect>>` 이벤트가 다음 이벤트 루프 turn에 선택을 `None`으로 되돌려서) 매핑 자체는 정확히 저장되지만 매핑 직후 UI에서 본 선택이 풀려버리는 문제였습니다 — 지우기 전 선택을 저장했다가 다시 그린 뒤 복원하는 방식으로 고쳤습니다.

## 4. 빠른 동작 확인 (합성 데이터로 전체 파이프라인 돌려보기)

실제 영상/리그 없이 프레임워크가 제대로 동작하는지만 빠르게 확인하고 싶다면, 테스트 스위트를 돌려보는 게 가장 빠릅니다:

```bash
pytest
```

220개 테스트가 전부 통과해야 정상입니다. 이 중 `tests/test_blender_executor.py`, `tests/test_apply_motion_script.py`는 가짜(fake) `bpy` 모듈로 Blender 없이도 동작 검증을 하고, `tests/test_rig_parser.py`, `tests/test_mmpose_adapter.py`, `tests/test_depth.py`는 가짜 노드/결과 객체로 assimp/mmpose/depth_anything_v2 없이도 핵심 로직을 검증합니다. `tests/test_ik_solver.py`의 IK 삼각형 계산과 `tests/test_axis_utils.py`의 rest-pose 보정은 직접 손으로 계산한 기하학적 예제로 검증되어 있습니다.

**실제 Blender까지 포함한 검증**을 하고 싶다면 Blender가 설치되어 있어야 하며, `export-blender`를 실행한 뒤 생성된 `.blend` 파일을 Blender에서 직접 열어 그래프 에디터에 키프레임이 보이는지 확인하면 됩니다.

## 5. 알려진 제약사항

- 다중 캐릭터 미지원, 물리 시뮬레이션 없음, 완전 자동 3D 모션캡처 아님 — 이건 의도된 설계입니다 (`Architecture_v2.md` 섹션 1.3 "Excluded From v2" 참고).
- SAM2, DWPose는 여전히 이번 버전(v2) 범위에서 제외되어 있습니다. **Depth Anything V2만 예외적으로 사용자 명시적 요청에 따라 통합했습니다** (`Architecture_v2.md` 문서 자체는 이를 제외하라고 되어 있지만, 리타게팅 품질 개선을 위해 의도적으로 문서 범위를 벗어난 것 — `result/result_mil8.txt` 참고).
- 리그의 임의 스켈레톤(사람이 아닌 크리처, 소품 등)을 지원하지만, 자동으로 어떤 본이 "진짜 관절"인지 판단하지는 않습니다 — 사용자가 `create-mapping`으로 직접 지정해야 합니다.
- 깊이 정보가 없으면 회전 리타게팅은 여전히 2D 이미지 평면 각도 변화만 사용합니다. 깊이가 있어도 카메라 캘리브레이션이나 물리 기반 재구성은 없으므로 완벽한 3D 재구성은 아닙니다.
- IK는 2-bone(코사인 법칙) 체인만 지원하며, 3개 이상의 중간 본이 있는 체인은 지원하지 않습니다.

## 6. 참고 문서

- `Architecture_v2.md` — 전체 설계 문서 (원본 스펙)
- `README.md` — 마일스톤별 구현 상태와 기술적 디테일 (영어, 개발자용)
- `result/result_mil1.txt` ~ `result_mil8.txt` — 마일스톤별 실제 구현/검증 내역 (로컬 전용, git 추적 안 함)
- `result/result_occurred_errors.txt` — 실제로 발견되고 고쳐진 버그 목록 (로컬 전용, git 추적 안 함)
