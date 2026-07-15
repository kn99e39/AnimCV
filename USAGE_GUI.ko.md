# AnimCV Motion Tool — GUI 사용법

영상에서 뼈대(리그) 애니메이션을 뽑아내는 모션캡처/리타게팅 도구입니다.
`motion-tool-gui.exe`(Windows) 또는 `motion-tool-gui.app`(macOS)를 실행하면
됩니다 — 별도 설치 없이, 필요한 건 전부 이 폴더 안에 들어있습니다.

창 하나에 탭 8개로 구성되어 있고, 왼쪽부터 순서대로 진행합니다 — 각 탭의
결과물이 다음 탭의 입력이 됩니다.

## 1. Frames (프레임 추출)
입력 영상과 출력 폴더를 지정하고, 필요하면 목표 FPS도 설정한 뒤
**Extract Frames**를 누릅니다. 이후 단계에서 쓰는 번호 매겨진 PNG 프레임
폴더가 만들어집니다.

영상 전체가 아니라 특정 구간만 레퍼런스로 쓰고 싶으면, `start`/`end`
프레임 번호를 직접 입력해도 되고, 영상 편집 프로그램처럼 **Load Preview**를
누른 뒤 **Start frame** / **End frame** 슬라이더를 드래그해도 됩니다 —
움직이는 쪽 슬라이더의 프레임이 위 미리보기에 실시간으로 표시되고,
슬라이더와 숫자 입력칸은 서로 동기화됩니다(둘 중 아무거나 편집 가능).
범위는 양 끝 포함이고 원본 영상 기준 인덱스이며, 비워두면 영상 전체가
대상입니다.

## 2. Pose (자세 추정)
1단계에서 만든 프레임 폴더와 MMPose 모델 설정 파일 + 체크포인트(`.py` + `.pth`)를
지정합니다. 이 앱에는 MMPose 자체가 포함되어 있지만, 검증된 기본 모델을 쓰거나
호환되는 다른 모델을 직접 선택할 수 있습니다.

### 가장 빠른 방법: 검증된 기본 모델

1. **Use Default Model (RTMPose-tiny)** 버튼을 누릅니다. MMPose에 포함된
   서로 맞는 config와 checkpoint 이름이 자동으로 채워집니다.
2. 호환되는 CUDA 환경이 없다면 Device는 `cpu`로 둡니다.
3. **Run Pose Estimation**을 누릅니다. 첫 실행 때만 약 13MB 체크포인트를
   OpenMMLab에서 `~/.cache/animcv/models`에 내려받고, 이후에는 그 파일을
   자동으로 재사용합니다.

### 다른 MMPose 모델을 직접 사용할 때

1. 공식 [MMPose model zoo](https://github.com/open-mmlab/mmpose)에서
   **top-down 2D body keypoint** 모델 중 **COCO 17-keypoint** 스키마로 학습된
   모델을 고릅니다. AnimCV는 이 스키마를 자체 신체 랜드마크로 변환하고,
   프레임마다 신뢰도가 가장 높은 한 사람만 처리합니다.
2. 하나의 모델 항목에서 함께 명시된 config와 checkpoint를 한 쌍으로 받습니다.
   서로 다른 행, 모델 크기, 입력 해상도, 데이터셋의 파일을 섞으면 안 됩니다.
3. 체크포인트 파일(`.pth`)은 예를 들어 `AnimCV-models/`처럼 사용자가 관리하는
   고정 폴더에 보관합니다. config는 공식 디렉터리 구조와 그것이 불러오는
   `_base_` 파일까지 함께 유지해야 합니다. config 파일 하나만 복사하면 로드에
   실패할 수 있습니다. 설치된 MMPose에 이미 포함된 config를 지정해도 됩니다.
4. Pose 탭에서 해당 config `.py`와 checkpoint `.pth`를 각각 찾아 지정합니다.
   Device는 `cpu`를 사용하거나, 설치된 PyTorch/CUDA 조합이 지원될 때만 `cuda`를
   사용합니다.
5. 전체 영상을 처리하기 전에 짧은 프레임 구간으로 먼저 실행하여 감지된
   랜드마크가 자연스러운지 확인합니다.

로드에 실패하면 먼저 config가 `_base_` 파일을 여전히 찾는지, checkpoint가
정확히 같은 model-zoo 항목에서 받은 것인지를 확인하세요. 임의의 pose/hand/face,
bottom-up 또는 COCO가 아닌 체크포인트는 이 파이프라인과 서로 바꿔 쓸 수 없습니다.

Depth Anything V2 체크포인트(`.pth`)를 추가로 지정하면 나중 단계에서 2D 근사
대신 실제 3D 정보를 활용한 리타게팅이 됩니다 — device는 `auto`로 두세요.
**Run Pose Estimation**을 누르면 `pose.json`이 만들어집니다.

## 3. Rig (리그 파싱)
캐릭터 리그 파일(`.fbx`, 또는 Assimp가 읽을 수 있는 형식)을 지정하고
**Parse Rig**를 누릅니다. 찾은 본 목록이 나오는데, 다음 Mapping 탭에서
필요합니다. 이 기능은 이 컴퓨터에 네이티브 `assimp` 라이브러리가 설치되어
있어야 합니다(번들에 포함 안 됨 — 아래 Setup 참고). 없으면 이 탭은 죽지
않고 에러 메시지만 보여줍니다.

## 4. Mapping (본 매핑)
클릭 기반으로 본을 매핑합니다: 프레임 + 감지된 랜드마크를 불러온 뒤, 왼쪽에서
리그 본을 클릭하고 랜드마크(회전 방향 매핑이면 랜드마크 두 개)를 클릭해서
지정합니다. 어깨→팔꿈치→손목처럼 2본 팔다리는 오른쪽에서 IK 체인(root→mid→end)을
추가로 설정할 수 있습니다. **Save Mapping**을 누르면 Retarget 단계에서 쓰는
매핑 프로필이 저장됩니다.

## 5. Motion (모션 그래프 생성)
2단계의 `pose.json`을 지정하고 **Build Motion Graph**를 누릅니다.
프레임별 랜드마크 데이터를 트랙 기반 모션 그래프로 바꿉니다.

## 6. Retarget (리타게팅)
5단계 모션 그래프 + 3단계 리그 파일 + 4단계 매핑 프로필을 합쳐서
애니메이션 클립을 만듭니다.

## 7. Optimize (키프레임 최적화)
프레임마다 촘촘한 애니메이션을 성긴 키프레임으로 압축합니다(RDP 방식 단순화).
프리셋(`light`/`medium`/`aggressive`) 중 고르거나, `custom`으로 직접 오차
임계값을 지정할 수 있습니다. `none`을 선택하면 이 단계를 건너뜁니다.

## 8. Export (Blender로 내보내기)
최종 애니메이션을 리그 파일에 적용해서 `.blend`로 저장합니다(선택적으로
`.fbx`도 함께). 이 컴퓨터에 실제 Blender가 설치되어 있어야 합니다 —
`PATH`나 OS별 기본 설치 위치에서 자동으로 찾거나, 직접 경로를 지정할 수
있습니다. Blender 자체는 이 프로그램에 포함되어 있지 않습니다.

## 이 프로그램이 대신 설치해주지 않는 것
- **Blender** (8단계 Export용): blender.org에서 설치하세요.
- **assimp** (3단계 Rig, 그리고 6단계 Retarget에서도 필요): 파이썬
  바인딩이 아니라 네이티브 라이브러리가 필요합니다 — Windows는
  `assimp.dll`을 `PATH`에 등록, macOS는 `brew install assimp`.

## 참고
이 GUI는 AnimCV 커맨드라인 도구(`motion-tool`)와 완전히 같은 파이프라인을
그대로 감싼 것입니다 — 각 탭은 CLI 명령어가 호출하는 것과 동일한 내부
함수를 호출합니다. 더 자세한 구조나 CLI 전체 명령어 목록이 필요하면 소스
저장소의 `README.md` / `README_EXEC.md`(한국어)를 참고하세요.
