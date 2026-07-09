# Architecture v2 — S-Tier Only

## Project Codename
Video-to-Armature Motion System

## Goal

This project converts a reference video, or a sequence of images extracted from a video, into editable animation keyframes applied to a user-provided armature such as an FBX or Blender armature.

The first implementation must prioritize **implementation success**, not maximum motion-capture accuracy. Therefore, this version uses only the S-Tier open-source dependencies and excludes depth estimation, segmentation, diffusion editing, and advanced alternative pose backends.

The system is intended for personal production use. The generated animation should be retouchable inside Blender, which means the system must avoid inserting raw keyframes on every frame unless explicitly requested.

---

# 1. Scope

## 1.1 Inputs

Required inputs:

- Reference video file, or pre-extracted image sequence
- Target rig file
  - `.fbx`
  - `.blend`
  - or another file format that can be imported into Blender
- User-defined bone-to-visual-point mapping

Optional inputs:

- Target FPS
- Start frame / end frame
- Bone mapping profile from a previous session
- Keyframe collapse threshold
- User-locked frames

## 1.2 Outputs

Primary outputs:

- Blender scene with generated animation curves
- Updated `.blend` file
- Optional exported `.fbx` animation

Intermediate outputs:

- Frame sequence cache
- Pose estimation cache
- Motion Graph file
- Rig Profile file
- Bone Mapping Profile file
- Keyframe Importance Report

## 1.3 Excluded From v2

The following systems must not be implemented in this version:

- Depth Anything V2
- Video Depth Anything
- SAM / SAM2
- DWPose
- Diffusion-based image or video editing
- Text-to-motion generation
- Audio analysis
- Multi-character tracking
- Full 3D reconstruction
- Physics-based motion synthesis

These may be added later through extension interfaces, but they must not be required for the MVP.

---

# 2. Design Principle

The project must be built around three internal representations:

1. **Pose Sequence**
2. **Motion Graph**
3. **Rig Profile**

External libraries should only be used for:

- Video and image loading
- Pose estimation
- FBX / armature parsing
- Blender execution

All project-specific logic should be implemented in-house:

- Interactive bone mapping
- Arbitrary skeleton mapping
- Motion Graph construction
- Retarget solving
- Keyframe importance scoring
- Keyframe collapsing
- Blender animation writing policy

---

# 3. S-Tier Dependencies

## 3.1 OpenCV

### Role

OpenCV is responsible for video and image sequence processing.

### Responsibilities

- Load video files
- Decode video into ordered frames
- Load image sequences
- Resize frames for pose estimation
- Convert color spaces
- Save debug overlays
- Optional optical-flow utility for future frame-to-frame point tracking

### Required Wrapper

Create a wrapper module:

```python
class VideoLoader:
    def load_video(self, path: str, target_fps: int | None = None) -> FrameSequence:
        pass

    def load_image_sequence(self, directory: str) -> FrameSequence:
        pass
```

### Output Type

```python
@dataclass
class Frame:
    index: int
    timestamp: float
    image: np.ndarray
    width: int
    height: int

@dataclass
class FrameSequence:
    frames: list[Frame]
    fps: float
    width: int
    height: int
    source_path: str
```

### Implementation Rule

No other module should directly call `cv2.VideoCapture` or raw OpenCV loading code. All video and frame access must go through `VideoLoader`.

---

## 3.2 MMPose

### Role

MMPose is the primary and only pose estimation backend in v2.

### Responsibilities

- Estimate visible body landmarks from each frame
- Optionally estimate hand and face landmarks if the selected MMPose model supports them
- Provide landmark confidence scores
- Produce a normalized pose sequence for Motion Graph construction

### Required Wrapper

Create a backend adapter:

```python
class PoseEstimator:
    def process_frame(self, frame: Frame) -> PoseFrame:
        pass

    def process_sequence(self, frames: FrameSequence) -> PoseSequence:
        pass
```

### Output Type

```python
@dataclass
class PoseLandmark:
    name: str
    x: float
    y: float
    confidence: float
    visible: bool

@dataclass
class PoseFrame:
    frame_index: int
    timestamp: float
    landmarks: dict[str, PoseLandmark]

@dataclass
class PoseSequence:
    frames: list[PoseFrame]
    source_fps: float
    landmark_schema: str
```

### Landmark Schema

The system must not assume a fixed humanoid-only target skeleton. However, the video-side landmarks from MMPose should be stored using semantic names such as:

- pelvis
- spine
- neck
- head
- left_shoulder
- left_elbow
- left_wrist
- right_shoulder
- right_elbow
- right_wrist
- left_hip
- left_knee
- left_ankle
- right_hip
- right_knee
- right_ankle

For arbitrary target rigs, the system maps these semantic landmarks to user-selected bones or control points through the Bone Mapping Profile.

### Implementation Rule

Project code must not depend on MMPose internals. MMPose must be replaceable by modifying only the adapter.

---

## 3.3 FBX SDK / Assimp

### Role

FBX SDK or Assimp is responsible for reading rig structure from external files.

### Responsibilities

- Parse target skeleton hierarchy
- Extract bone names
- Extract parent-child relationships
- Extract bind/rest transforms
- Extract local axes when available
- Identify root bone candidates

### Required Wrapper

Create a rig parser abstraction:

```python
class RigParser:
    def load(self, path: str) -> RigProfile:
        pass
```

### Output Type

```python
@dataclass
class BoneInfo:
    name: str
    parent: str | None
    children: list[str]
    rest_local_matrix: Matrix4
    rest_world_matrix: Matrix4
    local_axis_hint: dict[str, Vec3] | None

@dataclass
class RigProfile:
    rig_id: str
    source_path: str
    bones: dict[str, BoneInfo]
    root_bone: str | None
    scale: float
    metadata: dict
```

### Implementation Rule

All downstream modules must consume `RigProfile`, not raw FBX SDK or Assimp structures.

---

## 3.4 Blender Python API (`bpy`)

### Role

The Blender Python API is the final execution layer.

### Responsibilities

- Import rig file into Blender
- Locate armature object
- Apply generated animation curves
- Insert keyframes
- Configure interpolation
- Export `.blend` or `.fbx`
- Generate debug visualizations if needed

### Required Wrapper

```python
class BlenderExecutor:
    def import_rig(self, path: str) -> BlenderRigHandle:
        pass

    def apply_animation(self, animation: AnimationClip, rig_profile: RigProfile) -> None:
        pass

    def save_blend(self, path: str) -> None:
        pass

    def export_fbx(self, path: str) -> None:
        pass
```

### Implementation Rule

Core project logic must not import `bpy`. Only the Blender execution module may import and call `bpy`.

This allows core motion processing to be tested outside Blender.

---

# 4. Internal Architecture

## 4.1 Top-Level Pipeline

```text
Input Video / Image Sequence
        │
        ▼
OpenCV VideoLoader
        │
        ▼
FrameSequence
        │
        ▼
MMPose PoseEstimator
        │
        ▼
PoseSequence
        │
        ▼
MotionGraphBuilder
        │
        ▼
MotionGraph
        │
        ├───────────────┐
        │               │
        ▼               ▼
RigProfile        BoneMappingProfile
(FBX/Assimp)      (User Assisted Binding)
        │               │
        └───────┬───────┘
                ▼
          RetargetSolver
                │
                ▼
          Raw AnimationClip
                │
                ▼
       KeyframeOptimizer
                │
                ▼
       Optimized AnimationClip
                │
                ▼
        BlenderExecutor
                │
                ▼
        .blend / .fbx Output
```

---

# 5. Motion Graph

## 5.1 Purpose

The Motion Graph is the central intermediate representation. It stores interpreted motion independently from Blender, FBX, or any specific output software.

The Motion Graph should represent:

- Per-frame landmarks
- Estimated motion tracks
- Visibility and confidence
- User mapping anchors
- Importance scores
- User-locked frames
- Candidate keyframes

## 5.2 Data Model

```python
@dataclass
class MotionPoint:
    semantic_name: str
    frame_index: int
    position_2d: Vec2
    position_3d: Vec3 | None
    confidence: float
    visible: bool

@dataclass
class MotionTrack:
    semantic_name: str
    points: list[MotionPoint]

@dataclass
class MotionFrame:
    frame_index: int
    timestamp: float
    points: dict[str, MotionPoint]
    importance: float
    locked: bool

@dataclass
class MotionGraph:
    frames: list[MotionFrame]
    tracks: dict[str, MotionTrack]
    fps: float
    source_metadata: dict
```

## 5.3 Important Constraint

The Motion Graph is not the final animation. It is the project’s internal motion description.

The Retarget Solver converts Motion Graph data into rig-specific bone rotations and animation curves.

---

# 6. Assisted Bone Mapping

## 6.1 Motivation

The project targets arbitrary skeletons, not only humanoid rigs. Therefore, automatic bone mapping is unreliable.

The system must support user-assisted binding.

## 6.2 Required Workflow

1. Load target rig.
2. Load first usable reference frame.
3. Display rig bone list and video frame side by side.
4. User selects a bone.
5. User clicks or selects a corresponding visual landmark / body region in the frame.
6. System records the mapping.
7. User repeats this for important bones.
8. Mapping is saved as a reusable Bone Mapping Profile.

## 6.3 Mapping Types

The system should support three mapping types:

### Direct Landmark Mapping

Example:

```text
upper_arm.L -> left_shoulder / left_elbow vector
forearm.L   -> left_elbow / left_wrist vector
```

### Point Anchor Mapping

Example:

```text
custom_wing_01 -> user clicked point at frame 0
```

### Chain Mapping

Example:

```text
arm_chain.L = [upper_arm.L, forearm.L, hand.L]
source_chain = [left_shoulder, left_elbow, left_wrist]
```

## 6.4 Bone Mapping Profile

```python
@dataclass
class BoneMappingEntry:
    target_bone: str
    source_type: str
    source_names: list[str]
    mapping_mode: str
    weight: float
    axis_hint: str | None
    locked: bool

@dataclass
class BoneMappingProfile:
    rig_id: str
    entries: list[BoneMappingEntry]
    created_from_frame: int
    user_notes: str | None
```

## 6.5 Implementation Rule

The system must never require all bones to be mapped.

Only mapped bones should be animated in the first version. Unmapped bones should preserve rest pose unless affected by child/parent transforms or constraints.

---

# 7. Retarget Solver

## 7.1 Purpose

The Retarget Solver converts Motion Graph movement into target-rig animation curves.

It must support arbitrary bone structures by relying on user-provided Bone Mapping Profile data.

## 7.2 Initial Strategy

The v2 solver should start with a practical FK-oriented approach:

1. For each mapped bone or chain, compute source direction vector from pose landmarks.
2. Compare current source direction against reference source direction.
3. Convert direction delta into target bone local rotation.
4. Apply axis correction from Rig Profile and Bone Mapping Profile.
5. Generate per-frame raw transform samples.
6. Pass samples to Keyframe Optimizer.

## 7.3 Data Model

```python
@dataclass
class BoneTransformSample:
    frame_index: int
    bone_name: str
    location: Vec3 | None
    rotation: Quaternion
    scale: Vec3 | None
    confidence: float

@dataclass
class AnimationTrack:
    bone_name: str
    samples: list[BoneTransformSample]

@dataclass
class AnimationClip:
    name: str
    fps: float
    tracks: dict[str, AnimationTrack]
    frame_start: int
    frame_end: int
```

## 7.4 Important Limitation

Since v2 excludes depth estimation, the first solver may be primarily 2D-direction driven.

The goal is not perfect 3D motion capture. The goal is to generate a usable editable animation draft that can be manually refined.

---

# 8. Keyframe Importance and Collapse

## 8.1 Motivation

If every frame is inserted as a keyframe, the resulting Blender animation becomes difficult to retouch.

Therefore, the system must assign importance scores and collapse low-importance keyframes.

## 8.2 Importance Factors

Each frame or transform sample should receive an importance score based on:

- Rotation change
- Direction change
- Velocity change
- Acceleration change
- Confidence drop or recovery
- Local extrema
- User lock
- Start/end frames

## 8.3 Collapse Strategy

Initial implementation may use an error-based simplification algorithm:

1. Keep first and last keyframes.
2. Keep user-locked keyframes.
3. Estimate interpolation error if an intermediate keyframe is removed.
4. Remove the keyframe if error is below threshold.
5. Repeat until no more removable frames exist.

This can be implemented similarly to Ramer-Douglas-Peucker simplification, but applied to rotation/transform curves.

## 8.4 Data Model

```python
@dataclass
class KeyframeCandidate:
    frame_index: int
    bone_name: str
    transform: BoneTransformSample
    importance: float
    locked: bool
    reason: str

@dataclass
class KeyframeOptimizationReport:
    original_key_count: int
    optimized_key_count: int
    removed_key_count: int
    max_error: float
    threshold: float
```

## 8.5 Acceptance Criteria

The optimizer must be configurable:

- No collapse: insert all frames
- Light collapse
- Medium collapse
- Aggressive collapse
- Custom error threshold

---

# 9. Suggested Repository Structure

```text
project/
├── README.md
├── Architecture.md
├── pyproject.toml
├── src/
│   ├── app/
│   │   ├── cli.py
│   │   └── config.py
│   │
│   ├── io/
│   │   ├── video_loader.py        # OpenCV wrapper
│   │   ├── frame_sequence.py
│   │   └── cache.py
│   │
│   ├── pose/
│   │   ├── mmpose_adapter.py      # MMPose wrapper
│   │   ├── pose_types.py
│   │   └── schemas.py
│   │
│   ├── rig/
│   │   ├── rig_parser.py          # FBX SDK / Assimp wrapper
│   │   ├── rig_profile.py
│   │   └── bone_mapping.py
│   │
│   ├── motion/
│   │   ├── motion_graph.py
│   │   ├── motion_builder.py
│   │   └── motion_io.py
│   │
│   ├── retarget/
│   │   ├── solver.py
│   │   ├── fk_solver.py
│   │   └── axis_utils.py
│   │
│   ├── optimize/
│   │   ├── importance.py
│   │   ├── collapse.py
│   │   └── report.py
│   │
│   ├── blender/
│   │   ├── executor.py            # bpy wrapper
│   │   ├── keyframe_writer.py
│   │   └── export.py
│   │
│   └── ui/
│       ├── mapping_ui.py          # initial simple UI
│       └── debug_viewer.py
│
├── tests/
│   ├── test_motion_graph.py
│   ├── test_bone_mapping.py
│   ├── test_keyframe_collapse.py
│   └── test_retarget_solver.py
│
├── examples/
│   ├── sample_config.yaml
│   └── demo_mapping_profile.json
│
└── outputs/
```

---

# 10. CLI MVP

The first usable version should be executable through CLI before building a polished UI.

Example:

```bash
python -m app.cli \
  --video ./input/reference.mp4 \
  --rig ./input/character.fbx \
  --mapping ./profiles/character_mapping.json \
  --output ./outputs/result.blend \
  --collapse medium
```

Required CLI commands:

```bash
# 1. Extract frames
motion-tool extract-frames --video input.mp4 --out cache/frames

# 2. Run pose estimation
motion-tool estimate-pose --frames cache/frames --out cache/pose.json

# 3. Create or edit mapping profile
motion-tool create-mapping --rig character.fbx --frame cache/frames/0001.png --out profiles/mapping.json

# 4. Build motion graph
motion-tool build-motion --pose cache/pose.json --out cache/motion_graph.json

# 5. Retarget
motion-tool retarget --motion cache/motion_graph.json --rig character.fbx --mapping profiles/mapping.json --out cache/animation.json

# 6. Optimize keyframes
motion-tool optimize --animation cache/animation.json --collapse medium --out cache/animation_optimized.json

# 7. Export to Blender
motion-tool export-blender --animation cache/animation_optimized.json --rig character.fbx --out result.blend
```

---

# 11. Implementation Milestones

## Milestone 1 — Skeleton of the Project

- Create repository structure
- Define all core dataclasses
- Implement JSON serialization for FrameSequence metadata, PoseSequence, MotionGraph, RigProfile, BoneMappingProfile, AnimationClip
- Add basic CLI entry points

Acceptance criteria:

- Project can run without MMPose or Blender installed using mock inputs
- Unit tests pass for serialization and type conversion

## Milestone 2 — Video and Pose Pipeline

- Implement OpenCV VideoLoader
- Implement MMPose adapter
- Convert pose output into PoseSequence
- Build initial MotionGraph

Acceptance criteria:

- Given a video, system produces `pose.json` and `motion_graph.json`
- Debug overlay can be exported as image sequence or video

## Milestone 3 — Rig Profile Parser

- Implement FBX SDK or Assimp parser
- Extract bone hierarchy
- Save RigProfile JSON
- Add root bone candidate detection

Acceptance criteria:

- Given an FBX, system produces `rig_profile.json`
- Bone hierarchy can be printed and inspected

## Milestone 4 — Interactive Bone Mapping

- Implement minimal mapping UI or CLI-assisted mapping
- User can bind target bones to semantic pose landmarks
- Save BoneMappingProfile JSON

Acceptance criteria:

- User can map at least upper arm, forearm, thigh, shin, head, and root-like bones
- Mapping profile can be reused across runs

## Milestone 5 — First Retarget Solver

- Implement FK direction-based retargeting
- Generate raw per-frame AnimationClip
- Support arbitrary mapped bones

Acceptance criteria:

- Given MotionGraph + RigProfile + BoneMappingProfile, system produces `animation.json`
- At least one mapped chain moves according to video pose direction

## Milestone 6 — Keyframe Optimizer

- Implement importance scoring
- Implement collapse threshold presets
- Preserve locked frames
- Generate optimization report

Acceptance criteria:

- Optimized animation has fewer keys than raw animation
- User can disable collapse
- Locked keys are never removed

## Milestone 7 — Blender Export

- Implement BlenderExecutor
- Import rig
- Apply animation curves
- Insert optimized keyframes
- Save `.blend`
- Optional FBX export

Acceptance criteria:

- Result opens in Blender
- Target armature contains generated keyframes
- Animation is editable in Blender Graph Editor / Dope Sheet

---

# 12. Testing Strategy

## 12.1 Unit Tests

Required tests:

- MotionGraph serialization
- RigProfile serialization
- BoneMappingProfile validation
- Quaternion / axis conversion utilities
- Keyframe collapse correctness

## 12.2 Integration Tests

Required tests:

- Video → FrameSequence
- PoseSequence → MotionGraph
- RigProfile + BoneMappingProfile → AnimationClip
- AnimationClip → Optimized AnimationClip

## 12.3 Blender Test

Run Blender in background mode:

```bash
blender --background --python scripts/test_export.py
```

Acceptance criteria:

- No Python error
- Output `.blend` exists
- Armature has animation data

---

# 13. Future Extension Points

Although A-Tier components are excluded from v2, the architecture should leave room for them through optional interfaces.

Future interfaces:

```python
class DepthProvider:
    def estimate(self, frames: FrameSequence) -> DepthSequence:
        pass

class SegmentationProvider:
    def segment(self, frame: Frame, prompt: object) -> Mask:
        pass

class AlternativePoseProvider:
    def process_sequence(self, frames: FrameSequence) -> PoseSequence:
        pass
```

These must not be used in v2.

---

# 14. Coding Agent Instructions

When implementing this project, follow these rules:

1. Do not integrate Depth Anything, SAM, SAM2, DWPose, or diffusion models in v2.
2. Implement all external libraries behind adapter interfaces.
3. Keep core motion logic independent from Blender.
4. Keep MotionGraph, RigProfile, BoneMappingProfile, and AnimationClip serializable.
5. Prioritize a CLI-based MVP before GUI polish.
6. Do not assume the target rig is humanoid.
7. Allow partial bone mapping.
8. Preserve user-locked keyframes during collapse.
9. Make every pipeline stage cacheable to JSON or another inspectable format.
10. Favor debuggability over performance in the first implementation.

---

# 15. MVP Definition

The MVP is complete when the following is possible:

1. User provides a video and an FBX rig.
2. System extracts frames using OpenCV.
3. System estimates pose using MMPose.
4. User creates a Bone Mapping Profile.
5. System builds a Motion Graph.
6. System retargets mapped motion to the target rig.
7. System collapses low-importance keyframes.
8. System writes editable keyframes into Blender.
9. User can open the result and manually retouch the animation.

