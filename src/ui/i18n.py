"""Minimal in-app translation for the Tkinter GUI (ui/gui_app.py).

No dependency, no .po/.mo tooling -- just a nested dict keyed by a stable
string id, with an English + Korean entry each. ``Translator`` resolves
the current language and falls back to English (then to the key itself)
so a missing translation degrades visibly-but-safely rather than
crashing. Templated strings (results, prompts, dialogs) keep their
``{placeholders}`` in both languages and are ``.format()``-ed by the
caller.
"""

from __future__ import annotations

LANGUAGES = ("en", "ko")
LANGUAGE_LABELS = {"en": "English", "ko": "한국어"}

_STRINGS: dict[str, dict[str, str]] = {
    # window / tabs
    "app.title": {"en": "AnimCV Motion Tool", "ko": "AnimCV 모션 툴"},
    "tab.frames": {"en": "1. Frames", "ko": "1. 프레임"},
    "tab.pose": {"en": "2. Pose", "ko": "2. 포즈"},
    "tab.rig": {"en": "3. Rig", "ko": "3. 리그"},
    "tab.mapping": {"en": "4. Mapping", "ko": "4. 매핑"},
    "tab.motion": {"en": "5. Motion", "ko": "5. 모션"},
    "tab.retarget": {"en": "6. Retarget", "ko": "6. 리타깃"},
    "tab.optimize": {"en": "7. Optimize", "ko": "7. 최적화"},
    "tab.export": {"en": "8. Export", "ko": "8. 내보내기"},
    "tab.settings": {"en": "Settings", "ko": "설정"},
    # generic
    "btn.browse": {"en": "Browse...", "ko": "찾아보기..."},
    "status.ready": {"en": "Ready", "ko": "준비됨"},
    "dlg.error.title": {"en": "Error", "ko": "오류"},
    "dlg.missing_input.title": {"en": "Missing input", "ko": "입력 누락"},
    # settings tab
    "settings.language": {"en": "Language:", "ko": "언어:"},
    "settings.language_hint": {
        "en": "Applies immediately across all tabs.",
        "ko": "모든 탭에 즉시 적용됩니다.",
    },
    # ---- Frames tab ----
    "frames.video": {"en": "Video file:", "ko": "영상 파일:"},
    "frames.out": {"en": "Output frames dir:", "ko": "출력 프레임 폴더:"},
    "frames.target_fps": {"en": "Target FPS (optional):", "ko": "목표 FPS (선택):"},
    "frames.start": {"en": "Start frame:", "ko": "시작 프레임:"},
    "frames.end": {"en": "End frame:", "ko": "끝 프레임:"},
    "frames.range_hint": {
        "en": "(blank = whole video; both are source-video frame indices, end inclusive)",
        "ko": "(비우면 영상 전체; 둘 다 원본 영상 프레임 인덱스, 끝 포함)",
    },
    "frames.preview_hint": {
        "en": "Load a video to scrub it and set the reference range visually (optional).",
        "ko": "영상을 불러오면 스크러버로 구간을 눈으로 보고 지정할 수 있습니다 (선택).",
    },
    "btn.load_preview": {"en": "Load Preview", "ko": "미리보기 열기"},
    "btn.extract_frames": {"en": "Extract Frames", "ko": "프레임 추출"},
    "frames.progress.preparing": {"en": "preparing…", "ko": "준비 중…"},
    "frames.preview_status": {
        "en": "✓ {count} frames @ {fps:.2f} fps, {w}x{h}",
        "ko": "✓ {count}개 프레임 @ {fps:.2f} fps, {w}x{h}",
    },
    "frames.preview_info": {
        "en": "Frame {index} / {last}   (t = {t:.2f}s)",
        "ko": "프레임 {index} / {last}   (t = {t:.2f}초)",
    },
    "frames.preview_none": {"en": "(no frame at index {index})", "ko": "(인덱스 {index}에 프레임 없음)"},
    "frames.result": {
        "en": "✓ Extracted {count} frames -> {out}",
        "ko": "✓ {count}개 프레임 추출 완료 -> {out}",
    },
    "busy.extracting": {"en": "Extracting frames...", "ko": "프레임 추출 중..."},
    "dlg.frames_missing": {
        "en": "Video file and output directory are required.",
        "ko": "영상 파일과 출력 폴더가 필요합니다.",
    },
    "dlg.choose_video": {"en": "Choose a video file first.", "ko": "먼저 영상 파일을 선택하세요."},
    "dlg.open_video_fail.title": {"en": "Could not open video", "ko": "영상을 열 수 없음"},
    "dlg.output_not_empty.title": {"en": "Output folder not empty", "ko": "출력 폴더가 비어있지 않음"},
    "dlg.output_not_empty.msg": {
        "en": "'{out}' already contains files. Extracting here may overwrite existing "
        "frames and mix old ones in. Continue?",
        "ko": "'{out}'에 이미 파일이 있습니다. 여기에 추출하면 기존 프레임을 덮어쓰거나 "
        "이전 프레임이 섞일 수 있습니다. 계속할까요?",
    },
    # ---- Pose tab ----
    "pose.frames_dir": {"en": "Frames directory:", "ko": "프레임 폴더:"},
    "pose.config": {"en": "MMPose config:", "ko": "MMPose 설정 파일:"},
    "pose.checkpoint": {"en": "MMPose checkpoint:", "ko": "MMPose 체크포인트:"},
    "btn.use_default_model": {
        "en": "Use Default Model (RTMPose-tiny)",
        "ko": "기본 모델 사용 (RTMPose-tiny)",
    },
    "pose.device": {"en": "Device:", "ko": "장치:"},
    "pose.visibility": {"en": "Visibility threshold:", "ko": "가시성 임계값:"},
    "pose.depth_checkpoint": {"en": "Depth checkpoint (optional):", "ko": "깊이 체크포인트 (선택):"},
    "pose.depth_encoder": {"en": "Depth encoder:", "ko": "깊이 인코더:"},
    "pose.depth_device": {"en": "Depth device:", "ko": "깊이 장치:"},
    "pose.out": {"en": "Output pose.json:", "ko": "출력 pose.json:"},
    "btn.run_pose": {"en": "Run Pose Estimation", "ko": "포즈 추정 실행"},
    "pose.default_ready": {"en": "✓ RTMPose-tiny ready", "ko": "✓ RTMPose-tiny 준비됨"},
    "pose.result": {
        "en": "✓ Estimated pose for {count} frames -> {out}",
        "ko": "✓ {count}개 프레임 포즈 추정 완료 -> {out}",
    },
    "busy.default_model": {
        "en": "Fetching default model (downloads ~13MB on first use)...",
        "ko": "기본 모델 받는 중 (최초 1회 ~13MB 다운로드)...",
    },
    "busy.pose": {
        "en": "Running pose estimation (this can take a while)...",
        "ko": "포즈 추정 실행 중 (시간이 걸릴 수 있습니다)...",
    },
    "dlg.pose_missing": {
        "en": "Frames dir, MMPose config, checkpoint, and output path are required.",
        "ko": "프레임 폴더, MMPose 설정, 체크포인트, 출력 경로가 필요합니다.",
    },
    # ---- Rig tab ----
    "rig.file": {"en": "Rig file (.fbx etc.):", "ko": "리그 파일 (.fbx 등):"},
    "rig.out": {"en": "Output rig_profile.json:", "ko": "출력 rig_profile.json:"},
    "btn.parse_rig": {"en": "Parse Rig", "ko": "리그 파싱"},
    "rig.parsed_bones": {"en": "Parsed bones:", "ko": "파싱된 본:"},
    "rig.result": {
        "en": "✓ Parsed {count} bones (root={root}) -> {out}",
        "ko": "✓ 본 {count}개 파싱 (root={root}) -> {out}",
    },
    "busy.parsing": {"en": "Parsing rig...", "ko": "리그 파싱 중..."},
    "dlg.rig_missing": {
        "en": "Rig file and output path are required.",
        "ko": "리그 파일과 출력 경로가 필요합니다.",
    },
    # ---- Mapping tab ----
    "mapping.rig_file": {"en": "Rig file:", "ko": "리그 파일:"},
    "mapping.ref_frame": {"en": "Reference frame image:", "ko": "기준 프레임 이미지:"},
    "mapping.pose_json": {"en": "Pose JSON (for landmark dots):", "ko": "포즈 JSON (랜드마크 점 표시용):"},
    "btn.load_frame_landmarks": {"en": "Load Frame + Landmarks", "ko": "프레임 + 랜드마크 불러오기"},
    "mapping.bones_hint": {"en": "Rig bones (click to select):", "ko": "리그 본 (클릭해서 선택):"},
    "mapping.col_bone": {"en": "Bone", "ko": "본"},
    "mapping.col_mapping": {"en": "Assigned mapping", "ko": "지정된 매핑"},
    "mapping.mode_landmark": {"en": "Landmark (click 1 point)", "ko": "랜드마크 (점 1개 클릭)"},
    "mapping.mode_direction": {"en": "Direction (click 2 points)", "ko": "방향 (점 2개 클릭)"},
    "mapping.mode_custom": {"en": "Custom point (type id)", "ko": "커스텀 포인트 (ID 입력)"},
    "mapping.custom_id": {"en": "Custom point id:", "ko": "커스텀 포인트 ID:"},
    "btn.assign": {"en": "Assign", "ko": "지정"},
    "btn.clear_mapping": {
        "en": "Clear mapping for selected bone",
        "ko": "선택한 본의 매핑 지우기",
    },
    "mapping.ik_frame": {
        "en": "IK chains (optional, 2-bone: root -> mid -> end)",
        "ko": "IK 체인 (선택, 2-본: root -> mid -> end)",
    },
    "mapping.ik.root_bone": {"en": "root bone", "ko": "root 본"},
    "mapping.ik.mid_bone": {"en": "mid bone", "ko": "mid 본"},
    "mapping.ik.end_bone": {"en": "end bone", "ko": "end 본"},
    "mapping.ik.root_src": {"en": "root src", "ko": "root 소스"},
    "mapping.ik.mid_src": {"en": "mid src", "ko": "mid 소스"},
    "mapping.ik.end_src": {"en": "end src", "ko": "end 소스"},
    "btn.add_ik": {"en": "Add IK Chain", "ko": "IK 체인 추가"},
    "btn.remove_ik": {"en": "Remove selected chain", "ko": "선택한 체인 제거"},
    "mapping.save_to": {"en": "Save mapping to:", "ko": "매핑 저장 위치:"},
    "btn.save_mapping": {"en": "Save Mapping", "ko": "매핑 저장"},
    "mapping.status.begin": {
        "en": "Parse a rig and select a bone to begin.",
        "ko": "리그를 파싱하고 본을 선택하면 시작합니다.",
    },
    "mapping.status.select_first": {
        "en": "Select a bone from the list first.",
        "ko": "먼저 목록에서 본을 선택하세요.",
    },
    "mapping.status.click_first": {
        "en": "'{bone}': click the FIRST landmark point.",
        "ko": "'{bone}': 첫 번째 랜드마크 점을 클릭하세요.",
    },
    "mapping.status.click_one": {
        "en": "'{bone}': click a landmark point.",
        "ko": "'{bone}': 랜드마크 점을 클릭하세요.",
    },
    "mapping.status.type_id": {
        "en": "'{bone}': type a point id and click Assign.",
        "ko": "'{bone}': 포인트 ID를 입력하고 Assign을 누르세요.",
    },
    "mapping.status.no_landmark": {
        "en": "No landmark near that click -- try closer to a dot.",
        "ko": "클릭 위치 근처에 랜드마크가 없습니다 -- 점에 더 가까이 클릭하세요.",
    },
    "mapping.status.first_point": {
        "en": "First point: {name}. Now click the second point.",
        "ko": "첫 번째 점: {name}. 이제 두 번째 점을 클릭하세요.",
    },
    "mapping.status.cleared": {
        "en": "Cleared mapping for '{bone}'.",
        "ko": "'{bone}'의 매핑을 지웠습니다.",
    },
    "dlg.no_bone.title": {"en": "No bone selected", "ko": "선택된 본 없음"},
    "dlg.type_id": {"en": "Type a point id first.", "ko": "먼저 포인트 ID를 입력하세요."},
    "dlg.ik_missing": {
        "en": "All six IK chain fields are required.",
        "ko": "IK 체인 6개 필드가 모두 필요합니다.",
    },
    "dlg.no_rig.title": {"en": "No rig loaded", "ko": "리그 미로딩"},
    "dlg.no_rig.msg": {"en": "Parse a rig first.", "ko": "먼저 리그를 파싱하세요."},
    "dlg.choose_save_mapping": {
        "en": "Choose where to save the mapping profile.",
        "ko": "매핑 프로필 저장 위치를 선택하세요.",
    },
    "dlg.choose_ref_frame": {
        "en": "Choose a reference frame image first.",
        "ko": "먼저 기준 프레임 이미지를 선택하세요.",
    },
    "dlg.load_image_fail.title": {"en": "Could not load image", "ko": "이미지를 불러올 수 없음"},
    "dlg.load_image_fail.msg": {
        "en": "Only PNG frames are supported: {err}",
        "ko": "PNG 프레임만 지원합니다: {err}",
    },
    # ---- Motion tab ----
    "motion.pose_json": {"en": "Pose JSON:", "ko": "포즈 JSON:"},
    "motion.out": {"en": "Output motion_graph.json:", "ko": "출력 motion_graph.json:"},
    "btn.build_motion": {"en": "Build Motion Graph", "ko": "모션 그래프 생성"},
    "motion.result": {
        "en": "✓ Built motion graph with {count} frames -> {out}",
        "ko": "✓ {count}개 프레임 모션 그래프 생성 -> {out}",
    },
    "busy.building_motion": {"en": "Building motion graph...", "ko": "모션 그래프 생성 중..."},
    "dlg.motion_missing": {
        "en": "Pose JSON and output path are required.",
        "ko": "포즈 JSON과 출력 경로가 필요합니다.",
    },
    # ---- Retarget tab ----
    "retarget.motion_json": {"en": "Motion graph JSON:", "ko": "모션 그래프 JSON:"},
    "retarget.rig_file": {"en": "Rig file:", "ko": "리그 파일:"},
    "retarget.mapping_json": {"en": "Mapping JSON:", "ko": "매핑 JSON:"},
    "retarget.out": {"en": "Output animation.json:", "ko": "출력 animation.json:"},
    "btn.retarget": {"en": "Retarget", "ko": "리타깃"},
    "retarget.result": {
        "en": "✓ Retargeted {count} bone tracks -> {out}",
        "ko": "✓ 본 트랙 {count}개 리타깃 -> {out}",
    },
    "busy.retargeting": {"en": "Retargeting...", "ko": "리타깃 중..."},
    "dlg.retarget_missing": {"en": "All four fields are required.", "ko": "4개 필드가 모두 필요합니다."},
    # ---- Optimize tab ----
    "optimize.animation_json": {"en": "Animation JSON:", "ko": "애니메이션 JSON:"},
    "optimize.collapse_preset": {"en": "Collapse preset:", "ko": "압축 프리셋:"},
    "optimize.custom_threshold": {"en": "Custom threshold:", "ko": "커스텀 임계값:"},
    "optimize.out": {
        "en": "Output optimized animation.json:",
        "ko": "출력 optimized animation.json:",
    },
    "btn.optimize": {"en": "Optimize", "ko": "최적화"},
    "busy.optimizing": {"en": "Optimizing...", "ko": "최적화 중..."},
    "dlg.optimize_missing": {
        "en": "Animation JSON and output path are required.",
        "ko": "애니메이션 JSON과 출력 경로가 필요합니다.",
    },
    "dlg.optimize_custom": {
        "en": "'custom' preset needs a threshold value.",
        "ko": "'custom' 프리셋은 임계값이 필요합니다.",
    },
    # ---- Export tab ----
    "export.animation_json": {"en": "Optimized animation JSON:", "ko": "최적화된 애니메이션 JSON:"},
    "export.rig_file": {"en": "Rig file:", "ko": "리그 파일:"},
    "export.out": {"en": "Output .blend:", "ko": "출력 .blend:"},
    "export.fbx_out": {"en": "Output .fbx (optional):", "ko": "출력 .fbx (선택):"},
    "export.blender_exe": {
        "en": "Blender executable (optional override):",
        "ko": "Blender 실행파일 (선택, 직접 지정):",
    },
    "btn.export_blender": {"en": "Export to Blender", "ko": "Blender로 내보내기"},
    "export.result": {"en": "✓ Exported -> {out}", "ko": "✓ 내보내기 완료 -> {out}"},
    "busy.exporting": {
        "en": "Exporting to Blender (this can take a moment)...",
        "ko": "Blender로 내보내는 중 (잠시 걸릴 수 있습니다)...",
    },
    "dlg.export_missing": {
        "en": "Animation JSON, rig file, and output path are required.",
        "ko": "애니메이션 JSON, 리그 파일, 출력 경로가 필요합니다.",
    },
}


class Translator:
    def __init__(self, language: str = "en"):
        self.language = language if language in LANGUAGES else "en"

    def set_language(self, language: str) -> None:
        if language in LANGUAGES:
            self.language = language

    def __call__(self, key: str, **fmt) -> str:
        entry = _STRINGS.get(key)
        if entry is None:
            text = key
        else:
            text = entry.get(self.language) or entry.get("en") or key
        return text.format(**fmt) if fmt else text
