import cv2
import numpy as np
import pytest

from app.cli import main


def _write_synthetic_video(path, fps=12.0, num_frames=6, size=(32, 24)):
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(num_frames):
        writer.write(np.full((height, width, 3), fill_value=i, dtype=np.uint8))
    writer.release()


def test_cli_extract_frames_writes_images_and_metadata(tmp_path):
    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path)
    out_dir = tmp_path / "frames"

    exit_code = main(["extract-frames", "--video", str(video_path), "--out", str(out_dir)])

    assert exit_code == 0
    assert len(list(out_dir.glob("*.png"))) == 6
    assert (out_dir / "metadata.json").exists()


def test_cli_build_motion_from_pose_json(tmp_path):
    from common.serialization import write_json
    from pose.pose_types import PoseFrame, PoseLandmark, PoseSequence

    landmark = PoseLandmark(name="left_wrist", x=1.0, y=2.0, confidence=0.9, visible=True)
    poses = PoseSequence(
        frames=[PoseFrame(frame_index=0, timestamp=0.0, landmarks={"left_wrist": landmark})],
        source_fps=24.0,
    )
    pose_path = tmp_path / "pose.json"
    write_json(pose_path, poses.to_dict())
    out_path = tmp_path / "motion_graph.json"

    exit_code = main(["build-motion", "--pose", str(pose_path), "--out", str(out_path)])

    assert exit_code == 0
    assert out_path.exists()


def test_cli_parse_rig_reports_import_error_without_assimp(tmp_path, capsys, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyassimp":
            raise ModuleNotFoundError("No module named 'pyassimp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    exit_code = main(
        ["parse-rig", "--rig", "character.fbx", "--out", str(tmp_path / "rig_profile.json")]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "assimp" in captured.err.lower()


def test_cli_create_mapping_reports_import_error_without_assimp(tmp_path, capsys, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyassimp":
            raise ModuleNotFoundError("No module named 'pyassimp'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    exit_code = main(
        [
            "create-mapping",
            "--rig",
            "character.fbx",
            "--frame",
            "cache/frames/00003.png",
            "--out",
            str(tmp_path / "mapping.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "assimp" in captured.err.lower()


def test_cli_create_mapping_runs_interactively_against_a_parsed_rig(tmp_path, monkeypatch):
    import io

    from rig.rig_parser import RigParser
    from rig.rig_profile import BoneInfo, RigProfile

    fake_profile = RigProfile(
        rig_id="character_01",
        source_path="character.fbx",
        bones={
            "head": BoneInfo(name="head", parent="neck"),
            "hips": BoneInfo(name="hips", parent=None),
        },
        root_bone="hips",
    )
    monkeypatch.setattr(RigParser, "load", lambda self, path: fake_profile)
    monkeypatch.setattr("sys.stdin", io.StringIO("landmark head\nlandmark pelvis\n"))

    out_path = tmp_path / "mapping.json"
    exit_code = main(
        [
            "create-mapping",
            "--rig",
            "character.fbx",
            "--frame",
            "cache/frames/00003.png",
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    from rig.bone_mapping import load_bone_mapping_profile

    saved = load_bone_mapping_profile(out_path)
    assert saved.rig_id == "character_01"
    assert saved.created_from_frame == 3
    assert {entry.target_bone for entry in saved.entries} == {"head", "hips"}


def test_cli_retarget_runs_against_a_parsed_rig(tmp_path, monkeypatch):
    from common.serialization import write_json
    from motion.motion_graph import MotionFrame, MotionGraph, MotionPoint
    from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile, save_bone_mapping_profile
    from rig.rig_parser import RigParser
    from rig.rig_profile import BoneInfo, RigProfile

    def _point(name, x, y, frame_index):
        return MotionPoint(
            semantic_name=name,
            frame_index=frame_index,
            position_2d=(x, y),
            position_3d=None,
            confidence=0.9,
            visible=True,
        )

    motion_graph = MotionGraph(
        frames=[
            MotionFrame(
                frame_index=0,
                timestamp=0.0,
                points={
                    "left_shoulder": _point("left_shoulder", 0.0, 0.0, 0),
                    "left_elbow": _point("left_elbow", 1.0, 0.0, 0),
                },
            ),
            MotionFrame(
                frame_index=1,
                timestamp=1 / 24.0,
                points={
                    "left_shoulder": _point("left_shoulder", 0.0, 0.0, 1),
                    "left_elbow": _point("left_elbow", 0.0, 1.0, 1),
                },
            ),
        ],
        tracks={},
        fps=24.0,
    )
    motion_path = tmp_path / "motion_graph.json"
    write_json(motion_path, motion_graph.to_dict())

    mapping_path = tmp_path / "mapping.json"
    save_bone_mapping_profile(
        BoneMappingProfile(
            rig_id="character_01",
            entries=[
                BoneMappingEntry(
                    target_bone="upper_arm.L",
                    source_type="landmark",
                    source_names=["left_shoulder", "left_elbow"],
                    mapping_mode="direction",
                )
            ],
        ),
        mapping_path,
    )

    fake_rig = RigProfile(
        rig_id="character_01",
        source_path="character.fbx",
        bones={"upper_arm.L": BoneInfo(name="upper_arm.L", parent=None)},
    )
    monkeypatch.setattr(RigParser, "load", lambda self, path: fake_rig)

    out_path = tmp_path / "animation.json"
    exit_code = main(
        [
            "retarget",
            "--motion",
            str(motion_path),
            "--rig",
            "character.fbx",
            "--mapping",
            str(mapping_path),
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    from retarget.solver import load_animation_clip

    clip = load_animation_clip(out_path)
    assert "upper_arm.L" in clip.tracks
    assert len(clip.tracks["upper_arm.L"].samples) == 2


def test_cli_optimize_reduces_keyframe_count(tmp_path):
    import math

    from common.serialization import write_json
    from retarget.axis_utils import quaternion_from_axis_angle
    from retarget.solver import AnimationClip, AnimationTrack, BoneTransformSample

    samples = [
        BoneTransformSample(
            frame_index=i,
            bone_name="upper_arm.L",
            location=None,
            rotation=quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(i * 9)),
            scale=None,
            confidence=0.9,
        )
        for i in range(11)
    ]
    clip = AnimationClip(
        name="Generated_Motion",
        fps=24.0,
        tracks={"upper_arm.L": AnimationTrack(bone_name="upper_arm.L", samples=samples)},
        frame_start=0,
        frame_end=10,
    )
    animation_path = tmp_path / "animation.json"
    write_json(animation_path, clip.to_dict())
    out_path = tmp_path / "animation_optimized.json"

    exit_code = main(
        [
            "optimize",
            "--animation",
            str(animation_path),
            "--collapse",
            "medium",
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0
    from retarget.solver import load_animation_clip

    optimized = load_animation_clip(out_path)
    assert len(optimized.tracks["upper_arm.L"].samples) < 11


def test_cli_optimize_custom_preset_without_threshold_reports_error(tmp_path):
    from common.serialization import write_json
    from retarget.solver import AnimationClip

    animation_path = tmp_path / "animation.json"
    write_json(animation_path, AnimationClip(name="Generated_Motion", fps=24.0).to_dict())

    exit_code = main(
        [
            "optimize",
            "--animation",
            str(animation_path),
            "--collapse",
            "custom",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )

    assert exit_code == 1


def test_find_blender_executable_prefers_explicit_override():
    from app.cli import _find_blender_executable

    assert _find_blender_executable("C:/custom/blender.exe") == "C:/custom/blender.exe"


def test_find_blender_executable_uses_env_var(monkeypatch):
    from app.cli import _find_blender_executable

    monkeypatch.setenv("BLENDER_EXECUTABLE", "C:/env/blender.exe")

    assert _find_blender_executable() == "C:/env/blender.exe"


def test_find_blender_executable_raises_when_not_found(monkeypatch, tmp_path):
    from app.cli import _find_blender_executable

    monkeypatch.delenv("BLENDER_EXECUTABLE", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))  # empty dir: no Blender Foundation here

    with pytest.raises(FileNotFoundError):
        _find_blender_executable()


def test_cli_export_blender_invokes_expected_subprocess_command(tmp_path, monkeypatch):
    import subprocess

    captured_command = {}
    out_path = tmp_path / "result.blend"

    def fake_run(command, capture_output, text):
        captured_command["command"] = command
        # A real successful run leaves the output file on disk; the CLI
        # checks for that in addition to returncode (see the next test).
        out_path.write_text("fake blend contents")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr("app.cli.subprocess.run", fake_run)

    exit_code = main(
        [
            "export-blender",
            "--animation",
            "animation.json",
            "--rig",
            "character.fbx",
            "--out",
            str(out_path),
            "--blender-executable",
            "C:/fake/blender.exe",
        ]
    )

    assert exit_code == 0
    command = captured_command["command"]
    assert command[0] == "C:/fake/blender.exe"
    assert "--background" in command
    assert "--python" in command
    assert command[command.index("--rig") + 1] == "character.fbx"
    assert command[command.index("--animation") + 1] == "animation.json"
    assert command[command.index("--out") + 1] == str(out_path)


def test_cli_export_blender_reports_error_when_output_missing_despite_zero_exit(
    tmp_path, monkeypatch
):
    import subprocess

    # Confirmed against a real Blender build: an unhandled exception in
    # the --python script does not make blender.exe itself exit
    # non-zero, so returncode==0 alone can't be trusted. This simulates
    # that: "success" exit code, but no output file was ever written.
    def fake_run(command, capture_output, text):
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="hidden traceback")

    monkeypatch.setattr("app.cli.subprocess.run", fake_run)

    exit_code = main(
        [
            "export-blender",
            "--animation",
            "animation.json",
            "--rig",
            "character.fbx",
            "--out",
            str(tmp_path / "result.blend"),
            "--blender-executable",
            "C:/fake/blender.exe",
        ]
    )

    assert exit_code == 1


def test_cli_export_blender_reports_error_on_nonzero_exit(tmp_path, monkeypatch):
    import subprocess

    def fake_run(command, capture_output, text):
        return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("app.cli.subprocess.run", fake_run)

    exit_code = main(
        [
            "export-blender",
            "--animation",
            "animation.json",
            "--rig",
            "character.fbx",
            "--out",
            str(tmp_path / "result.blend"),
            "--blender-executable",
            "C:/fake/blender.exe",
        ]
    )

    assert exit_code == 1


def test_cli_requires_arguments():
    with pytest.raises(SystemExit):
        main(["extract-frames"])
