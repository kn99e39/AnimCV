"""CLI entry points (Architecture_v2.md section 10).

Every documented command is real as of Milestone 7. ``export-blender``
runs the actual ``blender`` executable as a subprocess (Architecture_v2.md
section 4.10's "Headless Batch Mode") since this project's own venv
Python cannot import ``bpy``.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motion-tool", description="Video-to-armature motion pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract-frames", help="Decode a video into a cached frame sequence")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="Output directory for extracted frame images")
    p.add_argument("--fps", type=float, default=None, help="Optional target fps override")

    p = sub.add_parser("estimate-pose", help="Run pose estimation over a frame sequence")
    p.add_argument("--frames", required=True, help="Directory of extracted frame images")
    p.add_argument("--out", required=True, help="Output pose.json path")
    p.add_argument("--pose-config", required=True, help="MMPose model config path")
    p.add_argument("--pose-checkpoint", required=True, help="MMPose model checkpoint path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--visibility-threshold", type=float, default=0.3)
    p.add_argument(
        "--depth-checkpoint",
        default=None,
        help="Optional Depth Anything V2 checkpoint; when given, samples "
        "relative depth at every landmark so retarget can use real 3D "
        "rotations instead of the 2D-plane approximation",
    )
    p.add_argument("--depth-encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    p.add_argument(
        "--depth-device",
        default="auto",
        help="'auto' (recommended), or an explicit device matching what "
        "depth_anything_v2 will actually use (see pose/depth_estimator.py)",
    )

    p = sub.add_parser("parse-rig", help="Parse a rig file into a RigProfile JSON")
    p.add_argument("--rig", required=True, help="Path to .fbx or any Assimp-readable rig file")
    p.add_argument("--out", required=True, help="Output rig_profile.json path")

    p = sub.add_parser("create-mapping", help="Create or edit a bone mapping profile")
    p.add_argument("--rig", required=True)
    p.add_argument("--frame", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("build-motion", help="Build a Motion Graph from a pose sequence")
    p.add_argument("--pose", required=True, help="Input pose.json path")
    p.add_argument("--out", required=True, help="Output motion_graph.json path")

    p = sub.add_parser("retarget", help="Retarget a Motion Graph onto a target rig")
    p.add_argument("--motion", required=True)
    p.add_argument("--rig", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("optimize", help="Collapse dense animation samples into sparse keyframes")
    p.add_argument("--animation", required=True)
    p.add_argument(
        "--collapse", default="medium", choices=["none", "light", "medium", "aggressive", "custom"]
    )
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out", required=True)

    p = sub.add_parser("export-blender", help="Write optimized animation into Blender and export")
    p.add_argument("--animation", required=True)
    p.add_argument("--rig", required=True)
    p.add_argument("--out", required=True, help="Output .blend path")
    p.add_argument("--fbx-out", default=None, help="Optional output .fbx path")
    p.add_argument(
        "--blender-executable", default=None, help="Path to blender(.exe); overrides autodetection"
    )

    return parser


def _extract_frames(args: argparse.Namespace) -> None:
    import cv2

    from mediaio.video_loader import VideoLoader

    sequence = VideoLoader().load_video(args.video, target_fps=args.fps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in sequence.frames:
        cv2.imwrite(str(out_dir / f"{frame.index:05d}.png"), frame.image)

    from common.serialization import write_json
    from mediaio.frame_sequence import FrameSequenceMetadata

    write_json(out_dir / "metadata.json", FrameSequenceMetadata.from_sequence(sequence).to_dict())
    print(f"[motion-tool] extracted {len(sequence.frames)} frames to {out_dir}")


def _estimate_pose(args: argparse.Namespace) -> None:
    from common.serialization import write_json
    from mediaio.video_loader import VideoLoader
    from pose.mmpose_adapter import MMPoseConfig, PoseEstimator
    from pose.pose_types import PoseSequence

    sequence = VideoLoader().load_image_sequence(args.frames)
    config = MMPoseConfig(
        config_path=args.pose_config,
        checkpoint_path=args.pose_checkpoint,
        device=args.device,
        visibility_threshold=args.visibility_threshold,
    )
    poses = PoseEstimator(config).process_sequence(sequence)

    if args.depth_checkpoint:
        from pose.depth_estimator import DepthEstimator, DepthEstimatorConfig
        from pose.depth_sampling import sample_depth_at_landmarks

        depth_estimator = DepthEstimator(
            DepthEstimatorConfig(
                checkpoint_path=args.depth_checkpoint,
                encoder=args.depth_encoder,
                device=args.depth_device,
            )
        )
        depth_sampled_frames = []
        for frame, pose_frame in zip(sequence.frames, poses.frames):
            depth_map = depth_estimator.infer_frame(frame.image)
            depth_sampled_frames.append(sample_depth_at_landmarks(pose_frame, depth_map))
        poses = PoseSequence(
            frames=depth_sampled_frames,
            source_fps=poses.source_fps,
            landmark_schema=poses.landmark_schema,
        )
        print(f"[motion-tool] sampled depth for {len(depth_sampled_frames)} frames")

    write_json(args.out, poses.to_dict())
    print(f"[motion-tool] estimated pose for {len(poses.frames)} frames -> {args.out}")


def _parse_rig(args: argparse.Namespace) -> None:
    from rig.rig_parser import RigParser
    from rig.rig_profile import save_rig_profile

    profile = RigParser().load(args.rig)
    save_rig_profile(profile, args.out)
    print(
        f"[motion-tool] parsed {len(profile.bones)} bones from {args.rig} -> {args.out} "
        f"(root_bone={profile.root_bone!r})"
    )


def _create_mapping(args: argparse.Namespace) -> None:
    from rig.bone_mapping import save_bone_mapping_profile
    from rig.rig_parser import RigParser
    from ui.mapping_ui import run_interactive_mapping

    profile = RigParser().load(args.rig)
    bone_names = sorted(profile.bones)

    mapping = run_interactive_mapping(
        bone_names=bone_names,
        rig_id=profile.rig_id,
        created_from_frame=_frame_index_from_path(args.frame),
    )
    save_bone_mapping_profile(mapping, args.out)
    print(f"[motion-tool] saved bone mapping with {len(mapping.entries)} entries -> {args.out}")


def _frame_index_from_path(frame_path: str) -> int:
    stem = Path(frame_path).stem
    return int(stem) if stem.isdigit() else 0


def _build_motion(args: argparse.Namespace) -> None:
    from common.serialization import read_json, write_json
    from motion.motion_builder import MotionGraphBuilder
    from pose.pose_types import PoseSequence

    poses = PoseSequence.from_dict(read_json(args.pose))
    graph = MotionGraphBuilder().build(poses, source_metadata={"pose_source": args.pose})

    write_json(args.out, graph.to_dict())
    print(f"[motion-tool] built motion graph with {len(graph.frames)} frames -> {args.out}")


def _retarget(args: argparse.Namespace) -> None:
    from motion.motion_io import load_motion_graph
    from retarget.solver import RetargetSolver, save_animation_clip
    from rig.bone_mapping import load_bone_mapping_profile
    from rig.rig_parser import RigParser

    motion_graph = load_motion_graph(args.motion)
    rig_profile = RigParser().load(args.rig)
    mapping_profile = load_bone_mapping_profile(args.mapping)

    clip = RetargetSolver().solve(motion_graph, rig_profile, mapping_profile)

    save_animation_clip(clip, args.out)
    print(f"[motion-tool] retargeted {len(clip.tracks)} bone tracks -> {args.out}")


def _optimize(args: argparse.Namespace) -> None:
    from optimize.collapse import collapse_animation_clip
    from retarget.solver import load_animation_clip, save_animation_clip

    clip = load_animation_clip(args.animation)
    optimized_clip, reports = collapse_animation_clip(
        clip, preset=args.collapse, custom_threshold=args.threshold
    )

    save_animation_clip(optimized_clip, args.out)
    for bone_name, report in reports.items():
        print(
            f"[motion-tool]   {bone_name}: {report.original_key_count} -> "
            f"{report.optimized_key_count} keys (removed {report.removed_key_count}, "
            f"max_error={report.max_error:.3f}, threshold={report.threshold:.3f})"
        )
    print(f"[motion-tool] optimized {len(optimized_clip.tracks)} bone tracks -> {args.out}")


def _default_blender_search_paths(
    system: str | None = None,
    *,
    windows_program_files: Path | None = None,
    macos_application_dirs: list[Path] | None = None,
    linux_candidates: list[Path] | None = None,
) -> list[Path]:
    """Candidate blender executable paths for the platforms Blender ships
    an installer for. Existence is checked by the caller — this only
    enumerates plausible locations, newest version first where a
    version number is part of the path.

    The keyword overrides exist so this can be unit-tested against a
    fake filesystem layout for every branch regardless of which OS
    actually runs the test suite; production callers never pass them.
    """
    system = system or platform.system()

    if system == "Windows":
        # Blender doesn't add itself to PATH on Windows.
        program_files = windows_program_files or Path(
            os.environ.get("ProgramFiles", r"C:\Program Files")
        )
        foundation_dir = program_files / "Blender Foundation"
        if not foundation_dir.is_dir():
            return []
        return [
            entry / "blender.exe" for entry in sorted(foundation_dir.iterdir(), reverse=True)
        ]

    if system == "Darwin":
        # The macOS installer drops Blender.app into /Applications (or
        # ~/Applications for a per-user install); it isn't added to PATH
        # either. Some installs are versioned ("Blender 4.5.app"), so
        # check both plain and per-user Applications, newest name first.
        application_dirs = macos_application_dirs or [
            Path("/Applications"),
            Path.home() / "Applications",
        ]
        candidates: list[Path] = []
        for apps_dir in application_dirs:
            if not apps_dir.is_dir():
                continue
            for entry in sorted(apps_dir.iterdir(), reverse=True):
                if entry.suffix == ".app" and entry.stem.lower().startswith("blender"):
                    candidates.append(entry / "Contents" / "MacOS" / "Blender")
        return candidates

    if system == "Linux":
        if linux_candidates is not None:
            return linux_candidates
        # Linux has no single standard install location (tarball
        # extracted anywhere, distro package, Snap, Flatpak); PATH
        # (already checked via shutil.which before this runs) is the
        # primary mechanism. These cover the most common non-PATH cases.
        return [
            Path("/usr/bin/blender"),
            Path("/usr/local/bin/blender"),
            Path("/opt/blender/blender"),
            Path("/snap/blender/current/blender"),
            Path("/var/lib/flatpak/exports/bin/org.blender.Blender"),
        ]

    return []


def _find_blender_executable(override: str | None = None) -> str:
    if override:
        return override

    env_path = os.environ.get("BLENDER_EXECUTABLE")
    if env_path:
        return env_path

    found = shutil.which("blender")
    if found:
        return found

    for candidate in _default_blender_search_paths():
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find the Blender executable. Set BLENDER_EXECUTABLE, add "
        "blender to PATH, or pass --blender-executable."
    )


def _export_blender(args: argparse.Namespace) -> None:
    blender_exe = _find_blender_executable(args.blender_executable)
    script_path = _PROJECT_ROOT / "scripts" / "apply_motion.py"

    command = [
        blender_exe,
        "--background",
        "--python",
        str(script_path),
        "--",
        "--rig",
        args.rig,
        "--animation",
        args.animation,
        "--out",
        args.out,
    ]
    if args.fbx_out:
        command += ["--fbx-out", args.fbx_out]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"blender exited with code {result.returncode}")
    if not Path(args.out).exists():
        # Confirmed by testing against a real Blender build: an
        # unhandled exception inside a --python script does NOT make
        # blender.exe's own exit code non-zero, so returncode==0 alone
        # doesn't prove apply_motion.py actually succeeded. Belt and
        # suspenders on top of that script's own try/except.
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"blender exited with code 0 but {args.out} was not created; see stderr above"
        )
    print(f"[motion-tool] exported Blender scene -> {args.out}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "extract-frames":
            _extract_frames(args)
        elif args.command == "estimate-pose":
            _estimate_pose(args)
        elif args.command == "build-motion":
            _build_motion(args)
        elif args.command == "parse-rig":
            _parse_rig(args)
        elif args.command == "create-mapping":
            _create_mapping(args)
        elif args.command == "retarget":
            _retarget(args)
        elif args.command == "optimize":
            _optimize(args)
        elif args.command == "export-blender":
            _export_blender(args)
        else:
            raise NotImplementedError(f"unhandled command: {args.command}")
    except ImportError as exc:
        print(f"[motion-tool] {exc}", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"[motion-tool] {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[motion-tool] {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"[motion-tool] {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"[motion-tool] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
