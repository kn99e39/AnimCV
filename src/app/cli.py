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

def _detect_project_root() -> Path:
    # When frozen by PyInstaller, __file__ points inside the bundle
    # (onedir extraction folder / onefile temp dir), not the real
    # source tree, so "scripts/apply_motion.py" must be located next
    # to the executable instead (see build_windows.py, which copies
    # scripts/ there as PyInstaller data).
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent.parent


_PROJECT_ROOT = _detect_project_root()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motion-tool", description="Video-to-armature motion pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract-frames", help="Decode a video into a cached frame sequence")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="Output directory for extracted frame images")
    p.add_argument("--fps", type=float, default=None, help="Optional target fps override")
    p.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Optional inclusive source-video frame index to start from (Architecture_v2.md "
        "section 1.1's 'Start frame')",
    )
    p.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Optional inclusive source-video frame index to stop at ('End frame')",
    )

    p = sub.add_parser("estimate-pose", help="Run pose estimation over a frame sequence")
    p.add_argument("--frames", required=True, help="Directory of extracted frame images")
    p.add_argument("--out", required=True, help="Output pose.json path")
    p.add_argument(
        "--pose-config",
        default=None,
        help="MMPose model config path (default: bundled RTMPose-tiny config)",
    )
    p.add_argument(
        "--pose-checkpoint",
        default=None,
        help="MMPose model checkpoint path (default: RTMPose-tiny, downloaded to "
        "~/.cache/animcv/models on first use if not already cached there)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--tracking-report-out", default=None, help="Output tracking gate report (default: beside --out)")

    p.add_argument(
        "--evaluation-ground-truth", default=None,
        help="Benchmark-only GT pose JSON used to supply per-frame person boxes; never use for production output",
    )

    p.add_argument("--visibility-threshold", type=float, default=0.3)
    p.add_argument("--subject-box", default=None, help="Track one subject from x1,y1,x2,y2; requires detector options")
    p.add_argument("--detector-config", default=None, help="MMDetection person-detector config for --subject-box")
    p.add_argument("--detector-checkpoint", default=None, help="MMDetection person-detector checkpoint for --subject-box")
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

    p = sub.add_parser("estimate-root-motion", help="Estimate root yaw and character-space 3D joints")
    p.add_argument("--lifted-pose", required=True, help="Input lifted_pose.json path")
    p.add_argument("--out", required=True, help="Output root_motion.json path")
    p.add_argument("--smoothing-window", type=int, default=5)
    p.add_argument("--max-yaw-step-degrees", type=float, default=20.0)

    p = sub.add_parser("audit-3d-pose", help="Evaluate temporal 3D pose and root-orientation quality")
    p.add_argument("--lifted-pose", required=True)
    p.add_argument("--root-motion", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("audit-supervised-3d", help="Compare a predicted 3D sequence with source-neutral ground truth")
    p.add_argument("--predicted", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--predicted-root", default=None)
    p.add_argument("--ground-truth-root", default=None)
    p.add_argument("--out", required=True)

    p = sub.add_parser("import-mpi3dhp-ground-truth", help="Import a locally licensed MPI-INF-3DHP camera sequence")
    p.add_argument("--annotation", required=True, help="Official annot.mat path (kept outside the repository)")
    p.add_argument("--calibration", required=True, help="Official camera.calibration path")
    p.add_argument("--camera-index", required=True, type=int)
    p.add_argument("--pose-out", required=True, help="Canonical GT 2D pose JSON output")
    p.add_argument("--lifted-out", required=True, help="Canonical root-relative GT 3D pose JSON output")
    p.add_argument("--calibration-out", required=True, help="AnimCV camera calibration JSON output")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)

    p = sub.add_parser("import-mpi3dhp-supervised-dataset", help="Build a trainable supervised clip directly from MPI-INF-3DHP GT")
    p.add_argument("--annotation", required=True)
    p.add_argument("--camera-index", required=True, type=int)
    p.add_argument("--image-width", required=True, type=int)
    p.add_argument("--image-height", required=True, type=int)
    p.add_argument("--sequence-id", required=True)
    p.add_argument("--split", choices=["train", "validation", "holdout"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)

    p = sub.add_parser("import-3dpw-supervised-dataset", help="Build a trainable supervised dataset from one official 3DPW sequence")
    p.add_argument("--sequence", required=True, help="Official 3DPW sequenceFiles/*/*.pkl path")
    p.add_argument("--split", choices=["train", "validation", "holdout"], required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("audit-mpi3dhp-2d", help="Evaluate estimated 2D pose against imported MPI-INF-3DHP GT")
    p.add_argument("--pose", required=True, help="Estimated canonical 2D pose JSON")
    p.add_argument("--ground-truth", required=True, help="Canonical GT 2D pose JSON from import-mpi3dhp-ground-truth")
    p.add_argument("--out", required=True)

    p = sub.add_parser("audit-mpi3dhp-3d", help="Evaluate lifted pose and root yaw against imported MPI-INF-3DHP GT")
    p.add_argument("--lifted-pose", required=True)
    p.add_argument("--ground-truth", required=True)
    p.add_argument("--root-motion", required=True)
    p.add_argument("--ground-truth-root-motion", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("build-supervised-3d-dataset", help="Pair licensed 2D pose and root-relative 3D GT for own-data training")
    p.add_argument("--pose", required=True)
    p.add_argument("--ground-truth", required=True, help="Root-relative LiftedPoseSequence ground truth")
    p.add_argument("--image-width", required=True, type=int)
    p.add_argument("--image-height", required=True, type=int)
    p.add_argument("--sequence-id", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("triangulate-supervised-3d-ground-truth", help="Triangulate synchronised calibrated camera poses into own-data 3D GT")
    p.add_argument("--observations", required=True, help="Comma-separated camera=pose.json pairs, e.g. front=front_pose.json,side=side_pose.json")
    p.add_argument("--calibration", required=True, help="animcv_multiview_calibration_v1 JSON in metres")
    p.add_argument("--reference-camera", required=True, help="Camera whose coordinates define emitted 3D GT")
    p.add_argument("--out", required=True, help="Output root-relative LiftedPoseSequence JSON")
    p.add_argument("--report-out", required=True, help="Output triangulation coverage/reprojection report")
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--max-reprojection-error-pixels", type=float, default=10.0)

    p = sub.add_parser("train-supervised-3d-lifter", help="Train the own-data temporal 2D-to-3D baseline")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True, help="Output checkpoint path")
    p.add_argument("--window", type=int, default=81)
    p.add_argument("--channels", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--distributed", action="store_true", help="Use DDP; launch this command with torchrun")
    p.add_argument("--no-mixed-precision", action="store_true", help="Disable CUDA AMP")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--inference-batch-size", type=int, default=1024)
    p.add_argument("--input-jitter-std", type=float, default=0.0,
                   help="Per-epoch normalized 2D coordinate noise for detector-domain augmentation")
    p.add_argument("--input-dropout-probability", type=float, default=0.0,
                   help="Per-observed-joint dropout probability for detector-domain augmentation")
    p.add_argument("--confidence-jitter-std", type=float, default=0.0,
                   help="Per-epoch confidence noise standard deviation")
    p.add_argument("--input-global-scale-std", type=float, default=0.0,
                   help="Per-frame global 2D scale noise around the normalized image center")
    p.add_argument("--input-translation-std", type=float, default=0.0,
                   help="Per-frame normalized 2D translation noise")
    p.add_argument("--input-rotation-degrees", type=float, default=0.0,
                   help="Maximum per-frame in-plane 2D rotation")
    p.add_argument("--temporal-occlusion-probability", type=float, default=0.0,
                   help="Approximate observed-joint fraction dropped in contiguous temporal spans")
    p.add_argument("--temporal-occlusion-frames", type=int, default=9,
                   help="Odd temporal span length for contiguous occlusion")
    p.add_argument("--source-balanced-sampling", action="store_true",
                   help="Sample equal frame mass per declared source dataset each epoch")
    p.add_argument("--architecture", choices=["legacy_tcn_v1", "dilated_tcn_v1"], default="dilated_tcn_v1")
    p.add_argument("--bone-loss-weight", type=float, default=0.0)
    p.add_argument("--torso-loss-weight", type=float, default=0.0)
    p.add_argument("--hinge-loss-weight", type=float, default=0.0)
    p.add_argument("--init-checkpoint", default=None,
                   help="Compatible checkpoint to initialize a pretrain→fine-tune run; optimizer is reset")
    p.add_argument("--report-out", required=True)

    p = sub.add_parser("preflight-training", help="Verify PyTorch and the requested training device before a server run")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)

    p = sub.add_parser("combine-supervised-3d-datasets", help="Combine complete supervised clips without crossing temporal boundaries")
    p.add_argument("--datasets", required=True, help="Comma-separated supervised dataset JSON paths")
    p.add_argument("--expected-split", choices=["train", "validation", "holdout"], default=None)
    p.add_argument("--out", required=True)

    p = sub.add_parser("evaluate-supervised-3d-lifter", help="Evaluate a trained own-data temporal baseline")
    p.add_argument("--dataset", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)

    p = sub.add_parser("lift-supervised-3d", help="Lift 2D pose with a trained own-data temporal checkpoint")
    p.add_argument("--pose", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-width", required=True, type=int)
    p.add_argument("--image-height", required=True, type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)

    p = sub.add_parser("lift-pose3d", help="Temporally lift tracked 2D pose into pelvis-relative 3D joints")
    p.add_argument("--pose", required=True, help="Input canonical pose.json path")
    p.add_argument("--out", required=True, help="Output lifted_pose.json path")
    p.add_argument("--image-width", required=True, type=int)
    p.add_argument("--image-height", required=True, type=int)
    p.add_argument("--checkpoint", required=True, help="MMPose VideoPose3D 81-frame checkpoint")
    p.add_argument("--config", default=None, help="Optional MMPose VideoPose3D config override")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--min-observation-confidence", type=float, default=None,
        help="Override the recorded 2D observation threshold; defaults to the input pose policy",
    )
    p.add_argument("--max-interpolation-gap", type=int, default=5)

    p = sub.add_parser("reconstruct-kinematic-pose", help="Apply subject-specific fixed-length reconstruction to lifted 3D pose")
    p.add_argument("--lifted-pose", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-confidence", type=float, default=0.3)

    p = sub.add_parser("stabilize-bend-planes", help="Stabilize knee/elbow bend direction in fixed-length 3D pose")
    p.add_argument("--lifted-pose", required=True)
    p.add_argument("--root-motion", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-bend-degrees", type=float, default=12.0)

    p = sub.add_parser("audit-reprojection", help="Check 2D preservation with a weak-perspective proxy")
    p.add_argument("--pose", required=True, help="Trusted tracked 2D pose JSON")
    p.add_argument("--baseline", required=True, help="Raw lifted 3D pose JSON")
    p.add_argument("--reconstructed", required=True, help="Kinematically reconstructed 3D pose JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--max-median-worsening-ratio", type=float, default=1.05)

    p = sub.add_parser("audit-calibrated-reprojection", help="Check 2D preservation with supplied pinhole camera calibration")
    p.add_argument("--pose", required=True, help="Trusted tracked 2D pose JSON")
    p.add_argument("--baseline", required=True, help="Raw lifted 3D pose JSON")
    p.add_argument("--reconstructed", required=True, help="Kinematically reconstructed 3D pose JSON")
    p.add_argument("--calibration", required=True, help="animcv_camera_calibration_v1 JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--max-median-worsening-ratio", type=float, default=1.05)

    p = sub.add_parser("estimate-camera-calibration", help="Conservatively self-calibrate a static camera from 2D/3D pose")
    p.add_argument("--pose", required=True, help="Trusted tracked 2D pose JSON")
    p.add_argument("--lifted-pose", required=True, help="Lifted camera-relative 3D pose JSON")
    p.add_argument("--image-width", required=True, type=int)
    p.add_argument("--image-height", required=True, type=int)
    p.add_argument("--out", required=True, help="Output animcv_camera_calibration_v1 JSON")
    p.add_argument("--report-out", required=True, help="Output self-calibration quality report JSON")
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--max-focal-uncertainty-ratio", type=float, default=1.5)

    p = sub.add_parser("prepare-constraint-targets", help="Run R2/R4/R3 in a safe order for future constraint retargeting")
    p.add_argument("--lifted-pose", required=True)
    p.add_argument("--pose-out", required=True, help="Final bend-stabilized lifted pose JSON")
    p.add_argument("--root-motion-out", required=True, help="Final matching root-motion JSON")
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--smoothing-window", type=int, default=5)
    p.add_argument("--max-yaw-step-degrees", type=float, default=15.0)
    p.add_argument("--min-bend-degrees", type=float, default=12.0)

    p = sub.add_parser("audit-pose-uncertainty", help="Write traceable per-joint 3D quality scores and limb gate rates")
    p.add_argument("--raw-lifted-pose", required=True)
    p.add_argument("--prepared-pose", required=True)
    p.add_argument("--root-motion", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--unsafe-threshold", type=float, default=0.55)

    p = sub.add_parser("render-3d-audit", help="Render front, side, and top SVG views for selected 3D audit frames")
    p.add_argument("--root-motion", required=True)
    p.add_argument("--out", required=True, help="Output SVG path")
    p.add_argument("--frames", default=None, help="Comma-separated frame indices; default is first/middle/last")

    p = sub.add_parser("build-constraint-targets", help="Build R3/R5-gated 3D end-effector and pole targets for a rig adapter")
    p.add_argument("--root-motion", required=True)
    p.add_argument("--uncertainty", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-limb-unsafe-rate", type=float, default=0.20)

    p = sub.add_parser("retarget-constraint-targets", help="Bake safe prepared 3D targets into rig direction FK tracks")
    p.add_argument("--root-motion", required=True)
    p.add_argument("--constraint-targets", required=True)
    p.add_argument("--rig", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--out", required=True)

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
    p.add_argument(
        "--min-visibility-rate", type=float, default=0.60,
        help="Minimum fraction of frames where every landmark of a mapping is visible",
    )
    p.add_argument(
        "--min-mean-confidence", type=float, default=0.30,
        help="Minimum mean confidence across each mapping's required landmarks",
    )
    p.add_argument(
        "--max-direction-step-degrees", type=float, default=120.0,
        help="Reject a direction mapping whose consecutive-frame turn exceeds this angle",
    )
    p.add_argument(
        "--skip-quality-check", action="store_true",
        help="Allow retargeting low-quality input; intended only for manual recovery workflows",
    )
    p.add_argument(
        "--quality-report", default=None,
        help="Optional JSON path for per-mapping quality metrics; written even if retarget is rejected",
    )
    p.add_argument(
        "--smoothing-window", type=int, default=3,
        help="Odd median-filter window for valid landmarks after quality validation (default: 3; 1 disables)",
    )

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

    sequence = VideoLoader().load_video(
        args.video, target_fps=args.fps, start_frame=args.start_frame, end_frame=args.end_frame
    )

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

    pose_config_path = args.pose_config
    pose_checkpoint_path = args.pose_checkpoint
    if pose_config_path is None:
        from pose.default_model import get_default_pose_config_path

        pose_config_path = get_default_pose_config_path()
    if pose_checkpoint_path is None:
        from pose.default_model import get_default_pose_checkpoint_path

        print("[motion-tool] no --pose-checkpoint given, using the default RTMPose-tiny "
              "model (downloading to ~/.cache/animcv/models if not already cached)...")
        pose_checkpoint_path = get_default_pose_checkpoint_path()

    sequence = VideoLoader().load_image_sequence(args.frames)
    config = MMPoseConfig(
        config_path=pose_config_path,
        checkpoint_path=pose_checkpoint_path,
        device=args.device,
        visibility_threshold=args.visibility_threshold,
        subject_box=_parse_subject_box(args.subject_box),
        detector_config_path=args.detector_config,
        detector_checkpoint_path=args.detector_checkpoint,
    )
    estimator = PoseEstimator(config)
    if args.evaluation_ground_truth:
        from common.serialization import read_json
        poses = estimator.process_sequence_with_evaluation_boxes(
            sequence, PoseSequence.from_dict(read_json(args.evaluation_ground_truth))
        )
        print("[motion-tool] evaluation-only GT boxes supplied; detector quality is not measured")
    else:
        poses = estimator.process_sequence(sequence)
        tracking_report = {
            "schema": "animcv_tracking_audit_v1",
            **estimator.last_tracking_report,
            "passed": estimator.last_tracking_report.get("tracking_success_rate", 0.0) >= 0.95,
            "gate": {"minimum_tracking_success_rate": 0.95},
        }
        tracking_path = Path(args.tracking_report_out) if args.tracking_report_out else Path(args.out).with_name(
            Path(args.out).stem + "_tracking_report.json"
        )
        write_json(tracking_path, tracking_report)
        print(f"[motion-tool] tracking audit {'passed' if tracking_report['passed'] else 'failed'} -> {tracking_path}")

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


def _lift_pose3d(args: argparse.Namespace) -> None:
    from common.serialization import read_json
    from pose.pose_lifter import VideoPose3DConfig, VideoPose3DLifter, save_lifted_pose_sequence
    from pose.pose_types import PoseSequence

    poses = PoseSequence.from_dict(read_json(Path(args.pose)))
    observation_threshold = args.min_observation_confidence
    if observation_threshold is None:
        observation_threshold = poses.observation_confidence_threshold
    if observation_threshold is None:
        observation_threshold = 0.3  # legacy pose artifacts have no recorded policy
    lifter = VideoPose3DLifter(
        VideoPose3DConfig(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device=args.device,
            min_observation_confidence=observation_threshold,
            max_interpolation_gap=args.max_interpolation_gap,
        )
    )
    lifted = lifter.lift(poses, (args.image_width, args.image_height))
    save_lifted_pose_sequence(lifted, args.out)
    print(f"[motion-tool] lifted {len(lifted.frames)} 3D pose frames -> {args.out}")


def _estimate_root_motion(args: argparse.Namespace) -> None:
    from pose.pose_lifter import load_lifted_pose_sequence
    from pose.root_motion import estimate_root_motion, save_root_motion_sequence

    root_motion = estimate_root_motion(
        load_lifted_pose_sequence(args.lifted_pose), args.smoothing_window,
        args.max_yaw_step_degrees,
    )
    save_root_motion_sequence(root_motion, args.out)
    print(f"[motion-tool] estimated root yaw for {len(root_motion.frames)} frames -> {args.out}")


def _audit_3d_pose(args: argparse.Namespace) -> None:
    from common.serialization import write_json
    from pose.pose_lifter import load_lifted_pose_sequence
    from pose.quality_audit import audit_3d_pose
    from pose.root_motion import load_root_motion_sequence

    report = audit_3d_pose(
        load_lifted_pose_sequence(args.lifted_pose), load_root_motion_sequence(args.root_motion)
    )
    write_json(args.out, report)
    status = "passed" if report["passed"] else "failed"
    print(f"[motion-tool] 3D pose audit {status} -> {args.out}")


def _audit_supervised_3d(args: argparse.Namespace) -> None:
    from common.serialization import write_json
    from pose.dataset_3d_audit import audit_supervised_3d
    from pose.pose_lifter import load_lifted_pose_sequence
    from pose.root_motion import load_root_motion_sequence

    report = audit_supervised_3d(
        load_lifted_pose_sequence(args.predicted), load_lifted_pose_sequence(args.ground_truth),
        load_root_motion_sequence(args.predicted_root) if args.predicted_root else None,
        load_root_motion_sequence(args.ground_truth_root) if args.ground_truth_root else None,
    )
    write_json(args.out, report)
    print(f"[motion-tool] supervised 3D audit -> {args.out}")


def _import_mpi3dhp_ground_truth(args: argparse.Namespace) -> None:
    from pose.camera_calibration import save_camera_calibration
    from pose.mpi3dhp_adapter import load_mpi3dhp_calibration, load_mpi3dhp_ground_truth
    from pose.pose_lifter import save_lifted_pose_sequence
    from common.serialization import write_json

    pose, lifted = load_mpi3dhp_ground_truth(
        args.annotation, args.camera_index, start_frame=args.start_frame, end_frame=args.end_frame
    )
    write_json(args.pose_out, pose.to_dict())
    save_lifted_pose_sequence(lifted, args.lifted_out)
    save_camera_calibration(load_mpi3dhp_calibration(args.calibration, args.camera_index), args.calibration_out)
    print(f"[motion-tool] imported {len(pose.frames)} MPI-INF-3DHP GT frames -> {args.pose_out}")


def _import_mpi3dhp_supervised_dataset(args: argparse.Namespace) -> None:
    from training.research_sources import import_mpi3dhp_dataset

    report = import_mpi3dhp_dataset(
        args.annotation, args.camera_index, (args.image_width, args.image_height), args.sequence_id,
        args.out, args.start_frame, args.end_frame, args.split,
    )
    print(f"[motion-tool] imported {report['frame_count']} MPI-INF-3DHP {report['split']} frames -> {args.out}")


def _import_3dpw_supervised_dataset(args: argparse.Namespace) -> None:
    from training.research_sources import import_3dpw_dataset

    report = import_3dpw_dataset(args.sequence, args.out, split=args.split)
    print(f"[motion-tool] imported {report['sequence_count']} 3DPW actor sequences / "
          f"{report['frame_count']} {report['split']} frames -> {args.out}")


def _audit_mpi3dhp_2d(args: argparse.Namespace) -> None:
    from common.serialization import read_json, write_json
    from pose.mpi3dhp_audit import audit_mpi3dhp_2d
    from pose.pose_types import PoseSequence

    report = audit_mpi3dhp_2d(
        PoseSequence.from_dict(read_json(args.pose)), PoseSequence.from_dict(read_json(args.ground_truth))
    )
    write_json(args.out, report)
    print(f"[motion-tool] MPI-INF-3DHP 2D audit {'passed' if report['passed'] else 'failed'} -> {args.out}")


def _audit_mpi3dhp_3d(args: argparse.Namespace) -> None:
    from common.serialization import write_json
    from pose.mpi3dhp_3d_audit import audit_mpi3dhp_3d
    from pose.pose_lifter import load_lifted_pose_sequence
    from pose.root_motion import load_root_motion_sequence

    report = audit_mpi3dhp_3d(
        load_lifted_pose_sequence(args.lifted_pose), load_lifted_pose_sequence(args.ground_truth),
        load_root_motion_sequence(args.root_motion), load_root_motion_sequence(args.ground_truth_root_motion),
    )
    write_json(args.out, report)
    print(f"[motion-tool] MPI-INF-3DHP 3D audit -> {args.out}")


def _build_supervised_3d_dataset(args: argparse.Namespace) -> None:
    from common.serialization import read_json
    from pose.pose_lifter import load_lifted_pose_sequence
    from pose.pose_types import PoseSequence
    from training.temporal_lifter import build_dataset, save_dataset

    dataset = build_dataset(
        PoseSequence.from_dict(read_json(args.pose)), load_lifted_pose_sequence(args.ground_truth),
        (args.image_width, args.image_height), args.sequence_id,
    )
    save_dataset(dataset, args.out)
    print(f"[motion-tool] built {len(dataset['frames'])} supervised 3D training frames -> {args.out}")


def _triangulate_supervised_3d_ground_truth(args: argparse.Namespace) -> None:
    from common.serialization import read_json, write_json
    from pose.multiview_triangulation import load_calibration, triangulate
    from pose.pose_lifter import save_lifted_pose_sequence
    from pose.pose_types import PoseSequence

    observations = {}
    for item in args.observations.split(","):
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in observations:
            raise ValueError("--observations must be unique comma-separated camera=pose.json pairs")
        observations[name] = PoseSequence.from_dict(read_json(path))
    result, report = triangulate(
        observations, load_calibration(args.calibration), args.reference_camera,
        args.min_confidence, args.max_reprojection_error_pixels,
    )
    save_lifted_pose_sequence(result, args.out)
    write_json(args.report_out, report)
    print(f"[motion-tool] triangulated {report['triangulated_joint_count']} joint samples; "
          f"coverage={report['coverage']:.1%} -> {args.out}")


def _train_supervised_3d_lifter(args: argparse.Namespace) -> None:
    from common.serialization import write_json
    from training.temporal_lifter import TrainingConfig, load_dataset, train

    report = train(load_dataset(args.dataset), args.out, TrainingConfig(
        window=args.window, channels=args.channels, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=args.device, distributed=args.distributed,
        mixed_precision=not args.no_mixed_precision, seed=args.seed,
        inference_batch_size=args.inference_batch_size, input_jitter_std=args.input_jitter_std,
        input_dropout_probability=args.input_dropout_probability,
        confidence_jitter_std=args.confidence_jitter_std,
        input_global_scale_std=args.input_global_scale_std, input_translation_std=args.input_translation_std,
        input_rotation_degrees=args.input_rotation_degrees,
        temporal_occlusion_probability=args.temporal_occlusion_probability,
        temporal_occlusion_frames=args.temporal_occlusion_frames,
        source_balanced_sampling=args.source_balanced_sampling, architecture=args.architecture,
        bone_loss_weight=args.bone_loss_weight, torso_loss_weight=args.torso_loss_weight,
        hinge_loss_weight=args.hinge_loss_weight, init_checkpoint=args.init_checkpoint,
    ))
    if report["is_primary"]:
        write_json(args.report_out, report)
        print(f"[motion-tool] trained supervised 3D lifter -> {args.out}")


def _preflight_training(args: argparse.Namespace) -> None:
    from common.serialization import write_json
    from training.temporal_lifter import preflight

    report = preflight(args.device)
    write_json(args.out, report)
    print(f"[motion-tool] training preflight passed on {args.device} -> {args.out}")


def _combine_supervised_3d_datasets(args: argparse.Namespace) -> None:
    from training.temporal_lifter import combine_datasets, load_dataset, save_dataset

    paths = [path for path in args.datasets.split(",") if path]
    if not paths:
        raise ValueError("--datasets must contain at least one path")
    combined = combine_datasets([load_dataset(path) for path in paths], args.expected_split)
    save_dataset(combined, args.out)
    print(f"[motion-tool] combined {len(combined['sequences'])} sequences / {len(combined['frames'])} frames -> {args.out}")


def _evaluate_supervised_3d_lifter(args: argparse.Namespace) -> None:
    from common.serialization import write_json
    from training.temporal_lifter import evaluate, load_dataset

    report = evaluate(load_dataset(args.dataset), args.checkpoint, args.device)
    write_json(args.out, report)
    print(f"[motion-tool] evaluated supervised 3D lifter -> {args.out}")


def _lift_supervised_3d(args: argparse.Namespace) -> None:
    from common.serialization import read_json
    from pose.pose_lifter import save_lifted_pose_sequence
    from pose.pose_types import PoseSequence
    from training.temporal_lifter import infer

    result = infer(
        PoseSequence.from_dict(read_json(args.pose)), args.checkpoint,
        (args.image_width, args.image_height), args.device,
    )
    save_lifted_pose_sequence(result, args.out)
    print(f"[motion-tool] lifted {len(result.frames)} frames with supervised temporal checkpoint -> {args.out}")


def _reconstruct_kinematic_pose(args: argparse.Namespace) -> None:
    from pose.kinematic_reconstruction import reconstruct_kinematic_pose
    from pose.pose_lifter import load_lifted_pose_sequence, save_lifted_pose_sequence

    result = reconstruct_kinematic_pose(load_lifted_pose_sequence(args.lifted_pose), args.min_confidence)
    save_lifted_pose_sequence(result, args.out)
    print(f"[motion-tool] reconstructed {len(result.frames)} fixed-length 3D frames -> {args.out}")


def _stabilize_bend_planes(args: argparse.Namespace) -> None:
    from pose.bend_plane import stabilize_bend_planes
    from pose.pose_lifter import load_lifted_pose_sequence, save_lifted_pose_sequence
    from pose.root_motion import load_root_motion_sequence

    result = stabilize_bend_planes(
        load_lifted_pose_sequence(args.lifted_pose), load_root_motion_sequence(args.root_motion),
        args.min_bend_degrees,
    )
    save_lifted_pose_sequence(result, args.out)
    print(f"[motion-tool] stabilized bend planes for {len(result.frames)} frames -> {args.out}")


def _audit_reprojection(args: argparse.Namespace) -> None:
    from common.serialization import read_json, write_json
    from pose.pose_lifter import load_lifted_pose_sequence
    from pose.pose_types import PoseSequence
    from pose.reprojection_audit import audit_weak_perspective_reprojection

    report = audit_weak_perspective_reprojection(
        PoseSequence.from_dict(read_json(args.pose)),
        load_lifted_pose_sequence(args.baseline),
        load_lifted_pose_sequence(args.reconstructed),
        args.min_confidence,
        args.max_median_worsening_ratio,
    )
    write_json(args.out, report)
    status = "passed" if report["passed"] else "failed"
    print(f"[motion-tool] weak-perspective reprojection audit {status} -> {args.out}")


def _audit_calibrated_reprojection(args: argparse.Namespace) -> None:
    from common.serialization import read_json, write_json
    from pose.calibrated_reprojection_audit import audit_calibrated_reprojection
    from pose.camera_calibration import load_camera_calibration
    from pose.pose_lifter import load_lifted_pose_sequence
    from pose.pose_types import PoseSequence

    report = audit_calibrated_reprojection(
        PoseSequence.from_dict(read_json(args.pose)),
        load_lifted_pose_sequence(args.baseline),
        load_lifted_pose_sequence(args.reconstructed),
        load_camera_calibration(args.calibration),
        args.min_confidence,
        args.max_median_worsening_ratio,
    )
    write_json(args.out, report)
    status = "passed" if report["passed"] else "failed"
    print(f"[motion-tool] calibrated reprojection audit {status} -> {args.out}")


def _estimate_camera_calibration(args: argparse.Namespace) -> None:
    from common.serialization import read_json, write_json
    from pose.camera_calibration import save_camera_calibration
    from pose.camera_self_calibration import estimate_static_camera_calibration
    from pose.pose_lifter import load_lifted_pose_sequence
    from pose.pose_types import PoseSequence

    calibration, report = estimate_static_camera_calibration(
        PoseSequence.from_dict(read_json(args.pose)), load_lifted_pose_sequence(args.lifted_pose),
        args.image_width, args.image_height, args.min_confidence, args.max_focal_uncertainty_ratio,
    )
    save_camera_calibration(calibration, args.out)
    write_json(args.report_out, report)
    status = "accepted" if report["accepted_for_limited_calibrated_audit"] else "rejected"
    print(f"[motion-tool] static camera self-calibration {status} -> {args.out}")


def _prepare_constraint_targets(args: argparse.Namespace) -> None:
    from pose.constraint_targets import prepare_constraint_targets
    from pose.pose_lifter import load_lifted_pose_sequence, save_lifted_pose_sequence
    from pose.root_motion import save_root_motion_sequence

    pose, root = prepare_constraint_targets(
        load_lifted_pose_sequence(args.lifted_pose), args.min_confidence, args.smoothing_window,
        args.max_yaw_step_degrees, args.min_bend_degrees,
    )
    save_lifted_pose_sequence(pose, args.pose_out)
    save_root_motion_sequence(root, args.root_motion_out)
    print(f"[motion-tool] prepared {len(pose.frames)} constraint-ready 3D target frames")


def _audit_pose_uncertainty(args: argparse.Namespace) -> None:
    from common.serialization import write_json
    from pose.pose_lifter import load_lifted_pose_sequence
    from pose.root_motion import load_root_motion_sequence
    from pose.uncertainty import audit_pose_uncertainty

    report = audit_pose_uncertainty(
        load_lifted_pose_sequence(args.raw_lifted_pose),
        load_lifted_pose_sequence(args.prepared_pose),
        load_root_motion_sequence(args.root_motion),
        args.unsafe_threshold,
    )
    write_json(args.out, report)
    print(f"[motion-tool] audited {report['point_count']} joint quality scores -> {args.out}")


def _render_3d_audit(args: argparse.Namespace) -> None:
    from pose.audit_visualization import render_audit_views
    from pose.root_motion import load_root_motion_sequence

    indices = [int(value) for value in args.frames.split(",")] if args.frames else None
    render_audit_views(load_root_motion_sequence(args.root_motion), args.out, indices)
    print(f"[motion-tool] rendered 3D audit views -> {args.out}")


def _build_constraint_targets(args: argparse.Namespace) -> None:
    from common.serialization import read_json, write_json
    from pose.constraint_target_builder import build_constraint_targets
    from pose.root_motion import load_root_motion_sequence

    result = build_constraint_targets(
        load_root_motion_sequence(args.root_motion), read_json(args.uncertainty), args.max_limb_unsafe_rate
    )
    write_json(args.out, result)
    print(f"[motion-tool] built {result['frame_count']} constraint target frames -> {args.out}")


def _retarget_constraint_targets(args: argparse.Namespace) -> None:
    from common.serialization import read_json
    from pose.root_motion import load_root_motion_sequence
    from retarget.constraint_target_solver import solve_constraint_target_animation
    from retarget.solver import save_animation_clip
    from rig.bone_mapping import load_bone_mapping_profile
    from rig.rig_profile import load_rig_profile

    clip = solve_constraint_target_animation(
        load_root_motion_sequence(args.root_motion), read_json(args.constraint_targets),
        load_rig_profile(args.rig), load_bone_mapping_profile(args.mapping),
    )
    save_animation_clip(clip, args.out)
    print(f"[motion-tool] baked {len(clip.tracks)} constraint-target FK tracks -> {args.out}")


def _parse_subject_box(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        box = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("--subject-box must be x1,y1,x2,y2") from exc
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("--subject-box must be x1,y1,x2,y2 with x2>x1 and y2>y1")
    return box


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
    from common.serialization import write_json
    from motion.motion_io import load_motion_graph
    from retarget.solver import RetargetSolver, save_animation_clip
    from rig.bone_mapping import load_bone_mapping_profile
    from rig.rig_parser import RigParser
    from retarget.quality import RetargetQualityConfig, RetargetQualityError, assess_retarget_quality

    motion_graph = load_motion_graph(args.motion)
    rig_profile = RigParser().load(args.rig)
    mapping_profile = load_bone_mapping_profile(args.mapping)

    quality_config = RetargetQualityConfig(
        min_visibility_rate=args.min_visibility_rate,
        min_mean_confidence=args.min_mean_confidence,
        max_direction_step_degrees=args.max_direction_step_degrees,
    )
    report = assess_retarget_quality(motion_graph, rig_profile, mapping_profile, quality_config)
    if args.quality_report:
        write_json(args.quality_report, report.to_dict())
    if not args.skip_quality_check and not report.passed:
        raise RetargetQualityError(report)

    clip = RetargetSolver().solve(
        motion_graph,
        rig_profile,
        mapping_profile,
        quality_config=quality_config,
        validate_quality=False,
        smoothing_window=args.smoothing_window,
    )

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


def run_export_blender(
    animation: str,
    rig: str,
    out: str,
    fbx_out: str | None = None,
    blender_executable: str | None = None,
) -> None:
    """Core export-blender logic, factored out of _export_blender so
    ui/gui_app.py can call it directly instead of duplicating the
    Blender-executable search + subprocess/exit-code handling below."""
    blender_exe = _find_blender_executable(blender_executable)
    script_path = _PROJECT_ROOT / "scripts" / "apply_motion.py"

    command = [
        blender_exe,
        "--background",
        "--python",
        str(script_path),
        "--",
        "--rig",
        rig,
        "--animation",
        animation,
        "--out",
        out,
    ]
    if fbx_out:
        command += ["--fbx-out", fbx_out]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"blender exited with code {result.returncode}")
    if not Path(out).exists():
        # Confirmed by testing against a real Blender build: an
        # unhandled exception inside a --python script does NOT make
        # blender.exe's own exit code non-zero, so returncode==0 alone
        # doesn't prove apply_motion.py actually succeeded. Belt and
        # suspenders on top of that script's own try/except.
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"blender exited with code 0 but {out} was not created; see stderr above")
    print(f"[motion-tool] exported Blender scene -> {out}")


def _export_blender(args: argparse.Namespace) -> None:
    run_export_blender(
        animation=args.animation,
        rig=args.rig,
        out=args.out,
        fbx_out=args.fbx_out,
        blender_executable=args.blender_executable,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "extract-frames":
            _extract_frames(args)
        elif args.command == "estimate-pose":
            _estimate_pose(args)
        elif args.command == "lift-pose3d":
            _lift_pose3d(args)
        elif args.command == "estimate-root-motion":
            _estimate_root_motion(args)
        elif args.command == "audit-3d-pose":
            _audit_3d_pose(args)
        elif args.command == "audit-supervised-3d":
            _audit_supervised_3d(args)
        elif args.command == "import-mpi3dhp-ground-truth":
            _import_mpi3dhp_ground_truth(args)
        elif args.command == "import-mpi3dhp-supervised-dataset":
            _import_mpi3dhp_supervised_dataset(args)
        elif args.command == "import-3dpw-supervised-dataset":
            _import_3dpw_supervised_dataset(args)
        elif args.command == "audit-mpi3dhp-2d":
            _audit_mpi3dhp_2d(args)
        elif args.command == "audit-mpi3dhp-3d":
            _audit_mpi3dhp_3d(args)
        elif args.command == "build-supervised-3d-dataset":
            _build_supervised_3d_dataset(args)
        elif args.command == "triangulate-supervised-3d-ground-truth":
            _triangulate_supervised_3d_ground_truth(args)
        elif args.command == "train-supervised-3d-lifter":
            _train_supervised_3d_lifter(args)
        elif args.command == "preflight-training":
            _preflight_training(args)
        elif args.command == "combine-supervised-3d-datasets":
            _combine_supervised_3d_datasets(args)
        elif args.command == "evaluate-supervised-3d-lifter":
            _evaluate_supervised_3d_lifter(args)
        elif args.command == "lift-supervised-3d":
            _lift_supervised_3d(args)
        elif args.command == "reconstruct-kinematic-pose":
            _reconstruct_kinematic_pose(args)
        elif args.command == "stabilize-bend-planes":
            _stabilize_bend_planes(args)
        elif args.command == "audit-reprojection":
            _audit_reprojection(args)
        elif args.command == "audit-calibrated-reprojection":
            _audit_calibrated_reprojection(args)
        elif args.command == "estimate-camera-calibration":
            _estimate_camera_calibration(args)
        elif args.command == "prepare-constraint-targets":
            _prepare_constraint_targets(args)
        elif args.command == "audit-pose-uncertainty":
            _audit_pose_uncertainty(args)
        elif args.command == "render-3d-audit":
            _render_3d_audit(args)
        elif args.command == "build-constraint-targets":
            _build_constraint_targets(args)
        elif args.command == "retarget-constraint-targets":
            _retarget_constraint_targets(args)
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
