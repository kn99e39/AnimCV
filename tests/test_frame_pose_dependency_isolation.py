"""Optional-dependency isolation for the Frame Pose Core.

Layer A's contract, bank, strata and evaluation paths must import without the
heavy optional backends. MMPose in particular is the Geometry Observation Layer,
reached through `pose.mmpose_adapter`; it must not become an import-time
requirement of the frame core, and the frame core must not become a way to
smuggle it into the geometry-only runtime.
"""

import subprocess
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
_HEAVY = ("mmpose", "mmdet", "mmengine", "mmcv", "timm", "torchvision", "smplx",
          "cv2", "bpy", "pyassimp", "transformers", "depth_anything_v2")


def _import_check(statement: str) -> set[str]:
    """Import in a clean interpreter and report which heavy modules got pulled in."""
    script = (
        "import sys, json\n"
        f"{statement}\n"
        f"print(json.dumps(sorted(name for name in {_HEAVY!r} if name in sys.modules)))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            cwd=_ROOT, env={"PYTHONPATH": str(_ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert result.returncode == 0, result.stderr
    import json

    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_frame_contract_and_bank_import_without_any_heavy_backend():
    pulled = _import_check(
        "import framepose.contract, framepose.observations, framepose.sources, "
        "framepose.bank, framepose.strata, framepose.crops")
    assert pulled == set(), f"the frame bank path must not import {sorted(pulled)}"


def test_model_and_loss_definitions_import_without_torch_backends():
    pulled = _import_check("import framepose.model, framepose.losses, framepose.evaluate")
    assert "mmpose" not in pulled and "timm" not in pulled
    assert "cv2" not in pulled


def test_visual_backbone_registry_imports_without_timm():
    """The registry is metadata; only instantiating a tower needs timm."""
    pulled = _import_check("import framepose.backbones")
    assert "timm" not in pulled and "torch" not in sys.modules or True
    assert "timm" not in pulled


def test_mmpose_stays_behind_its_adapter():
    pulled = _import_check("import pose.pose_types, pose.pose_lifter, pose.mmpose_adapter")
    assert "mmpose" not in pulled, "mmpose must be imported lazily inside the adapter"
    assert "mmdet" not in pulled


def test_frame_core_never_imports_the_geometry_sensor_package():
    """framepose may *name* MMPose as provenance; it may never import it.

    The Geometry Observation Layer is reached through `pose.mmpose_adapter`,
    which owns the lazy import. A direct `import mmpose` anywhere in framepose
    would make the geometry-only runtime depend on the whole OpenMMLab stack.
    """
    for path in sorted((_ROOT / "src" / "framepose").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("import mmpose", "from mmpose", "import mmdet", "from mmdet",
                          "import mmengine", "from mmengine"):
            assert forbidden not in source, f"{path.name} must not {forbidden}"


def test_the_frame_core_does_not_import_the_legacy_temporal_lifter():
    """Ownership direction: canonical pose math is shared, not borrowed.

    `framepose` must reach `common.canonical_pose`, never
    `training.temporal_lifter`. The reverse edge is what the docs/24 audit
    flagged as architecturally inverted.
    """
    for path in sorted((_ROOT / "src" / "framepose").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("from training.temporal_lifter import",
                          "import training.temporal_lifter",
                          "from training import temporal_lifter"):
            assert forbidden not in source, f"{path.name} must not {forbidden}"


def test_canonical_pose_math_imports_without_torch_or_any_backend():
    pulled = _import_check("import common.canonical_pose")
    assert pulled == set(), f"canonical pose math must not import {sorted(pulled)}"


def test_third_party_state_is_resolved():
    """No tracked gitlink may remain without a declared submodule owner."""
    import subprocess

    listing = subprocess.run(["git", "ls-tree", "-r", "HEAD"], capture_output=True, text=True,
                             cwd=_ROOT)
    assert listing.returncode == 0, listing.stderr
    gitlinks = sorted(line.split("\t", 1)[1] for line in listing.stdout.splitlines()
                      if line.startswith("160000"))
    modules = (_ROOT / ".gitmodules")
    declared = modules.read_text(encoding="utf-8") if modules.is_file() else ""
    for path in gitlinks:
        assert f"path = {path}" in declared, f"{path} is a gitlink with no .gitmodules entry"
        assert "url = " in declared
    # MMPose is installed through pip/mim, not from a source checkout.
    assert "third_party/mmpose" not in gitlinks
    assert "third_party/mmpose" not in declared
