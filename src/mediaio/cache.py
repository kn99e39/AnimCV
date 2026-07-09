"""Cache directory layout helper shared by perception/pose stages.

Implemented fully once Milestone 2 needs disk caching for decoded frames
and pose estimation results; provides just the directory layout for now.
"""

from __future__ import annotations

from pathlib import Path


def project_cache_dir(project_root: str | Path, project_id: str) -> Path:
    cache_dir = Path(project_root) / "outputs" / project_id / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
