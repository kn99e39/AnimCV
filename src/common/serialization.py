"""JSON file read/write helpers shared by every schema module.

Each schema dataclass implements its own ``to_dict``/``from_dict`` pair;
these helpers only handle the file <-> dict boundary so that on-disk
format stays plain, inspectable JSON (see Architecture_v2.md section 14.9).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
