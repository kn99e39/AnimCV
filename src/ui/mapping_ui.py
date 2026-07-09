"""Interactive CLI-assisted bone mapping (Architecture_v2.md section 6).

A full point-and-click UI (frame + bone list side by side, section 6.2
steps 3-5) is future work; a text-driven mapper is enough to build a
real, reusable ``BoneMappingProfile`` for a personal single-user tool
and satisfies Milestone 4 ("minimal mapping UI or CLI-assisted
mapping").

For each bone (in the order given), one line of input selects a
mapping:

    landmark <source_name> [axis_hint]
    direction <source_a> <source_b> [axis_hint]
    custom_point <point_id> [axis_hint]
    skip | <empty line>      -- leave this bone unmapped
    done                     -- stop prompting; keep entries so far

Chain Mapping (section 6.3) is not a separate command: a chain is just
several consecutive ``direction`` answers, e.g.
``spine_01 -> pelvis/chest`` then ``spine_02 -> chest/neck``.
"""

from __future__ import annotations

import sys
from typing import IO

from rig.bone_mapping import BoneMappingEntry, BoneMappingProfile


class MappingCommandError(ValueError):
    pass


def parse_mapping_line(bone_name: str, line: str) -> BoneMappingEntry | None:
    """Parse one line of interactive mapping input.

    Returns ``None`` for an explicit skip / empty line. Raises
    ``MappingCommandError`` for malformed input; the caller decides
    whether that means "skip this bone" or "re-prompt".
    """
    stripped = line.strip()
    if not stripped or stripped.lower() == "skip":
        return None

    parts = stripped.split()
    mode = parts[0].lower()

    if mode == "direction":
        if len(parts) not in (3, 4):
            raise MappingCommandError(
                "direction needs 2 source names and an optional axis hint: "
                "direction <source_a> <source_b> [axis_hint]"
            )
        return BoneMappingEntry(
            target_bone=bone_name,
            source_type="landmark",
            source_names=[parts[1], parts[2]],
            mapping_mode="direction",
            axis_hint=parts[3] if len(parts) == 4 else None,
        )

    if mode == "landmark":
        if len(parts) not in (2, 3):
            raise MappingCommandError(
                "landmark needs a source name and an optional axis hint: "
                "landmark <source_name> [axis_hint]"
            )
        return BoneMappingEntry(
            target_bone=bone_name,
            source_type="landmark",
            source_names=[parts[1]],
            mapping_mode="landmark",
            axis_hint=parts[2] if len(parts) == 3 else None,
        )

    if mode == "custom_point":
        if len(parts) not in (2, 3):
            raise MappingCommandError(
                "custom_point needs a point id and an optional axis hint: "
                "custom_point <point_id> [axis_hint]"
            )
        return BoneMappingEntry(
            target_bone=bone_name,
            source_type="custom_point",
            source_names=[parts[1]],
            mapping_mode="point",
            axis_hint=parts[2] if len(parts) == 3 else None,
        )

    raise MappingCommandError(
        f"unrecognized mapping command {mode!r}; expected one of "
        "landmark/direction/custom_point/skip/done"
    )


def run_interactive_mapping(
    bone_names: list[str],
    rig_id: str,
    created_from_frame: int = 0,
    input_stream: IO[str] | None = None,
    output_stream: IO[str] | None = None,
) -> BoneMappingProfile:
    # Resolved here rather than as parameter defaults: a default of
    # `sys.stdin`/`sys.stdout` would bind to whatever those names point
    # to at function-definition time, so monkeypatching sys.stdin later
    # (e.g. in tests, or after CLI stream redirection) would silently
    # have no effect.
    if input_stream is None:
        input_stream = sys.stdin
    if output_stream is None:
        output_stream = sys.stdout

    entries: list[BoneMappingEntry] = []

    for bone_name in bone_names:
        output_stream.write(
            f"{bone_name}> (landmark <name> | direction <a> <b> | custom_point <id> | skip) "
        )
        output_stream.flush()
        line = input_stream.readline()
        if line == "":  # EOF: treat like an early "done"
            break
        if line.strip().lower() == "done":
            break
        try:
            entry = parse_mapping_line(bone_name, line)
        except MappingCommandError as exc:
            output_stream.write(f"  ! {exc} -- skipping {bone_name}\n")
            continue
        if entry is not None:
            entries.append(entry)

    return BoneMappingProfile(
        rig_id=rig_id, entries=entries, created_from_frame=created_from_frame
    )
