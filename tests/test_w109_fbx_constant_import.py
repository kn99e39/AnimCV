"""W109 FBX importer CONSTANT-flag ownership: flag encoding, byte-exact patching, tree digests, declared functions,
classification, CASE mapping, code guards and stored-output consistency (no Blender)."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"scripts"))
import w109_fbx_constant_import_contract as C  # noqa: E402
import w108_fbx_subframe_contract as C8  # noqa: E402

BASE = Path(os.environ.get("W109_BASE", "C:/Users/dna10/animcv-w109"))
DECL = BASE/"results/declare/declaration.json"
ANALYSIS = BASE/"results/analysis/analysis_report.json"
W107_FBX = Path(os.environ.get("W107_BASE", "C:/Users/dna10/animcv-w107"))/"results/probe/complete_control/fbx/V1.fbx"


# --- flag semantics ------------------------------------------------------------------------------------------------
def test_declared_variant_flags_and_decoding():
    assert C.SOURCE_FLAG == (1 << 2 | 1 << 8 | 1 << 13 | 1 << 14)            # io_scene_fbx 5.15 exporter literal
    assert C.EXPECTED_FLAGS == {"CONSTANT_standard": 0x6002, "CONSTANT_next": 0x6102}
    assert C.decode_flag(C.SOURCE_FLAG)["interpolation"] == "LINEAR"
    assert C.decode_flag(C.SOURCE_FLAG)["constant_mode"] is None
    std, nxt = C.decode_flag(0x6002), C.decode_flag(0x6102)
    assert (std["interpolation"], std["constant_mode"]) == ("CONSTANT", "STANDARD")
    assert (nxt["interpolation"], nxt["constant_mode"]) == ("CONSTANT", "NEXT")
    # only the interpolation field and the constant-mode bit change; all other bits retained
    for v in C.VARIANTS:
        assert (C.EXPECTED_FLAGS[v] ^ C.SOURCE_FLAG) & ~C.ALTERABLE_BITS == 0
    # a naive interpolation-only edit of 0x6104 would silently produce CONSTANT-next (bit 0x100 aliasing)
    assert C.decode_flag((C.SOURCE_FLAG & ~C.FBX_INTERP_MASK) | C.FBX_CONSTANT)["constant_mode"] == "NEXT"
    with pytest.raises(ValueError):
        C.variant_flag(C.SOURCE_FLAG, "CUBIC")


# --- minimal binary FBX 7400 with a target and a non-target stack ---------------------------------------------------
def _prop(v):
    if isinstance(v, bytes):
        return b"S" + struct.pack("<I", len(v)) + v
    if isinstance(v, bool):
        return b"C" + (b"\x01" if v else b"\x00")
    if isinstance(v, int):
        return b"L" + struct.pack("<q", v)
    if isinstance(v, tuple):
        t, a, comp = v
        raw = a.tobytes()
        data = zlib.compress(raw) if comp else raw
        return t + struct.pack("<III", a.size, 1 if comp else 0, len(data)) + data
    raise TypeError(v)


def _node(name, props=(), children=(), off=0):
    pb = b"".join(_prop(p) for p in props)
    head_len = 13 + len(name)
    body = b""
    cur = off + head_len + len(pb)
    for ch in children:
        enc = _node(*ch, off=cur)
        body += enc
        cur += len(enc)
    if children:
        body += b"\x00"*13
    end = off + head_len + len(pb) + len(body)
    return struct.pack("<III", end, len(props), len(pb)) + bytes([len(name)]) + name + pb + body


def _curve(uid, flags, times, compress_flags=False):
    tick = C8.FBX_KTIME_PER_SECOND//30
    return (b"AnimationCurve", (uid, b"\x00\x01AnimCurve", b""), [
        (b"KeyTime", ((b"l", np.array(times, dtype=np.int64)*tick, True),)),
        (b"KeyValueFloat", ((b"f", np.arange(len(times), dtype=np.float32), False),)),
        (b"KeyAttrFlags", ((b"i", np.array([flags], dtype=np.int32), compress_flags),)),
        (b"KeyAttrDataFloat", ((b"f", np.zeros(4, dtype=np.float32), False),)),
        (b"KeyAttrRefCount", ((b"i", np.array([len(times)], dtype=np.int32), False),))])


def _fbx(flags=C.SOURCE_FLAG, other_flags=C.SOURCE_FLAG, compress_flags=False):
    objects = (b"Objects", (), [
        (b"AnimationStack", (1, b"root|ACT\x00\x01AnimStack", b"")), (b"AnimationLayer", (2, b"L\x00\x01AnimLayer", b"")),
        (b"AnimationCurveNode", (3, b"R\x00\x01AnimCurveNode", b"")),
        _curve(11, flags, [1, 2, 3, 4], compress_flags), _curve(12, flags, [1, 2, 3, 4], compress_flags),
        (b"AnimationStack", (4, b"root|OTHER\x00\x01AnimStack", b"")), (b"AnimationLayer", (5, b"L\x00\x01AnimLayer", b"")),
        (b"AnimationCurveNode", (6, b"R\x00\x01AnimCurveNode", b"")), _curve(21, other_flags, [1, 2, 3]),
        (b"Model", (7, b"root\x00\x01Model", b"Null"), [(b"Flag", (True,))])])
    conns = (b"Connections", (), [(b"C", (b"OO", 2, 1)), (b"C", (b"OO", 3, 2)), (b"C", (b"OP", 11, 3, b"d|X")),
                                  (b"C", (b"OP", 12, 3, b"d|Y")), (b"C", (b"OO", 5, 4)), (b"C", (b"OO", 6, 5)),
                                  (b"C", (b"OP", 21, 6, b"d|X")), (b"C", (b"OP", 3, 7, b"Lcl Rotation")),
                                  (b"C", (b"OP", 6, 7, b"Lcl Rotation"))])
    out = C8.FBX_MAGIC + struct.pack("<I", 7400)
    for n in (objects, conns):
        out += _node(*n, off=len(out))
    return out + b"\x00"*13


def _tree(data, tmp_path, name="t.fbx"):
    p = tmp_path/name
    p.write_bytes(data)
    return C8.fbx_read(p)[0]


def test_target_selection_is_stack_scoped(tmp_path):
    root = _tree(_fbx(), tmp_path)
    sel, tg, smap = C.target_uids(root, "ACT")
    assert sel == "root|ACT" and tg == [11, 12] and smap[21] == "root|OTHER"
    assert C.target_uids(root, "MISSING")[1] == []


def test_patch_is_byte_exact_isolated_and_tree_verified(tmp_path):
    data = _fbx()
    root0 = _tree(data, tmp_path, "a.fbx")
    noop, _ = C.patch_flags(data, [11, 12], C.SOURCE_FLAG)
    assert noop == data                                                # no-op identity
    for v in C.VARIANTS:
        new, recs = C.patch_flags(data, [11, 12], C.EXPECTED_FLAGS[v])
        assert len(new) == len(data) and len(recs) == 2
        d = C.byte_diff_spans(data, new, [(r["offset"], r["bytes"]) for r in recs])
        assert d["all_in_targets"] and 0 < d["diff_bytes"] <= 8
        root1 = _tree(new, tmp_path, f"{v}.fbx")
        r0 = C.file_tree_record(root0, "ACT", "control")
        r1 = C.file_tree_record(root1, "ACT", v)
        assert r1["digest_targets_masked"] == r0["digest_targets_masked"]
        assert r1["digest_full"] != r0["digest_full"]
        assert r1["target_flags_hex"] == [f"0x{C.EXPECTED_FLAGS[v]:x}"] and r1["nontarget_flags_hex"] == ["0x6104"]
        assert r1["target_flags_decoded"][0]["interpolation"] == "CONSTANT"
        st = C8.fbx_anim_summary(root1, 30.0)["stacks"]
        assert st["root|ACT"]["interpolation"] == ["CONSTANT"] and st["root|OTHER"]["interpolation"] == ["LINEAR"]
        assert st["root|ACT"]["keys"] == 8 and st["root|ACT"]["refcount_covers_keys"]
    # masking only the targets: a change to a non-target flag is still detected
    other = _tree(_fbx(other_flags=0x6102), tmp_path, "o.fbx")
    assert C.tree_digest(other, [11, 12]) != C.tree_digest(root0, [11, 12])


def test_patch_refuses_unexpected_source_or_compressed(tmp_path):
    with pytest.raises(ValueError):
        C.patch_flags(_fbx(flags=0x6102), [11], 0x6002)                 # source value contradicts 0x6104
    with pytest.raises(ValueError):
        C.patch_flags(_fbx(compress_flags=True), [11], 0x6002)           # not patchable in place
    with pytest.raises(ValueError):
        C.patch_flags(_fbx(), [99], 0x6002)


def test_index_agrees_with_independent_reader(tmp_path):
    data = _fbx()
    locs = C.curve_flag_locations(data)
    assert sorted(locs) == sorted(C.curve_flags(_tree(data, tmp_path)))
    for p in locs.values():
        assert struct.unpack_from("<i", data, p["value_offset"])[0] == C.SOURCE_FLAG


def test_tree_digest_parser_neutral_for_bool_and_arrays(tmp_path):
    import array
    root = _tree(_fbx(), tmp_path)

    def convert(e):   # mimic io_scene_fbx.parse_fbx value types: 'C' as 1-byte bytes, arrays as array.array
        props = []
        for t, v in zip(bytes(e.props_type).decode(), e.props):
            if t == "C":
                v = b"\x01" if v else b"\x00"
            elif t in "ilfd":
                v = array.array({"i": "i", "l": "q", "f": "f", "d": "d"}[t], np.asarray(v).tobytes())
            props.append(v)
        return C8.Elem(e.id, props, bytearray(e.props_type), [convert(c) for c in e.elems])
    alt = convert(root)
    assert C.tree_digest(alt) == C.tree_digest(root)
    assert C.tree_digest(alt, [11]) == C.tree_digest(root, [11])


@pytest.mark.skipif(not W107_FBX.exists(), reason="W107 FBX absent")
def test_real_v1_fbx_targets_are_patchable_in_place():
    data = W107_FBX.read_bytes()
    root = C8.fbx_read(W107_FBX)[0]
    sel, tg, _ = C.target_uids(root, "W91 complete_control FIXED")
    assert sel and len(tg) > 0
    locs = C.curve_flag_locations(data)
    assert all(locs[u]["encoding"] == 0 and locs[u]["type"] == b"i" for u in tg)
    noop, _ = C.patch_flags(data, tg, C.SOURCE_FLAG)
    assert noop == data


# --- declared functions, classification and CASE ---------------------------------------------------------------
def _samples(times, f):
    t = np.asarray(times, dtype=float)
    v = np.array([f(x) for x in t], dtype=float)
    wh = np.zeros((len(t), 1, 3))
    wh[:, 0, 0] = v
    return {"times": t, "bones": np.array(["b"]), "world_head": wh, "world_tail": wh + 1.0}


def test_declared_function_error_floor_and_ceil():
    times = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    keys = {1: 0.0, 2: 1.0, 3: 3.0}
    lin = _samples(times, lambda x: np.interp(x, [1, 2, 3], [0, 1, 3]))
    held = _samples(times, lambda x: keys[int(np.floor(x))])
    nxt = _samples(times, lambda x: keys[int(np.ceil(x))])
    cls = {1: "transition", 2: "transition"}
    assert C.declared_function_error(held, lin, times, "floor", cls)["max_m"] == 0.0
    assert C.declared_function_error(nxt, lin, times, "ceil", cls)["max_m"] == 0.0
    e = C.declared_function_error(lin, lin, times, "floor", cls)
    assert e["max_m"] == pytest.approx(1.5) and e["phase_0.75_m"] == pytest.approx(1.5)
    assert C.declared_function_error(lin, lin, times, "ceil", cls)["phase_0.25_m"] == pytest.approx(1.5)


def _rec(**kw):
    r = {"imported_interpolation": ["LINEAR"], "keydata_equal_all": True, "keydata_equal_excl_interpolation": True,
         "delta_vs_control_m": 0.0, "integer_vs_control_m": 0.0, "declared_error_m": 0.5, "discriminable": True}
    r.update(kw)
    return r


def test_classify():
    assert C.classify(_rec()) == "LINEAR_UNCHANGED"
    assert C.classify(_rec(imported_interpolation=["CONSTANT"], keydata_equal_all=False, delta_vs_control_m=0.4,
                           declared_error_m=1e-6)) == "FOLLOWS_DECLARED"
    assert C.classify(_rec(imported_interpolation=["CONSTANT"], keydata_equal_all=False, delta_vs_control_m=0.4,
                           declared_error_m=0.2)) == "OTHER"
    assert C.classify(_rec(keydata_equal_excl_interpolation=False, keydata_equal_all=False, delta_vs_control_m=0.1,
                           declared_error_m=0.0)) == "OTHER"
    assert C.classify(_rec(discriminable=False)) == "NOT_DISCRIMINABLE"


def _pre(outcome_std="LINEAR_UNCHANGED", outcome_next="LINEAR_UNCHANGED", **kw):
    res = {"CONSTANT_standard": {c: {"outcome": outcome_std, "keydata_ok": True, "integer_ok": True}
                                 for c in C.TARGET_CASES},
           "CONSTANT_next": {c: {"outcome": outcome_next, "keydata_ok": True, "integer_ok": True}
                             for c in C.TARGET_CASES}}
    p = {"preflight_ok": True, "encoding_valid": True, "sources_ok": True, "matching_ok": True, "deterministic": True,
         "originals_unchanged": True, "p0": {c: True for c in C.ALL_CASES}, "smooth_valid": True, "results": res}
    p.update(kw)
    return p


def test_decide_case():
    assert C.decide_case(_pre())[0] == "A"
    assert C.decide_case(_pre("OTHER", "OTHER"))[0] == "A"                 # fails to follow the declared function
    assert C.decide_case(_pre("FOLLOWS_DECLARED"))[0] == "B"
    assert C.decide_case(_pre("FOLLOWS_DECLARED", "FOLLOWS_DECLARED"))[0] == "B"
    assert C.decide_case(_pre(encoding_valid=False))[0] == "D"
    assert C.decide_case(_pre(deterministic=False))[0] == "D"
    assert C.decide_case(_pre(p0={c: c != "real_clip" for c in C.ALL_CASES}))[0] == "D"
    assert C.decide_case(_pre(smooth_valid=False))[0] == "C"
    assert C.decide_case(_pre("FOLLOWS_DECLARED", smooth_valid=False))[0] == "C"
    mixed = _pre()
    mixed["results"]["CONSTANT_standard"]["syn_step"]["outcome"] = "FOLLOWS_DECLARED"
    assert C.decide_case(mixed)[0] == "C"
    nd = _pre()
    nd["results"]["CONSTANT_next"]["real_clip"]["outcome"] = "NOT_DISCRIMINABLE"
    assert C.decide_case(nd)[0] == "C"
    broken = _pre()
    broken["results"]["CONSTANT_next"]["complete_control"]["keydata_ok"] = False
    assert C.decide_case(broken)[0] == "C"


# --- code guards -------------------------------------------------------------------------------------------------
def test_no_importer_or_exporter_patch_and_no_post_import_edit():
    for name in ("w109_fbx_constant_import_blender.py", "w109_fbx_constant_import.py",
                 "w109_fbx_constant_import_contract.py"):
        src = (ROOT/"scripts"/name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        assert "torch" not in src
        for bad in ("import io_scene_fbx.import_fbx", "from io_scene_fbx import import_fbx", "io_scene_fbx.import_fbx",
                    "export_fbx_bin", "blender.export"):
            assert bad not in src, (name, bad)                              # no importer/exporter internals touched
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute):
                assert n.attr not in ("foreach_set", "keyframe_insert", "keyframe_delete", "save_as_mainfile",
                                      "save_mainfile", "export_scene"), (name, n.attr)
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    assert not (isinstance(t, ast.Attribute) and t.attr in ("interpolation", "fps", "frame_end")), name
    blender = (ROOT/"scripts/w109_fbx_constant_import_blender.py").read_text(encoding="utf-8")
    assert blender.count("bpy.ops.import_scene.fbx(filepath=str(fbx))") == 1   # default options only


# --- stored outputs (only once the run exists) -------------------------------------------------------------------
@pytest.mark.skipif(not ANALYSIS.exists(), reason="W109 outputs absent")
def test_stored_declaration_and_analysis_consistent():
    decl_sha = hashlib.sha256(DECL.read_bytes()).hexdigest()
    decl = json.loads(DECL.read_text())
    rep = json.loads(ANALYSIS.read_text())
    assert decl["outcomes_read"] is False and rep["declaration_sha256"] == decl_sha
    assert set(decl["cases"]) == set(C.ALL_CASES) and decl["cases"]["syn_smooth"]["variants"] == []
    assert decl["intervention"]["variants"]["CONSTANT_standard"]["flag_hex"] == "0x6002"
    assert decl["intervention"]["variants"]["CONSTANT_next"]["flag_hex"] == "0x6102"
    assert rep["CASE"] == C.decide_case(rep["decision_inputs"])[0]
