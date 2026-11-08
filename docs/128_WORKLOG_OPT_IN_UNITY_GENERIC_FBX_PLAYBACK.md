# Worklog — Opt-in Unity Generic-Rig Playback from Verified FBX Interchange: STOPPED, DATA BLOCKED (2026-11-09)

Batch 128 ran unattended on `arch/single_frame_first` on Main:

- local RTX 5080 (not used);
- Python `C:\Projects\AnimCV\.venv_build_full`;
- Blender 5.1.0;
- Unity Editors 2022.3.8f1 and 6000.5.2f1 (6000.2.12f1 is an incomplete install);
- Windows 11 Pro 10.0.26200.

`docs/next_batch.md` was read only.

Everything below is **AGENT-RUN** on Main. No Reviewer has re-executed any of it.

## 1. Outcome

**DATA BLOCKED / NO CASE.** No installed Unity Editor will start under a valid licence on Main. That is the Batch's first
STOP condition ("Unity licensing/editor … unavailable"), so I stopped there.

Not done:

- no prospective declaration was frozen;
- no Unity project, FBX import, AnimationClip, prefab, sample or visual was produced;
- no CASE A–D is assigned.

| Requirement | State |
|---|---|
| Unity Editor executable | 2022.3.8f1 and 6000.5.2f1: present. 6000.2.12f1: **no `Unity.exe`** (only `Unity.com`); the Hub does not list it. |
| Unity licence (batchmode) | **BLOCKED** on both executable editors: "No valid Unity Editor license found", exit 1 (2022.3) and 198 (6000.5) |
| Unity licence (GUI mode) | **BLOCKED** (6000.5.2f1, one manual probe, §3) |
| W127 S1/S3 packages (6 states) | **AUTHENTICATED**: seal, schema, FBX SHA and W127 evidence manifest all match (§4) |
| Blender (for the evaluator side) | present, `Blender 5.1.0` |

The S1/S3 artifacts and Blender would have been sufficient. Unity execution was the missing requirement.

The historical verdicts are untouched: W127 B, W126 C (official N12 40/41), W125 A (transactions only) and W124 B. No earlier
Worklog, run, package or test was edited.

## 2. Starting state and revisions

| Item | Value |
|---|---|
| Instruction blob | `8838fd224d814787b9c6ce900a24f96a57763ebe` (verified first) |
| Starting commit | `6b197956bcd77286c6903eec4cd4c137b7cc34e7`. `git fetch` at start: the remote had not advanced. |
| W127 completion head (provenance) | `5c2782916ae3741753b6328add134b13033f37cf`, an ancestor of the starting commit |
| Final source revision | `910d778`: the preflight script, the only code added |
| Later commits | documentation only: this Worklog and one `docs/README.md` index line, then the COMPLETE marker |
| Push | **Not pushed by this agent.** The first fetch + push waited about 10 min on a Git Credential Manager sign-in prompt and was killed. A non-interactive `git fetch` then succeeded, showing the remote had not advanced. A non-interactive `git push` failed: `could not read Username … terminal prompts disabled`. I did not sign in. The commits stay local, for the worker to publish. |
| Machine | Main only. No LabServer, no training. Network use: `git fetch/push`, plus Unity's own licensing client contacting Unity servers during the probes. |

## 3. What was run, and why it is a licence block

**Disclosure of the order.** I ran the environment checks first, by hand, from Git Bash. Then I wrote and ran the
preflight script. No Unity import outcome exists, so no outcome was seen before any declaration.

**Manual checks** (logs in `C:/Users/dna10/animcv-w128/envcheck/`):

| # | Command (all `-quit -createProject` into a new scratch path) | Result | Log SHA-256 |
|---|---|---|---|
| 1 | 6000.2.12f1 `Unity.exe -batchmode -nographics` | the executable does not exist | — |
| 2 | 6000.5.2f1 `-batchmode -nographics` | exit 198, "No valid Unity Editor license found" | `c133c914…` |
| 3 | 2022.3.8f1 `-batchmode -nographics` | exit 1, same message | `3968f156…` |
| 4 | `Unity Hub.exe -- --headless editors -i` | lists 6000.5.2f1 and 2022.3.8f1 only | `1cfc4e52…` |
| 5 | 6000.5.2f1 `-batchmode -nographics`, repeated after #4 | exit 198, same | `45e71b9c…` |
| 6 | 6000.5.2f1 **GUI mode** (no `-batchmode`) | "No valid Unity Editor license found"; still waiting at 150 s, so `timeout` killed it | `baaab6d1…` |

After #6, `tasklist` showed no Unity or Hub process left running. Each project directory is partial, with no `Assets/`.

**Preflight script** (`scripts/w128_unity_generic_preflight.py`; output `C:/Users/dna10/animcv-w128/preflight/preflight.json`,
SHA-256 `f6bd3e0c602aa00e38f83f9eae7dddf96ffdd030460662999ce015ab8f1282d9`, `outcomes_inspected: false`). It repeated the
batchmode probe for each executable editor:

| Editor | Exit | Time | Project created | Editor log | Probe log SHA-256 |
|---|---|---|---|---|---|
| 2022.3.8f1 | 1 | 1.1 s | no | `License is not active (com.unity.editor.headless)`; `Access token is unavailable` | `9ac3030a…` |
| 6000.5.2f1 | 198 | 0.45 s | no | `'com.unity.editor.headless' was not found`; `Access token is unavailable` | `99f230ec…` |

**Reason, from the licensing client's own log.** These are the lines it wrote during the probes:

- `Ignoring license: this license does not contain any currently valid entitlement nor any revocation
  (LicenseGroupOfflineValidityPeriodIsExpired)`;
- `Failed to update license file. [Code: 401] Token not found in cache`;
- `No ULF license found`.

Other facts:

- The entitlement licence file was last written 2026-10-02 (`UpdateDate` 2026-11-01T10:18Z). The script records only
  these dates; no licence content, token or account data was copied.
- `license.unity3d.com` was reachable (HTTP 200), so this is not a network outage.

The licence's offline validity has lapsed, and there is no signed-in session that could refresh it.

**Possible side effect of check #4 (disclosed).** At initialisation the Hub's auth service logged:

- `Token exists, trying to login from existing tokens`;
- then `Logging out`.

The Hub log shows the same pair on 2026-10-02 at a GUI start. That time a sign-in followed about 50 s later; I infer that
sign-in was manual. So the stored session was most likely already unusable.

The licence failures in checks #2 and #3 came **before** check #4. Even so, the Hub may have cleared stored tokens while
logging out.

I did **not**:

- sign in, or enter any credential;
- activate or return a licence;
- change Hub or editor settings;
- install or repair an editor.

**Human action needed before any rerun.** On Main:

1. Sign in to Unity Hub.
2. Refresh or activate the Unity licence, so that `-batchmode` gets the `com.unity.editor.headless` entitlement.
3. Optionally, repair or remove the incomplete 6000.2.12f1 install.

To confirm, rerun the preflight script into a new `--out-dir`. It reports `READY_FOR_DECLARATION` once an editor creates a
project in batchmode. This is an environment step, not an architecture decision.

## 4. W127 / W126 input authentication (read-only)

The W127 evidence manifest `C:/Users/dna10/animcv-w127/results/evidence_manifest.json` has SHA-256 `8d70fa40…4a2c2`, equal
to W127 Worklog §12. For each state, all of these hold:

- manifest schema `animcv_exp_fbx_interchange_package_v1`;
- `seal_ok`;
- FBX SHA-256 equal to the manifest;
- both files equal to their entries in the W127 evidence manifest.

| State | Take | Frames | FBX SHA-256 | Manifest digest | FBX bytes | Keys | Source `.blend` SHA-256 |
|---|---|---|---|---|---|---|---|
| S1 baseline | `root\|AnimCV_EXP_1284d6ce91af` | 1..240 | `103f667e…` | `c95163c5…` | 1,148,220 | 112,320 | `0803ab2b…` |
| S1 W126V (AUTHORIZED_AT_COMMIT) | `root\|AnimCV_EXP_674ba278e9a0` | 1..240 | `f8b842e7…` | `07513622…` | 1,148,204 | 112,320 | `7e2ab165…` |
| S1 head (exact revert) | `root\|AnimCV_EXP_73dcd3dde3a6` | 1..240 | `103f616d…` | `738238b7…` | 1,148,236 | 112,320 | `1ec3cda7…` |
| S3 baseline | `root\|AnimCV_EXP_07667f65b21d` | 1..154 | `34c34a7f…` | `1381fbf8…` | 898,396 | 72,072 | `e1cfbae4…` |
| S3 W126V (AUTHORIZED_AT_COMMIT) | `root\|AnimCV_EXP_fd249fbb203f` | 1..154 | `b23f4cd5…` | `eac50645…` | 898,396 | 72,072 | `702ead26…` |
| S3 head (exact revert) | `root\|AnimCV_EXP_be9a1f09107c` | 1..154 | `9eb4c3bc…` | `d0600cff…` | 898,412 | 72,072 | `0de50493…` |

Every package is 30 fps with `fps_base` 1. The rig is armature `root`, 51 bones. The producer was Blender 5.1.0 with
io_scene_fbx 5.15.0. Full hashes are in `preflight.json`.

The script read only:

- the W127 **package** directories;
- the W127 evidence manifest.

It did not open the S1/S3 run copies (`runs/S1_replay`, `runs/S3_replay`), any `.blend`, or any W127 evaluation JSON.
Nothing was exported, imported, accepted or reverted. The run-immutability check (before/after the export/import) is
therefore **NOT APPLICABLE**: there was no operation to bracket.

## 5. Required evidence A–E: status

| Item | Status |
|---|---|
| A. Real Unity consumer, native import control, optional Editor adapter, trust boundary | **NOT RUN** (licence block) |
| B. Independent Blender↔Unity integer/subframe sampling, coordinate/timebase transform, pin residual | **NOT RUN**. No transform was derived or frozen. |
| C. Discriminating controls (held / smooth / mixed / static-rotated / raw-vs-package / bad manifest / ambiguity / repeat / killed or concurrent Editor) | **NOT RUN** |
| D. S1/S3 Unity playback and visual A/B | **NOT RUN**. Inputs authenticated only (§4). |
| E. Cost | Unity import cost unknown. Package sizes and key counts as in §4 (from W127). |

No Unity-side code was written: no consumer project, Editor script, adapter or evaluator. With no executable engine,
writing one could not have been tested. It would have been untested scaffolding outside the Batch's evidence standard.

## 6. Tests

| Set | Result |
|---|---|
| Preflight script, real Unity + Blender + W127 files | RUN once: verdict `DATA_BLOCKED_UNITY_LICENSE`, packages OK |
| Unity Editor batchmode tests | **NOT RUN** (blocked) |
| W124–W127 focused, isolation, gate and FBX regressions | **NOT RUN**. No `src/` or `tests/` file changed (`git diff 6b19795 -- src tests` is empty), so there was nothing to regress against. W126 official 40/41 and W127 official 29/31 (supplementary 31/31) stand as recorded. |
| Default CLI/GUI checks | **NOT RUN**, same reason |
| Independent Reviewer execution | none |

## 7. Negative findings

- **Unity is not usable on Main as configured.** "Installed" in W127 §6 meant the files are present. The licence's offline
  validity expired after the last Hub sign-in (2026-10-02). Unattended `-batchmode` requires the `com.unity.editor.headless`
  entitlement, and this machine cannot obtain it without a human sign-in.
- **6000.2.12f1 is an incomplete install.** There is no `Unity.exe`, and the Hub does not list it.
- **The Hub probe (#4) may have logged the Hub out**, if the stored session was not already invalid (§3).
- **Still open from W127:** whether Unity honours the CONSTANT `KeyAttrFlags`, the per-Action take time origin, the
  BindPose rest and the key density.

## 8. Unresolved uncertainties

- Whether the Hub session was already invalid before check #4. This is inferred from the identical 2026-10-02 log pattern,
  not proven.
- Whether the licence type, once refreshed, includes the headless entitlement. Usually yes for Personal, but not verified.
- Every Unity-side question in the Batch: native Generic import fidelity, held/smooth behaviour, axis/scale/time mapping,
  and whether an Editor adapter is needed.

## 9. Reproduction

Git Bash, repository root; any revision ≥ `910d778`. `--out-dir` must not exist.

```
PY=C:/Projects/AnimCV/.venv_build_full/Scripts/python.exe
PYTHONPATH=src $PY -X utf8 scripts/w128_unity_generic_preflight.py --out-dir C:/Users/dna10/animcv-w128/preflight_<n>
#  -> "verdict": DATA_BLOCKED_UNITY_LICENSE | DATA_BLOCKED_PACKAGES | READY_FOR_DECLARATION
```

The script only:

- launches each Unity editor with `-createProject` into its own new directory;
- reads the W127 packages and evidence manifest, and the licence file's dates;
- runs `blender --version`.

## 10. Changed files, resources, rollback

**Repository:**

- new `scripts/w128_unity_generic_preflight.py`;
- new: this Worklog;
- `docs/README.md`: one dated index line, status DATA BLOCKED.

Unchanged:

- `src/`, `tests/` and the W127 package, producer and consumer;
- the stock `fbx` path, CLI and GUI;
- the normative architecture documents;
- `docs/EXPERIMENTAL_VIDEO_ACTION_WORLD_PIN.md`, the master plan and framework status (no capability was demonstrated);
- the automation scripts and `docs/next_batch.md`.

**Outside the repository:** `C:/Users/dna10/animcv-w128/` (`envcheck/`, `preflight/`). It holds about 10 small logs and
partial Unity project stubs, 77 KB in total.

**Resources.** Each Unity probe took under 3 s, except the GUI probe (killed at 150 s). No GPU was used.

**Rollback:** delete the script and `C:/Users/dna10/animcv-w128/`. No AnimCV run or W127 artifact was written.
