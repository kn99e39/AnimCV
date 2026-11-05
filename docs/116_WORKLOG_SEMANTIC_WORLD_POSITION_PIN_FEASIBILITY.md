# Worklog — Semantic Constraint Architecture: World-Space Joint Position Pin Feasibility (2026-11-06)

Batch 116. Evidence-only scratch prototype on `arch/single_frame_first`, run unattended on Main (local RTX 5080 host;
no GPU inference was needed). No `src/` module, schema, model, threshold, rig, rest pose, FK/IK, FBX/Blender exporter or
importer, historical renderer, W102–W115 script, test, Worklog, CASE or artefact was changed. `docs/next_batch.md` was
read only.

## 1. Official verdict and scope

**Verdict: CASE B — PARTIAL SUPPORT** (frozen rule, `results/decision/decision.json`, `bcf6ee3a…`).

- **Validity gates V1–V9: all pass.** Authentication, parent reproduction, FK calibration, bindings, per-frame
  independence, meaningful edits, ground-truth soundness, repeatability, and the solver-claim-vs-evaluated-output control.
- **Minimum-guarantee gates GC1–GC5: all pass.** The contract expresses, binds, measures and safely rejects user
  constraints, and the commit is sound: no unsafe accept, no wrong failure class, refs unchanged on every reject, source
  unchanged.
- **Execution gates: GB1, GB2 and GB3 pass. GB4 and GB5 fail.** Both failures are in my declared programme and gate
  code. Neither is an observed resolver limitation or a failure of the semantic representation:
  - **GB4 (lineage) fails because declared step P5 could not run.** P5 was declared as "commit the smallest *REJECTed*
    `RB_neg` frame". `RB_neg` turned out to be feasible and accepted on 199/199 frames, so P5 had no input. All 138 side-bend
    rejections fell in `RB_pos`. My prediction said only "the sign that moves the shoulder away from the hand" would be
    infeasible; the programme hard-coded the sign. The rest of GB4 holds: P3 and S2 are `STALE_PARENT`, and every lineage
    id re-verifies.
  - **GB5 (atomicity) fails in the gate code, not in the commit.** The gate code counts a frame as passing only when its
    resolver status is `SOLVED`. The capture frame of an interval hold (RH1 frame 109, K8 frame 7) is already satisfied, so
    its status is `NO_CHANGE_NEEDED`. The gate therefore flags RH1 and K8 as inconsistent. The commit rule, and the gate's
    own declared text ("ACCEPT iff all its frames pass"), treat that frame as passing: its residual is 0.
- **Mechanical alternative, not assigned.** If both defects were repaired, the rule would output A. I do not assign A.
  The programme and the gate code were frozen before outcomes, and repairing them now would be an outcome-driven change.
- **Labelled post-hoc checks (§11; no CASE effect):**
  - A gate faithful to the declared GB5 text is consistent for all 6 multi-frame requests.
  - A P5-equivalent run using the first rejected `RB_pos` frame (frame 0) is REJECTed, and the head is unchanged.

So the B is about guarantees left unresolved or mis-checked by my own instrument. It does not come from feasible
requests the resolver could not satisfy: those number 0 of 736.

**What this Batch established, by evidence level.** All of these are separate from the CASE.

| Level | Finding |
|---|---|
| Implementation | A scratch WORLD_POSITION_PIN contract exists. It is declarative and solver-independent, with target provenance, binding identity, authorised DOF, parent identity, hard status and an explicit tolerance. It has separate binding, resolver, Blender-only evaluator and atomic commit/reject steps. |
| Synthetic contract correctness | All 31 synthetic requests returned their declared decision and failure class (31/31). That includes 25/25 declared rejections. |
| Synthetic kinematic feasibility | 5/5 feasible positives were accepted (K1, K2, K2b, K6b, K8) with evaluated residual ≤ 9.6e-5 m. Every GT-infeasible request was rejected as INFEASIBLE_REQUEST. |
| Real-rig Blender output | Across 995 per-frame positive requests, 736/736 GT-feasible frames were accepted. Accepted residual was ≤ 0.98 mm, and 734/736 were ≤ 6.8e-7 m. 258/258 GT-infeasible frames were rejected. Only the two authorised arm channels changed, with 0 unintended channel changes, 0 outside-domain point changes and 0 root moves. |
| Authentic replay | The real W102/W105 clip (199 frames, 50 fps) is GT-free. Pins validate rig output only, never the true 3D motion. |
| Animation usability | Pins are met with large compensating elbow changes (up to 34° for a 5° side bend), occasional straight-arm solutions and no temporal smoothing. Not production quality (§9). |
| Architecture viability | Bounded partial support only. The separation of intent, binding, solver and evaluator held in every tested case, but the experiment covers one constraint type, one chain and one rig. |

## 2. Historical baseline fingerprints

- **Starting commit:** `88ef04ea2c7db3cc6914cabe7c7123060cf47ddf`, equal to HEAD at start and to `origin/arch/single_frame_first`.
  The remote showed no divergence at start.
- **Instruction blob:** `docs/next_batch.md` = `3adfda964525cdd02bd315b07249087a102dd790`. It was verified before any change
  and was read only.
- **Main-local lock:** batch 116, this blob and this start commit.
- **Historical Worklogs:** the W113, W114 and W115 blobs are unchanged and their COMPLETE markers are present. The W115
  frozen CASE is D.
- **Inputs**, each re-hashed and matched to its manifest:
  - W102 observations/predictions `6784357f…`;
  - clip `0dda8a5f…`;
  - W105 manifest `cb0eca93…` and, bound through it, `video_cases.json`, `video_sources.npz` and `render_clip_audit.json`;
  - rig `BaseRig_Skinned.fbx` `c2947ba7…`;
  - rest snapshot `baserig_blender_rest.json` `1dcdc826…` (snapshot digest `8519d243…`);
  - W115 manifest `5988f4fb…` and, through it, the W115 decision, `posed.json`, `restq.json` and `blender_compare.json`.
- **Parent Animation State v0** = a copy of the W105 `real_clip` FIXED slot (M_Main, S0-only configuration). Its state id
  is `26aa62f4…d134e9`.
  - Keyed in Blender with the historical renderer's rules, it reproduces the W105 FIXED Action value digest
    `4d168d95234cc46e…` exactly, with 32 F-curves.
- **Synthetic parent s0:** state id `6f5bbb00…5e68`, a declared two-bone chain on a vertical base. Rig spec sha is in
  `preflight/syn_probe.json`.
- **Runtime:** Python 3.11.9, numpy 1.26.4, Blender 5.1.0, Windows 10.0.26200.

## 3. Semantic constraint schema and semantics

`w116_semantic_constraint_v1` (`scripts/w116_semantic_pin_base.py:make_pin`). Its fields:

- `constraint_id`;
- `type`: only `WORLD_POSITION_PIN` is supported. Orientation, motion, contact, edit-region and soft-style types are
  listed and rejected as `TYPE_NOT_IMPLEMENTED`;
- `semantic_joint` and `binding_id` (a verified binding);
- `coordinate_frame`: only `SCENE_WORLD`;
- `unit`: only `m`;
- `target`;
- `target_source`, which is either:
  - `CAPTURED_FROM_EVALUATED_STATE`, with state id, frame, binding and evaluator; or
  - `CAPTURED_PLUS_DECLARED_OFFSET`, which adds `offset_m`;
- `frames` (an evaluation frame or a declared interval);
- `hard` (only `true`);
- `tolerance_m` (only the declared 0.001);
- `authorized_channels`;
- `root_permitted` (`true` is rejected: root authorisation is not approved in W116);
- `parent_state_id`;
- `target_digest`, which binds target, source, binding and frames;
- `constraint_digest`.

**Contract validation never solves.** It recomputes both digests and recomputes the target from its declared source,
bit-exact against the Blender-evaluated capture table. It also checks:
- binding existence and joint/rig match;
- frame ∈ state frames;
- authorised ⊆ state channels;
- no root or `__object__` authorisation;
- no overlap between authorised channels and user-edit channels, so a solver may never override the user's edit;
- consistent authorisation across the pins of a request;
- conflicting hard pins: the same binding, overlapping frames, and targets more than 2·tol apart.

**Failure classes** are kept distinct: `INFEASIBLE_REQUEST`, `UNSUPPORTED_BINDING`, `SOLVER_FAILURE`,
`CONTRACT_VIOLATION`, and `INVALID_EXPERIMENT`, which is used for no-op evidence.

**Acceptance criterion.** r = ‖P_world_after − P_world_target‖ ≤ 0.001 m. P_world_after comes from Blender's evaluated
Armature Object `matrix_world` × posed bone point. This was declared before outcomes as a provisional diagnostic target,
**not** a production tolerance.

## 4. Rig binding and world-coordinate definitions

**WORLD** = the Blender scene world: Armature Object location, rotation (`rest_q · objq`) and scale 0.01 on the cm FBX
rig, times the posed head/tail. It is not reconstructed physical space.

**Bindings.** Each semantic joint has a declared list of candidate rig points. It is BOUND only if all candidates exist,
coincide at rest (≤ 1e-4 armature units) and coincide on every one of the 199 Blender-evaluated parent frames (≤ 1e-6 m).
The calibration was run on the unedited parents.

| Semantic joint | Declared candidates | Result |
|---|---|---|
| left_wrist | lowerarm_l.tail ≡ hand_l.head | BOUND (gap 0 / 0) |
| left_elbow | upperarm_l.tail ≡ lowerarm_l.head | BOUND (0 / 0) |
| left_shoulder | clavicle_l.tail ≡ upperarm_l.head | BOUND (3.2e-5 cm / 2.5e-7 m) |
| right_wrist | lowerarm_r.tail ≡ hand_r.head | BOUND (0 / 0) |
| right_shoulder | clavicle_r.tail ≡ upperarm_r.head | BOUND |
| pelvis_root | spine_01.head | BOUND |
| right_elbow | upperarm_r.tail vs lowerarm_r.head | **AMBIGUOUS**: 1.36e-4 cm at rest and 1.8e-6 m evaluated, both just above the pre-set thresholds. Fail-safe, not used. |
| thorax | spine_05.tail vs neck_01.head | AMBIGUOUS (4.2 cm), as declared |
| left_hand | hand_l.head vs hand_l.tail | AMBIGUOUS (3.8 cm), as declared: a name match is not a joint |
| nose | none | MISSING |

**Controllability (from the bound dependency chain ∩ state channels):**
- **left_wrist:** `__object__`, `spine_01`, `upperarm_l`, `lowerarm_l`.
- **right_wrist:** only `__object__` and `spine_01`. The right-arm bones exist in Blender, but no right-arm channel exists in
  the Animation State: it is unkeyed rest, with no source owner. This is an actual missing controllable channel, distinct
  from the available left-arm chain.

**Coordinate probes** on frame 0, with the Action detached:
- Translating the Armature Object by +0.1 m x shifts every evaluated point by exactly that (max error 2.4e-8 m), and a
  WORLD target does not move with it (synthetic C11, §6).
- A local `lowerarm_l` rotation moves only the 3 points it should: lowerarm_l.tail, hand_l.head and hand_l.tail.
  `upperarm_l` moves 5 points, and the parent `spine_01` moves 29 points, including the whole right arm and the neck.
- World bone length = rest length × 0.01, with max error 1.6e-7 m.

**FK model calibration.** The resolver's own Python FK (rest snapshot + binding) differs from Blender by ≤ 1.31e-6 m (real)
and ≤ 4.4e-8 m (synthetic) on every point and every frame of the parents. The gate was 1e-5 m.

## 5. Candidate solver and declared degrees of freedom

The resolver (`resolve_frame`) works on a copy of the control state's values. It never reads the immutable evidence or
the parent file.

**Primary resolver: bounded analytic.**
- `two_bone`: authorised = {proximal, distal} of the pinned point's chain. Law of cosines with the *control* elbow as
  pole. The reach band comes from the declared elbow-flex limit [0°, 150°]. Unreachable targets are clamped and returned
  as `UNREACHABLE`.
- `aim_distal` and `aim_rigid`: single-bone sphere tests.
- `two_pin_chain`: elbow + wrist pins.
- Any other DOF/pin structure returns `RESOLVER_UNSUPPORTED`.

**Secondary resolver:** bounded CCD, swing-only, single pin, with a declared `max_iter` and a 1e-4 m internal criterion.

**Declared authorisations:**
- Real: {`upperarm_l`, `lowerarm_l`}.
- Never authorised: user-edit channels and the root (`__object__`, `spine_01`, thighs).
- There is no retry, no relaxation and no secondary heuristic.

**Feasibility ground truth** (`gt_feasibility`) is computed independently. It uses *Blender-evaluated* control positions
and Blender rest lengths, with the reach band expanded by the 1 mm criterion. Margins within 1e-5 m are BOUNDARY and are
not counted.

**User edits** (declared; per-frame on all 199 frames):
- `RA_pos` / `RA_neg`: a `spine_01` local-X side bend whose angle moves the unpinned left wrist by exactly 0.01 m.
- `RB_pos` / `RB_neg`: a fixed ±5° side bend.
- `RC_raise`: no channel edit; the target is the wrist + 0.01 m world Z ("raise the hand 1 cm").
- Negatives:
  - `NR1`: right-wrist pin under the RA_pos edit, with left-arm DOF;
  - `NR2`: right-wrist pin authorising `upperarm_r`/`lowerarm_r`;
  - `NE1`: elbow swing of 0.01 m, with only `lowerarm_l` authorised;
  - `NC1`: two wrist pins 0.02 m apart;
  - `NC2`: elbow + wrist pins under RA_pos.
- Intervals: `RH1` holds the wrist at its frame-109 position over 109–118; `RH2` holds the frame-154 position over 154–181.
- `RW_pos`: RA_pos as one 199-pin atomic request.

Each per-frame request is an independent child of v0. Evaluating one frame of a combined Action equals evaluating that
single-frame state alone: V5 confirms this bit-exactly on frames 0, 113 and 160.

## 6. Synthetic feasibility results

Two-bone chain: upper 0.30 m, lower 0.25 m, lower bent 60°, on a 0.5 m base swinging 15°·sin(πf/20). The armature is
built in Blender. All results are evaluated in Blender.

| Case | Request | Decision / class | Evaluated residual |
|---|---|---|---|
| K1 | wrist +5 cm z, auth {upper, lower} | ACCEPT | 1.9e-8 m (control 0.050) |
| K2 | base +5° edit, wrist held | ACCEPT | 4.5e-8 (control 0.048) |
| K2b | wrist rotated 20° about the shoulder, auth {upper} | ACCEPT | 5.1e-8 (control 0.166) |
| K3 / K3b | beyond max reach / inside the 150° min reach | REJECT INFEASIBLE (UNREACHABLE) | clamped 0.050 / 0.0082 |
| K4 | two contradictory wrist pins | REJECT INFEASIBLE (CONFLICTING_HARD_CONSTRAINTS) | — |
| K4b | elbow + wrist pins inconsistent with bone lengths | REJECT INFEASIBLE (UNREACHABLE) | 1.5 mm |
| K5a | reachable only by moving the root, root locked | REJECT INFEASIBLE | 0.150 |
| K5b / K5d | authorises the base bone / `__object__` | REJECT CONTRACT (ROOT_NOT_PERMITTED) | — |
| K5c | no authorised DOF | REJECT INFEASIBLE (INSUFFICIENT_DOF) | 0.050 |
| K6a | CCD, max_iter 1 | REJECT SOLVER_FAILURE (NON_CONVERGED) | 4.1 mm |
| K6b | CCD, max_iter 64 | ACCEPT (12 iterations) | 9.6e-5 m |
| K7 | already satisfied, no edit | ACCEPT, flagged TRIVIALLY_SATISFIED_NOT_EVIDENCE (not counted) | 0 |
| C2 / C3 / C3b | missing / ambiguous / absent-bone binding | REJECT UNSUPPORTED_BINDING | — |
| C4 / C5a | ARMATURE_LOCAL frame / `cm` unit | REJECT CONTRACT | — |
| C5b | cm values labelled m | REJECT CONTRACT (TARGET_PROVENANCE_MISMATCH) | — |
| C6 / C6b | target altered without / with re-digest | REJECT CONTRACT (digest mismatches / provenance mismatch) | — |
| C9 / C10 | ORIENTATION_PIN / soft pin | REJECT CONTRACT (not implemented) | — |
| C12 | authorised set overlaps the user edit | REJECT CONTRACT | — |
| C13 | constraint bound to another parent | REJECT CONTRACT | — |
| C8 | tampered candidate changes the base by 1° | REJECT CONTRACT (UNAUTHORIZED_CHANNEL_CHANGE, OUTPUT_OUTSIDE_DECLARED_DOMAIN, pin unsatisfied) | 0.0103 |
| C15 | resolver claims SOLVED but returns the unpinned control | REJECT SOLVER_FAILURE (EVALUATED_PIN_UNSATISFIED) | 0.050 |
| C11 | Armature Object moved +0.1 m after capture | REJECT CONTRACT (ROOT_MOVED, pin unsatisfied). The WORLD target did not move with the object. | 0.1000 |
| K8 | hold the frame-7 wrist over 7–13 | ACCEPT atomically | ≤ 4.2e-8 |
| K8b | target 7 cm further out, held over 0–19 | REJECT atomically (frames 9–11 GT-infeasible; 17/20 frames pass) | frames 8/12 straight-arm 0.58 mm |

## 7. Real-rig A/B results

Population: the authentic clip, `real_animcv_observation`-derived W105 S0-only rig output, 199 frames. CONTROL = the
edit without the pin; CANDIDATE = the same edit plus the pin. Resolver, rig and evaluator are identical between the two.

| Family | GT feasible / infeasible / boundary | Accept / reject | Unpinned residual (median, range) | Accepted pinned residual |
|---|---|---|---|---|
| RA_pos (0.01 m) | 78 / 121 / 0 | 78 / 121 UNREACHABLE | 0.0100 (0.009999633–0.010000446) | median 3.7e-7, max 3.4e-4 |
| RA_neg (0.01 m) | 199 / 0 / 0 | 199 / 0 | 0.0100 | max 6.8e-7 |
| RB_pos (5°) | 61 / 137 / 1 | 61 / 138 UNREACHABLE | 0.0268 (0.0024–0.0377) | median 3.4e-7, max 9.8e-4 |
| RB_neg (5°) | 199 / 0 / 0 | 199 / 0 | 0.0268 | max 6.7e-7 |
| RC_raise (+1 cm) | 199 / 0 / 0 | 199 / 0 | 0.0100 | max 6.6e-7 |
| NR1 right wrist, left DOF | 199 infeasible | 0 / 199 INSUFFICIENT_DOF | 0.0157 (0.0112–0.177) | — |
| NR2 right-arm DOF | contract | 0 / 199 CONTRACT (DOF_NOT_IN_STATE) | — | — |
| NE1 elbow swing, forearm only | 39 / 160 | 39 / 160 UNREACHABLE | 0.0197 | max 0.98 mm |
| NC1 contradictory pins | contract-level infeasible | 0 / 199 CONFLICTING | — | — |
| NC2 elbow + wrist under RA_pos | 199 infeasible | 0 / 199 UNREACHABLE | — | — |
| RH1 hold 109–118 | 10/10 feasible | ACCEPT | up to 0.159 | ≤ 5.9e-7 |
| RH2 hold 154–181 | 22 feasible / 6 infeasible | REJECT atomically | up to 0.333 | 6 frames 5.0–21.8 mm |
| RW_pos (199 pins, atomic) | 78 pass | REJECT atomically | — | — |

**Scope of accepted candidates.** Against the control, only `upperarm_l` and `lowerarm_l` stored values changed. Against
the parent, the changes were those plus `spine_01` (the user edit).
- Unintended channel changes: 0.
- Evaluated points changed outside the declared output domain (bit-exact): 0.
- Armature Object matrix and stored root: identical to the control on every accepted frame.
- Max bone-length error: 3.3e-7 m.
- Numerical evaluation error (|solver internal residual − Blender residual|): ≤ 6.3e-7 m. The pins are met to the
  float32 evaluation level, not just within the 1 mm criterion.

**Ground truth vs outcome.** Every rejection of a positive-family frame was GT-INFEASIBLE, except one RB_pos BOUNDARY
frame, which was excluded. Every GT-INFEASIBLE frame's clamped best attempt had an evaluated residual > 1 mm (V7). The real
left arm is nearly straight in this clip: the reach slack is median 4.2 mm, minimum 1.6e-5 m. As a result, one side-bend
sign (the positive one) is mostly infeasible.

**Side effects of the edit itself** (present in both control and candidate; unspecified joints are not locked):
- The RA rule fixes the chord only at the left wrist. Where the wrist lies near the bend axis, the angle grows: median
  1.86°, max 21.1°, with 16 frames above 5°.
- So "0.01 m at the hand" can be a large torso edit. It moved the right wrist by up to 0.177 m and the neck_02 tail by up to
  0.227 m.
- The right wrist, which the user did not pin, moves in every side-bend edit. NR1 shows that it cannot be held with the
  authorised DOF.

**Local vs world judgement** (labelled post-hoc tabulation over frozen outputs):
- RA_pos/RA_neg **controls** keep the left-arm stored local values bit-identical on 199/199 frames. Yet their wrist
  world position is preserved on 0/199.
- The pinned **candidates** change the local arm values on 199/199 frames. Their wrist world position is preserved on all
  accepted frames (78/78 and 199/199).
- Example, frame 113 (source time 2.26 s):
  - `spine_01` is edited;
  - the arm locals are unchanged in the control;
  - the wrist moves from (−0.0029, 0.1569, 0.9988) to (−0.0011, 0.1519, 0.9903);
  - the pinned candidate returns it to (−0.0029, 0.1569, 0.9988), within 4.0e-7 m, by changing both arm channels.

## 8. Authentic observation replay limitations

- The real source is GT-free. No claim is made that any edit is closer to the performer's true 3D motion.
- Video subject identity across frames is **UNKNOWN on 199/199 frames**.
  - RH1/RH2 and RW_pos are cross-frame requests on the **known output rig** (rig sha and parent state id are bound).
  - They make no claim about the video subject.
  - Synthetic K8/K8b are the only cross-frame tests with a known rig identity throughout.
- The low-evidence wrist windows were treated as diagnostic subpopulations:
  - **109–118:** RA_pos, RA_neg, RB_pos, RB_neg and RC are feasible and accepted on 10/10 frames each. NE1 is infeasible on
    10/10.
  - **154–181:** the same five families are 28/28 feasible and accepted. NE1 is infeasible on 28/28.
  - These windows say nothing about whether the source wrist evidence there is correct. The pins only re-pose the rig
    around the frozen hypothesis.
- The source timebase is unchanged and not repaired: the Action is keyed at frame_index+1 at 30 fps, against 50 fps
  source time, exactly as W105/W115 report.
- **Mixed W102 ownership control** (contract level):
  - In `mixed_w102`, the arm channels are owned by J1X. An authorisation expressed as "S0 left arm" resolves to no channel
    and is rejected as UNSUPPORTED_BINDING `SOURCE_OWNS_NO_CHANNEL`.
  - In `real_s0_only`, the same authorisation resolves to `upperarm_l`/`lowerarm_l`. This agrees with W115 MA1.

## 9. Animation quality and temporal observations

Visual review renders are under `results/render/` (skinned mesh) and the post-hoc bone-proxy close-ups are under
`results/posthoc/closeup_*.png`. Each shows the parent, the unpinned control and the pinned candidate, viewed from −Y and
from +X.

- **Large compensating articulation.** The pins are met by re-bending the elbow, and the change is often large:
  - RB_neg: elbow flex change median +22.7°, max +34.0°, for a 5° side bend;
  - RA_neg: median +10.5°, max +19.2°;
  - RC_raise: median +11.0°, max +20.6°;
  - RH1 frame 116: flex goes from 64.7° (control) to 40.4° (pinned).
  This is a trade-off a user may not want. No secondary heuristic was added.
- **Straight-arm (reach-clamped) acceptances:**
  - RA_pos frame 194 is accepted at 0.34 mm with elbow flex 8.6e-5°. The bend plane is undefined there, so a bend flip is
    reported.
  - RB_pos has 1 accepted frame at 0.98 mm, and RH2 frame 166 is at 0.22 mm.
  - These satisfy the declared criterion only because the 1 mm tolerance extends the reach. A fully locked elbow is an
    unnatural pose.
  - Bend flips among accepted frames: RA_pos 1/78, NE1 8/39, and 0 elsewhere.
- **The real parent is itself temporally discontinuous.**
  - The W105 left wrist moves median 2.3 cm per frame, but up to 0.485 m (frame 153→154), with 7 transitions above 0.1 m.
  - The declared 3×-parent-max discontinuity rule therefore cannot flag anything on the real clip (limit 1.45 m). Applied
    post-hoc, it flags nothing for RH1, RH2, RA_neg, RB_neg or RC.
  - Raw steps: at RH1's release (118→119) the wrist steps 0.264 m, against the parent's own 0.309 m. Inside the hold, the
    pinned wrist steps 0 m while the body keeps moving.
  - Real-clip temporal continuity of pinned edits is therefore **not assessable** with this instrument.
- **Synthetic K8.** No discontinuity was flagged. My declared prediction of a release discontinuity was wrong for a
  specific reason: the base motion is symmetric about frame 10, so frames 7 and 13 share the same pose and the release does
  not jump.
  - Inside the hold the elbow steps ≤ 2e-4 m.
  - Entering and leaving the hold, the elbow and wrist steps fall from about 1.1 cm to 0 within one frame. This is an
    abrupt start/stop of motion, with no blending.
- **Bone lengths** are preserved (≤ 3.3e-7 m). There was no root displacement and no joint teleport in accepted output.

## 10. Hard-constraint failures and infeasible cases

- Hard constraints were never relaxed. The only within-tolerance acceptance beyond exact reach is the declared criterion
  itself (§9).
- Feasible requests the resolver failed: **0/736** real and 0/5 synthetic.
- Solver failures: 1 declared control, K6a (NON_CONVERGED at max_iter 1). C15 is the evaluator catching a false SOLVED
  claim.
- Infeasible-but-valid requests are reported as INFEASIBLE_REQUEST. They pass their rejection control and are not counted
  as pin successes: 258 positive-family frames, NR1 199, NE1 160, NC1 199, NC2 199, RH2, RW_pos, K3/K3b/K4/K4b/K5a/K5c and
  K8b.
- Unsupported bindings: right_elbow, thorax, left_hand, nose, syn_hand, syn_finger and syn_nose. All were refused; none
  was guessed.

## 11. Edit lineage and reversal evidence

States are content-addressed, read-only, verified on read, and have parent links. A ref moves only on an ACCEPT whose
expected parent is the head.

- **P1:** RW_pos on `real/whole_clip` was REJECTed (UNREACHABLE on 121 frames); the head stayed `26aa62f4`.
- **P2:** RC_raise frame 0 was ACCEPTed on `real/main`, moving the head from `26aa62f4` to `a50076d1`.
- **P3:** RA_pos frame 60, prepared against v0, was rejected as STALE_PARENT; the head stayed.
- **P4:** revert to `bc2b3a37`. Its values are byte-equal to v0, and its Blender evaluation equals v0 bit-exactly: same
  value digest, and identical world and stored values on all 199 frames.
- **P5:** not executed (no rejected RB_neg frame). This is GB4.
- **Synthetic:**
  - S1: K1 ACCEPTed.
  - S2: K2 against s0 is STALE_PARENT.
  - S3: the tampered C8 candidate is REJECTed and the head is unchanged.
  - S4: revert to `85063297`. Its values equal s0 and its Blender evaluation equals s0 bit-exactly.
- **Lineage audit:** every chain re-verifies.
- **Source guard:** all 14 declared input files are byte-unchanged after every stage, and both parent files still verify.

**Post-hoc (labelled):**
- `p5-equivalent`: the same P2 → P4 replay, then RB_pos frame 0 (REJECT UNREACHABLE) against the reverted head. The head
  is unchanged.
- `gb5-faithful`: all six multi-frame requests are consistent (RH1, RH2, K8, K8b, RW_pos and the per-frame atomics).

## 12. Alternative explanations

- **GB4/GB5 are instrument faults, not architecture faults.** This is the plain reading, and the post-hoc checks support
  it. It is still possible that a differently designed programme would expose a real lineage defect that this run did not
  exercise.
- **"Feasible" is defined by my GT model.** That is closed-form two-bone reach plus the 1 mm tolerance and the declared
  150° limit, on Blender positions. A different feasibility notion (anatomical limits, collision, twist) would reclassify
  frames. V7 only shows that GT-infeasible frames are not satisfiable by the clamped attempt.
- **The real positives are easy for an analytic solver.** They are a single pin on an exact two-bone chain with ball
  joints. The near-perfect residuals say little about multi-constraint or redundant-chain solving.
- **The 0.01 m chord edit is not a uniformly small edit.** Its body-wide effect varies by frame (§7). Population results
  for RA are therefore not "1 cm edits" in a whole-body sense.

## 13. Unresolved uncertainties and non-claims

- **Not shown:**
  - general production viability, or arbitrary rig/chain support;
  - orientation, contact, motion-preservation or soft constraints;
  - multi-pin redundancy;
  - temporal smoothness or animation quality;
  - correctness relative to the performer;
  - FBX/engine survival of pinned Actions (FBX was not exercised);
  - a Generator. Nothing here is learned motion generation; the edits are declared mock edits.
- **Policy questions left to the Architecture Owner:**
  - whether a hard pin may be satisfied by a straight-arm solution;
  - whether large compensating articulation is acceptable;
  - whether unpinned side effects (the right wrist, the head) should be bounded;
  - the local-vs-world choice. W116 demonstrates the divergence again, now on a world pin, but decides nothing.
- **Thresholds:**
  - The right_elbow AMBIGUOUS result rests on a 1 µm threshold set before calibration. It is not a rig fault.
  - The 1 mm criterion is provisional.
- **Narrow data:** one private real clip with one performer, one rig, one arm chain, and one synthetic chain.

## 14. Quantitative accounting

From `posthoc_accounting/accounting.json` (a post-hoc tabulation of frozen outputs):

**Constraints:**
- 2,622 constraints requested: 2,009 with a valid specification and 613 invalid. The invalid ones are NR2 199, NC1 398 and
  16 synthetic contract controls.
- Bound joints: 6 real and 3 synthetic. Unbound: 4 real and 3 synthetic.

**Real positive families** (5 × 199 = 995 frame requests):

| Measure | Count |
|---|---|
| GT-feasible | 736 |
| GT-infeasible | 258 |
| Boundary | 1 |
| Accepted / rejected | 736 / 259 |
| Hard pins resolved / unresolved among feasible | 736 / 0 |
| Criterion pass / fail | 736 / 259 |
| Intended channel changes vs control | 1,472 |
| Unintended channel changes | 0 |

**Other counts:**
- Root movement: 0 accepted frames.
- Source evidence preserved: 14/14 files, at every stage.
- Exact reversals: 2/2 (real and synthetic).
- Solver failures: 1 declared control (K6a). Resolver-unsupported: 0 in valid requests.
- Temporally discontinuous frames: 0 under the declared rule (synthetic K8/K8b); not assessable on the real clip (§9).
- Identity: video subject UNKNOWN on 199/199 real frames; output rig known on 199/199; synthetic rig known on 20/20.

## 15. Changed files, tests and reproducibility

**Changed files** (all new except the one W116-internal edit noted):
- `scripts/w116_semantic_pin_base.py`: contract, binding, FK model, resolver, evaluator, store.
- `scripts/w116_semantic_pin_contract.py`: stages preflight, calibrate, declare, build, evaluate, commit, render, decide
  and manifest.
- `scripts/w116_blender_stages.py`: Blender probe, evaluate and render. After the decision, a post-hoc `--hide-mesh`
  render option was added; its default is unchanged.
- `scripts/w116_semantic_pin_posthoc.py`: post-hoc and labelled, written after the decision.
- `tests/test_w116_semantic_pin.py`.
- This Worklog.

**Revisions:**
- start `88ef04e`;
- `d9bc41d`, the pre-outcome instrument (committed after the declaration `ad4fe0df…` was hashed);
- the Worklog/post-hoc commit, which is the final source revision;
- a final marker-only commit.

**Pre-declaration work, disclosed:**
- Calibration ran on the unedited parents. That was the FK model, bindings, probes and synthetic geometry; K8b's target
  was designed from the parent shoulder path at that stage.
- One test-fixture bug was fixed: a wrong local axis in the test rig.
- A crash-only smoke run of build → evaluate → commit → decide → render ran in a throwaway directory, with stdout discarded
  unread. The directory was deleted. It revealed one filename-split bug in render composition, which was fixed before the
  declaration.

**Tests (AGENT-RUN):**
- `tests/test_w116_semantic_pin.py`: **12 passed** (`w116_tests.xml`). They cover:
  - FK, scale and object translation;
  - descendant-only motion and the output domain;
  - pivoted world swing;
  - two-bone, aim and CCD reachability and non-convergence;
  - constraint round-trip and every validation code: frame, unit, soft, type, root, DOF-not-in-state, provenance,
    ambiguous binding, altered target, conflict, edit overlap;
  - evaluator independence from the solver's claim, including tampered and unauthorised changes;
  - GT classes;
  - store commit, stale rejection, unchanged ref on reject, revert and content binding;
  - the source guard and binding fail-safes.
- **Regressions: 86 passed** (`w116_regression.xml`): `test_w115_validity`, `test_w114_source_preserving_base`,
  `test_w105_rig_utility`, `test_fk_solver`, `test_canonical_pose_parity`, `test_retarget_solver`.
- **Failed:** none.
- **NOT RUN:** the full suite, FBX export/import, any game engine, and Blender-dependent repository suites. Blender was
  exercised only through the W116 stages.

**Reproduction** (PowerShell on Main, repo root; every stage refuses an existing output directory):

```
$py="C:\Projects\AnimCV\.venv_build_full\Scripts\python.exe"; $o="C:/Users/dna10/animcv-w116/results"
$s="ad4fe0df26a175316a060a000297796b02092c60e009325ebee685e8c6bd898f"; $d="--declaration $o/declare/declaration.json --declaration-sha $s"
& $py -X utf8 scripts/w116_semantic_pin_contract.py preflight --out-dir $o/preflight
& $py -X utf8 scripts/w116_semantic_pin_contract.py calibrate --preflight $o/preflight --out-dir $o/calibrate
& $py -X utf8 scripts/w116_semantic_pin_contract.py declare --preflight $o/preflight --calibration $o/calibrate --out-dir $o/declare
& $py -X utf8 scripts/w116_semantic_pin_contract.py build $d.Split() --tag r1 --out-dir $o/build_r1
& $py -X utf8 scripts/w116_semantic_pin_contract.py build $d.Split() --tag r2 --out-dir $o/build_r2
& $py -X utf8 scripts/w116_semantic_pin_contract.py evaluate --build-dir $o/build_r1 --preflight $o/preflight --out-dir $o/eval_e1
& $py -X utf8 scripts/w116_semantic_pin_contract.py evaluate --build-dir $o/build_r1 --preflight $o/preflight --out-dir $o/eval_e2
& $py -X utf8 scripts/w116_semantic_pin_contract.py commit $d.Split() --build-dir $o/build_r1 --eval-dir $o/eval_e1 --out-dir $o/commit
& $py -X utf8 scripts/w116_semantic_pin_contract.py render $d.Split() --build-dir $o/build_r1 --eval-dir $o/eval_e1 --commit-dir $o/commit --out-dir $o/render
& $py -X utf8 scripts/w116_semantic_pin_contract.py decide $d.Split() --build-dir $o/build_r1 --build-dir2 $o/build_r2 --eval-dir $o/eval_e1 --eval-dir2 $o/eval_e2 --commit-dir $o/commit --out-dir $o/decision
# labelled post-hoc (no CASE)
& $py -X utf8 scripts/w116_semantic_pin_posthoc.py --results $o --out-dir $o/posthoc --declaration-sha $s
& $py -X utf8 scripts/w116_semantic_pin_posthoc.py accounting $o $o/posthoc_accounting
& $py -X utf8 scripts/w116_semantic_pin_contract.py manifest --base C:/Users/dna10/animcv-w116 --declaration-sha $s
& $py -X utf8 -m pytest tests/test_w116_semantic_pin.py -q
```

The declaration embeds this host's absolute input paths. Blender is `C:/Program Files/Blender Foundation/Blender 5.1`.

**Key artefacts (SHA-256 prefixes):**

| Artefact | SHA-256 |
|---|---|
| `declare/declaration.json` | `ad4fe0df…` |
| `decision/decision.json` | `bcf6ee3a…` |
| `preflight/preflight.json` | `40ac51e9…` |
| `calibrate/calibration.json` | `95b5f0ac…` |
| `build_r1/real_states.json` (equal to `build_r2`) | `d99632c9…` |
| `eval_e1/real_eval.json` (equal to `e2`) | `7ca4c82e…` |
| `eval_e1/syn_eval.json` (equal to `e2`) | `3d9c0eb7…` |
| `commit/commit_report.json` | `7d324759…` |
| `posthoc/posthoc.json` | `dd39be0a…` |
| `posthoc_accounting/accounting.json` | `b91e36eb…` |

`evidence_manifest.json` = `117645586c1ab77450098ac5cde2f3ebca4ddc03bb2d14fbd425b9da168cc4ac` (246 files), written before
this Worklog.

**Completed and not completed.**
- **Completed:** every Required Evidence part (1–6) was executed.
- **Not completed in the declared sense:**
  - lineage step P5 on the real ref;
  - a correct GB5 implementation;
  - therefore a non-B CASE.
- **Not attempted:** FBX/engine survival, other constraint types, temporal smoothing, a Generator, and Production
  integration.

**Resource and environment constraints:**
- Each build took 1.7 s, each Blender evaluation about 4 s, and renders about 6 s; commit, decide and post-hoc took seconds. Tests took 0.2 s;
  regressions 7.1 s.
- Storage: 121 MB under `C:\Users\dna10\animcv-w116\`. It is local only and not published. The source clip was not copied.
- Main only. No GPU inference, network, model download, package change, elevated access, or LabServer63/69 use. No
  evaluation GT was read.
- Single branch `arch/single_frame_first`. No branch, merge, rebase, reset or force operation.

AUTOMATION_BATCH_STATUS: COMPLETE
BATCH_NUMBER: 116
