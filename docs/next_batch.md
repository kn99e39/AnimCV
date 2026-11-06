# BATCH 120 — Temporal Compensation Mechanism Attribution: World-Space Guide vs Flex-Floor

DIRECTION

Advance AnimCV's approved Semantic Constraint Architecture toward genuinely usable, inspectable rig animation. Perform ONE bounded, evidence-only causal ablation of the transition compensation mechanism already exercised in W119. Keep the frozen WORLD_POSITION_PIN and user-approved automatic, bounded, visible out-of-pin editing policy.

Do not attempt another unconstrained solver search. Do not treat a corrected test instrument as evidence of improved animation. Do not implement Production systems, new constraint types, a trained Generator, a style subsystem, or a new user joint-quality policy.

PURPOSE

W119 reported a small number of real-rig boundary improvements but CASE D INVALID due to two prospective instrument/declaration errors. Its reported RH1 gain was predominantly a near-straight flex-floor clamp; its RH2 gain appeared to come from world-space wrist guidance at one release event. On synthetic trajectories, Y provided no material benefit over clean X, and one real variant introduced a bend flip. The real clip is GT-free and has pre-existing discontinuities.

The next architectural uncertainty is NOT whether we can fix W119's declaration. It is whether world-space guidance provides an independently useful animation-editing mechanism, distinct from flex-floor regularization, under the same semantic hard pin, rig DOF, transition domain and output evaluator.

If the two effects cannot be separated or the evidence remains limited to one correlated event, report that honestly and do not promote either mechanism.

CENTRAL INTENT

Under identical per-request source, pin, temporal margins, authorized arm channels, rig, solver family and evaluation, can we causally distinguish the animation-quality contribution of (a) world-space wrist transition guidance and (b) near-straight flex-floor regularization, including their interaction?

Do either mechanism or their combination produce repeatable, practically visible improvements in actual Blender rig animation without new hard-pin, footprint, articulation, source-preservation or rollback failures?

This is a mechanism-attribution and bounded-utility test, not a permanent choice of solver, elbow limit or animation-quality policy.

HUMAN ASSESSMENT

Normative product direction:
AnimCV is a Video-to-Animation Generator/Editor, not only a pose estimator or mocap converter. Practical editable rig animation, source/reference fidelity, user intent and production utility are distinct evaluation axes.

Approved architecture:
- Semantic Constraint System is the intention/constraint boundary.
- WORLD_POSITION_PIN is the only operational semantic type in scope.
- The hard pin refers to Blender scene-world metres, NOT recovered physical video world.
- A hard pin cannot be silently relaxed.
- Constraint declaration, verified rig binding, bounded resolver, independent Blender output evaluator, immutable source evidence, versioned edit state and atomic accept/reject/revert remain separate.
- Unspecified joints are not all hard-locked, but solver DOF must be explicit.
- Automatic bounded transition editing outside a pin interval is allowed without per-extension user approval, provided the intended and ACTUAL frame/Bone footprint is truthful, intuitive and reversible.
- No permanent joint-angle, elbow-deviation, straight-arm, contact or smoothness tolerance has been approved.

Frozen evidence:
- W113/W114/W115: CASE D, preserved.
- W116: CASE B. Agent-run 736/736 feasible framewise real-rig pins accepted and 258/258 infeasible rejected; GB4/GB5 instruments invalid. Temporal quality not established.
- W117: CASE C. T failed meaningful synthetic benefit; T/TE had a feasible request rejected by convergence; TE modified out-of-pin frames with mixed animation quality.
- W118: CASE D. Family-shared F/T baselines contaminated six segment quality windows. Its footprint evaluator was truthful, but its V4 population contract was invalid.
- W119: CASE D. Clean isolated F_r/X_r baselines were reported, but the zero-budget Z_A control had Y's fixed 8-frame budget rather than X's zero budget, violating A/B fairness. Three pin-tamper controls were correctly rejected as CONTRACT_VIOLATION but incorrectly predeclared as SOLVER_FAILURE. Both are experiment-specification errors, not permission to reassign W119.
- W119 AGENT-RUN exploratory observations: no material synthetic Y-over-X gain (including held-out H_G); RH1 real release improvement associated with flex-floor clamp and guide shortfall up to 0.056 m; RH2 release improvement with reachable world guide; one RH1_RBneg bend flip; real evidence correlated within one clip. The frozen W119 verdict is D, not B.
- W119 agent-reported focused tests 19 and regression tests 128. These have NOT been independently re-executed by the Architecture Reviewer.

Reviewer source inspection:
W119's endpoint_transition_base.py always calls budget_plan(..., XB.M_MAX) for Y, whereas X uses m_max=0 for Z_A. W119 ab_fairness compares budget_frames and pin digests; pin tamper creates PIN_VALUES_DIFFER_FROM_F, whose CONTRACT_VIOLATION precedence is visible in decide_y. Do not carry those incorrect prospective expectations into W120.

HUMAN-READABLE SUMMARY

Direction:
Use a prospectively declared matched-factor ablation to separate world-space wrist-guide effects from near-straight flex-floor effects, on isolated per-request U/F/X baselines and actual Blender Actions.

Purpose:
Decide whether endpoint-space compensation adds meaningful, controllable animation utility, or whether observed gains are explained by a simple articulation guard, a single release event, or measurement/design artifacts.

Central Intent:
Does the guide improve output independently of the floor, does the floor improve output independently of the guide, and what does their combination cost in elbow motion and edit footprint?

Do:
- Authenticate historical inputs and reproduce clean isolated per-request baselines.
- Freeze a matched two-factor design before reading new outcomes.
- Run outcome-independent instrument controls, including zero-budget and failure-class precedence.
- Evaluate all four factor combinations with identical pins, margins, authorized channels and solver family.
- Compare to unchanged X and W119 Y as historical external references.
- Independently measure Blender world positions, actual footprint, raw vs source-relative temporal changes, articulation and visible motion.
- Preserve negative results, family correlation and exact reversible state lineage.

Do NOT:
- Retroactively fix/re-decide W118 or W119.
- Add a new optimizer, tuned transition margin, unapproved secondary constraint, contact policy or Production joint limit.
- Change source, rig, FBX, timebase, production modules, existing schemas or old Worklogs.
- Promote a single real event to generalized animation-quality proof.
- Ask the Agent for next-work recommendations.

Decision enabled:
Whether the WORLD-GUIDE and/or FLEX-FLOOR mechanism deserves continued architectural consideration, under what bounded conditions, and whether available evidence can support that distinction at all.

HISTORICAL BASELINE

Repository: kn99e39/AnimCV
Existing branch ONLY: arch/single_frame_first

Observed completion HEAD before issuance:
e0ccf74acb578db0f643571e4edd8f4f2b2ee36f

Verify the actual remote and local HEAD before doing anything. Never assume this reference is current. Do not create, switch, merge, rebase, reset or delete branches. Do not force-push.

Read and honor:
- AGENTS.md
- PRODUCT_VISION.md
- Architecture_v3_FramePose.md (normative perception contract)
- Architecture_v2.md (standing downstream MotionGraph, Rig and Blender contracts)
- docs/AUDIT_W111_W115_2026_11_05.md
- W116, W117, W118 and W119 COMPLETE Worklogs
- scripts/w116_semantic_pin_base.py and its Blender evaluator
- scripts/w117_temporal_pin_base.py
- scripts/w118_transition_edit_base.py and relevant visual/footprint logic
- scripts/w119_endpoint_transition_base.py
- scripts/w119_endpoint_transition.py
- W119 focused tests and relevant earlier focused tests
- docs/next_batch.md (verify that this is W120 before executing)

Use the frozen W102 real-observation clip, W105 rig/Action, W116 source and binding contracts, W118 X, and W119 per-request isolated F_r/X_r construction. Preserve original input fingerprints and historical decisions.

The authentic real source is GT-free; tracked video-subject continuity is UNKNOWN. Source observation is 50 fps, historical Action is 30 fps. Use explicit Action-frame semantics and label the mismatch. Do not repair or infer actor identity.

PRESERVE / CHANGE ONLY / DO NOT TUNE

PRESERVE:
- Every W116–W119 Worklog, official CASE, frozen declaration, test, script, output manifest and original evidence.
- Source video/observations, model weights, checkpoints, pose estimates, rig rest pose, source Actions, frame timing, FBX pipeline and original Blender files.
- W119 clean per-request U_r/F_r/X_r reference and historical Y output as comparison references.
- W116 1 mm experimental hard-pin criterion; all other hard ownership, locality, root and source guards.
- Existing Production code and schema.

CHANGE ONLY:
- New isolated W120 scratch scripts and focused tests.
- New prospectively declared ablation candidate records and generated scratch Actions.
- New W120 independent Blender evaluations, timelines, visual evidence and Worklog.

DO NOT TUNE:
- W117 or W118 smoothing/margin parameters.
- W119 guide weight, elbow-pole rule, solver tolerances, flex-floor formula, joint limits or post-hoc improvement metrics.
- Source and Action frame rates or sample mapping.
- Any historical threshold or frozen verdict.
- Any parameter after seeing outcome evidence.

A diagnostic threshold used in W119 may be reused as a labelled experiment-only comparator. It is NOT a Production policy.

REQUIRED EVIDENCE — PART 1: VALIDITY-FIRST PREFLIGHT

Before generating new ablation outputs, reproduce and verify:
1. The W119 official D decision, manifest and Worklog markers unchanged.
2. W118 family-shared baseline contamination, as historical evidence only.
3. W119 isolated F_r and X_r construction for every included request.
4. Correct source, rig, parent, U, pin, DOF and Blender Action identities.
5. Actual planned/authorized temporal domains for every request, including zero-budget Z_A.

Preflight the W120 instrument using synthetic, outcome-independent fixtures:
- A zero-budget request MUST have identical empty transition budget in every factor combination; it must not inherit the global M_MAX=8.
- An intentionally unequal budget MUST be rejected as an A/B contract violation.
- Pin-value tampering that also breaks F equality MUST be classified by the actual declared precedence (CONTRACT_VIOLATION), not falsely expected as SOLVER_FAILURE.
- An isolated nonconvergent solve with otherwise valid fairness MUST classify as SOLVER_FAILURE.
- A corrupted hard pin must be rejected even if the solver claims SOLVED.
- Wrong U, rig, constraint digest, source state, authorized Bone, margin, display footprint and stale parent must be independently rejected.
- Negative controls must exercise combined error conditions and explicitly test precedence, not only one error at a time.

Freeze a W120 error-class precedence table and test it before the outcome declaration. If instrument expectations contradict the actual contract, STOP before running the experimental population; do not repeat W119's avoidable D.

If historical outputs are missing, report NOT AVAILABLE. Do not fabricate them or silently use family-shared F as a substitute.

REQUIRED EVIDENCE — PART 2: MATCHED FACTORIAL MECHANISM

Use a minimal two-factor design. The Agent may implement a new isolated W120 wrapper, but must preserve the W119 analytic two-bone solver family, elbow pole reference, weight function, authorized DOF and matched X margins.

Factor G (wrist target):
- G0: the world-space wrist trajectory evaluated from the clean per-request X_r at each transition frame. This is a rotation-offset-derived target, not a claim that re-solving it reproduces X's exact joint rotations.
- G1: W119 world-space wrist guide G(k) = W_U(k) + w(k) [W_F(b) - W_U(b)].

Factor F (additional flex floor):
- F0: no W119 extra min(flex_U, flex_F(b)) floor; preserve the W116 physical reach/joint-limit band. Never lengthen bones or allow invalid articulation.
- F1: W119 extra flex-floor rule, unchanged.

Run all four combinations:
- G0F0
- G0F1
- G1F0
- G1F1

The same analytic solver, pole rule, branch-ambiguity handling, numerical tolerances, pin frames, side margins, root restriction and authorization must apply to all four. Do not choose different margins or different solver parameters per variant.

G1F1 should reproduce frozen W119 Y values/statuses on comparable non-Z_A cases. Report any differences. Z_A is a special zero-budget contract control, not evidence of W119 parity. The historical W119 Y remains unchanged and historically REJECTED on Z_A.

X_r is an EXTERNAL unchanged baseline, not identical by definition to G0F0. Independently quantify any G0F0-versus-X difference in wrist, elbow, rotations and bend branch. If the reference trajectory cannot be reconstructed under the shared solver, explicitly report the limitation and do not attribute that difference to the G factor.

The guide is a soft transition target, not a new hard pin. Record reachable, unreachable, floor-clamped and ambiguous cases, and the actual Blender guide shortfall. Never label an unreachable guide as accurately followed.

Pin frames must remain bit-equal to F_r. No hidden fallback to X, F or U. A valid infeasible request must reject with a distinct reason; an expected rejection is not a successful edit.

REQUIRED EVIDENCE — PART 3: PREDECLARATION AND POPULATION

Before seeing any W120 factor output, freeze and hash:
- Factor definitions and exact shared solver code revision.
- Source/rig/Action fingerprints.
- All requests, frames, pins, authorized channels and X margins.
- A/B fairness and failure-class precedence.
- Which comparisons constitute the primary mechanism test.
- Quality metrics, diagnostic materiality criteria, failure guards and CASE rule.
- Visual-review procedure and required examples.
- Expected negative-control outcomes.
- Reproducibility/repetition plan.

Use all relevant existing W119 synthetic evaluation cases, including H_A, H_B, H_C, H_E and H_G; retain near-singular, zero-margin and infeasible controls. Use a predeclared additional synthetic hold only if its trajectory is selected from parent/source geometry without viewing any W120 factor outcomes. No post-hoc successful-case selection.

Use the authentic W119 RH1 and RH2 real-rig families, including the RBneg bend-flip counterexample, the clean previously contaminated segment windows and low-evidence 109–118 and 154–181 windows. Retain zero-transition and rejected cases. Treat RH1 variants and DJ_R as correlated with their constituent windows; do not count them as independent clips or independent events.

If a requested factor combination is invalid or infeasible, retain its result as REJECTED; do not substitute another variant or silently exclude it from denominators.

REQUIRED EVIDENCE — PART 4: CAUSAL ATTRIBUTION

For each eligible request, independently evaluate Blender world positions and compute the two factor contrasts and their interaction on identical frames:
- Guide effect at F0: G1F0 minus G0F0.
- Guide effect at F1: G1F1 minus G0F1.
- Floor effect at G0: G0F1 minus G0F0.
- Floor effect at G1: G1F1 minus G1F0.
- Interaction: whether the joint effect differs from the sum of the individual effects.

Report both magnitude and direction, with request-level and correlated-family accounting. Do not pool raw effect sizes across unrelated scales without explaining the normalization.

Compare all four to unchanged X_r and historical W119 Y for context, not as a substitute for the factorial contrasts.

Measure:
- B1 edit-induced boundary wrist compensation step relative to U.
- Raw wrist step relative to U and original source discontinuities.
- D2 elbow compensation step and elbow deviation from U.
- D3 upper/lower arm rotation-step excess.
- Bend-plane branch flips and near-straight frames, including whether they are introduced relative to U.
- Wrist-guide reach and actual guide residual.
- Temporal endpoint jumps at pin entry/release and within transition.
- Hard pin residual, root, other-joint collateral and changed frame/channel footprint.
- Whether the same mechanism behaves differently in smooth synthetic versus discontinuous real Action.

Separate RH1's flex-floor/clamped-guide contribution from RH2's reachable-guide contribution. A reduction caused by clamping 56 mm short of a guide must not be described as successful guide tracking.

The historical W119 20% AND 10 mm B1 criterion and its articulation-worsening rule may be reused, explicitly labelled as frozen diagnostic comparators. Do not redefine them to manufacture a positive result.

REQUIRED EVIDENCE — PART 5: RIG, USER CONTRACT AND VISIBILITY

For every accepted candidate, independently evaluate the unchanged Blender rig and confirm:
- Every hard pin within the frozen 1 mm diagnostic criterion.
- Actual changed frames ⊆ planned frames ⊆ authorized budget.
- Actual changed channels ⊆ authorized upper/lower arm channels.
- No unapproved root or outside-arm changes.
- Unmodified user-edit channels and immutable source evidence.
- Explicit guide/floor status and any unreachable target.
- Atomic accept/reject, stale-parent protection and exact reversible state.

Generate a minimal scratch timeline that clearly separates:
1. Hard pin domain.
2. Authorized automatic transition domain.
3. Actual Blender-evaluated changed frame/Bone footprint.
4. Candidate status, guide clamp and rejection reasons.

Show source/U, clean F, X and the four factor variants on comparable frames, with stable labels. The timeline must not mark rejected edits as applied. Truth-check it against raw Blender evaluations. The UI is a diagnostic visualization, not a Production UI or validated human-usability study.

REQUIRED EVIDENCE — PART 6: PRACTICAL ANIMATION REVIEW

Produce synchronized actual-rig playback artefacts (GIF/video where feasible), plus framewise evidence, for:
- RH1 release near frame 122.
- RH2 release near frame 182.
- RH1_RBneg bend-flip case.
- H_G held-out synthetic and at least one near-singular synthetic case.
- A previously contaminated segment whose clean comparator now changes the interpretation.
- A case where no transition exists.

The Agent must inspect motion across frames, not only compare isolated static thumbnails. If actual playback cannot be inspected, label NOT REVIEWED and do not count it as visual utility evidence.

Report whether an apparent wrist improvement causes:
- unnatural elbow/forearm articulation;
- bend flips or near-straight poses;
- motion inconsistent with the source;
- possible torso/hip intersection or contact (UNKNOWN if proxies cannot decide);
- noticeable discontinuity moved to a neighboring frame.

A metric PASS is not a claim of anatomical plausibility, user satisfaction, reference fidelity, style transfer or engine readiness. An agent's review is not independent animator validation.

REQUIRED EVIDENCE — PART 7: NEGATIVE CONTROLS AND REPEATABILITY

Include at least:
- Z_A zero-budget parity.
- One deliberately mismatched budget and one mismatched margin.
- Wrong parent/U/rig/constraint digest.
- Wrong pin value combined with a failed output pin.
- Isolated solver nonconvergence with otherwise valid fairness.
- Unauthorized Bone and root edits.
- Out-of-domain actual change and false display footprint.
- Unreachable guide (disclose clamp) versus infeasible hard pin (reject).
- Pole ambiguity and branch flip detection.
- No-transition request.
- One stale candidate and one exact revert on synthetic and real.
- A negative case in which the proposed variant has a better B1 but worse articulation.
- Repeat builds and Blender evaluations with exact or prospectively justified numerical repeatability accounting.

Do not silently repair failed controls or modify frozen historical outputs. If a new validity instrument fails, retain the failure and stop any unsupported architecture claim.

QUANTITATIVE ACCOUNTING

Report exact denominators by factor and request family:
- Requested/valid/infeasible/unsupported/rejected/accepted.
- Blender hard-pin residuals and failures.
- Factorial comparison eligibility and missing contrasts.
- Budget, planned and actually changed frames/Bones.
- Source-relative and raw temporal effects.
- Guide reached/clamped/unreachable frames.
- New near-straight frames, bend flips and elbow deviations.
- Meaningful improvement and material worsening per frozen comparator.
- Visual inspection coverage and uninspected examples.
- Exact source, lineage, rollback and repeatability checks.

Do not hide a failure behind an aggregate mean. Distinguish synthetic kinematic correctness, real-rig output, authentic GT-free replay, animation usability and Architecture viability.

TESTING

Run focused tests after implementation units. In particular, execute tests for:
- Matched-factor parity and zero-budget contract.
- Predeclared failure-class precedence, including combined errors.
- Source, rig, U, pin and DOF equality across four variants.
- Frozen pin preservation and independent output evaluation.
- Guide reach/flex-floor effect and factor interaction.
- Actual frame/Bone footprint and timeline truth.
- Infeasible/unsupported/failure rejection, no fallback and rollback.
- Deterministic repeated builds and representative Blender replays.

Run relevant W116–W119 focused regressions. Do not run a full suite solely for isolated scratch diagnostics. Report NOT RUN tests accurately. Reviewer has not independently run Blender; do not imply otherwise.

STOP CONDITIONS

Stop with evidence if:
- Source/rig/Action or historical baseline authentication fails.
- Per-request isolated baselines cannot be established.
- A/B factor matching, zero-budget handling or error-class precedence fails before outcome evaluation.
- Factor isolation requires a different solver, pole rule, hidden margin or unapproved DOF for some variants.
- An important constraint cannot be measured on actual Blender output.
- A hard pin, user edit, source immutability, footprint or rollback guarantee is violated.
- A purported positive comparison depends on shared-state contamination or correlated-family double counting.
- The experiment requires a new user joint-quality limit, Production change or trained Generator.
- Results cannot distinguish guide from floor effects under the declared comparison.

Report a valid negative or inconclusive result instead of modifying gates, thresholds or algorithms after seeing results.

DECISION CONTRACT

Freeze the exact objective gates and the official A/B/C/D mapping before observing factor outputs. Apply the following meanings, without retrospective changes:

CASE A — BOUNDED MECHANISM SUPPORT:
All validity and hard contracts pass. The matched-factor results identify a mechanism with repeatable, material animation benefit on predeclared synthetic AND distinct real-rig release populations, with no material new articulation defect, truthful visible footprint and credible motion review. This is bounded evidence, not Production acceptance or real-world generalization.

CASE B — PARTIAL / CONTEXT-SPECIFIC SUPPORT:
All validity and hard contracts pass, and factor contrasts are interpretable, but benefit is limited to particular real/synthetic conditions, mixed with articulation costs, not replicated broadly, or not independently visually established. State which mechanism is supported and where; do not call this general solver success.

CASE C — NOT SUPPORTED:
Valid matched-factor evidence shows no meaningful practical benefit for the mechanisms under test, or a critical new output/constraint failure contradicts the proposed approach. A valid negative conclusion is useful.

CASE D — INVALID:
Source, baseline isolation, factor matching, prospective declaration, measurement, negative-control expectations, repeatability or attribution validity fails so the intended mechanism conclusion cannot be trusted. D supersedes all other CASEs.

Do not assign a mechanical alternative as the official verdict. Do not reclassify W116–W119.

FINAL REPORT

Create a W120 Worklog with:
1. Official CASE, declaration hash and exact gates.
2. Remote HEAD, source/rig/Action and frozen baseline fingerprints.
3. Independent source/code/diff scope and preservation.
4. Outcome-independent preflight and error-class precedence evidence.
5. Matched four-factor mechanism and solver parity.
6. Synthetic and real-rig population/denominators.
7. Blender-evaluated factorial contrasts and interaction.
8. Reach/flex-floor/guide shortfall accounting.
9. Elbow, bend flip, near-straight and temporal quality costs.
10. Source-relative vs raw motion accounting.
11. Actual visible frame/Bone edit footprint and viewer truth.
12. Playback/visual review coverage and limitations.
13. Negative controls, hard failures, exact revert and repeatability.
14. Alternative explanations, confounds and non-claims.
15. Changed files, tests, hashes and reproducibility.

Agent is an Evidence Producer. Do NOT provide Next Work Recommendation, Next Architecture Decision, Proposed Next Batch, or a plan for W121. Do not start W121.

PERIODIC AUDIT CHECKPOINT

W120 is the fifth Worklog after the W115 audit. The independent Architecture Reviewer must audit cumulative W116–W120 Git changes and Worklogs after W120 COMPLETE and BEFORE issuing W121. The Agent must not self-certify that independent audit or alter earlier Worklogs.

COMPLETION CONDITION

W120 is complete when the frozen, controlled evidence lets the Architecture Reviewer determine:
1. Whether clean, source-bound and fair four-factor comparisons are valid.
2. Whether the apparent RH1 and RH2 improvements arise from world-space guidance, flex-floor clamping, their interaction, or an unresolvable confound.
3. Whether any identified effect survives synthetic and authentic real-rig comparison without unacceptable unreported articulation or temporal side effects.
4. Whether hard WORLD_POSITION_PIN, bounded automatic edits, truthful user-visible footprint, immutable source and exact rollback survive all accepted candidates.
5. Whether endpoint-guided temporal compensation has defensible bounded engineering utility, or should remain unsupported/uncertain under the available evidence.

Negative and inconclusive findings are legitimate. Do not expand to Production integration, new semantic constraints, trained generation, FBX/timebase changes, or W121.

AUTOMATION_BATCH_STATUS: ISSUED
BATCH_NUMBER: 120
