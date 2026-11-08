# AnimCV — Product Vision and Animation Authoring North Star

**Status:** Normative product direction, not a claim of implemented functionality.  
**Confirmed:** 2026-10-19.  
**Applies to:** Product decisions, architecture trade-offs, engineering priorities and Agent/Codex task planning.

## 1. Non-negotiable product purpose

> **AnimCV is an animation creation framework. Video-based 3D motion reconstruction is an important input capability, not the final product.**
>
> AnimCV must ultimately enable users to **reconstruct, understand, edit, restyle and author rigged animation**. It should reproduce the **expressive intent and motion style** of a reference when asked, and support creating deliberately different motion when the user asks for it.

A more accurate 3D pose is useful, but it is not sufficient evidence of a better animation. Likewise, a smoother animation can be less faithful if it suppresses meaningful motion. Quality is judged at the level of editable, usable **rig animation**, not solely by joint-error metrics.

This is the project's enduring **north star**. New agents must not silently narrow AnimCV into a pose-estimation benchmark, pure motion-capture replica, or a sequence of increasingly isolated reconstruction experiments.

## 2. Four distinct capabilities

| Capability | User-facing result | Scope |
| --- | --- | --- |
| **Motion Reconstruction** | Recover 3D pose, movement and timing from visual evidence | Video/reference driven; preserve provenance, ambiguity and uncertainty |
| **Animation Reconstruction** | Produce coherent, rig-compatible, editable animation from reconstructed movement | Torso/root orientation, articulation, contact, timing, motion continuity, IK/retargeting |
| **Style Reconstruction and Transfer** | Recover a reference's expressive movement characteristics, or apply a chosen style to existing motion | Preserve relevant action/intent while allowing intentional timing, rhythm, posture and amplitude changes |
| **Animation Authoring / Synthesis** | Create or revise animation directly from user intent, constraints, manual edits or references | Video is optional in the long term; controllability and editable outputs are essential |

These are **product goals**, not claims that all four are implemented. The first two have partial working infrastructure; comprehensive style reconstruction/transfer and direct authoring remain future capabilities.

## 3. Motion fidelity and artistic control are different objectives

A physically faithful gait and a convincing animation gait are not always identical.

The framework must eventually support both:
- **Reference fidelity:** reconstruct the observed motion when the user wants faithful reproduction.
- **Expressive control:** deliberately alter timing, anticipation, follow-through, rhythm, posture, exaggeration, weight, trajectory, or other style characteristics when requested.
- **Constraint preservation:** keep necessary skeletal coherence, contacts, endpoints, continuity and rig compatibility while applying deliberate style changes.
- **Editability:** let an artist inspect, adjust, retime, replace and selectively lock meaningful motion, instead of delivering an opaque pose array or an uneditable dense bake.

Do not treat artistic deviations as reconstruction errors when they are intentional, and do not excuse accidental reconstruction errors as artistic style.

## 4. Intended architecture and ownership

An eventual product path may be represented as:

```text
Video / Motion Reference / User Instructions / Manual Keyframes
              |
              v
Observation + 3D Motion Reconstruction (where applicable)
              |
              v
Motion Understanding + Animation Semantics
              |
              v
Editable, rig-independent Animation Representation
              |
              v
Animation Reconstruction / Style Transfer / Motion Authoring
              |
              v
Constraints + IK / Retargeting + Keyframe Editing
              |
              v
Editable Blender / FBX Rig Animation
```

This is a **directional product architecture**, not an assertion that every block or interface exists today. Do not invent a large new intermediate-representation subsystem merely to satisfy this diagram. Establish its contract incrementally based on demonstrated editing operations.

Existing ownership remains intact:
- **FramePose:** visual observation and reconstruction of canonical 3D geometry.
- **Animation Semantics:** currently limited motion meaning such as root heading and contact evidence; future style/intent semantics require their own explicit design and validation.
- **Motion Graph / animation representation:** preserve rig-independent, time-aware, editable motion information; evolve its contract only when a concrete authoring use case demands it.
- **FK/IK / retargeting / Blender:** apply the motion to an actual rig and export editable animation.

Do not redefine primitive 3D pose accuracy, root/world movement, contact inference and expressive style as if they were interchangeable.

## 4.1 Confirmed architecture direction (2026-11-08)

The human confirmed the following **target architecture direction** on the existing research/development branch `arch/single_frame_first`. This is a product/system direction, **not** an assertion that the complete path has been implemented, integrated, trained, or accepted for Production.

```text
Video / Reference (immutable source, timestamps, provenance)
    -> Geometry Observation + FramePose (2D observations -> 3D estimate)
    -> Pose Hypothesis + Learned Uncertainty (per-frame/per-joint, parallel evidence)
    -> Reliability-Aware Animation Base (source + motion hypothesis + reliability + lineage)
    -> Conditional Motion Generator (future candidate, NOT implemented/authorized)
    -> Semantic Constraints + Rig Evaluator (hard authoring intent and actual-rig checks)
    -> Editable Blender Action / FBX (output, interchange, real animation review)
```

**Ownership and invariants**

- The original video, timestamps, subject/source identity and unmodified observations remain distinguishable from derived 3D pose hypotheses and their possible errors. Inference must not read evaluation ground truth.
- The learned uncertainty/error signal is an **independent, candidate estimate for a specific frozen predictor and joint/frame**, not raw RTMPose detector score, an infallible confidence value, or a hard user constraint. W123 is authorized **only** to test a frozen-model learned assessor, calibration, generalization and GT-free-at-decision utility in isolated scratch outputs. An unproven signal must not be presented as calibrated 3D trust.
- The Reliability-Aware Animation Base is the proposed source-bound handoff; W113-W122 supply **scratch contract/editability evidence**, not an approved Production replacement for MotionGraph or AnimationClip.
- A future Conditional Motion Generator could retain well-supported reference motion and propose completion/variation for poorly supported regions. Its architecture, training, inference policy and Production integration **remain unapproved and unimplemented**. The Pose/Motion Completion requirement in `docs/POSE_COMPLETION_PRODUCT_PLAN.md` remains active.
- Semantic **hard constraints** (e.g. an authorized WORLD_POSITION_PIN) take precedence over any learned soft reference-trust signal. The Rig Evaluator must check actual Blender-evaluated animation, authorized edit footprint, temporal behavior and reversibility. W116-W122 demonstrate limited scratch evidence, not a general-purpose Production Editor.
- Blender/FBX availability is not proof of faithful continuous-time animation or external consumer interoperability. Historical FBX findings W107-W112 remain unresolved in the stock Production export path.
- No future module, schema migration, generator training, new constraint type, solver policy, or branch operation is authorized merely by this diagram. Preserve official historical Worklog CASEs and require a separately bounded decision before implementation.

**Evidence maturity (as of 2026-11-08):** FramePose, legacy MotionGraph/retargeting and some semantics/FK/IK modules exist; the standard CLI/GUI retarget path remains legacy MotionGraph-based. W116-W122 constraint experiments are scratch. W123 is **issued, not reported complete**. Learned Reliability, an integrated Animation Base, Conditional Motion Generator, and full authoring/style functionality must not be claimed as shipped.

## 5. Engineering priority rules

1. **Working framework over research purity.** Scientific-style controlled evaluation is useful to avoid false progress, but publication novelty and perfect causal identification are not project goals.
2. **Heuristics are allowed.** Filtering, rules, blending, constraints and learned methods are legitimate when they measurably improve practical animation and their failure modes, latency and maintenance costs are acceptable. They are not automatically required.
3. **Do not get trapped in a small reconstruction defect.** Reconsider whether a problem matters more than missing end-to-end capability, editing control, runtime contracts or animation usability.
4. **Preserve baselines and evaluation provenance.** Avoid GT leakage into runtime, report what is implemented versus experimental, and keep opt-in defaults when evidence is limited.
5. **Validate the final medium.** Measure joint/rotation accuracy, temporal response and regressions, but also review actual rig animation and the usability of edit operations.
6. **Accept bounded imperfection.** When reconstruction is sufficiently useful to support a higher-level workflow, develop that workflow; do not require perfect pose estimation before any animation-authoring work.
7. **Do not auto-expand scope.** A long-term authoring vision is not permission to implement text-to-motion, a giant generative model, a new DCC editor or a full-body network in the next batch.

## 6. Distinct quality axes and future acceptance

**Reconstruction / animation fidelity**
- Correct and reasonably stable anatomical geometry, torso/root orientation, motion timing and continuity.
- Valid end-effector/IK behavior and appropriately handled contact, occlusion and uncertainty.
- A usable real-video → rig-action path with real timestamps and no privileged inference inputs.

**Animation controllability / authoring**
- A user can change at least one meaningful expressive property (for example motion timing, gait rhythm, arm-swing amplitude or anticipation) while protecting explicitly preserved constraints.
- The result is observable in rendered rig animation, not just in modified coordinates or labels.
- Style can be **reproduced from a reference** or **specified by the user**, with clear distinction between faithful recovery and intentional stylization.
- Edits remain reversible or inspectable, can be applied locally to tracks/segments where appropriate, and yield editable animation curves/keyframes.

Neither axis substitutes for the other. Full product acceptance requires both, even if delivered through stages.

## 7. Roadmap interpretation

Near-term work may continue on genuine blockers to a runtime-safe video → 3D pose → rig animation pipeline, such as GT-free root orientation, native-frame-cadence replay, broader body coverage, completion and reliable export. These are **enablers**, not a redefinition of the destination.

As the pipeline becomes usable, the roadmap must explicitly add bounded **animation editing, reference-style reconstruction/transfer and user-directed authoring** milestones. Their exact representation, UI and algorithm are not yet specified; select them from real user editing tasks and evidence, not by assuming a particular model architecture.

Existing Worklogs and technical architecture remain historical evidence. This document does **not** claim the style/authoring modules, user controls or production integration already exist.

## 8. Document precedence and context persistence

- **This document owns product purpose and long-term success criteria.**
- [Architecture_v3_FramePose.md](Architecture_v3_FramePose.md) owns the current perception / FramePose technical contracts.
- [Architecture_v2.md](Architecture_v2.md) remains the historical/standing technical source for areas not superseded by v3, especially Motion Graph, mapping, retargeting and Blender output.
- [docs/00_마스터_플랜.md](docs/00_마스터_플랜.md) records the current staged execution plan; its older reconstruction-focused gates are necessary but **not sufficient** for the product vision.
- [AGENTS.md](AGENTS.md) instructs future agents to read and honor this vision before scoping work.

**When technical choices conflict, preserve existing measured contracts unless a separately scoped change is authorized. When priorities conflict, use the product north star above to decide what is worth building next.**
