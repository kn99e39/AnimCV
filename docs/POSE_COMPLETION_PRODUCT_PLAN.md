# Pose / Motion Completion Product Plan

## Status

**Deferred required product stage — not abandoned.**

AnimCV must eventually recover usable motion when direct joint observation is temporarily unavailable because of occlusion, out-of-frame motion, or short tracking loss.

The current production-safe behavior is intentionally conservative:

- preserve observation validity / reliability metadata;
- represent unsupported pose evidence as `UNKNOWN`;
- do not silently replace missing evidence with arbitrary interpolation.

This is an interim contract, not the intended final product behavior.

## Why this remains required

AnimCV is intended to be a usable animation-production framework, not only an evaluation pipeline.

A practical video-driven animation system cannot allow an arm, leg, or other joint chain to collapse simply because the subject is briefly occluded or leaves direct observation.

The final requirement is therefore:

> Temporary observation loss should produce a plausible, continuous canonical pose trajectory whenever enough surrounding motion evidence exists.

Exact hidden ground-truth recovery is not required when it is unobservable. The product criterion is a stable and visually plausible animation that reconnects correctly to reliable observations.

## Evidence already established

The bounded experiment in `docs/50_WORKLOG_VLM_RELIABILITY_AND_NONLINEAR_RECONSTRUCTION.md` tested:

1. local VLM observation reliability;
2. deterministic nonlinear gap reconstruction from reliable pose anchors.

That experiment rejected the tested production candidates:

- the current Qwen2-VL-2B reliability backend did not provide sufficiently image-grounded reliability states;
- general Hermite/SQUAD gap filling degraded H0 overall;
- very short 1–2 frame gaps retained a positive reconstruction signal.

Therefore the **tested mechanisms were rejected**, not the Pose Completion product requirement.

## Intended architecture

```
Frame Pose
    ↓
Observation / Pose Reliability
    ↓
Reliable
    → preserve observed pose

Unreliable / Occluded / Out-of-Observation
    ↓
Pose / Motion Completion
    ↓
canonical completed pose trajectory
    ↓
Animation Semantics
    ↓
IK / Retargeting / Output
```

Pose Completion operates on the canonical skeleton before target-rig application.

## Recovery policy by evidence horizon

### Very short dropout

Typical target: approximately 1–2 missing frames.

Use deterministic local motion reconstruction when surrounding reliable poses provide enough support.

Candidate evidence:

- reliable pose anchors;
- position / angular velocity;
- local kinematic constraints;
- root orientation;
- stance / swing evidence.

A deterministic nonlinear curve remains appropriate here, but the exact curve family is not fixed by this document.

### Bounded occlusion

For a longer but still locally constrained hidden interval, reconstruction should use more than two endpoint poses.

Candidate evidence:

- surrounding reliable pose history;
- root orientation;
- contact / stance-swing state;
- limb kinematics;
- observation reliability;
- motion semantics where available.

The objective is a plausible trajectory consistent with the visible motion before and after the gap.

### Long occlusion / prolonged out-of-frame motion

Two-sided interpolation is insufficient when substantial hidden motion may occur.

This case may require a learned motion prior or dedicated motion-completion model.

The system must still preserve uncertainty and may refuse completion when evidence is insufficient.

## Role of VLM

VLM use remains allowed as a future evidence source, but it should not be the primary continuous-transform generator.

Preferred responsibilities:

- visibility / occlusion reasoning;
- coarse motion semantics;
- hidden-motion interpretation;
- ambiguity evidence.

Examples:

- "arm continues raising";
- "hand passes behind torso";
- "leg is in swing phase";
- "foot likely remains planted".

Actual XYZ / quaternion trajectories should remain owned by geometry / motion reconstruction.

The failed Qwen reliability experiment in Worklog 50 does not permanently reject this VLM role; it rejects that tested backend and contract.

## Reliability ownership

Until a better learned reliability source is demonstrated, authoritative runtime evidence remains structural:

- observation validity;
- in-frame state where available;
- detector / tracker state;
- provenance;
- other explicitly measured uncertainty signals.

`UNKNOWN` means "insufficient evidence".

It must never be serialized or interpreted as:

- zero transform;
- zero position;
- automatic linear interpolation;
- valid observed pose.

## Planned project order

The current project should first close the Animation Semantics contract and establish the minimal canonical-pose → target-rig path.

After that, Pose / Motion Completion must be reopened as a dedicated product stage before final end-to-end acceptance.

Current intended order:

```
Frame Pose
    ↓
Animation Semantics
    ↓
minimal IK / Retargeting integration
    ↓
Pose / Motion Completion
    ↓
foot locking / temporal polish
    ↓
FBX / Blender end-to-end validation
```

The exact integration boundary may move as retargeting is implemented, but Pose Completion is a required product capability and must not disappear from the roadmap.

## Completion criterion for the future Pose Completion stage

The future stage is complete when AnimCV can answer:

> When direct observation temporarily disappears, can the system produce a visually plausible, kinematically valid motion trajectory that reconnects to reliable observations without visible collapse, popping, or arbitrary pose invention?

Evaluation should include both quantitative and qualitative evidence.

Important qualitative failure modes include:

- joint collapse;
- visible popping at gap boundaries;
- implausible limb arcs;
- sudden bone-direction changes;
- contact-breaking foot motion;
- failure to reconnect to the recovered observation.

Ground-truth trajectory error remains useful, but it is not the only product criterion for an unobservable hidden interval.
