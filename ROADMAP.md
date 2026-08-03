# AnimCV implementation roadmap

This is the original staged roadmap for moving from 2D direction retargeting
to game-engine-ready FK animation output. It is retained as the baseline;
`QUALITY_REMEDIATION_PLAN.md` records the additional work discovered through
the production-demo audit.

## 1. Failure baseline

- Preserve the production demo, its 2D input, and its invalid knee-bend FBX as
  regression/audit evidence.
- Treat a clean FBX export as insufficient evidence of animation quality.

## 2. Coordinate contract

- Define image, camera, character-root, rig-local and Blender frames.
- Fix units, quaternion ordering and rest-pose conversion rules.
- Verify identity, rotation and parent-child transform round trips.

## 3. Temporal 3D pose lifting

- Lift tracked 2D poses into pelvis-relative metric 3D joint sequences.
- Preserve source confidence and depth uncertainty information.

## 4. Root and torso estimation

- Infer pelvis/root yaw and forward direction.
- Separate global motion from local joint motion.

## 5. Constraint-aware retargeting

- Solve arms and legs with two-bone IK.
- Add elbow/knee pole vectors, hinge axes, joint limits and rig-length
  adaptation.

## 6. FK bake and Blender export

- Bake constraint results as parent-relative local FK rotation keys.
- Verify Blender and FBX round trips.

## 7. Foot contact

- Detect contacts, reduce foot sliding and correct root movement.

## 8. Quality validation

- Exercise frontal, side-on, turning, walking, occluded and multi-person
  clips.
- Measure 3D joint quality, joint-limit violations, foot sliding and visual
  FBX integrity.

## 9. Productization

- Expose options in CLI/GUI.
- Improve mapping/profile UI, metadata, demos and user documentation.

## Status at 2026-08-01

Implementation work for stages 1–4 exists. The production audit does not yet
accept stages 3–4 as quality-ready: left forearm length variance and right-knee
bend-direction flips require remediation before stage 5 consumes their output.
