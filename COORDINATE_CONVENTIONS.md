# Coordinate conventions

This document is the binding contract for new pose-lifting, root-motion and
retarget code.  A value may cross a boundary only through the conversion named
below; no module may infer an axis from a mapping's `axis_hint`.

## Frames and units

| Name | Axes | Units | Use |
| --- | --- | --- | --- |
| Image `I` | +X right, +Y down | pixels | `PoseLandmark`, `MotionPoint.position_2d` |
| Camera `C` | +X right, +Y forward/away, +Z up | metric body-relative units | temporally lifted 3D pose |
| Character `R` | +X right, +Y forward, +Z up | metres | root-normalized target joints |
| Rig local `L` | Blender pose-bone local axes | radians/metres | final FK channels |

`image_delta_to_camera` maps `(dx, dy)` to `(dx, depth, -dy)`. The transform
from `C` to `R` always takes an explicit root yaw; fixed camera axes must not
be used as character axes after the performer turns.

## Calibrated camera input

`animcv_camera_calibration_v1` describes physical image intrinsics (`fx`,
`fy`, `cx`, `cy`) and optional OpenCV-compatible radial/tangential lens
distortion. Its `image_size` must match the frames given to the pose detector.
The canonical camera projection is `u=fx*(X/Y)+cx`, `v=cy-fy*(Z/Y)`, followed
by distortion. A lifted pose is pelvis-relative, so the calibrated audit fits
a per-frame pelvis translation solely to measure reprojection error. That fit
is not global root motion and must not be exported as such.

## Existing data compatibility

`MotionPoint.position_3d` currently stores `(image_x, image_y, relative_depth)`
when the optional depth sampler is used. Its components have mixed units and
it is **not** a `C` or `R` position. New 3D retarget code must reject it and
consume only a new metric lifted-pose representation introduced in phase 3.

## Rotation conventions

- Project `Quaternion` is `(x, y, z, w)`, unit length, active rotation:
  `v' = q * v * conjugate(q)`.
- `AnimationClip` direction samples are desired world/character deltas in that
  project layout.
- Blender RNA `PoseBone.rotation_quaternion` uses `(w, x, y, z)`.
- Before writing a child bone, Blender export uses its imported
  `bone.matrix_local` and converts its desired world delta to the parent
  relative local delta.  Assimp node matrices are not used for this step.

## Validation requirements

1. Identity clip must preserve the imported rig's rest pose.
2. A +90° character +Z turn must round-trip through Blender/FBX without a
   sign or component-order change.
3. Parent and child desired world rotations must produce the requested child
   direction once, rather than compound the parent rotation.
4. Any rig scale conversion must be recorded in output metadata.
