"""Layer A — Frame Pose Core.

`one frame -> one root-relative canonical 17-joint 3D pose`.

See `Architecture_v3_FramePose.md`. This package deliberately does not import
`training.temporal_lifter`'s training path: the Legacy Temporal Pose Baseline
stays byte-for-byte reproducible and is never modified from here.
"""
