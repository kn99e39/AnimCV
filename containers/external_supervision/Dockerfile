# Diagnostic runtime for Worklog 73 (external paired RGB<->3D supervision).
# Extends the pinned Worklog 67/68 Qwen3-VL runtime (same torch / numpy / cv2)
# with:
#   - the CMU OpenPose COCO-18 body model (PyTorch port, Hzzone/pytorch-openpose
#     network definition at a pinned commit; weights = the CMU COCO body model
#     converted, sha256-verified) -- the same detector family that produced the
#     3DPW-shipped poses2D used by the FrameBank benchmark observation;
#   - smplx for deriving BEDLAM camera-space joints from the dataset's SMPL-X
#     labels (model files are mounted from the host, never baked in).
# Not part of production FramePose.
FROM animcv-qwen3vl:cu124

RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/vlm/bin/pip install scipy==1.13.1 smplx==0.1.28 \
    && /opt/vlm/bin/pip freeze > /opt/vlm/freeze_external.txt

ARG OPENPOSE_COMMIT=5ee71dc10020403dc3def2bb68f9b77c40337ae2
ARG OPENPOSE_WEIGHTS_SHA256=25a948c16078b0f08e236bda51a385d855ef4c153598947c28c0d47ed94bb746
RUN git clone https://github.com/Hzzone/pytorch-openpose /opt/pytorch-openpose \
    && git -C /opt/pytorch-openpose checkout ${OPENPOSE_COMMIT} \
    && curl -sSL -o /opt/pytorch-openpose/body_pose_model.pth \
        https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth \
    && echo "${OPENPOSE_WEIGHTS_SHA256}  /opt/pytorch-openpose/body_pose_model.pth" | sha256sum -c -

ENV OPENPOSE_ROOT=/opt/pytorch-openpose
ENTRYPOINT ["/opt/vlm/bin/python"]
