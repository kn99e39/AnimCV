# Worklog 97: isolated environment for the official 4DHumans / HMR 2.0b baseline (does not modify animcv-external).
# Build context must contain the upstream clone at 4D-Humans/ (commit efe18deff163b29dff87ddbd575fa29b716a356c).
#   docker build -f w97_hmr2.Dockerfile -t animcv-w97-hmr2:cu124 <context>
FROM animcv-external:cu124
ENV PYOPENGL_PLATFORM=egl
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libgles2 libglib2.0-0 libx11-6 libxext6 libxrender1 libxi6 libxxf86vm1 libglu1-mesa && rm -rf /var/lib/apt/lists/*
# numpy < 1.24 so the upstream chumpy-based SMPL pickle conversion (hmr2.models.convert_pkl) runs unchanged
RUN /opt/vlm/bin/python -m pip install --no-cache-dir \
      numpy==1.23.5 pytorch-lightning==2.2.5 smplx==0.1.28 pyrender==0.1.45 trimesh==4.4.9 \
      opencv-python-headless==4.10.0.84 yacs==0.1.8 scikit-image==0.24.0 einops==0.8.0 timm==0.9.16 \
      webdataset==0.2.86 dill==0.3.8 pandas==2.2.2 hydra-core==1.3.2 omegaconf==2.3.0 rich==13.7.1 pyrootutils==1.0.4
# chumpy's setup.py imports pip, so it cannot build in pip's isolated build environment
RUN /opt/vlm/bin/python -m pip install --no-cache-dir wheel==0.44.0 \
 && /opt/vlm/bin/python -m pip install --no-cache-dir --no-build-isolation \
      "chumpy @ git+https://github.com/mattloper/chumpy@580566eafc9ac68b2614b64d6f7aaa84eebb70da"
COPY 4D-Humans /opt/4D-Humans
RUN /opt/vlm/bin/python -m pip install --no-cache-dir --no-deps -e /opt/4D-Humans \
 && /opt/vlm/bin/python -c "import numpy, torch, chumpy, smplx, pytorch_lightning, hmr2; print('ok', numpy.__version__, torch.__version__, pytorch_lightning.__version__)"
