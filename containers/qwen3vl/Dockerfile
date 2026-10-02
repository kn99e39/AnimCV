# Diagnostic VLM runtime for Worklog 67 (Qwen/Qwen3-VL-8B-Instruct, 4-bit NF4).
# Pinned once; not part of production FramePose.  Model weights are mounted
# from the host Hugging Face cache, never baked into the image.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/vlm && /opt/vlm/bin/pip install --upgrade pip==24.3.1 \
    && /opt/vlm/bin/pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.4.1 torchvision==0.19.1 \
    && /opt/vlm/bin/pip install \
        transformers==4.57.6 accelerate==1.10.1 bitsandbytes==0.45.5 \
        huggingface_hub==0.35.3 safetensors==0.6.2 \
        pillow==11.3.0 numpy==1.26.4 opencv-python-headless==4.11.0.86 pytest==8.3.5 \
    && /opt/vlm/bin/pip freeze > /opt/vlm/freeze.txt

ENTRYPOINT ["/opt/vlm/bin/python"]
