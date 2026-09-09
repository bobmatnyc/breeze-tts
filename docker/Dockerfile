# BASE_IMAGE selects the CUDA runtime. The -devel variant is only needed to
# compile FlashAttention; a serverless build passes the smaller -runtime image.
ARG BASE_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

ARG FLASH_ATTN_CUDA_ARCHS=90
# BUILD_FLASH_ATTN=1 keeps the pod/dev image unchanged. Serverless builds pass 0:
# every caller of load_runtime hardcodes attn_implementation="eager"
# (infer.py:73, breeze_infer/api.py:131), so the compiled kernels are dead weight.
ARG BUILD_FLASH_ATTN=1
# INSTALL_SERVERLESS_DEPS=1 adds the runpod SDK and huggingface_hub.
ARG INSTALL_SERVERLESS_DEPS=0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
        ninja-build \
        sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/breeze-infer

COPY requirements.txt docker/requirements-serverless.txt ./

RUN python -m pip install --upgrade pip setuptools wheel packaging ninja \
    && python -m pip install -r requirements.txt \
    && if [ "$INSTALL_SERVERLESS_DEPS" = "1" ]; then \
        python -m pip install -r requirements-serverless.txt; \
    fi \
    && if [ "$BUILD_FLASH_ATTN" = "1" ]; then \
        MAX_JOBS=8 FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS} python -m pip install \
            --no-build-isolation \
            --no-deps \
            "flash-attn==2.8.3"; \
    fi

COPY . .

RUN python docker/smoke_check.py \
    && python -m pytest -q \
        tests/test_logits_process.py \
        tests/test_rp_handler.py \
        tests/test_runpod_clone.py \
        tests/test_templates.py \
        tests/test_warmup_profile.py

ENTRYPOINT ["/opt/breeze-infer/docker/entrypoint.sh"]
CMD ["python", "-u", "rp_handler.py"]
