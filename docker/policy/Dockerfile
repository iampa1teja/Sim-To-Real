FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS base 

LABEL org.opencontainers.image.description="Multi-policy inference server (GR00T, MolmoAct2, ...)"

ENV DEBIAN_FRONTEND=noninteractive \
    GIT_LFS_SKIP_SMUDGE=1 \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git git-lfs ninja-build \
    python3 libpython3.12t64 ffmpeg libegl1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/0.8.14/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

ENV POLICY_CACHE=/opt/cache \ 
    UV_CACHE_DIR=/opt/cache/uv \ 
    UV_PYTHON_INSTALL_DIR=/opt/cache/python \ 
    UV_LINK_MODE=hardlink \
    POLICIES_DIR=/policies 

COPY src/core/policy_server /opt/src/policy_server
COPY src/core/policy_client /opt/src/policy_client
COPY docker/policy/entrypoint.sh /opt/src/entrypoint.sh

RUN chmod +x /opt/src/entrypoint.sh

ENTRYPOINT ["/opt/src/entrypoint.sh"]
CMD ["--list"]

FROM base AS train
LABEL org.opencontainers.image.description="Policy training runtime: runs a policy's declared train commands"
ENV S2R_ROLE=train

FROM base AS serve
LABEL org.opencontainers.image.description="Policy server for the SO-101 workshop (GR00T, MolmoAct2, ...)"
ENV S2R_ROLE=serve
