# Generic policy runtime: one uv venv per policy, built on first use. Two images come from this file and share
# every layer except the last label, so pushing both to a registry costs almost nothing extra:
#   --target serve   so101-serve  serves a policy (--policy <name>); the last stage, so it is the default target
#   --target train   so101-train  runs a policy's [train.commands] (--env <name> -- <command>)
# Build from the repo root:  so101 build serve|train   (or ./policy_servers/build.sh for serve)
# Policies are NOT baked in. They are bind-mounted at /policies, so adding one needs no rebuild.
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS base

LABEL org.opencontainers.image.description="Multi-policy inference server (GR00T, MolmoAct2, ...)"

# GIT_LFS_SKIP_SMUDGE: policy checkouts must not pull model weights through LFS.
# CUDA_HOME: flash-attn and similar packages compile or locate CUDA through it.
ENV DEBIAN_FRONTEND=noninteractive \
    GIT_LFS_SKIP_SMUDGE=1 \
    CUDA_HOME=/usr/local/cuda

# python3 here is only the system interpreter used by entrypoint.sh to read policy.toml (tomllib, 3.11+).
# Every policy gets its own interpreter from uv; this one never runs a model.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl git git-lfs ninja-build \
    python3 ffmpeg libegl1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Pinned uv. Policies pin their own Python versions, so uv downloads them on demand.
RUN curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/0.8.14/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Everything that must survive container restarts lives under /opt/cache (mount one named volume there):
#   uv/        downloaded wheels
#   python/    uv-managed interpreters
#   venvs/     one venv per policy
#   checkouts/ pinned git checkouts for policies built from an upstream repo
# UV_LINK_MODE=copy: cache and venvs sit on the same volume, but copy mode avoids hardlink warnings
# and keeps a venv valid if the cache is pruned.
ENV POLICY_CACHE=/opt/cache \
    UV_CACHE_DIR=/opt/cache/uv \
    UV_PYTHON_INSTALL_DIR=/opt/cache/python \
    UV_LINK_MODE=copy \
    POLICIES_DIR=/policies

# Shared, policy-agnostic code. Bind-mount over these paths during development to skip rebuilds.
# server_core: ZMQ server loop + launcher. policy_client: transport module reused for the wire format.
COPY policy_servers/core /opt/so101/server_core
COPY source/sim_to_real_so101/policy_client /opt/so101/policy_client
COPY policy_servers/entrypoint.sh /opt/so101/entrypoint.sh
RUN chmod +x /opt/so101/entrypoint.sh

ENTRYPOINT ["/opt/so101/entrypoint.sh"]
CMD ["--list"]

FROM base AS train
LABEL org.opencontainers.image.description="Policy training runtime: runs a policy's declared train commands"
ENV SO101_ROLE=train

# Last stage = default build target. Keep it last.
FROM base AS serve
LABEL org.opencontainers.image.description="Policy server for the SO-101 workshop (GR00T, MolmoAct2, ...)"
ENV SO101_ROLE=serve
