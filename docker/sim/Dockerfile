FROM nvcr.io/nvidia/isaac-lab:3.0.0

USER root 
ENV HOME=/root 
ENV PYTHON=/opt/isaaclab-venv/bin/python
ENV PATH="/opt/isaaclab-venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git \
    libxkbcommon0 libxkbcommon-x11-0 libxcursor1 libxrandr2 libxi6 libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone https://github.com/huggingface/lerobot.git
WORKDIR /workspace/lerobot
RUN git checkout v0.6.1

RUN printf '%s\n' \
    "numpy==2.5.1" \
    "packaging==26.0" \
    "torch==2.11.0" \
    "torchvision==0.26.0" \
    > /tmp/overrides.txt

RUN uv pip install --python $PYTHON --override /tmp/overrides.txt \
    -e ".[dataset,hardware,feetech,viz,pyzmq-dep]"

RUN $PYTHON -c "import numpy, cv2, torch, torchcodec, datasets, rerun, zmq, lerobot"

RUN uv pip install --python "$PYTHON" "imageio-ffmpeg>=0.6,<0.7"
RUN $PYTHON -c "import robomimic"

WORKDIR /workspace/s2r
COPY pyproject.toml ./
COPY src/ src/
RUN uv pip install --python "$PYTHON" --no-deps --no-build-isolation -e /workspace/s2r
RUN $PYTHON -c "import cli, core, sim"

ENV PYTHONPATH=/workspace/s2r/src:/workspace/s2r

COPY docker/sim/entrypoint.sh /usr/local/bin/s2r-entrypoint
RUN chmod +x /usr/local/bin/s2r-entrypoint

RUN mkdir -p /tmp/pycache && chmod 1777 /tmp/pycache
ENV PYTHONPYCACHEPREFIX=/tmp/pycache
ENV XDG_RUNTIME_DIR=/tmp/xdg-runtime-root
ENV ISAAC_PATH=/isaac-sim
ENV CARB_APP_PATH=/isaac-sim/kit
ENV EXP_PATH=/isaac-sim/apps
RUN install -d -m 0700 "$XDG_RUNTIME_DIR"

HEALTHCHECK NONE
ENTRYPOINT ["/usr/local/bin/s2r-entrypoint"]
CMD ["bash"]
