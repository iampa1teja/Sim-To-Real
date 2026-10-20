# Share the already-built, pinned GR00T runtime with N1.7 serving.
ARG GROOT_IMAGE=real-robot:n1.7
FROM ${GROOT_IMAGE}
ARG GROOT_REF=51d4c89f72fda44cbf77285c6a8114b52676b8a1
ARG LEROBOT_REF=c75455a6de5c818fa1bb69fb2d92423e86c70475
LABEL org.opencontainers.image.description="SO-101 GR00T N1.7 training and dataset preparation"
LABEL org.opencontainers.image.revision="${GROOT_REF}"
RUN test "$(git -C /Isaac-GR00T rev-parse HEAD)" = "${GROOT_REF}" \
    && git config --system --add safe.directory /Isaac-GR00T
# Keep the official Torch/Transformers lock. The converter's LeRobot utilities
# are installed without their separate dependency resolver replacing Torch.
COPY --from=training_wheels / /tmp/training-wheels/
RUN /Isaac-GR00T/.venv/bin/python -m ensurepip \
    && /Isaac-GR00T/.venv/bin/python -m pip install --no-cache-dir --no-deps \
      "lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_REF}" \
    && /Isaac-GR00T/.venv/bin/python -m pip install --no-cache-dir --no-index --no-deps \
      /tmp/training-wheels/*.whl \
    && rm -rf /tmp/training-wheels
ENV SO101_TRAIN_IN_CONTAINER=1 \
    PYTHONPATH=/Isaac-GR00T \
    PYTHONDONTWRITEBYTECODE=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
ENTRYPOINT []
CMD ["/bin/bash"]
