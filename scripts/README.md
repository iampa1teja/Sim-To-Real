# Prepare the pick-place dataset for GR00T

From the workshop root, with the LeRobot dataset dependencies (including
PyArrow and PyAV) installed and a GR00T checkout at
`ead52833afbbf4243f8cd5e7664f48a94de03b19`:

```bash
python scripts/prepare_pick_place_gr00t.py \
  --groot /path/to/Isaac-GR00T \
  --source datasets/pick_place_v1 \
  --output datasets/pick_place_v1_gr00t
```

This creates an independent v2.1 training copy, preserves the original dataset,
maps the recorded RGB cameras to `front` and `wrist`, copies the upstream
SO100/SO101 modality configuration, and computes relative-action statistics.
It uses the pinned upstream conversion functions without their source-directory
swap. The lossless video-copy path requires one complete episode per source
video, as produced by this workshop's recorder; other layouts fail explicitly.
The original joint values and cube-pose sidecars are retained.

The preparation checks every parquet value, decodes every video frame, verifies
frame counts/timestamps and source checksums, and exercises the actual GR00T
reader. It writes reports and checksum manifests in the output directory.
An interrupted `.building` directory can be retried with `--resume`; the
recorded source hashes must still match. Existing completed outputs are never
overwritten. FFmpeg must be available for the GR00T reader check.

To independently verify a prepared dataset using the GR00T checkout:

```bash
PYTHONPATH=/path/to/Isaac-GR00T python scripts/verify_pick_place_training.py \
  --dataset datasets/pick_place_v1_gr00t
```

The verifier checks the prepared manifest and compares GR00T's first, middle,
and last frame for both cameras in every episode against independent PyAV
decoding. Both scripts are CPU-only; they do not launch training. Dataset,
model, and runtime artifacts remain excluded from Git.
