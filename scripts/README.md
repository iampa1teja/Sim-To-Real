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

To omit failed demonstrations, pass a comma-separated list of **source** episode
indices (the default excludes none and preserves the existing conversion):

```bash
python scripts/prepare_pick_place_gr00t.py \
  --groot /path/to/Isaac-GR00T \
  --source datasets/pick_place_v1 \
  --output datasets/pick_place_v1_gr00t_filtered \
  --exclude_episodes 0,6
```

Retained episodes stay in source order and are numbered `0..N-1`. The copy's
parquet episode/global indices, metadata, totals, split boundaries, video paths,
and sidecar names/episode indices are updated. Video bytes and all other parquet
values are preserved. Unused tasks are removed; retained task IDs stay unchanged.
Unknown/negative indices and excluding every episode are rejected. Duplicate
indices are accepted once. The filtered `preparation_report.json` records
`excluded` and `index_map` (new episode ID to source ID).

At the pinned revision, `gr00t/data/dataset/lerobot_episode_loader.py` reads
`episodes.jsonl` in line order, uses its episode IDs and `info.json` chunk/path
templates to locate parquet/videos, and maps parquet task IDs through
`tasks.jsonl`. It loads `meta/stats.json` and optional `relative_stats.json`.
The generators in `gr00t/data/stats.py` skip existing valid statistics, so the
filtered conversion never copies source global stats and removes generated stats
on resume before recomputing both files from retained parquet. Per-episode stats
are retained with their episode/global index statistics shifted accordingly.

The preparation checks every parquet value, decodes every video frame, verifies
frame counts/timestamps and source checksums, and exercises the actual GR00T
reader. It writes reports and checksum manifests in the output directory.
An interrupted `.building` directory can be retried with `--resume`; the
recorded source hashes must still match. Existing completed outputs are never
overwritten. FFmpeg must be available for the GR00T reader check.
Resuming a filtered build also requires the same `--exclude_episodes` selection.

To independently verify a prepared dataset using the GR00T checkout:

```bash
PYTHONPATH=/path/to/Isaac-GR00T python scripts/verify_pick_place_training.py \
  --dataset datasets/pick_place_v1_gr00t
```

The verifier checks the prepared manifest and compares GR00T's first, middle,
and last frame for both cameras in every episode against independent PyAV
decoding. Both scripts are CPU-only; they do not launch training. Dataset,
model, and runtime artifacts remain excluded from Git.

Run the synthetic five-episode CPU integration tests with the same dependencies:

```bash
GROOT_CHECKOUT=/path/to/Isaac-GR00T \
  python -m unittest discover -s tests -p test_prepare_pick_place_gr00t.py -v
```

The tests exclude episodes 0 and 3, check retained values, contiguous indices,
video bytes, sidecars, tasks, splits, statistics, source hashes, resume, and the
real GR00T reader/verifier. Optionally set `PREPARATION_BASELINE` to a saved copy
of the pre-change script to compare default output bytes; only script provenance,
its checksum manifest, and report time/output path are allowed to differ.
