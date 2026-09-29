# Dataset preparation entry point

[Guide hub](../README.md)

See [05 — Training](../docs/05-training.md) for the complete workflow:

- [Pinned GR00T environment and dependencies](../docs/05-training.md#pin-the-training-environment)
- [Independent preparation, episode exclusion and verifier commands](../docs/05-training.md#prepare-an-independent-copy)
- [Fine-tuning and checkpoint selection](../docs/05-training.md#fine-tune)
- [CPU regression tests](../docs/05-training.md#cpu-preparation-regression-tests)

`prepare_pick_place_gr00t.py` preserves the source dataset and all existing
validation checks. `--exclude_episodes` filters source IDs and renumbers the
training copy. Filtered stats are recomputed; reports preserve the mapping.
`--resume` requires unchanged source hashes and the same selection.
`verify_pick_place_training.py` checks checksums and actual GR00T/PyAV frame reads.
Neither script trains a model or requires a GPU.
