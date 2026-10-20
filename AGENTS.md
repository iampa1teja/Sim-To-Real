# Project notes

GR00T training and preparation run through `./docker/train.sh`; see
[docs/05-training.md](docs/05-training.md). Policy serving and simulation evaluation
use `./docker/eval.sh`; see [docs/06-evaluation.md](docs/06-evaluation.md).
The host Isaac-GR00T checkout, its virtual environment and the physical-arm
rollout client were removed to reclaim space. Keep model weights, datasets and
completed evaluation results. The `real-robot:n1.7` image remains necessary for
GR00T serving and as the training image's base.
