"""Use LeRobot's trainer with the workshop state/action frame correction."""
import faulthandler
import json
import sys

import torch

from molmo_lora_support import install_training_correction

if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    print("WORKSHOP_RUNTIME " + json.dumps({
        "python": sys.version, "executable": sys.executable,
        "autograd_multithreading": False,
    }), flush=True)
    install_training_correction()
    from lerobot.scripts import lerobot_train

    # Two extra recovery points protect the long first interval. Keep the
    # configured periodic schedule and all completed checkpoints afterward.
    original_should_save = lerobot_train.should_save_checkpoint

    def should_save(step, save_freq, total_steps):
        return step in (512, 2048) or original_should_save(step, save_freq, total_steps)

    lerobot_train.should_save_checkpoint = should_save

    # Keep checkpoint recomputation on the calling Python thread. This retains
    # activation checkpointing and CUDA kernels, while avoiding the native
    # pt_autograd_0 unpack-hook crash observed in both Python patch versions.
    with torch.autograd.set_multithreading_enabled(False):
        lerobot_train.main()
