"""Keep milestone checkpoints next to a short rolling "last checkpoint" window.

The Trainer saves often (e.g. every 1k steps) with a small save_total_limit, so a crash loses little.
Every `every` steps the fresh checkpoint becomes a milestone; only the newest `keep` milestones are kept.
Milestones sit outside the Trainer's checkpoint-* rotation and resume search.

- No milestone dir (default): hard-link into <output_dir>/milestones/checkpoint-N (instant, no extra disk).
- Milestone dir on another disk (e.g. rolling checkpoints on a fast SSD, milestones on a big HDD):
  hard-link into <output_dir>/.milestone_staging/checkpoint-N (instant, survives Trainer rotation), then a
  detached process copies it to <milestone_dir>/checkpoint-N.partial and renames it into place, so training
  never waits for the slow disk and a training crash does not stop the copy. Leftover staging dirs (e.g.
  after a reboot) are copied again on the next start.
"""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

STAGING = ".milestone_staging"


def link_tree(src: Path, dst: Path):
    """Hard-link every file of src into a new dst tree (same filesystem)."""
    tmp = dst.with_name(dst.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    for root, _, files in os.walk(src):
        target = tmp / Path(root).relative_to(src)
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            os.link(Path(root) / name, target / name)
    tmp.rename(dst)


def step_of(path: Path):
    tail = path.name.split("-")[-1]
    return int(tail) if path.name.startswith("checkpoint-") and tail.isdigit() else None


def prune(folder: Path, keep: int):
    """Delete all but the newest `keep` complete checkpoint-N dirs in folder."""
    saved = sorted((p for p in folder.glob("checkpoint-*") if step_of(p) is not None), key=step_of)
    for old in saved[:-keep] if keep > 0 else []:
        shutil.rmtree(old)


def copy_milestone(staged: Path, milestone_dir: Path, keep: int):
    """Copy a staged checkpoint to milestone_dir (via .partial + rename), prune, then drop the staging link."""
    lock = open(staged.parent / f"{staged.name}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return  # another copier (e.g. from before a training restart) is already on it
    if not staged.is_dir():
        return
    milestone_dir.mkdir(parents=True, exist_ok=True)
    dst = milestone_dir / staged.name
    if not dst.exists():
        tmp = dst.with_name(dst.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(staged, tmp)
        tmp.rename(dst)
    prune(milestone_dir, keep)
    shutil.rmtree(staged)


def start_copy(staged: Path, milestone_dir: Path, keep: int):
    """Run copy_milestone in a detached, low-priority process; its log goes next to the staging dir."""
    log = open(staged.parent / f"{staged.name}.copy.log", "ab")
    command = [sys.executable, __file__, "copy", str(staged), str(milestone_dir), str(keep)]
    if shutil.which("ionice"):
        command = ["ionice", "-c3", *command]
    return subprocess.Popen(["nice", "-n", "10", *command], stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                            start_new_session=True)


def save_milestone(output_dir, step: int, every: int, keep: int, milestone_dir=None):
    """Turn checkpoint-<step> into a milestone when step is a multiple of every."""
    if every <= 0 or step <= 0 or step % every:
        return None
    output_dir = Path(output_dir)
    src = output_dir / f"checkpoint-{step}"
    if not src.is_dir():
        return None
    if milestone_dir is None:
        milestones = output_dir / "milestones"
        milestones.mkdir(exist_ok=True)
        dst = milestones / src.name
        if not dst.exists():
            link_tree(src, dst)
        prune(milestones, keep)
        return dst
    staging = output_dir / STAGING
    staging.mkdir(exist_ok=True)
    staged = staging / src.name
    if not staged.exists():
        link_tree(src, staged)
    start_copy(staged, Path(milestone_dir), keep)
    return Path(milestone_dir) / src.name


def resume_pending(output_dir, milestone_dir, keep: int):
    """Restart copies left unfinished by a crash or reboot."""
    staging = Path(output_dir) / STAGING
    if milestone_dir is None or not staging.is_dir():
        return []
    for partial in staging.glob("*.partial"):
        shutil.rmtree(partial)  # an interrupted link_tree; that milestone's source may be rotated away already
    pending = sorted((p for p in staging.glob("checkpoint-*") if p.is_dir() and step_of(p) is not None), key=step_of)
    return [start_copy(staged, Path(milestone_dir), keep) for staged in pending]


def make_callback(every: int, keep: int, milestone_dir=None):
    from transformers import TrainerCallback

    class MilestoneCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # On resume the Trainer reloads trainer_state.json after applying the current args, so the old
            # run's save_steps (e.g. 10000) would win over --save-steps. Re-apply the current ones.
            state.compute_steps(args, state.max_steps)
            if state.is_world_process_zero and every > 0:
                resume_pending(args.output_dir, milestone_dir, keep)

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                save_milestone(args.output_dir, state.global_step, every, keep, milestone_dir)

    return MilestoneCallback()


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "copy":
    copy_milestone(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
