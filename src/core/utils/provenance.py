"""Repository identity recorded in datasets, evaluation sets and telemetry."""
from pathlib import Path
import os
import subprocess


def _git(root, *args, timeout=None):
    root = Path(root).resolve()
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
        text=True, stderr=subprocess.DEVNULL, timeout=timeout,
    ).strip()


def git_commit(root, *, default="unknown", required=False):
    """Return the commit; strict callers propagate lookup errors instead of a fallback."""
    try:
        return _git(root, "rev-parse", "HEAD")
    except (OSError, subprocess.SubprocessError):
        if os.environ.get("S2R_GIT_COMMIT"):
            return os.environ["S2R_GIT_COMMIT"]
        if required:
            raise
        return default


def git_identity(root, *, timeout=None, untracked_files=None):
    """Return a commit and dirty flag together; unknown identity has null fields."""
    status_args = ("status", "--porcelain")
    if untracked_files is not None:
        status_args += (f"--untracked-files={untracked_files}",)
    try:
        return {"git_commit": _git(root, "rev-parse", "HEAD", timeout=timeout),
                "git_dirty": bool(_git(root, *status_args, timeout=timeout))}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": os.environ.get("S2R_GIT_COMMIT"), "git_dirty": None}
