"""Repository data folders (workspace/arms, scenes, tasks, ...) and default selectors, without any Isaac imports."""

from pathlib import Path

DEFAULT_ARM = "so101"
DEFAULT_SCENE = "rpd_meeting_room"
DEFAULT_TASK = "pick_place_cube"


def repo_root() -> Path:
    """Return the repository top: the nearest ancestor of this file that contains ``workspace/``."""
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "workspace").is_dir():
            return parent
    raise FileNotFoundError(f"No workspace/ directory above {here}")


def repo_dir(folder: str) -> Path:
    """Return ``workspace/<folder>`` (arms, scenes, tasks, ...) or, failing that, ``<folder>`` at the repo top."""
    root = repo_root()
    for path in (root / "workspace" / folder, root / folder):
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No {folder}/ directory in {root}/workspace or {root}")
