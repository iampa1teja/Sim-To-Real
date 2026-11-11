"""Repository data folders (arms/, scenes/, tasks/) and default selectors, without any Isaac imports."""

from pathlib import Path

DEFAULT_ARM = "so101"
DEFAULT_SCENE = "rpd_meeting_room"
DEFAULT_TASK = "pick_place_cube"


def repo_dir(folder: str) -> Path:
    """Return the first ``folder`` directory above this file (arms/, scenes/, tasks/)."""
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        # Skip Python packages of the same name (src/core/arms is code, not the arms/ data folder).
        if (parent / folder).is_dir() and not (parent / folder / "__init__.py").exists():
            return parent / folder
    raise FileNotFoundError(f"No {folder}/ directory above {here}")
