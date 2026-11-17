"""Paths, environment-variable patching and fixture copying (contracts §13). Standard library only."""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
FIXTURES_DIR = TESTS_DIR / "fixtures"

#: The variables temp_env may patch (contracts §13).
S2R_VARS = ("S2R_ARMS_DIR", "S2R_ARM", "S2R_SCENE_DIR", "S2R_TASKS_DIR", "S2R_TASK")


@contextlib.contextmanager
def temp_env(**vars: str | os.PathLike | None) -> Iterator[None]:
    """Patch ``os.environ`` for the S2R_* selection variables and restore it exactly on exit.

    Keys must be in ``S2R_VARS`` (``ValueError`` otherwise). ``Path`` values are converted with
    ``str``; ``None`` removes the variable for the duration of the block.
    Example: ``with temp_env(S2R_ARMS_DIR=tmp / "arms"): load_arm_spec("fx_arm6")``.
    """
    unknown = sorted(set(vars) - set(S2R_VARS))
    if unknown:
        raise ValueError(f"temp_env only patches {S2R_VARS}; got {unknown}")
    saved = {key: os.environ.get(key) for key in vars}
    try:
        for key, value in vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = os.fspath(value)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def copy_fixture(name: str, dst: str | os.PathLike, *, fixtures_dir: str | os.PathLike | None = None) -> Path:
    """Copy ``tests/fixtures/<name>`` (file or folder) to ``dst/<name>`` and return the new path.

    The relative path is kept, so ``copy_fixture("arms/fx_arm6", tmp)`` gives ``tmp/arms/fx_arm6``
    and ``temp_env(S2R_ARMS_DIR=tmp / "arms")`` then finds the arm. Raises ``FileNotFoundError``
    when the fixture does not exist. ``fixtures_dir`` overrides the source root (for self-tests).
    """
    source_root = Path(fixtures_dir) if fixtures_dir is not None else FIXTURES_DIR
    source = source_root / name
    if not source.exists():
        raise FileNotFoundError(f"No fixture {name!r} under {source_root}")
    target = Path(dst) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)
    return target


def copy_repo_folder(kind: str, name: str, dst: str | os.PathLike, *, ignore: tuple[str, ...] = ("*.usd", "*.usdz")) -> Path:
    """Copy a real repo data folder (``arms/so101``, ``tasks/pick_place_cube``, ``scenes/...``) to ``dst/<kind>/<name>``.

    ``kind`` is ``"arms"``, ``"tasks"`` or ``"scenes"``. Large binary USD files are skipped by default
    (pass ``ignore=()`` to keep them). The repo folder itself is never modified.
    """
    if kind not in ("arms", "tasks", "scenes"):
        raise ValueError(f"kind must be arms, tasks or scenes, got {kind!r}")
    source = REPO_ROOT / kind / name
    if not source.is_dir():
        raise FileNotFoundError(source)
    target = Path(dst) / kind / name
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(*ignore, "__pycache__"), dirs_exist_ok=True)
    return target


def clear_task_cache(*names: str) -> None:
    """Forget loaded tasks so a re-written task folder is imported again.

    Drops ``core.tasks.loader._loaded`` entries and ``s2r_tasks.<name>`` modules (all when no names
    are given). Safe to call before ``core.tasks.loader`` was ever imported.
    """
    loader = sys.modules.get("core.tasks.loader")
    loaded = getattr(loader, "_loaded", None) if loader is not None else None
    if isinstance(loaded, dict):
        for key in list(loaded):
            if not names or key in names:
                loaded.pop(key, None)
    for module in list(sys.modules):
        parts = module.split(".")
        if parts[0] == "s2r_tasks" and len(parts) > 1 and (not names or parts[1] in names):
            sys.modules.pop(module, None)


@dataclass(frozen=True)
class Sandbox:
    """Paths of a ``sandbox()`` temp tree: ``root/arms``, ``root/tasks``, ``root/scenes/<scene>``."""

    root: Path
    arms: Path
    tasks: Path
    scenes: Path
    scene: Path


@contextlib.contextmanager
def sandbox(*, arm: str | None = None, task: str | None = None, scene: str = "fx_scene") -> Iterator[Sandbox]:
    """Create a temp tree with empty ``arms/``, ``tasks/`` and ``scenes/<scene>/`` and point the env vars there.

    Sets ``S2R_ARMS_DIR``, ``S2R_TASKS_DIR`` and ``S2R_SCENE_DIR`` (the scene folder itself, as
    ``core.scenes.scene_dir`` expects), plus ``S2R_ARM`` / ``S2R_TASK`` when given. Fill it with
    ``write_arm`` / ``write_task`` / ``write_scene`` / ``copy_fixture``. The task cache is cleared on
    entry and exit; everything is deleted on exit.
    """
    with tempfile.TemporaryDirectory(prefix="s2r_v4_") as tmp:
        root = Path(tmp)
        box = Sandbox(root=root, arms=root / "arms", tasks=root / "tasks", scenes=root / "scenes",
                      scene=root / "scenes" / scene)
        for path in (box.arms, box.tasks, box.scene):
            path.mkdir(parents=True, exist_ok=True)
        patch = {"S2R_ARMS_DIR": box.arms, "S2R_TASKS_DIR": box.tasks, "S2R_SCENE_DIR": box.scene}
        if arm is not None:
            patch["S2R_ARM"] = arm
        if task is not None:
            patch["S2R_TASK"] = task
        clear_task_cache()
        try:
            with temp_env(**patch):
                yield box
        finally:
            clear_task_cache()
