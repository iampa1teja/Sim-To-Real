"""Shared prelaunch helpers, safe to import before Kit and arm/scene selection."""


import os


def rendering_kit_args(existing: str = "") -> str:
    """Enable the installed material context-menu dependency before Kit starts.

    The material library initializes its menus asynchronously during startup.
    Enabling this extension after app creation can miss that initialization.
    """
    extension = "omni.kit.context_menu"
    words = existing.split()
    for index, word in enumerate(words):
        if word == f"--enable={extension}" or (
            word == "--enable" and words[index + 1:index + 2] == [extension]
        ):
            return existing
    return f"{existing} --enable {extension}".strip()


def apply_selection(cfg) -> None:
    from core.paths import repo_dir

    for kind, name in (("arms", cfg.arm), ("scenes", cfg.scene)):
        if not (repo_dir(kind) / name).is_dir():
            available = sorted(p.name for p in repo_dir(kind).iterdir() if p.is_dir())
            raise ValueError(f"Unknown {kind[:-1]} {name!r}; available: {available}")
    if cfg.task and not (repo_dir("tasks") / cfg.task / "task.toml").is_file():
        available = sorted(p.name for p in repo_dir("tasks").iterdir() if (p / "task.toml").is_file())
        raise ValueError(f"Unknown task {cfg.task!r}; available: {available}")
    if cfg.variant and not cfg.task:
        raise ValueError("variant needs a task.")
    os.environ["S2R_ARM"] = cfg.arm
    os.environ["S2R_SCENE"] = cfg.scene
    os.environ["S2R_TASK"] = cfg.task

