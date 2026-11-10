"""Shared prelaunch helpers, safe to import before Kit and arm/scene selection."""


import os
from argparse import Namespace
import sys
import traceback


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
    os.environ["S2R_TASK"] = cfg.task or ""


def add_launch_arguments(parser):
    """Register SDK flags without starting Kit or importing simulator assets."""
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    return parser


def launch(args=None, *, selection=None, enable_cameras=None, **overrides):
    """Start the shared Kit app after selectors and rendering dependencies are prepared.

    Namespace callers retain SDK-resolved options, including environment-selected headless
    mode. The module itself stays safe to import before selecting arms/scenes/tasks.
    """
    if selection is not None:
        apply_selection(selection)
    options = Namespace(**(vars(args) if isinstance(args, Namespace) else dict(args or {})))
    vars(options).update(overrides)
    if enable_cameras is not None:
        options.enable_cameras = enable_cameras
    options.kit_args = rendering_kit_args(getattr(options, "kit_args", "") or "")

    from isaaclab.app import AppLauncher

    app = AppLauncher(options).app
    if isinstance(args, Namespace):
        vars(args).update(vars(options))
    return app


def close_app(app, *, success=True):
    """Flush pending failures before Kit's close can terminate the Python process."""
    exception = sys.exc_info()
    exit_code = 0 if success and exception[0] is None else 1
    if exception[0] is not None:
        traceback.print_exception(*exception)
    sys.stdout.flush()
    sys.stderr.flush()
    app.close(exit_code=exit_code)

