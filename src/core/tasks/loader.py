from __future__ import annotations

import importlib
import sys
import types
from typing import Any

from core.tasks.base import TASK_NAME, TASK_PACKAGE, Task
from utils.workspace import available, kind_dir

_loaded: dict[str, Task] = {}


def _ensure_package() -> None:
    package = sys.modules.get(TASK_PACKAGE)
    path = [str(kind_dir("tasks"))]
    if package is None:
        package = types.ModuleType(TASK_PACKAGE)
        package.__path__ = path
        package.__package__ = TASK_PACKAGE
        sys.modules[TASK_PACKAGE] = package
    elif list(package.__path__) != path:
        package.__path__ = path


def load_task(name: str | None = None) -> Task:
    name = name or TASK_NAME
    if name in _loaded:
        return _loaded[name]
    if not name.isidentifier():
        raise ValueError(f"Task folder name {name!r} must be a valid Python identifier")
    if name not in available("tasks", "task.toml"):
        raise FileNotFoundError(f"Unknown task {name!r}; available: {available("tasks", "task.toml")}")
    if not (kind_dir("tasks") / name / "__init__.py").is_file():
        raise FileNotFoundError(f"tasks/{name}/__init__.py is missing; it must define TASK")

    _ensure_package()
    module = importlib.import_module(f"{TASK_PACKAGE}.{name}")
    task = getattr(module, "TASK", None)
    if not isinstance(task, Task):
        raise TypeError(f"tasks/{name}/__init__.py must define TASK as a Task instance, got {type(task).__name__}")
    if task.name != name:
        raise ValueError(f"tasks/{name}: TASK was built from task {task.name!r}")
    _loaded[name] = task
    return task


def resolve(entry_point: str) -> Any:
    module_name, _, attr = entry_point.partition(":")
    if module_name.startswith(f"{TASK_PACKAGE}."):
        _ensure_package()
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise AttributeError(f"{module_name} has no {attr!r} (from entry point {entry_point!r})") from None


def load_task_module(module: str, task: str | None = None):
    # Import an Isaac-free task helper such as tasks/<task>/benchmark.py (S2R_TASK by default).
    task = task or TASK_NAME
    if task not in available("tasks", "task.toml"):
        raise FileNotFoundError(f"Unknown task {task!r}; available: {available("tasks", "task.toml")}")
    _ensure_package()
    try:
        return importlib.import_module(f"{TASK_PACKAGE}.{task}.{module}")
    except ModuleNotFoundError as error:
        if error.name != f"{TASK_PACKAGE}.{task}.{module}":
            raise
        raise FileNotFoundError(f"tasks/{task}/{module}.py is missing") from None
