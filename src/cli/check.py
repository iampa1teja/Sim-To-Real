"""Validate extension manifests and Python entrypoints without importing Isaac Sim."""
import argparse
import ast
from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def extension_roots(repo):
    values = {"S2R_ARMS_DIR": str(repo / "arms"), "S2R_SCENES_DIR": str(repo / "scenes"),
              "S2R_TASKS_DIR": str(repo / "tasks")}
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _folders(repo, kind, marker):
    return sorted(path for path in (repo / kind).iterdir()
                  if path.is_dir() and not path.name.startswith(("_", ".")))


def _python_files(folder):
    for path in folder.rglob("*.py"):
        if not set(path.parts) & {"__pycache__", ".venv", "venv"}:
            ast.parse(path.read_text(), filename=str(path))


def _entry(folder, entry):
    module, attribute = entry.split(":", 1)
    # Task entrypoints are names within their own folder, including subpackages.
    path = folder / (module.replace(".", "/") + ".py")
    if not path.is_file():
        raise ValueError(f"Entrypoint module is missing: {path}")
    tree = ast.parse(path.read_text(), filename=str(path))
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == attribute
               for node in tree.body):
        raise ValueError(f"{path} has no declared entrypoint {attribute!r}")


def check_static(repo, *, arm=None, scene=None, task=None, policy=None, unit=None):
    from core.arms.spec import load_arm_spec
    from core.scenes import load_scene_spec
    from core.tasks.loader import load_task
    from core.policy_client.discovery import load_manifest

    repo = Path(repo).resolve()
    rows, errors = [], []
    loaded_arms, loaded_scenes, loaded_tasks = {}, {}, {}
    with extension_roots(repo):
        for kind, marker, loader, loaded in (("arms", "arm.toml", lambda name: load_arm_spec(name)[0], loaded_arms),
                                            ("scenes", "scene.toml", load_scene_spec, loaded_scenes),
                                            ("tasks", "task.toml", load_task, loaded_tasks)):
            for folder in _folders(repo, kind, marker):
                try:
                    spec = loader(folder.name)
                    loaded[folder.name] = spec
                    _python_files(folder)
                    if kind == "scenes" and spec.hook:
                        filename, attribute = spec.hook.split(":", 1)
                        _entry(folder, filename.removesuffix(".py") + ":" + attribute)
                    if kind == "tasks":
                        for entry in spec.spec.variants.values():
                            _entry(folder, entry)
                    rows.append(f"{kind}/{folder.name}")
                except Exception as error:
                    errors.append(f"{kind}/{folder.name}: {error}")
        policies = {}
        for folder in _folders(repo, "policies", "policy.toml"):
            try:
                manifest = load_manifest(folder)
                _python_files(manifest.path)
                if manifest.enabled:
                    policies[manifest.name] = manifest
                rows.append(f"policies/{manifest.name}")
            except Exception as error:
                errors.append(f"policies/{folder.name}: {error}")
        for label, selected, available in (("arm", arm, loaded_arms), ("scene", scene, loaded_scenes),
                                            ("task", task, loaded_tasks), ("policy", policy, policies)):
            if selected and selected not in available:
                errors.append(f"Selected {label} {selected!r} is unavailable or invalid")
        if arm in loaded_arms and policy in policies:
            dimension = getattr(policies[policy], "action_dim", "arm")
            if dimension != "arm" and dimension != len(loaded_arms[arm].action_joints):
                errors.append(f"Policy {policy!r} has {dimension} action axes; arm {arm!r} has {len(loaded_arms[arm].action_joints)}")
        if unit and arm in loaded_arms:
            try:
                from core.arms.calibration import resolve_calibration
                resolve_calibration(loaded_arms[arm], unit=unit, environ={})
            except Exception as error:
                errors.append(f"Arm unit {unit!r}: {error}")
    return rows, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    for selector in ("arm", "scene", "task", "policy", "unit"):
        parser.add_argument("--" + selector)
    parser.add_argument("--static-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        rows, errors = check_static(args.repo, arm=args.arm, scene=args.scene, task=args.task, policy=args.policy, unit=args.unit)
    except ModuleNotFoundError as error:
        print(f"Host dependency {error.name!r} is missing. Run 'uv sync --group host', then 'uv run s2r check'.")
        return 2
    for row in rows:
        print("PASS " + row)
    for error in errors:
        print("FAIL " + error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
