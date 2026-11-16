"""Live Isaac cfg oracle for Phase 8.

Run in s2r-sim with ``s2r-entrypoint python tests/workspace/tasks/check_stack_two_cubes.py``.
It compares the complete legacy and generic cfg exports for both arms, scenes and variants.
Only the port's generic term function/parameter substitutions and disabled reset event are normalized.
"""
import importlib
import sys
import types


def main():
    from sim import launch, close_app
    # Select the repository's utils package before Kit adds its own search paths,
    # just as the production entry points' prelaunch task imports do.
    from core.tasks.loader import load_task

    app = launch(headless=True, enable_cameras=True, device="cuda:0")
    try:
        from sim.workspace.envs.scene_builder import build_scene_cfg
        from sim.workspace.envs.task_env import apply_variant
        from utils.arms.arms import load_arm_spec
        from task_oracle import legacy_task, LEGACY_TASKS

        old_task = legacy_task("stack_two_cubes", "StackTwoCubesTask")
        task = load_task("stack_two_cubes")
        package_name = "legacy_stack_cfg"
        package = types.ModuleType(package_name)
        package.__path__ = [str(LEGACY_TASKS / task.name)]
        package.TASK = old_task
        sys.modules[package_name] = package
        legacy_cfg = importlib.import_module(package_name + ".env_cfg")
        for arm_name in ("so101", "sp7"):
            arm = load_arm_spec(arm_name)[0]
            for scene in ("rpd_meeting_room", "blank_table"):
                for variant in ("teleop", "eval"):
                    old = build_scene_cfg(scene, arm)
                    new = build_scene_cfg(scene, arm)
                    getattr(legacy_cfg, "apply_" + variant)(old, old_task)
                    apply_variant(new, task, variant)
                    a, b = old.to_dict(), new.to_dict()
                    # These three hooks now receive explicit task/object parameters.
                    for event in ("reset_tracking", "reset_target"):
                        if a["events"].get(event) is not None:
                            a["events"][event]["func"] = b["events"][event]["func"]
                            a["events"][event]["params"] = b["events"][event]["params"]
                    if variant == "eval":
                        assert a["events"].pop("reset_target") is None
                        a["terminations"]["success"]["func"] = b["terminations"]["success"]["func"]
                        a["terminations"]["success"]["params"] = b["terminations"]["success"]["params"]
                    differences = list(diff(a, b))
                    assert not differences, (arm_name, scene, variant, differences)
                    print(f"MATCH {arm_name}/{scene}/{variant}", flush=True)
    finally:
        close_app(app)


def diff(a, b, path="cfg"):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a.keys() | b.keys():
            if key not in a or key not in b:
                yield path + "." + str(key), a.get(key), b.get(key)
            else:
                yield from diff(a[key], b[key], path + "." + str(key))
    elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            yield path, a, b
        else:
            for index, (x, y) in enumerate(zip(a, b)):
                yield from diff(x, y, f"{path}[{index}]")
    elif a != b:
        yield path, a, b


if __name__ == "__main__":
    main()
