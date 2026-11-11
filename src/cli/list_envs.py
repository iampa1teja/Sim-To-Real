# `list_envs`: print the arms, scenes and tasks in this repo and the gym ids they register.
#
# Isaac-free: sim.envs registration only reads scene.toml / task.toml and imports the task's __init__
# (TASK), so this runs on the host or in the sim image without launching Isaac Sim. Env cfg classes are
# not built here.

import argparse
import os


# Rows of (gym id, what it composes, env cfg entry) for the selected scene and every task.
def _rows(scene, tasks):
    import gymnasium as gym

    os.environ["S2R_SCENE"] = scene
    os.environ["S2R_TASK"] = ""
    import sim.envs as envs

    envs.register_envs(scene=scene, task="")
    rows = [(envs.BASE_GYM_ID, "arm only", f"{envs.__name__}.base_env_cfg:BaseEnvCfg"),
            (envs.scene_gym_id(scene), f"arm + {scene}", envs.scene_entry_point(scene))]
    for name in tasks:
        task = envs.register_envs(scene=scene, task=name)
        for variant in task.spec.all_variants():
            default = " (default)" if variant == task.spec.default_variant else ""
            rows.append((task.gym_id(variant), f"arm + {scene} + {name}:{variant}{default}",
                         task.spec.entry_point(variant)))
    registered = {spec.id for spec in gym.registry.values()}
    return [row for row in rows if row[0] in registered]


# Print the folders and the gym ids as aligned plain-text tables.
def main(argv=None):
    from core.paths import DEFAULT_SCENE, repo_dir
    from core.tasks.base import available_tasks

    parser = argparse.ArgumentParser(description="List arms, scenes, tasks and their S2R gym ids.")
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE", DEFAULT_SCENE))
    args = parser.parse_args(argv)

    def folders(kind, marker):
        return sorted(p.name for p in repo_dir(kind).iterdir() if not p.name.startswith("_") and (p / marker).is_file())

    tasks = available_tasks()
    print(f"arms:   {', '.join(folders('arms', 'arm.toml')) or '-'}")
    print(f"scenes: {', '.join(folders('scenes', 'scene.toml')) or '-'}")
    print(f"tasks:  {', '.join(tasks) or '-'}")
    print()
    rows = _rows(args.scene, tasks)
    width = [max(len(str(row[i])) for row in rows + [("gym id", "composes", "env cfg")]) for i in range(3)]
    for row in [("gym id", "composes", "env cfg"), *rows]:
        print("  ".join(str(value).ljust(width[i]) for i, value in enumerate(row)))


if __name__ == "__main__":
    main()
