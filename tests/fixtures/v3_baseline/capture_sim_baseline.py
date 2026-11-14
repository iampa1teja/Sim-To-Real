"""Records the v2 baseline, so run it from the Phase 0 commit (45b0731), where the v2 module paths it
imports (sim.app, sim.envs) still exist. It is not kept in step with later phases.

Record the v3 Phase 0 simulator-side baseline as JSON files.

Run once inside the ``s2r-sim`` container, from the repository root:

    s2r-entrypoint python tests/fixtures/v3_baseline/capture_sim_baseline.py --out tests/fixtures/v3_baseline/sim

Writes into ``--out``:

- ``environment.json``: Python/torch/Isaac Lab versions, git commit, and where ``import utils`` and
  ``import shared`` would resolve today (evaluated in this process, which never starts Kit), so we know
  whether a new top-level ``src/utils`` package would be shadowed in the container.
- ``pytest_outcomes.json``: ``python -m pytest tests -q -p no:cacheprovider -rA`` with src/ on PYTHONPATH;
  per-test outcome by node id (read from the same run's JUnit XML), the summary line and the exit code.
- ``env_cfg/<task>__<variant>.json``: the parsed env cfg ``to_dict()`` for every task and declared variant,
  made JSON-safe. Arm/scene/task are selected at import time, so each pair runs in a fresh headless
  subprocess of this script (``--one-cfg TASK VARIANT``), launched like ``src/sim/check.py``.

The arm and scene are ``--arm``/``--scene`` (default: ``S2R_ARM``/``S2R_SCENE``, else so101 and
rpd_meeting_room). File names carry only task and variant, so capture another arm or scene into its own
``--out``. ``--skip-pytest`` reruns only the cfg dumps.

Every section records its error text instead of aborting. Per-step signal traces and dataset frame hashes
are NOT captured here: they need the scripted datagen recipes and are a separate step.
"""
import argparse
import dataclasses
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
CFG_TIMEOUT_S = 1800


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def guarded(function, *args):
    try:
        return function(*args)
    except Exception:
        return {"error": traceback.format_exc()}


def spec_origin(name):
    try:
        spec = importlib.util.find_spec(name)
    except Exception as error:
        return f"error: {error!r}"
    if spec is None:
        return None
    return spec.origin or list(spec.submodule_search_locations or [])


def path_candidates(name):
    """Every sys.path entry that holds a top-level module or package called ``name``, in lookup order."""
    return [entry for entry in sys.path
            if (Path(entry or ".") / name).is_dir() or (Path(entry or ".") / f"{name}.py").is_file()]


def src_env():
    """Inherit the environment; append src/ only if missing, so the container's path order is kept."""
    entries = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if not any(Path(entry).resolve() == SRC for entry in entries):
        entries.append(str(SRC))
    return dict(os.environ, PYTHONPATH=os.pathsep.join(entries))


# --- environment.json --------------------------------------------------------------------------------------

def git_commit():
    if os.environ.get("S2R_GIT_COMMIT"):
        return {"source": "S2R_GIT_COMMIT", "commit": os.environ["S2R_GIT_COMMIT"]}
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return {"source": "git", "error": result.stderr.strip()}
    return {"source": "git", "commit": result.stdout.strip()}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception as error:
        return f"error: {error!r}"


def environment():
    # Nothing in this process launches Kit, so these lookups are "before any Kit launch".
    return {
        "python": sys.version,
        "executable": sys.executable,
        "sys_path": list(sys.path),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "find_spec_utils": spec_origin("utils"),
        "find_spec_shared": spec_origin("shared"),
        # sys.path[0] here is this script's folder, which normal `python -m` programs do not have.
        "utils_candidates": path_candidates("utils"),
        "shared_candidates": path_candidates("shared"),
        "versions": {name: package_version(name) for name in ("torch", "isaaclab", "isaacsim", "numpy", "pytest")},
        "git": guarded(git_commit),
        "selection_env": {key: value for key, value in os.environ.items() if key.startswith("S2R_")},
    }


# --- pytest_outcomes.json ----------------------------------------------------------------------------------

def junit_outcomes(path):
    outcomes = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        file, classname, name = case.get("file"), case.get("classname", ""), case.get("name")
        module = file[:-3].replace("/", ".") if file and file.endswith(".py") else classname
        parts = [file or classname.replace(".", "/") + ".py"]
        if classname.startswith(module + "."):
            parts += classname[len(module) + 1:].split(".")
        nodeid = "::".join(parts + [name])
        children = {child.tag for child in case}
        outcome = ("error" if "error" in children else "failed" if "failure" in children
                   else "skipped" if "skipped" in children else "passed")
        outcomes[nodeid] = outcome
    return outcomes


def pytest_outcomes():
    with tempfile.TemporaryDirectory() as temporary:
        junit = Path(temporary) / "junit.xml"
        command = [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider", "-rA",
                   f"--junitxml={junit}", "-o", "junit_family=xunit1"]
        result = subprocess.run(command, cwd=ROOT, env=src_env(), capture_output=True, text=True)
        lines = result.stdout.splitlines()
        summary = next((line for line in reversed(lines) if re.search(r"\b(passed|failed|errors?|skipped)\b", line)), None)
        data = {"command": command[1:], "pythonpath": src_env()["PYTHONPATH"], "exit_code": result.returncode,
                "summary": summary,
                "short_summary": [line for line in lines if line.startswith(("PASSED", "FAILED", "ERROR", "SKIPPED",
                                                                                 "XFAIL", "XPASS"))],
                "stderr_tail": result.stderr[-4000:]}
        data["outcomes"] = guarded(junit_outcomes, junit)
        counts = {}
        for outcome in (data["outcomes"].values() if "error" not in data["outcomes"] else ()):
            counts[outcome] = counts.get(outcome, 0) + 1
        data["counts"] = counts
        return data


# --- env_cfg/<task>__<variant>.json ------------------------------------------------------------------------

def json_safe(value, _active=None):
    """Plain JSON: tuples to lists, callables to 'module:qualname', other objects to an address-free repr."""
    active = _active if _active is not None else set()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if id(value) in active:
        return "<cycle>"
    active = active | {id(value)}
    if isinstance(value, dict):
        return {str(key): json_safe(item, active) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item, active) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((json_safe(item, active) for item in value), key=json.dumps)
    if callable(value) and hasattr(value, "__qualname__"):
        return f"{getattr(value, '__module__', None)}:{value.__qualname__}"
    if hasattr(value, "tolist"):
        return json_safe(value.tolist(), active)
    if dataclasses.is_dataclass(value):
        return {field.name: json_safe(getattr(value, field.name), active) for field in dataclasses.fields(value)}
    return re.sub(r" at 0x[0-9a-fA-F]+", "", repr(value))


def task_variants():
    sys.path.insert(0, str(SRC))
    from core.tasks.base import load_task_spec
    from utils.workspace import available

    return {name: load_task_spec(name).all_variants() for name in available("tasks", "task.toml") if not name.startswith("_")}


def one_cfg(args):
    """Child mode: launch Kit headless for one (task, variant), dump its parsed cfg, then close."""
    from sim.app import launch

    task_name, variant = args.one_cfg
    target = Path(args.out) / "env_cfg" / f"{task_name}__{variant}.json"
    result = {"task": task_name, "variant": variant, "arm": args.arm, "scene": args.scene, "device": args.device}
    selection = argparse.Namespace(arm=args.arm, scene=args.scene, task=task_name, variant=variant)
    app = launch(selection=selection, headless=True, enable_cameras=True, device=args.device)
    try:
        result["find_spec_utils_after_kit"] = spec_origin("utils")
        result["utils_candidates_after_kit"] = path_candidates("utils")
        result["selection_env"] = {key: value for key, value in os.environ.items() if key.startswith("S2R_")}
        from isaaclab_tasks.utils import parse_env_cfg

        import sim.envs as registry

        env_id = registry.TASK.gym_id(variant)
        result["gym_id"] = env_id
        cfg = parse_env_cfg(env_id, device=args.device, num_envs=1)
        try:
            result["cfg"] = json_safe(cfg.to_dict())
        except Exception:
            result["to_dict_error"] = traceback.format_exc()
            result["cfg"] = json_safe(vars(cfg))
    except Exception:
        result["error"] = traceback.format_exc()
    write_json(target, result)
    sys.stdout.flush()
    sys.stderr.flush()
    app.close()


def env_cfgs(args):
    variants = task_variants()
    summary = {}
    for task_name, names in variants.items():
        for variant in names:
            target = Path(args.out) / "env_cfg" / f"{task_name}__{variant}.json"
            if target.exists():
                target.unlink()
            command = [sys.executable, str(Path(__file__).resolve()), "--out", str(args.out), "--arm", args.arm,
                       "--scene", args.scene, "--device", args.device, "--one-cfg", task_name, variant]
            try:
                result = subprocess.run(command, cwd=ROOT, env=src_env(), capture_output=True, text=True,
                                        timeout=CFG_TIMEOUT_S)
                exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
            except subprocess.TimeoutExpired as error:
                exit_code, stdout, stderr = "timeout", str(error.stdout or ""), str(error.stderr or "")
            # The child writes its own file before closing Kit; judge by the file, not the exit code.
            if not target.exists():
                write_json(target, {"task": task_name, "variant": variant, "exit_code": exit_code,
                                    "error": "child wrote no result", "stdout_tail": stdout[-4000:],
                                    "stderr_tail": stderr[-4000:]})
            written = json.loads(target.read_text())
            summary[f"{task_name}__{variant}"] = "error" if "error" in written else "ok"
            print(f"[baseline] env_cfg {task_name}/{variant}: {summary[f'{task_name}__{variant}']}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", default=os.environ.get("S2R_ARM") or "so101")
    parser.add_argument("--scene", default=os.environ.get("S2R_SCENE") or "rpd_meeting_room")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--one-cfg", nargs=2, metavar=("TASK", "VARIANT"), help="internal: dump one env cfg")
    parser.add_argument("--skip-pytest", action="store_true", help="do not rerun the test suite")
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    if args.one_cfg:
        one_cfg(args)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "environment.json", guarded(environment))
    print("[baseline] environment.json written", flush=True)
    if not args.skip_pytest:
        write_json(args.out / "pytest_outcomes.json", guarded(pytest_outcomes))
        print("[baseline] pytest_outcomes.json written", flush=True)
    write_json(args.out / "env_cfg" / "_summary.json", guarded(env_cfgs, args))
    print("[baseline] env_cfg written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
