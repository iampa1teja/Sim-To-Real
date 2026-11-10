"""Create one extension folder from its versioned template."""
import argparse
from pathlib import Path
import re
import shutil

KINDS = {"arm": ("arms", "arm.toml"), "scene": ("scenes", "scene.toml"),
         "task": ("tasks", "task.toml"), "policy": ("policies", "policy.toml")}


def update_toml_name(path, section, name):
    text = path.read_text()
    pattern = rf'(\[{re.escape(section)}\][^\[]*?\bname\s*=\s*)"[^"]*"'
    text, count = re.subn(pattern, lambda match: match[1] + '"' + name + '"', text, count=1)
    if count != 1:
        raise ValueError(f"{path} needs a name in [{section}]")
    path.write_text(text)


def create_extension(kind, name, repo):
    if kind not in KINDS:
        raise ValueError(f"Unknown extension kind {kind!r}")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Names must start with a lowercase letter and contain only lowercase letters, digits or underscores")
    folder, marker = KINDS[kind]
    template, target = Path(repo) / folder / "_template", Path(repo) / folder / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Extension already exists: {target}")
    if not template.is_dir():
        raise FileNotFoundError(f"Template not found: {template}")
    shutil.copytree(template, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", ".pytest_cache"))
    try:
        update_toml_name(target / marker, kind, name)
        if kind == "policy":
            update_toml_name(target / "pyproject.toml", "project", name)
        if kind == "task":
            path = target / "__init__.py"
            path.write_text(path.read_text().replace('load_task_spec("new_task")', f'load_task_spec("{name}")'))
        if kind == "arm" and (target / "_utils.py").is_file():
            (target / "_utils.py").rename(target / f"{name}_utils.py")
            path = target / marker
            path.write_text(path.read_text().replace('utils = "_utils.py"', f'utils = "{name}_utils.py"'))
    except BaseException:
        shutil.rmtree(target)
        raise
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=KINDS)
    parser.add_argument("name")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    try:
        target = create_extension(args.kind, args.name, args.repo)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")
    print(f"Created {target}; edit its manifest and run s2r check --{args.kind} {args.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
