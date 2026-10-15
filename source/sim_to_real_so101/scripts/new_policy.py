from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


def validate_name(name: str) -> None:
    if not re.fullmatch(r"[a-z0-9_]+", name) or name.startswith("_"):
        raise ValueError(f"Invalid policy name {name!r}: use lowercase letters, digits, and underscores, without starting with '_'.")


def update_toml_name(path: Path, section: str, name: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_section = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        match = re.match(r"\[([^\]]+)\]\s*(?:#.*)?$", stripped)
        if match:
            in_section = match.group(1) == section
            continue

        if in_section and re.match(r"name\s*=", stripped):
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f'{indent}name = "{name}"'
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return

    raise ValueError(f"Could not find name in [{section}] in {path}")


def create_policy(name: str, policies_dir: Path) -> Path:
    validate_name(name)

    policies_dir = Path(policies_dir)
    template = policies_dir / "_template"
    target = policies_dir / name

    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Policy already exists: {target}")

    if not template.is_dir():
        raise FileNotFoundError(f"Template directory not found: {template}")

    shutil.copytree(template, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv", ),)

    try:
        update_toml_name(target / "policy.toml", "policy", name)
        update_toml_name(target / "pyproject.toml", "project", name)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise

    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a policy plugin from policies/_template.")
    parser.add_argument("name", help="Name of the new policy.")
    parser.add_argument("--policies-dir", type=Path, default=Path(__file__).resolve().parents[3] / "policies", help="Directory containing _template.", )
    args = parser.parse_args(argv)

    try:
        target = create_policy(args.name, args.policies_dir)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Created policy: {target}")
    print("Next steps:")
    print(f"  1. Edit {target / 'policy.toml'}")
    print(f"  2. Implement {target / 'server.py'} and {target / 'client.py'}")
    print(f"  3. Configure dependencies in {target / 'pyproject.toml'}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())