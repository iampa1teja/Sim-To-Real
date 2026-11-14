"""Static import-boundary rules, checked from the AST without importing the modules under test.

Stdlib only. Relative imports are resolved against src/, so ``from ..x import y`` is judged by where it lands.
Each rule lists offending imports as ``path:line -> module``.
"""
import ast
import sys
from pathlib import Path

import pytest

# --- _locate: the only place later restructure phases should need to edit --------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CORE = SRC / "core"
SHARED_PACKAGE = "utils"          # top-level shared package; the plan's fallback name is "shared"
SHARED = SRC / SHARED_PACKAGE
SIM_ACTIONS = SRC / "sim" / "actions"
STANDALONE_PACKAGES = ("policy_client", "policy_server")   # shipped alone in docker/policy/Dockerfile
# ----------------------------------------------------------------------------------------------------------

ISAAC_ROOTS = {"sim", "pxr", "carb", "warp", "newton", "mujoco_warp"}   # matched as exact top-level names
ISAAC_PREFIXES = ("isaaclab", "isaacsim", "omni")                      # isaaclab_tasks, omni.kit, ...

# Module-top simulator imports under src/core that already existed on v2, as "relative/path.py:module".
# v2 has none; keep this empty unless a violation is a deliberate, reviewed exception.
KNOWN_V2_EXCEPTIONS = set()

# core/sim/shared imports the standalone policy packages already made on v2, as "relative/path.py:module".
# v2 has none: they import each other only as bare ``policy_client.*`` (policy_server/launcher.py:9,
# policy_server/server.py:8), which is how they are laid out in the policy image.
POLICY_ALLOWED_V2 = set()


def _python_files(directory):
    return sorted(path for path in Path(directory).rglob("*.py") if "__pycache__" not in path.parts)


def _is_type_checking(test):
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _package(path):
    """Dotted package that holds ``path``, relative to src/ (e.g. src/core/a/b.py -> ['core', 'a'])."""
    parts = list(Path(path).resolve().relative_to(SRC).parts)
    return parts[:-1]


def _absolute(level, module, package):
    """Resolve ``from <level dots><module>`` against the file's package; climbing above src/ is an error marker."""
    if level == 0:
        return module or ""
    if level - 1 >= len(package):
        return "<relative import above src>"
    base = package[:len(package) - (level - 1)]
    return ".".join(base + ([module] if module else []))


def imports(path, package=None):
    """Yield (line, absolute module, top_level) for every import; relative imports are resolved.

    Top level means executed at import time: module body, class bodies, try/if blocks, but not function
    bodies and not ``if TYPE_CHECKING:`` blocks.
    """
    package = _package(path) if package is None else package
    tree = ast.parse(Path(path).read_text(), filename=str(path))

    def visit(nodes, top):
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield node.lineno, alias.name, top
            elif isinstance(node, ast.ImportFrom):
                yield node.lineno, _absolute(node.level, node.module, package), top
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                yield from visit(ast.iter_child_nodes(node), False)
            elif isinstance(node, ast.If) and _is_type_checking(node.test):
                yield from visit(node.body, False)
                yield from visit(node.orelse, top)
            else:
                yield from visit(ast.iter_child_nodes(node), top)

    yield from visit(tree.body, True)


def _root(module):
    return module.split(".")[0]


def is_isaac(module):
    root = _root(module)
    return root in ISAAC_ROOTS or root.startswith(ISAAC_PREFIXES)


def _relative(path):
    return path.relative_to(ROOT).as_posix()


def _report(rows):
    return "\n".join(f"{file}:{line} -> {module}" for file, line, module in rows)


def _require_files(directory):
    files = _python_files(directory)
    assert files, f"No Python files found under {directory}; update the _locate section"
    return files


def test_collector_classifies_top_level_and_lazy_imports(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import typing\n"
        "from typing import TYPE_CHECKING\n"
        "import pxr\n"
        "class A:\n"
        "    import omni.kit\n"
        "try:\n"
        "    import isaaclab_tasks\n"
        "except ImportError:\n"
        "    pass\n"
        "def f():\n"
        "    import sim.app\n"
        "if TYPE_CHECKING:\n"
        "    import warp\n"
        "if typing.TYPE_CHECKING:\n"
        "    import newton\n"
        "from . import sibling\n"
        "from ..arms.spec import load\n"
        "import simple_parsing\n")
    found = {module: top for _, module, top in imports(sample, package=["core", "policy_client"])}
    assert found == {"typing": True, "pxr": True, "omni.kit": True, "isaaclab_tasks": True, "sim.app": False,
                     "warp": False, "newton": False, "core.policy_client": True, "core.arms.spec": True,
                     "simple_parsing": True}
    assert [m for m in found if is_isaac(m)] == ["pxr", "omni.kit", "isaaclab_tasks", "sim.app", "warp", "newton"]


def test_core_has_no_module_top_simulator_imports():
    offending, seen = [], set()
    for path in _require_files(CORE):
        for line, module, top in imports(path):
            if top and is_isaac(module):
                key = f"{_relative(path)}:{module}"
                seen.add(key)
                if key not in KNOWN_V2_EXCEPTIONS:
                    offending.append((_relative(path), line, module))
    assert not offending, "src/core imports simulator modules at module top:\n" + _report(offending)
    stale = KNOWN_V2_EXCEPTIONS - seen
    assert not stale, f"KNOWN_V2_EXCEPTIONS entries no longer occur; remove them: {sorted(stale)}"


def test_shared_package_is_independent_of_core_sim_and_cli():
    if not SHARED.is_dir():
        pytest.skip(f"{_relative(SHARED)} does not exist yet")
    offending = []
    for path in _require_files(SHARED):
        for line, module, top in imports(path):
            if _root(module) in ("core", "sim", "cli") or (top and is_isaac(module)):
                offending.append((_relative(path), line, module))
    assert not offending, f"src/{SHARED_PACKAGE} imports core/sim/cli or simulator modules:\n" + _report(offending)


def test_sim_actions_module_tops_are_stdlib_and_numpy_only():
    if not SIM_ACTIONS.is_dir():
        pytest.skip(f"{_relative(SIM_ACTIONS)} does not exist yet")
    allowed = set(sys.stdlib_module_names) | {"numpy", "__future__", SHARED_PACKAGE}
    offending = []
    for path in _require_files(SIM_ACTIONS):
        for line, module, top in imports(path):
            if not top or _root(module) in allowed:
                continue
            if module == "sim.actions" or module.startswith("sim.actions."):
                continue
            offending.append((_relative(path), line, module))
    assert not offending, "src/sim/actions imports non-stdlib modules at module top:\n" + _report(offending)


def _standalone_dirs():
    found = {name: sorted(p for p in SRC.rglob(name) if p.is_dir() and (p / "__init__.py").is_file())
             for name in STANDALONE_PACKAGES}
    for name, dirs in found.items():
        assert len(dirs) == 1, f"Expected one {name} package under {SRC}, found {dirs}; update _locate"
    return [dirs[0] for dirs in found.values()]


def test_standalone_policy_packages_do_not_import_the_repository():
    directories = _standalone_dirs()
    # Imports between the two packages are fine, whether bare (policy_client.x) or resolved (core.policy_client.x).
    inside = tuple(".".join(d.relative_to(SRC).parts) for d in directories) + STANDALONE_PACKAGES
    offending = []
    for directory in directories:
        for path in _require_files(directory):
            for line, module, _ in imports(path):
                if module in inside or module.startswith(tuple(name + "." for name in inside)):
                    continue
                if _root(module) in ("core", "sim", SHARED_PACKAGE) or module.startswith("<"):
                    if f"{_relative(path)}:{module}" not in POLICY_ALLOWED_V2:
                        offending.append((_relative(path), line, module))
    assert not offending, ("Policy client/server are copied alone into the policy image "
                           "(docker/policy/Dockerfile) and must not import core/sim/shared:\n" + _report(offending))
