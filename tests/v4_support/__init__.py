"""Shared v4 test support (contracts §13). Owned by WP0; other packages use it but do not edit it.

Import as ``from v4_support import temp_env, assert_frozen`` (``tests/conftest.py`` puts ``tests/`` on
``sys.path``). Importing this package needs only the standard library and numpy; helpers that use v4
modules (``factories``) import them lazily inside each function. WP7 adds ``v4_support/harness.py``.

Modules:
    env          temp_env, copy_fixture, copy_repo_folder, clear_task_cache, sandbox, paths
    frozen       assert_frozen
    toml_writer  dump_toml
    fx_arms      synthetic arm data (kinematics.json, arm.toml, setup block), reference FK, write_arm
    tasks        write_task, write_scene, contract example task tables
    factories    make_* / random_* instances of the WP0 protocol dataclasses
    stubs        ListWriter (in-memory EpisodeWriter)
"""

from .env import (FIXTURES_DIR, REPO_ROOT, S2R_VARS, SRC_DIR, TESTS_DIR, Sandbox, clear_task_cache, copy_fixture,
                  copy_repo_folder, sandbox, temp_env)
from .frozen import assert_frozen
from .toml_writer import dump_toml

__all__ = ["FIXTURES_DIR", "REPO_ROOT", "S2R_VARS", "SRC_DIR", "TESTS_DIR", "Sandbox", "assert_frozen",
           "clear_task_cache", "copy_fixture", "copy_repo_folder", "dump_toml", "sandbox", "temp_env"]
