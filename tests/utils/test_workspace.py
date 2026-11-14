from pathlib import Path
import sys

import pytest

from utils.workspace import available, item_dir, kind_dir, load_file_module, relative_file, text


def test_kind_dir_overrides_keep_their_tilde_handling(monkeypatch):
    for key in ("S2R_ARMS_DIR", "S2R_SCENES_DIR", "S2R_TASKS_DIR"):
        monkeypatch.setenv(key, "~/items")
    assert kind_dir("arms") == kind_dir("scenes") == Path("~/items").expanduser()
    assert kind_dir("tasks") == Path("~/items")


def test_item_lookup_and_listing(monkeypatch, tmp_path):
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "task.toml").write_text("")
    (tmp_path / "beta").mkdir()
    monkeypatch.setenv("S2R_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("S2R_ARMS_DIR", str(tmp_path))
    assert available("tasks", "task.toml") == ["alpha"]
    assert item_dir("arms", "beta") == (tmp_path / "beta").resolve()
    with pytest.raises(ValueError, match="^Arm folder name '_x' must be a public Python identifier$"):
        item_dir("arms", "_x")
    with pytest.raises(ValueError, match="^Scene folder name 'a-b' must"):
        item_dir("scenes", "a-b")


def test_validator_wording_is_the_callers(tmp_path):
    with pytest.raises(ValueError, match=r"^usd_path must be relative to the arm folder without '\.\.'$"):
        relative_file(tmp_path, "../x.usd", "usd_path", "relative to the arm folder")
    with pytest.raises(ValueError, match="^label must be a nonempty string$"):
        text("", "label")
    assert text("", "label", empty=True) == ""


def test_failed_file_module_is_not_cached(tmp_path):
    path = tmp_path / "hook.py"
    path.write_text("raise RuntimeError('broken')\n")
    with pytest.raises(RuntimeError):
        load_file_module(path, "_s2r_test_hook")
    assert "_s2r_test_hook" not in sys.modules
    path.write_text("VALUE = 3\n")
    assert load_file_module(path, "_s2r_test_hook").VALUE == 3
    assert load_file_module(path, "_s2r_test_hook") is sys.modules.pop("_s2r_test_hook")
