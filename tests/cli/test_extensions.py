"""Scaffolds, static checking and dry runs work without the simulator SDK."""
from pathlib import Path
import shutil
import pytest

from cli.check import check_static
from cli.main import check_compositions, main
from cli.scaffold import create_extension

REPO = Path(__file__).resolve().parents[2]


def test_each_scaffold_has_a_matching_manifest_and_passes_static_checks(tmp_path):
    for kind, folder, marker in (("arm", "arms", "arm.toml"), ("scene", "scenes", "scene.toml"),
                                  ("task", "tasks", "task.toml"), ("policy", "policies", "policy.toml")):
        shutil.copytree(REPO / folder / "_template", tmp_path / folder / "_template")
        created = create_extension(kind, "test_" + kind, tmp_path)
        assert created == tmp_path / folder / ("test_" + kind)
        assert f'name = "test_{kind}"' in (created / marker).read_text()
        with pytest.raises(FileExistsError):
            create_extension(kind, "test_" + kind, tmp_path)
    rows, errors = check_static(tmp_path, arm="test_arm", scene="test_scene", task="test_task", policy="test_policy")
    assert not errors, errors
    assert len(rows) == 4
    (tmp_path / "policies" / "test_policy" / "policy.toml").write_text('[policy]\nname = "wrong_folder"\n')
    _, errors = check_static(tmp_path)
    assert any("policies/test_policy" in error for error in errors)


def test_new_dry_run_does_not_create_a_folder(tmp_path, capsys):
    (tmp_path / "policies").mkdir()
    assert main(["--dry-run", "new", "arm", "untouched"], repo=tmp_path) == 0
    assert not (tmp_path / "arms").exists()
    assert "-m cli.scaffold arm untouched" in capsys.readouterr().out


def test_default_check_covers_all_extensions_without_a_cartesian_product():
    rows = check_compositions(REPO)
    assert rows == [("so101", "rpd_meeting_room", "pick_place_cube"),
                    ("so101", "blank_table", "pick_place_cube"),
                    ("so101", "rpd_meeting_room", "stack_two_cubes")]
    assert check_compositions(REPO, scene="blank_table", task="stack_two_cubes") == [("so101", "blank_table", "stack_two_cubes")]
