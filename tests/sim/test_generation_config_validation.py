"""Code-bearing Task instances must not enter Isaac's recursive config validator."""
import pytest

pytest.importorskip("isaaclab")

from core.tasks.task import TaskSpec
from sim.datagen.rollout import setup_context
from sim.datagen.scripted import parse_args
from sim.datagen.starts import GridStarts


@pytest.mark.parametrize("name", ["pick_place_cube", "stack_two_cubes", "stack_bowls"])
@pytest.mark.parametrize("mode", ["random", "grid"])
def test_each_tasks_generation_config_validates_without_following_python_modules(tmp_path, monkeypatch, name, mode):
    monkeypatch.setenv("S2R_ARM", "so101")
    args = parse_args(["--destination", str(tmp_path / "dataset"), "--arm", "so101", "--scene", "blank_table",
                       "--task", name, "--unit", "rpd_follower", "--start_mode", mode])

    class Validated(Exception):
        pass

    def capture(cfg):
        params = cfg.events.reset_from_starts.params
        assert isinstance(params["task"], TaskSpec)
        if mode == "grid":
            assert isinstance(params["start_sampler"], GridStarts)
            assert isinstance(params["start_sampler"].spec, TaskSpec)
        cfg.validate()
        raise Validated

    with pytest.raises(Validated):
        setup_context(args, env_factory=capture)
