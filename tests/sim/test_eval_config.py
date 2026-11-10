"""Evaluation YAML must remain usable with the Lab 3 launcher argument set."""
from sim.eval import run


def test_gui_yaml_without_sdk_headless_flag(tmp_path, monkeypatch):
    def launcher_flags(parser):
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--kit_args", default="")

    monkeypatch.setattr(run, "add_launch_arguments", launcher_flags)
    monkeypatch.setattr(run, "apply_selection", lambda selection: None)
    config = tmp_path / "eval.yaml"
    config.write_text("policy:\n  name: example\neval:\n  action_horizon: 3\nrun:\n  gui: false\n")
    args = run.parse_args(["--config_path", str(config), "--action_horizon", "5"])
    assert args.headless is True
    assert args.action_horizon == 5
    config.write_text(config.read_text().replace("gui: false", "gui: true"))
    assert run.parse_args(["--config_path", str(config)]).headless is False
