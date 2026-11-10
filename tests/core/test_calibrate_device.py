"""Declared LeRobot plugins calibrate in the same process and always disconnect."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

from core.calibration.calibrate_arm import calibrate_device


@pytest.mark.parametrize("role", ["leader", "follower"])
@pytest.mark.parametrize("failure", [None, "connect", "calibrate"])
def test_custom_device_registration_and_calibration_lifecycle(monkeypatch, tmp_path, role, failure):
    events = []
    class Config:
        @staticmethod
        def get_choice_class(name):
            assert events == ["plugin"]
            assert name == "custom_" + role
            return lambda **kwargs: kwargs
    class Device:
        def connect(self, *, calibrate):
            assert not calibrate
            events.append("connect")
            if failure == "connect":
                raise RuntimeError("connect failed after opening transport")
        def calibrate(self):
            events.append("calibrate")
            if failure == "calibrate":
                raise RuntimeError("calibrate failed")
        def disconnect(self):
            events.append("disconnect")
    def make(config):
        assert config == dict(port="/dev/example", id="unit1", use_degrees=False, calibration_dir=tmp_path)
        events.append("make")
        return Device()
    robot_module, teleop_module, utils = ModuleType("lerobot.robots"), ModuleType("lerobot.teleoperators"), ModuleType("lerobot.teleoperators.utils")
    robot_module.RobotConfig, robot_module.make_robot_from_config = Config, make
    teleop_module.TeleoperatorConfig, utils.make_teleoperator_from_config = Config, make
    for name, module in (("lerobot.robots", robot_module), ("lerobot.teleoperators", teleop_module),
                         ("lerobot.teleoperators.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    def register(name):
        assert name == "custom_robot_package"
        events.append("plugin")
    monkeypatch.setattr("core.calibration.calibrate_arm.importlib.import_module", register)
    spec = SimpleNamespace(modules=("custom_robot_package",), follower_type="custom_follower",
                           leader_type="custom_leader", use_degrees=False)
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            calibrate_device(role, spec, port="/dev/example", device_id="unit1", directory=tmp_path)
    else:
        calibrate_device(role, spec, port="/dev/example", device_id="unit1", directory=tmp_path)
    assert events == ["plugin", "make", "connect"] + ([] if failure == "connect" else ["calibrate"]) + ["disconnect"]


@pytest.mark.parametrize("role", ["leader", "follower"])
def test_registered_config_can_omit_optional_degree_flag(monkeypatch, tmp_path, role):
    from dataclasses import dataclass
    @dataclass
    class DeviceConfig:
        port: str
        id: str
        calibration_dir: object
    class Registry:
        @staticmethod
        def get_choice_class(name):
            return DeviceConfig
    events = []
    device = SimpleNamespace(connect=lambda **kwargs: events.append("connect"),
                             calibrate=lambda: events.append("calibrate"), disconnect=lambda: events.append("disconnect"))
    def make(config):
        assert config == DeviceConfig("port", "device1", tmp_path)
        return device
    robot, teleop, utils = ModuleType("lerobot.robots"), ModuleType("lerobot.teleoperators"), ModuleType("lerobot.teleoperators.utils")
    robot.RobotConfig, robot.make_robot_from_config = Registry, make
    teleop.TeleoperatorConfig, utils.make_teleoperator_from_config = Registry, make
    for name, module in (("lerobot.robots", robot), ("lerobot.teleoperators", teleop), ("lerobot.teleoperators.utils", utils)):
        monkeypatch.setitem(sys.modules, name, module)
    spec = SimpleNamespace(modules=(), follower_type="custom", leader_type="custom", use_degrees=False)
    calibrate_device(role, spec, port="port", device_id="device1", directory=tmp_path)
    assert events == ["connect", "calibrate", "disconnect"]


def test_read_live_config_can_omit_optional_degree_flag(monkeypatch, tmp_path):
    from dataclasses import dataclass
    from core.calibration.check_calibration import read_live
    @dataclass
    class DeviceConfig:
        port: str
        id: str
        calibration_dir: object
    class Registry:
        @staticmethod
        def get_choice_class(name):
            return DeviceConfig
    events = []
    bus = SimpleNamespace(connect=lambda: events.append("connect"), is_connected=True,
                          sync_read=lambda *_args, **_kwargs: {"axis": 100},
                          disconnect=lambda **kwargs: events.append("disconnect"))
    def make(config):
        assert config == DeviceConfig("port", "device1", tmp_path)
        return SimpleNamespace(bus=bus)
    module = ModuleType("lerobot.robots")
    module.RobotConfig, module.make_robot_from_config = Registry, make
    monkeypatch.setitem(sys.modules, "lerobot.robots", module)
    monkeypatch.setattr("core.calibration.check_calibration.load_arm_spec", lambda: (
        SimpleNamespace(lerobot=SimpleNamespace(modules=(), follower_type="custom", use_degrees=False)), tmp_path))
    result = read_live("port", "device1", tmp_path / "device1.json", {"axis": dict(range_min=0, range_max=200)})
    assert result[0]["in_range"]
    assert events == ["connect", "disconnect"]
