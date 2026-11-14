"""Declared device plugins work without serial, camera, or per-unit calibration assumptions."""
from dataclasses import dataclass, replace
from types import ModuleType

import pytest

pytest.importorskip("lerobot")

from utils.arms.calibration import resolve_calibration
from utils.arms.spec import load_arm_spec
from sim.teleop import devices


@dataclass
class FactoryConfig:
    id: str | None = None


@pytest.mark.parametrize("role", ["leader", "follower"])
def test_factory_config_registers_before_lookup_and_uses_only_declared_fields(synthetic_arm, monkeypatch, role):
    events = []
    arm = replace(synthetic_arm, lerobot=replace(synthetic_arm.lerobot, modules=("custom_factory_plugin",)))
    selection = resolve_calibration(arm, follower_id="unit", leader_id="leader", environ={})
    original_import = devices.importlib.import_module

    def register(name):
        if name == "custom_factory_plugin":
            events.append("register")
            return ModuleType(name)
        return original_import(name)

    def choice(name):
        assert events == ["register"]
        assert name == getattr(arm.lerobot, role + "_type")
        events.append("lookup")
        return FactoryConfig

    monkeypatch.setattr(devices.importlib, "import_module", register)
    registry = devices.RobotConfig if role == "follower" else devices.TeleoperatorConfig
    monkeypatch.setattr(registry, "get_choice_class", staticmethod(choice))
    config = devices.make_device_config(role, "unused-for-this-driver", "unit", {}, 30, selection)
    assert config == FactoryConfig(id="unit")
    assert events == ["register", "lookup"]


def test_config_rejects_requested_cameras_or_options_unsupported_by_driver(synthetic_arm, monkeypatch):
    selection = resolve_calibration(synthetic_arm, environ={})
    monkeypatch.setattr(devices.RobotConfig, "get_choice_class", staticmethod(lambda _: FactoryConfig))
    with pytest.raises(ValueError, match="does not support cameras"):
        devices.make_device_config("follower", "unused", "unit", {"room": {}}, 30, selection)
    with pytest.raises(ValueError, match="does not support options"):
        devices.make_device_config("follower", "unused", "unit", {}, 30, selection, {"unknown": 1})


@pytest.mark.parametrize("role", ["leader", "follower"])
@pytest.mark.parametrize("fails", [False, True])
def test_factory_device_uses_driver_lifecycle_without_reading_unit_calibration(synthetic_arm, monkeypatch, role, fails):
    selection = resolve_calibration(synthetic_arm, follower_id="unit", leader_id="leader", environ={})
    assert getattr(selection, role + "_path") is not None
    events = []

    class Driver:
        is_connected = False
        is_calibrated = True

        def connect(self, *, calibrate):
            assert not calibrate
            events.append("connect")
            self.is_connected = True
            if fails:
                raise RuntimeError("transport failed")

        def disconnect(self):
            events.append("disconnect")
            self.is_connected = False

    def make(config):
        assert config == FactoryConfig(id="unit")
        events.append("make")
        return Driver()

    def no_calibration(*_):
        raise AssertionError("Factory drivers must not load per-unit calibration files")

    registry = devices.RobotConfig if role == "follower" else devices.TeleoperatorConfig
    monkeypatch.setattr(registry, "get_choice_class", staticmethod(lambda _: FactoryConfig))
    monkeypatch.setattr(devices, "make_robot_from_config" if role == "follower" else "make_teleoperator_from_config", make)
    monkeypatch.setattr(devices, "load_device_calibration", no_calibration)
    if fails:
        with pytest.raises(RuntimeError, match="transport failed"):
            devices.connect_device(role, "unused", "unit", {}, 30, selection, calibrate=False)
    else:
        device = devices.connect_device(role, "unused", "unit", {}, 30, selection, calibrate=False)
        assert device.is_connected
        device.disconnect()
        assert not device.is_connected
    assert events == ["make", "connect", "disconnect"]


def test_production_so101_config_keeps_serial_and_normalized_joint_mode():
    arm = load_arm_spec("so101")[0]
    selection = resolve_calibration(arm, unit="rpd_robot", environ={})
    config = devices.make_device_config("follower", "/dev/unopened", "unit", {}, 30, selection)
    assert config.port == "/dev/unopened"
    assert config.id == "unit"
    assert config.use_degrees is False
    assert config.cameras == {}
