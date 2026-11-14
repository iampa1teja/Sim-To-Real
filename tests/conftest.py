"""CPU fixtures for arms with layouts different from the production SO101."""

import pytest

from utils.arms.arms import load_arm_spec


SYNTHETIC_MANIFEST = '''
[arm]
name = "synthetic"
usd_path = "usd/arm.usda"
[arm.links]
base = "base"
ee = "tool"
[[arm.joints]]
name = "pan"
usd = "Pan"
type = "revolute"
role = "arm"
limits = [-180.0, 180.0]
initial = 0.0
[[arm.joints]]
name = "left"
usd = "Left"
type = "prismatic"
role = "gripper"
limits = [0.0, 0.04]
initial = 0.02
[[arm.joints]]
name = "slide"
usd = "Slide"
type = "prismatic"
role = "arm"
limits = [-0.2, 0.2]
initial = 0.0
[[arm.joints]]
name = "right"
usd = "Right"
type = "prismatic"
role = "gripper"
limits = [0.0, 0.04]
initial = 0.02
[arm.gripper]
kind = "parallel"
joints = ["left", "right"]
contact_bodies = ["left_link", "right_link"]
friction_bodies = ["left_link", "right_link"]
[arm.actuators.drive]
joint_names_expr = ["Pan|Left|Slide|Right"]
stiffness = 10.0
damping = 1.0
[arm.lerobot]
follower_type = "synthetic_follower"
leader_type = "synthetic_leader"
modules = []
use_degrees = false
calibration = "factory"
[arm.hardware]
motor = "test"
encoder_counts_per_turn = 4096
zero_count = 2048
'''


@pytest.fixture
def synthetic_arm(tmp_path, monkeypatch):
    directory = tmp_path / "arms/synthetic"
    (directory / "usd").mkdir(parents=True)
    (directory / "usd/arm.usda").write_text('#usda 1.0\n(defaultPrim="base")\ndef Xform "base" {}\n')
    (directory / "arm.toml").write_text(SYNTHETIC_MANIFEST)
    monkeypatch.setenv("S2R_ARMS_DIR", str(directory.parent))
    return load_arm_spec("synthetic")[0]
