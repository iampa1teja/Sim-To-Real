import math
from core.arms.spec import load_arm_spec
from core.scenes import load_setup
from .joint_calibration import FOLLOWER_JOINTS


def joint_name_pairs(spec=None):
    if spec is None:
        spec, _ = load_arm_spec()
    lerobot_names = tuple(spec["lerobot"]["names"])
    usd_names = tuple(spec["joints"]["names"])
    if len(lerobot_names) != len(usd_names):
        raise ValueError(f"arm.toml: [arm.lerobot] names {lerobot_names} and [arm.joints] names {usd_names} differ in length")
    return tuple(zip(lerobot_names, usd_names))


def runtime_joint_limits(setup=None):
    if setup is None:
        setup = load_setup()
    bounds = setup["runtime_joint_limits"]
    if set(bounds) != set(FOLLOWER_JOINTS):
        raise ValueError(f"runtime_joint_limits must specify exactly the follower joints {FOLLOWER_JOINTS}")
    for name, pair in bounds.items():
        values = [pair.get("joint_min"), pair.get("joint_max")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"Runtime limits for {name} must be finite")
        if values[0] >= values[1]:
            raise ValueError(f"Runtime limits for {name} must increase")
    return {name: dict(bounds[name]) for name in FOLLOWER_JOINTS}
