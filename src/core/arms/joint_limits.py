import math
from collections.abc import Mapping as _Mapping
from core.arms.spec import load_arm_spec
from core.arms import spec as _spec
from core.scenes import load_setup


def joint_name_pairs(spec=None):
    if spec is None:
        spec, _ = load_arm_spec()
    lerobot_names = tuple(spec["lerobot"]["names"])
    usd_names = tuple(spec["joints"]["names"])
    if len(lerobot_names) != len(usd_names):
        raise ValueError(f"arm.toml: [arm.lerobot] names {lerobot_names} and [arm.joints] names {usd_names} differ in length")
    return tuple(zip(lerobot_names, usd_names))


def runtime_joint_limits(setup=None, *, arm=None):
    if setup is None:
        setup = load_setup()
    names = tuple(load_arm_spec(arm)[0]["lerobot"]["names"])
    arms = setup.get("arms", {})
    block = arms.get(arm or _spec.ARM_NAME, setup) if isinstance(arms, _Mapping) else setup
    bounds = block["runtime_joint_limits"]
    if set(bounds) != set(names):
        raise ValueError(f"runtime_joint_limits must specify exactly the follower joints {names}")
    for name, pair in bounds.items():
        values = [pair.get("joint_min"), pair.get("joint_max")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError(f"Runtime limits for {name} must be finite")
        if values[0] >= values[1]:
            raise ValueError(f"Runtime limits for {name} must increase")
    return {name: dict(bounds[name]) for name in names}
