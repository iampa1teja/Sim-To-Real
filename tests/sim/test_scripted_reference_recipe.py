"""The scripted reference start keeps its physical jaw opening across unit codecs."""
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from utils.arms.calibration import resolve_calibration
from utils.arms.mapping import JointMapping
from utils.arms.arms import load_arm_spec
from sim.datagen.scripted import GenerationContext


def test_scripted_reference_start_preserves_the_historical_physical_jaw_opening(monkeypatch):
    monkeypatch.delenv("S2R_ARMS_DIR", raising=False)
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "workspace/configs/datagen/scripted.yaml").read_text())
    # Phase 5's retained 391-frame reference used this seeded task sample, without starts_dir.
    assert config["seed"] == 101
    assert not config.get("starts_dir")
    arm = load_arm_spec(config["arm"])[0]
    current = resolve_calibration(arm, unit=config["unit"], environ={})
    historical = resolve_calibration(arm, unit="rpd_robot", environ={})
    targets = []
    for selection, fraction in ((current, config["expert_options"]["gripper_open_fraction"]),
                                (historical, 0.45)):
        mapping = JointMapping.from_arm(arm, selection.mapping, device="cpu")
        context = SimpleNamespace(arm=arm, mapping=mapping)
        command = GenerationContext.gripper_target(context, torch.zeros((1, len(arm.joints))), fraction)
        targets.append(mapping.sim_to_physical(command))
    grippers = [index for index, joint in enumerate(arm.joints) if joint.role == "gripper"]
    torch.testing.assert_close(targets[0][:, grippers], targets[1][:, grippers], atol=1e-5, rtol=0)
