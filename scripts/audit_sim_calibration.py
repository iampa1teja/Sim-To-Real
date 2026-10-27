"""Inspect the original arm with pxr in the teleop image, before changing it."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def joint_frame(position, rotation):
    matrix = Gf.Matrix4d(1).SetRotate(Gf.Quatd(rotation))
    matrix.SetTranslateOnly(Gf.Vec3d(position))
    return matrix


def forward_kinematics(stage, angles):
    result, cache = {}, UsdGeom.XformCache()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        parent, child = joint.GetBody0Rel().GetTargets()[0], joint.GetBody1Rel().GetTargets()[0]
        parent_matrix = result.get(str(parent), cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent)))
        frame0 = joint_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
        frame1 = joint_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
        rotation = Gf.Matrix4d(1).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(angles[prim.GetName()])))
        result[str(child)] = frame1.GetInverse() * rotation * frame0 * parent_matrix
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="2055ffd33d53b773433871263ac3354800911286")
    args = parser.parse_args()
    assets = ROOT / "source/sim_to_real_so101/assets"
    def baseline_file(relative):
        return subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT),
                                        "show", f"{args.baseline_ref}:{relative}"])
    # The obsolete layers are recovered only in a temporary audit directory,
    # so this investigation remains reproducible after their removal.
    temporary = tempfile.TemporaryDirectory()
    audit_assets = Path(temporary.name)
    (audit_assets / "SO-ARM101-USD.usd").symlink_to(assets / "usd/SO-ARM101-USD.usd")
    for name in ("ARM_new.usd", "SO-ARM101-USD-NO-CAMERA.usd"):
        (audit_assets / name).write_bytes(baseline_file(f"source/sim_to_real_so101/assets/usd/{name}"))
    stage = Usd.Stage.Open(str(assets / "usd/SO-ARM101-USD.usd"))
    variant = Usd.Stage.Open(str(audit_assets / "ARM_new.usd"))
    limits = {p.GetName(): [UsdPhysics.RevoluteJoint(p).GetLowerLimitAttr().Get(),
                          UsdPhysics.RevoluteJoint(p).GetUpperLimitAttr().Get()]
              for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)}
    changes = []
    for prim in variant.Traverse():
        original = stage.GetPrimAtPath(prim.GetPath())
        if not original:
            changes.append(dict(new_prim=str(prim.GetPath())))
            continue
        for attribute in prim.GetAttributes():
            old = original.GetAttribute(attribute.GetName())
            if not old or attribute.Get() != old.Get():
                changes.append(dict(path=str(attribute.GetPath()), base=str(old.Get()) if old else None,
                                    variant=str(attribute.Get())))
        for relation in prim.GetRelationships():
            old = original.GetRelationship(relation.GetName())
            if not old or old.GetTargets() != relation.GetTargets():
                changes.append(dict(changed_relationship=str(relation.GetPath())))
    for prim in stage.Traverse():
        if not variant.GetPrimAtPath(prim.GetPath()):
            changes.append(dict(removed_prim=str(prim.GetPath())))
    cameras = {}
    for filename in ("SO-ARM101-USD.usd", "SO-ARM101-USD-NO-CAMERA.usd", "ARM_new.usd"):
        asset = Usd.Stage.Open(str(audit_assets / filename))
        cameras[filename] = [str(p.GetPath()) for p in asset.TraverseAll() if p.IsA(UsdGeom.Camera)]
    error = 0.0
    for seed in range(100):
        names = list(limits)
        angles = dict(zip(names, np.random.default_rng(seed).uniform(-60, 60, len(names))))
        base_angles = dict(angles, Wrist_Roll=angles["Wrist_Roll"] + 90)
        base, other = forward_kinematics(stage, base_angles), forward_kinematics(variant, angles)
        error = max(error, max(float(np.max(np.abs(np.asarray(base[k]) - np.asarray(other[k])))) for k in base))
    tree = ast.parse(baseline_file("source/sim_to_real_so101/utils/lerobot_interface.py").decode())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LeRobotSO101Interface")
    bounds = next(ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "SO101_USD_MAPPING" for t in n.targets))
    setup = json.loads(baseline_file("source/sim_to_real_so101/assets/real_setup.json"))
    result = dict(baseline_ref=args.baseline_ref, usd_limits=limits,
                  old_mapping={**bounds, **setup["sim_joint_mapping"]}, cameras=cameras,
                  arm_variant_changes=changes, wrist_base_equals_variant_plus_degrees=90,
                  wrist_fk_100_poses_max_matrix_difference=error,
                  calibration=json.loads(args.calibration.read_text()),
                  calibration_sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                  source_hashes=snapshot(ROOT / "datasets/pick_place_v1"),
                  real_hashes=snapshot(ROOT / "datasets/pick_place_v1-real"))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("hashes")}, indent=2))
    temporary.cleanup()


if __name__ == "__main__":
    main()
