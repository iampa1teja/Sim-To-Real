"""Extract an arm's USD revolute-joint frames into kinematics schema 1."""

import argparse as _argparse
import copy as _copy
import json as _json
import math as _math
import os as _os
import subprocess as _subprocess
import tempfile as _tempfile
from pathlib import Path as _Path

from core.generate.errors import KinematicsError as _KinematicsError


def build_kinematics(joints, base_link, arm, source) -> dict:
    """Convert USD-like rows in stable topological order; leave inputs untouched."""
    try:
        pending = list(joints)
        children = [row["child"] for row in pending]
        if len(set(children)) != len(children) or base_link in children:
            raise ValueError("A child has multiple parents, or the base has a parent")
        names = [row["name"] for row in pending]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate joint name")
        available, output = {base_link}, []
        while pending:
            progressed = False
            for row in pending[:]:
                if row["parent"] not in available:
                    continue
                axis = row["axis"]
                if axis not in ("X", "Y", "Z"):
                    raise ValueError("Unknown joint axis")
                limits = [float(row["lower_deg"]), float(row["upper_deg"])]
                if not all(_math.isfinite(v) for v in limits) or limits[0] >= limits[1]:
                    raise ValueError("Invalid joint limits")
                def frame(position, rotation):
                    p, q = [float(v) for v in position], [float(v) for v in rotation]
                    if len(p) != 3 or len(q) != 4 or not all(_math.isfinite(v) for v in p + q):
                        raise ValueError("Invalid joint frame")
                    return {"position": p, "quat_wxyz": q}
                output.append({"name": row["name"], "type": "revolute", "parent": row["parent"], "child": row["child"],
                               "axis": [float(i == "XYZ".index(axis)) for i in range(3)],
                               "parent_frame": frame(row["pos0"], row["rot0_wxyz"]),
                               "child_frame": frame(row["pos1"], row["rot1_wxyz"]),
                               "limits_rad": [_math.radians(v) for v in limits]})
                available.add(row["child"])
                pending.remove(row)
                progressed = True
            if not progressed:
                raise ValueError("Joint chain is disconnected or cyclic")
        return {"schema_version": 1, "arm": arm, "base_link": base_link, "source": _copy.deepcopy(source), "joints": output}
    except (ValueError, TypeError, KeyError) as exc:
        raise _KinematicsError(f"Cannot extract joint chain: {exc}") from exc


def _write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with _tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = _Path(stream.name)
            _json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            _os.fsync(stream.fileno())
        _os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = _argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm")
    parser.add_argument("--usd")
    parser.add_argument("--out")
    args = parser.parse_args()
    from pxr import Usd, UsdPhysics
    from core.arms import spec as arm_specs
    from core.utils.hashing import file_sha256

    spec, folder = arm_specs.load_arm_spec(args.arm)
    usd = _Path(args.usd).resolve() if args.usd else folder / spec["usd_path"]
    stage = Usd.Stage.Open(str(usd))
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        def quat(value):
            return [float(value.GetReal()), *[float(v) for v in value.GetImaginary()]]
        rows.append({"name": prim.GetName(), "parent": joint.GetBody0Rel().GetTargets()[0].name,
                     "child": joint.GetBody1Rel().GetTargets()[0].name, "axis": joint.GetAxisAttr().Get(),
                     "pos0": list(joint.GetLocalPos0Attr().Get()), "rot0_wxyz": quat(joint.GetLocalRot0Attr().Get()),
                     "pos1": list(joint.GetLocalPos1Attr().Get()), "rot1_wxyz": quat(joint.GetLocalRot1Attr().Get()),
                     "lower_deg": joint.GetLowerLimitAttr().Get(), "upper_deg": joint.GetUpperLimitAttr().Get()})
    try:
        commit = _subprocess.check_output(["git", "-C", str(folder), "rev-parse", "HEAD"], text=True, stderr=_subprocess.DEVNULL).strip()
    except (OSError, _subprocess.CalledProcessError):
        commit = "unknown"
    source = {"usd": _os.path.relpath(usd, folder), "usd_sha256": file_sha256(usd), "git_commit": commit}
    payload = build_kinematics(rows, spec["links"]["base"], args.arm or arm_specs.ARM_NAME, source)
    _write_atomic(_Path(args.out) if args.out else folder / "kinematics.json", payload)


if __name__ == "__main__":
    main()
