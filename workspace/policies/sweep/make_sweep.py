"""Generate a calibrated, speed-limited single-joint sweep (host, PYTHONPATH=src)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def make_sweep(*, arm='so101', unit='rpd_follower', scene='rpd_meeting_room', fps=30,
               hold_s=2.0, speed_deg_s=15.0, holds=9, margin_deg=5.0, exclude=''):
    from utils.arms.spec import load_arm_spec
    from utils.arms.calibration import load_unit
    from utils.arms.mapping import physical_to_units, raw_joint_limits
    from utils.geometry import mount_joint_positions
    from utils.scene import load_scene_spec, load_mount_spec

    if fps <= 0 or hold_s <= 0 or speed_deg_s <= 0 or holds < 1 or margin_deg < 0:
        raise ValueError('fps, hold-s, speed and holds must be positive; margin must be nonnegative')
    hold_ticks = round(hold_s * fps)
    if hold_ticks < 1 or not math.isclose(hold_ticks / fps, hold_s, abs_tol=1e-9):
        raise ValueError('hold-s must be an exact, positive number of ticks')
    spec = load_arm_spec(arm)[0]
    calibration = load_unit(spec, unit)
    names = list(spec.joint_names)
    omitted = set()
    for token in filter(None, exclude.split(',')):
        name, index = token.rsplit(':', 1)
        pair = (name, int(index))
        if name not in names or not 0 <= pair[1] < holds:
            raise ValueError(f'Unknown hold {token!r}; indices are zero based')
        omitted.add(pair)
    mount = load_mount_spec(load_scene_spec(scene), spec)
    starts = mount_joint_positions(spec, mount)
    physical = np.array([np.rad2deg(starts[j.usd]) if j.type == 'revolute' else starts[j.usd]
                         for j in spec.action_joints])
    limits = raw_joint_limits(spec, calibration.mapping)
    raw_min, raw_max = np.array([limits[n] for n in names]).T
    factors = np.array([(calibration.mapping[n]['joint_max'] - calibration.mapping[n]['joint_min'])
                        / (raw_max[i] - raw_min[i]) for i, n in enumerate(names)])
    home = physical_to_units(physical, calibration.mapping, names, gripper_joints=spec.gripper_joint_names)
    home = np.clip(home, raw_min, raw_max)
    if any(not j.limits[0] <= value <= j.limits[1] for j, value in zip(spec.action_joints, physical)):
        raise ValueError('Home is outside runtime joint limits')
    ticks, records = [home.tolist()], []
    current = home.copy()

    def move(destination):
        nonlocal current
        # Smoothstep peaks at 1.5 times average velocity. Omit the endpoint:
        # the following hold supplies it, so each hold has exactly hold_ticks rows.
        count = max(1, math.ceil(1.5 * np.max(np.abs(destination - current) * factors) / speed_deg_s * fps))
        for k in range(1, count):
            t = k / count
            ticks.append(np.clip(current + (destination - current) * t*t*(3 - 2*t), raw_min, raw_max).tolist())
        current = destination.copy()

    for i, joint in enumerate(spec.action_joints):
        span = calibration.mapping[joint.name]
        lower = max(joint.limits[0], span['joint_min']) + margin_deg
        upper = min(joint.limits[1], span['joint_max']) - margin_deg
        if lower > upper:
            raise ValueError(f'Margin leaves no range for {joint.name}')
        for index, angle in enumerate(np.linspace(lower, upper, holds)):
            if (joint.name, index) in omitted:
                continue
            pose = physical.copy()
            pose[i] = angle
            raw = np.clip(physical_to_units(pose, calibration.mapping, names,
                                           gripper_joints=spec.gripper_joint_names), raw_min, raw_max)
            move(raw)
            start = len(ticks)
            ticks.extend([raw.tolist() for _ in range(hold_ticks)])
            counts = []
            for j, n in enumerate(names):
                source = calibration.source['joints'][n]
                counts.append(source['range_min'] + (raw[j] - raw_min[j]) / (raw_max[j] - raw_min[j])
                              * (source['range_max'] - source['range_min']))
            records.append(dict(joint=joint.name, index=index, angle_deg=float(angle), raw=raw.tolist(),
                                start_tick=start, end_tick=len(ticks), expected_encoder_counts=counts))
        move(home)
        ticks.append(home.tolist())
    return dict(fps=fps, unit=unit, arm=arm, scene=scene,
                unit_sha256=hashlib.sha256(calibration.path.read_bytes()).hexdigest(),
                calibration_sha256=calibration.source['calibration_sha256'],
                joint_names=names, home_raw=home.tolist(), deg_per_raw=factors.tolist(),
                raw_min=raw_min.tolist(), raw_max=raw_max.tolist(), hold_s=hold_s,
                speed_deg_s=speed_deg_s, excluded=[f'{n}:{i}' for n, i in sorted(omitted)],
                ticks=ticks, holds=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in [('arm', 'so101'), ('unit', 'rpd_follower'), ('scene', 'rpd_meeting_room')]:
        parser.add_argument('--' + name, default=default)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--hold-s', type=float, default=2.0)
    parser.add_argument('--speed-deg-s', type=float, default=15.0)
    parser.add_argument('--holds', type=int, default=9)
    parser.add_argument('--margin-deg', type=float, default=5.0)
    parser.add_argument('--exclude', default='', help='Comma-separated joint:index holds; indices start at zero')
    parser.add_argument('--out', type=Path, default=Path('workspace/policies/sweep/sweep.json'))
    options = vars(parser.parse_args())
    out = options.pop('out')
    result = make_sweep(**options)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(f'{len(result["ticks"]) / result["fps"]:.3f} seconds ({len(result["ticks"])} ticks), {len(result["holds"])} holds; {out}')


if __name__ == '__main__':
    main()
