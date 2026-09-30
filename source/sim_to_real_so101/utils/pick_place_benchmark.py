"""CPU-only fixed-start benchmark. Metres and radians in robot base-link frame."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import numpy as np
from .pick_place_eval import (RecordedStarts, footprint,
                             footprints_overlap, resting_pose, rotation_matrix,
                             upright_orientation, results_report)

SPLITS = ('id', 'ood', 'yaw', 'train')
STAGES = ('reached', 'grasped', 'lifted', 'over_box', 'placed', 'success')
FAILURES = ('never_reached', 'missed_grasp', 'dropped_before_box', 'dropped_outside',
            'placed_not_confirmed', 'timeout_holding')


def hull(points):
    """CCW monotone-chain convex hull, without a SciPy dependency."""
    points = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    halves = []
    for ordered in (points, points[::-1]):
        half = []
        for p in ordered:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.extend(half[:-1])
    if len(halves) < 3:
        raise ValueError('Training starts must cover a non-degenerate 2D convex hull')
    return np.array(halves)


def inside_polygon(point, polygon):
    polygon = np.asarray(polygon)
    edges = np.roll(polygon, -1, axis=0) - polygon
    delta = np.asarray(point) - polygon
    return bool(np.all(edges[:, 0]*delta[:, 1] - edges[:, 1]*delta[:, 0] >= -1e-12))


def hull_distance(point, polygon):
    polygon = np.asarray(polygon)
    edges = np.roll(polygon, -1, axis=0) - polygon
    t = np.clip(np.sum((point-polygon)*edges, axis=1) / np.sum(edges*edges, axis=1), 0, 1)
    return float(np.linalg.norm(point - (polygon + t[:, None]*edges), axis=1).min())


def face_yaw(quaternion):
    rotation = rotation_matrix(quaternion)
    axis = np.argmax(np.linalg.norm(rotation[:2], axis=0))
    return float(np.arctan2(rotation[1, axis], rotation[0, axis]) % (np.pi/2))


def wrap_yaw(angle):
    """Square-symmetric relative yaw in [-pi/4, pi/4)."""
    return (np.asarray(angle) + np.pi/4) % (np.pi/2) - np.pi/4


def sample_yaw(xy, training_xy, training_relative_yaws, mode, rng):
    """Return (absolute yaw modulo pi/2, relative yaw, copied source index)."""
    bearing = math.atan2(xy[1], xy[0])
    source = int(np.argmin(np.linalg.norm(training_xy-xy, axis=1))) if mode == 'matched' else None
    if mode == 'matched':
        yaw = (bearing + training_relative_yaws[source]) % (np.pi/2)
    elif mode == 'radial':
        yaw = bearing % (np.pi/2)
    elif mode == 'random':
        yaw = rng.uniform(0, np.pi/2)
    else:
        raise ValueError('Unknown yaw mode')
    return float(yaw), float(wrap_yaw(yaw-bearing)), source


def circular_mean(angles, period):
    mean = np.mean(np.exp(2j*np.pi*np.asarray(angles)/period))
    if abs(mean) < 1e-6:
        raise ValueError('Recorded orientations have no stable circular alignment rule')
    return float(np.angle(mean)*period/(2*np.pi))


def pose_for_xy(xy, yaw, scene):
    q = upright_orientation(scene['base_quaternion_wxyz'], scene['table_plane_world'],
                            [math.cos(yaw/2), 0, 0, math.sin(yaw/2)])
    world, rotation = resting_pose(np.r_[xy, 0], q, scene['base_position_world'],
                                  scene['base_quaternion_wxyz'], scene['table_plane_world'], scene['cube_size_m'])
    return q, world, rotation


def clear_of_box(xy, yaw, scene):
    _, world, rotation = pose_for_xy(xy, yaw, scene)
    box = footprint(scene['box_position_world'], rotation_matrix(scene['box_quaternion_wxyz']),
                    scene['box_size_m'], bottom_origin=True)
    return not footprints_overlap(footprint(world, rotation, scene['cube_size_m']), box)


def supported_on_table(xy, yaw, scene):
    _, world, rotation = pose_for_xy(xy, yaw, scene)
    base_rotation = rotation_matrix(scene['base_quaternion_wxyz'])
    centre = base_rotation.T @ (world-np.asarray(scene['base_position_world']))
    projected = footprint(centre, base_rotation.T @ rotation, scene['cube_size_m'])
    return all(inside_polygon(p, scene['table_outline_base_xy']) for p in projected)


def eval_set_digest(data):
    payload = {key: data[key] for key in ('schema_version', 'parameters', 'scene', 'training_region', 'starts')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def generate_eval_set(dataset, scene, *, seed=1984, n_id=60, n_ood=20, n_yaw=20,
                      min_dist_m=.02, ood_margin_m=(.02, .05), yaw_mode='matched', git_commit='unknown'):
    if any(type(n) is not int or n < 0 for n in (n_id, n_ood, n_yaw)) or n_yaw > n_id:
        raise ValueError('Counts must be nonnegative integers, with n_yaw <= n_id')
    if (not math.isfinite(min_dist_m) or min_dist_m < 0 or len(ood_margin_m) != 2
            or not np.isfinite(ood_margin_m).all() or not 0 < ood_margin_m[0] < ood_margin_m[1]):
        raise ValueError('Require min_dist_m >= 0 and 0 < ood_margin_m[0] < ood_margin_m[1]')
    if yaw_mode not in ('matched', 'radial', 'random'):
        raise ValueError('yaw_mode must be matched, radial or random')
    recorded = RecordedStarts(Path(dataset)/'pick_place_meta', robot_states_root=dataset)
    xy = np.array([p[:2] for p, _ in recorded.starts])
    polygon = hull(xy)
    bearings = np.arctan2(xy[:, 1], xy[:, 0])
    # Pan coverage means the polar sweep of the recorded cube starts. Resting
    # first-frame arm states are used for resetting, not inverse kinematics.
    pan_offset = circular_mean(bearings, 2*np.pi)
    pan = (bearings-pan_offset+np.pi) % (2*np.pi)-np.pi
    pan_limits = [float(pan.min()-np.deg2rad(10)), float(pan.max()+np.deg2rad(10))]
    recorded_yaws = np.array([face_yaw(q) for _, q in recorded.starts])
    relative_yaws = wrap_yaw(recorded_yaws-bearings)
    mean_state = np.mean(recorded.robot_states, axis=0).tolist()
    rng = np.random.default_rng(seed)
    starts = []
    def row(split, p, yaw, **extra):
        return dict(id=f'{split}-{sum(s["split"] == split for s in starts):03d}', split=split,
                    x=float(p[0]), y=float(p[1]), yaw=float(yaw), zone='OUT' if split == 'ood' else recorded.zone(p),
                    arm_start_state=mean_state.copy(),
                    rel_yaw=float(wrap_yaw(yaw-math.atan2(p[1], p[0]))), source_start_id=None, **extra)
    for split, count in (('id', n_id), ('ood', n_ood)):
        accepted = 0
        margin = ood_margin_m[1] if split == 'ood' else 0
        for _ in range(500000):
            if accepted == count:
                break
            p = rng.uniform(xy.min(0)-margin, xy.max(0)+margin)
            inside = inside_polygon(p, polygon)
            if split == 'id' and not inside:
                continue
            if split == 'ood':
                distance = hull_distance(p, polygon)
                estimated_pan = (np.arctan2(p[1], p[0])-pan_offset+np.pi) % (2*np.pi)-np.pi
                if inside or not ood_margin_m[0] <= distance <= ood_margin_m[1] or not pan_limits[0] <= estimated_pan <= pan_limits[1]:
                    continue
            if np.linalg.norm(xy-p, axis=1).min() < min_dist_m:
                continue
            yaw, _, source = sample_yaw(p, xy, relative_yaws, yaw_mode, rng)
            if not clear_of_box(p, yaw, scene) or not supported_on_table(p, yaw, scene):
                continue
            paired_yaw = rng.uniform(0, np.pi/2) if split == 'id' and accepted < n_yaw else None
            if paired_yaw is not None and (not clear_of_box(p, paired_yaw, scene) or not supported_on_table(p, paired_yaw, scene)):
                continue
            starts.append(row(split, p, yaw, **({'paired_yaw': float(paired_yaw)} if paired_yaw is not None else {})))
            starts[-1]['source_start_id'] = f'train-{source:03d}' if source is not None else None
            accepted += 1
        if accepted != count:
            raise ValueError(f'Cannot fit {count} {split} starts with these constraints (accepted {accepted})')
    for source in list(starts[:n_yaw]):
        starts.append(row('yaw', [source['x'], source['y']], source.pop('paired_yaw'), paired_id=source['id']))
    for index, (p, q) in enumerate(recorded.starts):
        starts.append(row('train', p[:2], face_yaw(q), orientation_wxyz=q.tolist(), source_episode=recorded.episode_ids[index]))
        starts[-1]['source_start_id'] = starts[-1]['id']
    data = dict(schema_version=1, reference_frame='robot base link (base)', units='m', angle_units='rad',
                dataset_path=str(Path(dataset).resolve()), seed=seed, git_commit=git_commit,
                created_at=datetime.now(timezone.utc).isoformat(), scene=scene,
                parameters=dict(seed=seed, n_id=n_id, n_ood=n_ood, n_yaw=n_yaw, min_dist_m=min_dist_m,
                                ood_margin_m=list(ood_margin_m), yaw_mode=yaw_mode, require_table_support=True),
                training_region=dict(hull_xy=polygon.tolist(), low=recorded.low.tolist(), high=recorded.high.tolist(),
                                     pan_limits_rad=pan_limits, pan_centre_rad=pan_offset,
                                     pan_rule='cube-start azimuth relative to circular centre, expanded by 10 degrees',
                                     rel_yaw_distribution=dict(mean=float(relative_yaws.mean()), std=float(relative_yaws.std()),
                                                               min=float(relative_yaws.min()), max=float(relative_yaws.max())),
                                     yaw_rule='matched: bearing plus nearest recorded relative yaw; radial: bearing; random: uniform [0, pi/2)' ), starts=starts)
    data['eval_set_id'] = eval_set_digest(data)
    validate_eval_set(data)
    return data


def validate_eval_set(data):
    if data.get('schema_version') != 1 or data.get('reference_frame') != 'robot base link (base)' or data.get('units') != 'm' or data.get('angle_units') != 'rad':
        raise ValueError('Unsupported eval set schema or coordinate units')
    if data.get('eval_set_id') != eval_set_digest(data):
        raise ValueError('Eval set content hash mismatch')
    ids = set()
    for start in data['starts']:
        if start['id'] in ids or start['split'] not in SPLITS:
            raise ValueError('Duplicate start ID or invalid split')
        ids.add(start['id'])
        if not np.isfinite([start['x'], start['y'], start['yaw']]).all():
            raise ValueError('Nonfinite start pose')
        state = np.asarray(start['arm_start_state'])
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError('Arm start state must contain six finite values')
    if not ids:
        raise ValueError('Empty eval set')
    return data


def load_eval_set(path):
    return validate_eval_set(json.loads(Path(path).read_text()))


class EvalSetStarts:
    """Ordered trials, repeated with deterministic seeds; same reset interface."""
    def __init__(self, path, splits=None, repeats=1, seed=1984):
        self.data = load_eval_set(path)
        if isinstance(splits, str):
            splits = splits.split(',')
        available = {s['split'] for s in self.data['starts']}
        if repeats < 1 or (splits is not None and (len(splits) != len(set(splits)) or not set(splits) <= available)):
            raise ValueError('repeats must be positive; splits must be unique names present in eval set')
        self.starts = [s for s in self.data['starts'] if splits is None or s['split'] in splits]
        if not self.starts:
            raise ValueError('No selected eval starts')
        self.schedule = [dict(s, repeat=r, policy_seed=int(np.random.SeedSequence([seed, r, self.data['starts'].index(s)]).generate_state(1)[0]))
                         for r in range(repeats) for s in self.starts]
        self.cursor = 0
        self.current = None

    def sample(self, accepts_random=None):
        # Isaac performs one unused automatic reset on the final done step.
        index = min(self.cursor, len(self.schedule)-1)
        self.current = self.schedule[index]
        self.cursor += 1
        s = self.current
        q = s.get('orientation_wxyz', [math.cos(s['yaw']/2), 0, 0, math.sin(s['yaw']/2)])
        return index, np.array([s['x'], s['y'], 0.]), np.asarray(q), s['zone']

    def robot_state(self, index):
        return np.array(self.schedule[index]['arm_start_state'])


def wilson_interval(successes, episodes):
    if not 0 <= successes <= episodes:
        raise ValueError('Require 0 <= successes <= episodes')
    if episodes == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes/episodes
    denom = 1+z*z/episodes
    centre = (p+z*z/(2*episodes))/denom
    half = z*math.sqrt(p*(1-p)/episodes+z*z/(4*episodes**2))/denom
    return [max(0., centre-half), min(1., centre+half)]


def mcnemar(first, second):
    """Exact two-sided paired binomial McNemar, one Boolean per shared start."""
    if len(first) != len(second):
        raise ValueError('Paired observations must have equal length')
    b = sum(bool(a) and not bool(c) for a, c in zip(first, second))
    c = sum(not bool(a) and bool(d) for a, d in zip(first, second))
    n = b+c
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(b, c)+1)) / (2**n)) if n else 1.
    return dict(shared_starts=len(first), first_only=b, second_only=c, discordant=n,
                p_value=p, method='exact two-sided McNemar')


class StageTrace:
    def __init__(self):
        self.stages = dict.fromkeys(STAGES, False)
        self.holding = False

    def observe(self, *, distance, grasped, lift, min_lift, held, inside_box, placed, success=False):
        for name, value in dict(reached=distance <= .03, grasped=grasped, lifted=lift >= min_lift,
                                over_box=held and inside_box, placed=placed, success=success).items():
            self.stages[name] |= bool(value)
        self.holding = bool(held)

    def finish(self, success):
        stages = dict(self.stages, success=bool(success))
        if success:
            failure = None
        elif stages['placed']:
            failure = 'placed_not_confirmed'
        elif not stages['reached']:
            failure = 'never_reached'
        elif not stages['grasped']:
            failure = 'missed_grasp'
        elif self.holding:
            failure = 'timeout_holding'
        elif not stages['over_box']:
            failure = 'dropped_before_box'
        else:
            failure = 'dropped_outside'
        return dict(stages=stages, failure_mode=failure)


def summarize(rows, step_dt):
    n = len(rows)
    successes = sum(r['success'] for r in rows)
    times = [r['success_step']*step_dt for r in rows if r['success']]
    return dict(episodes=n, successes=successes, success_rate=successes/n if n else None,
                success_rate_ci95=wilson_interval(successes, n),
                stage_rates={s: sum(r['stages'][s] for r in rows)/n if n else None for s in STAGES},
                failure_modes={f: sum(r['failure_mode'] == f for r in rows) for f in FAILURES},
                median_time_to_success_s=median(times) if times else None)


def benchmark_report(episodes, eval_set, repeats, **kwargs):
    report = results_report(episodes, **kwargs)
    report.update(schema_version=2, eval_set_id=eval_set['eval_set_id'], eval_set=eval_set, repeats=repeats,
                  ci_method='95% Wilson, episode-level; repeats are not independent starts',
                  overall=summarize(episodes, kwargs['step_dt']))
    for key in ('split', 'zone'):
        report['by_'+key] = {name: summarize([r for r in episodes if r[key] == name], kwargs['step_dt'])
                             for name in sorted({r[key] for r in episodes})}
    report['per_start_consistency'] = {
        name: dict(successes=sum(r['success'] for r in episodes if r['start_id'] == name),
                   repeats=sum(r['start_id'] == name for r in episodes),
                   success_fraction=float(np.mean([r['success'] for r in episodes if r['start_id'] == name])))
        for name in sorted({r['start_id'] for r in episodes})}
    return report


def save_heatmap(report, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    fig, (ax, overview) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
    data = report['eval_set']
    for key, label in (('table_outline_base_xy', 'Table'), ('box_footprint_base_xy', 'Box')):
        polygon = np.array(data['scene'][key])
        polygon = np.vstack([polygon, polygon[0]])
        for target in (ax, overview):
            target.plot(*polygon.T, label=label, color='black' if label == 'Box' else 'tan')
    train = [s for s in data['starts'] if s['split'] == 'train']
    ax.scatter([s['x'] for s in train], [s['y'] for s in train], c='gray', s=15, label='Training starts', alpha=.5)
    overview.scatter([s['x'] for s in train], [s['y'] for s in train], c='gray', s=5)
    overview.set(title='Full tabletop', aspect='equal', xlabel='Base X (m)', ylabel='Base Y (m)')
    cmap = LinearSegmentedColormap.from_list('success', ['#c43c39', '#eac75a', '#22965c'])
    for split, marker in zip(SPLITS, ('o', '^', 's', 'D')):
        starts = [s for s in data['starts'] if s['split'] == split and s['id'] in report['per_start_consistency']]
        if not starts:
            continue
        ax.scatter([s['x'] for s in starts], [s['y'] for s in starts],
                   c=[report['per_start_consistency'][s['id']]['success_fraction'] for s in starts],
                   cmap=cmap, vmin=0, vmax=1, marker=marker, s={'id': 110, 'yaw': 25, 'ood': 65, 'train': 30}[split],
                   edgecolors='black', linewidths=.4, alpha=.8, label=split)
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=ax, label='Success fraction', shrink=.7)
    region = np.array([[s['x'], s['y']] for s in data['starts']] + data['scene']['box_footprint_base_xy'])
    low, high = region.min(0), region.max(0)
    pad = np.maximum(high-low, .01)*.1
    ax.set_xlim(low[0]-pad[0], high[0]+pad[0])
    ax.set_ylim(low[1]-pad[1], high[1]+pad[1])
    ax.set(xlabel='Base-link X (m)', ylabel='Base-link Y (m)', title='Pick/place fixed-start benchmark', aspect='equal')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
