"""Generate the fixed benchmark using a captured live-scene geometry snapshot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from sim_to_real_so101.utils.pick_place_benchmark import generate_eval_set


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1984)
    parser.add_argument('--n_id', type=int, default=60)
    parser.add_argument('--n_ood', type=int, default=20)
    parser.add_argument('--n_yaw', type=int, default=20)
    parser.add_argument('--min_dist_m', type=float, default=.02)
    parser.add_argument('--ood_margin_m', type=float, nargs=2, default=[.02, .05])
    parser.add_argument('--yaw_mode', choices=['matched', 'radial', 'random'], default='matched')
    parser.add_argument('--scene_json', type=Path,
                        default=Path(__file__).resolve().parents[3]/'eval_sets/pick_place_scene.json',
                        help='Live geometry snapshot from capture_eval_scene.py')
    args = parser.parse_args(argv)
    scene = json.loads(args.scene_json.read_text())
    commit = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', 'HEAD'], text=True).strip()
    data = generate_eval_set(args.dataset, scene, seed=args.seed, n_id=args.n_id, n_ood=args.n_ood,
                             n_yaw=args.n_yaw, min_dist_m=args.min_dist_m, ood_margin_m=args.ood_margin_m,
                             yaw_mode=args.yaw_mode, git_commit=commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'eval_set_id': data['eval_set_id'], 'splits': dict(Counter(s['split'] for s in data['starts'])),
                      'zones': {split: dict(Counter(s['zone'] for s in data['starts'] if s['split'] == split))
                                for split in ('id', 'ood', 'yaw', 'train')}}, indent=2))


if __name__ == '__main__':
    main()
