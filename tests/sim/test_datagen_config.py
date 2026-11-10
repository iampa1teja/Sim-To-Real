"""DR replay YAML uses declared parser options and keeps CLI flags authoritative."""
import pytest

from sim.datagen import dr_replay


def test_dr_replay_config_flags_and_source_validation(tmp_path, monkeypatch):
    def launcher_arguments(parser):
        parser.add_argument('--headless', action='store_true')
        return parser

    monkeypatch.setattr(dr_replay, 'add_launch_arguments', launcher_arguments)
    path = tmp_path / 'datagen.yaml'
    path.write_text('''arm: so101
scene: blank_table
task: pick_place_cube
variant: teleop
source: dr_replay
src_root: original
src_repo_id: local/original
dst_root: generated
dst_repo_id: local/generated
seed: 101
include_originals: true
headless: true
''')
    args = dr_replay.parse_args(['--config_path', str(path), '--seed', '17', '--dst_root', 'override'])
    assert (args.arm, args.scene, args.task, args.variant) == ('so101', 'blank_table', 'pick_place_cube', 'teleop')
    assert args.src_root == 'original' and args.dst_root == 'override'
    assert args.seed == 17 and args.include_originals is True and args.headless is True
    assert args.source == 'dr_replay'
    path.write_text(path.read_text().replace('source: dr_replay', 'source: mimic'))
    with pytest.raises(SystemExit):
        dr_replay.parse_args(['--config_path', str(path)])
