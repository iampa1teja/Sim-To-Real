"""URDF planning prints executable container commands without loading an SDK."""
from pathlib import Path
import subprocess
import sys

import pytest

from cli.main import main
from cli.plan import CliError,plan_new

REPO=Path(__file__).resolve().parents[2]
SOURCE=REPO/'tests/fixtures/urdf/seven_parallel_pkg/robot.urdf.xacro'


def test_urdf_dry_run_passes_roles_packages_and_camera(capsys):
    assert main(['new','arm','imported','--urdf',str(SOURCE),'--ee_link','Tool_0',
                 '--gripper','Grip_L','--camera','Camera_1=wrist_cam','--dry-run'])==0
    command=capsys.readouterr().out
    assert 'python -m sim.arms.urdf_import imported' in command
    assert '--ee_link Tool_0 --gripper Grip_L' in command
    assert '--camera Camera_1=wrist_cam' in command
    assert '--package_path seven_parallel_pkg=' in command


def test_external_description_paths_have_readonly_mounts(tmp_path):
    source=tmp_path/'robot.urdf';source.write_text('<robot/>')
    plan=plan_new('arm','imported',REPO,urdf=source,ee_link='Tool',gripper=['Grip'])
    assert plan.argv[:3]==('docker','run','--rm')
    assert f'type=bind,source={tmp_path},target={tmp_path},readonly' in plan.argv
    assert str(source) in plan.argv
    assert 'sim.arms.urdf_import' in plan.argv


def test_template_plan_is_unchanged_and_flags_need_import():
    plan=plan_new('arm','imported',REPO)
    assert plan.argv==(sys.executable,'-m','cli.scaffold','arm','imported','--repo',str(REPO))
    with pytest.raises(CliError,match='requires --ee_link'):
        plan_new('arm','imported',REPO,urdf=SOURCE)
    with pytest.raises(CliError,match='only supported'):
        plan_new('scene','imported',REPO,urdf=SOURCE,ee_link='Tool',gripper=['Grip'])


def test_planner_imports_no_simulator_libraries():
    result=subprocess.run([sys.executable,'-c',
        'from cli.plan import plan_new; import sys; '
        'assert not any(n.split(".")[0] in {"isaaclab","isaacsim","pxr","newton","omni","torch"} for n in sys.modules)'],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_staging_preserves_sources_and_force_never_overwrites_manual_arm(tmp_path):
    from sim.arms.urdf_import import parser,stage_sources
    import hashlib
    args=parser().parse_args(['staged','--urdf',str(SOURCE),'--ee_link','Tool_0',
                             '--gripper','Grip_L','--repo',str(tmp_path)])
    model,paths=stage_sources(args)
    copies=list((paths['directory']/'source').rglob('robot.urdf.xacro'))
    assert len(copies)==1
    assert copies[0].read_bytes()==SOURCE.read_bytes()
    choices=paths['choices']
    assert choices.source_hashes[str(copies[0].relative_to(paths['directory']))]==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError,match='exists'):
        stage_sources(args)
    (paths['directory']/'arm.toml').write_text('[arm]\nname="manual"\n')
    args.force=True
    with pytest.raises(ValueError,match='missing \\[arm.import\\]'):
        stage_sources(args)


def test_invalid_source_roles_publish_no_arm_folder(tmp_path):
    from sim.arms.urdf_import import parser,stage_sources
    args=parser().parse_args(['invalid','--urdf',str(SOURCE),'--ee_link','Tool_0',
                             '--gripper','Axis_2','--repo',str(tmp_path)])
    with pytest.raises(ValueError,match='base-to-ee chain'):
        stage_sources(args)
    assert not (tmp_path/'arms'/'invalid').exists()


def test_native_worker_failure_keeps_a_complete_failure_report(tmp_path,monkeypatch):
    import json
    import subprocess
    from sim.arms.urdf_import import main
    directory=tmp_path/'arms'/'supervised_arm';directory.mkdir(parents=True)
    report_path=directory/'import-report.json'
    report_path.write_text(json.dumps(dict(arm='supervised_arm',fields={},checks=[
        dict(name='convert_to_usd',passed=True)],fallbacks=[])))
    monkeypatch.delenv('S2R_IMPORT_WORKER',raising=False)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess(a[0],139))
    assert main(['supervised_arm','--urdf',str(SOURCE),'--ee_link','Tool_0',
                 '--gripper','Grip_L','--repo',str(tmp_path)])==0
    report=json.loads(report_path.read_text())
    checks={row['name']:row for row in report['checks']}
    assert checks['convert_to_usd']['passed']
    assert checks['simulator_worker']['exit_code']==139
    assert not checks['gripper_cube_lift']['passed']
    assert report['fallbacks'][-1]['step']=='simulator_worker'


def test_native_startup_has_a_published_manifest_and_report(tmp_path,monkeypatch):
    import sim.app
    from sim.arms.urdf_import import run_import
    directory=tmp_path/'arms'/'startup_probe'
    observed=[]
    def startup_failure(**kwargs):
        observed.append(((directory/'arm.toml').is_file(),(directory/'import-report.json').is_file()))
        raise RuntimeError('diagnostic native startup failure')
    monkeypatch.setattr(sim.app,'launch',startup_failure)
    assert run_import(['startup_probe','--urdf',str(SOURCE),'--ee_link','Tool_0',
                       '--gripper','Grip_L','--repo',str(tmp_path)])==0
    assert observed==[(True,True)]
