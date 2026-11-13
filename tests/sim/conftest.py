"""Opt-in live import acceptance uses subprocess Kit apps in temporary repositories."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _import_urdf_arm(arm_name,tmp_path_factory):
    if os.environ.get('S2R_RUN_IMPORT_TESTS')!='1':
        pytest.skip('Set S2R_RUN_IMPORT_TESTS=1 inside s2r-sim for live import acceptance')
    pytest.importorskip('isaaclab')
    repository=Path(__file__).resolve().parents[2]
    temporary=tmp_path_factory.mktemp('urdf_acceptance_'+arm_name)
    for folder in ('src','scenes','tasks','configs'):
        shutil.copytree(repository/folder,temporary/folder,symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    (temporary/'arms').mkdir()
    # Reference selection is data-driven, as it is in the importer.
    from core.paths import DEFAULT_ARM
    shutil.copytree(repository/'arms'/DEFAULT_ARM,temporary/'arms'/DEFAULT_ARM,symlinks=True)
    package=repository/'tests'/'fixtures'/'urdf'/(arm_name+'_pkg')
    source=next(package.glob('robot.urdf*'))
    arguments=[arm_name,'--urdf',str(source),'--ee_link','Tool_0','--gripper',
               'Grip_L' if arm_name=='seven_parallel' else 'GripJaw','--repo',str(temporary)]
    if arm_name=='seven_parallel':
        arguments+=['--camera','Camera_1=wrist_cam']
    # The entrypoint prepends /workspace/s2r/src, so set path precedence *after*
    # SDK setup. This proves the temporary repository and its mount are used.
    bootstrap='import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); sys.argv[0]="sim.arms.urdf_import"; runpy.run_module("sim.arms.urdf_import",run_name="__main__")'
    environment=dict(os.environ)
    for key in ('S2R_ARM','S2R_ARMS_DIR','S2R_SCENES_DIR','S2R_TASKS_DIR'):
        environment.pop(key,None)
    result=subprocess.run([sys.executable,'-c',bootstrap,str(temporary/'src'),*arguments],
                          cwd=temporary,env=environment,text=True,capture_output=True,timeout=600)
    (temporary/'import.log').write_text(result.stdout+result.stderr)
    report_path=temporary/'arms'/arm_name/'import-report.json'
    assert report_path.exists(),result.stdout[-3000:]+result.stderr[-3000:]
    report=json.loads(report_path.read_text())
    return dict(name=arm_name,repo=temporary,report=report,returncode=result.returncode)


@pytest.fixture(scope='session')
def urdf_imports(tmp_path_factory):
    """Share each expensive live import across generic and embodiment-specific checks."""
    results={}
    def imported(name):
        if name not in results:
            results[name]=_import_urdf_arm(name,tmp_path_factory)
        return results[name]
    return imported


@pytest.fixture(scope='session',params=['five_jaw','seven_parallel'])
def imported_urdf_arm(request,urdf_imports):
    return urdf_imports(request.param)


@pytest.fixture(scope='session')
def imported_parallel_arm(urdf_imports):
    return urdf_imports('seven_parallel')
