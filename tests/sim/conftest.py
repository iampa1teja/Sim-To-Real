"""Opt-in live import acceptance uses subprocess Kit apps in temporary repositories."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture(scope='session',params=['five_jaw','seven_parallel'])
def imported_urdf_arm(request,tmp_path_factory):
    if os.environ.get('S2R_RUN_IMPORT_TESTS')!='1':
        pytest.skip('Set S2R_RUN_IMPORT_TESTS=1 inside s2r-sim for live import acceptance')
    pytest.importorskip('isaaclab')
    repository=Path(__file__).resolve().parents[2]
    temporary=tmp_path_factory.mktemp('urdf_acceptance_'+request.param)
    shutil.copytree(repository/'src',temporary/'src',symlinks=True,ignore=shutil.ignore_patterns('__pycache__'))
    for folder in ('scenes','tasks','configs'):
        shutil.copytree(repository/'workspace'/folder,temporary/'workspace'/folder,symlinks=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    (temporary/'workspace'/'arms').mkdir()
    # Reference selection is data-driven, as it is in the importer.
    from utils.paths import DEFAULT_ARM
    shutil.copytree(repository/'workspace'/'arms'/DEFAULT_ARM,temporary/'workspace'/'arms'/DEFAULT_ARM,symlinks=True)
    package=repository/'tests'/'fixtures'/'urdf'/(request.param+'_pkg')
    source=next(package.glob('robot.urdf*'))
    arguments=[request.param,'--urdf',str(source),'--ee_link','Tool_0','--gripper',
               'Grip_L' if request.param=='seven_parallel' else 'GripJaw','--repo',str(temporary)]
    if request.param=='seven_parallel':
        arguments+=['--camera','Camera_1=wrist_cam']
    # The entrypoint prepends /workspace/s2r/src, so set path precedence *after*
    # SDK setup. This proves the temporary repository and its mount are used.
    bootstrap='import sys,runpy; sys.path.insert(0,sys.argv.pop(1)); sys.argv[0]="sim.workspace.arms.urdf_import"; runpy.run_module("sim.workspace.arms.urdf_import",run_name="__main__")'
    environment=dict(os.environ)
    for key in ('S2R_ARM','S2R_ARMS_DIR','S2R_SCENES_DIR','S2R_TASKS_DIR'):
        environment.pop(key,None)
    result=subprocess.run([sys.executable,'-c',bootstrap,str(temporary/'src'),*arguments],
                          cwd=temporary,env=environment,text=True,capture_output=True,timeout=600)
    (temporary/'import.log').write_text(result.stdout+result.stderr)
    report_path=temporary/'workspace'/'arms'/request.param/'import-report.json'
    assert report_path.exists(),result.stdout[-3000:]+result.stderr[-3000:]
    report=json.loads(report_path.read_text())
    return dict(name=request.param,repo=temporary,report=report,returncode=result.returncode)
