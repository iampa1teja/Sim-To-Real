"""Diagnostic reproducibility harness; production manifests and physics stay untouched."""
import importlib
from pathlib import Path
import sys
checkout=Path(sys.argv.pop(1)).resolve()
source=sys.argv.pop(1)
sys.path.insert(0,str(checkout/'src'))
from sim.datagen import scripted
original=scripted.setup_context

def setup(*args,**kwargs):
    from sim.envs import scene_builder
    apply=scene_builder._apply_physics
    def reproducible(cfg,settings):
        apply(cfg,settings)
        cfg.sim.physics.deterministic_mode='run_to_run'
        cfg.sim.physics.solver_cfg.disable_sensors=True
    scene_builder._apply_physics=reproducible
    return original(*args,**kwargs)

scripted.setup_context=setup
print(f'[diagnostic] checkout={checkout}; run_to_run scheduling; MuJoCo internal sensors disabled',flush=True)
if source=='scripted':
    code=scripted.main(sys.argv[1:])
else:
    code=importlib.import_module('sim.datagen.'+source).main(sys.argv[1:])
raise SystemExit(code)
