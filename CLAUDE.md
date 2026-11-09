# Repository conventions

The active restructure is on branch `v2`. See `docs/restructure-status.md` for phase gates and verified evidence.
Do not start the next phase until the current phase's CPU suite and simulator checks pass.

- `src/core/` must not import Isaac Sim, Isaac Lab, Kit, or `pxr`. Simulator-specific code belongs in `src/sim/`.
- Choose arm, scene and task before importing modules that read `S2R_ARM`, `S2R_SCENE` and `S2R_TASK` at import time.
  The current shared selection and Kit argument helpers live in `sim.app` and are safe before Kit startup.
- Isaac Lab 3.0 / Newton live tensors use XYZW quaternions. Dataset sidecars and benchmark JSON use WXYZ.
  Convert explicitly at those boundaries. Positions and translations are metres; articulation angles are radians.
- LeRobot raw body joints currently use -100..100, and gripper joints 0..100. Calibrated spans and runtime clamps
  are distinct. Use `core.arms.mapping` conversions; do not duplicate interpolation formulas.
- Newton imports scene topology once. Declare objects before model construction, then park/reposition them;
  do not add rigid bodies during a running episode.
- Table planes and link-local jaw geometry are static across resets and cached per USD stage. Call
  `clear_geometry_cache(stage)` after explicitly editing those meshes or table transforms.
- Preserve fixed evaluation schedules, seeds and content hashes. Schema conversion must retain the legacy hash.
- `gui/` and `src/sim/recording/web_gui.py` are outside this restructure. The local untracked code explorer is also
  outside the repository; its files are preserved.
- Run CPU checks with `PYTHONPATH=src python -m pytest tests -q`. The existing `s2r-sim` container includes the
  required Python/Isaac dependencies; run simulator scripts through `s2r-entrypoint python` there.
- Keep local hardware settings in ignored `.env`; `.env.example` contains portable example values.
- Do not change production physics to make a diagnostic pass. Diagnostic spawn fixtures must start with enough
  jaw clearance for their object. Closed-jaw object overlap is a known placement issue, not a safe simulator input.
