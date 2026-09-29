# Pick-place evaluation validation record

[Guide hub](../README.md) · [Evaluation guide](../docs/06-evaluation.md)

## Scope and historical results

The implementation's earlier CPU validation ran 22 tests covering host runner
lifecycle and argv, streamed logs after auto-removal, interrupted evaluation,
recorded-start transforms and seeded sampling, zone definitions, overlap handling,
colour instructions, grouped reports and auto-reset behavior. All 22 passed in
2.523 seconds in that environment. Those numbers are historical, not a new test
claim made by this documentation rewrite.

The earlier record also reports syntax, whitespace and targeted static checks,
and stubbed event-manager ordering checks. GPU rendering/physics checks were
explicitly outstanding. Current behavior, CLI flags and JSON schema are explained
in [Evaluation](../docs/06-evaluation.md); measured dimensions and DR ranges remain
in code and must not be treated as universal calibration.

## Repeat CPU checks

**Host or training machine**, repository root, Python with the test dependencies:

```bash
python -m unittest discover -s tests -v
bash -n docker/eval_pick_place.sh
git diff --check
```

The preparation integration test additionally needs `GROOT_CHECKOUT` and its
runtime dependencies; see [Training tests](../docs/05-training.md#cpu-preparation-regression-tests).
Do not infer a preparation pass from a skipped test.

## GPU/container checklist

First start teleop with the [Setup](../docs/01-setup.md) mounts. Use your own
sidecars and checkpoint. None of these checks is fulfilled by a dry run.

**Host**, attach a shell:

```bash
docker exec -it teleop bash
```

**Teleop container**, inspect plain resets followed by DR resets without a policy:

```bash
export PICK_PLACE_EVAL_STARTS=/workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1/pick_place_meta
zero_agent --task Lerobot-So101-Teleop-Pick-Place-Eval --num_steps 2250
zero_agent --task Lerobot-So101-Teleop-Pick-Place-DR-Eval --num_steps 4500
```

Pass criteria: startup validates robot-reset → cube-spawn → tracking-reset order;
cubes rest on the live tabletop outside the box; DR alters the applicable light,
cameras, robot colour and blue/red cube colour. Consecutive colours may repeat.
MyRoom has no HDRI/mat DR targets. Custom rooms without the measured tube light
skip that term. Confirm geometry and success behavior have not changed.

**Host**, repository root, dry-run both variants and then run five episodes:

```bash
./docker/eval_pick_place.sh --model so101_pick_place_v1/checkpoint-10000 --models_dir ~/sim2real/models --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 --dry_run
./docker/eval_pick_place.sh --model so101_pick_place_v1/checkpoint-10000 --models_dir ~/sim2real/models --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 --dr --dry_run
./docker/eval_pick_place.sh --model so101_pick_place_v1/checkpoint-10000 --models_dir ~/sim2real/models --dataset /workspace/Sim-to-Real-SO-101-Workshop/datasets/pick_place_v1 --episodes 5
docker ps -a --filter name=groot-eval-server --format '{{.Names}}'
```

Pass criteria: exactly five completed episodes, readable saved JSON,
`complete=true`, `overall.episodes=5`, and no server from this run left behind.
Check Ctrl+C/error cleanup as well. Repeat DR with the [colour instruction example](../docs/06-evaluation.md#recorded-starts-and-dr)
and inspect whether instructions match the actual cube. Compare grouped rates
alongside episode counts; do not conflate policy performance with infrastructure
validation.
