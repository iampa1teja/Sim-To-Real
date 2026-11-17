"""Report deterministic generation capacity without starting a simulator."""

import argparse as _argparse
import json as _json
import sys as _sys

from core.generate.config import load_generate_config as _load_generate_config
from core.generate.errors import ConfigError as _ConfigError, GenerationError as _GenerationError
from core.generate.loop import make_spec_fn as _make_spec_fn
from core.generate.plan import KINDS as _KINDS, check_capacity as _check_capacity, shard_quota as _shard_quota
from core.tasks.loader import load_task as _load_task


def main(argv=None):
    parser = _argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--task")
    parser.add_argument("--shard")
    args = parser.parse_args(argv)
    try:
        cfg = _load_generate_config(args.config_path, {"shard": args.shard} if args.shard is not None else None)
        tasks = (args.task,) if args.task else cfg.tasks
        for name in tasks:
            if name not in cfg.tasks:
                raise _ConfigError(f"Task is not configured: {name}")
            task = _load_task(name)
            fn = _make_spec_fn(task, cfg.arm, scene=cfg.scene, noise=cfg.noise,
                               weights_override=cfg.perturbation_weights, fps=cfg.fps)
            report = _check_capacity(name, cfg.master_seed, fn, cfg.quotas[name],
                                     max_attempt_factor=cfg.max_attempt_factor,
                                     max_consecutive_skips=cfg.max_consecutive_skips,
                                     unique_across_kinds=cfg.scenario_unique_across_kinds)
            quotas = {kind.value: getattr(cfg.quotas[name], kind.value) for kind in _KINDS}
            print(_json.dumps(dict(task=name, shard=list(cfg.shard), quotas=quotas,
                                    shard_quotas={k: _shard_quota(v, *cfg.shard) for k, v in quotas.items()},
                                    non_skipped={k.value: report.non_skipped[k] for k in _KINDS}, horizon=report.horizon), sort_keys=True))
        return 0
    except _GenerationError as exc:
        print(f"{type(exc).__name__}: {exc}", file=_sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
