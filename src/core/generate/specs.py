"""Construct episode specs using independent deterministic random streams."""

from core.generate.perturbations import draw_perturbation as _draw_perturbation
from core.generate.protocols import EpisodeSpec as _EpisodeSpec, Kind as _Kind, NoiseSpec as _NoiseSpec
from core.generate.seeds import rng_for as _rng_for


def build_spec(task, kind, attempt_index, seed, *, arm, scene, home_q, reach_fn=None, noise=None,
               weights_override=None, fps=30, joint_names=None, joint_limits=None, arm_joints=None):
    kind = _Kind(kind)
    start = task.sample_episode_start(_rng_for(seed, "start"), reach_fn=reach_fn, home_q=home_q,
                                      joint_limits=joint_limits, instruction_rng=_rng_for(seed, "instruction"),
                                      joint_names=joint_names, arm_joints=arm_joints)
    perturbation = _draw_perturbation(task, kind, _rng_for(seed, "perturbation"), weights_override, start=start)
    drawn_noise = None
    if kind is _Kind.SUCCESS and noise is not None and _rng_for(seed, "noise").random() < noise.fraction:
        drawn_noise = _NoiseSpec(noise.theta, noise.sigma_rad, noise.max_abs_rad)
    return _EpisodeSpec(task=task.name, arm=arm, scene=scene, kind=kind, attempt_index=attempt_index, seed=seed,
                        objects=start.objects, roles=start.roles, goal=start.goal, instruction=start.instruction,
                        initial_q=tuple(home_q) if start.initial_q is None else start.initial_q,
                        held_object=start.held_object, perturbation=perturbation, noise=drawn_noise,
                        max_steps=task.spec.raw.get("expert", {}).get("max_steps", 600), fps=fps)
