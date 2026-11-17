"""End-to-end generation harness on the CPU (contracts §11, §13; WP7). Shared by WP7, WP9, WP10 and WP13.

Everything runs on ``FakeEnv`` + ``MemoryWriter`` (no Isaac) inside a ``sandbox()``:

* ``prepare_sandbox(box, task_names, arms=...)`` copies the fx arms, ``fx_scene`` and the named task
  folders into the sandbox. A task is looked up in ``tests/fixtures/v4_wp7/tasks``, then
  ``tests/fixtures/v4_wp1/tasks``, then the repository ``tasks/`` folder (WP9/WP10/WP13 tasks).
* ``write_config(path, ...)`` writes a ``configs/generate``-style YAML (contracts §8.5).
* ``run_shard(cfg_path, task, shard, kills=...)`` runs ``run_generation`` for one shard and, for each
  kill point, lets the run crash (``SimulatedCrash``) and resumes it with a fresh env and writer, exactly
  as a restarted container would. Kill points: ``("env", step)`` (``FakeEnv.crash_at_step``),
  ``("before_save", episode)`` / ``("after_save", episode)`` (``MemoryWriter``), and ``("torn", None)``
  (a half-written manifest line appended after the previous crash).
* ``generate_small(...)`` runs every shard with its kill points and merges the shards with WP12.
* ``fake_dr(src_root, dst_root, ...)`` builds a DR dataset without rendering: the originals plus
  ``copies`` appearance copies per original (``draw_appearance`` / ``dr_copy_sidecar``, WP5), frames
  copied from the source (a re-render keeps the trajectory).
* ``assert_no_duplicates(test, root, source_root=None)`` recomputes every fingerprint from the stored
  data independently of ``core.datasets.audit`` and asserts the owner's no-duplicates requirement.
* ``run_no_duplicates_suite(test, task_names, ...)`` is the main gate: generate (3 shards, random kills,
  resume), merge, fake DR x2, audit both datasets with ``core.datasets.audit``, and check them again with
  ``assert_no_duplicates``. WP9, WP10 and WP13 call it for their real tasks.

All v4 imports are inside functions, so importing this module needs only the standard library and numpy.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .env import FIXTURES_DIR, REPO_ROOT, clear_task_cache, copy_fixture

KINDS = ("success", "failure", "recovery")
TASK_DIRS = (FIXTURES_DIR / "v4_wp7" / "tasks", FIXTURES_DIR / "v4_wp1" / "tasks", REPO_ROOT / "tasks")
DEFAULT_ARM = "fx_arm5"
SCENE = "fx_scene"
ROBOT_COLORS = ("black", "white", "orange", "blue")
EXPOSURE = (0.5, 1.5)


# ---------------------------------------------------------------------------------------------------
# sandbox and config

def find_task_dir(name: str) -> Path:
    for base in TASK_DIRS:
        if (base / name / "task.toml").is_file():
            return base / name
    raise FileNotFoundError(f"no task folder {name!r} in {[str(b) for b in TASK_DIRS]}")


def prepare_sandbox(box, task_names: Iterable[str], *, arms: Sequence[str] = ("fx_arm5", "fx_arm6")) -> None:
    """Copy arms, the fx scene and the task folders into ``box`` (a ``v4_support.sandbox()``)."""
    for arm in arms:
        copy_fixture(f"arms/{arm}", box.root)
    copy_fixture(f"scenes/{SCENE}", box.root)
    for name in task_names:
        dst = box.tasks / name
        if not dst.exists():
            shutil.copytree(find_task_dir(name), dst, ignore=shutil.ignore_patterns("__pycache__", "*.usd", "*.usdz"))
    clear_task_cache()


def _yaml_value(v):
    return json.dumps(v)          # JSON is valid YAML flow syntax


def write_config(path, *, tasks: Sequence[str], dst_root, arm: str = DEFAULT_ARM, quotas: Mapping[str, int] | None = None,
                 per_task_quotas: Mapping[str, Mapping[str, int]] | None = None, shard: str = "0/1",
                 master_seed: int = 0, max_attempt_factor: float = 8.0, max_consecutive_skips: int = 200,
                 scenario_unique_across_kinds: bool = True, noise_fraction: float = 0.5, dr_copies: int = 2,
                 dr_seed: int = 0, perturbation_weights: Mapping[str, float] | None = None) -> Path:
    """Write a generate config (contracts §8.5) as YAML and return its path."""
    cfg = {"arm": arm, "scene": SCENE, "tasks": list(tasks), "master_seed": int(master_seed),
           "quotas": dict(quotas or {"success": 3, "failure": 3, "recovery": 3}),
           "per_task_quotas": {k: dict(v) for k, v in (per_task_quotas or {}).items()},
           "shard": shard, "dst_root": str(dst_root), "repo_id": "local/{task}", "fps": 30,
           "max_attempt_factor": float(max_attempt_factor), "max_consecutive_skips": int(max_consecutive_skips),
           "scenario_unique_across_kinds": bool(scenario_unique_across_kinds),
           "noise": {"fraction": float(noise_fraction), "theta": 0.15, "sigma_rad": 0.01, "max_abs_rad": 0.02},
           "perturbation_weights": None if perturbation_weights is None else dict(perturbation_weights),
           "cameras": False,
           "dr": {"copies": int(dr_copies), "terms": ["light", "robot_color"], "recolor": True, "seed": int(dr_seed)}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{k}: {_yaml_value(v)}\n" for k, v in cfg.items()))
    return path


# ---------------------------------------------------------------------------------------------------
# one shard

class Rig:
    """Arm objects shared by every episode of one arm (chain, gripper, mapping)."""

    def __init__(self, arm: str):
        from core.arms.gripper import Gripper
        from core.arms.kinematics import KinematicChain
        from core.arms.raw_mapping import RawMapping
        self.arm = arm
        self.chain = KinematicChain.from_arm(arm)
        self.gripper = Gripper.from_arm(arm)
        self.mapping = RawMapping.from_setup(arm=arm)


def _manifest(root) -> Path:
    return Path(root) / "meta" / "generation_manifest.jsonl"


def tear_manifest(root, text: str = '{"v": 1, "task": "torn') -> None:
    """Append a half-written manifest line (no newline), as a crash in the middle of a write leaves it."""
    path = _manifest(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def run_shard(cfg_path, task: str, *, shard: str | None = None, kills: Sequence[tuple] = (), rig: Rig | None = None,
              env_factory=None, expert_factory=None, git_commit: str = "0" * 40):
    """Run ``run_generation`` for ``task`` on one shard, crashing at each kill point and resuming.

    Returns ``(root, summary, crashes)``: the dataset root, the final ``GenerationSummary`` and the number
    of crashes that actually happened (a kill point after the end of the work never fires).
    ``env_factory(task_obj, rig, crash_at_step)`` and ``expert_factory`` replace FakeEnv / the scripted
    expert (tests of the loop's duplicate path use them).
    """
    from core.datasets.frames import MemoryWriter
    from core.generate.config import load_generate_config
    from core.generate.fake_env import FakeEnv, SimulatedCrash
    from core.generate.loop import make_expert_factory, make_spec_fn, run_generation
    from core.tasks.loader import load_task
    cfg = load_generate_config(cfg_path, {"shard": shard} if shard else None)
    rig = rig or Rig(cfg.arm)
    task_obj = load_task(task)
    root = cfg.dataset_root(task)
    experts = expert_factory or make_expert_factory(task_obj, cfg.arm)
    spec_fn = make_spec_fn(task_obj, cfg.arm, scene=cfg.scene, noise=cfg.noise,
                           weights_override=cfg.perturbation_weights, fps=cfg.fps)
    make_env = env_factory or (lambda t, r, crash: FakeEnv(t, r.chain, r.gripper, fps=cfg.fps, crash_at_step=crash))
    crashes = 0
    plan = list(kills) + [None]
    for kill in plan:
        env_crash = writer_crash = writer_ep = None
        if kill is not None:
            what, at = kill
            if what == "env":
                env_crash = int(at)
            elif what in ("before_save", "after_save"):
                writer_crash, writer_ep = what, at
            elif what == "torn":
                tear_manifest(root)
                continue
            else:
                raise ValueError(f"unknown kill point {kill!r}")
        env = make_env(task_obj, rig, env_crash)
        writer = MemoryWriter(root, fps=cfg.fps, joint_names=tuple(rig.chain.q_names), crash=writer_crash,
                              crash_on_episode=writer_ep)
        try:
            summary = run_generation(cfg, task, env, writer, expert_factory=experts, mapping=rig.mapping,
                                     spec_fn=spec_fn, git_commit=git_commit)
        except SimulatedCrash:
            crashes += 1
            continue
        finally:
            env.close()
        if kill is None:
            return root, summary, crashes
    raise AssertionError("unreachable")


def random_kills(rng, *, n: int = 3, max_step: int = 400, max_episode: int = 2) -> list[tuple]:
    """``n`` random kill points of every kind (env step, before/after save, torn manifest line)."""
    out = []
    for _ in range(n):
        what = ("env", "before_save", "after_save", "torn")[int(rng.integers(4))]
        if what == "env":
            out.append(("env", int(rng.integers(1, max_step))))
        elif what == "torn":
            out.append(("torn", None))
        else:
            out.append((what, int(rng.integers(0, max_episode))))
    return out


# ---------------------------------------------------------------------------------------------------
# several shards, merge, audit

def audit(root, *, source_root=None):
    """Run the WP7 audit and write ``meta/audit_report.json``; return the report."""
    from core.datasets.audit import audit_dataset, write_audit_report
    report = audit_dataset(root, source_root=source_root)
    write_audit_report(root, report)
    return report


def merge(shard_roots: Sequence[Path], dst_root, *, repo_id: str = "local/merged") -> Path:
    """Audit each shard (WP7) and merge them with WP12 (metadata + lite frames)."""
    from core.datasets.merge import ShardSummary, apply_merge_meta, merge_frames, plan_merge
    for root in shard_roots:
        report = audit(root)
        if not report.passed:
            failed = [c for c in report.checks if not c["passed"]]
            raise AssertionError(f"shard {root} failed its audit: {failed}")
    shards = [ShardSummary.from_root(r) for r in shard_roots]
    plan = plan_merge(shards)
    dst_root = Path(dst_root)
    apply_merge_meta(plan, shards, dst_root)
    merge_frames(plan, shards, dst_root, repo_id)
    return dst_root


def generate_small(task_names: Sequence[str], kinds: Sequence[str] = KINDS, quotas: Mapping[str, int] | None = None,
                   shards: int = 3, kill_points: Mapping[tuple[str, int], Sequence[tuple]] | None = None, *,
                   work: Path, arm: str = DEFAULT_ARM, master_seed: int = 0, max_attempt_factor: float = 8.0,
                   per_task_quotas: Mapping[str, Mapping[str, int]] | None = None,
                   stats: dict | None = None) -> dict[str, Path]:
    """Generate every task in ``shards`` shards with kill points, merge per task; return {task: merged root}.

    ``quotas`` are totals per kind (default 3 each); kinds not in ``kinds`` get 0. ``kill_points`` maps
    ``(task, shard)`` to a list of kill points (see ``run_shard``). Must run inside a prepared sandbox.
    ``stats`` (optional dict) receives ``{(task, shard): crashes that fired}``.
    """
    quotas = dict(quotas or {k: 3 for k in KINDS})
    quotas = {k: (int(quotas.get(k, 0)) if k in kinds else 0) for k in KINDS}
    work = Path(work)
    out = {}
    rig = Rig(arm)
    for task in task_names:
        roots = []
        for i in range(shards):
            cfg = write_config(work / f"cfg_{task}_{i}.yaml", tasks=[task], dst_root=work / f"shard{i}", arm=arm,
                               quotas=quotas, per_task_quotas=per_task_quotas, shard=f"{i}/{shards}",
                               master_seed=master_seed, max_attempt_factor=max_attempt_factor)
            root, _, crashes = run_shard(cfg, task, kills=(kill_points or {}).get((task, i), ()), rig=rig)
            if stats is not None:
                stats[(task, i)] = crashes
            roots.append(root)
        out[task] = merge(roots, work / "merged" / task, repo_id=f"local/{task}")
    return out


# ---------------------------------------------------------------------------------------------------
# fake DR

def fake_dr(src_root, dst_root, *, copies: int = 2, dr_seed: int = 0, terms=("light", "robot_color"),
            recolor: bool = True, include_originals: bool = True, source_repo_id: str = "local/source") -> Path:
    """Build a DR dataset at ``dst_root`` from the audited dataset at ``src_root`` without rendering.

    For each source episode (in index order): the original (if ``include_originals``), then ``copies``
    copies with appearance drawn by ``draw_appearance`` (``taken`` = parent + earlier siblings). Frames are
    the source frames; each copy's ``task`` column is its new instruction.
    """
    from core.datasets.frames import LiteDatasetWriter, read_frames
    from core.datasets.task_meta import list_sidecars, read_sidecar, write_sidecar
    from core.generate.appearance import draw_appearance, dr_copy_sidecar
    from core.generate.protocols import FrameLabels
    from core.tasks.base import load_task_spec
    src_root, dst_root = Path(src_root), Path(dst_root)
    frames = read_frames(src_root)
    info = json.loads((src_root / "meta" / "info.json").read_text())
    names = tuple(info["features"]["action"]["names"])
    writer = LiteDatasetWriter(dst_root, fps=int(info["fps"]), joint_names=names)
    specs = {}

    def put(sidecar, data):
        index = writer.episode_count()
        for t in range(sidecar["num_frames"]):
            ee = np.asarray(data["label.ee_pose"][t], np.float64)
            writer.add_frame(data["action"][t], data["observation.state"][t], {}, sidecar["instruction"] or "",
                             FrameLabels(stage=int(data["label.stage"][t]), imitate=bool(data["label.imitate"][t]),
                                         ee_pose_wxyz=ee))
        sidecar = dict(sidecar, episode_index=index)
        write_sidecar(dst_root, sidecar)
        writer.save_episode()

    for i in list_sidecars(src_root):
        parent = read_sidecar(src_root, i)
        if include_originals:
            put(parent, frames[i])
        task = parent["task"]
        if task not in specs:
            specs[task] = load_task_spec(task)
        taken = {parent["fingerprints"]["appearance_fp"]}
        for c in range(copies):
            draw = draw_appearance(parent, task_spec=specs[task], terms=terms, exposure_range=EXPOSURE,
                                   robot_colors=ROBOT_COLORS, recolor=recolor, dr_seed=dr_seed, copy_index=c,
                                   taken=set(taken))
            taken.add(draw.appearance_fp)
            put(dr_copy_sidecar(parent, draw, episode_index=writer.episode_count(), source_repo_id=source_repo_id),
                frames[i])
    writer.finalize()
    return dst_root


# ---------------------------------------------------------------------------------------------------
# independent duplicate check

def _q(a):
    return np.floor(np.asarray(a, np.float32).astype(np.float64) * 100 + 0.5).astype("<i4").tobytes()


def assert_no_duplicates(test, root, *, source_root=None) -> dict:
    """Recompute traj / appearance / episode fingerprints from the stored data with the contract formulas
    (§8.2) and assert: every stored fingerprint matches; no two episodes share ``episode_fp``; no two
    originals share ``traj_fp`` or ``scenario_key``; every DR copy shares ``traj_fp`` with its parent and
    differs from it and from its siblings in ``appearance_fp``; no two episodes have identical frames.
    Returns {"originals": n, "copies": m}.
    """
    import hashlib

    from core.datasets.frames import read_frames
    from core.datasets.task_meta import list_sidecars, read_sidecar
    from core.utils.hashing import json_sha256
    root = Path(root)
    frames = read_frames(root)
    sidecars = {i: read_sidecar(root, i) for i in list_sidecars(root)}
    test.assertEqual(sorted(frames), sorted(sidecars), "data and sidecars hold different episodes")
    test.assertGreater(len(sidecars), 0, "empty dataset")

    def quant(x, step):
        return int(math.floor(float(x) / step + 0.5))

    episode_fps, raw_hashes = {}, {}
    trajs, scenarios, appearance = {}, {}, {}
    copies = {}
    for i, sc in sorted(sidecars.items()):
        f = frames[i]
        fp = sc["fingerprints"]
        payload = {"v": 1, "shape": [int(f["action"].shape[0]), int(f["action"].shape[1])],
                   "action": hashlib.sha256(_q(f["action"])).hexdigest(),
                   "state": hashlib.sha256(_q(f["observation.state"])).hexdigest(),
                   "objects0": {n: [quant(c, 1e-4) for c in o["position"][0]] for n, o in sc["objects"].items()}}
        traj = json_sha256(payload)
        light = sc["appearance"]["light_exposure"]
        app = json_sha256({"v": 1, "colors": {n: o["color"] for n, o in sc["objects"].items()},
                           "instruction": sc["instruction"], "light": None if light is None else quant(light, 1e-3),
                           "robot_color": sc["appearance"]["robot_color"]})
        epi = hashlib.sha256((traj + app).encode("ascii")).hexdigest()
        with test.subTest(episode=i):
            test.assertEqual(fp["traj_fp"], traj, f"episode {i}: stored traj_fp does not match the data")
            test.assertEqual(fp["appearance_fp"], app, f"episode {i}: stored appearance_fp does not match")
            test.assertEqual(fp["episode_fp"], epi, f"episode {i}: stored episode_fp does not match")
        test.assertNotIn(epi, episode_fps, f"episodes {episode_fps.get(epi)} and {i} are duplicates")
        episode_fps[epi] = i
        appearance[i] = app
        raw = hashlib.sha256(np.ascontiguousarray(f["action"]).tobytes() + sc["instruction"].encode()
                             + json.dumps(sc["objects"], sort_keys=True).encode()
                             + json.dumps(sc["appearance"], sort_keys=True).encode()).hexdigest()
        test.assertNotIn(raw, raw_hashes, f"episodes {raw_hashes.get(raw)} and {i} store identical content")
        raw_hashes[raw] = i
        if sc["source"] is None:
            test.assertNotIn(traj, trajs, f"originals {trajs.get(traj)} and {i} share a trajectory")
            test.assertNotIn(fp["scenario_key"], scenarios,
                             f"originals {scenarios.get(fp['scenario_key'])} and {i} share a start scenario")
            trajs[traj] = i
            scenarios[fp["scenario_key"]] = i
        else:
            copies.setdefault(sc["source"]["episode_fp"], []).append(i)
            test.assertEqual(sc["source"]["traj_fp"], traj, f"copy {i} changed the trajectory")
    parents = {sc["fingerprints"]["episode_fp"]: sc for sc in sidecars.values() if sc["source"] is None}
    if source_root is not None:
        for j in list_sidecars(source_root):
            sc = read_sidecar(source_root, j)
            parents.setdefault(sc["fingerprints"]["episode_fp"], sc)
    for parent_fp, members in copies.items():
        test.assertIn(parent_fp, parents, f"copies {members} have no parent")
        parent = parents[parent_fp]
        seen = {parent["fingerprints"]["appearance_fp"]}
        for i in members:
            test.assertEqual(sidecars[i]["fingerprints"]["traj_fp"], parent["fingerprints"]["traj_fp"])
            test.assertNotIn(appearance[i], seen, f"copy {i} repeats the appearance of its parent or a sibling")
            seen.add(appearance[i])
    return {"originals": len(trajs), "copies": sum(len(m) for m in copies.values())}


def assert_audit_passes(test, report):
    failed = [(c["name"], c["details"][:3]) for c in report.checks if not c["passed"]]
    test.assertTrue(report.passed, f"audit failed: {failed}")


def run_no_duplicates_suite(test, task_names: Sequence[str], *, work: Path, arm: str = DEFAULT_ARM,
                            quotas: Mapping[str, int] | None = None, shards: int = 3, kills_per_shard: int = 2,
                            dr_copies: int = 2, seed: int = 0, max_attempt_factor: float = 8.0,
                            per_task_quotas: Mapping[str, Mapping[str, int]] | None = None) -> dict:
    """The no-duplicates gate (contracts §13) for ``task_names``; call it inside a prepared sandbox.

    For each task: ``shards`` shards with ``kills_per_shard`` random kill points each (seeded), merge, audit
    (must pass), ``assert_no_duplicates``; then fake DR x``dr_copies`` into a DR dataset, audit it with the
    merged dataset as ``source_root`` (must pass) and ``assert_no_duplicates`` again. Every kind must reach
    its quota exactly. Returns {task: {"merged": Path, "dr": Path, "counts": {...}}}.
    """
    from core.datasets.task_meta import list_sidecars, read_sidecar
    rng = np.random.default_rng(seed)
    quotas = dict(quotas or {k: 3 for k in KINDS})
    kills = {(t, i): random_kills(rng, n=kills_per_shard) for t in task_names for i in range(shards)}
    stats = {}
    merged = generate_small(task_names, KINDS, quotas, shards, kills, work=Path(work) / "gen", arm=arm,
                            master_seed=seed, max_attempt_factor=max_attempt_factor, per_task_quotas=per_task_quotas,
                            stats=stats)
    if kills_per_shard:
        test.assertGreater(sum(stats.values()), 0, "no kill point fired; the resume path was not exercised")
    out = {}
    for task, root in merged.items():
        with test.subTest(task=task):
            assert_audit_passes(test, audit(root))
            want = dict(quotas, **(per_task_quotas or {}).get(task, {}))
            kinds = [read_sidecar(root, i)["kind"] for i in list_sidecars(root)]
            test.assertEqual({k: kinds.count(k) for k in KINDS}, {k: int(want.get(k, 0)) for k in KINDS},
                             f"{task}: quotas not met exactly")
            counts = assert_no_duplicates(test, root)
            dr = fake_dr(root, Path(work) / "dr" / task, copies=dr_copies, dr_seed=seed,
                         source_repo_id=f"local/{task}")
            assert_audit_passes(test, audit(dr, source_root=root))
            dr_counts = assert_no_duplicates(test, dr, source_root=root)
            test.assertEqual(dr_counts["copies"], dr_copies * counts["originals"])
            out[task] = {"merged": root, "dr": dr, "counts": counts}
    return out
