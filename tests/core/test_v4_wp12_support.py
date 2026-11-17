"""WP12: shared builders for the WP12 tests (small on-disk datasets, audit-record stub, storage fakes).

The other WP12 test files import these helpers with ``from test_v4_wp12_support import ...`` (pytest
puts ``tests/core`` on ``sys.path``; basenames are unique, contracts §13). The tests at the bottom
check the builders themselves against the contract, so a broken builder cannot hide a broken package.

* ``make_sidecar`` builds a valid v2 sidecar (§9.1) with unique fingerprints derived from a tag.
* ``build_dataset`` writes a dataset in the lite LeRobot-v3 layout MemoryWriter persists (§11):
  ``meta/info.json`` ``{"codebase_version": "v3.0-lite", "fps", "total_episodes", "features"}``,
  ``data/chunk-000/file-{index:06d}.parquet`` (``action``, ``observation.state`` list<float32>,
  ``episode_index``, ``frame_index``, ``index`` int64, ``timestamp`` float32, ``task`` string and the
  ``LABEL_FEATURES`` columns), ``task_meta/`` sidecars, ``meta/generation_manifest.jsonl`` (§8.3),
  ``meta/generation_plan.json`` (§8.5) and the audit record ``meta/audit_report.json`` (§10.1). With
  ``layout="v3"`` the data looks like a real LeRobot v3 dataset instead (several episodes per file,
  fixed-size-list vectors, scalar shape-(1,) label columns, ``codebase_version "v3.0"``).
* ``content_digest`` is the §10.1 digest as WP12 must compute it (see docs/v4/prompts/WP12.md):
  ``json_sha256({posix path relative to root: file sha256})`` over every regular file under
  ``meta/``, ``data/``, ``videos/`` and ``task_meta/``, except ``meta/audit_report.json``.
* ``write_audit`` is the stub for WP7's ``write_audit_report``: it writes the AuditReport fields
  (§10.1) as JSON, passing by default, with the current content digest.
* ``FakeHubApi`` / ``FakeFs`` are in-memory stand-ins for ``huggingface_hub.HfApi`` and an fsspec
  filesystem with the signatures the contract names (§10.3).
"""

import copy
import dataclasses
import hashlib
import json
import os
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from core.datasets.task_meta import LABEL_FEATURES, list_sidecars, read_sidecar, validate_sidecar, write_sidecar
from core.generate.manifest import Manifest, ManifestLine, read_plan_json, write_plan_json
from core.utils.hashing import canonical_json, json_sha256, tree_sha256
from v4_support.factories import hex_digest, make_noise, make_perturbation, make_recovery_info
from v4_support.fx_arms import fx_joint_names, fx_lerobot_names

ARM = "fx_arm5"
JOINTS = list(fx_joint_names(ARM))
LEROBOT = list(fx_lerobot_names(ARM))
J = len(JOINTS)
STAGES = ["reached", "grasped", "lifted", "placed"]
KINDS = ("success", "failure", "recovery")
TREES = ("meta", "data", "videos", "task_meta")
AUDIT_REL = "meta/audit_report.json"
MANIFEST_REL = "meta/generation_manifest.jsonl"
CHECK_NAMES = ("sidecars_valid", "counts_match", "labels_match", "fingerprints_recomputed", "episode_fp_unique",
               "traj_fp_unique_originals", "scenario_key_unique_originals", "holdout_absent", "dr_sources_exist",
               "dr_appearance_distinct")
#: Forbidden at import time for every WP12 module (§1.6 and the WP12 brief).
HEAVY_ROOTS = ("isaaclab", "isaacsim", "omni", "carb", "pxr", "warp", "lerobot", "torch", "huggingface_hub",
               "boto3", "botocore", "s3fs", "fsspec", "datasets", "sim")


# ---------------------------------------------------------------------------------------------
# Sidecars


def _unit_quats(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q


def _track(rng, base, n):
    pos = np.asarray(base) + rng.uniform(-0.01, 0.01, (n, 3))
    return {"position": [[round(float(v), 6) for v in row] for row in pos],
            "orientation_wxyz": [[round(float(v), 6) for v in row] for row in _unit_quats(rng, n)]}


def make_fingerprints_json(tag):
    """The five §9.1 fingerprint strings, unique per ``tag``."""
    return {"scenario_key": hex_digest(f"scenario-{tag}"), "spec_fp": hex_digest(f"spec-{tag}"),
            "traj_fp": hex_digest(f"traj-{tag}"), "appearance_fp": hex_digest(f"app-{tag}"),
            "episode_fp": hex_digest(f"episode-{tag}")}


def make_sidecar(kind="success", *, index=0, n=12, tag=None, rng=None, task="fx_pick", shard=(0, 1),
                 attempt_index=None, seed=None, colors=("red", None), sets=("cubes", None), fps=30):
    """A valid v2 sidecar (§9.1) of ``kind`` with ``n`` frames and fingerprints derived from ``tag``.

    Objects: ``cube_0`` (target, colour ``colors[0]``, set ``sets[0]``) and ``box`` (container).
    Poses are already 6-decimal (as ``write_sidecar`` stores them), so ``read_sidecar`` returns an
    equal dict. ``attempt`` carries ``shard``; ``seed`` defaults to a value above 2**63.
    """
    if n < 8:
        raise ValueError("make_sidecar needs n >= 8 (recovery segments)")
    rng = np.random.default_rng(index) if rng is None else rng
    tag = f"{task}-{kind}-{index}" if tag is None else tag
    attempt_index = index if attempt_index is None else attempt_index
    onset = max(1, n // 4)
    if kind == "success":
        bounds = [("nominal", n)]
    elif kind == "failure":
        bounds = [("nominal", onset), ("perturbation", onset + 2), ("failure", n)]
    else:
        bounds = [("nominal", onset), ("perturbation", onset + 2), ("recovery", onset + 4), ("nominal", n)]
    segments, start = [], 0
    for label, end in bounds:
        segments.append({"label": label, "start": start, "end": end, "phase": "approach"})
        start = end
    success = kind != "failure"
    perturbation = None if kind == "success" else dict(make_perturbation().to_json(), onset_frame=onset, applied=True)
    payload = {
        "schema_version": 2, "episode_index": index, "num_frames": n, "fps": fps, "task": task, "arm": ARM,
        "scene": "fx_scene", "reference_frame": "robot base link (fx5_base)", "units": "m", "angle_units": "rad",
        "joint_names": list(JOINTS),
        "objects": {
            "cube_0": {"set": sets[0], "size_m": [0.02, 0.02, 0.02], "color": colors[0], "active": True,
                       "role": "target", **_track(rng, (0.2, 0.05, 0.01), n)},
            "box": {"set": sets[1], "size_m": [0.12, 0.1, 0.05], "color": colors[1], "active": True,
                    "role": "container", **_track(rng, (0.25, -0.05, 0.0), n)},
        },
        "ee_pose": _track(rng, (0.2, 0.0, 0.15), n),
        "instruction": f"put the {colors[0]} cube in the box", "roles": {"target": "cube_0", "container": "box"},
        "kind": kind, "success": success, "failure_mode": None if success else "missed_grasp",
        "furthest_stage": "placed" if success else "grasped", "stages": list(STAGES),
        "stage": [min(len(STAGES) - 1, t * len(STAGES) // n) if t else -1 for t in range(n)],
        "imitate": [not (kind != "success" and onset <= t < onset + 2) for t in range(n)],
        "phases": [{"name": "approach", "start": 0, "end": n}], "segments": segments,
        "perturbation": perturbation,
        "recovery": make_recovery_info(detected_frame=onset + 2, back_on_plan_frame=onset + 4).to_json()
        if kind == "recovery" else None,
        "noise": make_noise().to_json() if kind == "success" else None,
        "attempt": {"index": attempt_index, "seed": 2 ** 64 - 1 - attempt_index if seed is None else seed,
                    "shard": list(shard), "spec_fp": hex_digest(f"spec-{tag}")},
        "fingerprints": make_fingerprints_json(tag),
        "appearance": {"light_exposure": None, "robot_color": None}, "source": None,
        "generator": {"name": "scripted_expert", "git_commit": "0123abcd"}, "legacy_v1": False,
    }
    validate_sidecar(payload)
    return payload


def make_dr_copy(parent, *, index, tag):
    """A DR copy of ``parent`` (§8.6 shape): same traj and scenario key, new appearance and episode fp."""
    copy_ = copy.deepcopy(parent)
    copy_["episode_index"] = index
    copy_["fingerprints"]["appearance_fp"] = hex_digest(f"app-{tag}")
    copy_["fingerprints"]["episode_fp"] = hex_digest(f"episode-{tag}")
    copy_["appearance"] = {"light_exposure": 0.5, "robot_color": "black"}
    copy_["source"] = {"repo_id": "local/fx_source", "episode_index": parent["episode_index"],
                       "traj_fp": parent["fingerprints"]["traj_fp"], "episode_fp": parent["fingerprints"]["episode_fp"]}
    copy_["generator"] = {"name": "dr_replay", "git_commit": None}
    validate_sidecar(copy_)
    return copy_


def make_legacy(index=0, n=8):
    """A legacy_v1 v2 sidecar (nulls where §9.1 allows them)."""
    payload = make_sidecar("success", index=index, n=n)
    payload.update(legacy_v1=True, fingerprints=None, attempt=None, task=None, arm=None, scene=None, stage=None,
                   ee_pose=None, phases=[], stages=[], success=None, noise=None, instruction=None)
    validate_sidecar(payload)
    return payload


# ---------------------------------------------------------------------------------------------
# Frames (data/) and info.json


def features(*, J=J, cameras=()):
    """The info.json ``features`` of a lite dataset: action, state, the label columns (and cameras)."""
    out = {"action": {"dtype": "float32", "shape": [J], "names": list(LEROBOT[:J])},
           "observation.state": {"dtype": "float32", "shape": [J], "names": list(LEROBOT[:J])}}
    for name, spec in LABEL_FEATURES.items():
        out[name] = {"dtype": spec["dtype"], "shape": list(spec["shape"]), "names": list(spec["names"])}
    for cam in cameras:
        out[f"observation.images.{cam}"] = {"dtype": "video", "shape": [4, 6, 3], "names": ["height", "width", "channel"]}
    return out


def frame_arrays(sidecar, rng):
    """Per-frame arrays of one episode, consistent with ``sidecar`` (labels from stage/imitate/ee_pose)."""
    n = sidecar["num_frames"]
    stage = sidecar["stage"] if sidecar["stage"] is not None else [-1] * n
    if sidecar["ee_pose"] is not None:
        ee = np.concatenate([np.asarray(sidecar["ee_pose"]["position"]),
                             np.asarray(sidecar["ee_pose"]["orientation_wxyz"])], axis=1)
    else:
        ee = np.zeros((n, 7))
    return {"action": rng.uniform(-100, 100, (n, J)).astype(np.float32),
            "observation.state": rng.uniform(-100, 100, (n, J)).astype(np.float32),
            "label.stage": np.asarray(stage, np.int64), "label.imitate": np.asarray(sidecar["imitate"], np.int64),
            "label.ee_pose": ee.astype(np.float32), "frame_index": np.arange(n, dtype=np.int64)}


def _episode_table(index, arrays, start, fps, task, layout):
    n = len(arrays["frame_index"])
    if layout == "lite":
        vec = lambda a: pa.array([list(map(float, row)) for row in a], pa.list_(pa.float32()))  # noqa: E731
        lab = lambda a: pa.array([[int(v)] for v in a], pa.list_(pa.int64()))  # noqa: E731
    else:
        vec = lambda a: pa.FixedSizeListArray.from_arrays(pa.array(np.asarray(a, np.float32).ravel()), a.shape[1])  # noqa: E731
        lab = lambda a: pa.array(np.asarray(a, np.int64))  # noqa: E731
    return pa.table({
        "action": vec(arrays["action"]), "observation.state": vec(arrays["observation.state"]),
        "episode_index": pa.array([index] * n, pa.int64()), "frame_index": pa.array(arrays["frame_index"], pa.int64()),
        "index": pa.array(range(start, start + n), pa.int64()),
        "timestamp": pa.array(np.arange(n, dtype=np.float32) / np.float32(fps), pa.float32()),
        "task": pa.array([task] * n, pa.string()),
        "label.stage": lab(arrays["label.stage"]), "label.imitate": lab(arrays["label.imitate"]),
        "label.ee_pose": vec(arrays["label.ee_pose"]),
    })


def write_frames(root, sidecars, *, rng, layout="lite", fps=30, files=2):
    """Write ``data/`` for ``sidecars`` (in index order) and return ``{episode_index: arrays}``.

    ``layout="lite"``: one file ``data/chunk-000/file-{index:06d}.parquet`` per episode (§11).
    ``layout="v3"``: the episodes spread over ``files`` files, rows of a file in random order.
    """
    root = Path(root)
    out, tables, start = {}, [], 0
    for sc in sorted(sidecars, key=lambda s: s["episode_index"]):
        arrays = frame_arrays(sc, rng)
        out[sc["episode_index"]] = arrays
        tables.append((sc["episode_index"], _episode_table(sc["episode_index"], arrays, start, fps,
                                                           sc["instruction"] or "", layout)))
        start += sc["num_frames"]
    folder = root / "data" / "chunk-000"
    folder.mkdir(parents=True, exist_ok=True)
    if layout == "lite":
        for index, table in tables:
            pq.write_table(table, folder / f"file-{index:06d}.parquet")
    else:
        groups = [[t for k, (_, t) in enumerate(tables) if k % files == f] for f in range(files)]
        for f, group in enumerate(groups):
            if group:
                table = pa.concat_tables(group)
                table = table.take(pa.array(rng.permutation(table.num_rows)))
                pq.write_table(table, folder / f"file-{f:03d}.parquet")
    return out


def write_info(root, *, total, fps=30, layout="lite", feats=None, extra=None):
    info = {"codebase_version": "v3.0-lite" if layout == "lite" else "v3.0", "fps": fps, "total_episodes": total,
            "features": features() if feats is None else feats}
    info.update(extra or {})
    path = Path(root) / "meta" / "info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, sort_keys=True))
    return info


# ---------------------------------------------------------------------------------------------
# Manifest, plan, audit stub


def accepted_line(sc):
    fp = sc["fingerprints"]
    return {"v": 1, "task": sc["task"], "kind": sc["kind"], "attempt_index": sc["attempt"]["index"],
            "seed": sc["attempt"]["seed"], "shard": list(sc["attempt"]["shard"]), "status": "accepted", "reason": None,
            "episode_index": sc["episode_index"], "scenario_key": fp["scenario_key"], "spec_fp": fp["spec_fp"],
            "traj_fp": fp["traj_fp"], "appearance_fp": fp["appearance_fp"], "episode_fp": fp["episode_fp"],
            "success": sc["success"], "failure_mode": sc["failure_mode"], "furthest_stage": sc["furthest_stage"],
            "perturbation": sc["perturbation"]["type"] if sc["perturbation"] else None,
            "num_frames": sc["num_frames"], "wall_s": 1.25}


def other_line(task, kind, attempt_index, shard, status, tag):
    skipped = status == "skipped"
    return {"v": 1, "task": task, "kind": kind, "attempt_index": attempt_index, "seed": 1000 + attempt_index,
            "shard": list(shard), "status": status, "reason": "scenario_seen" if skipped else "expert_failed",
            "episode_index": None, "scenario_key": hex_digest(f"s-{tag}"), "spec_fp": hex_digest(f"p-{tag}"),
            "traj_fp": None if skipped else hex_digest(f"t-{tag}"),
            "appearance_fp": None if skipped else hex_digest(f"a-{tag}"),
            "episode_fp": None if skipped else hex_digest(f"e-{tag}"), "success": None if skipped else False,
            "failure_mode": None, "furthest_stage": None, "perturbation": None,
            "num_frames": None if skipped else 7, "wall_s": 0.5}


def plan_json(*, task="fx_pick", shard=(0, 1), config_sha="c" * 64, task_toml_sha="d" * 64, master_seed=7):
    return {"v": 1, "config_sha256": config_sha, "config": {"note": "test"}, "master_seed": master_seed,
            "git_commit": "0123abcd", "shard": list(shard), "task": task, "task_toml_sha256": task_toml_sha,
            "arm": ARM, "arm_files_sha256": {"arm.toml": "a" * 64, "kinematics.json": "b" * 64}, "scene": "fx_scene",
            "quotas": {"success": 10, "failure": 5, "recovery": 5}, "holdout": {"colors": [], "counts": []},
            "created_at": "2026-01-01T00:00:00+00:00"}


def content_digest(root):
    """§10.1 content digest as pinned for WP12 (see the module docstring)."""
    root = Path(root)
    tree = {}
    for top in TREES:
        if (root / top).is_dir():
            tree.update({f"{top}/{rel}": sha for rel, sha in tree_sha256(root / top).items()})
    tree.pop(AUDIT_REL, None)
    return json_sha256(tree)


def write_audit(root, *, passed=True, digest=None):
    """Stub of WP7 ``write_audit_report``: the §10.1 AuditReport fields as JSON in meta/audit_report.json."""
    root = Path(root)
    fps = _episode_fps(root)
    record = {"v": 1, "root": str(root), "passed": passed, "episodes": len(list_sidecars(root)),
              "checks": [{"name": name, "passed": passed, "details": [] if passed else ["stub failure"]}
                         for name in CHECK_NAMES],
              "content_digest": content_digest(root) if digest is None else digest,
              "fingerprints_digest": json_sha256(fps), "created_at": "2026-01-01T00:00:00+00:00"}
    path = root / AUDIT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def _episode_fps(root):
    try:
        sidecars = [read_sidecar(root, i) for i in list_sidecars(root)]
    except Exception:  # noqa: BLE001 - placeholder sidecar files in the upload tests
        return []
    return sorted(sc["fingerprints"]["episode_fp"] for sc in sidecars if sc["fingerprints"] is not None)


def build_dataset(root, sidecars, *, rng=None, layout="lite", fps=30, plan=None, manifest_extra=(), audit=True,
                  feats=None, info_extra=None, manifest=True, consolidate=False):
    """Write a complete small dataset at ``root`` and return ``{episode_index: frame arrays}``.

    ``sidecars`` are written with ``write_sidecar``; accepted manifest lines are built from the
    originals among them (DR copies have no generation attempt), and
    ``manifest_extra`` (rejected / skipped line dicts, each with an ``"after"`` attempt index) are
    interleaved by attempt index. ``plan`` (a dict) is written with ``write_plan_json``. ``audit``
    writes a passing audit record last (after ``consolidate`` when requested).
    """
    root = Path(root)
    rng = np.random.default_rng(0) if rng is None else rng
    for sc in sidecars:
        write_sidecar(root, sc)
    frames = write_frames(root, sidecars, rng=rng, layout=layout, fps=fps)
    write_info(root, total=len(sidecars), fps=fps, layout=layout, feats=feats, extra=info_extra)
    if manifest:
        lines = [accepted_line(sc) for sc in sidecars if sc["source"] is None and sc["attempt"] is not None] + [dict(l) for l in manifest_extra]
        lines.sort(key=lambda l: (l["attempt_index"], KINDS.index(l["kind"])))
        m = Manifest(root / MANIFEST_REL)
        for line in lines:
            m.append(ManifestLine.from_json(line))
    if plan is not None:
        write_plan_json(root, plan)
    if consolidate:
        from core.datasets.consolidate import consolidate as _consolidate
        _consolidate(root)
    if audit:
        write_audit(root)
    return frames


def build_shard(root, shard, n, *, episodes, task="fx_pick", rng_seed=None, plan_kwargs=None, tag_prefix="",
                sidecar_hook=None, with_plan=True, extra_lines=(), **kwargs):
    """Shard ``shard`` of ``n`` of one generation run: owned attempt indices ``shard + n·j``.

    Episode ``e`` comes from attempt ``shard + n·(2e + 1)``; attempts ``shard + n·2e`` are rejected
    (even e) or skipped (odd e), so the manifest interleaves non-accepted lines. ``sidecar_hook(list)``
    may edit the sidecars before writing (e.g. to force a collision); ``extra_lines`` are more
    manifest line dicts; ``with_plan=False`` writes no generation_plan.json.
    """
    rng = np.random.default_rng(1000 + shard if rng_seed is None else rng_seed)
    sidecars, extra = [], []
    for e in range(episodes):
        kind = KINDS[(e + shard) % 3]
        attempt = shard + n * (2 * e + 1)
        sidecars.append(make_sidecar(kind, index=e, n=int(rng.integers(8, 15)), tag=f"{tag_prefix}{task}-{shard}-{e}",
                                     rng=rng, task=task, shard=(shard, n), attempt_index=attempt))
        extra.append(other_line(task, kind, shard + n * 2 * e, (shard, n), "rejected" if e % 2 == 0 else "skipped",
                                f"{tag_prefix}{task}-{shard}-{e}-x"))
    if sidecar_hook is not None:
        sidecar_hook(sidecars)
    plan = plan_json(task=task, shard=(shard, n), **(plan_kwargs or {})) if with_plan else None
    frames = build_dataset(root, sidecars, rng=rng, plan=plan, manifest_extra=extra + list(extra_lines), **kwargs)
    return sidecars, frames


def read_table_by_episode(root):
    """``{episode_index: pyarrow.Table sorted by frame_index}`` over ``root/data/**/*.parquet``."""
    out = {}
    for path in sorted((Path(root) / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        for e in sorted(set(table.column("episode_index").to_pylist())):
            part = table.filter(pc.equal(table.column("episode_index"), e))
            out[e] = pa.concat_tables([out[e], part]) if e in out else part
    return {e: t.sort_by("frame_index") for e, t in out.items()}


def tree_bytes(root):
    """``{relative posix path: sha256}`` of every file under ``root`` (for byte-level comparisons)."""
    return tree_sha256(root)


# ---------------------------------------------------------------------------------------------
# Storage fakes (§10.3)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class FakeLfs(dict):
    """Like ``huggingface_hub.hf_api.BlobLfsInfo``: a dict that also exposes ``size``/``sha256`` attributes."""

    def __init__(self, size, sha256):
        super().__init__(size=size, sha256=sha256, pointer_size=134)
        self.size, self.sha256, self.pointer_size = size, sha256, 134


@dataclasses.dataclass
class FakeRepoFile:
    path: str
    size: int
    blob_id: str
    lfs: FakeLfs | None = None


class SimulatedNetworkError(ConnectionError):
    pass


class FakeHubApi:
    """In-memory ``HfApi`` subset with huggingface_hub signatures (keyword-only where the real one is).

    ``files`` maps path → bytes; ``uploads`` lists the paths of successful ``upload_file`` calls.
    ``fail_at=k`` makes the k-th ``upload_file`` call (1-based, counted over the object's life) raise
    ``SimulatedNetworkError`` without storing anything. ``lfs`` decides per path whether the file is
    stored as LFS (so ``get_paths_info`` reports a sha256); default: every file.
    """

    def __init__(self, repo_id="org/fx", *, revision="main", fail_at=None, lfs=None):
        self.repo_id, self.revision = repo_id, revision
        self.files = {".gitattributes": b"*.parquet filter=lfs diff=lfs merge=lfs -text\n"}
        self.uploads, self.calls, self.commit_messages = [], [], []
        self.fail_at, self.lfs = fail_at, (lambda path: True) if lfs is None else lfs
        self._upload_calls = 0

    def _check(self, repo_id, repo_type, revision):
        assert repo_id == self.repo_id, repo_id
        assert repo_type == "dataset", repo_type
        assert revision in (self.revision, None if self.revision == "main" else self.revision), revision

    def list_repo_files(self, repo_id, *, revision=None, repo_type=None, token=None):
        self.calls.append("list_repo_files")
        self._check(repo_id, repo_type, revision)
        return sorted(self.files)

    def get_paths_info(self, repo_id, paths, *, expand=False, revision=None, repo_type=None, token=None):
        self.calls.append("get_paths_info")
        self._check(repo_id, repo_type, revision)
        out = []
        for path in ([paths] if isinstance(paths, str) else list(paths)):
            if path in self.files:
                data = self.files[path]
                lfs = FakeLfs(len(data), sha256_bytes(data)) if self.lfs(path) and path != ".gitattributes" else None
                out.append(FakeRepoFile(path=path, size=len(data), blob_id=hashlib.sha1(data).hexdigest(), lfs=lfs))
        return out

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, token=None, repo_type=None, revision=None,
                    commit_message=None, commit_description=None, create_pr=None, parent_commit=None,
                    run_as_future=False):
        self.calls.append("upload_file")
        self._check(repo_id, repo_type, revision)
        self._upload_calls += 1
        if self.fail_at is not None and self._upload_calls == self.fail_at:
            raise SimulatedNetworkError(f"simulated failure uploading {path_in_repo}")
        if isinstance(path_or_fileobj, (str, os.PathLike)):
            data = Path(path_or_fileobj).read_bytes()
        elif isinstance(path_or_fileobj, bytes):
            data = path_or_fileobj
        else:
            data = path_or_fileobj.read()
        self.files[path_in_repo] = data
        self.uploads.append(path_in_repo)
        self.commit_messages.append(commit_message)

    def create_repo(self, repo_id, *, token=None, private=None, repo_type=None, exist_ok=False, **kwargs):
        self.calls.append("create_repo")
        return repo_id


class FakeFs:
    """In-memory fsspec-like filesystem: ``find(path)``, ``info(path)``, ``put_file(lpath, rpath)``.

    Paths are stored without protocol (fsspec strips ``s3://`` and returns bare paths from ``find``).
    ``info`` returns ``{"name", "size", "type", "sha256"}`` (``with_sha=False`` omits the sha256).
    ``fail_at`` works as in ``FakeHubApi``.
    """

    def __init__(self, *, fail_at=None, with_sha=True):
        self.objects, self.uploads = {}, []
        self.fail_at, self.with_sha, self._put_calls = fail_at, with_sha, 0

    @staticmethod
    def strip(path):
        return str(path).split("://", 1)[-1].rstrip("/")

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        bare = self.strip(path)
        return sorted(k for k in self.objects if k.startswith(bare + "/"))

    def info(self, path, **kwargs):
        bare = self.strip(path)
        if bare not in self.objects:
            raise FileNotFoundError(path)
        data = self.objects[bare]
        out = {"name": bare, "size": len(data), "type": "file"}
        if self.with_sha:
            out["sha256"] = sha256_bytes(data)
        return out

    def put_file(self, lpath, rpath, callback=None, **kwargs):
        self._put_calls += 1
        if self.fail_at is not None and self._put_calls == self.fail_at:
            raise SimulatedNetworkError(f"simulated failure uploading {rpath}")
        self.objects[self.strip(rpath)] = Path(lpath).read_bytes()
        self.uploads.append(self.strip(rpath))


# ---------------------------------------------------------------------------------------------
# Self-tests of the builders (they encode contract clauses the WP12 tests rely on)


class BuilderSelfTests(unittest.TestCase):
    """WP12 test builders: the datasets they write follow §8.3, §8.5, §9.1, §10.1 and §11."""

    def test_sidecars_valid_and_round_trip(self):
        """§9.1 built sidecars validate and read back equal; DR copies and legacy sidecars too."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            items = [make_sidecar(k, index=i) for i, k in enumerate(KINDS)]
            items.append(make_dr_copy(items[0], index=3, tag="copy"))
            items.append(make_legacy(index=4))
            for sc in items:
                write_sidecar(tmp, sc)
                self.assertEqual(read_sidecar(tmp, sc["episode_index"]), sc)

    def test_shard_layout(self):
        """§11 lite layout, §8.3 manifest (accepted lines 0..M−1) and §8.5 plan json of a built shard."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "s1"
            sidecars, frames = build_shard(root, 1, 3, episodes=3)
            info = json.loads((root / "meta" / "info.json").read_text())
            self.assertEqual(set(info), {"codebase_version", "fps", "total_episodes", "features"})
            self.assertEqual((info["codebase_version"], info["total_episodes"]), ("v3.0-lite", 3))
            self.assertEqual(sorted(p.name for p in (root / "data" / "chunk-000").iterdir()),
                             [f"file-{i:06d}.parquet" for i in range(3)])
            lines = Manifest(root / MANIFEST_REL).lines()
            self.assertEqual(len(lines), 6)
            self.assertEqual([l.episode_index for l in Manifest(root / MANIFEST_REL).accepted()], [0, 1, 2])
            self.assertTrue(all(l.attempt_index % 3 == 1 for l in lines))
            self.assertEqual(read_plan_json(root)["shard"], [1, 3])
            tables = read_table_by_episode(root)
            for e, sc in enumerate(sidecars):
                self.assertEqual(tables[e].num_rows, sc["num_frames"])
                np.testing.assert_array_equal(np.asarray(tables[e].column("label.stage").to_pylist()).ravel(),
                                              frames[e]["label.stage"])

    def test_audit_stub(self):
        """§10.1 the stub record has the AuditReport fields and the digest of the current content."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            build_shard(tmp, 0, 1, episodes=2)
            record = json.loads((Path(tmp) / AUDIT_REL).read_text())
            self.assertEqual(set(record), {"v", "root", "passed", "episodes", "checks", "content_digest",
                                           "fingerprints_digest", "created_at"})
            self.assertTrue(record["passed"])
            self.assertEqual(record["content_digest"], content_digest(tmp))
            (Path(tmp) / "README.md").write_text("outside the digested trees")
            self.assertEqual(record["content_digest"], content_digest(tmp))
            (Path(tmp) / "meta" / "info.json").write_text("{}")
            self.assertNotEqual(record["content_digest"], content_digest(tmp))

    def test_fakes_record_uploads(self):
        """§10.3 the fakes store bytes and report sha256 like the clients they stand for."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.bin"
            path.write_bytes(b"abc")
            api = FakeHubApi()
            api.upload_file(path_or_fileobj=str(path), path_in_repo="a/x.bin", repo_id="org/fx", repo_type="dataset",
                            revision="main", commit_message="m")
            info = api.get_paths_info("org/fx", ["a/x.bin"], repo_type="dataset", revision="main")[0]
            self.assertEqual(info.lfs.sha256, sha256_bytes(b"abc"))
            self.assertEqual(info.lfs["sha256"], sha256_bytes(b"abc"))
            with self.assertRaises(TypeError):
                api.upload_file(str(path), "a/y.bin", "org/fx")
            fs = FakeFs()
            fs.put_file(str(path), "s3://bkt/ds/a/x.bin")
            self.assertEqual(fs.find("s3://bkt/ds"), ["bkt/ds/a/x.bin"])
            self.assertEqual(fs.info("bkt/ds/a/x.bin")["sha256"], sha256_bytes(b"abc"))


if __name__ == "__main__":
    unittest.main()
