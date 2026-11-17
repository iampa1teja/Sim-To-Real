"""Independent checks of persisted dataset labels, provenance and uniqueness."""

from dataclasses import dataclass as _dataclass, asdict as _asdict
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path as _Path
import json as _json
import re as _re

import numpy as _np

from core.datasets._audit_record import _atomic_text, _content_digest
from core.datasets.frames import read_frames as _read_frames
from core.datasets.task_meta import list_sidecars as _list_sidecars, read_sidecar as _read_sidecar
from core.generate.errors import SchemaError as _SchemaError
from core.generate.fingerprint import traj_fp as _traj_fp, appearance_fp as _appearance_fp, episode_fp as _episode_fp
from core.generate.manifest import Manifest as _Manifest, read_plan_json as _read_plan_json
from core.tasks.base import load_task_spec as _load_task_spec
from core.utils.hashing import json_sha256 as _json_sha256


CHECKS = ("sidecars_valid", "counts_match", "labels_match", "fingerprints_recomputed", "episode_fp_unique",
          "traj_fp_unique_originals", "scenario_key_unique_originals", "holdout_absent", "dr_sources_exist",
          "dr_appearance_distinct")


@_dataclass(frozen=True)
class AuditReport:
    v: int
    root: str
    passed: bool
    episodes: int
    checks: tuple[dict, ...]
    content_digest: str
    fingerprints_digest: str
    created_at: str

    def __post_init__(self):
        object.__setattr__(self, "checks", tuple({**c, "details": list(c["details"])} for c in self.checks))


def content_digest(root):
    return _content_digest(root)


def _check_report(record):
    if (type(record) is not dict or set(record) != set(AuditReport.__dataclass_fields__)
            or type(record["v"]) is not int or record["v"] != 1
            or type(record["root"]) is not str or type(record["passed"]) is not bool
            or type(record["episodes"]) is not int or record["episodes"] < 0
            or type(record["created_at"]) is not str or type(record["checks"]) is not list):
        raise _SchemaError("Invalid audit report fields")
    for key in ("content_digest", "fingerprints_digest"):
        if type(record[key]) is not str or _re.fullmatch(r"[0-9a-f]{64}", record[key]) is None:
            raise _SchemaError("Invalid audit digest")
    for check in record["checks"]:
        if (type(check) is not dict or set(check) != {"name", "passed", "details"}
                or type(check["name"]) is not str or type(check["passed"]) is not bool
                or type(check["details"]) is not list or len(check["details"]) > 50
                or any(type(d) is not str for d in check["details"])
                or not check["passed"] and not check["details"]):
            raise _SchemaError("Invalid audit check")
    if tuple(c["name"] for c in record["checks"]) != CHECKS:
        raise _SchemaError("Invalid audit check order")


def write_audit_report(root, report):
    record = _asdict(report)
    record["checks"] = list(record["checks"])
    _check_report(record)
    path = _Path(root) / "meta" / "audit_report.json"
    _atomic_text(path, _json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return path


def read_audit_report(root):
    try:
        record = _json.loads((_Path(root) / "meta" / "audit_report.json").read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise _SchemaError("Invalid audit report JSON") from exc
    _check_report(record)
    return AuditReport(**record)


def audit_dataset(root, *, source_root=None, check_manifest=True):
    root = _Path(root)
    checks = {name: dict(name=name, passed=True, details=[]) for name in CHECKS}

    def fail(name, message):
        check = checks[name]
        check["passed"] = False
        if len(check["details"]) < 50:
            check["details"].append(str(message))

    sidecars = {}
    indices = _list_sidecars(root)
    for i in indices:
        try:
            sidecars[i] = _read_sidecar(root, i)
        except (ValueError, OSError) as exc:
            fail("sidecars_valid", f"Episode {i}: {exc}")
    try:
        frames = _read_frames(root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        frames = {}
        for name in ("counts_match", "labels_match", "fingerprints_recomputed"):
            fail(name, f"Cannot read frame data: {exc}")
    try:
        info = _json.loads((root / "meta" / "info.json").read_text())
        count = info["total_episodes"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid total_episodes")
        if indices != list(range(count)) or sorted(frames) != indices or sorted(sidecars) != indices:
            fail("counts_match", "info.json, sidecar and data episode indices differ")
    except (ValueError, OSError, TypeError, KeyError) as exc:
        fail("counts_match", f"Cannot read episode count: {exc}")
    originals = {i: sc for i, sc in sidecars.items() if sc["source"] is None}
    manifest_path = root / "meta" / "generation_manifest.jsonl"
    if check_manifest and manifest_path.exists():
        try:
            lines = _Manifest(manifest_path).accepted()
            if sorted(line.episode_index for line in lines) != sorted(originals):
                fail("counts_match", "Accepted manifest episodes differ from originals")
            for line in lines:
                sc = originals.get(line.episode_index)
                if sc is None:
                    continue
                fp, attempt = sc["fingerprints"] or {}, sc["attempt"] or {}
                pairs = [(line.kind.value, sc["kind"]), (line.attempt_index, attempt.get("index")),
                         (line.seed, attempt.get("seed")), (line.num_frames, sc["num_frames"])]
                pairs.extend((getattr(line, key), fp.get(key)) for key in ("scenario_key", "spec_fp", "traj_fp", "appearance_fp", "episode_fp"))
                if any(a != b for a, b in pairs):
                    fail("counts_match", f"Manifest disagrees with episode {line.episode_index}")
        except (ValueError, OSError, TypeError, KeyError) as exc:
            fail("counts_match", f"Unreadable manifest: {exc}")
    for index, sc in sidecars.items():
        if index not in frames:
            fail("counts_match", f"Missing data for episode {index}")
            continue
        data = frames[index]
        if len(data["frame_index"]) != sc["num_frames"]:
            fail("counts_match", f"Frame count differs for episode {index}")
        if (sc["stage"] is not None and not _np.array_equal(data["label.stage"], sc["stage"])
                or not _np.array_equal(data["label.imitate"], sc["imitate"])):
            fail("labels_match", f"Stage or imitate differs for episode {index}")
        if sc["ee_pose"] is not None:
            ee = _np.concatenate([sc["ee_pose"]["position"], sc["ee_pose"]["orientation_wxyz"]], axis=1)
            if data["label.ee_pose"].shape != ee.shape or not _np.allclose(data["label.ee_pose"], ee, atol=1e-5, rtol=0):
                fail("labels_match", f"EE pose differs for episode {index}")
        try:
            fp = sc["fingerprints"]
            traj = _traj_fp(data["action"], data["observation.state"], {name: o["position"][0] for name, o in sc["objects"].items()})
            appearance = _appearance_fp({name: o["color"] for name, o in sc["objects"].items()}, sc["instruction"],
                                         sc["appearance"]["light_exposure"], sc["appearance"]["robot_color"],
                                         sc["appearance"].get("camera_offsets"))
            episode = _episode_fp(traj, appearance)
            if fp is None or any(fp[key] != value for key, value in (("traj_fp", traj), ("appearance_fp", appearance), ("episode_fp", episode))):
                fail("fingerprints_recomputed", f"Fingerprints differ for episode {index}")
        except (ValueError, TypeError, KeyError) as exc:
            fail("fingerprints_recomputed", f"Cannot recompute episode {index}: {exc}")
    for name, subset, key in (("episode_fp_unique", sidecars, "episode_fp"),
                               ("traj_fp_unique_originals", originals, "traj_fp"),
                               ("scenario_key_unique_originals", originals, "scenario_key")):
        seen = {}
        for index, sc in subset.items():
            value = (sc["fingerprints"] or {}).get(key)
            if value is not None:
                if value in seen:
                    fail(name, f"Episodes {seen[value]} and {index} share {key}")
                seen[value] = index
    holdout = None
    if (root / "meta" / "generation_plan.json").exists():
        try:
            holdout = _read_plan_json(root)["holdout"]
        except (ValueError, OSError) as exc:
            fail("holdout_absent", f"Cannot read holdouts: {exc}")
    task_holdouts = {}
    for index, sc in sidecars.items():
        try:
            if holdout is not None:
                colors, counts = holdout["colors"], holdout["counts"]
            else:
                task_name = sc["task"]
                if task_name not in task_holdouts:
                    task_holdouts[task_name] = _load_task_spec(task_name).holdout
                spec = task_holdouts[task_name]
                colors, counts = spec.colors, spec.counts
            objects = sc["objects"]
            if any(obj["color"] in colors for obj in objects.values()):
                fail("holdout_absent", f"Held-out color in episode {index}")
            target = objects.get(sc["roles"].get("target"), {})
            target_set = target.get("set")
            if target_set is not None and sum(obj["set"] == target_set for obj in objects.values()) in counts:
                fail("holdout_absent", f"Held-out object count in episode {index}")
        except (ValueError, OSError, TypeError, KeyError) as exc:
            fail("holdout_absent", f"Cannot check episode {index}: {exc}")
    sources = {}
    if source_root is None:
        checks["dr_sources_exist"]["details"].append("Source lookup skipped: no source_root")
    else:
        try:
            sources = {sc["fingerprints"]["episode_fp"]: sc for sc in
                       (_read_sidecar(source_root, i) for i in _list_sidecars(source_root)) if sc["fingerprints"] is not None}
        except (ValueError, OSError, KeyError) as exc:
            fail("dr_sources_exist", f"Cannot read source: {exc}")
    parents = {sc["fingerprints"]["episode_fp"]: sc for sc in originals.values() if sc["fingerprints"] is not None}
    parents.update(sources)
    groups = {}
    for index, sc in sidecars.items():
        source = sc["source"]
        if source is None:
            continue
        parent_fp = source["episode_fp"]
        if source_root is not None:
            parent = sources.get(parent_fp)
            if parent is None or parent["fingerprints"]["traj_fp"] != source["traj_fp"]:
                fail("dr_sources_exist", f"Missing source for episode {index}")
        fp = sc["fingerprints"] or {}
        if fp.get("traj_fp") != source["traj_fp"]:
            fail("dr_appearance_distinct", f"Episode {index} changed its source trajectory")
        if parent_fp not in groups:
            parent = parents.get(parent_fp)
            groups[parent_fp] = {parent["fingerprints"]["appearance_fp"]} if parent else set()
        appearance = fp.get("appearance_fp")
        if appearance in groups[parent_fp]:
            fail("dr_appearance_distinct", f"Episode {index} repeats parent or sibling appearance")
        groups[parent_fp].add(appearance)
    fps = sorted(sc["fingerprints"]["episode_fp"] for sc in sidecars.values()
                 if sc["fingerprints"] is not None and sc["fingerprints"]["episode_fp"] is not None)
    return AuditReport(1, str(root), all(c["passed"] for c in checks.values()), len(indices), tuple(checks.values()),
                        content_digest(root), _json_sha256(fps), _datetime.now(_timezone.utc).isoformat())
