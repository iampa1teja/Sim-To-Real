"""Atomic evaluation checkpoints and validation of fixed-benchmark continuations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile


def write_report(path, report):
    """Replace the report atomically; an interrupted write retains the last episode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_report(path):
    try:
        report = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot resume results {path}: {error}") from error
    if not isinstance(report, dict) or not isinstance(report.get("episodes"), list):
        raise ValueError(f"Invalid evaluation report: {path}")
    return report


def find_report(target, pattern, fingerprint, checkpoint, task):
    """Select the newest matching report, including already complete evaluations."""
    target = Path(target)
    if not target.is_dir():
        candidates = [target]
    else:
        candidates = sorted(target.glob(pattern), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        try:
            report = read_report(path)
        except ValueError:
            if not target.is_dir():
                raise
            continue
        if (report.get("checkpoint") == checkpoint and report.get("task") == task
                and report.get("config_fingerprint", fingerprint) == fingerprint):
            return path.resolve()
    raise ValueError("No matching evaluation results to resume; run once without --resume or name its results JSON.")


def load_episodes(path, expected, schedule):
    """Require completed episodes to be an exact prefix of this benchmark schedule."""
    report = read_report(path)
    for key, value in expected.items():
        # Older reports lack a launch fingerprint and detailed continuation identity.
        # Their recorded settings and schedule are still checked below.
        if key in ("config_fingerprint", "resume_settings") and key not in report:
            continue
        if report.get(key) != value:
            raise ValueError(f"Cannot resume: {key} differs from the saved evaluation")
    episodes = report["episodes"]
    if len(episodes) > len(schedule):
        raise ValueError("Cannot resume: saved episodes exceed the selected benchmark")
    for index, (row, start) in enumerate(zip(episodes, schedule)):
        if not isinstance(row, dict) or row.get("episode") != index:
            raise ValueError("Cannot resume: saved episodes are not a contiguous prefix")
        for key, value in (("start_id", start["id"]), ("repeat", start["repeat"]),
                           ("policy_seed", start["policy_seed"]), ("split", start["split"])):
            if row.get(key) != value:
                raise ValueError(f"Cannot resume: episode {index + 1} has a different {key}")
        if (not isinstance(row.get("success"), bool) or type(row.get("steps")) is not int
                or row["steps"] < 1 or (row["success"] and row.get("success_step") != row["steps"])):
            raise ValueError(f"Cannot resume: episode {index + 1} is not a completed attempt")
    return report


def next_video_attempt(directory):
    """Keep completed recordings and abandoned partial attempts without overwriting."""
    indices = []
    for path in Path(directory).glob("episode_*_*.mp4"):
        match = re.fullmatch(r"episode_(\d+)_.+\.mp4", path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


if __name__ == "__main__":
    import sys

    try:
        print(find_report(*sys.argv[1:]))
    except ValueError as error:
        sys.exit(f"ERROR: {error}")
