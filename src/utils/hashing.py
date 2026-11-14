# Content hashes recorded as provenance in eval sets, results and prepared datasets.
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_json(payload):
    # Sorted keys, no whitespace, no NaN: the same payload always hashes the same.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_sha256(payload):
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def legacy_json_sha256(payload):
    # Sorted keys with json's default ", "/": " separators and NaN allowed. Stored values depend on
    # this exact form: recording-alignment markers' calibration_sha256 (utils/datasets/alignment.py)
    # and the "-cal-<12 hex>" dataset-root suffix (sim/recording/setup.py). Use json_sha256 for new hashes.
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def tree_sha256(root, pattern="**/*"):
    # Per-file hashes keyed by posix path relative to root.
    root = Path(root)
    return {p.relative_to(root).as_posix(): file_sha256(p)
            for p in sorted(root.glob(pattern)) if p.is_file()}
