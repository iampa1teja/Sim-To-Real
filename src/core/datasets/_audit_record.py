"""Read WP7 audit records without importing the audit implementation."""

from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
import json as _json
import os as _os
import re as _re
import tempfile as _tempfile

from core.generate.errors import AuditFailed as _AuditFailed
from core.utils.hashing import file_sha256 as _file_sha256, json_sha256 as _json_sha256


@_dataclass(frozen=True)
class _AuditRecord:
    v: int
    root: str
    passed: bool
    episodes: int
    checks: tuple
    content_digest: str
    fingerprints_digest: str
    created_at: str


def _content_digest(root):
    root = _Path(root)
    files = {}
    for top in ("meta", "data", "videos", "task_meta"):
        for path in (root / top).rglob("*"):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and relative != "meta/audit_report.json":
                files[relative] = _file_sha256(path)
    return _json_sha256(files)


def _read_audit(root):
    try:
        record = _json.loads((_Path(root) / "meta" / "audit_report.json").read_text(encoding="utf-8"))
        valid = (type(record) is dict and set(record) == set(_AuditRecord.__dataclass_fields__)
                 and type(record["v"]) is int and record["v"] == 1
                 and type(record["root"]) is str and type(record["passed"]) is bool
                 and type(record["episodes"]) is int and record["episodes"] >= 0
                 and type(record["checks"]) is list
                 and type(record["content_digest"]) is str
                 and _re.fullmatch(r"[0-9a-f]{64}", record["content_digest"]) is not None
                 and type(record["fingerprints_digest"]) is str and type(record["created_at"]) is str)
        if not valid:
            raise ValueError("Invalid audit record fields")
        return _AuditRecord(**{**record, "checks": tuple(record["checks"])})
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        raise _AuditFailed(f"Missing or malformed audit at {root}: {exc}") from exc


def _require_current(root, report):
    if report.passed is not True or report.content_digest != _content_digest(root):
        raise _AuditFailed(f"Audit failed or is stale: {root}")


def _atomic_text(path, text):
    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = _tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            _os.fsync(stream.fileno())
        _os.replace(name, path)
    finally:
        _Path(name).unlink(missing_ok=True)
