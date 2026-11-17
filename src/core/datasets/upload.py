"""Resumable uploads through injected storage clients, gated by a current audit."""

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import Protocol as _Protocol, runtime_checkable as _runtime_checkable
import tempfile as _tempfile

from core.datasets._audit_record import _read_audit, _require_current
from core.datasets.task_meta import list_sidecars as _list_sidecars, sidecar_path as _sidecar_path
from core.generate.errors import AuditFailed as _AuditFailed, GenerationError as _GenerationError
from core.utils.hashing import file_sha256 as _file_sha256


@_runtime_checkable
class StorageClient(_Protocol):
    def list_files(self) -> dict[str, str | None]: ...
    def upload_file(self, local: _Path, remote: str) -> None: ...
    def finalize(self, message: str) -> None: ...


class HubStorage:
    def __init__(self, api, repo_id, *, revision="main", repo_type="dataset"):
        self._api, self._repo_id = api, repo_id
        self._revision, self._repo_type = revision, repo_type

    def list_files(self):
        paths = self._api.list_repo_files(self._repo_id, repo_type=self._repo_type, revision=self._revision)
        if not paths:
            return {}
        result = {}
        for info in self._api.get_paths_info(self._repo_id, paths, repo_type=self._repo_type, revision=self._revision):
            lfs = info.lfs
            result[info.path] = (lfs.get("sha256") if isinstance(lfs, _Mapping) else getattr(lfs, "sha256", None))
        return result

    def upload_file(self, local, remote):
        self._api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=self._repo_id,
                              repo_type=self._repo_type, revision=self._revision, commit_message=f"Upload {remote}")

    def finalize(self, message):
        pass


class BucketStorage:
    def __init__(self, fs, prefix):
        self._fs, self._prefix = fs, str(prefix).rstrip("/")

    def list_files(self):
        prefix = self._prefix.split("://", 1)[-1] + "/"
        result = {}
        for path in self._fs.find(self._prefix):
            bare = str(path).split("://", 1)[-1]
            if bare.startswith(prefix):
                remote = bare[len(prefix):]
                if remote and remote != "_COMPLETE":
                    result[remote] = self._fs.info(path).get("sha256")
        return result

    def upload_file(self, local, remote):
        self._fs.put_file(str(local), f"{self._prefix}/{remote}")

    def finalize(self, message):
        with _tempfile.TemporaryDirectory() as folder:
            path = _Path(folder) / "_COMPLETE"
            path.write_text(message, encoding="utf-8")
            self.upload_file(path, "_COMPLETE")


@_dataclass(frozen=True)
class UploadItem:
    local: _Path
    remote: str
    sha256: str
    size: int

    def __post_init__(self):
        object.__setattr__(self, "local", _Path(self.local))


@_dataclass(frozen=True)
class UploadPlan:
    items: tuple[UploadItem, ...]
    skipped: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "skipped", tuple(self.skipped))


def plan_upload(root, storage, *, allow_overwrite=False):
    root = _Path(root)
    _require_current(root, _read_audit(root))
    if not (root / "task_meta" / "episodes.parquet").is_file():
        raise _AuditFailed("Consolidated sidecar parquet is missing")
    if any(path.is_file() and path.name.endswith(".tmp") for path in root.rglob("*")):
        raise _GenerationError("Dataset contains temporary files")
    if (root / "mp4").exists():
        raise _GenerationError("Dataset contains an mp4 preview tree")
    if any(path.is_file() for path in (root / "images").rglob("*")):
        raise _GenerationError("Dataset contains an open episode image buffer")
    indices = _list_sidecars(root)
    excluded = {_sidecar_path(root, i) for i in indices} if len(indices) >= 1000 else set()
    paths = []
    audit = root / "meta" / "audit_report.json"
    for top in ("data", "videos", "task_meta", "meta"):
        paths.extend(sorted((p for p in (root / top).rglob("*") if p.is_file() and p not in excluded and p != audit),
                            key=lambda p: p.relative_to(root).as_posix()))
    paths.append(audit)
    remote_files = storage.list_files()
    items, skipped = [], []
    for path in paths:
        remote = path.relative_to(root).as_posix()
        digest = _file_sha256(path)
        existing = remote_files.get(remote)
        if existing == digest:
            skipped.append(remote)
        elif existing is not None and not allow_overwrite:
            raise _GenerationError(f"Remote file differs: {remote}")
        else:
            items.append(UploadItem(path, remote, digest, path.stat().st_size))
    return UploadPlan(tuple(items), tuple(skipped))


def run_upload(plan, storage, *, message):
    for item in plan.items:
        storage.upload_file(item.local, item.remote)
    storage.finalize(message)
