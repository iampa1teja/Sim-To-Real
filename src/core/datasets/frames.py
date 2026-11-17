"""Persistent lite dataset writer and parquet frame reader."""

from pathlib import Path as _Path
import json as _json
import os as _os
import re as _re
import tempfile as _tempfile

import numpy as _np
import pyarrow as _pa
import pyarrow.parquet as _pq

from core.datasets._audit_record import _atomic_text
from core.datasets.streaming import _episode_tables
from core.datasets.task_meta import LABEL_FEATURES as _LABEL_FEATURES
from core.generate.fake_env import SimulatedCrash as _SimulatedCrash
from core.generate.protocols import FrameLabels as _FrameLabels


def read_frames(root):
    return {index: {name: _np.asarray(table.column(name).to_pylist(), dtype=dtype).reshape(table.num_rows, *shape)
                    for name, dtype, shape in (
                        ("action", _np.float32, (-1,)), ("observation.state", _np.float32, (-1,)),
                        ("label.stage", _np.int64, ()), ("label.imitate", _np.int64, ()),
                        ("label.ee_pose", _np.float32, (7,)), ("frame_index", _np.int64, ()))}
            for index, table in sorted(_episode_tables(root).items())}


class LiteDatasetWriter:
    def __init__(self, root, *, fps, joint_names):
        self.root, self._fps, self._names = _Path(root), fps, tuple(joint_names)
        self._buffer = []
        vector = {"dtype": "float32", "shape": [len(self._names)], "names": list(self._names)}
        features = {"action": dict(vector), "observation.state": dict(vector)}
        features.update({key: {**value, "shape": list(value["shape"]), "names": list(value["names"])}
                         for key, value in _LABEL_FEATURES.items()})
        expected = dict(codebase_version="v3.0-lite", fps=fps, total_episodes=0, features=features)
        self._info_path = self.root / "meta" / "info.json"
        if self._info_path.exists():
            self._info = _json.loads(self._info_path.read_text())
            if any(self._info.get(key) != expected[key] for key in ("codebase_version", "fps", "features")):
                raise ValueError("Dataset layout, fps or features differ")
        else:
            self._info = expected
            _atomic_text(self._info_path, _json.dumps(self._info, indent=2, sort_keys=True))
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            match = _re.fullmatch(r"file-([0-9]+)\.parquet", path.name)
            if path.name.endswith(".tmp") or (match and self.root / "data" in path.parents
                                             and int(match[1]) >= self.episode_count()):
                path.unlink()
        self._lengths = [_pq.read_metadata(self._data_path(i)).num_rows for i in range(self.episode_count())]

    def _data_path(self, index):
        return self.root / "data" / "chunk-000" / f"file-{index:06d}.parquet"

    def episode_count(self):
        return self._info["total_episodes"]

    def episode_lengths(self):
        return list(self._lengths)

    def add_frame(self, action_raw, state_raw, images, task, labels):
        if not isinstance(labels, _FrameLabels):
            raise TypeError("labels must be FrameLabels")
        action, state = _np.asarray(action_raw), _np.asarray(state_raw)
        if action.shape != (len(self._names),) or state.shape != action.shape:
            raise ValueError("Incorrect action or state shape")
        for image in images.values():
            if not isinstance(image, _np.ndarray) or image.dtype != _np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Images must be uint8 H x W x 3 arrays")
        self._buffer.append((_np.array(action, _np.float32, copy=True), _np.array(state, _np.float32, copy=True), task, labels))

    def discard_episode(self):
        self._buffer.clear()

    def save_episode(self):
        if not self._buffer:
            raise RuntimeError("Cannot save an empty episode")
        index, n = self.episode_count(), len(self._buffer)
        start = sum(self._lengths)
        table = _pa.table({
            "action": _pa.array([a.tolist() for a, _, _, _ in self._buffer], _pa.list_(_pa.float32())),
            "observation.state": _pa.array([s.tolist() for _, s, _, _ in self._buffer], _pa.list_(_pa.float32())),
            "episode_index": _pa.array([index]*n, _pa.int64()), "frame_index": _pa.array(range(n), _pa.int64()),
            "index": _pa.array(range(start, start+n), _pa.int64()),
            "timestamp": _pa.array(_np.arange(n, dtype=_np.float32)/_np.float32(self._fps), _pa.float32()),
            "task": _pa.array([task for _, _, task, _ in self._buffer], _pa.string()),
            "label.stage": _pa.array([[lab.stage] for _, _, _, lab in self._buffer], _pa.list_(_pa.int64())),
            "label.imitate": _pa.array([[int(lab.imitate)] for _, _, _, lab in self._buffer], _pa.list_(_pa.int64())),
            "label.ee_pose": _pa.array([lab.ee_pose_wxyz.tolist() for _, _, _, lab in self._buffer], _pa.list_(_pa.float32())),
        })
        path = self._data_path(index)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = _tempfile.mkstemp(dir=path.parent, prefix=".episode.", suffix=".tmp")
        _os.close(fd)
        try:
            _pq.write_table(table, temporary)
            with open(temporary, "rb") as stream:
                _os.fsync(stream.fileno())
            _os.replace(temporary, path)
        finally:
            _Path(temporary).unlink(missing_ok=True)
        info = dict(self._info, total_episodes=index+1)
        _atomic_text(self._info_path, _json.dumps(info, indent=2, sort_keys=True))
        self._info = info
        self._lengths.append(n)
        self._buffer.clear()
        return index

    def finalize(self):
        pass


class MemoryWriter(LiteDatasetWriter):
    def __init__(self, root, *, fps, joint_names, crash=None, crash_on_episode=None):
        if crash not in (None, "before_save", "after_save"):
            raise ValueError("Unknown writer crash mode")
        super().__init__(root, fps=fps, joint_names=joint_names)
        self._crash, self._crash_episode, self._fired = crash, crash_on_episode, False

    def save_episode(self):
        fire = (not self._fired and self._crash is not None
                and (self._crash_episode is None or self.episode_count() == self._crash_episode))
        if fire:
            self._fired = True
            if self._crash == "before_save":
                raise _SimulatedCrash("Interrupted before episode commit")
        index = super().save_episode()
        if fire:
            raise _SimulatedCrash("Interrupted after episode commit")
        return index
