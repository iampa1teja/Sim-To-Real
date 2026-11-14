"""Each caller moved onto atomic_write_text writes the same bytes and file mode as its old inline code."""
import json
import os
from pathlib import Path
import stat
import tempfile

import numpy as np
import pytest

from utils.io import atomic_write_text

PAYLOAD = {"episodes": [{"x": 0.1, "tiny": 1e-7, "big": 1e20, "np": np.float64(2.5), "n": 3, "ok": True}],
           "Ünïcødé ✓": ["line\nbreak", None, -0.0, [1, [2, {"z": "é"}]]], "b": False}


def _old_resume_write_report(path, report):  # core/eval/resume.py write_report
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


def _old_task_meta_write(path, payload):  # utils/datasets/task_meta.py _write
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _old_rollout_write_json(path, data):  # sim/datagen/rollout.py write_json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _old_rerecording_journal(journal, entries):  # sim/recording/rerecording.py journal write
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=journal.parent, prefix=journal.name,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(entries, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, journal)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _new_rerecording_journal(journal, entries):  # the replacement line, verbatim
    atomic_write_text(journal, json.dumps(entries), fsync=True)


def _callers():
    from core.eval.resume import write_report
    from sim.datagen.rollout import write_json
    from utils.datasets.task_meta import _write

    return [(_old_resume_write_report, write_report), (_old_task_meta_write, _write),
            (_old_rollout_write_json, write_json), (_old_rerecording_journal, _new_rerecording_journal)]


@pytest.mark.parametrize("index", range(4))
def test_migrated_writers_keep_bytes_and_mode(tmp_path, index):
    old, new = _callers()[index]
    payloads = [PAYLOAD, [{"root": "/data/ü", "stage": "x"}], {}]
    for number, payload in enumerate(payloads):
        (tmp_path / "old").mkdir(exist_ok=True)
        (tmp_path / "new").mkdir(exist_ok=True)
        before, after = tmp_path / "old" / f"{number}.json", tmp_path / "new" / f"{number}.json"
        old(before, payload)
        new(after, payload)
        assert after.read_bytes() == before.read_bytes()
        assert stat.S_IMODE(after.stat().st_mode) == stat.S_IMODE(before.stat().st_mode)
        new(after, payload)  # replacing an existing file
        assert after.read_bytes() == before.read_bytes()
    assert sorted(p.name for p in (tmp_path / "new").iterdir()) == [f"{n}.json" for n in range(len(payloads))]


def test_failed_serialisation_leaves_target_and_no_temporaries(tmp_path):
    from core.eval.resume import write_report

    target = tmp_path / "report.json"
    write_report(target, {"ok": 1})
    with pytest.raises(ValueError):
        write_report(target, {"bad": float("nan")})
    assert json.loads(target.read_text()) == {"ok": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["report.json"]
