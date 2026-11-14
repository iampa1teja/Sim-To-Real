"""Atomic file replacement shared by reports, sidecars and journals."""
import os
from pathlib import Path
import tempfile


def atomic_write_text(path, text, *, fsync=False, encoding=None, newline=None):
    """Write ``text`` to a temporary file beside ``path``, then replace ``path`` with it.

    ``encoding``/``newline`` are passed to ``open``; ``fsync`` forces the bytes to disk
    before the replace. The parent folder must already exist.
    """
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline=newline, dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            if fsync:
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
