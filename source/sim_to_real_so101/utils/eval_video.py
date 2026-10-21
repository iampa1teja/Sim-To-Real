"""Optional camera videos with bounded background encoding and no dropped frames."""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
from queue import Full, Queue
import re
from threading import Thread

import numpy as np


class _VideoEncoder:
    """Own the PyAV containers on a single encoding thread."""

    def __init__(self, output_dir: Path, fps: float, rename_map: dict[str, str] | None, encoder: str):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video fps must be finite and positive")
        if encoder not in ("libx264", "h264_nvenc"):
            raise ValueError("Video encoder must be libx264 or h264_nvenc")
        import av

        self._av = av
        self.encoder = encoder
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.rate = Fraction(str(float(fps))).limit_denominator(100000)
        self.cameras = dict(rename_map or {"realsense_rgb": "room", "wrist_cam": "wrist"})
        self._filenames = {camera: re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip("._") or "camera"
                           for camera, name in self.cameras.items()}
        if len(set(self._filenames.values())) != len(self._filenames):
            raise ValueError("Video camera names must produce distinct filenames")
        self._episode = None
        self._writers = {}
        self._paths = {}

    def start_episode(self, index: int) -> None:
        if self._episode is not None:
            raise RuntimeError("Finish the current video episode before starting another")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Video episode index must be a nonnegative integer")
        self._episode = index
        self._paths = {}

    @staticmethod
    def _rgb(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        image = np.asarray(value)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError(f"Video camera needs RGB shape (H,W,3) or (1,H,W,3), got {image.shape}")
        if image.dtype != np.uint8:
            if not np.isfinite(image).all():
                raise ValueError("Video camera contains nonfinite pixels")
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image)

    def _open(self, camera: str, image: np.ndarray):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"episode_{self._episode:06d}_{self._filenames[camera]}.mp4"
        # Explicit output directories may be reused: preserve any existing recording.
        handle = path.open("xb")
        container = None
        try:
            container = self._av.open(handle, mode="w", format="mp4")
            stream = container.add_stream(self.encoder, rate=self.rate)
            stream.height, stream.width = image.shape[:2]
            stream.pix_fmt = "yuv420p" if stream.width % 2 == stream.height % 2 == 0 else "yuv444p"
            if self.encoder == "h264_nvenc":
                stream.options = {"preset": "p1", "tune": "ll", "rc": "vbr", "cq": "23", "bf": "0"}
            else:
                stream.options = {"preset": "veryfast", "crf": "23", "tune": "zerolatency"}
                stream.codec_context.thread_count = 2
        except BaseException:
            try:
                if container is not None:
                    container.close()
            finally:
                handle.close()
            raise
        self._paths[self.cameras[camera]] = str(path)
        self._writers[camera] = (container, stream, handle)
        return self._writers[camera]

    def write(self, visualobs: dict) -> None:
        if self._episode is None:
            raise RuntimeError("Start a video episode before writing frames")
        missing = [f"rgb_{camera}" for camera in self.cameras if f"rgb_{camera}" not in visualobs]
        if missing:
            raise KeyError(f"Missing video camera observations: {missing}")
        for camera in self.cameras:
            image = self._rgb(visualobs[f"rgb_{camera}"])
            container, stream, _handle = self._writers.get(camera) or self._open(camera, image)
            if image.shape[:2] != (stream.height, stream.width):
                raise ValueError(f"Video camera {camera} changed image dimensions during an episode")
            frame = self._av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

    def finish_episode(self) -> dict[str, str]:
        errors = []
        for container, stream, handle in self._writers.values():
            try:
                for packet in stream.encode():
                    container.mux(packet)
            except BaseException as error:
                errors.append(error)
            finally:
                try:
                    container.close()
                except BaseException as error:
                    errors.append(error)
                finally:
                    try:
                        handle.close()
                    except BaseException as error:
                        errors.append(error)
        self._writers.clear()
        self._episode = None
        if errors:
            raise errors[0]
        return self._paths.copy()


class EpisodeVideoRecorder:
    """Snapshot camera observations before stepping, then encode them in order.

    At most ``queue_size`` frame sets wait for the worker. A full queue slows the
    producer instead of dropping frames. ``finish_episode`` drains and finalizes
    both recordings before returning their paths, including after an exception.
    """

    def __init__(self, output_dir: Path, fps: float, rename_map: dict[str, str] | None = None,
                 *, encoder: str = "libx264", queue_size: int = 8):
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError("Video queue_size must be a positive integer")
        self._encoder = _VideoEncoder(output_dir, fps, rename_map, encoder)
        self.encoder = encoder
        self._queue_size = queue_size
        self._queue = None
        self._worker = None
        self._episode = None
        self._error = None
        self._error_delivered = False

    def start_episode(self, index: int) -> None:
        if self._episode is not None:
            raise RuntimeError("Finish the current video episode before starting another")
        self._encoder.start_episode(index)
        self._episode = index
        self._queue = Queue(maxsize=self._queue_size)
        self._error = None
        self._error_delivered = False

    def _raise_error(self) -> None:
        if self._error is not None:
            self._error_delivered = True
            raise self._error

    def _encode(self) -> None:
        try:
            while True:
                frames = self._queue.get()
                if frames is None:
                    break
                if self._error is None:
                    try:
                        self._encoder.write(frames)
                    except BaseException as error:
                        self._error = error
                # On failure, drain the bounded queue so shutdown cannot deadlock.
        except BaseException as error:
            self._error = self._error or error
        finally:
            try:
                self._encoder.finish_episode()
            except BaseException as error:
                self._error = self._error or error

    def _put(self, frames) -> None:
        while True:
            if frames is not None:
                self._raise_error()
            if not self._worker.is_alive():
                if frames is None:
                    return
                self._raise_error()
                raise RuntimeError("Video encoding worker exited unexpectedly")
            try:
                self._queue.put(frames, timeout=0.1)
                return
            except Full:
                pass

    def write(self, visualobs: dict) -> None:
        if self._episode is None:
            raise RuntimeError("Start a video episode before writing frames")
        self._raise_error()
        missing = [f"rgb_{camera}" for camera in self._encoder.cameras if f"rgb_{camera}" not in visualobs]
        if missing:
            raise KeyError(f"Missing video camera observations: {missing}")
        if self._worker is None:
            for filename in self._encoder._filenames.values():
                path = self._encoder.output_dir / f"episode_{self._episode:06d}_{filename}.mp4"
                if path.exists():
                    raise FileExistsError(f"Recording already exists: {path}")
        # Isaac reuses camera buffers. Copy on the producer before the next env.step;
        # a contiguous NumPy view alone would let queued frames change underneath us.
        frames = {f"rgb_{camera}": self._encoder._rgb(visualobs[f"rgb_{camera}"]).copy()
                  for camera in self._encoder.cameras}
        if self._worker is None:
            self._worker = Thread(target=self._encode, name="so101-video-encoder", daemon=True)
            self._worker.start()
        self._put(frames)
        self._raise_error()

    def finish_episode(self) -> dict[str, str]:
        if self._worker is None:
            try:
                return self._encoder.finish_episode()
            finally:
                self._episode = None
        self._put(None)
        self._worker.join()
        self._worker = None
        self._queue = None
        self._episode = None
        if not self._error_delivered:
            self._raise_error()
        return self._encoder._paths.copy()

    def close(self) -> None:
        self.finish_episode()
