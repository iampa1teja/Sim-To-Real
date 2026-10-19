"""Optional, streaming camera videos for evaluation attempts (no frame accumulation)."""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
import re

import numpy as np


class EpisodeVideoRecorder:
    """Write one MP4 per camera/attempt; instantiate only when recording is requested.

    ``write`` expects raw RGB camera observations in 0..255, optionally batched
    with one environment. Call it before stepping the environment so an automatic
    terminal reset never contributes frames from the following attempt.
    """

    def __init__(self, output_dir: Path, fps: float, rename_map: dict[str, str] | None = None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video fps must be finite and positive")
        import av

        self._av = av
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
            stream = container.add_stream("libx264", rate=self.rate)
            stream.height, stream.width = image.shape[:2]
            stream.pix_fmt = "yuv420p" if stream.width % 2 == stream.height % 2 == 0 else "yuv444p"
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

    def close(self) -> None:
        self.finish_episode()
