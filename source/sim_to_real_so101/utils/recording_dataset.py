"""Batch encoding compatibility for LeRobot commit e670ac5daf9b76.

That revision's batch encoder reads unflushed episode metadata and uses data
file indices for metadata files. Keep its recording API, but encode one video
file per episode and publish metadata before removing source images.
"""

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import subprocess
import sys
import threading

import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_EPISODES_PATH, load_episodes, write_info


# Exec a fresh interpreter, never fork Isaac/CUDA or import the control-loop entry
# point. Only paths and FPS cross this boundary. Codec arguments are deliberately
# omitted: these are encode_video_frames defaults at e670ac5daf9b76.
_ENCODER = """
import ctypes, json, os, signal, sys
from pathlib import Path
parent = int(sys.argv[2])
if sys.platform == 'linux':
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)
    if os.getppid() != parent:
        sys.exit(1)
try:
    os.nice(10)
except (AttributeError, OSError):
    pass
from lerobot.datasets.video_utils import encode_video_frames, get_video_info
job = json.loads(Path(sys.argv[1]).read_text())
for camera in job['cameras']:
    encode_video_frames(camera['images'], camera['temporary'], job['fps'], overwrite=True)
    camera['info'] = get_video_info(camera['temporary'])
Path(sys.argv[1]).write_text(json.dumps(job))
"""


class EncodingCancelled(Exception):
    """Encoding stopped before publication; source PNGs are still recoverable."""


class RecordingDataset(LeRobotDataset):
    """Use the pinned dataset writer with durable deferred video metadata."""

    def __new__(cls, *args, **kwargs):
        # LeRobot.create uses __new__ directly and does not call __init__.
        obj = super().__new__(cls)
        obj.recording_lock = threading.RLock()
        return obj

    @property
    def pending_video_indices(self) -> list[int]:
        # meta.episodes is an immutable snapshot, replaced only after publication.
        # Reading it does not block the 30 Hz control loop on a metadata commit.
        episodes = self.meta.episodes
        if episodes is None:
            return []
        return [int(row["episode_index"]) for row in episodes
                if any(pd.isna(row.get(f"videos/{key}/chunk_index")) for key in self.meta.video_keys)]

    def finalize(self):
        with self.recording_lock:
            super().finalize()

    def clear_episode_buffer(self, delete_images=True):
        with self.recording_lock:
            super().clear_episode_buffer(delete_images=delete_images)

    def _check_cached_episodes_sufficient(self) -> bool:
        """Allow locally committed PNG-only episodes to reach recovery validation.

        The pinned parent checks video paths here before its constructor returns.
        Missing video metadata is intentional during deferred recording; data
        coverage and already published video files must still be checked.
        """
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False
        available = {int(index) for index in self.hf_dataset.unique("episode_index")}
        requested = set(range(self.meta.total_episodes)) if self.episodes is None else set(self.episodes)
        if not requested.issubset(available):
            return False
        for index in requested:
            row = self.meta.episodes[index]
            for key in self.meta.video_keys:
                if pd.isna(row.get(f"videos/{key}/chunk_index")):
                    continue
                if not (self.root / self.meta.get_video_file_path(index, key)).is_file():
                    raise ValueError(f"Published video is missing for episode {index}, {key} at {self.root}.")
        return True

    def save_episode(self, episode_data=None, parallel_encoding=True):
        """Close every saved episode before allowing any encoding to start."""
        with self.recording_lock:
            batch_size = self.batch_encoding_size
            # Suppress the parent's synchronous trigger while it owns mutable
            # buffers. Immediate encoding remains available to lerobot_agent.
            self.batch_encoding_size = sys.maxsize
            try:
                super().save_episode(episode_data=episode_data, parallel_encoding=parallel_encoding)
                self._close_episode_files()
            finally:
                self.batch_encoding_size = batch_size
        pending = self.pending_video_indices
        if pending and len(pending) >= batch_size:
            self._batch_save_episode_video(pending[0], self.num_episodes)

    def _close_episode_files(self) -> None:
        """Publish readable Parquet files without encoding or overwriting prior saves."""
        self.finalize()
        self.meta.episodes = load_episodes(self.root)
        self.latest_episode = None
        self.meta.latest_episode = None

    def queue_unencoded_episodes(self) -> int:
        """Recover a suffix of committed episodes whose video metadata is absent."""
        # A killed child or interrupted atomic write may leave staging files.
        for temporary in self.root.rglob(".encoding-*"):
            if temporary.is_dir():
                shutil.rmtree(temporary)
            else:
                temporary.unlink()
        episodes = self.meta.episodes
        if episodes is None:
            episodes = []
        missing = [
            int(row["episode_index"]) for row in episodes
            if any(pd.isna(row.get(f"videos/{key}/chunk_index")) for key in self.meta.video_keys)
        ]
        expected = list(range(self.num_episodes - len(missing), self.num_episodes))
        if missing != expected:
            raise ValueError(f"Unencoded episodes must be a contiguous suffix; found indices {missing}.")
        for index in missing:
            length = int(episodes[index]["length"])
            for key in self.meta.video_keys:
                directory = self._get_image_file_path(index, key, 0).parent
                count = len(list(directory.glob("*.png")))
                if count != length:
                    raise ValueError(
                        f"Cannot recover episode {index}, {key}: expected {length} PNGs, found {count} in {directory}."
                    )
        # Only uncommitted indices may be reused. Never remove committed PNGs.
        for key in self.meta.camera_keys:
            parent = self._get_image_file_path(self.num_episodes, key, 0).parent.parent
            for directory in parent.glob("episode-*"):
                suffix = directory.name.removeprefix("episode-")
                if directory.is_dir() and suffix.isdecimal() and int(suffix) >= self.num_episodes:
                    print(f"[INFO]: Removing orphan images for uncommitted episode {int(suffix)}: {directory}")
                    shutil.rmtree(directory)
        # A crash after publication but before PNG cleanup is also recoverable.
        for row in episodes:
            index = int(row["episode_index"])
            if index not in missing:
                for key in self.meta.video_keys:
                    directory = self._get_image_file_path(index, key, 0).parent
                    if directory.is_dir():
                        shutil.rmtree(directory)
        self.episodes_since_last_encoding = len(missing)
        return len(missing)

    def encode_episode(self, index: int, stop: threading.Event | None = None) -> None:
        """Prepare without dataset mutation, encode in a child, then publish."""
        stop = stop or threading.Event()
        with self.recording_lock:
            if index not in self.pending_video_indices:
                return
            episode = dict(self.meta.episodes[index])
            expected = int(episode["length"])
            chunk, file_index = divmod(index, self.meta.chunks_size)
            cameras = []
            for key in self.meta.video_keys:
                directory = self._get_image_file_path(index, key, 0).parent
                if (len(list(directory.glob("*.png"))) != expected or
                        any(not self._get_image_file_path(index, key, frame).is_file()
                            for frame in range(expected))):
                    raise ValueError(f"Episode {index}, {key}: missing or extra source images.")
                cameras.append({"key": key, "images": str(directory), "target": str(
                    self.root / self.meta.video_path.format(
                        video_key=key, chunk_index=chunk, file_index=file_index))})
            fps = self.fps
        with tempfile.TemporaryDirectory(prefix=".encoding-", dir=self.root) as temporary:
            staging = Path(temporary)
            for number, camera in enumerate(cameras):
                camera["temporary"] = str(staging / f"{number}.mp4")
            job_path = staging / "job.json"
            job_path.write_text(json.dumps({"fps": fps, "cameras": cameras}))
            with (staging / "encoder.log").open("w+") as log:
                process = subprocess.Popen(
                    [sys.executable, "-c", _ENCODER, str(job_path), str(os.getpid())],
                    stdout=log, stderr=log,
                )
                try:
                    while process.poll() is None:
                        if stop.wait(0.05):
                            raise EncodingCancelled()
                    if stop.is_set():
                        raise EncodingCancelled()
                    if process.returncode:
                        log.seek(0)
                        raise RuntimeError(log.read()[-2000:].strip() or f"Encoder exited {process.returncode}")
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
            cameras = json.loads(job_path.read_text())["cameras"]
            with self.recording_lock:
                # Stop is checked before the commit. Once begun, publication and
                # PNG deletion finish as one operation without cancellation.
                if stop.is_set():
                    raise EncodingCancelled()
                self._commit_episode_video(index, episode, cameras, chunk, file_index, staging)

    def _commit_episode_video(self, index, episode, cameras, chunk, file_index, staging):
        path = self.root / DEFAULT_EPISODES_PATH.format(
            chunk_index=episode["meta/episodes/chunk_index"],
            file_index=episode["meta/episodes/file_index"],
        )
        table = pd.read_parquet(path)
        mask = table["episode_index"] == index
        if int(mask.sum()) != 1:
            raise ValueError(f"Expected one metadata row for episode {index} in {path}.")
        info_changed = False
        info = copy.deepcopy(self.meta.info)
        for camera in cameras:
            target = Path(camera["target"])
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(camera["temporary"], target)
            key = camera["key"]
            for suffix, value in {
                "chunk_index": chunk, "file_index": file_index,
                "from_timestamp": 0.0, "to_timestamp": int(episode["length"]) / self.fps,
            }.items():
                table.loc[mask, f"videos/{key}/{suffix}"] = value
            if not self.meta.features[key].get("info"):
                info["features"][key]["info"] = camera["info"]
                info_changed = True
        # Publish info first so a published episode always has video info, even
        # if the process dies between the two atomic replacements.
        if info_changed:
            write_info(info, staging)
            os.replace(staging / "meta/info.json", self.root / "meta/info.json")
            self.meta.info = info
        for column in table.columns:
            if column.startswith("videos/") and column.endswith(("/chunk_index", "/file_index")):
                table[column] = table[column].astype("Int64")
        temporary_metadata = staging / "episodes.parquet"
        table.to_parquet(temporary_metadata, index=False)
        os.replace(temporary_metadata, path)
        self.meta.episodes = load_episodes(self.root)
        for camera in cameras:
            shutil.rmtree(camera["images"])

    def _batch_save_episode_video(self, start_episode: int, end_episode: int | None = None) -> None:
        end_episode = self.num_episodes if end_episode is None else end_episode
        for index in self.pending_video_indices:
            if start_episode <= index < end_episode:
                self.encode_episode(index)
