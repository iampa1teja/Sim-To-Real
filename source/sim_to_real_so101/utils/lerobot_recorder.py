# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot 0.4.3 dataset recording helpers.

Frame and episode ownership deliberately stay in the caller's control loop. The
episode timeline stays synchronous. A group worker can encode saved PNGs in a
child process; only publication takes the dataset recording lock.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES, validate_episode_buffer

from .recording_dataset import EncodingCancelled, RecordingDataset


class LeRobotRecorder:
    """Synchronous owner of one official :class:`LeRobotDataset`.

    ``features`` is supplied for a physical follower and is built from the
    installed LeRobot processors. When omitted, the workshop's existing
    six-joint simulation schema is retained.
    """

    SO101_ACTION_NAMES = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    def __init__(
        self,
        task_name: str,
        repo_id: str,
        dataset_root: str,
        fps: int,
        device: str,
        cameras: dict | None = None,
        save_mp4: bool = False,
        depth: bool = False,
        instance_id_seg: bool = False,
        *,
        features: dict | None = None,
        robot_type: str = "so101_follower",
        image_writer_threads_per_camera: int = 4,
        batch_encoding_size: int = 1,
    ):
        self.task_name = task_name
        self.repo_id = repo_id
        self.dataset_root = Path(dataset_root)
        self.fps = fps
        self.dt = 1 / fps
        self.device = device
        self.cameras = cameras or {}
        self.robot_type = robot_type
        self.save_mp4 = save_mp4
        self.depth = depth
        self.instance_id_seg = instance_id_seg
        self.image_writer_threads_per_camera = image_writer_threads_per_camera
        # >1 defers video encoding: saved episodes keep their PNG frames on disk
        # and are encoded in batches, with the remainder encoded in finalize().
        self.batch_encoding_size = batch_encoding_size

        self.dataset_features = features or self._make_sim_features(self.cameras)
        self.dataset: LeRobotDataset | None = None
        self._closed = False
        self._save_failed = False
        self._frame_count = 0
        self._aux_frames: dict[str, dict[str, list[np.ndarray]]] = {
            "rgb": {},
            "depth": {},
            "instance_id_seg": {},
        }

    @classmethod
    def _make_sim_features(cls, cameras: dict) -> dict:
        observation_features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": cls.SO101_ACTION_NAMES.copy(),
            }
        }
        for camera_name, camera in cameras.items():
            observation_features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": (camera["height"], camera["width"], 3),
                "names": ["height", "width", "channels"],
            }

        return {
            **observation_features,
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": cls.SO101_ACTION_NAMES.copy(),
            },
        }

    @staticmethod
    def _feature_signature(feature: dict) -> tuple:
        return (
            feature.get("dtype"),
            tuple(feature.get("shape", ())),
            tuple(feature.get("names") or ()),
        )

    def _check_existing_dataset(self) -> None:
        assert self.dataset is not None
        if self.dataset.fps != self.fps:
            raise ValueError(
                f"Dataset {self.dataset_root} uses {self.dataset.fps} FPS; "
                f"the control loop uses {self.fps} FPS."
            )

        expected = {
            key: self._feature_signature(value)
            for key, value in self.dataset_features.items()
        }
        actual = {
            key: self._feature_signature(value)
            for key, value in self.dataset.features.items()
            if key not in DEFAULT_FEATURES
        }
        if actual != expected or not set(expected).issubset(self.dataset.features):
            raise ValueError(
                f"Dataset feature mismatch at {self.dataset_root}. "
                "Use a new dataset root for this recording configuration."
            )

    def init_dataset(self) -> None:
        """Create or resume the dataset using the installed LeRobot 0.4.3 API."""
        info_path = self.dataset_root / "meta" / "info.json"
        if info_path.is_file():
            self.dataset = RecordingDataset(
                self.repo_id, root=self.dataset_root, batch_encoding_size=self.batch_encoding_size
            )
            self._check_existing_dataset()
            lengths = self.episode_lengths()
            frame_episodes = np.asarray(self.dataset.hf_dataset["episode_index"])
            expected_episodes = np.repeat(np.arange(len(lengths)), lengths)
            if not np.array_equal(frame_episodes, expected_episodes):
                raise ValueError(f"Frame data and episode lengths disagree at {self.dataset_root}.")
            recovered = self.dataset.queue_unencoded_episodes()
            if recovered:
                print(f"[INFO]: Recovered {recovered} unencoded episode(s); they will be encoded with this session's.")
            camera_count = len(self.dataset.meta.camera_keys)
            if camera_count and self.image_writer_threads_per_camera:
                self.dataset.start_image_writer(
                    num_processes=0,
                    num_threads=self.image_writer_threads_per_camera * camera_count,
                )
            print(f"[INFO]: Existing dataset initialized - {self.dataset.root}")
        else:
            if self.dataset_root.exists() and any(self.dataset_root.iterdir()):
                raise ValueError(
                    f"Dataset root exists but is not a LeRobot dataset: {self.dataset_root}"
                )
            if self.dataset_root.exists():
                self.dataset_root.rmdir()

            camera_count = sum(
                feature.get("dtype") in ("image", "video")
                for feature in self.dataset_features.values()
            )
            self.dataset = RecordingDataset.create(
                self.repo_id,
                fps=self.fps,
                features=self.dataset_features,
                root=self.dataset_root,
                robot_type=self.robot_type,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=(
                    self.image_writer_threads_per_camera * camera_count
                    if camera_count
                    else 0
                ),
                batch_encoding_size=self.batch_encoding_size,
            )
            print(f"[INFO]: New dataset initialized - {self.dataset.root}")

    def episode_lengths(self) -> list[int]:
        """Return committed episode lengths in index order for resume checks."""
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        episodes = self.dataset.meta.episodes
        if episodes is None:
            if self.episode_index:
                raise ValueError(f"Missing episode metadata at {self.dataset_root}.")
            return []
        rows = list(episodes)
        if [row["episode_index"] for row in rows] != list(range(self.episode_index)):
            raise ValueError(f"Incomplete episode metadata at {self.dataset_root}.")
        return [int(row["length"]) for row in rows]

    @property
    def episode_index(self) -> int:
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        return self.dataset.meta.total_episodes

    @property
    def pending_video_episodes(self) -> int:
        """Saved episodes whose videos have not been encoded yet."""
        if self.dataset is None:
            return 0
        return len(self.dataset.pending_video_indices)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @staticmethod
    def _as_numpy(value: Any, *, dtype=None) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        # Async writers must own their pixels even when simulation runs on CPU.
        return array.astype(dtype, copy=True) if dtype is not None else array.copy()

    def make_sim_frame(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        visual_buffers: dict[str, torch.Tensor],
    ) -> dict:
        frame = {
            "action": self._as_numpy(action, dtype=np.float32),
            "observation.state": self._as_numpy(observation, dtype=np.float32),
            "task": self.task_name,
        }
        for camera_name in self.cameras:
            frame[f"observation.images.{camera_name}"] = self._as_numpy(
                visual_buffers[camera_name], dtype=np.uint8
            )
        return frame

    def make_auxiliary_frame(
        self,
        visual_buffers: dict[str, torch.Tensor],
        depth_buffers: dict[str, torch.Tensor],
        instance_id_seg_buffers: dict[str, torch.Tensor],
    ) -> dict | None:
        if not self.save_mp4:
            return None

        auxiliary: dict[str, dict[str, np.ndarray]] = {"rgb": {}}
        for camera_name in self.cameras:
            auxiliary["rgb"][camera_name] = self._as_numpy(
                visual_buffers[camera_name], dtype=np.uint8
            )
        if self.depth:
            auxiliary["depth"] = {
                name: self._as_numpy(depth_buffers[name], dtype=np.float32)
                for name in self.cameras
            }
        if self.instance_id_seg:
            auxiliary["instance_id_seg"] = {
                name: self._as_numpy(instance_id_seg_buffers[name], dtype=np.uint8)
                for name in self.cameras
            }
        return auxiliary

    def add_frame(self, frame: dict, auxiliary: dict | None = None) -> None:
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        if self._closed or self._save_failed:
            raise RuntimeError("Cannot append after shutdown or a failed episode save.")
        self.dataset.add_frame(frame)
        self._frame_count += 1
        if auxiliary:
            for data_type, camera_frames in auxiliary.items():
                for camera_name, image in camera_frames.items():
                    self._aux_frames[data_type].setdefault(camera_name, []).append(image)

    def validate_episode(self) -> None:
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        validate_episode_buffer(
            self.dataset.episode_buffer,
            self.dataset.meta.total_episodes,
            self.dataset.features,
        )

    def save_episode(self) -> int:
        """Synchronously commit the current episode and return its index."""
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        self.validate_episode()
        episode_index = self.episode_index

        # Optional exports are written before committing the official episode.
        # A failure leaves the official buffer cancellable by the coordinator.
        if self.save_mp4:
            self._save_auxiliary_videos(episode_index)

        # False keeps LeRobot's multi-camera encoding in this process.
        try:
            self.dataset.save_episode(parallel_encoding=False)
        except BaseException:
            # LeRobot mutates the buffer and may have committed metadata already.
            # Preserve its images even if interrupted before save_episode returns.
            self._save_failed = True
            raise
        self._frame_count = 0
        self._clear_auxiliary_frames()
        print(
            f"[INFO]: Saved episode {episode_index} to {self.dataset_root} "
            f"({self.dataset.meta.total_episodes} total)."
        )
        return episode_index

    def clear_episode_buffer(self) -> None:
        if self.dataset is not None and self.dataset.episode_buffer is not None:
            self.dataset.clear_episode_buffer(delete_images=not self._save_failed)
        self._frame_count = 0
        self._clear_auxiliary_frames()

    def _clear_auxiliary_frames(self) -> None:
        self._aux_frames = {"rgb": {}, "depth": {}, "instance_id_seg": {}}

    def _save_auxiliary_videos(self, episode_index: int) -> None:
        for camera_name, frames in self._aux_frames["rgb"].items():
            self._save_video(np.stack(frames), camera_name, "rgb", episode_index)
        for camera_name, frames in self._aux_frames["depth"].items():
            self.save_depth_video(np.stack(frames), camera_name, episode_index)
        for camera_name, frames in self._aux_frames["instance_id_seg"].items():
            self._save_video(
                np.stack(frames), camera_name, "instance_id_segmentation", episode_index
            )

    def _save_video(
        self,
        frames_rgb: np.ndarray,
        camera_name: str,
        data_type: str,
        episode_index: int,
    ) -> None:
        filename = f"{camera_name}_{data_type}_{episode_index:03d}.mp4"
        filepath = (
            self.dataset_root
            / "mp4"
            / self.repo_id.split("/")[-1]
            / camera_name
            / filename
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)

        frames_rgb = np.ascontiguousarray(frames_rgb, dtype=np.uint8)
        _, height, width, _ = frames_rgb.shape
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            os.fspath(filepath),
        ]
        result = subprocess.run(
            command,
            input=frames_rgb.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"ffmpeg failed for {filepath}: {result.stderr.decode(errors='replace')}"
            )
        print(f"[INFO]: Saved {data_type} video to {filepath}")

    def save_depth_video(
        self, frames: np.ndarray, camera_name: str, episode_index: int
    ) -> None:
        if frames.shape[-1] == 1:
            frames = frames.squeeze(-1)
        valid_mask = np.isfinite(frames)
        if valid_mask.any():
            min_val = np.percentile(frames[valid_mask], 1)
            max_val = np.percentile(frames[valid_mask], 99)
        else:
            min_val, max_val = 0.0, 1.0
        if max_val - min_val < 1e-6:
            max_val = min_val + 1.0
        normalized = np.clip((frames - min_val) / (max_val - min_val), 0, 1)
        gray = (normalized * 255).astype(np.uint8)
        self._save_video(
            np.stack([gray] * 3, axis=-1), camera_name, "depth", episode_index
        )

    def finalize(self, encode: bool = True) -> None:
        if self._closed:
            return
        if self.dataset is None:
            return
        try:
            try:
                if self.frame_count:
                    print(f"[WARNING]: Discarding {self.frame_count} unsaved frames at {self.dataset_root}.")
                    self.clear_episode_buffer()
            finally:
                pending = self.pending_video_episodes
                if encode and pending:
                    self.dataset._batch_save_episode_video(0, self.episode_index)
        finally:
            try:
                self.dataset.finalize()
            finally:
                try:
                    self.dataset.stop_image_writer()
                finally:
                    self._closed = True


class SynchronizedLeRobotRecorders:
    """Coordinate one or more datasets from one master episode timeline."""

    def __init__(self, recorders: dict[str, LeRobotRecorder]):
        if not recorders:
            raise ValueError("At least one recorder is required.")
        self.recorders = recorders
        self._work_lock = threading.RLock()
        self._worker = None
        self._stop_encoding = threading.Event()
        self._closing = False
        self._background = False
        self._encoding = False
        self._encode_error = ""
        self._encoded_count = 0
        self._encode_total = 0

    def enable_background_encoding(self) -> None:
        """Opt in for pick_place_agent; lerobot_agent retains immediate saves."""
        self._background = True

    @property
    def pending_video_indices(self) -> list[int]:
        return sorted({index for recorder in self.recorders.values() if recorder.dataset is not None
                       for index in recorder.dataset.pending_video_indices})

    @property
    def encoding(self) -> bool:
        return self._encoding

    @property
    def encode_progress(self) -> str:
        if self._encode_error:
            return self._encode_error
        if self.encoding:
            return f"Encoding in background: {self._encoded_count}/{self._encode_total} episodes"
        pending = self.pending_video_episodes
        return (f"{pending} saved episode(s) not yet encoded" if pending
                else "All saved episodes are encoded")

    def start_encoding(self) -> bool:
        with self._work_lock:
            if self._closing or self._encoding or not self.pending_video_episodes:
                return False
            self._stop_encoding.clear()
            self._start_encoder()
            return True

    def _start_encoder(self) -> None:
        self._encode_error = ""
        self._encoded_count = 0
        self._encode_total = self.pending_video_episodes
        self._encoding = True
        self._worker = threading.Thread(target=self._encode_pending, name="recording-encoder", daemon=True)
        self._worker.start()

    def _encode_pending(self) -> None:
        index = None
        try:
            while not self._stop_encoding.is_set():
                with self._work_lock:
                    pending = self.pending_video_indices
                    if not pending:
                        # Serialized with save_episode, including the transition
                        # to idle, so a save cannot fall through a drain race.
                        self._encoding = False
                        return
                    index = pending[0]
                    self._encode_total = self._encoded_count + len(pending)
                for recorder in self.recorders.values():
                    if recorder.dataset is not None and index in recorder.dataset.pending_video_indices:
                        recorder.dataset.encode_episode(index, self._stop_encoding)
                self._encoded_count += 1
        except EncodingCancelled:
            pass
        except BaseException as exc:
            reason = str(exc).strip().splitlines()[-1][:160] if str(exc).strip() else type(exc).__name__
            self._encode_error = f"Encoding failed for episode {index}: {reason} — press Encode videos to retry"
            print(f"[ERROR]: {self._encode_error}")
        finally:
            with self._work_lock:
                if self._worker is threading.current_thread():
                    self._encoding = False

    def init_datasets(self) -> None:
        for recorder in self.recorders.values():
            recorder.init_dataset()
            if self._background:
                recorder.dataset.batch_encoding_size = sys.maxsize
        self._assert_aligned("initialization")
        lengths = {name: recorder.episode_lengths() for name, recorder in self.recorders.items()}
        if len({tuple(value) for value in lengths.values()}) != 1:
            raise ValueError(f"Existing datasets have different per-episode frame counts: {lengths}")

        if self._background and self.pending_video_episodes:
            self.start_encoding()

    @property
    def episode_index(self) -> int:
        self._assert_aligned("episode lookup")
        return next(iter(self.recorders.values())).episode_index

    @property
    def frame_count(self) -> int:
        counts = {name: recorder.frame_count for name, recorder in self.recorders.items()}
        if len(set(counts.values())) != 1:
            raise RuntimeError(f"Recorder frame counts diverged: {counts}")
        return next(iter(counts.values()))

    @property
    def pending_video_episodes(self) -> int:
        return len(self.pending_video_indices)

    def _assert_aligned(self, operation: str) -> None:
        episodes = {
            name: recorder.episode_index for name, recorder in self.recorders.items()
        }
        if len(set(episodes.values())) != 1:
            raise RuntimeError(
                f"Cannot continue synchronized recording during {operation}; "
                f"dataset episode counts differ: {episodes}. Use matching/new roots."
            )

    def add_frames(
        self,
        frames: dict[str, dict],
        auxiliary: dict[str, dict | None] | None = None,
    ) -> None:
        if set(frames) != set(self.recorders):
            raise ValueError(
                f"Frame set {set(frames)} does not match recorders {set(self.recorders)}."
            )
        self._assert_aligned("frame append")
        _ = self.frame_count
        auxiliary = auxiliary or {}
        try:
            for name, recorder in self.recorders.items():
                recorder.add_frame(frames[name], auxiliary.get(name))
            _ = self.frame_count
        except Exception as exc:
            self.cancel_episode()
            raise RuntimeError(
                "A synchronized frame write failed; all in-progress episode buffers "
                "were discarded."
            ) from exc

    def save_episode(self) -> int | None:
        with self._work_lock:
            result = self._save_episode()
            threshold = min(recorder.batch_encoding_size for recorder in self.recorders.values())
            if self._background and not self._encode_error and self.pending_video_episodes >= threshold:
                self.start_encoding()
            return result

    def _save_episode(self) -> int | None:
        self._assert_aligned("episode save")
        count = self.frame_count
        if count == 0:
            print("[WARNING]: Episode has no frames; nothing was saved.")
            return None

        for recorder in self.recorders.values():
            recorder.validate_episode()

        expected_index = self.episode_index
        saved: list[str] = []
        try:
            for name, recorder in self.recorders.items():
                actual_index = recorder.save_episode()
                if actual_index != expected_index:
                    raise RuntimeError(
                        f"Recorder {name} saved episode {actual_index}; expected {expected_index}."
                    )
                saved.append(name)
        except Exception as exc:
            for name, recorder in self.recorders.items():
                if name not in saved:
                    recorder.clear_episode_buffer()
            raise RuntimeError(
                f"Synchronized episode {expected_index} failed while saving. "
                f"Already committed datasets: {saved}. Recording stopped to prevent "
                "silent timeline divergence."
            ) from exc

        self._assert_aligned("post-save verification")
        print(
            f"[INFO]: Synchronized episode {expected_index} saved with {count} frames "
            f"to {list(self.recorders)}."
        )
        return expected_index

    def cancel_episode(self) -> None:
        counts = {name: recorder.frame_count for name, recorder in self.recorders.items()}
        errors = []
        for name, recorder in self.recorders.items():
            try:
                recorder.clear_episode_buffer()
            except BaseException as exc:
                errors.append((name, exc))
        if errors:
            raise RuntimeError(f"Episode cancellation failed: {errors}")
        if any(counts.values()):
            print(f"[INFO]: Discarded synchronized episode buffers: {counts}")

    def begin_shutdown(self, encode: bool) -> None:
        """Reject new encode requests and cancel promptly for Exit without encoding."""
        with self._work_lock:
            self._closing = True
            if not encode:
                self._stop_encoding.set()

    def finalize(self, encode: bool = True, progress=None) -> None:
        """Drain or cancel safely, then always close every recorder resource."""
        try:
            self.begin_shutdown(encode)
            with self._work_lock:
                if encode and not self._stop_encoding.is_set() and not self._encoding and self.pending_video_episodes:
                    self._start_encoder()
            while self._worker is not None and self._worker.is_alive():
                if progress is not None:
                    progress(self.pending_video_episodes)
                self._worker.join(timeout=0.1)
        except BaseException:
            # Signals during exit encoding cancel the child, never the commit.
            # Repeated signals must not skip image-writer/resource cleanup.
            self._stop_encoding.set()
        finally:
            while self._worker is not None and self._worker.is_alive():
                try:
                    self._worker.join(timeout=0.1)
                except BaseException:
                    self._stop_encoding.set()
            errors = []
            for name, recorder in self.recorders.items():
                try:
                    recorder.finalize(encode=False)
                except BaseException as exc:
                    errors.append((name, exc))
            pending = self.pending_video_episodes
            if pending:
                print(f"[INFO]: {pending} episode(s) left unencoded; they will be encoded next session.")
            if errors:
                details = "; ".join(f"{name}: {exc}" for name, exc in errors)
                raise RuntimeError(f"Dataset finalization failed: {details}")
