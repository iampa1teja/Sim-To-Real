# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot 0.4.3 dataset recording helpers.

Frame and episode ownership deliberately stay in the caller's control loop. The
only asynchronous work used here is LeRobot's own image writer. In particular,
there is no per-dataset episode-processing thread, so a real and simulated
dataset cannot advance on independent timelines.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import validate_episode_buffer
from lerobot.datasets.video_utils import VideoEncodingManager


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

        self.dataset_features = features or self._make_sim_features(self.cameras)
        self.dataset: LeRobotDataset | None = None
        self._video_manager: VideoEncodingManager | None = None
        self._closed = False
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
            if key in expected
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
            self.dataset = LeRobotDataset(self.repo_id, root=self.dataset_root)
            self._check_existing_dataset()
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
            self.dataset = LeRobotDataset.create(
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
                batch_encoding_size=1,
            )
            print(f"[INFO]: New dataset initialized - {self.dataset.root}")

        self._video_manager = VideoEncodingManager(self.dataset)
        self._video_manager.__enter__()

    @property
    def episode_index(self) -> int:
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        return self.dataset.meta.total_episodes

    @property
    def frame_count(self) -> int:
        if self.dataset is None or self.dataset.episode_buffer is None:
            return 0
        return int(self.dataset.episode_buffer["size"])

    @staticmethod
    def _as_numpy(value: Any, *, dtype=None) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        return array.astype(dtype, copy=False) if dtype is not None else array

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
        self.dataset.add_frame(frame)
        if auxiliary:
            for data_type, camera_frames in auxiliary.items():
                for camera_name, image in camera_frames.items():
                    self._aux_frames[data_type].setdefault(camera_name, []).append(image)

    def push_frame_to_buffer(
        self,
        action,
        observation,
        visual_buffers,
        depth_buffers,
        instance_id_seg_buffers,
    ) -> None:
        """Backward-compatible wrapper used by older workshop callers."""
        frame = self.make_sim_frame(action, observation, visual_buffers)
        auxiliary = self.make_auxiliary_frame(
            visual_buffers, depth_buffers, instance_id_seg_buffers
        )
        self.add_frame(frame, auxiliary)

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
        self.dataset.save_episode(parallel_encoding=False)
        self._clear_auxiliary_frames()
        print(
            f"[INFO]: Saved episode {episode_index} to {self.dataset_root} "
            f"({self.dataset.meta.total_episodes} total)."
        )
        return episode_index

    def clear_episode_buffer(self) -> None:
        if self.dataset is not None and self.dataset.episode_buffer is not None:
            self.dataset.clear_episode_buffer()
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

    def finalize(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.dataset is None:
            return

        # An incomplete episode is never silently committed at shutdown.
        if self.frame_count:
            print(
                f"[WARNING]: Discarding {self.frame_count} unsaved frames from "
                f"{self.dataset_root} during shutdown."
            )
            self.clear_episode_buffer()

        try:
            if self._video_manager is not None:
                self._video_manager.__exit__(None, None, None)
            else:
                self.dataset.finalize()
        finally:
            self.dataset.stop_image_writer()


class SynchronizedLeRobotRecorders:
    """Coordinate one or more datasets from one master episode timeline."""

    def __init__(self, recorders: dict[str, LeRobotRecorder]):
        if not recorders:
            raise ValueError("At least one recorder is required.")
        self.recorders = recorders

    def init_datasets(self) -> None:
        for recorder in self.recorders.values():
            recorder.init_dataset()
        self._assert_aligned("initialization")

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
        for recorder in self.recorders.values():
            recorder.clear_episode_buffer()
        if any(counts.values()):
            print(f"[INFO]: Discarded synchronized episode buffers: {counts}")

    def finalize(self) -> None:
        errors = []
        for name, recorder in self.recorders.items():
            try:
                recorder.finalize()
            except Exception as exc:
                errors.append((name, exc))
        if errors:
            details = "; ".join(f"{name}: {exc}" for name, exc in errors)
            raise RuntimeError(f"Dataset finalization failed: {details}")
