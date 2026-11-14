# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.feature_utils import validate_episode_buffer
from lerobot.datasets.io_utils import load_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import DEFAULT_FEATURES

from utils.arms.spec import ArmSpec, load_arm_spec


class RecordingDataset(LeRobotDataset):

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.recording_lock = threading.RLock()
        return obj

    @classmethod
    def open(
        cls,
        repo_id: str,
        root: str | Path,
        fps: int,
        features: dict,
        *,
        robot_type: str | None = None,
        use_videos: bool = True,
        image_writer_threads: int = 4,
        streaming_encoding: bool = True,
        depth_encoder=None,
    ) -> "RecordingDataset":
        root = Path(root)
        if (root / "meta" / "info.json").is_file():
            dataset = cls.resume(
                repo_id,
                root=root,
                image_writer_threads=image_writer_threads,
                streaming_encoding=streaming_encoding,
                depth_encoder=depth_encoder,
            )
            dataset.writer.cleanup_interrupted_episode(dataset.meta.total_episodes)
            return dataset

        if root.exists() and any(root.iterdir()):
            raise ValueError(f"{root} is not empty but has no meta/info.json; choose a new dataset root.")
        
        if root.exists(): root.rmdir()  

        return cls.create(
            repo_id,
            fps,
            features,
            root=root,
            robot_type=robot_type,
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
            streaming_encoding=streaming_encoding,
            depth_encoder=depth_encoder,
        )

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        with self.recording_lock:
            super().save_episode(episode_data=episode_data, parallel_encoding=parallel_encoding)
            self._commit_episode_files()

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        with self.recording_lock:
            super().clear_episode_buffer(delete_images=delete_images)

    def finalize(self) -> None:
        with self.recording_lock:
            super().finalize()

    def _commit_episode_files(self) -> None:
        self.writer.close_writer()
        self.meta._close_writer()
        self.meta.episodes = load_episodes(self.root)
        self.meta.latest_episode = None
        self.writer._latest_episode = None


def split_visual_obs(visual_obs: dict, cameras) -> tuple[dict, dict, dict]:
    rgb, depth, instance_id_seg = {}, {}, {}
    for camera in cameras:
        rgb[camera] = visual_obs[f"rgb_{camera}"][0]
        if f"depth_{camera}" in visual_obs:
            depth[camera] = visual_obs[f"depth_{camera}"][0]
        if f"instance_id_seg_{camera}" in visual_obs:
            instance_id_seg[camera] = visual_obs[f"instance_id_seg_{camera}"][0][..., :3]
    return rgb, depth, instance_id_seg


class LeRobotRecorder:
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
        robot_type: str | None = None,
        arm: ArmSpec | None = None,
        image_writer_threads_per_camera: int = 4,
    ):
        self.task_name = task_name
        self.repo_id = repo_id
        self.dataset_root = Path(dataset_root)
        self.fps = fps
        self.device = device
        self.cameras = cameras or {}
        self.arm = arm or load_arm_spec()[0]
        self.joint_feature_names = list(self.arm.feature_names)
        self.robot_type = robot_type or self.arm.lerobot.follower_type
        self.save_mp4 = save_mp4
        self.depth = depth
        self.instance_id_seg = instance_id_seg
        self.image_writer_threads_per_camera = image_writer_threads_per_camera

        self.dataset_features = features or self._make_sim_features(self.cameras)
        self.dataset: RecordingDataset | None = None
        self._closed = False
        self._save_failed = False
        self._frame_count = 0
        self._aux_frames: dict[str, dict[str, list[np.ndarray]]] = {"rgb": {}, "depth": {}, "instance_id_seg": {}}

    def _make_sim_features(self, cameras: dict) -> dict:
        joints = len(self.joint_feature_names)
        features = {
            "observation.state": {"dtype": "float32", "shape": (joints,), "names": self.joint_feature_names.copy()},
        }
        for camera_name, camera in cameras.items():
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": (camera["height"], camera["width"], 3),
                "names": ["height", "width", "channels"],
            }
        features["action"] = {"dtype": "float32", "shape": (joints,), "names": self.joint_feature_names.copy()}
        return features

    @staticmethod
    def _feature_signature(feature: dict) -> tuple:
        return (feature.get("dtype"), tuple(feature.get("shape", ())), tuple(feature.get("names") or ()))

    def _check_existing_dataset(self) -> None:
        if self.dataset.fps != self.fps:
            raise ValueError(f"Dataset {self.dataset_root} uses {self.dataset.fps} FPS; the control loop uses {self.fps} FPS.")
        expected = {key: self._feature_signature(value) for key, value in self.dataset_features.items()}
        actual = {
            key: self._feature_signature(value)
            for key, value in self.dataset.features.items()
            if key not in DEFAULT_FEATURES
        }
        if actual != expected:
            raise ValueError(f"Dataset feature mismatch at {self.dataset_root}. Use a new dataset root for this recording configuration.")

    def init_dataset(self) -> None:
        """Create the dataset, or resume it after checking fps, features and episode bookkeeping."""
        camera_count = sum(feature.get("dtype") in ("image", "video") for feature in self.dataset_features.values())
        resuming = (self.dataset_root / "meta" / "info.json").is_file()
        self.dataset = RecordingDataset.open(
            self.repo_id,
            self.dataset_root,
            self.fps,
            self.dataset_features,
            robot_type=self.robot_type,
            use_videos=True,
            image_writer_threads=self.image_writer_threads_per_camera * camera_count,
        )
        if resuming:
            self._check_existing_dataset()
            lengths = self.episode_lengths()
            frame_episodes = np.asarray(self.dataset.hf_dataset["episode_index"]) if lengths else np.array([])
            if not np.array_equal(frame_episodes, np.repeat(np.arange(len(lengths)), lengths)):
                raise ValueError(f"Frame data and episode lengths disagree at {self.dataset_root}.")
            print(f"[INFO]: Existing dataset initialized - {self.dataset.root}")
        else:
            print(f"[INFO]: New dataset initialized - {self.dataset.root}")

    def episode_lengths(self) -> list[int]:
        """Committed episode lengths in index order, for resume checks."""
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
    def frame_count(self) -> int:
        return self._frame_count

    @staticmethod
    def _as_numpy(value: Any, *, dtype=None) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        # Async writers must own their pixels even when simulation runs on CPU.
        return array.astype(dtype, copy=True) if dtype is not None else array.copy()

    def make_sim_frame(self, action: torch.Tensor, observation: torch.Tensor, visual_buffers: dict) -> dict:
        """One simulation frame in the arm's declared LeRobot device units (JointMapping.to_raw)."""
        frame = {
            "action": self._as_numpy(action, dtype=np.float32),
            "observation.state": self._as_numpy(observation, dtype=np.float32),
            "task": self.task_name,
        }
        for name in ("action", "observation.state"):
            if frame[name].shape != (len(self.joint_feature_names),) or not np.isfinite(frame[name]).all():
                raise ValueError(f"{name} must contain {len(self.joint_feature_names)} finite values "
                                 f"in arm {self.arm.name!r} action order")
        for camera_name in self.cameras:
            frame[f"observation.images.{camera_name}"] = self._as_numpy(visual_buffers[camera_name], dtype=np.uint8)
        return frame

    def make_auxiliary_frame(self, visual_buffers: dict, depth_buffers: dict, instance_id_seg_buffers: dict) -> dict | None:
        """Frames for the optional ``--save_mp4`` preview videos; None when disabled."""
        if not self.save_mp4:
            return None
        auxiliary: dict[str, dict[str, np.ndarray]] = {
            "rgb": {name: self._as_numpy(visual_buffers[name], dtype=np.uint8) for name in self.cameras}
        }
        if self.depth:
            auxiliary["depth"] = {name: self._as_numpy(depth_buffers[name], dtype=np.float32) for name in self.cameras}
        if self.instance_id_seg:
            auxiliary["instance_id_seg"] = {
                name: self._as_numpy(instance_id_seg_buffers[name], dtype=np.uint8) for name in self.cameras
            }
        return auxiliary

    def add_frame(self, frame: dict, auxiliary: dict | None = None) -> None:
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        if self._closed or self._save_failed:
            raise RuntimeError("Cannot append after shutdown or a failed episode save.")
        self.dataset.add_frame(frame)
        self._frame_count += 1
        for data_type, camera_frames in (auxiliary or {}).items():
            for camera_name, image in camera_frames.items():
                self._aux_frames[data_type].setdefault(camera_name, []).append(image)

    def validate_episode(self) -> None:
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        validate_episode_buffer(self.dataset.writer.episode_buffer, self.dataset.meta.total_episodes, self.dataset.features)

    def save_episode(self) -> int:
        """Commit the current episode and return its index."""
        if self.dataset is None:
            raise RuntimeError("Dataset has not been initialized.")
        self.validate_episode()
        episode_index = self.episode_index

        if self.save_mp4:
            self._save_auxiliary_videos(episode_index)

        try:
            self.dataset.save_episode(parallel_encoding=False)
        except BaseException:
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
        if self.dataset is not None and self.dataset.writer is not None:
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
            self._save_video(np.stack(frames), camera_name, "instance_id_segmentation", episode_index)

    def _save_video(self, frames_rgb: np.ndarray, camera_name: str, data_type: str, episode_index: int) -> None:
        filename = f"{camera_name}_{data_type}_{episode_index:03d}.mp4"
        filepath = self.dataset_root / "mp4" / self.repo_id.split("/")[-1] / camera_name / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        frames_rgb = np.ascontiguousarray(frames_rgb, dtype=np.uint8)
        _, height, width, _ = frames_rgb.shape
        command = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24", "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
            os.fspath(filepath),
        ]
        result = subprocess.run(command, input=frames_rgb.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=False)
        if result.returncode:
            raise RuntimeError(f"ffmpeg failed for {filepath}: {result.stderr.decode(errors='replace')}")
        print(f"[INFO]: Saved {data_type} video to {filepath}")

    def save_depth_video(self, frames: np.ndarray, camera_name: str, episode_index: int) -> None:
        """Preview video of depth, normalized to the 1st-99th percentile of finite values."""
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
        gray = (np.clip((frames - min_val) / (max_val - min_val), 0, 1) * 255).astype(np.uint8)
        self._save_video(np.stack([gray] * 3, axis=-1), camera_name, "depth", episode_index)

    def finalize(self) -> None:
        """Discard unsaved frames, then flush and close the dataset (also stops the image writer)."""
        if self._closed or self.dataset is None:
            return
        try:
            if self.frame_count:
                print(f"[WARNING]: Discarding {self.frame_count} unsaved frames at {self.dataset_root}.")
                self.clear_episode_buffer()
        finally:
            try:
                self.dataset.finalize()
            finally:
                self._closed = True


class SynchronizedLeRobotRecorders:
    def __init__(self, recorders: dict[str, LeRobotRecorder]):
        if not recorders:
            raise ValueError("At least one recorder is required.")
        self.recorders = recorders

    def init_datasets(self) -> None:
        for recorder in self.recorders.values():
            recorder.init_dataset()
        self._assert_aligned("initialization")
        lengths = {name: recorder.episode_lengths() for name, recorder in self.recorders.items()}
        if len({tuple(value) for value in lengths.values()}) != 1:
            raise ValueError(f"Existing datasets have different per-episode frame counts: {lengths}")

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
        episodes = {name: recorder.episode_index for name, recorder in self.recorders.items()}
        if len(set(episodes.values())) != 1:
            raise RuntimeError(
                f"Cannot continue synchronized recording during {operation}; "
                f"dataset episode counts differ: {episodes}. Use matching/new roots."
            )

    def add_frames(self, frames: dict[str, dict], auxiliary: dict[str, dict | None] | None = None) -> None:
        if set(frames) != set(self.recorders):
            raise ValueError(f"Frame set {set(frames)} does not match recorders {set(self.recorders)}.")
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
                "A synchronized frame write failed; all in-progress episode buffers were discarded."
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
                    raise RuntimeError(f"Recorder {name} saved episode {actual_index}; expected {expected_index}.")
                saved.append(name)
        except Exception as exc:
            for name, recorder in self.recorders.items():
                if name not in saved:
                    recorder.clear_episode_buffer()
            raise RuntimeError(
                f"Synchronized episode {expected_index} failed while saving. "
                f"Already committed datasets: {saved}. Recording stopped to prevent silent timeline divergence."
            ) from exc

        self._assert_aligned("post-save verification")
        print(f"[INFO]: Synchronized episode {expected_index} saved with {count} frames to {list(self.recorders)}.")
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

    def finalize(self) -> None:
        """Close every recorder, attempting all of them even if one fails."""
        errors = []
        for name, recorder in self.recorders.items():
            try:
                recorder.finalize()
            except BaseException as exc:
                errors.append((name, exc))
        if errors:
            details = "; ".join(f"{name}: {exc}" for name, exc in errors)
            raise RuntimeError(f"Dataset finalization failed: {details}")
