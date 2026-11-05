from __future__ import annotations

import threading
from pathlib import Path

from lerobot.datasets.io_utils import load_episodes
from lerobot.datasets.lerobot_dataset import LeRobotDataset

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
