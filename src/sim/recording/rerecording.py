# sim/recording/rerecording.py  (ported from v1 utils/rerecording.py for LeRobot 0.6.1; used by the browser GUI)
"""Replace a recorded episode in private copies, keeping recoverable originals.

Recording always appends using LeRobot's writer. Only after Save do we splice
that new episode into its selected slot and publish the complete datasets.

LeRobot 0.6.1 differences from v1: ``finalize()`` is final (it stops the image writer and the
streaming encoder), so the live datasets are copied without finalizing them. That is safe between
episodes because RecordingDataset closes its parquet writers after every save. Videos are encoded
while recording, so there is no pending-encode step.
"""

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from core.datasets.task_meta import SIDECAR_DIR, LEGACY_SIDECAR_DIR, sidecar_path, replace_episode_meta


def transaction_path(root):
    root = Path(root).absolute()
    return root.with_name(f".{root.name}.rerecord-transaction.json")


def recover_replacement(root):
    """Roll back an interrupted paired publication before opening either dataset."""
    journal = transaction_path(root)
    if not journal.exists():
        return
    for entry in json.loads(journal.read_text()):
        original, backup = Path(entry["root"]), Path(entry["backup"])
        if backup.exists():
            if original.exists():
                original.rename(original.with_name(f".{original.name}.interrupted-{uuid.uuid4().hex}"))
            backup.rename(original)
    journal.unlink()
    print("[INFO]: Restored original datasets after interrupted re-record publication.")


def splice_last_episode(root, target):
    """Replace target with the appended episode in an expendable v3 copy.

Preserve Arrow feature metadata and video bytes. Rebuild all frame offsets and
numeric stats; camera stats come from the corresponding recorded episode.
"""
    # Heavy imports stay here so recover_replacement can run before Isaac / LeRobot start.
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
    from lerobot.datasets.io_utils import write_stats
    from lerobot.datasets.utils import DEFAULT_EPISODES_PATH
    from lerobot.utils.utils import unflatten_dict

    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    # Arrow's Python lists retain nested camera-stat dimensions; pandas' reader
    # produces object arrays whose np.asarray shape loses those dimensions.
    rows = pd.DataFrame([row for p in sorted((root / "meta/episodes").rglob("*.parquet"))
                         for row in pq.read_table(p).to_pylist()])
    rows = rows.sort_values("episode_index").reset_index(drop=True)
    last = len(rows) - 1
    if type(target) is not int or not 0 <= target < last:
        raise ValueError("Re-record episode must be an existing episode index.")
    if rows.episode_index.tolist() != list(range(last + 1)):
        raise ValueError("Episode metadata is not contiguous.")
    selected = [last if i == target else i for i in range(last)]
    videos = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    # Keep tasks stable: original task IDs remain valid, including unused tasks.
    rebuilt = root / ".rerecord-rebuilt"
    rebuilt.mkdir()
    output_rows, stats = [], []
    offset = 0
    for index, source in enumerate(selected):
        row = rows.iloc[source].to_dict()
        path = root / info["data_path"].format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
        table = pq.read_table(path)
        data = table.to_pandas()
        data = data.loc[data.episode_index == source].copy().reset_index(drop=True)
        if len(data) != int(row["length"]):
            raise ValueError(f"Episode {source} length disagrees with its parquet data.")
        data["episode_index"] = index
        data["index"] = np.arange(offset, offset + len(data), dtype=np.int64)
        chunk, file = divmod(index, info["chunks_size"])
        destination = rebuilt / info["data_path"].format(chunk_index=chunk, file_index=file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(data, schema=table.schema, preserve_index=False), destination)
        row.update(episode_index=index, dataset_from_index=offset, dataset_to_index=offset + len(data))
        row.update({"data/chunk_index": chunk, "data/file_index": file,
                    "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        numeric = {k: np.stack(data[k].to_numpy()) for k in data
                   if k in info["features"] and info["features"][k]["dtype"] not in ("video", "image", "string")}
        numeric_stats = compute_episode_stats(numeric, info["features"])
        for key, values in numeric_stats.items():
            for stat, value in values.items():
                row[f"stats/{key}/{stat}"] = value.tolist()
        for key in videos:
            old = root / info["video_path"].format(video_key=key,
                                                   chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                                   file_index=int(row[f"videos/{key}/file_index"]))
            new = rebuilt / info["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
            row[f"videos/{key}/chunk_index"] = chunk
            row[f"videos/{key}/file_index"] = file
            # Preserve timestamps: pre-existing files can contain shared videos.
        stats.append(unflatten_dict({k.removeprefix("stats/"): np.asarray(v)
                                     for k, v in row.items() if k.startswith("stats/")}))
        output_rows.append(row)
        offset += len(data)
    for name in ("data", "videos"):
        if (rebuilt / name).exists():
            shutil.rmtree(root / name)
            (rebuilt / name).rename(root / name)
    rebuilt.rmdir()
    shutil.rmtree(root / "meta/episodes")
    metadata = root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    metadata.parent.mkdir(parents=True)
    pd.DataFrame(output_rows).to_parquet(metadata, index=False)
    info.update(total_episodes=last, total_frames=offset, splits={"train": f"0:{last}"})
    (root / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    write_stats(aggregate_stats(stats), root)
    try:
        sidecar = sidecar_path(root, last)
    except ValueError:
        sidecar = None
    if sidecar is not None:
        replace_episode_meta(root, last, target)
    for path in (root / "mp4").rglob(f"*_{last:03d}.mp4"):
        os.replace(path, path.with_name(path.name[:-len(f"{last:03d}.mp4")] + f"{target:03d}.mp4"))


# Same recorders as ``group``, opened on other dataset roots (a staged copy, or the published originals).
def _fresh_group(group, roots):
    from .lerobot_recorder import LeRobotRecorder, SynchronizedLeRobotRecorders
    recorders = {}
    for name, old in group.recorders.items():
        recorders[name] = LeRobotRecorder(
            old.task_name, old.repo_id, str(roots[name]), old.fps, old.device,
            old.cameras, old.save_mp4, old.depth, old.instance_id_seg,
            features=copy.deepcopy(old.dataset_features), robot_type=old.robot_type,
            arm=old.arm,
            image_writer_threads_per_camera=old.image_writer_threads_per_camera,
        )
    result = SynchronizedLeRobotRecorders(recorders)
    try:
        result.init_datasets()
    except BaseException:
        result.finalize()
        raise
    return result


class RerecordingSession:
    """Control-loop adapter; originals are untouched until a replacement Save."""

    def __init__(self, group):
        self.group = group
        self.original = None
        self.target = None
        self.roots = None
        self._save_started = False

    def __getattr__(self, name):
        return getattr(self.group, name)

    @property
    def episode_index(self):
        return self.target if self.target is not None else self.group.episode_index

    @property
    def saved_episodes(self):
        return (self.original or self.group).episode_index

    def begin(self, index):
        if self.target is not None or self.frame_count:
            raise ValueError("Finish the current recording before re-recording.")
        if type(index) is not int or not 0 <= index < self.saved_episodes:
            raise ValueError("Choose an existing episode index (starting at 0).")
        roots = {name: rec.dataset_root.absolute() for name, rec in self.group.recorders.items()}
        stages = {name: root.with_name(f".{root.name}.rerecord-{uuid.uuid4().hex}") for name, root in roots.items()}
        try:
            # No finalize here: in LeRobot 0.6.1 it would end the original datasets, which Discard keeps using.
            for name in self.group.recorders:
                shutil.copytree(roots[name], stages[name])
            staged = _fresh_group(self.group, stages)
        except BaseException:
            for stage in stages.values():
                if stage.exists():
                    shutil.rmtree(stage)
            raise
        self.original, self.group = self.group, staged
        self.target, self.roots = index, roots
        self._save_started = False

    def cancel_episode(self):
        self.group.cancel_episode()
        if self.original is not None:
            staged = self.group
            staged.finalize()
            self.group, self.original = self.original, None
            self.target = None
            for rec in staged.recorders.values():
                shutil.rmtree(rec.dataset_root)

    # Save the episode. ``sidecar(root, index)`` writes its object trajectory next to the sim dataset first.
    # A normal save returns the new index; a re-record splices it into the target slot and returns the target.
    def save_episode(self, sidecar=None):
        from core.datasets.task_meta import validate_task_trajectories
        self._save_started = self.original is not None
        index = self.group.save_episode()
        if index is None:
            self._save_started = False
            return None
        if sidecar is not None:
            sidecar(self.group.recorders["sim"].dataset_root, index)
        if self.original is None:
            return index
        target = self.target
        fps = self.group.recorders["sim"].fps
        self.group.finalize()
        for rec in self.group.recorders.values():
            splice_last_episode(rec.dataset_root, target)
        # Reopen every staged dataset to validate its frame/episode alignment.
        stages = {name: rec.dataset_root for name, rec in self.group.recorders.items()}
        checked = _fresh_group(self.group, stages)
        try:
            if any((Path(stages["sim"]) / name).is_dir() for name in (SIDECAR_DIR, LEGACY_SIDECAR_DIR)):
                validate_task_trajectories(stages["sim"], checked.recorders["sim"].episode_lengths(), fps)
        finally:
            checked.finalize()
        self.original.finalize()
        entries = [{"root": str(root), "backup": str(root.with_name(f"{root.name}.before-rerecord-{uuid.uuid4().hex}")),
                    "stage": str(stages[name])} for name, root in self.roots.items()]
        journal = transaction_path(self.roots["sim"])
        if journal.exists():
            raise RuntimeError(f"Unresolved re-record transaction: {journal}. Restart the recorder.")
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
        try:
            for entry in entries:
                Path(entry["root"]).rename(entry["backup"])
                Path(entry["stage"]).rename(entry["root"])
            reopened = _fresh_group(self.original, self.roots)
        except BaseException:
            recover_replacement(self.roots["sim"])
            raise
        journal.unlink()
        self.group, self.original, self.target = reopened, None, None
        for entry in entries:
            print(f"[INFO]: Original dataset backup: {entry['backup']}")
        return target

    def finalize(self):
        if self.original is not None and not self._save_started:
            self.cancel_episode()
        if self.original is not None:
            # Failed saves keep their staging copies for diagnosis/recovery.
            self.group.finalize()
            self.original.finalize()
        else:
            self.group.finalize()
