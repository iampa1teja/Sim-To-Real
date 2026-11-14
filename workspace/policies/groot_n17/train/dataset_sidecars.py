"""Preserve S2R object metadata while GR00T preparation renumbers episodes."""
import json
from pathlib import Path
import sys


def copy_task_sidecars(source, output, index_map, lengths):
    """Copy and validate either sidecar schema; keep all named schema-2 objects."""
    # The train image mounts the repository read-only at its working directory;
    # direct repository invocations can find it above this policy folder.
    for root in (Path.cwd(), *Path(__file__).resolve().parents):
        if (root / 'src/utils/datasets/task_meta.py').is_file():
            path = str(root / 'src')
            if path not in sys.path:
                sys.path.insert(0, path)
            break
    from utils.datasets.task_meta import read_task_meta, sidecar_path

    copied = []
    for new_index, old_index in index_map.items():
        path = sidecar_path(source, old_index)
        normalized = read_task_meta(path, episode_index=old_index, num_frames=lengths[old_index])
        original = json.loads(path.read_text())
        payload = normalized if original.get('schema_version', 1) == 2 else original
        payload['episode_index'] = new_index
        destination = Path(output) / path.parent.name / f'episode_{new_index:06d}.json'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        copied.append(destination)
    return copied
