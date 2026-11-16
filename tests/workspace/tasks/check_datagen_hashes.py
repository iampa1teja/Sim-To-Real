"""Compare real scripted datasets before/after Phase 12, without simulator imports.

Run with --before DATASET --after DATASET --output comparison.json. This retains
the Phase 0 hash convention: primitive action/state values in frame order,
stored dtype, C-order bytes; encoded videos are excluded.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from utils.io import write_json


def frame_hashes(root):
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise ValueError(f"No accepted dataset frames in {root}")
    columns = ["episode_index", "frame_index", "action", "observation.state"]
    table = pa.concat_tables([pq.ParquetFile(path).read(columns=columns) for path in files])
    result = {}
    for index in sorted(set(table["episode_index"].to_pylist())):
        episode = table.filter(pc.equal(table["episode_index"], index))
        episode = episode.sort_by([("frame_index", "ascending")])
        if episode["frame_index"].to_pylist() != list(range(episode.num_rows)):
            raise ValueError(f"Episode {index} has missing or duplicate frame indices")
        result[str(index)] = {}
        for name in columns[2:]:
            column = episode[name].combine_chunks()
            if column.null_count:
                raise ValueError(f"Null {name} values in episode {index}")
            values = column.flatten().to_numpy(zero_copy_only=False)
            array = np.ascontiguousarray(values.reshape(episode.num_rows, -1))
            result[str(index)][name] = {
                "shape": list(array.shape), "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before, after = frame_hashes(args.before), frame_hashes(args.after)
    match = before == after
    write_json(args.output, {"before": str(args.before), "after": str(args.after),
                            "before_hashes": before, "after_hashes": after, "identical": match})
    print(f"Dataset action/state frame hashes identical: {match}")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
