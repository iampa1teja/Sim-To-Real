"""Shared hash helpers equal the inline expressions they replaced at each call site."""
import hashlib
import json

from utils.hashing import file_sha256, legacy_json_sha256

CALIBRATION = dict(arm="so101", mapping={"shoulder_pan": {"raw_min": -100.0, "raw_max": 100.0, "reverse": False}},
                   runtime_joint_limits={"gripper": dict(joint_min=0.0, joint_max=1.7453292519943295)},
                   parameters={"use_degrees": True, "nan": float("nan")}, selected_calibrations={"follower": "ü" * 3})
CONTRACT = dict(version=4, arm="so101", unit=None, joint_names=["a", "b"], joint_units=["deg", "m"],
                calibration_sha256="0" * 64, wrist_rezero=True, synced_joints=["a", "b"], x=1e-7)


def test_legacy_json_sha256_matches_alignment_and_dataset_suffix_expressions():
    for payload in (CALIBRATION, CONTRACT, {}, [1, {"b": 2, "a": 1}]):
        # utils/datasets/alignment.py calibration_sha256 and sim/recording/setup.py "-cal-" suffix
        old = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        assert legacy_json_sha256(payload) == old
        assert "-cal-" + legacy_json_sha256(payload)[:12] == "-cal-" + old[:12]
    # Default separators and ASCII escaping, unlike the canonical json_sha256 form.
    assert legacy_json_sha256({"b": 1, "a": [1.5, "é"]}) == hashlib.sha256(b'{"a": [1.5, "\\u00e9"], "b": 1}').hexdigest()


def test_file_sha256_matches_read_bytes_digest(tmp_path):
    # sim/datagen/mimic.py and sim/workspace/arms/urdf_import.py source hashes
    for index, data in enumerate((b"", b"x" * 1_000_003, bytes(range(256)) * 97)):
        path = tmp_path / f"{index}.bin"
        path.write_bytes(data)
        assert file_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
