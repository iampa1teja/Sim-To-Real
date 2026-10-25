"""Opt-in real Docker regression for CPU decoding in the generic training image."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


IMAGE = os.environ.get("SO101_TRAIN_RUNTIME_TEST_IMAGE", "")


@unittest.skipUnless(IMAGE, "set SO101_TRAIN_RUNTIME_TEST_IMAGE for the real Docker test")
class TrainingImageRuntimeTest(unittest.TestCase):
    def test_prepared_copy_belongs_to_the_host_output_owner(self):
        with tempfile.TemporaryDirectory(prefix="so101-prepare-owner-") as folder:
            root = Path(folder)
            train = root / "train"
            train.mkdir()
            wrapper = Path(__file__).resolve().parents[1] / "policies/groot_n17/train/prepare.sh"
            (train / "prepare.sh").write_bytes(wrapper.read_bytes())
            (train / "prepare_pick_place_gr00t.py").write_text(
                "import pathlib, sys\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "out.parent.mkdir()\n"
                "out.parent.chmod(0o777)\n"
                "out.mkdir()\n"
                "stats = out / 'stats.json'\n"
                "stats.write_text('{}')\n"
                "stats.chmod(0o600)\n"
            )
            result = subprocess.run(
                ["docker", "run", "--rm", "--network", "none", "--mount",
                 f"type=bind,source={root},target=/work", "-e", "SO101_PYTHON=/usr/bin/python3",
                 "-e", "SO101_PROJECT_DIR=/unused", "-e", "SO101_DATASET=/unused",
                 "-e", "SO101_OUTPUT=/work/root-owned-parent/prepared",
                 "-e", f"SO101_HOST_UID={root.stat().st_uid}",
                 "-e", f"SO101_HOST_GID={root.stat().st_gid}", "--entrypoint", "bash", IMAGE,
                 "/work/train/prepare.sh"], capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            stats = root / "root-owned-parent/prepared/stats.json"
            self.assertEqual(stats.stat().st_uid, root.stat().st_uid)
            self.assertEqual(stats.stat().st_gid, root.stat().st_gid)
            self.assertEqual(stats.read_text(), "{}")

    def test_torchcodec_decodes_without_gpu_access(self):
        # The cached venv stays read-only; only the disposable container's /tmp is written.
        python = os.environ.get("SO101_TRAIN_RUNTIME_TEST_PYTHON",
                                "/opt/cache/venvs/groot_n17-train/bin/python")
        cache = os.environ.get("SO101_TRAIN_RUNTIME_TEST_CACHE", "policy-cache")
        script = """
import subprocess
from torchcodec.decoders import VideoDecoder

subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i',
                'color=c=red:s=16x16:r=2:d=1', '-c:v', 'libx264', '-threads', '1',
                '-pix_fmt', 'yuv420p', '/tmp/so101-decoder-test.mp4'], check=True)
frames = VideoDecoder('/tmp/so101-decoder-test.mp4', device='cpu').get_frames_at(indices=[0, 1]).data
assert tuple(frames.shape) == (2, 3, 16, 16), frames.shape
assert frames[:, 0].float().mean() > 200
assert frames[:, 1:].float().mean() < 30
print('PASS: TorchCodec CPU video decoding')
"""
        result = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", "-e", "PYTHONDONTWRITEBYTECODE=1",
             "-v", f"{cache}:/opt/cache:ro", "--entrypoint", python, IMAGE, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS: TorchCodec CPU video decoding", result.stdout)
