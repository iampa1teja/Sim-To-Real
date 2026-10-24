"""Every discovered policy must keep its client importable without torch and honour the contract."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
os.environ.setdefault("POLICIES_DIR", str(ROOT / "policies"))

from sim_to_real_so101.policy_client.contract import load_client_policy  # noqa: E402
from sim_to_real_so101.policy_client.discovery import _parse_train, discover  # noqa: E402

POLICIES = discover(ROOT / "policies")


class _TorchBlocked:
    """Make `import torch` fail while a client module is loaded."""

    def __enter__(self):
        self._saved = {k: sys.modules.get(k) for k in ("torch", "transformers")}
        for k in self._saved:
            sys.modules[k] = None
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class ConformanceTest(unittest.TestCase):
    def test_at_least_one_policy_is_discovered(self):
        self.assertTrue(POLICIES)

    def test_each_policy(self):
        for name, manifest in POLICIES.items():
            with self.subTest(policy=name):
                client_py = Path(manifest.path) / "client.py"
                if not client_py.is_file() or client_py.stat().st_size == 0:
                    self.skipTest(f"{name}: client.py not written yet")
                with _TorchBlocked():
                    client = load_client_policy(name, manifest, {})
                    joints = [f"j{i}" for i in range(6)]
                    img = np.zeros((8, 8, 3), dtype=np.uint8)
                    request = client.build_request(
                        np.zeros(len(joints), dtype=np.float32),
                        {camera: img for camera in manifest.cameras},
                        "pick the cube",
                    )
                self.assertIsInstance(request, dict)
                client.reset()

    def test_train_commands_exist(self):
        for name, manifest in POLICIES.items():
            if manifest.train is None:
                continue
            for command, script in manifest.train.commands.items():
                with self.subTest(policy=name, command=command):
                    self.assertTrue((Path(manifest.path) / script).is_file(), f"{name}: {script} is missing")

    def test_template_train_block_is_valid(self):
        import tomllib

        template = ROOT / "policies" / "_template"
        data = tomllib.loads((template / "policy.toml").read_text(encoding="utf-8"))
        train = _parse_train(data["train"])
        for script in train.commands.values():
            self.assertTrue((template / script).is_file(), f"template is missing {script}")


if __name__ == "__main__":
    unittest.main()
