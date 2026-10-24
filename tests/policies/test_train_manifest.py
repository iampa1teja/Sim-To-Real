"""The optional [train] table of policy.toml: accepted shapes and every way it can be wrong."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT))

from policy_servers.core.discovery import load_manifest as load_server_manifest  # noqa: E402
from sim_to_real_so101.policy_client.discovery import load_manifest  # noqa: E402

BASE = (ROOT / "policies" / "hold_position" / "policy.toml").read_text(encoding="utf-8")


def manifest_with(extra, loader=load_manifest):
    with tempfile.TemporaryDirectory() as root:
        folder = Path(root) / "hold_position"
        folder.mkdir()
        (folder / "policy.toml").write_text(BASE + extra, encoding="utf-8")
        return loader(folder)


def train_table(commands, header=""):
    return f"\n[train]\n{header}[train.commands]\n{commands}\n"


class TrainManifestTest(unittest.TestCase):
    def test_policy_without_train_table_still_loads(self):
        self.assertIsNone(manifest_with("").train)

    def test_full_table(self):
        extra = train_table("prepare = 'train/prepare.sh'\ntrain = 'train/train.sh'",
                            "dataset_format = 'lerobot_v3'\nextras = ['train']\n")
        for loader in (load_manifest, load_server_manifest):
            train = manifest_with(extra, loader).train
            self.assertEqual(train.dataset_format, "lerobot_v3")
            self.assertEqual(train.extras, ("train",))
            self.assertEqual(train.commands, {"prepare": "train/prepare.sh", "train": "train/train.sh"})

    def test_dataset_format_defaults_to_v2_1(self):
        self.assertEqual(manifest_with(train_table("train = 'train.sh'")).train.dataset_format, "lerobot_v2.1")

    def test_bad_tables_are_rejected(self):
        bad = {
            "unknown format": train_table("train = 'a.sh'", "dataset_format = 'v9'\n"),
            "no train command": train_table("prepare = 'a.sh'"),
            "no commands": "\n[train]\ndataset_format = 'lerobot_v3'\n",
            "unknown command": train_table("train = 'a.sh'\nserve = 'b.sh'"),
            "parent path": train_table("train = '../a.sh'"),
            "absolute path": train_table("train = '/a.sh'"),
            "non-string path": train_table("train = 3"),
            "bad extras": train_table("train = 'a.sh'", "extras = [1]\n"),
        }
        for label, extra in bad.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    manifest_with(extra)

    def test_backslash_parent_path_is_rejected(self):
        with self.assertRaises(ValueError):
            manifest_with(train_table(r"train = 'x\..\a.sh'"))


if __name__ == "__main__":
    unittest.main()
