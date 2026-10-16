"""Launcher: option parsing and error exit codes."""
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "source")]
# server.py imports policy_client as a top-level package, as in the container.
sys.modules.setdefault("policy_client", importlib.import_module("sim_to_real_so101.policy_client"))

from policy_servers.core import launcher  # noqa: E402


class ParseOptionsTest(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertEqual(launcher.parse_options(None), {})
        self.assertEqual(launcher.parse_options([]), {})

    def test_key_value_pairs(self):
        self.assertEqual(launcher.parse_options(["a=1", "b=2"]), {"a": "1", "b": "2"})

    def test_splits_on_first_equals_only(self):
        self.assertEqual(launcher.parse_options(["url=http://x?a=b"]), {"url": "http://x?a=b"})

    def test_missing_equals_raises(self):
        with self.assertRaises(ValueError):
            launcher.parse_options(["novalue"])

    def test_empty_key_raises(self):
        with self.assertRaises(ValueError):
            launcher.parse_options(["=3"])


class MainErrorsTest(unittest.TestCase):
    def test_missing_policy_dir_returns_1(self):
        self.assertEqual(launcher.main(["--policy-dir", str(ROOT / "policies" / "does_not_exist")]), 1)

    def test_bad_option_returns_1(self):
        policy_dir = ROOT / "policies" / "hold_position"
        self.assertEqual(launcher.main(["--policy-dir", str(policy_dir), "--option", "oops"]), 1)

    def test_parser_defaults(self):
        args = launcher.build_parser().parse_args(["--policy-dir", "x"])
        self.assertEqual((args.host, args.port, args.device, args.checkpoint), ("*", None, "cuda", None))


if __name__ == "__main__":
    unittest.main()
