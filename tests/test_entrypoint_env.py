"""policy_servers/entrypoint.sh with fake uv: serve is unchanged, --env uses its own venv and installs extras."""
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "policy_servers" / "entrypoint.sh"

FAKE_UV = """#!/usr/bin/env bash
echo "uv $*" >> "$FAKE_LOG"
echo "  env=${UV_PROJECT_ENVIRONMENT:-}" >> "$FAKE_LOG"
if [[ ${1:-} == sync ]]; then
    mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"; : > "$UV_PROJECT_ENVIRONMENT/bin/python"
fi
"""
MANIFEST = """[policy]
name = "demo"
default_port = 5599
[environment]
python = "3.12"
[observation]
cameras = ["room"]
[action]
default_horizon = 4
[runtime.env]
DEMO_FLAG = "1"
"""


def find_bash():
    """The bash on PATH, by full path: on Windows a bare 'bash' resolves to the WSL relay, not Git Bash."""
    path = shutil.which("bash")
    if not path:
        return None
    try:
        out = subprocess.run([path, "-c", "echo ok"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return path if out.stdout.strip() == "ok" else None


def find_python311():
    """An interpreter with tomllib for the entrypoint's `python3` (it reads policy.toml with it)."""
    if sys.version_info >= (3, 11):
        return [sys.executable]
    for candidate in (["py", "-3.14"], ["py", "-3.13"], ["py", "-3.12"], ["py", "-3.11"], ["python3.12"], ["python3.11"]):
        if shutil.which(candidate[0]):
            try:
                if subprocess.run([*candidate, "-c", "import tomllib"], capture_output=True, timeout=20).returncode == 0:
                    return candidate
            except (OSError, subprocess.TimeoutExpired):
                pass
    return None


BASH, PYTHON = find_bash(), find_python311()


@unittest.skipUnless(BASH and PYTHON, "needs a working bash and a Python 3.11+ for tomllib")
class EntrypointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        self.bin, self.cache, self.policies, self.log = tmp / "bin", tmp / "cache", tmp / "policies", tmp / "uv.log"
        self.bin.mkdir()
        (self.bin / "uv").write_text(FAKE_UV, encoding="utf-8", newline="\n")
        (self.bin / "python3").write_text(f"#!/usr/bin/env bash\nexec {shlex.join(PYTHON)} \"$@\"\n", encoding="utf-8", newline="\n")
        demo = self.policies / "demo"
        (demo / "train").mkdir(parents=True)
        (demo / "policy.toml").write_text(MANIFEST, encoding="utf-8", newline="\n")
        (demo / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8", newline="\n")
        (demo / "uv.lock").write_text("", encoding="utf-8", newline="\n")
        (demo / "train" / "requirements.txt").write_text("av==16.0.1\n", encoding="utf-8", newline="\n")

    def run_entrypoint(self, *args):
        import os

        env = {**os.environ, "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}", "FAKE_LOG": str(self.log),
               "POLICIES_DIR": str(self.policies), "POLICY_CACHE": str(self.cache)}
        result = subprocess.run([BASH, str(ENTRYPOINT), *args], capture_output=True, text=True, env=env, timeout=60)
        lines = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
        return result, [line.strip() for line in lines]

    def test_serve_syncs_the_serving_venv_and_runs_the_launcher(self):
        result, log = self.run_entrypoint("--policy", "demo", "--port", "7")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(log[0].startswith("uv sync --frozen --project") and log[0].endswith("--python 3.12"))
        self.assertTrue(log[1].endswith(str(self.cache / "venvs" / "demo")) or log[1].endswith("venvs/demo"))
        self.assertIn("uv run --no-sync --project", log[2])
        self.assertTrue(log[2].endswith("python -m server_core.launcher --policy-dir " + log[2].split("--policy-dir ")[1]))
        self.assertIn("--port 7", log[2])
        self.assertFalse((self.cache / "venvs" / "demo-train").exists())

    def test_env_mode_uses_a_separate_venv_installs_extras_and_runs_the_command(self):
        result, log = self.run_entrypoint("--env", "demo", "--", "printenv", "SO101_PYTHON", "SO101_PROJECT_DIR", "DEMO_FLAG")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(log[0].startswith("uv sync --frozen --project"))
        self.assertTrue(log[1].replace("\\", "/").endswith("venvs/demo-train"))
        install = log[2]
        self.assertTrue(install.startswith("uv pip install --python "))
        self.assertIn("demo-train/bin/python --no-deps -r ", install.replace("\\", "/"))
        self.assertTrue(install.replace("\\", "/").endswith("policies/demo/train/requirements.txt"))
        self.assertEqual(len([line for line in log if line.startswith("uv ")]), 2)
        python, project, flag = result.stdout.split()
        self.assertTrue(python.replace("\\", "/").endswith("venvs/demo-train/bin/python"))
        self.assertTrue(project.replace("\\", "/").endswith("policies/demo"))
        self.assertEqual(flag, "1")
        self.assertFalse((self.cache / "venvs" / "demo").exists(), "the serving venv must not be created or touched")

    def test_env_mode_without_requirements_skips_the_install(self):
        (self.policies / "demo" / "train" / "requirements.txt").unlink()
        result, log = self.run_entrypoint("--env", "demo", "--", "true")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([line for line in log if line.startswith("uv ")][-1].split()[:2], ["uv", "sync"])

    def test_env_mode_without_the_dashes_still_works(self):
        result, _ = self.run_entrypoint("--env", "demo", "printenv", "DEMO_FLAG")
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "1"))

    def test_env_mode_errors(self):
        for args, message in ((("--env",), "needs a name"), (("--env", "demo"), "needs a command"),
                              (("--env", "Bad Name", "--", "true"), "Invalid policy name"),
                              (("--env", "nope", "--", "true"), "No ")):
            with self.subTest(args=args):
                result, _ = self.run_entrypoint(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_list_and_plain_commands_are_unchanged(self):
        result, _ = self.run_entrypoint("--list")
        self.assertEqual(result.stdout.split(), ["demo"])
        result, _ = self.run_entrypoint("echo", "passthrough")
        self.assertEqual(result.stdout.strip(), "passthrough")


if __name__ == "__main__":
    unittest.main()
