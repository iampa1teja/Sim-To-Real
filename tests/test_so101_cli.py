"""so101 CLI: exact commands for each subcommand, local and remote, and every refusal."""
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "cli"), str(ROOT / "source")]

from so101_cli import main as cli  # noqa: E402
from so101_cli import plan  # noqa: E402
from sim_to_real_so101.policy_client.discovery import discover  # noqa: E402

POLICIES = discover(ROOT / "policies")
GENERIC_TOML = """[policy]
name = "demo"
default_port = 5599
[environment]
python = "3.12"
[observation]
cameras = ["room"]
[action]
default_horizon = 4
[train]
dataset_format = "lerobot_v3"
[train.commands]
prepare = "train/prepare.sh"
train = "train/train.py"
"""


def demo_manifest():
    from sim_to_real_so101.policy_client.discovery import load_manifest

    folder = Path(tempfile.mkdtemp()) / "demo"
    folder.mkdir()
    (folder / "policy.toml").write_text(GENERIC_TOML, encoding="utf-8")
    return load_manifest(folder)


class PolicyListingTest(unittest.TestCase):
    def test_lists_policies_with_train_commands(self):
        text = cli.describe(POLICIES)
        self.assertIn("groot_n17", text)
        self.assertRegex(text, r"molmoact2\s+5556\s+prepare,train,verify\s+lerobot_v3")
        self.assertRegex(text, r"hold_position\s+\d+\s+-\s+-")


class WrapperTrainTest(unittest.TestCase):
    def setUp(self):
        self.groot, self.molmo = POLICIES["groot_n17"], POLICIES["molmoact2"]
        self.train_py = str(ROOT / "docker" / "train.py")

    def test_groot_train_maps_flags_to_environment(self):
        result = plan.plan_train("train", self.groot, ROOT, dataset="/d", output="/o", base_model="/m", recipe="sim")
        self.assertEqual(result.argv, ("python3", self.train_py, "sim"))
        self.assertEqual(result.env, {"DATASET": "/d", "OUT": "/o", "BASE_MODEL": "/m"})
        self.assertFalse(result.docker_host)

    def test_groot_prepare_and_verify_use_the_scripts_flags(self):
        prepared = plan.plan_train("prepare", self.groot, ROOT, dataset="/src", output="/out", extra=["--resume"])
        self.assertEqual(prepared.argv, ("python3", self.train_py, "prepare", "--resume", "--source", "/src", "--output", "/out"))
        verified = plan.plan_train("verify", self.groot, ROOT, dataset="/out", dry_run=True)
        self.assertEqual(verified.argv, ("python3", self.train_py, "verify", "--dry_run", "--dataset", "/out"))
        self.assertTrue(verified.runs_on_dry_run)

    def test_molmo_commands(self):
        script = str(ROOT / "docker" / "molmo_train.sh")
        prepared = plan.plan_train("prepare", self.molmo, ROOT, dataset="/s", output="/o")
        self.assertEqual(prepared.argv, (script, "python", "policies/molmoact2/train/prepare_molmo_lora.py",
                                         "--source", "/s", "--output", "/o"))
        trained = plan.plan_train("train", self.molmo, ROOT, dataset="/d", output="/o", base_model="/b")
        self.assertEqual(trained.argv, (script, "train"))
        self.assertEqual(trained.env, {"MOLMO_DATASET": "/d", "MOLMO_OUT": "/o", "MOLMO_BASE": "/b"})
        probed = plan.plan_train("verify", self.molmo, ROOT, dataset="/d")
        self.assertEqual(probed.argv, (script, "probe"))
        self.assertFalse(probed.runs_on_dry_run)

    def test_refusals(self):
        with self.assertRaisesRegex(plan.CliError, "needs --dataset"):
            plan.plan_train("train", self.groot, ROOT, output="/o")
        with self.assertRaisesRegex(plan.CliError, "needs --output"):
            plan.plan_train("prepare", self.groot, ROOT, dataset="/s")
        with self.assertRaisesRegex(plan.CliError, "only applies to the train command"):
            plan.plan_train("verify", self.groot, ROOT, dataset="/s", recipe="sim")
        with self.assertRaisesRegex(plan.CliError, "not used by molmoact2"):
            plan.plan_train("train", self.molmo, ROOT, dataset="/s", output="/o", recipe="sim")
        with self.assertRaisesRegex(plan.CliError, "same absolute paths"):
            plan.plan_train("train", self.groot, ROOT, dataset="/s", output="/o", host="u@b", remote_root="/r")
        with self.assertRaisesRegex(plan.CliError, "no \\[train\\] table"):
            plan.plan_train("train", POLICIES["hold_position"], ROOT, dataset="/s", output="/o")


class GenericTrainTest(unittest.TestCase):
    def test_shell_command_with_all_mounts(self):
        manifest = demo_manifest()
        result = plan.plan_train("prepare", manifest, ROOT, dataset="/data/src", output="/data/out", extra=["--x"],
                                 host="gpu@box", remote_root="/srv/so101/")
        self.assertEqual(result.docker_host, "ssh://gpu@box")
        self.assertEqual(result.argv, (
            "docker", "run", "--rm", "--init", "--ipc", "host", "--shm-size", "8g", "--workdir", "/workspace/repo",
            "--mount", "type=bind,source=/srv/so101,target=/workspace/repo,readonly",
            "--mount", "type=bind,source=/srv/so101/policies,target=/policies",
            "--mount", "type=bind,source=/data/src,target=/data/dataset,readonly",
            "--mount", "type=bind,source=/data,target=/data/output",
            "-v", "policy-cache:/opt/cache",
            "--env", "SO101_DATASET=/data/dataset", "--env", "SO101_OUTPUT=/data/output/out",
            "--env", "SO101_POLICY=demo", "--env", "SO101_POLICY_DIR=/policies/demo",
            "so101-train:latest", "--env", "demo", "--", "bash", "/policies/demo/train/prepare.sh", "--x"))

    def test_python_command_uses_the_policy_interpreter_and_gpu(self):
        result = plan.plan_train("train", demo_manifest(), ROOT, dataset="/d", output="/o", base_model="/b",
                                 image="img:1", generic=True)
        self.assertIn("--gpus", result.argv)
        self.assertEqual(result.argv[result.argv.index("--pid") + 1], "host")
        self.assertEqual(result.argv[-9:], ("img:1", "--env", "demo", "--", "sh", "-c",
                                            'exec "${SO101_PYTHON:-python3}" "$@"', "so101",
                                            "/policies/demo/train/train.py"))
        self.assertIn("SO101_BASE_MODEL=/data/base_model", result.argv)

    def test_generic_plan_agrees_with_the_image_about_where_policies_live(self):
        import re

        dockerfile = (ROOT / "policy_servers/Dockerfile").read_text(encoding="utf-8")
        policies_dir = re.search(r"POLICIES_DIR=(\S+)", dockerfile).group(1)
        result = plan.plan_train("train", demo_manifest(), ROOT, dataset="/d", output="/o", host="u@b", remote_root="/srv/so101")
        mounts = [part.split(",") for part in result.argv if part.startswith("type=bind")]
        policy_mount = [m for m in mounts if f"target={policies_dir}" in m]
        self.assertEqual(len(policy_mount), 1, f"nothing mounted at {policies_dir}")
        self.assertIn("source=/srv/so101/policies", policy_mount[0])
        self.assertNotIn("readonly", policy_mount[0], "uv must be able to write a missing uv.lock")
        self.assertIn(f"SO101_POLICY_DIR={policies_dir}/demo", result.argv)
        script = result.argv[result.argv.index("--") + 2 if result.argv[result.argv.index("--") + 1] == "bash" else -1]
        self.assertTrue(script.startswith(f"{policies_dir}/demo/train/"), script)

    def test_local_output_folder_is_created_before_running_but_not_on_dry_run_or_remote(self):
        manifest = demo_manifest()
        with tempfile.TemporaryDirectory() as folder:
            output = str(Path(folder) / "new" / "out")
            local = plan.plan_train("train", manifest, ROOT, dataset="/d", output=output)
            self.assertEqual(local.make_dirs, (str(Path(output).resolve()),))
            remote = plan.plan_train("train", manifest, ROOT, dataset="/d", output="/o", host="u@b", remote_root="/r")
            self.assertEqual(remote.make_dirs, ())
            with mock.patch.object(cli.subprocess, "run") as run:
                run.return_value.returncode = 0
                cli.execute(local, dry_run=True)
                self.assertFalse(Path(output).exists())
                cli.execute(local, dry_run=False)
                self.assertTrue(Path(output).is_dir())

    def test_prepare_keeps_destination_absent_for_atomic_dataset_build(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "new" / "prepared"
            result = plan.plan_train("prepare", demo_manifest(), ROOT, dataset="/d", output=str(output))
            self.assertIn(f"type=bind,source={output.parent},target=/data/output", result.argv)
            self.assertIn("SO101_OUTPUT=/data/output/prepared", result.argv)
            with mock.patch.object(cli.subprocess, "run") as run:
                run.return_value.returncode = 0
                cli.execute(result, dry_run=False)
            self.assertTrue(output.parent.is_dir())
            self.assertFalse(output.exists(), "preparation refuses an existing destination")

    def test_generic_mounts_hf_cache_readonly_for_offline_backbones(self):
        local = plan.plan_train("train", demo_manifest(), ROOT, dataset="/d", output="/o")
        self.assertIn(f"type=bind,source={Path.home() / '.cache/huggingface'},"
                      "target=/root/.cache/huggingface,readonly", local.argv)
        for command in ("prepare", "verify"):
            data_tool = plan.plan_train(command, POLICIES["groot_n17"], ROOT, dataset="/d", output="/o", generic=True)
            self.assertFalse(any("target=/root/.cache/huggingface" in arg for arg in data_tool.argv))
        remote = plan.plan_train("train", demo_manifest(), ROOT, dataset="/d", output="/o",
                                 host="u@b", remote_root="/r", hf_home="/srv/hf")
        self.assertIn("type=bind,source=/srv/hf,target=/root/.cache/huggingface,readonly", remote.argv)
        with self.assertRaisesRegex(plan.CliError, "absolute path"):
            plan.plan_train("train", demo_manifest(), ROOT, dataset="/d", output="/o",
                            host="u@b", remote_root="/r", hf_home="relative")

    def test_host_output_owner_is_forwarded_only_for_local_generic_commands(self):
        local = plan.plan_train("prepare", demo_manifest(), ROOT, dataset="/d", output="/o",
                                host_uid=1234, host_gid=5678)
        self.assertIn("SO101_HOST_UID=1234", local.argv)
        self.assertIn("SO101_HOST_GID=5678", local.argv)
        remote = plan.plan_train("prepare", demo_manifest(), ROOT, dataset="/d", output="/o",
                                 host="u@b", remote_root="/r", host_uid=1234, host_gid=5678)
        self.assertFalse(any("SO101_HOST_UID=" in arg or "SO101_HOST_GID=" in arg for arg in remote.argv))

    def test_missing_command_and_relative_remote_path(self):
        with self.assertRaisesRegex(plan.CliError, "declares no 'verify'"):
            plan.plan_train("verify", demo_manifest(), ROOT, dataset="/d")
        with self.assertRaisesRegex(plan.CliError, "absolute path on the remote host"):
            plan.plan_train("prepare", demo_manifest(), ROOT, dataset="rel", output="/o", host="u@b", remote_root="/r")
        with self.assertRaisesRegex(plan.CliError, "--host needs --remote-root"):
            plan.plan_train("prepare", demo_manifest(), ROOT, dataset="/d", output="/o", host="u@b")


class ServeTest(unittest.TestCase):
    def test_local_serve_mirrors_eval_sh(self):
        result = plan.plan_serve(POLICIES["molmoact2"], ROOT, checkpoint="molmoact2_so101", models_dir="/m",
                                 hf_home="/hf", options=["a=1"])
        root = str(ROOT.resolve())
        self.assertEqual(result.argv, (
            "docker", "run", "-d", "--rm", "--name", "so101-serve-molmoact2-5556", "--network", "host", "--gpus", "all",
            "-v", f"{root}/policies:/policies", "-v", "policy-cache:/opt/cache",
            "-v", f"{Path('/hf').resolve()}:/root/.cache/huggingface", "-v", f"{Path('/m').resolve()}:/workspace/models",
            "policy-server:latest", "--policy", "molmoact2", "--checkpoint", "/workspace/models/molmoact2_so101",
            "--port", "5556", "--option", "a=1"))
        self.assertEqual(result.docker_host, "")

    def test_remote_serve_uses_remote_paths_and_foreground(self):
        result = plan.plan_serve(POLICIES["groot_n17"], ROOT, checkpoint="run/checkpoint-200000", models_dir="/mnt/models",
                                 port=6000, host="gpu@box", remote_root="/srv/so101", foreground=True)
        self.assertEqual(result.docker_host, "ssh://gpu@box")
        self.assertNotIn("-d", result.argv)
        self.assertIn("/srv/so101/policies:/policies", result.argv)
        self.assertIn("/mnt/models:/workspace/models", result.argv)
        self.assertFalse(any("huggingface" in part for part in result.argv))
        self.assertEqual(result.argv[-6:], ("--policy", "groot_n17", "--checkpoint",
                                            "/workspace/models/run/checkpoint-200000", "--port", "6000"))

    def test_refusals(self):
        for bad in ("../x", "/abs", ""):
            with self.assertRaises(plan.CliError):
                plan.plan_serve(POLICIES["groot_n17"], ROOT, checkpoint=bad)
        with self.assertRaisesRegex(plan.CliError, "key=value"):
            plan.plan_serve(POLICIES["groot_n17"], ROOT, checkpoint="c", options=["nokey"])
        with self.assertRaisesRegex(plan.CliError, "absolute path on the remote host"):
            plan.plan_serve(POLICIES["groot_n17"], ROOT, checkpoint="c", host="u@b", remote_root="/r")


class RemoteToolsTest(unittest.TestCase):
    def test_eval_is_delegated_unchanged(self):
        result = plan.plan_eval(ROOT, ["pick_place.yaml", "--external_server", "--dry_run"])
        self.assertEqual(result.argv, ("bash", str(ROOT / "docker" / "eval.sh"), "pick_place.yaml", "--external_server", "--dry_run"))

    def test_teleop_commands_exec_into_the_container_with_env_loaded(self):
        shell = ('source /root/env 2>/dev/null; console_script=$(command -v "$1") || exit; shift; '
                 'exec /workspace/isaaclab/_isaac_sim/python.sh "$console_script" "$@"')
        record = plan.plan_teleop("pick_place_agent", ["--task", "T", "--repo_id", "u/d"])
        self.assertEqual(record.argv, (
            "docker", "exec", "-it", "-w", "/workspace/Sim-to-Real-SO-101-Workshop", "teleop", "bash", "-c", shell,
            "so101", "pick_place_agent", "--task", "T", "--repo_id", "u/d"))
        self.assertEqual(plan.plan_teleop("real_rollout", ["--policy", "p"], container="t2").argv[5], "t2")
        self.assertFalse(record.docker_host)

    def test_tunnel(self):
        self.assertEqual(plan.plan_tunnel("gpu@box", 5560).argv, (
            "ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-L", "5560:localhost:5560", "gpu@box"))

    def test_sync_defaults_and_extra_paths(self):
        result = plan.plan_sync(ROOT, "gpu@box", "/srv/so101/", ["datasets/pick_place_v1"])
        self.assertEqual(result.argv, ("rsync", "-azR", "--info=progress2", "--exclude=__pycache__", "--exclude=.git",
                                       "./datasets/pick_place_v1", "gpu@box:/srv/so101/"))
        defaults = plan.plan_sync(ROOT, "gpu@box", "/srv/so101")
        self.assertEqual(defaults.argv[5:8], ("./policies", "./configs", "./eval_sets"))
        self.assertNotIn("--delete", defaults.argv)

    def test_sync_refusals(self):
        for host, root, paths in (("", "/r", []), ("h", "rel", []), ("h", "/r", ["../x"]), ("h", "/r", ["/etc"])):
            with self.assertRaises(plan.CliError):
                plan.plan_sync(ROOT, host, root, paths)


class ImageTest(unittest.TestCase):
    def test_names_follow_registry_and_tag(self):
        self.assertEqual(plan.image_name("serve"), "so101-serve:latest")
        self.assertEqual(plan.image_name("train", "myuser", "v2"), "myuser/so101-train:v2")
        self.assertEqual(plan.image_name("teleop", "reg.example.com/team/", "v2"), "reg.example.com/team/so101-teleop:v2")
        for bad in (("gpu",), ("serve", "bad space"), ("serve", "", "-x"), ("serve", "", "a b")):
            with self.assertRaises(plan.CliError):
                plan.image_name(*bad)

    def test_default_image_uses_the_registry_when_set(self):
        self.assertEqual(plan.default_image("serve", {}), "policy-server:latest")
        self.assertEqual(plan.default_image("train", {}), "so101-train:latest")
        self.assertEqual(plan.default_image("serve", {"SO101_REGISTRY": "me", "SO101_TAG": "v3"}), "me/so101-serve:v3")

    def test_build_all_uses_one_dockerfile_for_serve_and_train_and_keeps_legacy_names(self):
        steps = plan.plan_build(plan.IMAGE_ROLES, ROOT, registry="me", tag="v1")
        root = str(ROOT.resolve())
        self.assertEqual([s.argv for s in steps], [
            ("docker", "build", "-t", "me/so101-teleop:v1", "-t", "teleop-docker:latest", "-f", "docker/sim/Dockerfile", "."),
            ("docker", "build", "--target", "serve", "-t", "me/so101-serve:v1", "-t", "policy-server:latest",
             "-f", "policy_servers/Dockerfile", "."),
            ("docker", "build", "--target", "train", "-t", "me/so101-train:v1", "-f", "policy_servers/Dockerfile", ".")])
        self.assertTrue(all(s.cwd == root for s in steps))

    def test_dockerfile_stages_match_the_build_planner_and_serve_is_the_default(self):
        import re

        stages = {}
        for dockerfile in {item[0] for item in plan.DOCKERFILES.values()}:
            text = (ROOT / dockerfile).read_text(encoding="utf-8")
            stages[dockerfile] = re.findall(r"^FROM\s+\S+(?:\s+AS\s+(\S+))?\s*$", text, flags=re.M | re.I)
        for role, (dockerfile, target) in plan.DOCKERFILES.items():
            if target:
                self.assertIn(target, stages[dockerfile], f"{role}: no stage {target!r} in {dockerfile}")
        self.assertEqual(stages["policy_servers/Dockerfile"][-1], "serve", "a plain docker build must give the server")
        self.assertIn("--target serve", (ROOT / "policy_servers/build.sh").read_text(encoding="utf-8"))

    def test_push_needs_a_registry_and_an_explicit_isaac_confirmation(self):
        with self.assertRaisesRegex(plan.CliError, "needs --registry"):
            plan.plan_build(["serve"], ROOT, push=True)
        with self.assertRaisesRegex(plan.CliError, "licence"):
            plan.plan_build(["teleop"], ROOT, registry="me", push=True)
        steps = plan.plan_build(["serve", "teleop"], ROOT, registry="me", push=True, allow_isaac_push=True)
        self.assertEqual([s.argv[:2] + s.argv[-1:] for s in steps if s.argv[1] == "push"],
                         [("docker", "push", "me/so101-serve:latest"), ("docker", "push", "me/so101-teleop:latest")])
        self.assertNotIn("push", [s.argv[1] for s in plan.plan_build(["teleop"], ROOT, registry="me")])

    def test_pull_retags_the_legacy_names(self):
        steps = plan.plan_pull(plan.IMAGE_ROLES, registry="me", tag="v1")
        self.assertEqual([s.argv for s in steps], [
            ("docker", "pull", "me/so101-teleop:v1"), ("docker", "tag", "me/so101-teleop:v1", "teleop-docker:latest"),
            ("docker", "pull", "me/so101-serve:v1"), ("docker", "tag", "me/so101-serve:v1", "policy-server:latest"),
            ("docker", "pull", "me/so101-train:v1")])
        with self.assertRaisesRegex(plan.CliError, "needs --registry"):
            plan.plan_pull(["serve"], registry="")


class MainTest(unittest.TestCase):
    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch.object(cli.subprocess, "run") as run:
            run.return_value.returncode = 0
            code = cli.main(argv, repo=ROOT)
        return code, out.getvalue(), err.getvalue(), run

    def test_dry_run_prints_and_does_not_execute(self):
        code, out, _, run = self.run_main(["serve", "--policy", "groot_n17", "--checkpoint", "c", "--models-dir", "/m",
                                           "--hf-home", "/h", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("+ docker run -d --rm --name so101-serve-groot_n17-5555"))
        run.assert_not_called()

    def test_wrapper_with_its_own_dry_run_is_executed(self):
        code, out, _, run = self.run_main(["train", "--policy", "groot_n17", "--dataset", "/d", "--output", "/o",
                                           "--dry-run", "--", "--lr", "1"])
        self.assertEqual(code, 0)
        argv = run.call_args.args[0]
        self.assertEqual(argv[-3:], ("--dry_run", "--lr", "1"))
        self.assertEqual(run.call_args.kwargs["env"]["DATASET"], "/d")

    def test_remote_sets_docker_host_in_the_child_environment(self):
        _, out, _, run = self.run_main(["serve", "--policy", "groot_n17", "--checkpoint", "c", "--models-dir", "/m",
                                        "--host", "u@b", "--remote-root", "/r"])
        self.assertEqual(run.call_args.kwargs["env"]["DOCKER_HOST"], "ssh://u@b")
        self.assertIn("DOCKER_HOST=ssh://u@b docker run", out)

    def test_eval_passthrough_keeps_flags(self):
        _, _, _, run = self.run_main(["eval", "pick_place.yaml", "--smoke-test", "--save_videos"])
        self.assertEqual(run.call_args.args[0][2:], ("pick_place.yaml", "--smoke-test", "--save_videos"))

    def test_record_and_rollout_pass_flags_through(self):
        _, out, _, run = self.run_main(["rollout", "--policy", "molmoact2", "--enable-motion", "--lang_instruction", "go"])
        self.assertEqual(run.call_args.args[0][-5:], ("real_rollout", "--policy", "molmoact2", "--enable-motion",
                                                       "--lang_instruction", "go")[-5:])
        self.assertIn("so101 real_rollout --policy molmoact2 --enable-motion --lang_instruction go", out)
        _, _, _, run = self.run_main(["eval", "--help"])
        self.assertEqual(run.call_args.args[0][2:], ("--help",))
        _, _, _, run = self.run_main(["record", "--task", "Lerobot-So101-Teleop-Pick-Place"])
        self.assertEqual(run.call_args.args[0][10:], ("pick_place_agent", "--task", "Lerobot-So101-Teleop-Pick-Place"))

    def test_sim_real_eval_passes_flags_through(self):
        _, out, _, run = self.run_main(["sim-real-eval", "--policy", "groot_n17", "--eval_set", "eval_sets/x.json"])
        self.assertEqual(run.call_args.args[0][-5:], ("sim_real_eval", "--policy", "groot_n17", "--eval_set", "eval_sets/x.json"))
        self.assertIn("so101 sim_real_eval --policy groot_n17 --eval_set eval_sets/x.json", out)

    def test_leading_dry_run_prints_pass_through_commands_without_running_them(self):
        code, out, _, run = self.run_main(["--dry-run", "rollout", "--policy", "molmoact2", "--dry-run"])
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertIn("real_rollout --policy molmoact2 --dry-run", out)

    def test_build_dry_run_prints_every_step_and_runs_nothing(self):
        code, out, _, run = self.run_main(["build", "serve", "--registry", "me", "--push", "--dry-run"])
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertIn("+ docker build --target serve -t me/so101-serve:latest -t policy-server:latest", out)
        self.assertIn("+ docker push me/so101-serve:latest", out)

    def test_pull_runs_each_step_and_stops_on_failure(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), mock.patch.object(cli.subprocess, "run") as run:
            run.return_value.returncode = 1
            code = cli.main(["pull", "all", "--registry", "me"], repo=ROOT)
        self.assertEqual((code, run.call_count), (1, 1))

    def test_push_without_the_isaac_flag_is_refused_before_anything_runs(self):
        code, _, err, run = self.run_main(["build", "teleop", "--registry", "me", "--push"])
        self.assertEqual(code, 2)
        self.assertIn("licence", err)
        run.assert_not_called()

    def test_errors_exit_2_with_a_message(self):
        code, _, err, run = self.run_main(["train", "--policy", "nope", "--dataset", "/d", "--output", "/o"])
        self.assertEqual(code, 2)
        self.assertIn("unknown policy 'nope'", err)
        self.assertIn("groot_n17", err)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
