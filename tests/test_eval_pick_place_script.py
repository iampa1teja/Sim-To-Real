"""Host orchestration tests using fake Docker; no daemon or simulator needed."""
import json
import os
from pathlib import Path
import shutil
import signal
import time
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'docker/eval_pick_place.sh'


class HostScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        checkpoint = self.root / 'models/model/checkpoint-10'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'config.json').write_text('{}')
        (checkpoint / 'model.safetensors').touch()
        self.log = self.root / 'docker.jsonl'
        self.env = {**os.environ, 'PATH': f'{self.bin}:{os.environ["PATH"]}', 'DOCKER_TEST_LOG': str(self.log)}
        # Delegate Python validation to the real interpreter; readiness is a
        # deterministic fake so these tests never bind/connect to host ports.
        real_python = shutil.which('python3')
        wrapper = self.bin / 'python3'
        wrapper.write_text(f'''#!/bin/bash
if [[ $1 == - ]]; then
    source_text=$(cat)
    if [[ $source_text == *socket.create_connection* || $source_text == *sock.bind* ]]; then exit 0; fi
    exec {real_python} "$@" <<< "$source_text"
fi
exec {real_python} "$@"
''')
        wrapper.chmod(0o755)
        docker = self.bin / 'docker'
        docker.write_text(f'''#!{real_python}
import json, os, sys
args = sys.argv[1:]
with open(os.environ['DOCKER_TEST_LOG'], 'a') as f:
    f.write(json.dumps(args) + '\\n')
scenario = os.environ.get('DOCKER_TEST_SCENARIO', '')
if args[0] == 'inspect':
    print('false' if scenario in ('dead', 'removed') and args[-1] != 'teleop' else 'true')
elif args[0] == 'logs':
    if scenario == 'removed' and '--follow' not in args: sys.exit(1)
    print('mock server startup error')
elif args[:2] == ['exec', 'teleop'] and 'sim_to_real_so101.scripts.lerobot_eval' in args:
    print('mock evaluation summary')
    if scenario == 'eval_fail': sys.exit(42)
    if scenario == 'hang':
        import time
        time.sleep(30)
''')
        docker.chmod(0o755)

    def run_script(self, *extra, scenario=''):
        return subprocess.run([str(SCRIPT), '--model', 'model/checkpoint-10', '--dataset', '/workspace/datasets/demo',
                               '--models_dir', str(self.root / 'models'), *extra],
                              env={**self.env, 'DOCKER_TEST_SCENARIO': scenario}, capture_output=True, text=True, timeout=10)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_plain_success_cleanup_and_argv(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        run = next(c for c in calls if c[0] == 'run')
        self.assertIn('--gpus', run)
        self.assertIn('/workspace/models/model/checkpoint-10', run)
        evaluation = next(c for c in calls if c[:2] == ['exec', 'teleop'] and 'sim_to_real_so101.scripts.lerobot_eval' in c)
        self.assertIn('Lerobot-So101-Teleop-Pick-Place-Eval', evaluation)
        self.assertIn('--headless', evaluation)
        self.assertEqual(calls[-2][0], 'stop')
        self.assertEqual(calls[-1][:2], ['rm', '-f'])
        self.assertEqual(calls[-1][-1], run[run.index('--name')+1])

    def test_dr_gui_language_is_literal(self):
        malicious = 'Pick cube; $(touch /tmp/should-not-execute-eval-test)'
        result = self.run_script('--dr', '--gui', '--lang', malicious, '--port', '5556')
        self.assertEqual(result.returncode, 0, result.stderr)
        evaluation = next(c for c in self.calls() if c[:2] == ['exec', 'teleop'] and 'sim_to_real_so101.scripts.lerobot_eval' in c)
        self.assertIn('Lerobot-So101-Teleop-Pick-Place-DR-Eval', evaluation)
        self.assertIn(malicious, evaluation)
        self.assertNotIn('--headless', evaluation)
        self.assertIn('5556', evaluation)
        self.assertFalse(Path('/tmp/should-not-execute-eval-test').exists())

    def test_eval_failure_cleans_up_and_preserves_status(self):
        result = self.run_script(scenario='eval_fail')
        self.assertEqual(result.returncode, 42)
        self.assertEqual([c[0] for c in self.calls()][-2:], ['stop', 'rm'])

    def test_failed_server_logs_and_cleanup(self):
        result = self.run_script(scenario='dead')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('mock server startup error', result.stderr)
        self.assertEqual([c[0] for c in self.calls()][-3:], ['logs', 'stop', 'rm'])

    def test_auto_removed_server_uses_streamed_logs(self):
        result = self.run_script(scenario='removed')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('mock server startup error', result.stderr)
        self.assertEqual([c[0] for c in self.calls()][-2:], ['stop', 'rm'])

    def test_ctrl_c_cleans_up_server(self):
        process = subprocess.Popen(
            [str(SCRIPT), '--model', 'model/checkpoint-10', '--dataset', '/workspace/datasets/demo',
             '--models_dir', str(self.root / 'models')],
            env={**self.env, 'DOCKER_TEST_SCENARIO': 'hang'}, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.log.exists() and 'sim_to_real_so101.scripts.lerobot_eval' in self.log.read_text():
                    break
                time.sleep(.02)
            else:
                self.fail('Mock evaluator did not start')
            os.killpg(process.pid, signal.SIGINT)
            process.communicate(timeout=5)
            self.assertEqual(process.returncode, 130)
            self.assertEqual([c[0] for c in self.calls()][-2:], ['stop', 'rm'])
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()

    def test_dry_runs_never_execute_docker(self):
        for options in ((), ('--dr',)):
            result = self.run_script('--dry_run', *options)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('docker run -d', result.stdout)
            self.assertIn('docker stop', result.stdout)
            self.assertIn('docker rm -f', result.stdout)
        self.assertFalse(self.log.exists())

    def test_path_escape_rejected(self):
        result = self.run_script('--model', '../escape')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('relative path under', result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == '__main__':
    unittest.main()
