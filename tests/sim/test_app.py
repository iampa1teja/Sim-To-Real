"""Prelaunch argument regression checks without importing Kit."""

import importlib.util
from pathlib import Path
import unittest


source = Path(__file__).resolve().parents[2] / "src/sim/__init__.py"
spec = importlib.util.spec_from_file_location("s2r_app", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rendering_kit_args = module.rendering_kit_args


class RenderingKitArgsTests(unittest.TestCase):
    def test_default_enables_dependency(self):
        self.assertEqual(rendering_kit_args(), "--enable omni.kit.context_menu")

    def test_preserves_existing_settings_and_extensions(self):
        original = "--enable other.extension --/renderer/multiGpu/enabled=false"
        self.assertEqual(rendering_kit_args(original), original + " --enable omni.kit.context_menu")

    def test_existing_dependency_is_not_added_twice(self):
        for original in ("--enable omni.kit.context_menu", "--enable=omni.kit.context_menu"):
            with self.subTest(original=original):
                self.assertEqual(rendering_kit_args(original), original)

    def test_similar_extension_name_does_not_satisfy_dependency(self):
        original = "--enable omni.kit.context_menu_extra"
        self.assertEqual(rendering_kit_args(original), original + " --enable omni.kit.context_menu")

    def test_repeated_preparation_is_idempotent(self):
        prepared = rendering_kit_args("--/app/window/width=640")
        self.assertEqual(rendering_kit_args(prepared), prepared)


if __name__ == "__main__":
    unittest.main()


def test_launch_prepares_selection_and_sdk_namespace_without_starting_kit(monkeypatch):
    from argparse import Namespace
    from types import ModuleType, SimpleNamespace
    import sys

    events = []
    app = object()
    sdk = ModuleType('isaaclab.app')

    def launcher(options):
        events.append('launch')
        assert options.enable_cameras is True
        assert options.kit_args == '--custom --enable omni.kit.context_menu'
        options.headless = True  # SDK resolves an environment default into the caller's namespace.
        return SimpleNamespace(app=app)

    sdk.AppLauncher = launcher
    monkeypatch.setitem(sys.modules, 'isaaclab.app', sdk)
    monkeypatch.setattr(module, 'apply_selection', lambda cfg: events.append(('selection', cfg)))
    args = Namespace(kit_args='--custom', enable_cameras=False)
    selection = object()
    assert module.launch(args, selection=selection, enable_cameras=True) is app
    assert events == [('selection', selection), 'launch']
    assert args.headless is True and args.enable_cameras is True


def test_shared_app_import_does_not_import_simulator_sdk():
    import os
    import subprocess
    import sys

    source_root = str(source.parents[1])
    env = dict(os.environ, PYTHONPATH=source_root)
    result = subprocess.run([sys.executable, '-c',
        "import sys; import sim; assert not any(name.startswith(('isaaclab', 'omni', 'pxr')) "
        "for name in sys.modules)"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_close_app_flushes_pending_trace_and_sets_failure_status(monkeypatch):
    from io import StringIO
    from types import SimpleNamespace
    import pytest

    class Stream(StringIO):
        flushed = False

        def flush(self):
            self.flushed = True

    stdout, stderr = Stream(), Stream()
    monkeypatch.setattr(module.sys, 'stdout', stdout)
    monkeypatch.setattr(module.sys, 'stderr', stderr)
    statuses = []

    def close(*, exit_code):
        assert stdout.flushed and stderr.flushed
        statuses.append(exit_code)
        raise SystemExit(exit_code)

    app = SimpleNamespace(close=close)
    with pytest.raises(SystemExit) as stopped:
        try:
            raise RuntimeError('original failure before Kit shutdown')
        finally:
            module.close_app(app)
    assert stopped.value.code == 1
    assert 'RuntimeError: original failure before Kit shutdown' in stderr.getvalue()
    with pytest.raises(SystemExit):
        module.close_app(app, success=False)
    with pytest.raises(SystemExit):
        module.close_app(app)
    assert statuses == [1, 1, 0]
