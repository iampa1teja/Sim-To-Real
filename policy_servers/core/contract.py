import importlib.util
import sys
from pathlib import Path


def load_module_from_path(name, path):
    """Import the file at `path` as module `name`, register it in sys.modules, return it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name!r} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise

    return module


def load_policy_module(policy_dir, filename):
    """Load <policy_dir>/<filename> as module 'policies.<folder>.<stem>'."""
    policy_dir = Path(policy_dir)
    file_path = policy_dir / filename
    module_name = f"policies.{policy_dir.name}.{file_path.stem}"

    folder = str(policy_dir.resolve())
    added = folder not in sys.path
    if added:
        sys.path.insert(0, folder)

    try:
        return load_module_from_path(module_name, file_path)
    finally:
        if added:
            try:
                sys.path.remove(folder)
            except ValueError:
                pass


def load_server_policy(policy_dir):
    return load_policy_module(policy_dir, "server.py")


def load_client_policy(policy_dir):
    return load_policy_module(policy_dir, "client.py")
