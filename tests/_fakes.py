"""CPU stand-ins for Isaac Sim / Isaac Lab / USD so domain-randomization code can
be unit tested without launching the simulator. Only what the tested modules
touch is faked.

Typical use in a test module:

    from _fakes import CURRENT, FakeStage, install_stubs, load_module
    def setUpModule():
        global PATCHER; PATCHER = install_stubs()
    def tearDownModule():
        PATCHER.stop()
"""
from __future__ import annotations

import importlib.util
import math
import re
import sys
import types
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "source" / "sim_to_real_so101"
REAL_SETUP_JSON = PKG / "assets" / "real_setup.json"


# ── USD stand-ins ────────────────────────────────────────────────────────────

class FakeAttr:
    def __init__(self, value=None, valid=True):
        self.value, self.valid, self.history = value, valid, []

    def Get(self):
        return self.value

    def Set(self, value):
        self.value = value
        self.history.append(value)
        return True

    def IsValid(self):
        return self.valid

    def __bool__(self):
        return self.valid


class FakeShader:
    """UsdShade.Shader: GetInput(name) returns a falsy input when absent."""

    def __init__(self, path, inputs=None):
        self.path = path
        self.inputs = {k: FakeAttr(v) for k, v in (inputs or {}).items()}

    def GetInput(self, name):
        return self.inputs.get(name, FakeAttr(valid=False))

    def GetPath(self):
        return self.path


class FakeMaterial:
    def __init__(self, shader):
        self.shader = shader

    def ComputeSurfaceSource(self):
        return self.shader, "surface", None

    def __bool__(self):
        return True


class FakePrim:
    def __init__(self, path, attrs=None, material=None, children=(), valid=True):
        self.path = path
        self.attrs = {k: FakeAttr(v) for k, v in (attrs or {}).items()}
        self.material = material
        self.children = list(children)
        self.valid = valid

    def GetAttribute(self, name):
        return self.attrs.get(name, FakeAttr(valid=False))

    def IsValid(self):
        return self.valid

    def GetPath(self):
        return self.path

    def __repr__(self):
        return f"FakePrim({self.path!r})"


class FakeStage:
    def __init__(self, prims=()):
        self.prims = {}
        for prim in prims:
            self.add(prim)

    def add(self, prim):
        self.prims[prim.path] = prim
        for child in prim.children:
            self.add(child)
        return prim

    def GetPrimAtPath(self, path):
        return self.prims.get(str(path), FakePrim(str(path), valid=False))

    def find(self, pattern):
        regex = re.compile("^" + pattern + "$")
        return [prim for path, prim in sorted(self.prims.items()) if regex.match(path)]


# The stubs read the stage from here at call time; tests assign CURRENT.stage.
CURRENT = types.SimpleNamespace(stage=FakeStage())


class _Vec(tuple):
    def __new__(cls, *values):
        return super().__new__(cls, values)


class Quatd:
    def __init__(self, w, x=0.0, y=0.0, z=0.0):
        self.w, self.xyz = float(w), (float(x), float(y), float(z))

    def GetReal(self):
        return self.w

    def GetImaginary(self):
        return self.xyz

    def as_tuple(self):
        return (self.w, *self.xyz)


class _ChangeBlock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _prim_range(root, *args):
    queue = [root]
    while queue:
        prim = queue.pop(0)
        yield prim
        queue.extend(prim.children)


class _MaterialBindingAPI:
    def __init__(self, prim):
        self.prim = prim

    def ComputeBoundMaterial(self):
        return self.prim.material, None


# ── Isaac Lab math (wxyz; same formulas as isaaclab.utils.math) ──────────────

def sample_uniform(lower, upper, size, device="cpu"):
    if isinstance(size, int):
        size = (size,)
    return torch.rand(*size, device=device) * (upper - lower) + lower


def quat_from_euler_xyz(roll, pitch, yaw):
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    return torch.stack([
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    ], dim=-1)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def yaw_quat(yaw):
    """wxyz quaternion for a rotation of `yaw` radians about +Z."""
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


class SceneEntityCfg:
    def __init__(self, name, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)


# ── sys.modules patching ─────────────────────────────────────────────────────

def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def install_stubs(extra: dict | None = None):
    """Start a sys.modules patch with fake pxr / isaaclab / isaacsim packages.
    Returns the patcher; call .stop() in tearDownModule."""
    pxr = _module("pxr")
    pxr.Gf = _module("pxr.Gf", Vec3d=_Vec, Vec3f=_Vec, Quatd=Quatd, Quatf=Quatd)
    pxr.Sdf = _module("pxr.Sdf", ChangeBlock=_ChangeBlock)
    pxr.Usd = _module("pxr.Usd", PrimRange=_prim_range, TraverseInstanceProxies=lambda: None)
    pxr.UsdShade = _module("pxr.UsdShade", MaterialBindingAPI=_MaterialBindingAPI)
    pxr.UsdGeom = _module("pxr.UsdGeom")

    isaaclab = _module("isaaclab")
    isaaclab.sim = _module(
        "isaaclab.sim",
        find_matching_prims=lambda pattern: CURRENT.stage.find(pattern),
        get_current_stage=lambda: CURRENT.stage,
    )
    isaaclab.utils = _module("isaaclab.utils")
    isaaclab.utils.math = _module(
        "isaaclab.utils.math", sample_uniform=sample_uniform,
        quat_from_euler_xyz=quat_from_euler_xyz, quat_mul=quat_mul,
    )
    isaaclab.managers = _module("isaaclab.managers", SceneEntityCfg=SceneEntityCfg)
    isaaclab.assets = _module("isaaclab.assets", Articulation=object, RigidObject=object)
    isaaclab.sensors = _module("isaaclab.sensors", FrameTransformer=object)
    isaacsim = _module("isaacsim")
    isaacsim.core = _module("isaacsim.core")
    isaacsim.core.prims = _module("isaacsim.core.prims", XFormPrim=object)

    s2r = _module("sim_to_real_so101")
    s2r.__path__ = [str(PKG)]
    s2r.utils = _module("sim_to_real_so101.utils")
    s2r.utils.__path__ = [str(PKG / "utils")]
    s2r.utils.gripper_geometry = _module(
        "sim_to_real_so101.utils.gripper_geometry",
        gripper_tip_local_offset=lambda *a, **k: (0.0, 0.0, 0.0),
        table_plane=lambda *a, **k: (0.0, 0.0, 0.0),
    )

    modules = {
        "pxr": pxr, "pxr.Gf": pxr.Gf, "pxr.Sdf": pxr.Sdf, "pxr.Usd": pxr.Usd,
        "pxr.UsdShade": pxr.UsdShade, "pxr.UsdGeom": pxr.UsdGeom,
        "isaaclab": isaaclab, "isaaclab.sim": isaaclab.sim, "isaaclab.utils": isaaclab.utils,
        "isaaclab.utils.math": isaaclab.utils.math, "isaaclab.managers": isaaclab.managers,
        "isaaclab.assets": isaaclab.assets, "isaaclab.sensors": isaaclab.sensors,
        "isaacsim": isaacsim, "isaacsim.core": isaacsim.core,
        "isaacsim.core.prims": isaacsim.core.prims,
        "sim_to_real_so101": s2r, "sim_to_real_so101.utils": s2r.utils,
        "sim_to_real_so101.utils.gripper_geometry": s2r.utils.gripper_geometry,
    }
    try:
        __import__("yaml")
    except ImportError:
        modules["yaml"] = _module("yaml")
    modules.update(extra or {})
    patcher = _KeyPatcher(modules)
    patcher.start()
    return patcher


class _KeyPatcher:
    """Like patch.dict(sys.modules, ...) but on stop() restores ONLY the keys it set.
    (patch.dict would also drop every module imported while active, e.g. torch
    internals, and torch cannot be re-imported in the same process.)"""

    _MISSING = object()

    def __init__(self, modules):
        self.modules = modules
        self.saved = {}

    def start(self):
        for name, module in self.modules.items():
            self.saved[name] = sys.modules.get(name, self._MISSING)
            sys.modules[name] = module
        return self

    def stop(self):
        for name, old in self.saved.items():
            if old is self._MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
        self.saved = {}

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def load_module(name: str, relative_path: str):
    """Import a repo source file under `name`. Skips the calling test (instead of
    erroring) when the file has not been written yet."""
    path = REPO / relative_path
    if not path.is_file():
        raise unittest.SkipTest(f"{relative_path} not implemented yet")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def block_imports(*names):
    """Patch that makes importing these packages raise ImportError (purity checks)."""
    return _KeyPatcher({name: None for name in names})
