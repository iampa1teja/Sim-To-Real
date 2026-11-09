"""Optional wall-clock attribution for the headless recording benchmark.

GPU barriers make stage boundaries explicit. Nested measurements are subtracted
from their caller, so render and camera transfers are not also charged to the
observation manager. No hooks are installed unless ``install`` is called.
"""

from collections import defaultdict
from contextlib import contextmanager
import ctypes
import functools
import threading
import time


class TickProfiler:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.current = None
        self.calls = None
        self._stack = []
        self._restore = []

    def begin_tick(self):
        self.current = defaultdict(float)
        self.calls = defaultdict(int)

    @contextmanager
    def measure(self, name):
        if self.current is None:
            yield
            return
        self.synchronize()
        self.calls[name] += 1
        entry = [name, time.perf_counter(), 0.0]
        self._stack.append(entry)
        try:
            yield
        finally:
            self.synchronize()
            elapsed = time.perf_counter() - entry[1]
            self._stack.pop()
            self.current[name] += elapsed - entry[2]
            if self._stack:
                self._stack[-1][2] += elapsed

    def wrap(self, owner, name, stage):
        original = getattr(owner, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            with self.measure(stage):
                return original(*args, **kwargs)

        setattr(owner, name, measured)
        self._restore.append((owner, name, original))

    def install(self):
        from isaaclab.managers import ObservationManager
        from isaaclab.sim import SimulationContext
        from isaaclab_newton.physics import NewtonManager
        from isaaclab_physx.renderers import isaac_rtx_renderer, isaac_rtx_renderer_utils
        from isaaclab_visualizers.kit import KitVisualizer

        self.wrap(NewtonManager, "step", "physics")
        self.wrap(SimulationContext, "render", "render_dispatch")
        self.wrap(KitVisualizer, "_pump_kit_app_update", "kit_app_update")
        self.wrap(isaac_rtx_renderer, "ensure_isaac_rtx_render_update", "rtx_render")
        self.wrap(isaac_rtx_renderer.IsaacRtxRenderer, "_read_annotator_output", "camera_readback")
        self.wrap(ObservationManager, "compute", "observation_processing")
        original = isaac_rtx_renderer_utils._get_stage_streaming_busy
        def count_completed_app_update(*args, **kwargs):
            if self.current is not None:
                self.calls["sensor_app_updates"] += 1
            return original(*args, **kwargs)
        isaac_rtx_renderer_utils._get_stage_streaming_busy = count_completed_app_update
        self._restore.append((isaac_rtx_renderer_utils, "_get_stage_streaming_busy", original))

    def close(self):
        for owner, name, original in reversed(self._restore):
            setattr(owner, name, original)
        self._restore.clear()


class GpuSampler:
    """Read GPU utilization and memory through NVML without spawning polling commands."""

    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

    class Utilization(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint) for name in ("gpu", "memory")]

    def __init__(self, interval=0.2):
        self.samples = []
        self.phase = "startup"
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def start(self):
        try:
            self._nvml = ctypes.CDLL("libnvidia-ml.so.1")
            self._check(self._nvml.nvmlInit_v2())
            self._handle = ctypes.c_void_p()
            self._check(self._nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self._handle)))
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        except Exception as error:
            self.error = str(error)

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"NVML returned {code}")

    def _poll(self):
        while not self._stop.is_set():
            try:
                memory, util = self.Memory(), self.Utilization()
                self._check(self._nvml.nvmlDeviceGetMemoryInfo(self._handle, ctypes.byref(memory)))
                self._check(self._nvml.nvmlDeviceGetUtilizationRates(self._handle, ctypes.byref(util)))
                self.samples.append({"time_s": time.time(), "phase": self.phase,
                                     "gpu_util_percent": util.gpu, "memory_util_percent": util.memory,
                                     "vram_used_mib": memory.used / 2**20, "vram_total_mib": memory.total / 2**20})
            except Exception as error:
                self.error = str(error)
                break
            self._stop.wait(self.interval)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if hasattr(self, "_nvml"):
            self._nvml.nvmlShutdown()
