# Object spawner for teleop / paired sim+real recording (replaces v1's cube_spawner).
#
# Newton builds its physics model once, so objects cannot be created while the sim runs (v1 added a new
# USD cube prim per request; that only worked on PhysX). Instead, spawnable task objects are declared up
# front in the task's env cfg with a `park_object` reset event: they sit parked out of the scene, this
# module re-parks them after substantial drift, and a spawn request teleports one next to the fixed jaw tip (same
# placement as v1: jaw-tip offset, oriented like the end effector, snapped onto the table plane).
# One copy per object: spawning an object that is already out re-places it at the jaw.
#
# Requests come from `s2r spawn --object <name>` (cli/spawn.py) over a localhost HTTP listener;
# they are queued on the HTTP thread and applied on the simulation thread in `poll()`.

import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue
from threading import Event, Thread

# Parked objects are already two metres below the environment. Let gravity move
# them within this offscreen region instead of invalidating renderer transforms
# with a teleport on every recording tick.
PARK_DRIFT_LIMIT_M = 0.25


# Localhost listener: validates the auth key and request, queues it, waits (10 s) for the sim thread to answer.
class ObjectSpawnServer:
    def __init__(self, port=6001, authkey="so101-spawn"):
        self.pending = Queue(maxsize=16)
        pending = self.pending

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != "/spawn":
                    self.send_error(404)
                    return
                if not hmac.compare_digest(self.headers.get("X-Spawn-Key", ""), authkey):
                    self.send_error(403)
                    return
                done, result = Event(), {}
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length < 1024:
                        raise ValueError("Invalid request size")
                    message = json.loads(self.rfile.read(length))
                    if not isinstance(message, dict) or message.get("cmd") != "spawn":
                        raise ValueError('Expected {"cmd": "spawn", "object": <name or null>}')
                    name = message.get("object")
                    if name is not None and not isinstance(name, str):
                        raise ValueError("object must be a string or null")
                    pending.put_nowait((name, done, result))
                except Exception as exc:
                    self.send_error(400, str(exc))
                    return
                if not done.wait(10):
                    result["cancelled"] = True
                    self.send_error(503, "Simulation is not processing spawn requests")
                    return
                payload = json.dumps(result).encode()
                self.send_response(500 if "error" in result else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    # Run every queued request through callback(name) on the calling (sim) thread; returns how many ran.
    def poll(self, callback):
        handled = 0
        while True:
            try:
                name, done, result = self.pending.get_nowait()
            except Empty:
                return handled
            if result.get("cancelled"):
                continue
            try:
                result.update(callback(name))
                handled += 1
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                done.set()

    # Stop the listener thread.
    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


# Names and park offsets of every object the env cfg parks on reset: {name: park_offset}.
def parkable_objects(env_cfg, task=None):
    from sim.envs.mdp import park_object

    declared = None if task is None else {
        name for name, spec in task.spec.objects.items() if spec.params.get("spawnable", False)
    }
    found = {}
    for term in vars(env_cfg.events).values():
        func = getattr(term, "func", None)
        if func is not park_object:
            continue
        params = term.params or {}
        name = params["asset_cfg"].name
        if declared is None or name in declared:
            found[name] = tuple(params.get("park_offset", (0.0, 0.0, -2.0)))
    return found


# Half-extents (m) of a task object from task.toml: `size`, else `size_outer`; None lets the spawn cfg decide.
def _half_extents(task, name):
    if task is None or name not in task.spec.objects:
        return None
    spec = task.spec.objects[name]
    size = spec.size or spec.params.get("size_outer")
    return None if size is None else tuple(float(v) / 2.0 for v in size)


# Spawns parked objects at the jaw and keeps the rest parked. Optional HTTP listener (port 0 = none).
class ObjectSpawner:
    def __init__(self, env, task=None, port=0, authkey="so101-spawn"):
        self.env = env.unwrapped
        self.task = task
        self.offsets = parkable_objects(self.env.cfg, task)
        self.parked = set(self.offsets)
        self.server = None
        if port and self.offsets:
            try:
                self.server = ObjectSpawnServer(port, authkey)
                print(f"[INFO]: spawn listener on localhost:{port}; spawnable objects: {sorted(self.offsets)}")
            except OSError as exc:
                print(f"[WARNING]: Object spawner unavailable on port {port}: {exc}. Set spawn_port to another port.")

    # After env.reset(): the reset events parked every spawnable object again.
    def reset(self):
        self.parked = set(self.offsets)

    # Re-park only environments whose waiting object has left the offscreen region.
    def hold(self):
        if not self.parked:
            return
        from isaaclab.managers import SceneEntityCfg

        from sim.envs.mdp import as_torch, park_object

        for name in self.parked:
            position = as_torch(self.env.scene[name].data.root_pos_w)
            target = as_torch(self.env.scene.env_origins) + position.new_tensor(self.offsets[name])
            drifted = ((position - target).square().sum(dim=-1) > PARK_DRIFT_LIMIT_M ** 2).nonzero().flatten()
            if len(drifted):
                park_object(self.env, drifted, SceneEntityCfg(name), self.offsets[name])

    # Teleport `name` (default: the first spawnable object) beside the jaw tip; returns its name and position.
    def spawn(self, name=None, offset=(0.0, 0.0, 0.0)):
        from isaaclab.managers import SceneEntityCfg

        from sim.envs.mdp import as_torch, reset_object_pose

        if name is None:
            if not self.offsets:
                raise ValueError("This env has no spawnable objects (no park_object reset event).")
            name = sorted(self.offsets)[0]
        if name not in self.env.scene.keys():
            raise ValueError(f"Unknown object {name!r}; spawnable: {sorted(self.offsets)}")
        x, y, z = offset
        reset_object_pose(self.env, None, SceneEntityCfg(name), {"x": (x, x), "y": (y, y), "z": (z, z)},
                          half_extents=_half_extents(self.task, name))
        self.parked.discard(name)
        position = as_torch(self.env.scene[name].data.root_pos_w)[0].tolist()
        print(f"[INFO]: Spawned {name!r} at the jaw: {[round(v, 4) for v in position]}")
        return {"object": name, "position": position}

    # Apply queued spawn requests; returns how many objects moved (the caller then recomputes observations).
    def poll(self):
        if self.server is None:
            return 0
        return self.server.poll(self.spawn)

    # Stop the listener, if any.
    def close(self):
        if self.server is not None:
            self.server.close()
