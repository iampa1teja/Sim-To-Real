"""Apply local spawn requests on the simulation thread, not the HTTP thread."""
import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue
from threading import Event, Thread


class CubeSpawnServer:
    def __init__(self, port=6001, authkey='so101-spawn'):
        self.pending = Queue(maxsize=16)
        pending = self.pending

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != '/spawn':
                    self.send_error(404)
                    return
                if not hmac.compare_digest(self.headers.get('X-Spawn-Key', ''), authkey):
                    self.send_error(403)
                    return
                done, result = Event(), {}
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length < 1024:
                        raise ValueError('Invalid request size')
                    message = json.loads(self.rfile.read(length))
                    if message != {'cmd': 'spawn_blue_cube'}:
                        raise ValueError('Unsupported spawn command')
                    pending.put_nowait((done, result))
                except Exception as exc:
                    self.send_error(400, str(exc))
                    return
                if not done.wait(10):
                    result['cancelled'] = True
                    self.send_error(503, 'Simulation is not processing spawn requests')
                    return
                payload = json.dumps(result).encode()
                self.send_response(500 if 'error' in result else 200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def poll(self, callback):
        while True:
            try:
                done, result = self.pending.get_nowait()
            except Empty:
                return
            if result.get('cancelled'):
                continue
            try:
                result.update(callback())
            except Exception as exc:
                result['error'] = str(exc)
            finally:
                done.set()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def spawn_blue_cube(stage, eef_position, eef_quaternion):
    """Place the cube beside the fixed jaw tip with 3 mm surface clearance."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, Sdf

    from sim_to_real_so101.assets.real_setup import SETUP
    appearance = SETUP["object_materials"]["blue_cube"]
    units = UsdGeom.GetStageMetersPerUnit(stage)
    q = Gf.Quatd(float(eef_quaternion[0]), Gf.Vec3d(*map(float, eef_quaternion[1:])))
    half = .01 / units
    # Measure the stationary finger in gripper-local coordinates; use the live
    # sensor pose rather than potentially stale USD/Fabric world transforms.
    gripper = next((p for p in stage.Traverse() if p.GetName() == 'gripper'
                    and p.GetChild('visuals').IsValid()), None)
    if not gripper:
        raise RuntimeError('Fixed-jaw gripper link not found.')
    mesh_prim = next((p for p in Usd.PrimRange(gripper, Usd.TraverseInstanceProxies())
                      if p.IsA(UsdGeom.Mesh) and '/visuals/wrist_roll_follower_so101_v1/'
                      in str(p.GetPath())), None)
    if not mesh_prim:
        raise RuntimeError('Fixed-jaw mesh not found; cannot locate its tip.')
    local_transform, _ = UsdGeom.XformCache().ComputeRelativeTransform(mesh_prim, gripper)
    jaw_points = np.array([local_transform.Transform(Gf.Vec3d(*map(float, p)))
                           for p in UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()])
    tip_z = jaw_points[:, 2].min()
    tip_band = jaw_points[jaw_points[:, 2] <= tip_z + 2 * half]
    # Opposite side (+X): viewer's right when facing the arm from the front.
    # Measure from this face of the fixed finger, retaining 3 mm clearance.
    local_center = Gf.Vec3d(float(tip_band[:, 0].max() + half + .003 / units),
                           float((tip_band[:, 1].min() + tip_band[:, 1].max()) / 2),
                           float(tip_z + half))
    rotation = Gf.Rotation(q)
    center = np.array(eef_position, dtype=float) + np.array(rotation.TransformDir(local_center))
    table = next((p for p in stage.Traverse() if p.GetName() == 'Tabletop' and p.IsA(UsdGeom.Mesh)), None)
    if not table:
        raise RuntimeError('No Tabletop mesh found; cannot place the cube on the table.')
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(table)
    points = np.array([transform.Transform(Gf.Vec3d(*map(float, p)))
                       for p in UsdGeom.Mesh(table).GetPointsAttr().Get()])
    a, b, c = np.linalg.lstsq(np.c_[points[:, :2], np.ones(len(points))], points[:, 2], rcond=None)[0]
    # Project the oriented cube onto the table plane to prevent penetration.
    plane_normal = Gf.Vec3d(-a, -b, 1)
    clearance = half * sum(abs(Gf.Dot(plane_normal, rotation.TransformDir(axis)))
                           for axis in (Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)))
    table_clearance = a * center[0] + b * center[1] + c + clearance + .0002 / units
    center[2] = max(center[2], table_clearance)
    UsdGeom.Xform.Define(stage, '/World/SpawnedCubes')
    index = 1
    while stage.GetPrimAtPath(f'/World/SpawnedCubes/BlueCube_{index:03d}'):
        index += 1
    path = f'/World/SpawnedCubes/BlueCube_{index:03d}'
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(.02 / units)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddOrientOp().Set(Gf.Quatf(q))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*appearance["color"])])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(.008)
    material = UsdShade.Material.Define(stage, '/World/SpawnedCubes/BlueMaterial')
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*appearance["color"]))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(appearance['roughness'])
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(appearance['metallic'])
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    return {'prim_path': path, 'position': center.tolist()}
