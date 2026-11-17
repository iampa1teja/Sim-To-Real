"""WP5: ``core.generate.fingerprint`` (contracts §8.2, §1.1 quaternion sign, §1.3 dtypes)."""

import hashlib
import json
import math
import unittest

import numpy as np

from core.generate.fingerprint import appearance_fp, episode_fp, quantize, scenario_key, spec_fp, traj_fp
from core.generate.protocols import EpisodeSpec, ObjectState
from core.tasks.base import Pose
from core.utils.hashing import json_sha256
from v4_support import FIXTURES_DIR
from v4_support import factories as fx

GOLDEN = json.loads((FIXTURES_DIR / "fingerprints" / "golden.json").read_text())["cases"]
HEX_CHARS = set("0123456789abcdef")


def _is_hex(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX_CHARS


def _with(spec, **changes):
    """Copy of ``spec`` with fields replaced (EpisodeSpec is frozen)."""
    fields = {name: getattr(spec, name) for name in spec.__dataclass_fields__}
    fields.update(changes)
    return EpisodeSpec(**fields)


def _move(spec, name, *, dx=0.0, dy=0.0, dz=0.0, dyaw=0.0, color=..., active=None):
    """Copy of ``spec`` with object ``name`` moved/rotated about +z/recoloured."""
    st = spec.objects[name]
    x, y, z = st.pose.position
    qx, qy, qz, qw = st.pose.quat_xyzw
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)) + dyaw
    objects = dict(spec.objects)
    objects[name] = ObjectState(pose=fx.make_pose(x + dx, y + dy, z + dz, yaw), active=st.active if active is None else active,
                                color=st.color if color is ... else color)
    return _with(spec, objects=objects)


def _base_spec(kind="success", **overrides):
    """Spec whose positions sit at 1e-3 bin centres, yaw at a 1° bin centre and q0 at π/360 bin centres."""
    fields = dict(
        objects={"cube_0": fx.make_object_state(0.200, 0.050, 0.010, yaw=math.radians(30.0), color="red"),
                 "cube_2": fx.make_object_state(0.150, -0.080, 0.010, yaw=math.radians(-12.0), color="green"),
                 "cube_1": fx.make_object_state(0.1, 0.0, -2.0, active=False),
                 "box": fx.make_object_state(0.250, -0.050, 0.000)},
        roles={"target": "cube_0", "container": "box"},
        goal={"point": [0.210, -0.050, 0.120], "gap": 0.015, "count": 2, "flag": True},
        initial_q=tuple(k * math.pi / 360 for k in (0, -40, 180, 140, 3, 60)),
    )
    fields.update(overrides)
    return fx.make_episode_spec(kind, **fields)


class GoldenTests(unittest.TestCase):
    """WP5 §8.2 golden values (tests/fixtures/fingerprints/golden.json)."""

    def test_fixture_payloads_hash_to_expected(self):
        """§8.2 every golden payload hashes (json_sha256) to its expected value."""
        for name, cases in GOLDEN.items():
            for case in cases:
                if "payload" in case:
                    with self.subTest(fp=name, case=case["name"]):
                        self.assertEqual(json_sha256(case["payload"]), case["expected"])

    def test_scenario_key_golden(self):
        """§8.2 scenario_key golden values."""
        for case in GOLDEN["scenario_key"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(scenario_key(EpisodeSpec.from_json(case["spec"])), case["expected"])

    def test_spec_fp_golden(self):
        """§8.2 spec_fp golden values."""
        for case in GOLDEN["spec_fp"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(spec_fp(EpisodeSpec.from_json(case["spec"])), case["expected"])

    def test_traj_fp_golden(self):
        """§8.2 traj_fp golden values (lists, float64 and float32 arrays)."""
        for case in GOLDEN["traj_fp"]:
            for label, conv in (("list", lambda v: v), ("f64", lambda v: np.asarray(v, np.float64)),
                                ("f32", lambda v: np.asarray(v, np.float32))):
                with self.subTest(case=case["name"], input=label):
                    self.assertEqual(traj_fp(conv(case["action_raw"]), conv(case["state_raw"]), case["objects0"]),
                                     case["expected"])

    def test_appearance_fp_golden(self):
        """§8.2 appearance_fp golden values."""
        for case in GOLDEN["appearance_fp"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(appearance_fp(case["colors"], case["instruction"], case["light_exposure"],
                                               case["robot_color"]), case["expected"])

    def test_episode_fp_golden(self):
        """§8.2 episode_fp = sha256((traj_fp + appearance_fp).encode('ascii'))."""
        for case in GOLDEN["episode_fp"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(episode_fp(case["traj_fp"], case["appearance_fp"]), case["expected"])
                self.assertEqual(case["expected"],
                                 hashlib.sha256((case["traj_fp"] + case["appearance_fp"]).encode("ascii")).hexdigest())

    def test_outputs_are_lower_hex(self):
        """§3.2 Fingerprints are lower-case hex SHA-256 strings."""
        spec = _base_spec()
        t = traj_fp(np.zeros((3, 6)), np.zeros((3, 6)), {"cube_0": (0.2, 0.0, 0.01)})
        a = appearance_fp({"cube_0": "red"}, "x", None, None)
        for value in (scenario_key(spec), spec_fp(spec), t, a, episode_fp(t, a)):
            self.assertTrue(_is_hex(value), value)


class QuantizeTests(unittest.TestCase):
    """WP5 §8.2 quantize."""

    def test_formula_and_python_int(self):
        """§8.2 quantize(x, step) = floor(x / step + 0.5) as Python int."""
        for k in range(3):
            rng = np.random.default_rng(k)
            for _ in range(200):
                x, step = float(rng.uniform(-5, 5)), float(rng.choice([1e-3, 1e-4, math.pi / 180, math.pi / 360, 0.01]))
                with self.subTest(x=x, step=step):
                    value = quantize(x, step)
                    self.assertIs(type(value), int)
                    self.assertEqual(value, math.floor(x / step + 0.5))

    def test_half_rounds_up(self):
        """§8.2 floor(x/step + 0.5): exact halves round towards +inf (not banker's rounding)."""
        self.assertEqual(quantize(0.5, 1.0), 1)
        self.assertEqual(quantize(1.5, 1.0), 2)
        self.assertEqual(quantize(2.5, 1.0), 3)
        self.assertEqual(quantize(-0.5, 1.0), 0)
        self.assertEqual(quantize(-1.5, 1.0), -1)

    def test_negative_zero(self):
        """§8.2 -0.0 and 0.0 quantize to the same int."""
        for step in (1e-3, 1e-4, 0.01):
            self.assertEqual(quantize(-0.0, step), quantize(0.0, step))
            self.assertEqual(quantize(-0.0, step), 0)

    def test_float32_and_numpy_scalars_agree_with_python_float(self):
        """§1.3/§8.2 a float32 (or float64) numpy scalar quantizes like the Python float of the same value."""
        rng = np.random.default_rng(3)
        for _ in range(300):
            v32 = np.float32(rng.uniform(-3, 3))
            for step in (1e-3, 1e-4, math.pi / 180):
                with self.subTest(v=float(v32), step=step):
                    self.assertEqual(quantize(v32, step), quantize(float(v32), step))
                    self.assertEqual(quantize(np.float64(v32), step), quantize(float(v32), step))
                    self.assertIs(type(quantize(v32, step)), int)

    def test_array_form(self):
        """§8.2 numpy form: np.floor(x / step + 0.5).astype(np.int64)."""
        a = np.random.default_rng(0).uniform(-2, 2, (5, 3))
        out = quantize(a, 1e-3)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.int64)
        np.testing.assert_array_equal(out, np.floor(a / 1e-3 + 0.5).astype(np.int64))

    def test_rejects_non_finite_and_bad_step(self):
        """§1.5 plain ValueError for bad arguments of pure functions (non-finite values, step <= 0)."""
        for x in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(x=x), self.assertRaises(ValueError):
                quantize(x, 1e-3)
        for step in (0.0, -1e-3):
            with self.subTest(step=step), self.assertRaises(ValueError):
                quantize(1.0, step)


class ScenarioKeyTests(unittest.TestCase):
    """WP5 §8.2 scenario_key (D12: excludes slot names, kind, instruction; includes colours)."""

    def test_deterministic(self):
        """§8.2 equal specs give equal keys."""
        self.assertEqual(scenario_key(_base_spec()), scenario_key(_base_spec()))

    def test_excluded_fields(self):
        """§8.2 kind, attempt index, seed, instruction, perturbation, noise, max_steps and fps are excluded."""
        base = _base_spec("success")
        key = scenario_key(base)
        variants = {
            "failure": _base_spec("failure"),
            "recovery": _base_spec("recovery", perturbation=fx.make_perturbation("object_shift", params={"xy_m": 0.03})),
            "attempt_index": _with(base, attempt_index=999),
            "seed": _with(base, seed=2**64 - 1),
            "instruction": _with(base, instruction="Something else entirely"),
            "noise": _with(base, noise=fx.make_noise()),
            "max_steps": _with(base, max_steps=123),
            "fps": _with(base, fps=60),
        }
        for name, spec in variants.items():
            with self.subTest(changed=name):
                self.assertEqual(scenario_key(spec), key)

    def test_slot_names_excluded(self):
        """§8.2 renaming set slots (cube_0 -> cube_4, ...) does not change the key."""
        base = _base_spec()
        renamed = {"cube_0": "cube_5", "cube_2": "cube_3", "cube_1": "cube_0"}
        objects = {renamed.get(n, n): s for n, s in base.objects.items()}
        roles = {r: renamed.get(n, n) for r, n in base.roles.items()}
        self.assertEqual(scenario_key(_with(base, objects=objects, roles=roles)), scenario_key(base))

    def test_inactive_objects_excluded(self):
        """§8.2 only active objects enter the key: moving or recolouring an inactive object changes nothing."""
        base = _base_spec()
        self.assertEqual(scenario_key(_move(base, "cube_1", dx=0.37, color="pink")), scenario_key(base))

    def test_included_fields_change_the_key(self):
        """§8.2 task, arm, scene, colours, roles, positions, yaw, goal, q0, held and activity enter the key."""
        base = _base_spec()
        key = scenario_key(base)
        q0 = list(base.initial_q)
        q0[2] += math.pi / 360
        variants = {
            "task": _with(base, task="fx_other"), "arm": _with(base, arm="fx_arm6"), "scene": _with(base, scene="s2"),
            "color": _move(base, "cube_2", color="blue"),
            "color_none": _move(base, "cube_0", color=None),
            "swap_roles": _with(base, roles={"target": "cube_2", "container": "box"}),
            "drop_role": _with(base, roles={"target": "cube_0"}),
            "x": _move(base, "cube_0", dx=1e-3), "y": _move(base, "box", dy=-1e-3), "z": _move(base, "cube_2", dz=1e-3),
            "yaw": _move(base, "cube_0", dyaw=math.radians(1.0)),
            "goal_value": _with(base, goal={**base.goal, "gap": 0.016}),
            "goal_key": _with(base, goal={**base.goal, "extra": 1}),
            "q0": _with(base, initial_q=tuple(q0)),
            "held": _with(base, held_object="cube_0"),
            "activate": _move(base, "cube_1", active=True),
        }
        for name, spec in variants.items():
            with self.subTest(changed=name):
                self.assertNotEqual(scenario_key(spec), key)

    def test_quantisation_sensitivity(self):
        """§8.2 a move of 0.4·step keeps the key; 1.0·step changes it (p 1e-3, yaw π/180, goal 1e-3, q0 π/360)."""
        base = _base_spec()
        key = scenario_key(base)

        def q0_shift(i, d):
            q0 = list(base.initial_q)
            q0[i] += d
            return _with(base, initial_q=tuple(q0))

        def goal_shift(d):
            p = list(base.goal["point"])
            p[1] += d
            return _with(base, goal={**base.goal, "point": p, "gap": base.goal["gap"] + d})

        for factor, same in ((0.4, True), (-0.4, True), (1.0, False), (-1.0, False)):
            cases = {
                "x": _move(base, "cube_0", dx=factor * 1e-3), "y": _move(base, "cube_2", dy=factor * 1e-3),
                "z": _move(base, "box", dz=factor * 1e-3), "yaw": _move(base, "cube_2", dyaw=factor * math.pi / 180),
                "goal": goal_shift(factor * 1e-3), "q0": q0_shift(4, factor * math.pi / 360),
            }
            for name, spec in cases.items():
                with self.subTest(field=name, factor=factor):
                    (self.assertEqual if same else self.assertNotEqual)(scenario_key(spec), key)

    def test_negative_zero_positions(self):
        """§8.2 -0.0 and 0.0 coordinates, goal values and joints give the same key."""
        a = _base_spec(objects={"cube_0": ObjectState(Pose((0.2, 0.0, 0.01), (0.0, 0.0, 0.0, 1.0)), color="red"),
                                "box": ObjectState(Pose((0.25, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))},
                       goal={"point": [0.0, 0.1, 0.0]}, initial_q=(0.0,) * 6)
        b = _base_spec(objects={"cube_0": ObjectState(Pose((0.2, -0.0, 0.01), (0.0, 0.0, -0.0, 1.0)), color="red"),
                                "box": ObjectState(Pose((0.25, -0.0, -0.0), (-0.0, 0.0, 0.0, 1.0)))},
                       goal={"point": [-0.0, 0.1, -0.0]}, initial_q=(-0.0,) * 6)
        self.assertEqual(scenario_key(a), scenario_key(b))

    def test_float32_inputs_agree(self):
        """§1.3 float32 coordinates give the same key as their float64 values."""
        rng = np.random.default_rng(11)
        for _ in range(30):
            xyz = rng.uniform(-0.3, 0.3, 3).astype(np.float32)
            yaw = np.float32(rng.uniform(-3, 3))
            q32 = tuple(np.float32(v) for v in fx.yaw_quat_xyzw(float(yaw)))
            p32 = ObjectState(Pose(tuple(xyz), q32), color="red")
            p64 = ObjectState(Pose(tuple(float(v) for v in xyz), tuple(float(v) for v in q32)), color="red")
            box = fx.make_object_state(0.25, -0.05, 0.0)
            with self.subTest(xyz=xyz.tolist()):
                self.assertEqual(scenario_key(_base_spec(objects={"cube_0": p32, "box": box})),
                                 scenario_key(_base_spec(objects={"cube_0": p64, "box": box})))

    def test_quaternion_sign(self):
        """§1.1 fingerprints canonicalise the quaternion sign: q and -q give the same key."""
        base = _base_spec()
        objects = {n: ObjectState(Pose(s.pose.position, tuple(-v for v in s.pose.quat_xyzw)), s.active, s.color)
                   for n, s in base.objects.items()}
        self.assertEqual(scenario_key(_with(base, objects=objects)), scenario_key(base))

    def test_dict_order_independent(self):
        """§8.2 canonical JSON: insertion order of objects, roles and goal does not matter."""
        base = _base_spec()
        rev = _with(base, objects=dict(reversed(list(base.objects.items()))),
                    roles=dict(reversed(list(base.roles.items()))), goal=dict(reversed(list(base.goal.items()))))
        self.assertEqual(scenario_key(rev), scenario_key(base))

    def test_identical_objects_swapped_slots(self):
        """§8.2 two same-kind, same-colour slot objects exchanging poses are the same scenario."""
        base = fx.make_episode_spec(objects={"cube_0": fx.make_object_state(0.20, 0.05, 0.01, color="red"),
                                             "cube_1": fx.make_object_state(0.15, -0.05, 0.01, color="blue"),
                                             "cube_2": fx.make_object_state(0.25, 0.10, 0.01, color="blue"),
                                             "box": fx.make_object_state(0.25, -0.05, 0.0)})
        swapped = fx.make_episode_spec(objects={"cube_0": fx.make_object_state(0.20, 0.05, 0.01, color="red"),
                                                "cube_2": fx.make_object_state(0.15, -0.05, 0.01, color="blue"),
                                                "cube_1": fx.make_object_state(0.25, 0.10, 0.01, color="blue"),
                                                "box": fx.make_object_state(0.25, -0.05, 0.0)})
        self.assertEqual(scenario_key(swapped), scenario_key(base))

    def test_random_specs_distinct_and_stable(self):
        """§8.2 random distinct scenarios get distinct keys; a JSON round-trip keeps the key."""
        keys = set()
        for k in range(4):
            rng = np.random.default_rng(k)
            for _ in range(25):
                spec = fx.random_episode_spec(rng)
                again = EpisodeSpec.from_json(json.loads(json.dumps(spec.to_json())))
                self.assertEqual(scenario_key(again), scenario_key(spec))
                keys.add(scenario_key(spec))
        self.assertEqual(len(keys), 100)


class SpecFpTests(unittest.TestCase):
    """WP5 §8.2 spec_fp."""

    def test_excludes_attempt_index_and_seed_only(self):
        """§8.2 spec_fp ignores attempt_index and seed but sees kind, instruction, perturbation, noise, steps."""
        base = _base_spec("failure")
        fp = spec_fp(base)
        self.assertEqual(spec_fp(_with(base, attempt_index=77)), fp)
        self.assertEqual(spec_fp(_with(base, seed=1)), fp)
        changed = {
            "kind": _base_spec("recovery"), "instruction": _with(base, instruction="other"),
            "perturbation": _with(base, perturbation=fx.make_perturbation(params={"xy_m": 0.02, "dir_rad": 1.0})),
            "max_steps": _with(base, max_steps=601), "fps": _with(base, fps=15),
            "slot_name": _with(base, objects={("cube_9" if n == "cube_2" else n): s for n, s in base.objects.items()}),
            "inactive_moved": _move(base, "cube_1", dx=0.01),
        }
        for name, spec in changed.items():
            with self.subTest(changed=name):
                self.assertNotEqual(spec_fp(spec), fp)
        success = _base_spec("success")
        self.assertNotEqual(spec_fp(_with(success, noise=fx.make_noise())), spec_fp(success))

    def test_quantisation_sensitivity(self):
        """§8.2 floats quantized at 1e-4: ±0.4e-4 keeps, ±1e-4 changes (positions, goal, q0, params)."""
        base = _with(_base_spec("failure"), initial_q=(0.0, -0.3, 1.6, 1.2, 0.0, 0.5),
                     perturbation=fx.make_perturbation(params={"xy_m": 0.0125, "dir_rad": 1.0}))
        fp = spec_fp(base)
        for factor, same in ((0.4, True), (-0.4, True), (1.0, False), (-1.0, False)):
            d = factor * 1e-4
            q0 = list(base.initial_q)
            q0[1] += d
            cases = {
                "x": _move(base, "cube_0", dx=d), "goal": _with(base, goal={**base.goal, "gap": base.goal["gap"] + d}),
                "q0": _with(base, initial_q=tuple(q0)),
                "param": _with(base, perturbation=fx.make_perturbation(params={"xy_m": 0.0125 + d, "dir_rad": 1.0})),
            }
            for name, spec in cases.items():
                with self.subTest(field=name, factor=factor):
                    (self.assertEqual if same else self.assertNotEqual)(spec_fp(spec), fp)

    def test_ints_and_bools_unchanged(self):
        """§8.2 ints and bools are not quantized: goal ints 2 vs 3 differ, True vs False differ."""
        base = _base_spec()
        self.assertNotEqual(spec_fp(_with(base, goal={**base.goal, "count": 3})), spec_fp(base))
        self.assertNotEqual(spec_fp(_with(base, goal={**base.goal, "flag": False})), spec_fp(base))

    def test_negative_zero_sign_and_order(self):
        """§1.1/§8.2 -0.0 == 0.0, q == -q, and mapping order is irrelevant."""
        base = _base_spec()
        objects = {n: ObjectState(Pose(s.pose.position, tuple(-v for v in s.pose.quat_xyzw)), s.active, s.color)
                   for n, s in reversed(list(base.objects.items()))}
        self.assertEqual(spec_fp(_with(base, objects=objects, goal=dict(reversed(list(base.goal.items()))))),
                         spec_fp(base))
        a = _with(base, initial_q=(0.0,) * 6, goal={"point": [0.0, 0.0, 0.1]})
        b = _with(base, initial_q=(-0.0,) * 6, goal={"point": [-0.0, -0.0, 0.1]})
        self.assertEqual(spec_fp(a), spec_fp(b))

    def test_json_round_trip_stable(self):
        """§3.4/§8.2 spec_fp survives EpisodeSpec JSON round-trips (random specs, all kinds)."""
        fps = set()
        for k in range(4):
            rng = np.random.default_rng(100 + k)
            for _ in range(25):
                spec = fx.random_episode_spec(rng)
                again = EpisodeSpec.from_json(json.loads(json.dumps(spec.to_json())))
                with self.subTest(k=k):
                    self.assertEqual(spec_fp(again), spec_fp(spec))
                fps.add(spec_fp(spec))
        self.assertEqual(len(fps), 100)


def _traj(rng, n=20, j=6):
    """Raw trajectory at 0.01-bin centres (multiples of 0.01, exactly representable in float32 well enough)."""
    action = np.round(rng.uniform(-90, 90, (n, j)), 2)
    state = np.round(rng.uniform(-90, 90, (n, j)), 2)
    objects0 = {"cube_0": (0.2001, 0.0503, 0.0100), "box": (0.2500, -0.0500, 0.0000)}
    return action, state, objects0


class TrajFpTests(unittest.TestCase):
    """WP5 §8.2 traj_fp."""

    def test_reference_formula(self):
        """§8.2 traj_fp payload: shape, sha256 of Q(action) / Q(state) as '<i4' bytes, objects0 at 1e-4."""
        def Q(a):
            return np.floor(np.asarray(a, np.float32).astype(np.float64) * 100 + 0.5).astype("<i4").tobytes()
        for k in range(5):
            rng = np.random.default_rng(k)
            n, j = int(rng.integers(1, 40)), int(rng.integers(5, 8))
            action, state = rng.uniform(-100, 100, (n, j)), rng.uniform(-100, 100, (n, j))
            objects0 = {f"o{i}": tuple(rng.uniform(-0.5, 0.5, 3)) for i in range(int(rng.integers(0, 4)))}
            payload = {"v": 1, "shape": [n, j], "action": hashlib.sha256(Q(action)).hexdigest(),
                       "state": hashlib.sha256(Q(state)).hexdigest(),
                       "objects0": {name: [math.floor(c / 1e-4 + 0.5) for c in p] for name, p in objects0.items()}}
            with self.subTest(k=k):
                self.assertEqual(traj_fp(action, state, objects0), json_sha256(payload))

    def test_float32_and_float64_agree(self):
        """§8.2 Q casts to float32 first: float64 data and its float32 cast give the same traj_fp."""
        for k in range(10):
            rng = np.random.default_rng(k)
            action, state, objects0 = rng.uniform(-100, 100, (15, 6)), rng.uniform(-100, 100, (15, 6)), {"a": (0.1, 0.2, 0.3)}
            with self.subTest(k=k):
                self.assertEqual(traj_fp(action, state, objects0),
                                 traj_fp(action.astype(np.float32), state.astype(np.float32), objects0))
                self.assertEqual(traj_fp(action, state, objects0), traj_fp(action.tolist(), state.tolist(), objects0))

    def test_quantisation_sensitivity(self):
        """§8.2 raw values quantized at 0.01: ±0.004 keeps, ±0.01 changes; objects0 at 1e-4 likewise."""
        action, state, objects0 = _traj(np.random.default_rng(0))
        fp = traj_fp(action, state, objects0)
        for factor, same in ((0.4, True), (-0.4, True), (1.0, False), (-1.0, False)):
            a2 = action.copy()
            a2[7, 3] += factor * 0.01
            s2 = state.copy()
            s2[0, 0] += factor * 0.01
            o2 = dict(objects0)
            o2["cube_0"] = (0.2001 + factor * 1e-4, 0.0503, 0.0100)
            for name, args in (("action", (a2, state, objects0)), ("state", (action, s2, objects0)),
                               ("objects0", (action, state, o2))):
                with self.subTest(field=name, factor=factor):
                    (self.assertEqual if same else self.assertNotEqual)(traj_fp(*args), fp)

    def test_negative_zero(self):
        """§8.2 -0.0 and 0.0 give the same traj_fp."""
        a = np.zeros((4, 6))
        self.assertEqual(traj_fp(a, a, {"c": (0.0, 0.0, 0.0)}), traj_fp(-a, -a, {"c": (-0.0, -0.0, 0.0)}))

    def test_structure_changes(self):
        """§8.2 shape, order of frames, swapping action/state and object names all matter."""
        action, state, objects0 = _traj(np.random.default_rng(1))
        fp = traj_fp(action, state, objects0)
        self.assertNotEqual(traj_fp(action[:-1], state[:-1], objects0), fp)
        self.assertNotEqual(traj_fp(action[::-1], state[::-1], objects0), fp)
        self.assertNotEqual(traj_fp(state, action, objects0), fp)
        self.assertNotEqual(traj_fp(action, state, {"cube_9" if n == "cube_0" else n: p for n, p in objects0.items()}), fp)
        self.assertNotEqual(traj_fp(action, state, {"cube_0": objects0["cube_0"]}), fp)
        self.assertNotEqual(traj_fp(action.reshape(10, 12), state.reshape(10, 12), objects0), fp)

    def test_objects0_order_and_pose_input(self):
        """§8.2 objects0 order is irrelevant; a Pose value contributes its position."""
        action, state, objects0 = _traj(np.random.default_rng(2))
        fp = traj_fp(action, state, objects0)
        self.assertEqual(traj_fp(action, state, dict(reversed(list(objects0.items())))), fp)
        poses = {n: Pose(tuple(p), fx.yaw_quat_xyzw(0.3)) for n, p in objects0.items()}
        self.assertEqual(traj_fp(action, state, poses), fp)
        self.assertEqual(traj_fp(action, state, {n: np.asarray(p) for n, p in objects0.items()}), fp)

    def test_bad_shapes_raise_value_error(self):
        """§1.5 mismatched or non-2-D arrays raise ValueError."""
        with self.assertRaises(ValueError):
            traj_fp(np.zeros((3, 6)), np.zeros((3, 5)), {})
        with self.assertRaises(ValueError):
            traj_fp(np.zeros(6), np.zeros(6), {})
        with self.assertRaises(ValueError):
            traj_fp(np.zeros((3, 6)), np.zeros((3, 6)), {"a": (0.0, 0.0)})


class AppearanceAndEpisodeFpTests(unittest.TestCase):
    """WP5 §8.2 appearance_fp and episode_fp."""

    def test_appearance_fields(self):
        """§8.2 colours, instruction, light (1e-3) and robot colour all enter appearance_fp; order does not."""
        colors = {"cube_0": "red", "cube_1": "blue", "box": None}
        fp = appearance_fp(colors, "Pick red", 0.800, "black")
        self.assertEqual(appearance_fp(dict(reversed(list(colors.items()))), "Pick red", 0.800, "black"), fp)
        changed = [({**colors, "cube_1": "green"}, "Pick red", 0.8, "black"),
                   ({**colors, "box": "white"}, "Pick red", 0.8, "black"),
                   ({"cube_0": "red", "cube_1": "blue"}, "Pick red", 0.8, "black"),
                   (colors, "Pick blue", 0.8, "black"), (colors, "Pick red", None, "black"),
                   (colors, "Pick red", 0.8, None), (colors, "Pick red", 0.8, "white"),
                   (colors, "Pick red", 0.801, "black")]
        for args in changed:
            with self.subTest(args=args):
                self.assertNotEqual(appearance_fp(*args), fp)
        self.assertEqual(appearance_fp(colors, "Pick red", 0.8 + 0.4e-3, "black"), fp)
        self.assertEqual(appearance_fp(colors, "Pick red", 0.8 - 0.4e-3, "black"), fp)
        self.assertEqual(appearance_fp(colors, "Pick red", np.float32(0.8), "black"), fp)

    def test_original_payload(self):
        """§8.2 originals: light = robot_color = null; no 'nominal' marker (D13)."""
        colors = {"cube_0": "red", "box": None}
        expected = json_sha256({"v": 1, "colors": colors, "instruction": "x", "light": None, "robot_color": None})
        self.assertEqual(appearance_fp(colors, "x", None, None), expected)

    def test_episode_fp_changes_with_either_part(self):
        """§8.2 episode_fp changes when traj_fp or appearance_fp changes; it is not symmetric."""
        rng = np.random.default_rng(4)
        trajs = [traj_fp(*_traj(rng)) for _ in range(3)]
        apps = [appearance_fp({"c": c}, "x", None, None) for c in ("red", "green", "blue")]
        values = {episode_fp(t, a) for t in trajs for a in apps}
        self.assertEqual(len(values), 9)
        self.assertNotEqual(episode_fp(trajs[0], apps[0]), episode_fp(apps[0], trajs[0]))

    def test_episode_fp_rejects_non_hex(self):
        """§1.5/§3.2 episode_fp takes two lower-case hex digests; anything else is a ValueError."""
        good = appearance_fp({}, "x", None, None)
        for bad in ("", "abc", good.upper(), good[:-1] + "g", None, good + "0"):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                episode_fp(bad, good)


if __name__ == "__main__":
    unittest.main()
