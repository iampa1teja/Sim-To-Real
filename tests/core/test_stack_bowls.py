import itertools
import json

import numpy as np
import pytest
import torch
from core.tasks.loader import load_task, load_task_module


def test_all_six_color_orders_are_accepted_geometrically():
    task = load_task('stack_bowls')
    geometry = load_task_module('geometry', task.name)
    p = torch.tensor([[[.15,-.1,.1], [.15,-.1,.122], [.15,-.1,.144]]])
    q = torch.tensor([[[0.,0.,0.,1.]]*3])
    v = torch.zeros((1,3,6))
    for permutation in itertools.permutations(range(3)):
        paired, upright, stable = geometry.nesting_metrics(p[:,permutation], q, v, task.spec.raw['success'])
        assert paired.all() and upright.all() and stable.all()
    assert task.spec.raw['success']['color_order'] == 'any'


@pytest.mark.parametrize('case', ['separate', 'overlap', 'upside_down', 'moving'])
def test_invalid_stacks_are_rejected(case):
    task = load_task('stack_bowls')
    geometry = load_task_module('geometry', task.name)
    p = torch.tensor([[[.15,-.1,.1], [.15,-.1,.122], [.15,-.1,.144]]])
    q = torch.tensor([[[0.,0.,0.,1.]]*3])
    v = torch.zeros((1,3,6))
    if case == 'separate': p[0,2,0] += .1
    if case == 'overlap': p[0,2,2] = .122
    if case == 'upside_down': q[0,1] = torch.tensor([1.,0.,0.,0.])
    if case == 'moving': v[0,2,0] = .2
    paired, upright, stable = geometry.nesting_metrics(p,q,v,task.spec.raw['success'])
    assert not (paired.all() and upright.all() and stable.all())


def test_sampled_starts_separate_all_three_original_size_bowls():
    task = load_task('stack_bowls')
    for seed in range(100):
        starts = task.sample_start(np.random.default_rng(seed))
        assert set(starts) == set(task.spec.tracked_objects)
        for a,b in itertools.combinations(starts.values(),2):
            assert np.linalg.norm(np.array(a.position[:2])-b.position[:2]) >= task.spec.start['minimum_separation_m']


def test_bowl_colliders_preserve_open_interior():
    from pxr import Usd, UsdGeom, UsdPhysics
    from scipy.spatial import ConvexHull
    task = load_task('stack_bowls')
    stage = Usd.Stage.Open(str(task.spec.objects['yellow_bowl'].usd))
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) > 24
    def occupied(point):
        for prim in colliders:
            vertices = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            hull = ConvexHull(vertices)
            if np.all(hull.equations[:,:3] @ point + hull.equations[:,3] <= 1e-7):
                return True
        return False
    assert not occupied(np.array([0.,0.,0.]))
    assert not occupied(np.array([0.,0.,.024]))
    assert occupied(np.array([0.,0.,-.024]))
    provenance = json.loads((task.spec.task_dir/'assets/provenance.json').read_text())
    assert provenance['source_declared_meters_per_unit'] == 1.
    assert np.isclose(provenance['scale_to_metres'], .001)
    assert .0019 < provenance['floor_thickness_m'] < .0021
    assert np.allclose(provenance['size_m'], [.12,.12,.05], atol=.0001)


def test_first_placement_ignores_unstacked_third_bowl_height():
    task = load_task('stack_bowls')
    geometry = load_task_module('geometry', task.name)
    # A sloping table can put the third bowl between the first two in Z.
    p = torch.tensor([[[.1,-.1,.100], [.1,-.1,.117], [.3,-.2,.103]]])
    assert geometry.any_nested_pair(p,task.spec.raw['success']).item()


def test_rim_grasps_follow_pan_instead_of_a_fixed_world_direction():
    from core.tasks.base import Pose, StepState
    from core.tasks.expert import ExpertContext
    from utils.arms.arms import load_arm_spec
    from utils.geometry import rotation_matrix_xyzw
    task = load_task('stack_bowls')
    objects = dict(yellow_bowl=Pose((.18,-.12,.05)), red_bowl=Pose((-.05,-.22,.05)),
                   blue_bowl=Pose((-.22,-.14,.05)))
    arm = load_arm_spec('so101')[0]
    context = ExpertContext(StepState(objects,Pose((0,0,.3)),0.), Pose((0,0,.3)), (0.,0.,1.),
                            arm_base=Pose((0.,0.,0.)), arm=arm)
    plan = task.expert().plan(context)
    directions = []
    for name in ['red_bowl','blue_bowl']:
        grasp = next(p for p in plan if p.name == name+'_grasp')
        lift = next(p for p in plan if p.name == name+'_lift')
        transport = next(p for p in plan if p.name == name+'_transport')
        assert transport.pose.position[2] > max(p.position[2] for p in objects.values())+.05
        centre = np.asarray(objects[name].position)
        offset = np.asarray(grasp.pose.position)-centre
        rotation = rotation_matrix_xyzw(grasp.pose.quat_xyzw)
        closing = rotation[:,0]
        directions.append(closing)
        assert np.isclose(np.linalg.norm(offset[:2]),.058)
        assert np.isclose(offset[2],.012)
        assert np.isclose(np.dot(np.asarray(grasp.pose.position)[:2],closing[:2]),0.,atol=1e-8)
        assert np.isclose(rotation[2,2],np.cos(np.deg2rad(21.)))
    assert not np.allclose(*directions)
    heights = [p.pose.position[2] for p in plan if p.name.endswith('_transport')]
    assert np.ptp(heights) < 1e-9


def test_replans_next_pickup_after_the_support_bowl_moves():
    from dataclasses import replace
    from core.tasks.base import Pose, StepState
    from core.tasks.expert import ExpertContext
    from utils.arms.arms import load_arm_spec
    task = load_task('stack_bowls')
    objects = dict(yellow_bowl=Pose((.185,-.24,.05)), red_bowl=Pose((0.,-.24,.05)),
                   blue_bowl=Pose((-.185,-.24,.05)))
    context = ExpertContext(StepState(objects,Pose((0,0,.3)),0.), Pose((0,0,.3)), (0.,0.,1.),
        {'order':['red_bowl','yellow_bowl','blue_bowl']}, Pose((0.,0.,0.)), load_arm_spec('so101')[0])
    expert = task.expert()
    plan = expert.plan(context)
    lower = next(p for p in plan if p.name=='blue_bowl_lower')
    approach = next(p for p in plan if p.name=='blue_bowl_approach')
    moved = {**objects, 'red_bowl':Pose((.02,-.23,.05))}
    current = replace(context,state=replace(context.state,objects=moved))
    expert.adapt_waypoint(approach,current)
    adapted = expert.adapt_waypoint(lower,current)
    np.testing.assert_allclose(np.array(adapted.pose.position)-lower.pose.position,[.02,.01,0.])


def test_bowl_orientation_feedback_reduces_tilt_and_freezes_on_release():
    from core.tasks.base import Pose, StepState
    from core.tasks.expert import ExpertContext
    from utils.arms.arms import load_arm_spec
    from utils.geometry import rotation_matrix_xyzw
    from scipy.spatial.transform import Rotation
    task = load_task('stack_bowls')
    tilted = tuple(Rotation.from_euler('x',17,degrees=True).as_quat())
    objects = dict(yellow_bowl=Pose((.185,-.24,.05)), red_bowl=Pose((0.,-.24,.05)),
                   blue_bowl=Pose((-.185,-.24,.05),tilted))
    context = ExpertContext(StepState(objects,Pose((0,0,.3)),0.,{'holding':True}),Pose((0,0,.3)),
        (0.,0.,1.),{},Pose((0.,0.,0.)),load_arm_spec('so101')[0])
    expert = task.expert();plan = expert.plan(context)
    waypoint = next(p for p in plan if p.name=='blue_bowl_transport')
    result = expert.control_orientation(waypoint,context.state,context.control_pose,[0.,0.,0.,1.],context,1.)
    original_up = rotation_matrix_xyzw(tilted)[:,2]
    corrected_up = rotation_matrix_xyzw(result) @ original_up
    assert corrected_up[2] > original_up[2]
    context.state.signals['holding'] = False
    frozen = expert.control_orientation(waypoint,context.state,context.control_pose,[0.,0.,1.,0.],context,1.)
    np.testing.assert_allclose(frozen,result)


def test_active_bowl_preserves_the_supplied_usd_mesh_and_normals():
    import hashlib
    from pxr import Usd, UsdGeom, UsdLux
    task = load_task('stack_bowls')
    root = task.spec.task_dir/'assets'
    provenance = json.loads((root/'provenance.json').read_text())
    source = task.spec.task_dir/provenance['source_copy']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance['source_sha256']
    supplied = Usd.Stage.Open(str(source))
    active = Usd.Stage.Open(str(task.spec.objects['yellow_bowl'].usd))
    assert task.spec.objects['yellow_bowl'].usd.name == 'bowl.usd'
    original = [UsdGeom.Mesh(p) for p in supplied.Traverse() if p.IsA(UsdGeom.Mesh)]
    visual = [UsdGeom.Mesh(p) for p in Usd.PrimRange(active.GetPrimAtPath('/Bowl/Visual'))
              if p.IsA(UsdGeom.Mesh)]
    assert len(original) == len(visual) == 1
    for left, right in zip(original, visual):
        np.testing.assert_array_equal(left.GetPointsAttr().Get(), right.GetPointsAttr().Get())
        np.testing.assert_array_equal(left.GetFaceVertexIndicesAttr().Get(), right.GetFaceVertexIndicesAttr().Get())
        np.testing.assert_array_equal(left.GetNormalsAttr().Get(), right.GetNormalsAttr().Get())
        assert left.GetNormalsInterpolation() == right.GetNormalsInterpolation() == 'faceVarying'
    assert provenance['visual_triangles'] == provenance['source_triangles'] == 412122
    assert not any(p.HasAPI(UsdLux.LightAPI) or p.IsA(UsdGeom.Camera) for p in active.Traverse())
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        active.GetPrimAtPath('/Bowl/Visual')).ComputeAlignedRange()
    low, high = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
    np.testing.assert_allclose((low+high)/2, [0.,0.,0.], atol=1e-6)
    np.testing.assert_allclose(high-low, [.12,.12,.05], atol=.0001)


def test_nested_stack_allows_cumulative_lean_but_requires_adjacent_alignment():
    from scipy.spatial.transform import Rotation
    task = load_task('stack_bowls')
    geometry = load_task_module('geometry',task.name)
    p = torch.tensor([[[.15,-.1,.1],[.15,-.1,.118],[.15,-.1,.136]]])
    q = torch.tensor(Rotation.from_euler('x',[[0.],[10.],[20.]],degrees=True).as_quat()[None])
    velocity = torch.zeros((1,3,6))
    paired,upright,stable = geometry.nesting_metrics(p,q,velocity,task.spec.raw['success'])
    assert paired.all() and upright.all() and stable.all()
    q[0,1] = torch.tensor(Rotation.from_euler('x',28.,degrees=True).as_quat())
    paired,upright,stable = geometry.nesting_metrics(p,q,velocity,task.spec.raw['success'])
    assert not paired.all()
