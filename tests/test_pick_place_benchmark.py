"""Benchmark invariants with real parquet input and CPU-only geometry."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from sim_to_real_so101.utils import pick_place_benchmark as b


def fixture(root):
    (root/'pick_place_meta').mkdir()
    (root/'data/chunk-000').mkdir(parents=True)
    xy = np.array([[.15, -.15], [.4, -.15], [.15, .15], [.4, .15], [.26, .02]])
    states = []
    for i, (x, y) in enumerate(xy):
        yaw = math.atan2(y, x)
        (root/f'pick_place_meta/episode_{i:06d}.json').write_text(json.dumps(dict(
            units='m', position=[[x, y, .01]], orientation_wxyz=[[math.cos(yaw/2), 0, 0, math.sin(yaw/2)]])))
        states.append([math.degrees(yaw)/1.1, 0, 0, 0, 0, 50])
    pq.write_table(pa.table({'episode_index': list(range(len(xy))), 'frame_index': [0]*len(xy),
                            'observation.state': states}), root/'data/chunk-000/episodes.parquet')
    scene = dict(base_position_world=[0, 0, 0], base_quaternion_wxyz=[1, 0, 0, 0],
                 table_plane_world=[0, 0, 0], cube_size_m=[.02]*3,
                 box_position_world=[.35, .1, 0], box_quaternion_wxyz=[1, 0, 0, 0], box_size_m=[.1,.08,.05],
                 table_outline_base_xy=[[0,-.3],[.6,-.3],[.6,.3],[0,.3]],
                 box_footprint_base_xy=[[.3,.06],[.4,.06],[.4,.14],[.3,.14]])
    return xy, states, scene


class GenerationTests(unittest.TestCase):
    def test_geometry_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xy, states, scene = fixture(root)
            data = b.generate_eval_set(root, scene, seed=17)
            again = b.generate_eval_set(root, scene, seed=17)
        self.assertEqual(data['starts'], again['starts'])
        self.assertEqual(data['eval_set_id'], again['eval_set_id'])
        polygon = np.array(data['training_region']['hull_xy'])
        self.assertEqual({s: sum(r['split'] == s for r in data['starts']) for s in b.SPLITS},
                         dict(id=60, ood=20, yaw=20, train=5))
        for row in data['starts']:
            np.testing.assert_allclose(row['arm_start_state'], np.mean(states, axis=0))
            if row['split'] == 'train':
                continue
            point = np.array([row['x'],row['y']])
            self.assertGreaterEqual(np.linalg.norm(xy-point,axis=1).min(), .02)
            self.assertTrue(b.clear_of_box(point,row['yaw'],scene))
            self.assertTrue(b.supported_on_table(point,row['yaw'],scene))
            if row['split'] == 'ood':
                self.assertFalse(b.inside_polygon(point,polygon))
                self.assertGreaterEqual(b.hull_distance(point,polygon),.02)
                self.assertLessEqual(b.hull_distance(point,polygon),.05)
                self.assertEqual(row['zone'],'OUT')
                region=data['training_region']
                pan=(math.atan2(point[1],point[0])-region['pan_centre_rad']+math.pi)%(2*math.pi)-math.pi
                self.assertTrue(region['pan_limits_rad'][0] <= pan <= region['pan_limits_rad'][1])
            else:
                self.assertTrue(b.inside_polygon(point,polygon))
            if row['split']=='yaw':
                parent=next(r for r in data['starts'] if r['id']==row['paired_id'])
                self.assertEqual((row['x'],row['y']),(parent['x'],parent['y']))
                self.assertTrue(0 <= row['yaw'] < math.pi/2)

    def test_yaw_rules_and_wrap_boundaries(self):
        np.testing.assert_allclose(np.degrees(b.wrap_yaw(np.radians([-135,-45,45,135,44.999,-45.001]))),
                                   [-45,-45,-45,-45,44.999,44.999],atol=1e-10)
        xy=np.array([[.2,.1],[-.2,.1],[-.1,-.3]])
        yaw=np.radians([3,89,42])
        relative=b.wrap_yaw(yaw-np.arctan2(xy[:,1],xy[:,0]))
        for i,p in enumerate(xy):
            actual, rel, source=b.sample_yaw(p,xy,relative,'matched',np.random.default_rng(3))
            self.assertAlmostEqual(float(b.wrap_yaw(actual-yaw[i])),0)
            self.assertAlmostEqual(rel,relative[i])
            self.assertEqual(source,i)
            actual, rel, source=b.sample_yaw(p,xy,relative,'radial',np.random.default_rng(3))
            self.assertAlmostEqual(rel,0)
            self.assertIsNone(source)
        for mode in ('matched','radial','random'):
            self.assertEqual(b.sample_yaw([.24,-.12],xy,relative,mode,np.random.default_rng(8)),
                             b.sample_yaw([.24,-.12],xy,relative,mode,np.random.default_rng(8)))

    def test_yaw_metadata_and_always_random_yaw_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); xy,_,scene=fixture(root)
            for mode in ('matched','radial','random'):
                data=b.generate_eval_set(root,scene,yaw_mode=mode,n_id=4,n_yaw=3,n_ood=2)
                self.assertEqual(data['parameters']['yaw_mode'],mode)
                self.assertEqual(set(data['training_region']['rel_yaw_distribution']),{'mean','std','min','max'})
                for row in data['starts']:
                    if row['split']=='train':
                        self.assertEqual(row['source_start_id'],row['id'])
                    elif row['split']=='yaw' or mode=='random':
                        self.assertIsNone(row['source_start_id'])
                        self.assertTrue(0 <= row['yaw'] < math.pi/2)
                    elif mode=='radial':
                        self.assertAlmostEqual(row['rel_yaw'],0)
                        self.assertIsNone(row['source_start_id'])
                    else:
                        source=next(x for x in data['starts'] if x['id']==row['source_start_id'])
                        self.assertAlmostEqual(row['rel_yaw'],source['rel_yaw'])

    def test_hull_not_bounding_box(self):
        polygon=b.hull([[0,0],[1,0],[0,1],[.2,.2]])
        self.assertFalse(b.inside_polygon([.9,.9],polygon))
        self.assertTrue(b.inside_polygon([.2,.2],polygon))
        self.assertAlmostEqual(b.hull_distance(np.array([1,1]),polygon),math.sqrt(.5))
        with self.assertRaises(ValueError): b.hull([[0,0],[1,0],[2,0]])

    def test_invalid_inputs_and_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,_,scene=fixture(root)
            for opts in ({'n_id':1,'n_yaw':2},{'min_dist_m':float('nan')},{'ood_margin_m':[.05,.02]}):
                with self.assertRaises(ValueError): b.generate_eval_set(root,scene,**opts)
            data=b.generate_eval_set(root,scene,seed=2,n_id=2,n_ood=1,n_yaw=1)
            path=root/'set.json'; path.write_text(json.dumps(data))
            schedule=b.EvalSetStarts(path,'id,yaw',2,42)
            rows=[]
            for _ in range(6):
                idx,p,q,z=schedule.sample()
                rows.append(dict(schedule.current))
                np.testing.assert_allclose(schedule.robot_state(idx), rows[-1]['arm_start_state'])
            self.assertEqual([r['id'] for r in rows[:3]], [r['id'] for r in rows[3:]])
            self.assertEqual(len({r['policy_seed'] for r in rows}),6)
            schedule.sample(); self.assertEqual(schedule.current, rows[-1])
            for splits in ('oops','id,id'):
                with self.assertRaises(ValueError): b.EvalSetStarts(path,splits)
            data['starts'][0]['x']+=.01
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'hash'): b.load_eval_set(path)


class StatisticsTests(unittest.TestCase):
    def test_wilson_known_values(self):
        np.testing.assert_allclose(b.wilson_interval(15,20),[.531299122381256,.888138298592334],atol=1e-12)
        self.assertEqual(b.wilson_interval(0,0),[None,None])
        self.assertAlmostEqual(b.wilson_interval(0,10)[1],.2775327998628892)
        self.assertAlmostEqual(b.wilson_interval(10,10)[0],.7224672001371107)

    def test_mcnemar_exact(self):
        self.assertEqual(b.mcnemar([True]*6,[False]*6)['p_value'],.03125)
        self.assertEqual(b.mcnemar([True,False],[False,True])['p_value'],1.)
        self.assertEqual(b.mcnemar([True],[True])['p_value'],1.)
        from scipy.stats import binomtest
        for n in range(1,25):
            for k in range(n+1):
                a=[True]*k+[False]*(n-k); c=[not x for x in a]
                self.assertAlmostEqual(b.mcnemar(a,c)['p_value'],binomtest(k,n,.5).pvalue)

    def test_synthetic_stage_traces(self):
        def frame(trace,**kw):
            data=dict(distance=.1,grasped=False,lift=0,min_lift=.01,held=False,inside_box=False,placed=False)
            data.update(kw); trace.observe(**data)
        for mode in b.FAILURES:
            t=b.StageTrace(); frame(t)
            if mode!='never_reached': frame(t,distance=.03)
            if mode not in ('never_reached','missed_grasp'):
                frame(t,distance=.02,grasped=True,lift=.01,held=True)
                self.assertTrue(t.stages['lifted'])
                if mode in ('dropped_outside','placed_not_confirmed'): frame(t,held=True,inside_box=True)
                if mode!='timeout_holding': frame(t,placed=mode=='placed_not_confirmed')
            self.assertEqual(t.finish(False)['failure_mode'],mode)
            self.assertIsNone(t.finish(True)['failure_mode'])
            self.assertTrue(t.finish(True)['stages']['success'])

    def test_report_heatmap_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); _,_,scene=fixture(root)
            data=b.generate_eval_set(root,scene,n_id=2,n_yaw=1,n_ood=1)
            rows=[]
            for repeat in range(2):
                for i,s in enumerate(data['starts']):
                    success=(i+repeat)%2==0
                    rows.append(dict(start_id=s['id'],split=s['split'],zone=s['zone'],start_kind='recorded',cube_color='blue',
                                     repeat=repeat,policy_seed=repeat*100+i,success=success,success_step=30 if success else None,
                                     stages={k:success for k in b.STAGES}, failure_mode=None if success else 'never_reached'))
            report=b.benchmark_report(rows,data,2,task='test',checkpoint='A',seed=0,step_dt=1/30)
            report.update(complete=True,robot_start='recorded')
            self.assertEqual(report['schema_version'],2)
            self.assertEqual(report['overall']['success_rate'],.5)
            self.assertEqual(report['overall']['median_time_to_success_s'],1.)
            self.assertTrue(all(x['success_fraction']==.5 for x in report['per_start_consistency'].values()))
            self.assertEqual(set(report['by_split']),set(b.SPLITS))
            json.dumps(report,allow_nan=False)
            path=root/'heatmap.png'; b.save_heatmap(report,path)
            self.assertEqual(path.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
            self.assertGreater(path.stat().st_size,1000)
            spec=importlib.util.spec_from_file_location('comparison',ROOT/'scripts/compare_eval_results.py')
            comparison=importlib.util.module_from_spec(spec); spec.loader.exec_module(comparison)
            pairs=comparison.compare([report,report])
            self.assertEqual(pairs[0]['shared_starts'],len(data['starts']))
            self.assertEqual(pairs[0]['p_value'],1)
            with self.assertRaises(ValueError): comparison.compare([report,dict(report,eval_set_id='other')])



class RuntimeTests(unittest.TestCase):
    def test_stage_hook_observes_terminal_frame_and_preserves_success(self):
        import torch
        from types import SimpleNamespace as NS
        from sim_to_real_so101.utils.pick_place_benchmark_runtime import install_stage_tracking
        cube=NS(data=NS(root_pos_w=torch.tensor([[0.,0.,.03]])))
        box=NS(data=NS(root_pos_w=torch.tensor([[0.,0.,0.]]),root_quat_w=torch.tensor([[1.,0.,0.,0.]])))
        ee=NS(data=NS(target_pos_w=torch.tensor([[[0.,0.,.04]]])))
        result=torch.tensor([True]); calls=[]
        def success(env,**params):
            calls.append('original')
            return result
        params=dict(contact_sensor_cfg=NS(name='contact_grasp'),object_name='cube',container_name='box',
                    container_size=(.1,.1,.05),container_floor_z=.001,container_inner_half_size=(.04,.04),
                    min_lift=.01,warmup_steps=30,force_threshold=2,confirm_steps=25)
        cfg=NS(func=success,params=params)
        manager=NS(get_term_cfg=lambda name:cfg,set_term_cfg=lambda name,c:None)
        raw=NS(termination_manager=manager,scene={'cube':cube,'box':box,'ee_frame':ee},
               _pick_place_rest_z=torch.tensor([.01]),_pick_place_holding=torch.tensor([True]))
        def grasp(env,**kw):
            self.assertNotIn('confirm_steps',kw);self.assertNotIn('container_name',kw)
            return torch.tensor([[1.]])
        terms=NS(object_grasped=grasp,object_placed_in_container=lambda env,**kw:torch.tensor([[1.]]))
        with patch.dict(sys.modules,{'sim_to_real_so101.mdp.terms':terms}):
            install_stage_tracking(raw)
            self.assertIs(cfg.func(raw,**params),result)
        self.assertEqual(calls,['original'])
        self.assertTrue(all(raw._benchmark_trace.stages.values()))
        # Auto-reset destroys task trackers, not the completed benchmark trace.
        raw._pick_place_holding[:]=False
        raw._pick_place_rest_z[:]=0
        self.assertTrue(raw._benchmark_trace.finish(True)['stages']['placed'])

    def test_real_runner_schedule_repeat_seeding_and_final_autoreset(self):
        import test_lerobot_eval_rollout as rollout
        from sim_to_real_so101.utils import pick_place_benchmark_runtime as runtime
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,scene=fixture(root)
            data=b.generate_eval_set(root,scene,n_id=1,n_ood=0,n_yaw=1)
            path=root/'set.json';path.write_text(json.dumps(data))
            sampler=b.EvalSetStarts(path,'id,yaw',2,1984)
            class Environment(rollout.FakeEnvironment):
                def metadata(self):
                    super().metadata()
                    sampler.sample()
                    self._benchmark_start=dict(sampler.current)
                def step(self,actions):
                    self._benchmark_trace.observe(distance=.01,grasped=True,lift=.02,min_lift=.01,
                                                  held=False,inside_box=True,placed=True)
                    return super().step(actions)
            class Policy(rollout.FakePolicy):
                seeds=[]
                def reset(self,seed=None):
                    self.seeds.append(seed)
                    super().reset()
            def install(env): env._benchmark_trace=b.StageTrace()
            harness=rollout.EvaluatorRolloutTests()
            with patch.object(rollout,'FakeEnvironment',Environment), patch.object(rollout,'FakePolicy',Policy), \
                 patch.object(runtime,'install_stage_tracking',install),patch.object(runtime,'capture_scene',lambda env:scene):
                env,cfg=harness.evaluate(root/'results.json',args_override=dict(eval_set=path,splits='id,yaw',repeats=2),
                                         outcomes=[(1,True,False),(1,False,True)]*2)
            report=json.loads((root/'results.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['requested_episodes'],4)
            self.assertEqual(env.explicit_resets,1)
            self.assertEqual([(r['start_id'],r['repeat']) for r in report['episodes']],
                             [('id-000',0),('yaw-000',0),('id-000',1),('yaw-000',1)])
            self.assertEqual(Policy.seeds,[r['policy_seed'] for r in report['episodes']])
            self.assertEqual(len(set(Policy.seeds)),4)
            self.assertEqual(report['overall']['successes'],2)
            self.assertEqual(report['overall']['failure_modes']['placed_not_confirmed'],2)
            self.assertTrue((root/'results.png').is_file())

    def test_seed_endpoint_reseeds_actual_rngs(self):
        import torch, random
        spec=importlib.util.spec_from_file_location('seed_server',ROOT/'docker/real/scripts/benchmark_server.py')
        server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
        with patch.object(torch.cuda,'manual_seed_all'):
            self.assertEqual(server.seed_policy(123),{'seed':123})
            first=(random.random(),np.random.random(),torch.rand(1).item())
            server.seed_policy(123)
            second=(random.random(),np.random.random(),torch.rand(1).item())
            self.assertEqual(first,second)
            server.seed_policy(124)
            self.assertNotEqual(first,(random.random(),np.random.random(),torch.rand(1).item()))
        for bad in (-1,2**32,True,'123'):
            with self.assertRaises(ValueError):server.seed_policy(bad)

    def test_host_passes_benchmark_flags_and_selects_seed_server(self):
        import test_eval_pick_place_script as host
        harness=host.HostScriptTests();harness.setUp()
        try:
            result=harness.run_script('--eval_set','/workspace/set.json','--splits','id,ood',
                                      '--repeats','3','--robot_start','default','--dr')
            self.assertEqual(result.returncode,0,result.stderr)
            calls=harness.calls()
            server=next(c for c in calls if c[0]=='run')
            self.assertIn('/Isaac-GR00T/gr00t/eval/real_robot/SO100/benchmark_server.py',server)
            evaluation=next(c for c in calls if 'sim_to_real_so101.scripts.lerobot_eval' in c)
            for key,value in (('--eval_set','/workspace/set.json'),('--splits','id,ood'),('--repeats','3'),('--robot_start','default')):
                self.assertEqual(evaluation[evaluation.index(key)+1],value)
            self.assertIn('Lerobot-So101-Teleop-Pick-Place-DR-Eval',evaluation)
        finally:
            harness.doCleanups()


if __name__ == '__main__':
    unittest.main()
