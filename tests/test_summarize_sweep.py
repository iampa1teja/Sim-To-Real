"""CPU checks for sweep aggregation, comparisons, artifacts and schedule validation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from sim_to_real_so101.utils import pick_place_benchmark as b
from sim_to_real_so101.scripts.summarize_sweep import complete_report, paired, summarize_folder
spec=importlib.util.spec_from_file_location('sweep',ROOT/'scripts/sweep_action_horizon.py')
sweep=importlib.util.module_from_spec(spec);spec.loader.exec_module(sweep)


def report(horizon, outcomes, repeats=1):
    data=json.loads((ROOT/'eval_sets/pick_place_v1_seed1984.json').read_text())
    data['starts']=[data['starts'][0],data['starts'][1],next(s for s in data['starts'] if s['split']=='ood'),next(s for s in data['starts'] if s['split']=='yaw')]
    data['eval_set_id']=b.eval_set_digest(data)
    rows=[]
    for repeat in range(repeats):
        for i,(s,success) in enumerate(zip(data['starts'],outcomes)):
            rows.append(dict(start_id=s['id'],split=s['split'],repeat=repeat,policy_seed=i+repeat*10,
                start_kind='recorded',zone=s['zone'],cube_color='blue',start_index=i,success=success,
                success_step=10+i*10 if success else None,steps=10+i*10 if success else 200,
                stages={stage:success for stage in b.STAGES},failure_mode=None if success else 'never_reached',
                inference_calls=10+i,wall_time_s=20+i))
    r=b.benchmark_report(rows,data,repeats,task='test',checkpoint='model/checkpoint-1',seed=1984,
                         action_horizon=horizon,step_dt=.1,episode_length_s=20)
    r.update(complete=True,requested_episodes=len(rows),robot_start='recorded')
    return r


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)

    def setup_folder(self, reports, horizons=None):
        first=next(iter(reports.values()))
        expected=[(r['start_id'],r['repeat']) for r in first['episodes']]
        config=dict(horizons=horizons or list(reports),selected_splits=['id','ood','yaw'],
                    expected_schedule=expected,eval_set_id=first['eval_set_id'])
        (self.folder/'sweep_config.json').write_text(json.dumps(config))
        for h,r in reports.items():
            run=self.folder/f'ah_{h}';run.mkdir();(run/'results.json').write_text(json.dumps(r))
        return expected

    def test_aggregation_ci_and_plots(self):
        self.setup_folder({8:report(8,[True,False,True,False])})
        result=summarize_folder(self.folder)
        row=next(r for r in result['rows'] if r['split']=='all')
        self.assertEqual((row['episodes'],row['successes'],row['never_reached']),(4,2,2))
        self.assertEqual(row['success_rate'],.5)
        self.assertAlmostEqual(row['ci95_low'],.15003898915214947)
        self.assertAlmostEqual(row['ci95_high'],.8499610108478506)
        self.assertEqual(row['median_time_to_success_s'],2)
        self.assertEqual(row['mean_time_to_success_s'],2)
        self.assertEqual(row['success_within_15s'],.5)
        self.assertEqual(row['mean_inference_calls_per_episode'],11.5)
        self.assertEqual(row['mean_wall_time_per_episode_s'],21.5)
        for name in ('success_vs_horizon.png','stages_vs_horizon.png','time_to_success_vs_horizon.png','failure_modes_vs_horizon.png'):
            self.assertGreater((self.folder/'plots'/name).stat().st_size,1000)
        self.assertTrue((self.folder/'summary.csv').is_file());self.assertTrue((self.folder/'summary.md').is_file())

    def test_best_rate_then_id_then_time(self):
        a=report(2,[True,True,False,False]);c=report(16,[False,False,True,True]);d=report(8,[True,True,False,False])
        for r in d['episodes']:
            if r['success']:r['success_step']=5
        self.setup_folder({2:a,16:c,8:d})
        result=summarize_folder(self.folder)
        self.assertEqual(result['best_horizon'],8)
        self.assertEqual(result['paired_comparisons']['runner_up']['second'],2)
        self.assertFalse(result['paired_comparisons']['horizon_16']['statistically_better'])

    def test_id_near_tie_break_when_present(self):
        a=report(2,[True,True,False,False]);c=report(16,[False,False,True,True])
        a['by_split']['id_near']={'success_rate':.1};c['by_split']['id_near']={'success_rate':.9}
        self.setup_folder({2:a,16:c})
        self.assertEqual(summarize_folder(self.folder)['best_horizon'],16)

    def test_higher_all_rate_wins(self):
        self.setup_folder({2:report(2,[True,True,False,False]),16:report(16,[True,True,True,False])})
        self.assertEqual(summarize_folder(self.folder)['best_horizon'],16)

    def test_mcnemar_shared_starts_repeat_zero(self):
        a=report(2,[True,False,True,False],2);c=report(16,[False,True,True,False],2)
        c['episodes']=[r for r in c['episodes'] if r['start_id']!=c['eval_set']['starts'][3]['id']]
        test=next(p for p in paired(a,c) if p['split']=='all')
        self.assertEqual((test['shared_starts'],test['first_only'],test['second_only']),(3,1,1))
        self.assertEqual(test['p_value'],1)
        c['episodes'][0]['policy_seed']=99
        with self.assertRaisesRegex(ValueError,'policy seeds'):paired(a,c)

    def test_resume_count_identity_and_schedule(self):
        r=report(8,[True]*4,2);expected=[(e['start_id'],e['repeat']) for e in r['episodes']]
        self.assertTrue(complete_report(r,expected,r['eval_set_id'],8))
        self.assertFalse(complete_report(r,expected,r['eval_set_id'],16))
        self.assertFalse(complete_report(r,expected,'different',8))
        r['episodes'][-1]=r['episodes'][0]
        self.assertFalse(complete_report(r,expected))
        r=report(8,[True]*4);r['complete']=False
        self.assertFalse(complete_report(r,expected[:4]))

    def test_failed_horizon_and_no_false_winner(self):
        self.setup_folder({8:report(8,[True]*4)},[8,16])
        result=summarize_folder(self.folder)
        failed=next(r for r in result['rows'] if r['action_horizon']==16 and r['split']=='all')
        self.assertEqual(failed['status'],'failed');self.assertEqual(failed['episodes'],0)
        self.assertIsNone(failed['success_rate']);self.assertEqual(result['best_horizon'],8)
        self.assertFalse(result['paired_comparisons']['horizon_16']['available'])
        (self.folder/'ah_8/status.json').write_text('{"status":"failed"}')
        self.assertIsNone(summarize_folder(self.folder)['best_horizon'])

    def test_consistency_only_with_repeats_and_missing_telemetry(self):
        self.setup_folder({8:report(8,[True,False,True,False],2)})
        row=next(r for r in summarize_folder(self.folder)['rows'] if r['split']=='all')
        self.assertEqual(sorted(row['per_start_consistency'].values()),[0,0,1,1])
        r=report(8,[True]*4)
        for e in r['episodes']:e.pop('inference_calls');e.pop('wall_time_s')
        (self.folder/'ah_8/results.json').write_text(json.dumps(r))
        config=json.loads((self.folder/'sweep_config.json').read_text());config['expected_schedule']=config['expected_schedule'][:4]
        (self.folder/'sweep_config.json').write_text(json.dumps(config))
        row=next(r for r in summarize_folder(self.folder)['rows'] if r['split']=='all')
        self.assertNotIn('per_start_consistency',row);self.assertIsNone(row['mean_inference_calls_per_episode'])

    def test_horizon_parsing(self):
        self.assertEqual(sweep.parse_horizons('1,2,4'),[1,2,4]);self.assertEqual(sweep.parse_horizons('1-16'),list(range(1,17)))
        self.assertEqual(sweep.parse_horizons('1-2,2,4'),[1,2,4])
        for value in ('0','17','3-1','1.5','1,,2',''):
            with self.assertRaises(ValueError):sweep.parse_horizons(value)
        with self.assertRaises(ValueError):sweep.parse_horizons('8',4)

    def test_processor_horizon_over_padded_capacity(self):
        (self.folder/'config.json').write_text('{"action_horizon":40}')
        (self.folder/'processor_config.json').write_text(json.dumps(dict(processor_kwargs=dict(modality_configs=dict(new_embodiment=dict(action=dict(delta_indices=list(range(16)))))))))
        self.assertEqual(sweep.chunk_length(self.folder,'NEW_EMBODIMENT'),16)



class OrchestrationTests(unittest.TestCase):
    def test_one_server_failure_continues_resume_and_cleanup(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'docker').mkdir();bin_dir=root/'bin';bin_dir.mkdir()
            model_dir=root/'models/model/checkpoint-1';model_dir.mkdir(parents=True);(model_dir/'config.json').write_text('{}')
            r=report(16,[True,False,True,False]);r['task']='Lerobot-So101-Teleop-Pick-Place-Eval'
            r['checkpoint']='model/checkpoint-1';r['eval_set']['dataset_path']=str(root/'dataset')
            r['eval_set']['eval_set_id']=b.eval_set_digest(r['eval_set']);r['eval_set_id']=r['eval_set']['eval_set_id']
            fixture_path=root/'fixture.json';fixture_path.write_text(json.dumps(r))
            eval_path=root/'eval.json';eval_path.write_text(json.dumps(r['eval_set']))
            call_path=root/'calls.txt'
            fake=root/'docker/eval_pick_place.sh'
            fake.write_text('#!'+sys.executable+'\n'+f'''
import json,sys,time,signal
from pathlib import Path
args=sys.argv[1:]
p=Path({str(call_path)!r})
with p.open('a') as f:f.write(' '.join(args)+'\\n')
if '--server_only' in args:
    def stopped(*a):
        with p.open('a') as f:f.write('STOPPED\\n')
        sys.exit(0)
    signal.signal(signal.SIGTERM,stopped)
    print('SERVER_READY',flush=True)
    while True:time.sleep(.1)
h=int(args[args.index('--action_horizon')+1])
if h==2:sys.exit(42)
r=json.loads(Path({str(fixture_path)!r}).read_text());r['action_horizon']=h
Path(args[args.index('--results_json')+1]).write_text(json.dumps(r))
print('[EPISODE 4]',flush=True)
''');fake.chmod(0o755)
            for name,body in [('nvidia-smi',"import sys;print('Mock GPU' if '--query-gpu=name' in sys.argv else '',end='')"),('git',"print('abc123')"),('docker',f"import json;print(json.dumps([{{'Mounts':[{{'Source':{str(root)!r},'Destination':{str(root)!r}}}]}}]))")]:
                script=bin_dir/name;script.write_text('#!'+sys.executable+'\n'+body+'\n');script.chmod(0o755)
            args=['--model','model/checkpoint-1','--models_dir',str(root/'models'),'--eval_set',str(eval_path),'--horizons','2,16','--out_root',str(root/'outputs')]
            with patch.object(sweep,'ROOT',root),patch.dict(os.environ,PATH=str(bin_dir)+':'+os.environ['PATH']):
                self.assertEqual(sweep.main(args),1)
                folder=next((root/'outputs').iterdir());summary=json.loads((folder/'summary.json').read_text())
                self.assertEqual(summary['best_horizon'],16)
                calls=call_path.read_text();self.assertEqual(calls.count('--server_only'),1);self.assertIn('STOPPED',calls)
                self.assertIn('"exit_code": 42',(folder/'ah_2/status.json').read_text())
                self.assertEqual(sweep.main(args+['--resume']),1)
                calls=call_path.read_text();self.assertEqual(calls.count('--action_horizon 16'),1)
                self.assertEqual(calls.count('--server_only'),2)


class ServerModeTests(unittest.TestCase):
    def setUp(self):
        from test_eval_pick_place_script import HostScriptTests
        HostScriptTests.setUp(self)
        self.run_script=HostScriptTests.run_script.__get__(self)
        self.calls=HostScriptTests.calls.__get__(self)

    def test_external_server_and_explicit_results(self):
        result=self.run_script('--external_server','--results_json','/workspace/results/ah_8/results.json')
        self.assertEqual(result.returncode,0,result.stderr)
        calls=self.calls();self.assertFalse(any(c[0] in ('run','stop','rm') for c in calls))
        client=next(c for c in calls if 'sim_to_real_so101.scripts.lerobot_eval' in c)
        self.assertEqual(client[client.index('--results_json')+1],'/workspace/results/ah_8/results.json')

    def test_server_only_ctrl_c_owns_cleanup(self):
        import os
        import signal
        import subprocess
        import time
        from test_eval_pick_place_script import SCRIPT
        proc=subprocess.Popen([str(SCRIPT),'--model','model/checkpoint-10','--models_dir',str(self.root/'models'),
                               '--dataset','/workspace/datasets/demo','--server_only'],
                              env=self.env,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                if self.log.exists() and any(c[0]=='inspect' and c[-1]!='teleop' for c in self.calls()):break
                time.sleep(.02)
            else:self.fail('Server startup was not reached')
            os.killpg(proc.pid,signal.SIGINT);proc.communicate(timeout=5)
            self.assertEqual(proc.returncode,130)
            self.assertEqual([c[0] for c in self.calls()][-2:],['stop','rm'])
            self.assertFalse(any('sim_to_real_so101.scripts.lerobot_eval' in c for c in self.calls()))
        finally:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.communicate()


if __name__=='__main__':unittest.main()
