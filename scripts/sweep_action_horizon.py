#!/usr/bin/env python3
"""Host orchestration for a fixed-benchmark horizon sweep; evaluation stays in the existing runner."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source'))
from sim_to_real_so101.utils.pick_place_benchmark import load_eval_set
from sim_to_real_so101.scripts.summarize_sweep import complete_report, summarize_folder


def parse_horizons(value, chunk_length=16):
    values=[]
    for token in value.split(','):
        token=token.strip()
        if re.fullmatch(r'\d+-\d+', token):
            lo,hi=map(int,token.split('-'))
            if lo>hi: raise ValueError('Descending horizon range')
            if lo<1 or hi>chunk_length: raise ValueError(f'Horizons must be in 1..{chunk_length}')
            values.extend(range(lo,hi+1))
        elif re.fullmatch(r'\d+',token): values.append(int(token))
        else: raise ValueError('Use comma-separated integers or ranges, e.g. 1,2,4 or 1-16')
    if not values or any(not 1<=v<=chunk_length for v in values):
        raise ValueError(f'Horizons must be in 1..{chunk_length}')
    return list(dict.fromkeys(values))


def chunk_length(checkpoint, tag):
    """Processor's embodiment horizon takes precedence over padded model capacity."""
    processor=checkpoint/'processor_config.json'
    if processor.exists():
        d=json.loads(processor.read_text())
        configs=d.get('processor_kwargs',{}).get('modality_configs',{})
        for name,config in configs.items():
            if name.lower()==tag.lower():
                indices=config.get('action',{}).get('delta_indices',[])
                if indices: return len(indices)
    config=checkpoint/'config.json'
    if config.exists():
        d=json.loads(config.read_text())
        for key in ('chunk_size','action_chunk_length','action_horizon'):
            if key in d:
                n=d[key]
                if type(n) is not int or n<1: raise ValueError('Invalid checkpoint action chunk length')
                return n
    return 16


def host_path(path):
    path=Path(path).expanduser()
    prefix=Path('/workspace/Sim-to-Real-SO-101-Workshop')
    return ROOT/path.relative_to(prefix) if path.is_relative_to(prefix) else path.resolve()


def container_path(path, mounts):
    path=Path(path).resolve()
    for mount in sorted(mounts,key=lambda m:len(m['Source']),reverse=True):
        source=Path(mount['Source'])
        if path.is_relative_to(source): return str(Path(mount['Destination'])/path.relative_to(source))
    raise ValueError(f'Path is not mounted in teleop: {path}')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='Relative model/checkpoint path, or model directory with --checkpoint')
    p.add_argument('--checkpoint',help='Optional checkpoint subdirectory, e.g. checkpoint-50000')
    p.add_argument('--models_dir',type=Path,default=Path(os.environ.get('MODELS_DIR','~/models')))
    p.add_argument('--eval_set',type=Path,default=ROOT/'eval_sets/pick_place_v1_seed1984.json')
    p.add_argument('--dataset',type=Path,help='Defaults to eval-set dataset_path')
    p.add_argument('--splits',default='all');p.add_argument('--horizons',default='1,2,4,6,8,12,16')
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--episode_length_s',type=float,default=20)
    p.add_argument('--out_root',type=Path,default=ROOT/'datasets/hyperparameters')
    p.add_argument('--lang',default='Pick up the blue cube and place it in the white box')
    p.add_argument('--lang_by_color',default='');p.add_argument('--port',type=int,default=5555)
    for key in ('dr','resume','dry_run','gui'): p.add_argument('--'+key,action='store_true')
    p.add_argument('--render_warmup',type=int,default=0,help='Render only the N steps before each policy query (0 = every step; see lerobot_eval.py)')
    p.add_argument('--shared_gpu',action='store_true',help='Skip the free-GPU check (parallel sweeps; see scripts/sweep_checkpoints.sh)')
    args=p.parse_args(argv)
    model=Path(args.model)
    if args.checkpoint: model=model/args.checkpoint
    if model.is_absolute() or any(part in ('.','..') for part in str(model).split('/')):
        p.error('--model must be relative without . or .. components')
    args.model=str(model);args.models_dir=args.models_dir.expanduser().resolve()
    checkpoint=(args.models_dir/model).resolve()
    if not checkpoint.is_relative_to(args.models_dir): p.error('Checkpoint outside models directory')
    args.eval_set=host_path(args.eval_set);args.out_root=host_path(args.out_root)
    data=load_eval_set(args.eval_set)
    selected=list(dict.fromkeys(s['split'] for s in data['starts'])) if args.splits=='all' else args.splits.split(',')
    if len(selected)!=len(set(selected)) or not set(selected)<=set(s['split'] for s in data['starts']): p.error('Invalid/duplicate splits')
    if args.render_warmup<0: p.error('--render_warmup must be >= 0')
    if args.repeats<1 or not math.isfinite(args.episode_length_s) or args.episode_length_s<=0: p.error('Positive repeats and finite episode length required')
    if not 1<=args.port<=65535: p.error('Port must be in 1..65535')
    try: horizons=parse_horizons(args.horizons,chunk_length(checkpoint,os.environ.get('EMBODIMENT_TAG','NEW_EMBODIMENT')))
    except ValueError as e: p.error(str(e))
    if max(horizons)>16: p.error('Current SO101 evaluator supports execution horizons up to 16')
    args.dataset=host_path(args.dataset or data['dataset_path'])
    if args.dry_run:
        mounts=[dict(Source=str(ROOT),Destination='/workspace/Sim-to-Real-SO-101-Workshop')]
    else:
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        if busy and not args.shared_gpu: p.error('GPU is busy; stopped before launching: '+busy)
        mounts=json.loads(subprocess.check_output(['docker','inspect','teleop'],text=True))[0]['Mounts']
    eval_path=container_path(args.eval_set,mounts);dataset_path=container_path(args.dataset,mounts)
    expected=[(s['id'],r) for r in range(args.repeats) for s in data['starts'] if s['split'] in selected]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    prefix=f"{str(model).replace('/','_')}_{args.eval_set.stem}_"
    # render_warmup joins the identity only when used, so --resume still matches sweeps made before it existed.
    flags={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
           if k not in ('resume','dry_run','shared_gpu','port') and not (k=='render_warmup' and not v)}
    flags.update(server_image=os.environ.get('SERVER_IMAGE','real-robot:n1.7'),
                 embodiment_tag=os.environ.get('EMBODIMENT_TAG','NEW_EMBODIMENT'),
                 rename_map=os.environ.get('RENAME_MAP','{"realsense_rgb":"room","wrist_cam":"wrist"}'))
    settings=dict(checkpoint=str(model),episode_length_s=args.episode_length_s,robot_start='recorded',seed=1984,
                  task='Lerobot-So101-Teleop-Pick-Place'+('-DR' if args.dr else '')+'-Eval',repeats=args.repeats)
    identity=dict(flags=flags,eval_set_id=data['eval_set_id'],eval_set_sha256=hashlib.sha256(args.eval_set.read_bytes()).hexdigest())
    args.out_root.mkdir(parents=True,exist_ok=True)
    matches=sorted(d for d in args.out_root.glob(prefix+'*') if d.is_dir() and (d/'sweep_config.json').exists())
    folder=matches[-1] if args.resume and matches else args.out_root/(prefix+stamp)
    if args.resume and matches:
        old=json.loads((folder/'sweep_config.json').read_text())
        if any(old.get(k)!=v for k,v in identity.items()): p.error('Latest matching sweep has different flags or eval set; start a new sweep')
    else:
        folder.mkdir(exist_ok=False)
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
        config=dict(**identity,horizons=horizons,selected_splits=selected,expected_schedule=expected,
                    report_settings=settings,checkpoint_path=str(checkpoint),eval_set_path=str(args.eval_set),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    host=socket.gethostname(),gpu_name=gpu,start_time=datetime.now(timezone.utc).isoformat())
        (folder/'sweep_config.json').write_text(json.dumps(config,indent=2)+'\n')
    base=[str(ROOT/'docker/eval_pick_place.sh'),'--model',str(model),'--models_dir',str(args.models_dir),
          '--dataset',dataset_path,'--eval_set',eval_path,'--splits',','.join(selected),'--repeats',str(args.repeats),
          '--episode_length_s',str(args.episode_length_s),'--lang',args.lang,'--port',str(args.port)]
    if args.dr: base+=['--dr']
    if args.gui: base+=['--gui']
    if args.lang_by_color: base+=['--lang_by_color',args.lang_by_color]
    if args.render_warmup: base+=['--render_warmup',str(args.render_warmup)]
    server=None;client=None;current_result=None
    def terminate(process):
        if process and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL);process.wait()
    def interrupt(signum,frame): raise KeyboardInterrupt
    previous={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        pending=[]
        for horizon in horizons:
            run=folder/f'ah_{horizon}';run.mkdir(exist_ok=True)
            report_path=run/'results.json'
            try: report=json.loads(report_path.read_text())
            except (OSError,ValueError): report={}
            if args.resume and complete_report(report,expected,data['eval_set_id'],horizon,settings):
                tqdm.write(f'AH {horizon}: skipping complete {len(expected)} episodes')
                (run/'status.json').write_text(json.dumps(dict(status='complete',resumed=True)))
                continue
            pending.append((horizon,run))
        server_command=base+['--server_only']
        if args.dry_run: server_command+=['--dry_run']
        (folder/'server_cmd.txt').write_text(shlex.join(server_command)+'\n')
        print('Sweep folder:',folder,flush=True)
        if pending:
            if args.dry_run:
                subprocess.run(server_command,check=True)
            else:
                server_log=(folder/'server.log').open('w')
                server=subprocess.Popen(server_command,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
                deadline=time.monotonic()+330
                while 'SERVER_READY' not in (folder/'server.log').read_text():
                    if server.poll() is not None: raise RuntimeError('Server failed; see server.log')
                    if time.monotonic()>deadline: raise RuntimeError('Server readiness timed out; see server.log')
                    time.sleep(1)
        for horizon,run in pending:
            if server is not None and server.poll() is not None:
                (run/'run.log').write_text('Shared server exited; horizon failed without launching a client.\n')
                (run/'status.json').write_text(json.dumps(dict(status='failed',exit_code=server.returncode)))
                print(f'AH {horizon}: failed; shared server exited',flush=True)
                continue
            current_result=container_path(run/'results.json',mounts)
            command=base+['--external_server','--action_horizon',str(horizon),'--results_json',current_result]
            if args.dry_run: command+=['--dry_run']
            (run/'cmd.txt').write_text(shlex.join(command)+'\n')
            # Existing incomplete results must not qualify a failed retry as complete.
            if not args.dry_run:
                for name in ('results.json','results.png'):
                    if (run/name).exists(): (run/name).rename(run/(name+'.previous'))
            with (run/'run.log').open('w') as log, (run/'run.log').open() as progress_log, tqdm(
                total=len(expected), desc=f'AH {horizon} ({horizons.index(horizon)+1}/{len(horizons)})',
                unit='ep', mininterval=1, dynamic_ncols=True, leave=True, disable=args.dry_run,
            ) as progress:
                client=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                tail=''
                def update_progress():
                    nonlocal tail
                    # Read only new log output; retain a partial episode marker between reads.
                    text=tail+progress_log.read()
                    tail=text[-80:]
                    done=max([int(n) for n in re.findall(r'\[EPISODE (\d+)\]',text)],default=progress.n)
                    progress.update(max(0,min(done,len(expected))-progress.n))
                while client.poll() is None:
                    if not args.dry_run:
                        update_progress()
                        progress.refresh()
                    time.sleep(1)
                if not args.dry_run:
                    update_progress()
            code=client.returncode;client=None
            if args.dry_run:
                print((run/'run.log').read_text());continue
            try: report=json.loads((run/'results.json').read_text())
            except (OSError,ValueError): report={}
            status='complete' if code==0 and complete_report(report,expected,data['eval_set_id'],horizon,settings) else 'failed'
            (run/'status.json').write_text(json.dumps(dict(status=status,exit_code=code))+'\n')
            tqdm.write(f'AH {horizon}: {status}, {len(report.get("episodes",[]))}/{len(expected)} episodes')
        if not args.dry_run:
            summarize_folder(folder);print((folder/'summary.md').read_text())
        return 0 if args.dry_run or all(json.loads((folder/f'ah_{h}'/'status.json').read_text())['status']=='complete' for h in horizons) else 1
    except KeyboardInterrupt:
        if current_result and not args.dry_run:
            # docker exec's host process exiting does not terminate its container child.
            code="import os,signal; marker="+repr(current_result.encode())+"; me=os.getpid(); "\
                 "matches=[int(p) for p in os.listdir('/proc') if p.isdigit() and int(p)!=me and marker in open('/proc/'+p+'/cmdline','rb').read() and b'-m\\x00sim_to_real_so101.scripts.lerobot_eval\\x00' in open('/proc/'+p+'/cmdline','rb').read()]; "\
                 "[os.kill(p,signal.SIGTERM) for p in matches]"
            subprocess.run(['docker','exec','-u','root','teleop','python3','-c',code],check=False)
        return 130
    finally:
        terminate(client);terminate(server)
        if not args.dry_run and not (folder/'summary.json').exists():
            summarize_folder(folder)
        if server is not None: server_log.close()
        for sig,handler in previous.items(): signal.signal(sig,handler)


if __name__=='__main__': sys.exit(main())
