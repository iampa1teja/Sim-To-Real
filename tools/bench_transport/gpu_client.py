"""Measure a real s2r policy server from a CPU client; no robot/device is opened."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from common import summary
from core.arms.calibration import resolve_calibration
from core.arms.mapping import raw_joint_limits
from core.arms.spec import load_arm_spec
from core.policy_client.contract import ArmInfo
from core.policy_client.discovery import get_manifest
from core.policy_client.remote_policy import RemotePolicy


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--arm', default='so101')
    parser.add_argument('--unit')
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--policy-option', action='append',default=[])
    parser.add_argument('--action-horizon', type=int,default=30)
    parser.add_argument('--seconds', type=float,default=60.)
    parser.add_argument('--fps', type=int,default=30)
    parser.add_argument('--height', type=int,default=480)
    parser.add_argument('--width', type=int,default=640)
    parser.add_argument('--async-inference', action='store_true')
    parser.add_argument('--refill-seconds', type=float,default=.5)
    parser.add_argument('--starvation-grace', type=float,default=5.)
    parser.add_argument('--timeout-ms', type=int,default=15000)
    parser.add_argument('--output', type=Path,required=True)
    args=parser.parse_args()
    if args.seconds<=0 or args.fps<=0 or min(args.height,args.width)<=0:
        parser.error('seconds, fps and image dimensions must be positive')
    if any('=' not in option for option in args.policy_option):
        parser.error('--policy-option must use KEY=VALUE')
    arm=load_arm_spec(args.arm)[0]
    selection=resolve_calibration(arm,unit=args.unit)
    info=ArmInfo.from_arm(arm,selection.mapping)
    bounds=raw_joint_limits(arm,selection.mapping)
    state=np.array([sum(bounds[n])/2 for n in arm.joint_names],dtype=np.float32)
    rng=np.random.default_rng(20261114)
    images={name:rng.integers(0,256,(args.height,args.width,3),dtype=np.uint8)
            for name in get_manifest(args.policy).cameras}
    policy=RemotePolicy(info.feature_names,policy_name=args.policy,host=args.host,port=args.port,
        arm_info=info,action_horizon=args.action_horizon,lang_instruction=args.instruction,
        async_inference=args.async_inference,refill_seconds=args.refill_seconds,
        starvation_grace=args.starvation_grace,control_dt=1/args.fps,timeout_ms=args.timeout_ms,
        policy_options=dict(option.split('=',1) for option in args.policy_option))
    rows,turnarounds=[],[]
    try:
        policy.connect(); policy.reset()
        started=time.monotonic()
        count=round(args.seconds*args.fps)
        i=0; last_chunk=-1; last_origin=None; replacements=0
        while i<count:
            time.sleep(max(0.,started+i/args.fps-time.monotonic()))
            before=time.monotonic(); calls=policy.inference_calls
            action=policy.get_raw_action(state,images)
            if not np.isfinite(list(action.values())).all():
                raise ValueError('Non-finite model output')
            now=time.monotonic()
            completed_slot=max(i,int((now-started)*args.fps))
            while i<min(count,completed_slot):
                rows.append(dict(tick=i,held=True,age_ms=None if last_origin is None else
                                 (started+i/args.fps-last_origin)*1000,chunk_id=last_chunk))
                i+=1
            if i>=count:
                break
            if args.async_inference:
                tick=policy.last_tick
                fresh=not tick.held
                chunk=tick.chunk_id
                origin=policy._async.playback.last_origin
                turnaround=tick.turnaround*1000
            else:
                fresh=True; chunk=policy.inference_calls-1
                origin=before if policy.inference_calls!=calls else last_origin
                turnaround=(now-before)*1000
            if fresh:
                if chunk!=last_chunk:
                    replacements+=last_chunk>=0
                    turnarounds.append(turnaround)
                last_chunk,last_origin=chunk,origin
            rows.append(dict(tick=i,held=not fresh,age_ms=None if last_origin is None else
                             (now-last_origin)*1000,chunk_id=last_chunk))
            i+=1
        time.sleep(max(0.,started+args.seconds-time.monotonic()))
        result=dict(policy=args.policy,server=policy.server_info,async_inference=args.async_inference,
                    elapsed_s=time.monotonic()-started,ticks=count,held_ticks=sum(r['held'] for r in rows),
                    chunk_replacements=replacements,turnaround_ms=summary(turnarounds),
                    action_age_ms=summary([r['age_ms'] for r in rows if r['age_ms'] is not None]),
                    calibration=selection.provenance,rows=rows)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','calibration')},indent=2))
    finally:
        policy.close()


if __name__=='__main__':
    main()
