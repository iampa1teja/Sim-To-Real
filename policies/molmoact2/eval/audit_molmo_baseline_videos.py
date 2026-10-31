"""Read-only decode/hash/blank-frame audit of all listed completed attempts.

No frames are exported and no simulator/model process is touched. Evidence is
RAM-only. This checks every decoded frame's recording health, not its semantics.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
ROOT = Path(__file__).resolve().parents[3]
BASE = Path('/dev/shm') / f'molmo-video-audit-{time.time_ns()}-{os.getpid()}'
BASE.mkdir()
SOURCE = ROOT / 'datasets/eval_results/molmoact2_so101_nodr_20261021T133545904060377.json'
DEST = BASE / 'video_frame_health.json'
source_bytes = SOURCE.read_bytes()
source_sha = hashlib.sha256(source_bytes).hexdigest()
assert source_sha == '658a4ea997804d03fceed9d0cd2120dd942ed25b30ab35a5cf42f1852357f1a2'
report = json.loads(source_bytes)
assert report['complete'] and len(report['episodes']) == 150
jobs = []
for row in report['episodes']:
    for camera, path in row['videos'].items():
        prefix = '/workspace/Sim-to-Real-SO-101-Workshop/'
        assert path.startswith(prefix)
        jobs.append(dict(episode=row['episode'], start_id=row['start_id'], zone=row['zone'], success=row['success'], failure_mode=row['failure_mode'], camera=camera, path=str(ROOT / path[len(prefix):]), expected_frames=row['steps']))
assert len(jobs) == 300 and len({j['path'] for j in jobs}) == 300
begin = time.monotonic()
results = []


def audit(job):
    item = dict(job)
    path = Path(job['path'])
    initial = path.stat()
    item['bytes'] = initial.st_size
    command = [shutil.which('ffmpeg') or '/usr/bin/ffmpeg', '-nostdin', '-hide_banner', '-nostats', '-loglevel', 'info', '-threads', '2', '-filter_threads', '1', '-hwaccel', 'none', '-err_detect', 'explode', '-xerror', '-i', str(path), '-map', '0:v:0', '-an', '-vf', 'blackdetect=d=0.05:pix_th=0.02:pic_th=0.99', '-fps_mode', 'passthrough', '-threads', '1', '-f', 'framemd5', '-']
    frame_count = 0
    unique = set()
    unique_after = set()
    previous = None
    duplicate_count = 0
    streak = 0
    longest = 0
    last_pts = None
    bad_pts = 0
    sequence = hashlib.sha256()
    headers = []
    timed_out = []
    tick = time.monotonic()
    with tempfile.TemporaryFile(mode='w+', dir=BASE) as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)

        def timeout():
            timed_out.append(True)
            process.kill()
        timer = threading.Timer(180, timeout)
        timer.daemon = True
        timer.start()
        try:
            for line in process.stdout:
                if line.startswith('#'):
                    headers.append(line.strip())
                    continue
                fields = [x.strip() for x in line.split(',')]
                if len(fields) != 6:
                    continue
                digest = fields[-1]
                pts = int(fields[2])
                sequence.update(bytes.fromhex(digest))
                unique.add(digest)
                if frame_count >= 10:
                    unique_after.add(digest)
                if digest == previous:
                    duplicate_count += 1
                    streak += 1
                else:
                    streak = 1
                longest = max(longest, streak)
                if last_pts is not None and pts <= last_pts:
                    bad_pts += 1
                previous = digest
                last_pts = pts
                frame_count += 1
            code = process.wait()
        finally:
            timer.cancel()
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        errors.seek(0)
        log = errors.read()
    spans = []
    for match in re.finditer(r'black_start:([\d.e+-]+) black_end:([\d.e+-]+) black_duration:([\d.e+-]+)', log):
        a, b, d = map(float, match.groups())
        spans.append(dict(start_s=a, end_s=b, duration_s=d))
    final = path.stat()
    item.update(decoded_frames=frame_count, frame_count_matches=frame_count == job['expected_frames'], decode_exit_code=code, timed_out=bool(timed_out), unique_frames=len(unique), unique_frames_after_settling=len(unique_after), adjacent_duplicate_frames=duplicate_count, longest_identical_run_frames=longest, nonincreasing_timestamps=bad_pts, decoded_sequence_sha256=sequence.hexdigest(), black_spans=spans, black_after_settling_s=sum(max(0, s['end_s'] - max(s['start_s'], 10 * report['step_dt'])) for s in spans), source_stat_unchanged=(initial.st_size, initial.st_mtime_ns, initial.st_ino) == (final.st_size, final.st_mtime_ns, final.st_ino), headers=headers, wall_s=time.monotonic() - tick)
    if code or timed_out:
        item['decode_diagnostic_tail'] = log[-3000:]
    return item


def save(status):
    data = dict(status=status, pid=__import__('os').getpid(), source_report=str(SOURCE), source_report_sha256=source_sha, source_report_unchanged=hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_sha, requested_videos=len(jobs), completed_videos=len(results), elapsed_wall_s=time.monotonic() - begin, frame_audit_scope='All frames decoded and hashed; >99% black-frame spans checked. No claim that all frames were semantically interpreted.', limitations='Exact duplicate frames can reflect a stationary scene; they alone do not prove stale model inputs. Saved lossy videos are not wire-level model inputs. Recording health cannot prove a policy failure cause.', videos=sorted(results, key=lambda r: (r['episode'], r['camera'])))
    data['summary'] = dict(total_decoded_frames=sum(r['decoded_frames'] for r in results), decode_errors=sum(r['decode_exit_code'] != 0 for r in results), frame_count_mismatches=sum(not r['frame_count_matches'] for r in results), source_changes=sum(not r['source_stat_unchanged'] for r in results), black_after_settling_videos=sum(r['black_after_settling_s'] > .05 for r in results), entirely_static_after_settling_videos=sum(r['unique_frames_after_settling'] <= 1 for r in results), nonincreasing_timestamp_videos=sum(r['nonincreasing_timestamps'] > 0 for r in results))
    if status == 'complete':
        identical = []
        for episode in range(150):
            group = [r for r in results if r['episode'] == episode]
            if len(group) == 2 and group[0]['decoded_sequence_sha256'] == group[1]['decoded_sequence_sha256']:
                identical.append(episode)
        data['summary']['identical_room_wrist_episode_streams'] = identical
    temporary = DEST.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(DEST)
    print('[VIDEO_AUDIT] ' + json.dumps(dict(status=status, completed=len(results), requested=len(jobs), **data['summary'])), flush=True)


save('running')
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {executor.submit(audit, job): job for job in jobs}
    for future in as_completed(futures):
        results.append(future.result())
        if len(results) % 10 == 0:
            save('running')
save('complete')
print(f'RAM audit report: {DEST}', flush=True)
