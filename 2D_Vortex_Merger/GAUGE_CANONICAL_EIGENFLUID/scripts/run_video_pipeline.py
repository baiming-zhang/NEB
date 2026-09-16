"""Resume the authorized K1024 ablation and generate the final 2x4 videos.

Every stage writes an independent completion artifact. Transient subprocess
failures resume from the latest checkpoint up to three times; persistent
failures are recorded for the task heartbeat to diagnose and repair.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
S=ROOT/'scripts'; OUT=ROOT/'outputs/comparison_64_1024'

def write(value):
    value.update(pid=os.getpid(),updated_unix=time.time())
    temp=OUT/'pipeline_status.tmp'; temp.write_text(json.dumps(value,indent=2)); temp.replace(OUT/'pipeline_status.json')

def train_command():
    run=OUT/'k1024_raw/run_001'; latest=run/'checkpoints/hardcoded_latest.npz'
    initial=OUT/'k1024_raw/inputs/initial_epoch0000.npz'
    return [sys.executable,'-u',str(S/'train_gauge_ablation.py'),'--resume',str(latest if latest.exists() else initial),
            '--run-dir',str(run),'--validation-pack',str(OUT/'k1024_raw/inputs/fixed_validation_iid_uniform_v1.npz')]

def stages():
    yield 'train_k1024_raw',OUT/'k1024_raw/run_001/completion.json',train_command
    for k,arm,checkpoint in [
        (1024,'canonical',OUT/'k1024_canonical/checkpoint.npz'),
        (1024,'raw',OUT/'k1024_raw/run_001/checkpoints/hardcoded_sss_frozen.npz'),
        (64,'canonical',ROOT/'outputs/neural_k64_e2000/run_001/checkpoints/hardcoded_sss_frozen.npz'),
        (64,'raw',ROOT/'outputs/gauge_ablation/raw/run_001/checkpoints/hardcoded_sss_frozen.npz')]:
        folder=OUT/f'k{k}_{arm}/evaluation'
        command=[sys.executable,'-u',str(S/'evaluate_large_comparison.py'),'--checkpoint',str(checkpoint),'--output',str(folder)]
        if k==64:
            command+=['--check-against',str(ROOT/f'outputs/gauge_ablation/{arm}/evaluation/fields.npz')]
        yield f'evaluate_k{k}_{arm}',folder/'summary.json',lambda cmd=command:cmd
    yield 'analytic_relations',OUT/'analytic_eigenrelation.json',lambda:[sys.executable,'-u',str(S/'check_analytic_relations.py')]
    yield 'render_videos',OUT/'completion.json',lambda:[sys.executable,'-u',str(S/'render_comparison_video.py')]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--train-only',action='store_true'); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'logs').mkdir(exist_ok=True)
    started=time.time()
    for name,sentinel,command in stages():
        if args.train_only and not name.startswith('train'): break
        if sentinel.exists():
            print('Reusing completed stage:',name,flush=True); continue
        for attempt in range(1,4):
            logfile=OUT/'logs'/f'{name}_attempt{attempt}.log'
            with logfile.open('a',encoding='utf-8') as log:
                proc=subprocess.Popen(command(),cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT)
                write(dict(status='running',stage=name,attempt=attempt,child_pid=proc.pid,log=str(logfile),started_unix=started))
                print(f'{name}: attempt {attempt}, PID {proc.pid}',flush=True)
                code=proc.wait()
            if code==0 and sentinel.exists(): break
            write(dict(status='retrying',stage=name,attempt=attempt,exit_code=code,log=str(logfile),started_unix=started))
            if attempt==3:
                write(dict(status='needs_repair',stage=name,exit_code=code,log=str(logfile),started_unix=started))
                raise RuntimeError(f'{name} failed; inspect {logfile}, repair then rerun pipeline')
            time.sleep(5)
    write(dict(status='training_complete' if args.train_only else 'complete',started_unix=started,elapsed_seconds=time.time()-started))

if __name__=='__main__': main()
