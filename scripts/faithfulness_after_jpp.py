"""Run the complete held-out faithfulness grid after J++ has a terminal outcome."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from ilslens.experiment import write_json
from ilslens.runtime import resolve_device
from scripts.run_primary_after_validation import process_alive
from scripts.calibration_after_jpp import jpp_available


def validate_grid(result, config, include_jpp):
    rows=result['rows']
    lenses={r['lens'] for r in rows}
    required={'logit','J-reference','R-reference','ILS'}
    if not required<=lenses:raise ValueError('Missing required faithfulness baseline')
    if ('J++ (label-tuned)' in lenses)!=include_jpp:raise ValueError('J++ inclusion differs from controller outcome')
    expected={(name,layer,epsilon,gamma) for name in lenses for layer in config['layers']
              for epsilon in config['epsilons'] for gamma in config['gammas']}
    keys=[(r['lens'],r['layer'],r['epsilon'],r['gamma']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Incomplete or duplicate faithfulness grid')
    for row in rows:
        if row['N']!=config['prompts']*config['samples_per_prompt']:raise ValueError('Faithfulness sample budget changed')
        if row['label_tuned']!=(row['lens']=='J++ (label-tuned)'):raise ValueError('Invalid label-tuned marking')
    return sorted(lenses)


def main():
    parser=argparse.ArgumentParser()
    for name in ('jpp-control','jpp-artifact','dataset','run','control','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--config',default='configs/faithfulness.yaml')
    args=parser.parse_args()
    predecessor=Path(args.jpp_control);launch=json.loads((predecessor/'launch.json').read_text())
    control=Path(args.control);control.mkdir(parents=True,exist_ok=True)
    lock=(control/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.perf_counter()
    def record(state,**extra):
        write_json(control/'status.json',dict(state=state,pid=os.getpid(),jpp_pid=launch['pid'],
            elapsed_seconds=time.perf_counter()-start,**extra))
    try:
        record('waiting_for_jpp')
        while process_alive(launch['pid']):time.sleep(15)
        include_jpp=jpp_available(launch,json.loads((predecessor/'status.json').read_text()))
        if include_jpp and not Path(args.jpp_artifact).is_file():raise RuntimeError('Completed J++ stage is missing its artifact')
        while True:
            try:resolve_device('cuda',min_free_gib=9.5);break
            except RuntimeError as error:
                if not str(error).startswith('GPU busy:'):raise
                record('waiting_for_gpu',reason=str(error));time.sleep(15)
        command=[sys.executable,'-u','scripts/evaluate_faithfulness.py','--device','cuda']
        for name in ('dataset','run','output','config'):command+=['--'+name,getattr(args,name)]
        if include_jpp:command+=['--jpp',args.jpp_artifact]
        record('running_faithfulness',command=command,jpp_included=include_jpp)
        with (control/'faithfulness.log').open('ab') as log:
            subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((Path(args.output)/'results.json').read_text())
        lenses=validate_grid(result,yaml.safe_load(Path(args.config).read_text()),include_jpp)
        record('faithfulness_completed',lenses=lenses,rows=len(result['rows']),
               scope='Held-out residual R² and top-1000 token-change Spearman; not full-study completion')
    except Exception as error:
        record('failed',error=str(error));raise


if __name__=='__main__':main()
