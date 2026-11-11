"""Score the completed chat capability cohort after calibration succeeds."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from ilslens.experiment import write_json
from ilslens.runtime import resolve_device
from scripts.run_primary_after_validation import process_alive


def calibration_ready(launch,status):
    if status.get('pid')!=launch['pid']:raise ValueError('Calibration controller identity changed')
    if status.get('state')!='calibration_completed':raise RuntimeError('Calibration did not complete successfully')
    if not isinstance(status.get('jpp_included'),bool):raise ValueError('Missing explicit J++ calibration status')
    return status['jpp_included']


def validate_readouts(result, include_jpp, expected_tasks):
    if not result.get('complete_calibration') or result.get('missing_calibration'):
        raise ValueError('Readouts have missing calibration')
    labels=result['metadata']['label_tuned']
    if not {'logit','J-reference','R-reference','ILS'}<=labels.keys():raise ValueError('Missing readout baseline')
    if ('J++ (label-tuned)' in labels)!=include_jpp:raise ValueError('J++ inclusion differs from calibration outcome')
    for name,tuned in labels.items():
        if tuned!=(name=='J++ (label-tuned)'):raise ValueError('Incorrect label-tuned marking')
    expected={(name,task,layer,mode,k) for name in labels for task in expected_tasks
              for layer in list(range(24))+['any'] for mode in ('raw','z') for k in (1,10)}
    keys=[(r['lens'],r['task'],r['layer'],r['mode'],r['k']) for r in result['rows'] if 'k' in r]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Incomplete or duplicate readout metric grid')
    # Empty task families remain unknown; never require nonzero recall or select items here.
    if result['retained_test_items']<1:raise ValueError('No capability-filtered test items available for readout scoring')


def main():
    parser=argparse.ArgumentParser()
    for name in ('calibration-control','calibration-root','jpp-artifact','source','run','control','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--config',default='configs/readout-chat.yaml')
    args=parser.parse_args()
    predecessor=Path(args.calibration_control);launch=json.loads((predecessor/'launch.json').read_text())
    control=Path(args.control);control.mkdir(parents=True,exist_ok=True)
    lock=(control/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.perf_counter()
    def record(state,**extra):
        write_json(control/'status.json',dict(state=state,pid=os.getpid(),calibration_pid=launch['pid'],
            elapsed_seconds=time.perf_counter()-start,**extra))
    try:
        record('waiting_for_calibration')
        while process_alive(launch['pid']):time.sleep(15)
        include_jpp=calibration_ready(launch,json.loads((predecessor/'status.json').read_text()))
        if include_jpp and not Path(args.jpp_artifact).is_file():raise RuntimeError('Calibrated J++ artifact is missing')
        gates=Path(args.output)/'gates.json'
        before=hashlib.sha256(gates.read_bytes()).hexdigest()
        while True:
            try:resolve_device('cuda',min_free_gib=9.5);break
            except RuntimeError as error:
                if not str(error).startswith('GPU busy:'):raise
                record('waiting_for_gpu',reason=str(error));time.sleep(15)
        command=[sys.executable,'-u','scripts/evaluate_readouts.py','--device','cuda']
        for name in ('source','run','config','output','calibration-root'):
            command+=['--'+name,getattr(args,name.replace('-','_'))]
        if include_jpp:command+=['--jpp',args.jpp_artifact]
        record('running_readouts',command=command,jpp_included=include_jpp,gate_sha256=before)
        with (control/'readouts.log').open('ab') as log:
            subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if hashlib.sha256(gates.read_bytes()).hexdigest()!=before:raise ValueError('Readout scoring changed the capability cohort')
        result=json.loads((Path(args.output)/'results.json').read_text())
        tasks={r['item']['task'] for r in json.loads(gates.read_text())}
        validate_readouts(result,include_jpp,tasks)
        record('readouts_completed',jpp_included=include_jpp,retained_test_items=result['retained_test_items'],
            scope='Raw/z-scored five-task readouts with calibrated FDR and sparse pursuit; not causal validation or full-study completion')
    except Exception as error:
        record('failed',error=str(error));raise


if __name__=='__main__':main()
