"""Queue only the transcoder throughput benchmark after primary GPU stages.

Run from a fixed checkout. A stopped or failed primary controller is not success;
this does not enable full-study execution or launch transcoder training.
"""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from ilslens.experiment import write_json
from ilslens.runtime import resolve_device
from scripts.run_primary_after_validation import process_alive


def require_primary_success(launch,status):
    if status.get('pid')!=launch['pid']:
        raise ValueError('Primary controller identity differs from its launch')
    if status.get('state')!='primary_stages_completed':
        raise RuntimeError('Primary controller did not complete successfully: '+str(status.get('state')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--primary-control',required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--control',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    primary=Path(a.primary_control);launch=json.loads((primary/'launch.json').read_text())
    control=Path(a.control);control.mkdir(parents=True,exist_ok=True)
    lock=(control/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    begin=time.perf_counter()
    def record(state,**extra):
        write_json(control/'status.json',dict(state=state,pid=os.getpid(),
            elapsed_seconds=time.perf_counter()-begin,primary_pid=launch['pid'],**extra))
    try:
        record('waiting_for_primary')
        while process_alive(launch['pid']):time.sleep(15)
        require_primary_success(launch,json.loads((primary/'status.json').read_text()))
        while True:
            try:resolve_device('cuda',min_free_gib=9.5);break
            except RuntimeError as error:
                if not str(error).startswith('GPU busy:'):raise
                record('waiting_for_gpu',reason=str(error));time.sleep(15)
        command=[sys.executable,'-u','scripts/benchmark_transcoders.py','--dataset',a.dataset,
            '--output',a.output,'--device','cuda','--warmup','3','--steps','10']
        record('running_benchmark',command=command)
        with (control/'benchmark.log').open('ab') as log:
            subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        record('benchmark_completed',scope='Throughput benchmark only; weights discarded; no transcoder training or full-study completion')
    except Exception as error:
        record('failed',error=str(error));raise


if __name__=='__main__':main()
