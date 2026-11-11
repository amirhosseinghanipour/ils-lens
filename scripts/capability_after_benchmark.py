"""Queue the registered capability filter after the GPU benchmark succeeds.

Run from a fixed checkout. This launches no lens fitting, readout evaluation,
transcoder training or graph experiment, and does not enable the full study.
"""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from ilslens.experiment import write_json
from ilslens.runtime import resolve_device
from scripts.run_primary_after_validation import process_alive


def require_benchmark_success(launch,status):
    if status.get('pid')!=launch['pid']:
        raise ValueError('Benchmark controller identity differs from its launch')
    if status.get('state')!='benchmark_completed':
        raise RuntimeError('Benchmark controller did not complete successfully: '+str(status.get('state')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark-control',required=True)
    p.add_argument('--source',required=True);p.add_argument('--control',required=True)
    p.add_argument('--output',required=True);p.add_argument('--config',default='configs/readout.yaml')
    p.add_argument('--after-process',type=int,help='Also wait for this existing independent GPU process to exit')
    a=p.parse_args()
    predecessor=Path(a.benchmark_control);launch=json.loads((predecessor/'launch.json').read_text())
    control=Path(a.control);control.mkdir(parents=True,exist_ok=True)
    lock=(control/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    begin=time.perf_counter()
    def record(state,**extra):
        write_json(control/'status.json',dict(state=state,pid=os.getpid(),
            elapsed_seconds=time.perf_counter()-begin,benchmark_pid=launch['pid'],**extra))
    try:
        record('waiting_for_benchmark')
        while process_alive(launch['pid']):time.sleep(15)
        require_benchmark_success(launch,json.loads((predecessor/'status.json').read_text()))
        if a.after_process is not None:
            record('waiting_for_process',after_process=a.after_process)
            while process_alive(a.after_process):time.sleep(15)
        while True:
            try:resolve_device('cuda',min_free_gib=9.5);break
            except RuntimeError as error:
                if not str(error).startswith('GPU busy:'):raise
                record('waiting_for_gpu',reason=str(error));time.sleep(15)
        command=[sys.executable,'-u','scripts/evaluate_readouts.py','--source',a.source,
            '--output',a.output,'--config',a.config,'--device','cuda','--gate-only']
        record('running_capability_filter',command=command)
        with (control/'capability.log').open('ab') as log:
            subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        gates=Path(a.output)/'gates.json'
        if not gates.is_file():raise RuntimeError('Capability filter exited without gates.json')
        record('capability_filter_completed',scope='Capability filtering only; no readout or full-study completion')
    except Exception as error:
        record('failed',error=str(error));raise


if __name__=='__main__':main()
