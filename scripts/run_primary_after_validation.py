"""Serial GPU primary collection after the registered numerical test passes.

This deliberately has no graph, transcoder, label-tuned or readout stages. It
does not enable the full study. Run from an immutable checkout; retain failures.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from ilslens.experiment import write_json
from ilslens.validation_gate import validate_numerical_result
from ilslens.runtime import resolve_device


def process_alive(pid):
    path=Path('/proc')/str(pid)/'stat'
    try:return path.read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def main():
    p=argparse.ArgumentParser();p.add_argument('--numerical',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--control',required=True);args=p.parse_args()
    numerical=Path(args.numerical).resolve();control=Path(args.control).resolve();control.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(control/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    launch=json.loads((numerical/'launch.json').read_text());pid=launch['pid'];start=time.perf_counter()
    status=control/'status.json'
    def record(state,**details):
        write_json(status,dict(state=state,pid=os.getpid(),elapsed_seconds=time.perf_counter()-start,**details))
    try:
        record('waiting_for_numerical_validation',numerical_pid=pid)
        while process_alive(pid):time.sleep(15)
        result_path=numerical/'results.json'
        if not result_path.exists():raise RuntimeError('Numerical process exited without a result; inspect its log')
        gate=validate_numerical_result(json.loads(result_path.read_text()),json.loads((numerical/'manifest.json').read_text()))
        record('numerical_validation_passed',validation=gate)
        commands=[
            ('primary-collection',[sys.executable,'-u','scripts/run_qwen_main.py','--dataset',args.dataset,
                '--reference',args.reference,'--output',args.output,'--phase','all','--device','cuda']),
            ('primary-fitting',[sys.executable,'-u','scripts/fit_qwen_offline.py','--run',args.output,'--device','cuda'])]
        for stage,command in commands:
            # Each runner independently checks the free GPU and writes its cost
            # estimate before expensive work. Any failure stops this sequence.
            while True:
                try:resolve_device('cuda');break
                except RuntimeError as error:
                    if not str(error).startswith('GPU busy:'):raise
                    record('waiting_for_gpu',stage=stage,reason=str(error));time.sleep(15)
            record('running',stage=stage,command=command)
            with (control/(stage+'.log')).open('ab') as log:
                subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        record('primary_stages_completed',scope='Primary fitting only; full research is not complete')
    except Exception as error:
        record('failed',error=str(error));raise


if __name__=='__main__':main()
