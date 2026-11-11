"""Queue the label-tuned J++ baseline after capability filtering succeeds."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from ilslens.experiment import write_json
from ilslens.runtime import resolve_device
from scripts.run_primary_after_validation import process_alive


def require_capability_success(launch,status):
    if status.get('pid')!=launch['pid']:
        raise ValueError('Capability controller identity differs from its launch')
    if status.get('state')!='capability_filter_completed':
        raise RuntimeError('Capability filter did not complete successfully: '+str(status.get('state')))


def retained_dev_items(gates):
    return [row['item'] for row in gates if row['retained'] and row['item']['split']=='dev']


def main():
    p=argparse.ArgumentParser()
    for name in ('capability-control','gates','dataset','reference','jpp','run','control','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--after-process',type=int,help='Also wait for an independent GPU job before fitting')
    a=p.parse_args()
    predecessor=Path(a.capability_control);launch=json.loads((predecessor/'launch.json').read_text())
    control=Path(a.control);control.mkdir(parents=True,exist_ok=True)
    lock=(control/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    begin=time.perf_counter()
    def record(state,**extra):
        write_json(control/'status.json',dict(state=state,pid=os.getpid(),
            elapsed_seconds=time.perf_counter()-begin,capability_pid=launch['pid'],label_tuned=True,**extra))
    try:
        record('waiting_for_capability_filter')
        while process_alive(launch['pid']):time.sleep(15)
        require_capability_success(launch,json.loads((predecessor/'status.json').read_text()))
        dev=retained_dev_items(json.loads(Path(a.gates).read_text()))
        if not dev:
            record('unavailable_no_retained_dev_items',reason='No capability-filtered labeled dev items; no J++ measurement')
            return
        if a.after_process is not None:
            record('waiting_for_process',after_process=a.after_process)
            while process_alive(a.after_process):time.sleep(15)
        while True:
            try:resolve_device('cuda',min_free_gib=9.5);break
            except RuntimeError as error:
                if not str(error).startswith('GPU busy:'):raise
                record('waiting_for_gpu',reason=str(error));time.sleep(15)
        command=[sys.executable,'-u','scripts/fit_jpp_qwen.py']
        for name in ('dataset','reference','jpp','gates','run','output'):
            command+=['--'+name,getattr(a,name)]
        command+=['--device','cuda']
        record('running_jpp',command=command,retained_dev_items=len(dev))
        with (control/'jpp.log').open('ab') as log:
            subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (Path(a.output)/'J++-label-tuned.pt').is_file():
            raise RuntimeError('J++ fitter exited without the labeled baseline artifact')
        record('jpp_completed',scope='Label-tuned baseline fitting only; readout evaluation remains unrun')
    except Exception as error:
        record('failed',error=str(error));raise


if __name__=='__main__':main()
