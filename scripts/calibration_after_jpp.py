"""Queue full reference calibration after the registered J++ stage terminates."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from ilslens.experiment import write_json
from ilslens.runtime import resolve_device
from scripts.run_primary_after_validation import process_alive


def jpp_available(launch,status):
    if status.get('pid')!=launch['pid']:
        raise ValueError('J++ controller identity differs from its launch')
    if status.get('state')=='jpp_completed':return True
    if status.get('state')=='unavailable_no_retained_dev_items':return False
    raise RuntimeError('J++ stage has not completed or established dev-set unavailability: '+str(status.get('state')))


def main():
    p=argparse.ArgumentParser()
    for name in ('jpp-control','jpp-artifact','dataset','frequencies','run','control','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    predecessor=Path(a.jpp_control);launch=json.loads((predecessor/'launch.json').read_text())
    control=Path(a.control);control.mkdir(parents=True,exist_ok=True)
    lock=(control/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    begin=time.perf_counter()
    def record(state,**extra):
        write_json(control/'status.json',dict(state=state,pid=os.getpid(),
            elapsed_seconds=time.perf_counter()-begin,jpp_pid=launch['pid'],**extra))
    try:
        record('waiting_for_jpp')
        while process_alive(launch['pid']):time.sleep(15)
        include_jpp=jpp_available(launch,json.loads((predecessor/'status.json').read_text()))
        if include_jpp and not Path(a.jpp_artifact).is_file():
            raise RuntimeError('Completed J++ stage is missing its artifact')
        while True:
            try:resolve_device('cuda',min_free_gib=9.5);break
            except RuntimeError as error:
                if not str(error).startswith('GPU busy:'):raise
                record('waiting_for_gpu',reason=str(error));time.sleep(15)
        command=[sys.executable,'-u','scripts/calibrate_qwen.py','--config','configs/calibration.yaml']
        for name in ('dataset','frequencies','run','output'):
            command+=['--'+name,getattr(a,name)]
        command+=['--device','cuda']
        if include_jpp:command+=['--jpp',a.jpp_artifact]
        record('running_calibration',command=command,jpp_included=include_jpp)
        with (control/'calibration.log').open('ab') as log:
            subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (Path(a.output)/'timing.json').is_file():
            raise RuntimeError('Calibration exited without its completion timing artifact')
        record('calibration_completed',jpp_included=include_jpp,
            missing_baselines=[] if include_jpp else ['J++ (label-tuned): no retained dev items'],
            scope='Calibration of available lenses only; not readout evaluation or full-study completion')
    except Exception as error:
        record('failed',error=str(error));raise


if __name__=='__main__':main()
