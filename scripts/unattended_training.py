"""Run registered transcoder and planted-organism training after calibration.

No hyperparameter selection, automatic retries, or evaluation-label tuning.
Failures retain logs/checkpoints and stop the queue.
"""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
import yaml
from ilslens.experiment import write_json
from ilslens.runtime import resolve_device
from scripts.run_primary_after_validation import process_alive


def require_calibration_success(launch,status):
    if status.get('pid')!=launch['pid']:
        raise ValueError('Calibration controller identity differs from its launch')
    if status.get('state')!='calibration_completed':
        raise RuntimeError('Calibration did not complete successfully: '+str(status.get('state')))


def validate_benchmark(config,manifest,result):
    if manifest['protocol']['training_config']!=config:
        raise ValueError('Training configuration differs from the benchmark')
    if (result['registered_token_budget']!=100000000 or result['layers']!=24
        or result['dtype']!='float32' or result['tf32'] is not False
        or config['unique_training_windows'] is not True):
        raise ValueError('Benchmark does not cover the registered fp32 all-layer budget')


def main():
    p=argparse.ArgumentParser()
    for name in ('calibration-control','benchmark','corpus','primary-run','control','transcoders','planted'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();control=Path(a.control);control.mkdir(parents=True,exist_ok=True)
    lock=(control/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    predecessor=Path(a.calibration_control)
    launch=json.loads((predecessor/'launch.json').read_text());begin=time.perf_counter()
    config=yaml.safe_load(Path('configs/transcoders.yaml').read_text())
    benchmark=Path(a.benchmark)
    result=json.loads((benchmark/'results.json').read_text())
    validate_benchmark(config,json.loads((benchmark/'manifest.json').read_text()),result)
    write_json(control/'estimate.json',dict(
        transcoder_benchmark_projected_update_seconds=result['projected_update_seconds'],
        transcoder_registered_conservative_seconds=config['estimated_seconds'],
        transcoder_disk_mb=config['estimated_disk_mb'],
        planted_registered_seconds=28800,planted_disk_mb=800,
        limitation='Benchmark projection excludes tokenization, corpus audit, checkpoint I/O and evaluation. Registered conservative allowance retained; no budget reduction.'))
    def record(state,**extra):
        write_json(control/'status.json',dict(state=state,pid=os.getpid(),
            elapsed_seconds=time.perf_counter()-begin,calibration_pid=launch['pid'],**extra))
    commands=[
        ('transcoders',[sys.executable,'-u','scripts/train_transcoders.py','--corpus',a.corpus,
            '--config','configs/transcoders.yaml','--output',a.transcoders,'--device','cuda']),
        ('planted-train-behavior',[sys.executable,'-u','scripts/run_planted.py',
            '--config','configs/planted.yaml','--prompts','configs/stress_prompts.json',
            '--output',a.planted,'--run',a.primary_run,'--phase','train-behavior','--device','cuda'])]
    try:
        record('waiting_for_calibration')
        while process_alive(launch['pid']):time.sleep(15)
        require_calibration_success(launch,json.loads((predecessor/'status.json').read_text()))
        for stage,command in commands:
            if shutil.disk_usage(control).free<20*1024**3:
                raise RuntimeError('Less than 20 GiB free disk; retaining checkpoints and stopping queue')
            while True:
                try:resolve_device('cuda',min_free_gib=9.5);break
                except RuntimeError as error:
                    if not str(error).startswith('GPU busy:'):raise
                    record('waiting_for_gpu',stage=stage,reason=str(error));time.sleep(15)
            with (control/f'{stage}.log').open('ab') as log:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                record('running',stage=stage,child_pid=child.pid,command=command)
                code=child.wait()
                if code:raise RuntimeError(f'{stage} exited with code {code}; no automatic retry')
            destination=Path(a.transcoders if stage=='transcoders' else a.planted)
            if not (destination/'results.json').is_file():
                raise RuntimeError(f'{stage} exited without results.json')
        record('training_stages_completed',scope='Transcoder training/evaluation and planted training/behavior only; no full-study completion')
    except Exception as error:
        record('failed',error=str(error));raise


if __name__=='__main__':main()
