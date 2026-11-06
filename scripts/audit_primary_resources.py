"""Read-only storage accounting and a measured suffix-only timing projection."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import yaml
from ilslens.budget import baseline_count
from ilslens.experiment import write_json


def estimate(config,benchmark,layers=24,dimension=896):
    samples=config['samples_per_layer_epsilon']+config['heldout_samples_per_layer_epsilon']
    epsilons=len(config['epsilons']);gammas=len(config['gammas'])
    if benchmark['epsilon_batch']!=epsilons:raise ValueError('Timing batch differs from the registered epsilon batch')
    points=sorted(benchmark['measurements'],key=lambda row:row['layer'])
    positions=[p['layer'] for p in points];times=[p['seconds_per_six_epsilon_sample'] for p in points]
    if len(set(positions))!=len(positions) or positions[0]!=0 or positions[-1]!=layers-1:
        raise ValueError('Unique timing anchors must span all registered layers')
    if not np.isfinite(times).all() or min(times)<0:raise ValueError('Invalid measured timings')
    interpolated=np.interp(np.arange(layers),positions,times)
    cells=layers*epsilons*gammas
    tensors=dict(intervention_deltas=samples*layers*epsilons*dimension*4,
        intervention_responses=samples*layers*epsilons*gammas*dimension*4,
        intervention_indices=samples*layers*4*8,
        per_prompt_J_R=2*baseline_count(config)*layers*dimension**2*4,
        selected_ILS=cells*dimension**2*4)
    return dict(dimension=dimension,layers=layers,sample_draws_per_layer=samples,
        batched_epsilon_count=epsilons,horizons_per_forward_pair=gammas,selected_fit_cells=cells,
        tensor_bytes=tensors,tensor_bytes_lower_bound=sum(tensors.values()),
        suffix_projection_seconds=float(samples*interpolated.sum()),
        seconds_per_six_epsilon_sample_by_layer=interpolated.tolist(),
        original_registered_estimates={k:v for k,v in config.items() if 'estimated' in k},
        timing_method='Piecewise-linear interpolation between measured layer anchors; multiply once by train plus held-out sample draws, with epsilons and horizons already batched',
        exclusions=['Timing is not a full-run forecast or confidence interval',
            'Benchmark uses one source position and prompt; production positions and prompts vary',
            'Excludes donor-bank construction, repeated host transfers outside suffix timing, checkpoint serialization, baseline fitting, CV fitting, and all downstream evaluations',
            'Storage excludes serialization metadata, logs, atomic temporary copies, baseline-mean cache, and all downstream stages'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/qwen-main.yaml')
    p.add_argument('--benchmark',default='artifacts/qwen-gpu-suffix-benchmark.json')
    p.add_argument('--output',default='artifacts/primary-resource-projection.json');a=p.parse_args()
    config=Path(a.config);benchmark=Path(a.benchmark)
    result=estimate(yaml.safe_load(config.read_text()),json.loads(benchmark.read_text()))
    result.update(config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),benchmark_sha256=hashlib.sha256(benchmark.read_bytes()).hexdigest(),status='pre_run_projection_not_actual_cost')
    write_json(a.output,result)
    print(json.dumps(dict(tensor_bytes_lower_bound=result['tensor_bytes_lower_bound'],suffix_projection_seconds=result['suffix_projection_seconds'])))

if __name__=='__main__':main()
