"""Read saved JVP errors without rerunning or changing numerical validation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--max-layer',type=int,required=True)
    p.add_argument('--output',required=True);a=p.parse_args();start=time.perf_counter();torch.set_num_threads(2)
    root=Path(a.run);manifest=json.loads((root/'manifest.json').read_text());launch=json.loads((root/'launch.json').read_text())
    config=manifest['protocol']['config'];rows=[];sources=[];rng=np.random.default_rng(1101)
    for layer in config['layers']:
        if layer>a.max_layer:continue
        errors={epsilon:[] for epsilon in config['epsilons']}
        for index in range(config['contexts']):
            path=root/f'layer-{layer:02d}-context-{index:02d}.pt'
            saved=torch.load(path,weights_only=True,map_location='cpu')
            if saved['fingerprint']!=manifest['fingerprint']:raise ValueError('Checkpoint provenance mismatch')
            with path.open('rb') as file:digest=hashlib.file_digest(file,'sha256').hexdigest()
            sources.append(dict(file=path.name,sha256=digest,source_position=saved['position']))
            for epsilon in errors:
                value=saved['jvp_errors'][epsilon]
                if not np.isfinite(value) or value<0:raise ValueError('Invalid saved JVP error')
                errors[epsilon].append(value)
        for epsilon,values in errors.items():
            values=np.asarray(values);means=values[rng.integers(len(values),size=(1000,len(values)))].mean(1)
            rows.append(dict(layer=layer,epsilon=epsilon,contexts=len(values),relative_errors=values.tolist(),
                mean=float(values.mean()),median=float(np.median(values)),maximum=float(values.max()),
                mean_context_bootstrap_ci=np.quantile(means,[.025,.975]).tolist()))
    summary=[]
    for epsilon in config['epsilons']:
        selected=[r for r in rows if r['epsilon']==epsilon]
        values=np.asarray([r['relative_errors'] for r in selected]);by_context=values.mean(0)
        means=by_context[rng.integers(len(by_context),size=(1000,len(by_context)))].mean(1)
        worst_layer,worst_context=np.unravel_index(values.argmax(),values.shape)
        summary.append(dict(epsilon=epsilon,mean=float(values.mean()),mean_context_bootstrap_ci=np.quantile(means,[.025,.975]).tolist(),
            maximum=float(values.max()),worst_layer=selected[worst_layer]['layer'],worst_context=int(worst_context)))
    write_json(a.output,dict(status='partial_saved_measurement_audit',rows=rows,summary=summary,sources=sources,
        source_fingerprint=manifest['fingerprint'],source_commit=launch['commit'],max_layer=a.max_layer,
        seconds=time.perf_counter()-start,bootstrap_seed=1101,bootstrap_resamples=1000,
        definition='norm(antithetic finite difference - autograd JVP) / max(norm(JVP), 1e-12); one saved isotropic direction per context/epsilon',
        precision='Original function evaluations and JVPs are fp32; only summary statistics use NumPy float64',
        limitations=['Original prefix-sampled numerical contexts, not the shuffled primary corpus',
            'Context-resampling intervals are descriptive; they do not address shared-article dependence or generalize to all prompts/directions',
            'Partial layers only; not a full validation result or a newly selected tolerance',
            'Matrix cosine can remain high while individual directional finite differences have appreciable relative error',
            'Increasing error at smaller epsilon is consistent with cancellation/roundoff, but this audit does not isolate its cause']))
    print(json.dumps(dict(rows=len(rows),max_relative_error=max(r['maximum'] for r in rows),seconds=time.perf_counter()-start)))

if __name__=='__main__':main()
