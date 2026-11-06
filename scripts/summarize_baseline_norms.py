"""Describe existing primary J/R matrices; no model, refit or label selection."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--audit',default='artifacts/qwen-primary-baselines.json')
    p.add_argument('--output',default='artifacts/qwen-baseline-norms.json');a=p.parse_args()
    print(json.dumps(dict(estimated_seconds=60,estimated_output_bytes=50000,model_loaded=False)),flush=True)
    begin=time.perf_counter();torch.set_num_threads(2)
    root=Path(a.run);audit=json.loads(Path(a.audit).read_text());manifest=json.loads((root/'manifest.json').read_text())
    if audit['primary_fingerprint']!=manifest['fingerprint']:raise ValueError('Baseline audit provenance differs')
    totals={};norms={}
    for record in audit['files']:
        path=root/'baselines'/record['name']
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=record['sha256']:raise ValueError('Baseline artifact changed since validation')
        saved=torch.load(path,weights_only=True,map_location='cpu')
        for layer,matrix in saved['matrices'].items():
            key=(record['lens'],layer);value=matrix.double()
            if key not in totals:totals[key]=torch.zeros_like(value);norms[key]=[]
            totals[key]+=value;norms[key].append(float(value.norm()))
    rows=[];rng=np.random.default_rng(4101)
    n=audit['baseline_prompt_count'];indices=rng.integers(n,size=(1000,n))
    for (name,layer),values in sorted(norms.items()):
        if len(values)!=n:raise ValueError('Unequal prompt counts in norm summary')
        values=np.asarray(values);mean_matrix=totals[name,layer]/n;matrix_norm=float(mean_matrix.norm())
        rows.append(dict(lens=name,layer=layer,prompts=n,prompt_matrix_norms=values.tolist(),
            mean_prompt_norm=float(values.mean()),mean_prompt_norm_ci=np.quantile(values[indices].mean(1),[.025,.975]).tolist(),
            minimum_prompt_norm=float(values.min()),maximum_prompt_norm=float(values.max()),
            norm_of_mean_matrix=matrix_norm,
            norm_of_mean_over_mean_norm=matrix_norm/float(values.mean()) if values.mean()>0 else None))
    comparisons=[]
    for layer in range(24):
        j=totals['J',layer];r=totals['R',layer]
        comparisons.append(dict(layer=layer,J_R_matrix_cosine=float((j*r).sum()/(j.norm()*r.norm()))))
    write_json(a.output,dict(primary_fingerprint=manifest['fingerprint'],rows=rows,comparisons=comparisons,
        audit_sha256=hashlib.sha256(Path(a.audit).read_bytes()).hexdigest(),
        source_commit=audit['source_commit'],dtype_of_saved_matrices='float32',reporting_reductions='float64',
        bootstrap=dict(seed=4101,repeats=1000,unit='fitting prompt',scope='Descriptive mean per-prompt Frobenius norm; same resamples for J/R and layers'),
        target_reduction='upstream sum over future targets',final_layer='Analytic identity',
        actual_seconds=time.perf_counter()-begin,estimate=dict(seconds=60,output_bytes=50000),
        interpretation='Descriptive fitting-corpus norms, not held-out performance, causal sensitivity, or evidence separating rotation from curvature. Norm of mean / mean norm measures cancellation under averaging; magnitude depends on the native summed-target convention. Bootstrap does not model possible shared-article dependence. No settings selected from this diagnostic.'))


if __name__=='__main__':main()
