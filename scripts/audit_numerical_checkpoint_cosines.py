"""Read-only partial checkpoint audit; never substitutes for full validation.

Only reporting reductions use float64. Models, Jacobians and fitted matrices
remain fp32. The running experiment and its registered decision are unchanged.
"""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--max-layer',type=int,required=True)
    p.add_argument('--output',required=True);a=p.parse_args();start=time.perf_counter();torch.set_num_threads(2)
    root=Path(a.run);manifest=json.loads((root/'manifest.json').read_text());launch=json.loads((root/'launch.json').read_text())
    config=manifest['protocol']['config'];rows=[]
    for layer in config['layers']:
        if layer>a.max_layer:continue
        for epsilon in config['epsilons']:
            path=root/f'matrices-{layer}-{epsilon}.pt'
            if not path.exists():raise FileNotFoundError('Requested snapshot is not yet complete: '+str(path))
            saved=torch.load(path,weights_only=True,map_location='cpu');meta=saved['metadata']
            if meta['fingerprint']!=manifest['fingerprint']:raise ValueError('Matrix provenance mismatch')
            x=saved['ILS'];y=saved['J_normalized']
            if x.dtype!=torch.float32 or y.dtype!=torch.float32 or x.shape!=y.shape:raise ValueError('Expected matching fp32 maps')
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise FloatingPointError('Nonfinite matrix')
            raw=float(torch.nn.functional.cosine_similarity(x.flatten(),y.flatten(),dim=0))
            xd=x.flatten().double();yd=y.flatten().double();denominator=xd.norm()*yd.norm()
            stable=float(xd.dot(yd)/denominator) if denominator>0 else None
            with path.open('rb') as file:digest=hashlib.file_digest(file,'sha256').hexdigest()
            rows.append(dict(layer=layer,epsilon=epsilon,matrix_cosine_fp32_reduction=raw,
                matrix_cosine_float64_reduction=stable,source_file=path.name,source_sha256=digest,
                N=meta['N'],gamma=meta['gamma'],lambda_=meta['lambda_']))
    write_json(a.output,dict(status='partial_reporting_audit',rows=rows,source_fingerprint=manifest['fingerprint'],
        source_commit=launch['commit'],max_layer=a.max_layer,seconds=time.perf_counter()-start,
        heldout_top10_overlap='Not available in intermediate matrix checkpoints; await full experiment result',
        sampling='Original prefix-sampled numerical contexts; not the corrected shuffled primary-fitting bank',
        limitation='Does not establish all-layer equivalence, readout overlap, causal faithfulness, or method superiority. No experiment inputs or validation thresholds changed.'))

if __name__=='__main__':main()
