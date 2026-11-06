"""Validate the complete registered Qwen grid before exporting research rows."""
import json
import math
from pathlib import Path
import torch


def primary_rows(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    config=manifest['config']
    expected={(layer,epsilon,gamma) for layer in range(24)
              for epsilon in config['epsilons'] for gamma in config['gammas']}
    paths=sorted((root/'fits').glob('layer-*-epsilon-*-gamma-*.pt'))
    if len(paths)!=len(expected):
        raise ValueError(f'Only {len(paths)}/{len(expected)} fitted cells; refusing complete report')
    seen=set();rows=[]
    for path in paths:
        data=torch.load(path,weights_only=True,map_location='cpu')
        meta=data['metadata']
        key=(meta['layer'],meta['epsilon'],meta['gamma'])
        if key not in expected or key in seen:
            raise ValueError(f'Unexpected or duplicate fitted cell: {key}')
        seen.add(key)
        required=dict(fingerprint=manifest['fingerprint'],target_layer=23,
                      source='block_output',target='pre_final_norm',
                      baseline_reduction='upstream_sum',ils_reduction='normalized_horizon',
                      label_tuned=False,N=config['samples_per_layer_epsilon'])
        if any(meta.get(k)!=v for k,v in required.items()):
            raise ValueError(f'Fitted cell provenance or conventions mismatch: {key}')
        if set(data['heldout'])!={'I','J','R','ILS'}:
            raise ValueError(f'Incomplete baseline measurements: {key}')
        for lens,metrics in data['heldout'].items():
            values=[metrics['r2'],*metrics['ci']]
            if len(metrics['ci'])!=2 or any(v is not None and not math.isfinite(v) for v in values):
                raise ValueError(f'Invalid held-out statistics: {key}, {lens}')
            rows.append(dict(**meta,lens=lens,**metrics,selection=data['selection']))
    if seen!=expected:raise ValueError('Incomplete fitted grid')
    return config,rows
