"""Strict metadata-aware lens-bank loading for evaluation."""
import json
from pathlib import Path
import torch
from .budget import baseline_count

def primary_bank(root,epsilon=.05,gamma=.9):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());config=manifest['config']
    banks={'logit':{layer:torch.eye(896) for layer in range(24)},'J-reference':{},'R-reference':{},'ILS':{}}
    mean_path=root/'baselines'/'means.pt'
    count=baseline_count(config)
    files=[root/'baselines'/f'{name}-prompt-{i:03d}.pt' for name in ('J','R') for i in range(count)]
    file_state=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    cached=torch.load(mean_path,weights_only=True,map_location='cpu') if mean_path.exists() else None
    if cached is not None and cached['fingerprint']==manifest['fingerprint'] and cached['file_state']==file_state:
        banks.update(cached['matrices'])
    else:
        for short,name in [('J','J-reference'),('R','R-reference')]:
            means={layer:torch.zeros(896,896) for layer in range(24)}
            for prompt in range(count):
                artifact=torch.load(root/'baselines'/f'{short}-prompt-{prompt:03d}.pt',weights_only=True,map_location='cpu')
                if artifact['metadata']['fingerprint']!=manifest['fingerprint']:raise ValueError('Mismatched baseline provenance')
                for layer in means:means[layer]+=artifact['matrices'][layer]/count
            banks[name]=means
        from .experiment import save_tensor
        save_tensor(mean_path,dict(fingerprint=manifest['fingerprint'],file_state=file_state,
            matrices={name:banks[name] for name in ('J-reference','R-reference')}))
    for layer in range(24):
        artifact=torch.load(root/'fits'/f'layer-{layer:02d}-epsilon-{epsilon}-gamma-{gamma}.pt',weights_only=True,map_location='cpu')
        if artifact['metadata']['fingerprint']!=manifest['fingerprint'] or artifact['metadata']['label_tuned']:
            raise ValueError('Invalid primary ILS provenance')
        banks['ILS'][layer]=artifact['matrix']
    extras={};released_path=root.parent/'released-baselines/released-jlens.pt'
    if released_path.exists():
        released=torch.load(released_path,weights_only=True,map_location='cpu');info=released['metadata']
        if info['model']!='Qwen/Qwen2.5-0.5B-Instruct' or not info['extra_baseline'] or info['prior_eligible'] or info['label_tuned']:
            raise ValueError('Invalid released extra-baseline metadata')
        name=info['name'];banks[name]=released['matrices'];extras[name]=info
    return banks,dict(fingerprint=manifest['fingerprint'],epsilon=epsilon,gamma=gamma,extra_baselines=extras,
                      prior_selection='interventional_CV_only',label_tuned={name:False for name in banks})
