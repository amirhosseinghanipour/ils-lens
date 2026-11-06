"""Strict metadata-aware lens-bank loading for evaluation."""
import json
from pathlib import Path
import torch

def primary_bank(root,epsilon=.05,gamma=.9):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());config=manifest['config']
    banks={'logit':{layer:torch.eye(896) for layer in range(24)},'J-reference':{},'R-reference':{},'ILS':{}}
    for layer in range(24):
        for short,name in [('J','J-reference'),('R','R-reference')]:
            maps=[]
            for prompt in range(config['source_prompts']):
                artifact=torch.load(root/'baselines'/f'{short}-prompt-{prompt:03d}.pt',weights_only=True,map_location='cpu')
                if artifact['metadata']['fingerprint']!=manifest['fingerprint']:raise ValueError('Mismatched baseline provenance')
                maps.append(artifact['matrices'][layer])
            banks[name][layer]=torch.stack(maps).mean(0)
        artifact=torch.load(root/'fits'/f'layer-{layer:02d}-epsilon-{epsilon}-gamma-{gamma}.pt',weights_only=True,map_location='cpu')
        if artifact['metadata']['fingerprint']!=manifest['fingerprint'] or artifact['metadata']['label_tuned']:
            raise ValueError('Invalid primary ILS provenance')
        banks['ILS'][layer]=artifact['matrix']
    return banks,dict(fingerprint=manifest['fingerprint'],epsilon=epsilon,gamma=gamma,
                      prior_selection='interventional_CV_only',label_tuned={name:False for name in banks})
