"""Strict import of a pinned external lens; never a main-result prior."""
import hashlib
from pathlib import Path
import torch

def load_released(path,config):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=config['sha256']:raise ValueError('Released lens checksum mismatch')
    artifact=torch.load(path,weights_only=True,map_location='cpu')
    expected=list(range(config['source_layers']))
    if artifact['source_layers']!=expected or sorted(artifact['J'])!=expected:raise ValueError('Unexpected released source layers')
    if artifact['n_prompts']!=config['n_prompts'] or artifact['d_model']!=config['d_model']:raise ValueError('Released fitting metadata mismatch')
    matrices={}
    for layer,matrix in artifact['J'].items():
        if matrix.shape!=(config['d_model'],config['d_model']) or str(matrix.dtype)!=config['stored_dtype'] or not torch.isfinite(matrix).all():
            raise ValueError('Invalid released lens tensor')
        matrices[layer]=matrix.float()
    if config['target_layer']!=config['source_layers']:raise ValueError('Unverified released target convention')
    matrices[config['target_layer']]=torch.eye(config['d_model'])
    return dict(matrices=matrices,metadata=dict(config,evaluation_dtype='torch.float32',extra_baseline=True,
        prior_eligible=False,final_layer='Analytic identity appended under the upstream summed-target convention'))
