from pathlib import Path
import torch
from .train import SkipTranscoder

def load_transcoders(root,model):
    result=[]
    for layer in range(len(model.model.layers)):
        artifact=torch.load(Path(root)/f'layer-{layer:02d}.pt',weights_only=True,map_location='cpu')
        metadata=artifact['metadata']
        if metadata['model_type']!=model.config.model_type or metadata['hidden_size']!=model.config.hidden_size:
            raise ValueError('Transcoder architecture mismatch')
        tc=SkipTranscoder(metadata['hidden_size'],metadata['width'],metadata['top_k']).to(model.device)
        tc.load_state_dict(artifact['state_dict']);tc.eval().requires_grad_(False);result.append(tc)
    return result
