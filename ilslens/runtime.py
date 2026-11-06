"""Explicit execution devices. Large jobs never silently fall back to CPU."""
import os
import subprocess
import torch

def resolve_device(requested='cuda',min_free_gib=6.,max_utilization=10):
    device=torch.device(requested)
    if device.type=='cpu':return device
    if device.type!='cuda':raise ValueError('Only CPU and CUDA are supported')
    index=device.index or 0
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    identifier=visible.split(',')[index] if visible else str(index)
    result=subprocess.run(['nvidia-smi',f'--id={identifier}',
        '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],
        check=True,capture_output=True,text=True)
    free_mib,utilization=map(float,result.stdout.strip().split(','))
    if free_mib<min_free_gib*1024 or utilization>max_utilization:
        raise RuntimeError(f'GPU busy: {free_mib/1024:.2f} GiB free, {utilization:.0f}% utilization; no job started')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU fallback is disabled')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return device

def synchronize(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
