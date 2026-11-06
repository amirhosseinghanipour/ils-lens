"""Report observed checkpoint storage and the tensor-only completion lower bound."""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import torch
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.run);manifest=json.loads((root/'manifest.json').read_text());config=manifest['protocol']['config']
    launch=json.loads((root/'launch.json').read_text());paths=sorted(root.rglob('*'));files=[path for path in paths if path.is_file()]
    contexts=sorted(root.glob('layer-*-context-*.pt'));matrices=sorted(root.glob('matrices-*.pt'))
    if not contexts:raise ValueError('No completed context checkpoint to inspect')
    sample=torch.load(contexts[0],weights_only=True,map_location='cpu');d=sample['j'].shape[0]
    if sample['j'].shape!=(d,d) or sample['fingerprint']!=manifest['fingerprint']:raise ValueError('Unexpected context shape/provenance')
    tensors=[sample['j'],*sample['inputs'].values(),*sample['outputs'].values()]
    if any(t.dtype!=torch.float32 or t.shape!=(d,d) for t in tensors):raise ValueError('Expected square fp32 basis checkpoints')
    layers=len(config['layers']);epsilons=len(config['epsilons']);prompts=config['contexts'];matrix_bytes=d*d*4
    expected_contexts=layers*prompts;expected_matrices=layers*epsilons
    now=datetime.now(timezone.utc);started=datetime.fromisoformat(launch['started_utc'].replace('Z','+00:00'))
    write_json(a.output,dict(status='partial_storage_audit',observed_utc=now.isoformat(),source_commit=launch['commit'],
        source_fingerprint=manifest['fingerprint'],original_estimate=json.loads((root/'estimate.json').read_text()),
        elapsed_wall_seconds=(now-started).total_seconds(),observed_file_bytes=sum(path.stat().st_size for path in files),
        completed_context_checkpoints=len(contexts),required_context_checkpoints=expected_contexts,
        completed_matrix_checkpoints=len(matrices),required_matrix_checkpoints=expected_matrices,
        dimension=d,bytes_per_fp32_matrix=matrix_bytes,
        final_context_tensor_bytes=expected_contexts*(1+2*epsilons)*matrix_bytes,
        final_matrix_tensor_bytes=expected_matrices*2*matrix_bytes,
        final_tensor_bytes_lower_bound=(expected_contexts*(1+2*epsilons)+expected_matrices*2)*matrix_bytes,
        scope='Storage formula applies to the current numerical validation only, not primary fitting or the full study.',
        limitation='Snapshot while the job is running; elapsed wall time is not GPU kernel time. Tensor lower bound excludes serialization metadata, logs and temporary atomic-save copies. Original estimate is preserved.'))

if __name__=='__main__':main()
