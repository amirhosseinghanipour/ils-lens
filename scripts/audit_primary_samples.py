"""Read-only integrity snapshot of atomic primary intervention checkpoints."""
import argparse,hashlib,json,re,time
from pathlib import Path
import torch
from ilslens.experiment import write_json


def validate_shard(saved,manifest,split,prompt,layer,d=896,layers=24):
    c=manifest['config'];meta=saved['metadata']
    expected=dict(fingerprint=manifest['fingerprint'],layer=layer,target_layer=layers-1,
        split=split,epsilon=c['epsilons'],gamma=c['gammas'],delta_dist='donor',
        source_convention='block_output',target='pre_final_norm',execution_device='cuda')
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Sample provenance/conventions differ')
    prompts=c['source_prompts'] if split=='train' else c['heldout_prompts']
    total=c['samples_per_layer_epsilon'] if split=='train' else c['heldout_samples_per_layer_epsilon']
    if total%prompts or not 0<=prompt<prompts or not 0<=layer<layers:raise ValueError('Unexpected shard index/budget')
    n=total//prompts;x=saved['delta'];y=saved['response']
    if x.shape!=(n,len(c['epsilons']),d) or y.shape!=(n,len(c['epsilons']),len(c['gammas']),d):raise ValueError('Sample tensor shape differs')
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in (x,y)):raise ValueError('Samples must be finite fp32')
    positions=saved['source_positions'];donors=saved['donor_sites'];groups=saved['prompt_ids']
    if positions.shape!=(n,) or donors.shape!=(n,2) or groups.shape!=(n,):raise ValueError('Sample index shapes differ')
    if any(v.dtype!=torch.int64 for v in (positions,donors,groups)):raise ValueError('Sample indices must be int64')
    end=c['sequence_length']-1;start=c['skip_first']
    donor_count=c['donor_prompts'] if split=='train' else c['heldout_prompts']
    if not ((positions>=start)&(positions<end)).all() or not ((donors[:,1]>=start)&(donors[:,1]<end)).all():raise ValueError('Invalid source/donor position')
    if not ((donors[:,0]>=0)&(donors[:,0]<donor_count)).all() or not (groups==prompt).all():raise ValueError('Invalid donor/prompt index')
    eps=x.new_tensor(c['epsilons']);scaled=x[:,0,None,:]*(eps/eps[0])[None,:,None]
    torch.testing.assert_close(x,scaled,rtol=2e-6,atol=1e-7)
    final_error=None
    if layer==layers-1:
        weights=torch.stack([torch.stack([x.new_tensor(g).pow(torch.arange(end-int(t))).sum().reciprocal()
            for g in c['gammas']]) for t in positions])
        expected=x[:,:,None,:]*weights[:,None,:,None]
        final_error=float(((y-expected).norm(dim=-1)/expected.norm(dim=-1).clamp_min(1e-12)).max())
        if final_error>=1e-4:raise ValueError('Final-layer horizon identity failed')
    return dict(samples=n,final_layer_max_relative_error=final_error)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    begin=time.perf_counter();torch.set_num_threads(2);root=Path(a.run)
    manifest=json.loads((root/'manifest.json').read_text());files=sorted((root/'samples').glob('*/*.pt'))
    rows=[]
    for path in files:
        match=re.fullmatch(r'prompt-(\d+)-layer-(\d+)\.pt',path.name)
        if not match or path.parent.name not in ('train','validation'):raise ValueError('Unexpected sample filename')
        prompt,layer=map(int,match.groups())
        saved=torch.load(path,weights_only=True,map_location='cpu')
        row=validate_shard(saved,manifest,path.parent.name,prompt,layer)
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        rows.append(dict(path=str(path.relative_to(root)),sha256=digest,bytes=path.stat().st_size,**row))
    expected=24*(manifest['config']['source_prompts']+manifest['config']['heldout_prompts'])
    write_json(a.output,dict(status='integrity_snapshot',primary_fingerprint=manifest['fingerprint'],files=rows,
        snapshot_shards=len(rows),expected_shards=expected,all_shards_present=len(rows)==expected,
        audit_seconds=time.perf_counter()-begin,final_identity_relative_tolerance=1e-4,
        limitation='Snapshot of completed atomic files only; no fitting, held-out performance or study completion claim. Shapes and bounds do not establish statistical independence of donor draws.'))
    print(json.dumps(dict(shards=len(rows),expected=expected,seconds=time.perf_counter()-begin)))


if __name__=='__main__':main()
