"""Fit 100k-reference-activation readout calibration in vocabulary blocks."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from ilslens.model import load_local
from ilslens.data import token_windows
from ilslens.artifacts import primary_bank
from ilslens.hooks import capture_blocks
from ilslens.calibration import fit_calibration

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--frequencies',required=True)
    parser.add_argument('--run',default='runs/qwen-main');parser.add_argument('--output',default='runs/calibration')
    parser.add_argument('--device',default='cuda');parser.add_argument('--count',type=int,default=100000)
    parser.add_argument('--epsilon',type=float,default=.05);parser.add_argument('--gamma',type=float,default=.9)
    args=parser.parse_args()
    if args.count<100000:raise ValueError('Production calibration requires at least 100000 activations')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    estimate=dict(reference_count=args.count,activation_bytes=args.count*896*24*4,
        estimated_seconds=86400,note='Unbenchmarked CPU GPD fitting may dominate GPU score calculation',split='test')
    (root/'estimate.json').write_text(json.dumps(estimate,indent=2))
    frequencies=torch.load(args.frequencies,weights_only=True,map_location='cpu')
    if frequencies['metadata']['split']!='train' or not frequencies['metadata'].get('complete_split'):
        raise ValueError('Require complete training-corpus frequencies')
    bank,metadata=primary_bank(args.run,args.epsilon,args.gamma)
    model,tokenizer=load_local(args.device)
    manifest_path=root/'activation-manifest.json'
    activation_key=dict(reference_count=args.count,split='test',dataset=str(Path(args.dataset).resolve()),
        model='Qwen/Qwen2.5-0.5B-Instruct',primary_fingerprint=metadata['fingerprint'])
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=activation_key:
        raise ValueError('Calibration activation provenance changed')
    complete=(root/'activations-complete.json').exists()
    maps={layer:np.memmap(root/f'layer-{layer:02d}.f32',mode='r+' if complete else 'w+',dtype='float32',shape=(args.count,896)) for layer in range(24)}
    if not complete:
        manifest_path.write_text(json.dumps(activation_key,indent=2));count=0
        for tokens,offset in token_windows(args.dataset,tokenizer,'test'):
            with torch.no_grad(),capture_blocks(model,range(24)) as states:model.model(tokens[None].to(model.device),use_cache=False)
            n=min(111,args.count-count)
            for layer in maps:maps[layer][count:count+n]=states[layer][0,16:16+n].cpu().numpy()
            count+=n
            if count>=args.count:break
        if count<args.count:raise ValueError(f'Only {count} reference activations available; refusing calibration')
        for mmap in maps.values():mmap.flush()
        (root/'activations-complete.json').write_text(json.dumps(dict(count=count,last_token_window_offset=offset)))
    for name,matrices in bank.items():
        destination=root/name;destination.mkdir(exist_ok=True)
        for layer,matrix in matrices.items():
            path=destination/f'layer-{layer:02d}.pt'
            if path.exists():continue
            begin=time.perf_counter()
            h=torch.from_numpy(np.array(maps[layer])).to(model.device)
            with torch.no_grad():transported=h@matrix.to(h).T
            calibration=fit_calibration(transported,model.lm_head.weight,model.model.norm.weight,frequencies['counts'])
            calibration['metadata']=dict(primary_fingerprint=metadata['fingerprint'],lens=name,layer=layer,
                epsilon=args.epsilon,gamma=args.gamma,reference_split='test',label_tuned=False,seconds=time.perf_counter()-begin)
            temporary=path.with_suffix('.tmp');torch.save(calibration,temporary);temporary.replace(path)
            del h,transported
    (root/'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start)))

if __name__=='__main__':main()
