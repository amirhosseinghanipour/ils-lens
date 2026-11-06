"""Fit 100k-reference-activation readout calibration in vocabulary blocks."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import yaml
from ilslens.model import load_local
from ilslens.data import token_windows
from ilslens.artifacts import primary_bank
from ilslens.hooks import capture_blocks
from ilslens.calibration import (fit_calibration,matrix_fingerprint,
    calibration_input_fingerprint,validate_calibration_cache)

def parse_args(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--frequencies',required=True)
    parser.add_argument('--run',default='runs/qwen-main');parser.add_argument('--output',default='runs/calibration')
    parser.add_argument('--config',default='configs/calibration.yaml')
    parser.add_argument('--device');parser.add_argument('--count',type=int)
    parser.add_argument('--epsilon',type=float);parser.add_argument('--gamma',type=float)
    parser.add_argument('--jpp')
    args=parser.parse_args(argv)
    config=yaml.safe_load(Path(args.config).read_text())
    for key in ('device','count','epsilon','gamma'):
        if getattr(args,key) is None:setattr(args,key,config[key])
    if args.count<100000:raise ValueError('Production calibration requires at least 100000 activations')
    conventions=dict(split='test',sequence_length=128,skip_first=16,exclude_final=True)
    if any(config.get(key)!=value for key,value in conventions.items()):
        raise ValueError('Calibration requires held-out test data and the registered 128/skip-16/exclude-final conventions')
    args.resolved_config={**config,**{key:getattr(args,key) for key in ('device','count','epsilon','gamma')}}
    return args

def main():
    args=parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    estimate=dict(reference_count=args.count,activation_bytes=args.count*896*24*4,
        estimated_seconds=86400,note='Unbenchmarked CPU GPD fitting may dominate GPU score calculation',split='test')
    (root/'estimate.json').write_text(json.dumps(estimate,indent=2))
    frequencies=torch.load(args.frequencies,weights_only=True,map_location='cpu')
    if frequencies['metadata']['split']!='train' or not frequencies['metadata'].get('complete_split'):
        raise ValueError('Require complete training-corpus frequencies')
    bank,metadata=primary_bank(args.run,args.epsilon,args.gamma)
    if args.jpp:
        artifact=torch.load(args.jpp,weights_only=True,map_location='cpu')
        if not artifact['metadata']['label_tuned'] or artifact['metadata']['primary_fingerprint']!=metadata['fingerprint']:
            raise ValueError('J++ provenance mismatch')
        bank['J++ (label-tuned)']=artifact['matrices']
    model,tokenizer=load_local(args.device)
    manifest_path=root/'activation-manifest.json'
    activation_key=dict(reference_count=args.count,split='test',dataset=str(Path(args.dataset).resolve()),
        model='Qwen/Qwen2.5-0.5B-Instruct',primary_fingerprint=metadata['fingerprint'])
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=activation_key:
        raise ValueError('Calibration activation provenance changed')
    (root/'resolved-config.json').write_text(json.dumps(args.resolved_config,indent=2))
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
            input_fingerprint=calibration_input_fingerprint(
                dict(**activation_key,layer=layer),matrix,frequencies['counts'])
            if path.exists():
                existing=torch.load(path,weights_only=True,map_location='cpu')
                validate_calibration_cache(existing,input_fingerprint,args.count,model.config.vocab_size)
                continue
            begin=time.perf_counter()
            h=torch.from_numpy(np.array(maps[layer])).to(model.device)
            with torch.no_grad():transported=h@matrix.to(h).T
            calibration=fit_calibration(transported,model.lm_head.weight,model.model.norm.weight,frequencies['counts'])
            calibration['metadata']=dict(primary_fingerprint=metadata['fingerprint'],lens=name,layer=layer,
                matrix_sha256=matrix_fingerprint(matrix),
                input_fingerprint=input_fingerprint,
                calibration_config=args.resolved_config,
                epsilon=args.epsilon,gamma=args.gamma,reference_split='test',label_tuned=name.startswith('J++'),seconds=time.perf_counter()-begin)
            temporary=path.with_suffix('.tmp');torch.save(calibration,temporary);temporary.replace(path)
            del h,transported
    (root/'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start)))

if __name__=='__main__':main()
