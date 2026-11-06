"""Small CPU numerical/cost comparison; not a model-quality experiment."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from ilslens.fit import select
from ilslens.templatelens import fit_phrase_rows
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/phrase-fit-benchmark.json');args=p.parse_args()
    config=dict(seed=109,dimension=32,samples=256,phrases=24,prompts=32,penalties=[0.,.01,.1,1.],repeats=3,threads=2)
    print(json.dumps(dict(estimated_seconds=15,estimated_disk_bytes=10000,scope='Small synthetic regression validation')),flush=True)
    torch.set_num_threads(config['threads']);torch.manual_seed(config['seed'])
    x=torch.randn(config['samples'],config['dimension'])
    y=x@torch.randn(config['dimension'],config['phrases'])+1.5*torch.randn(config['samples'],config['phrases'])
    groups=torch.arange(config['prompts']).repeat_interleave(config['samples']//config['prompts'])
    def scalar():
        results=[select(x,y[:,c:c+1],groups,{'I':torch.zeros(1,config['dimension'])},config['penalties'],estimators=('ridge',),seed=config['seed']) for c in range(config['phrases'])]
        return torch.cat([r[0] for r in results]),[r[1] for r in results]
    def batched():return fit_phrase_rows(x,y,groups,config['penalties'],seed=config['seed'])[:2]
    scalar();batched();timings={'scalar':[],'batched':[]};outputs={}
    for repeat in range(config['repeats']):
        for name,function in ([('scalar',scalar),('batched',batched)] if repeat%2==0 else [('batched',batched),('scalar',scalar)]):
            start=time.perf_counter();outputs[name]=function();timings[name].append(time.perf_counter()-start)
    first,second=outputs['scalar'],outputs['batched']
    torch.testing.assert_close(first[0],second[0],atol=2e-6,rtol=2e-5)
    penalties_match=[r['lam'] for r in first[1]]==[r['lam'] for r in second[1]]
    if not penalties_match:raise AssertionError('Selected penalties differ; inspect floating-point CV behavior')
    record=dict(config=config,dtype='float32',device='cpu',timings_seconds=timings,
        selected_penalties_match=penalties_match,max_matrix_absolute_difference=float((first[0]-second[0]).abs().max()),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path('ilslens/templatelens.py'),Path('ilslens/fit.py'))},
        interpretation='Same estimator and per-phrase selection; tiny synthetic timing does not predict Qwen GPU wall time or readout quality')
    write_json(args.output,record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
