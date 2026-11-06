"""Analytic HMM observability diagnostic; no transformer inference or training."""
import argparse,time
from pathlib import Path
import torch
from ilslens.eval.toy_hmm import sequences
from ilslens.eval.hmm_oracle import posterior_states
from ilslens.experiment import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/toy-bayes');p.add_argument('--prompts',type=int,default=4000);args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    config=dict(seed=160000,prompts=args.prompts,length=128,estimated_seconds=10,estimated_disk_mb=1,
        purpose='Exact Bayes hidden-state accuracy on non-report positions; no fitted parameters')
    write_json(root/'config.json',config);start=time.perf_counter();torch.set_num_threads(2)
    ids,truth,reports=sequences(args.prompts,config['seed']);posterior=posterior_states(ids)
    mask=~reports;mask[:,:16]=False;mask[:,-1]=False
    correct=(posterior.argmax(-1)==truth)&mask
    by_prompt=correct.sum(1)/mask.sum(1);expected=(posterior.max(-1).values*mask).sum(1)/mask.sum(1)
    gen=torch.Generator().manual_seed(config['seed']);indices=torch.randint(args.prompts,(1000,args.prompts),generator=gen)
    result=dict(config=config,accuracy=float(by_prompt.mean()),ci=by_prompt[indices].mean(1).quantile(torch.tensor([.025,.975])).tolist(),
        posterior_expected_accuracy=float(expected.mean()),valid_positions=int(mask.sum()),seconds=time.perf_counter()-start,
        interpretation='Bayes optimal expected classification ceiling for observed histories under the specified generator; empirical accuracy varies with sampled latent states',
        caveat='Separate diagnostic prompts; not a re-estimation of lens performance or a training target')
    write_json(root/'results.json',result);print(result)

if __name__=='__main__':main()
