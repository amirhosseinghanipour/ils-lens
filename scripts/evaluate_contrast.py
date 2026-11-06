import argparse,json,time
from pathlib import Path
import torch,yaml
from ilslens.model import load_local
from ilslens.artifacts import primary_bank
from ilslens.eval.readout_tasks import generate_answers,candidate_ids
from ilslens.eval.contrastive import render_condition,matched_condition_states
from ilslens.contrast import contrast_with_null
from ilslens.readout import raw

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prompts',required=True)
    parser.add_argument('--config',default='configs/contrast.yaml');parser.add_argument('--run',default='runs/qwen-main')
    parser.add_argument('--output',default='runs/contrast');parser.add_argument('--device',default='cuda');args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());users=json.loads(Path(args.prompts).read_text())
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);(root/'estimate.json').write_text(json.dumps(config,indent=2))
    start=time.perf_counter();model,tokenizer=load_local(args.device)
    bank,metadata=primary_bank(args.run,config['epsilon'],config['gamma'])
    completions={arm:[] for arm in ('condition','control')}
    for i,user in enumerate(users):
        for arm in completions:
            prefix=render_condition(tokenizer,config[arm],user)
            text=tokenizer.decode(prefix)
            completions[arm].append(generate_answers(model,tokenizer,text,draws=1,max_new_tokens=32,seed=config['seed']+i)[0])
    layers=config['layers'];states,groups,excluded=matched_condition_states(model,tokenizer,users,
        config['condition'],config['control'],completions['condition'],completions['control'],layers,config['offsets'])
    target_ids=candidate_ids(tokenizer,config['target'])
    if not target_ids:raise ValueError('Target is not single-token; use the phrase experiment')
    rows=[]
    for name,maps in bank.items():
        for layer in layers:
            matrix=maps[layer].to(model.device)
            def read(h):return raw(h,matrix,model.lm_head.weight,model.model.norm.weight).softmax(-1)
            for mode,arms in states.items():
                statistics=contrast_with_null(arms['condition'][layer],arms['control'][layer],read,target_ids,
                    groups.to(model.device),seed=config['seed'],repeats=1000)
                for read_mode,statistic in statistics.items():
                    score_file=root/f'{name}-L{layer}-{mode}-{read_mode}.pt'
                    torch.save(statistic.pop('contrast_scores'),score_file)
                    rows.append(dict(lens=name,layer=layer,mode=mode,read_mode=read_mode,**statistic))
    (root/'results.json').write_text(json.dumps(dict(config=config,rows=rows,metadata=metadata,
        excluded_short_completion_pairs=excluded,completions=completions,seconds=time.perf_counter()-start),indent=2))

if __name__=='__main__':main()
