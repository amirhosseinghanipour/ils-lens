"""Recapture the vendored WSB chat prompts on this model; exact evidence only."""
import argparse,importlib.util,json,subprocess
from pathlib import Path
import torch,yaml
from ilslens.model import load_local
from ilslens.artifacts import primary_bank
from ilslens.eval.contrastive import render_condition
from ilslens.eval.readout_tasks import generate_token_completions
from ilslens.eval.hallucination import exact_readout_evidence
from ilslens.hooks import capture_blocks
from ilslens.readout import raw
from ilslens.experiment import Experiment,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--config',default='configs/hallucination.yaml')
    p.add_argument('--run',default='runs/qwen-main');p.add_argument('--output',default='runs/hallucination');p.add_argument('--device',default='cuda');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());revision=subprocess.check_output(['git','-C',args.reference,'rev-parse','HEAD'],text=True).strip()
    source=json.loads((Path(args.reference)/'data/wsbench_chat/items.json').read_text())['items']
    run=Experiment(args.output,dict(config=config,reference_revision=revision,prompts=[dict(id=r['id'],prompt=r['prompt']) for r in source],device=args.device))
    path=Path(args.reference)/'src/hallucination_bench/evals/wsbench_chat/expand.py'
    spec=importlib.util.spec_from_file_location('_ils_hallucination_sites',path);sites=importlib.util.module_from_spec(spec);spec.loader.exec_module(sites)
    model,tokenizer=load_local(args.device);bank,metadata=primary_bank(args.run,config['epsilon'],config['gamma']);rows=[];excluded=[]
    for index,item in enumerate(source):
        captured=run.path/f'capture-{index:04d}.json'
        if captured.exists():record=json.loads(captured.read_text())
        else:
            prefix=render_condition(tokenizer,None,item['prompt'])
            generation=generate_token_completions(model,tokenizer,prefix,draws=1,seed=config['seed']+index,
                max_new_tokens=config['max_new_tokens'],temperature=1.)[0]
            generated=generation['token_ids'];special=set(tokenizer.all_special_ids)
            while generated and generated[-1] in special:generated.pop()
            ids=prefix+generated;selected=sites.item_sites(tokenizer.decode,ids,len(prefix),config['max_sites'])
            record=dict(id=item['id'],prompt=item['prompt'],input_ids=ids,response=tokenizer.decode(generated),response_start=len(prefix),sites=selected)
            write_json(captured,record)
        if not record['sites']:excluded.append(item['id']);continue
        tokens=torch.tensor([record['input_ids']],device=model.device)
        with torch.no_grad(),capture_blocks(model,config['layers']) as states:model.model(tokens,use_cache=False)
        for name,matrices in bank.items():
            for layer in config['layers']:
                for site in record['sites']:
                    with torch.no_grad():scores=raw(states[layer][0,site['pos']],matrices[layer].to(model.device),model.lm_head.weight,model.model.norm.weight)
                    token_ids=scores.topk(10).indices.tolist();readouts=[tokenizer.decode([i]) for i in token_ids]
                    rows.append(dict(item_id=item['id'],lens=name,layer=layer,site=site,readouts=readouts,token_ids=token_ids,
                        **exact_readout_evidence(readouts,record['response'])))
        run.event('hallucination-capture',item=item['id'],sites=len(record['sites']))
    run.result(dict(rows=rows,metadata=metadata,excluded_no_sites=excluded,hallucination_rate=None,
        unjudged_cells=len(rows),judge='exact_match_only',official_index_computed=False,
        limitation='No configured external judge. Exact token evidence is exported; unjudged contradictions remain unknown, never counted as zero hallucinations.',
        adaptation='Original WSB chat prompts, new on-policy completions and tokenizer-specific punctuation sites; original 27B responses are not reused'))

if __name__=='__main__':main()
