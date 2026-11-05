import argparse,json,os,resource,time
from pathlib import Path
from contextlib import nullcontext
import torch,yaml
from transformers import AutoModelForCausalLM,AutoTokenizer
from ilslens.data import wikitext_sequences
from ilslens.reference import load_reference,token_adapter
from ilslens.baselines import lrp_rules

parser=argparse.ArgumentParser()
parser.add_argument('--dataset',required=True)
parser.add_argument('--reference',required=True)
parser.add_argument('--config',default='configs/reference-benchmark.yaml')
parser.add_argument('--output',default='runs/reference-benchmark')
args=parser.parse_args()
config=yaml.safe_load(Path(args.config).read_text())
output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
(output/'estimate.json').write_text(json.dumps(config,indent=2))
torch.set_num_threads(config['threads']);torch.manual_seed(config['seed'])
path=os.environ['ILS_MODEL_PATH'];start=time.perf_counter()
model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32,
    attn_implementation='eager').eval().requires_grad_(False)
tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
ids,provenance=wikitext_sequences(args.dataset,tokenizer,'train',config['prompts'])
reference=load_reference(args.reference)
adapter=token_adapter(reference,model,tokenizer,ids)
timings={}
for name,context in [('J-reference',nullcontext()),('R-reference',lrp_rules(model))]:
    before=time.perf_counter()
    with context:
        lens=reference.fit(adapter,[str(i) for i in range(len(ids))],
            dim_batch=config['dim_batch'],max_seq_len=128,skip_first=16,
            checkpoint_path=str(output/f'{name}-checkpoint.pt'),resume=True)
    matrices=lens.jacobians
    matrices[len(model.model.layers)-1]=torch.eye(model.config.hidden_size)
    metadata=dict(config,model='Qwen/Qwen2.5-0.5B-Instruct',data=provenance,
                  target_layer=len(model.model.layers)-1,label_tuned=False,
                  reference_revision='581d398613e5602a5af361e1c34d3a92ea82ba8e')
    torch.save(dict(matrices=matrices,metadata=metadata),output/f'{name}.pt')
    timings[name]=time.perf_counter()-before
    print(name,timings[name],flush=True)
timings.update(total_seconds=time.perf_counter()-start,
    maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(output/'timing.json').write_text(json.dumps(timings,indent=2))
