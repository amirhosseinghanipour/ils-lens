"""Explicit offline token-count preparation; reads the complete training split."""
import argparse,json,time,os
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',default='runs/corpus-frequencies.pt');args=parser.parse_args()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    (output.parent/'frequencies-estimate.json').write_text(json.dumps(dict(estimated_seconds=7200,estimated_disk_mb=2)))
    start=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(os.environ['ILS_MODEL_PATH'],local_files_only=True)
    # Config vocabulary may include rows absent from the tokenizer vocabulary.
    config=json.loads((Path(os.environ['ILS_MODEL_PATH'])/'config.json').read_text())
    counts=torch.zeros(config['vocab_size'],dtype=torch.long);records=0;buffer=[]
    for record in load_from_disk(args.dataset)['train']:
        ids=tokenizer.encode(record['text']+'\n',add_special_tokens=False)
        buffer.extend(ids);records+=1
        if len(buffer)>=100000:
            counts+=torch.bincount(torch.tensor(buffer,dtype=torch.long),minlength=len(counts));buffer=[]
    if buffer:counts+=torch.bincount(torch.tensor(buffer,dtype=torch.long),minlength=len(counts))
    torch.save(dict(counts=counts,metadata=dict(split='train',complete_split=True,records=records,
        token_count=int(counts.sum()),seconds=time.perf_counter()-start)),output)

if __name__=='__main__':main()
