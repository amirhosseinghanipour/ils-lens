"""Tokenizer-only complete training-split counts; no model loading or fitting."""
import argparse,json,time,os,hashlib
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from ilslens.experiment import fingerprint,save_tensor,write_json


def count_tokens(dataset,tokenizer,vocabulary,flush_tokens=100000,progress=None):
    if flush_tokens<1 or vocabulary<1:raise ValueError('Positive vocabulary and flush size required')
    counts=torch.zeros(vocabulary,dtype=torch.long);records=0;buffer=[]
    def flush():
        if buffer:
            ids=torch.tensor(buffer,dtype=torch.long)
            if int(ids.min())<0 or int(ids.max())>=vocabulary:raise ValueError('Token outside model vocabulary')
            counts.add_(torch.bincount(ids,minlength=vocabulary));buffer.clear()
    for record in dataset:
        buffer.extend(tokenizer.encode(record['text']+'\n',add_special_tokens=False));records+=1
        if len(buffer)>=flush_tokens:flush()
        if progress is not None and records%10000==0:progress(records,int(counts.sum())+len(buffer))
    flush()
    return counts,records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',default='runs/corpus-frequencies.pt');parser.add_argument('--threads',type=int,default=2);args=parser.parse_args()
    torch.set_num_threads(args.threads)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output.with_suffix('.estimate.json'),dict(estimated_seconds=7200,estimated_disk_mb=2,scope='Tokenizer-only complete training split; no model fitting'))
    start=time.perf_counter();model=Path(os.environ['ILS_MODEL_PATH']);dataset_path=Path(args.dataset)
    config=json.loads((model/'config.json').read_text());dataset=load_from_disk(args.dataset)['train']
    provenance_path=dataset_path.parent/'wikitext-manifest.json';source=json.loads(provenance_path.read_text())
    if len(dataset)!=source['rows']['train'] or dataset._fingerprint!=source['fingerprints']['train']:
        raise ValueError('Loaded training split differs from its pinned dataset manifest')
    provenance=dict(dataset=source,dataset_split_fingerprint=dataset._fingerprint,
        tokenizer_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (model/'tokenizer.json',model/'tokenizer_config.json',model/'config.json')},
        tokenization="Every training row text plus newline; add_special_tokens=False",vocabulary=config['vocab_size'])
    identity=fingerprint(provenance)
    if output.exists():
        existing=torch.load(output,weights_only=True,map_location='cpu');meta=existing['metadata'];counts=existing['counts']
        if meta.get('fingerprint')!=identity or not meta.get('complete_split') or meta.get('records')!=len(dataset):
            raise ValueError('Existing frequencies have unverified or different provenance; choose another output')
        if counts.dtype!=torch.long or counts.shape!=(config['vocab_size'],) or bool((counts<0).any()) or int(counts.sum())!=meta['token_count']:
            raise ValueError('Invalid stored frequency counts')
        print(json.dumps(dict(status='verified_existing',records=meta['records'],token_count=meta['token_count'])),flush=True);return
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    def progress(records,tokens):
        record=dict(stage='token-counting',records=records,total_records=len(dataset),tokens=tokens,seconds=time.perf_counter()-start)
        write_json(output.with_suffix('.status.json'),record);print(json.dumps(record),flush=True)
    counts,records=count_tokens(dataset,tokenizer,config['vocab_size'],progress=progress)
    if records!=len(dataset):raise AssertionError('Training split was not fully counted')
    metadata=dict(split='train',complete_split=True,records=records,token_count=int(counts.sum()),
        seconds=time.perf_counter()-start,fingerprint=identity,provenance=provenance,threads=args.threads)
    save_tensor(output,dict(counts=counts,metadata=metadata))
    write_json(output.with_suffix('.metadata.json'),metadata)
    write_json(output.with_suffix('.status.json'),dict(status='completed',**metadata));print(json.dumps(dict(status='completed',records=records,token_count=metadata['token_count'],seconds=metadata['seconds'])),flush=True)

if __name__=='__main__':main()
