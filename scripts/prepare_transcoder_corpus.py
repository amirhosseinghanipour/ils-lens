"""Download a pinned FineWeb stream sample and add a supplied chat JSONL bank."""
import argparse,hashlib,json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi
from ilslens.experiment import write_json

def split(identity):
    return 'validation' if int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)%10==0 else 'train'

def main():
    p=argparse.ArgumentParser();p.add_argument('--chat',required=True,help='JSONL with messages; source and id optional')
    p.add_argument('--output',default='data/transcoder-corpus.jsonl');p.add_argument('--web-documents',type=int,default=4096)
    p.add_argument('--revision');args=p.parse_args();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    revision=args.revision or HfApi().dataset_info('HuggingFaceFW/fineweb').sha
    manifest=dict(dataset='HuggingFaceFW/fineweb',configuration='sample-10BT',revision=revision,web_documents=args.web_documents,
        chat_sha256=hashlib.sha256(Path(args.chat).read_bytes()).hexdigest(),split_rule='sha256 document ID modulo 10; validation residue 0')
    if path.exists():raise FileExistsError('Use a new output path to preserve dataset provenance')
    write_json(path.with_suffix('.manifest.json'),manifest)
    dataset=load_dataset('HuggingFaceFW/fineweb','sample-10BT',split='train',revision=revision,streaming=True)
    temporary=path.with_suffix('.tmp');counts={};seen=set()
    with temporary.open('w') as file:
        for index,row in enumerate(dataset):
            if index>=args.web_documents:break
            identity='fineweb:'+hashlib.sha256(row['text'].encode()).hexdigest()
            if identity in seen:continue
            seen.add(identity);partition=split(identity);counts[partition]=counts.get(partition,0)+1
            file.write(json.dumps(dict(id=identity,source='FineWeb/sample-10BT',split=partition,text=row['text']))+'\n')
        with Path(args.chat).open() as chat:
            for line in chat:
                row=json.loads(line);messages=row['messages']
                if not messages or not all(m.get('role') in ('system','user','assistant') and isinstance(m.get('content'),str) for m in messages):
                    raise ValueError('Invalid chat messages')
                identity='chat:'+hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()
                if identity in seen:continue
                seen.add(identity);partition=split(identity);counts[partition]=counts.get(partition,0)+1
                file.write(json.dumps(dict(id=identity,source=row.get('source','supplied_chat'),split=partition,messages=messages))+'\n')
    temporary.replace(path);write_json(path.with_suffix('.manifest.json'),dict(manifest,document_counts=counts,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

if __name__=='__main__':main()
