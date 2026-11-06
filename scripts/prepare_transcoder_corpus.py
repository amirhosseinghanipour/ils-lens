"""Download a pinned FineWeb stream sample and add a supplied chat JSONL bank."""
import argparse,hashlib,json,time
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi
from ilslens.experiment import write_json

def split(identity):
    return 'validation' if int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)%10==0 else 'train'

def main():
    p=argparse.ArgumentParser();p.add_argument('--chat',help='Optional local messages JSONL; otherwise download a pinned UltraChat sample')
    p.add_argument('--web-parquet',help='Verified local FineWeb parquet shard; requires --revision')
    p.add_argument('--chat-parquet',help='Verified local UltraChat parquet shard; requires --chat-revision')
    p.add_argument('--chat-documents',type=int,default=512);p.add_argument('--chat-revision')
    p.add_argument('--output',default='data/transcoder-corpus.jsonl');p.add_argument('--web-documents',type=int,default=4096)
    p.add_argument('--revision');args=p.parse_args();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if args.web_parquet and not args.revision:raise ValueError('Local web data needs an explicit revision')
    if args.chat_parquet and (not args.chat_revision or args.chat):raise ValueError('Local chat parquet needs a revision and cannot be combined with --chat')
    if path.exists():raise FileExistsError('Use a new output path to preserve dataset provenance')
    revision=args.revision or HfApi().dataset_info('HuggingFaceFW/fineweb',timeout=30).sha
    chat_revision=None if args.chat else (args.chat_revision or HfApi().dataset_info('HuggingFaceH4/ultrachat_200k',timeout=30).sha)
    manifest=dict(dataset='HuggingFaceFW/fineweb',configuration='sample-10BT',revision=revision,web_documents=args.web_documents,
        chat_sha256=hashlib.sha256(Path(args.chat).read_bytes()).hexdigest() if args.chat else None,
        chat_dataset=None if args.chat else 'HuggingFaceH4/ultrachat_200k',chat_split=None if args.chat else 'train_sft',
        chat_revision=chat_revision,chat_documents=args.chat_documents,split_rule='sha256 document ID modulo 10; validation residue 0')
    if path.exists():raise FileExistsError('Use a new output path to preserve dataset provenance')
    factor=max(args.web_documents/4096,args.chat_documents/512)
    write_json(path.with_suffix('.estimate.json'),dict(estimated_seconds=120*factor,estimated_output_mb=64*factor,web_documents=args.web_documents,chat_documents=args.chat_documents))
    write_json(path.with_suffix('.manifest.json'),manifest)
    def parquet_rows(filename,columns):
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(filename).iter_batches(batch_size=256,columns=columns):
            yield from batch.to_pylist()
    dataset=parquet_rows(args.web_parquet,['text']) if args.web_parquet else load_dataset('HuggingFaceFW/fineweb','sample-10BT',split='train',revision=revision,streaming=True)
    temporary=path.with_suffix('.tmp');counts={};seen=set()
    with temporary.open('w') as file:
        for index,row in enumerate(dataset):
            if index>=args.web_documents:break
            identity='fineweb:'+hashlib.sha256(row['text'].encode()).hexdigest()
            if identity in seen:continue
            seen.add(identity);partition=split(identity);counts[partition]=counts.get(partition,0)+1
            file.write(json.dumps(dict(id=identity,source='FineWeb/sample-10BT',split=partition,text=row['text']))+'\n')
        if args.chat:
            chat=[json.loads(line) for line in Path(args.chat).read_text().splitlines()]
        else:
            from itertools import islice
            stream=parquet_rows(args.chat_parquet,['messages']) if args.chat_parquet else load_dataset('HuggingFaceH4/ultrachat_200k',split='train_sft',revision=chat_revision,streaming=True)
            chat=[dict(messages=row['messages'],source='UltraChat/train_sft') for row in islice(stream,args.chat_documents)]
        for row in chat:
            messages=row['messages']
            if not messages or not all(m.get('role') in ('system','user','assistant') and isinstance(m.get('content'),str) for m in messages):
                raise ValueError('Invalid chat messages')
            identity='chat:'+hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()
            if identity in seen:continue
            seen.add(identity);partition=split(identity);counts[partition]=counts.get(partition,0)+1
            file.write(json.dumps(dict(id=identity,source=row.get('source','supplied_chat'),split=partition,messages=messages))+'\n')
    def file_sha(filename):
        if not filename:return None
        digest=hashlib.sha256()
        with open(filename,'rb') as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
        return digest.hexdigest()
    temporary.replace(path);write_json(path.with_suffix('.manifest.json'),dict(manifest,document_counts=counts,seconds=time.perf_counter()-start,disk_bytes=path.stat().st_size,sha256=file_sha(path),web_parquet_sha256=file_sha(args.web_parquet),chat_parquet_sha256=file_sha(args.chat_parquet)))

if __name__=='__main__':main()
