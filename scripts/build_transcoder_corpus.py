"""Verify pinned shards, prepare a larger corpus, and audit its training budget."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import yaml
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--shards',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/transcoder-corpus.yaml');p.add_argument('--download-pid',type=int)
    args=p.parse_args();config=yaml.safe_load(Path(args.config).read_text());output=Path(args.output).resolve()
    shards=Path(args.shards).resolve();started=time.perf_counter();status=output.with_suffix('.build-status.json')
    def record(state,**details):write_json(status,dict(state=state,seconds=time.perf_counter()-started,**details))
    try:
        if args.download_pid:
            record('waiting_for_download',download_pid=args.download_pid)
            proc=Path('/proc')/str(args.download_pid)/'stat'
            while proc.exists():
                try:
                    if proc.read_text().rsplit(')',1)[1].split()[0]=='Z':break
                except FileNotFoundError:break
                time.sleep(15)
        record('verifying_shards',estimate={k:v for k,v in config.items() if k.startswith('estimated_')})
        subprocess.run([sys.executable,'scripts/download_transcoder_shards.py','--output',str(shards)],check=True)
        # Existing corpus data must be independently audited rather than silently
        # overwritten or assumed to have been built with this configuration.
        if output.exists():raise FileExistsError('Use a new corpus output path')
        record('preparing_corpus')
        subprocess.run([sys.executable,'scripts/prepare_transcoder_corpus.py','--web-parquet',str(shards/'fineweb-000.parquet'),
            '--chat-parquet',str(shards/'ultrachat-train-sft-000.parquet'),'--web-documents',str(config['web_documents']),
            '--chat-documents',str(config['chat_documents']),'--revision',config['web_revision'],
            '--chat-revision',config['chat_revision'],'--output',str(output)],check=True)
        record('auditing_budget')
        subprocess.run([sys.executable,'scripts/audit_transcoder_corpus.py','--corpus',str(output),
            '--output',str(output.with_suffix('.budget.json'))],check=True)
        record('completed',scope='Verified corpus and sufficient distinct-window budget; no model training')
    except Exception as error:
        record('failed',error=str(error));raise


if __name__=='__main__':main()
