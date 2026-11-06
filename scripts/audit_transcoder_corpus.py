"""Count eligible distinct training windows before any model training."""
import argparse,hashlib,json,os,time
from pathlib import Path
import yaml
from transformers import AutoTokenizer
from ilslens.transcoders.corpus import audit_budget
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True)
    p.add_argument('--config',default='configs/transcoders.yaml')
    p.add_argument('--output',default='runs/transcoder-corpus-audit.json');args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text());start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(os.environ['ILS_MODEL_PATH'],local_files_only=True)
    result=audit_budget(args.corpus,tokenizer,config['token_presentations'],config['chat_every'])
    result.update(seconds=time.perf_counter()-start,corpus_sha256=hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
                  token_budget=config['token_presentations'],chat_every=config['chat_every'])
    write_json(args.output,result);print(json.dumps(result,indent=2))
    if not result['passes']:raise SystemExit(1)


if __name__=='__main__':main()
