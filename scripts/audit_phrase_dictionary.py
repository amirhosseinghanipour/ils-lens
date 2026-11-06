"""Audit continuation token lengths without loading model weights."""
import argparse,hashlib,json,os,time
from pathlib import Path
from transformers import AutoTokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--dictionary',default='configs/phrases.json')
    p.add_argument('--model',default=os.environ.get('ILS_MODEL_PATH'))
    p.add_argument('--output',required=True);a=p.parse_args()
    if not a.model:raise ValueError('Supply --model or ILS_MODEL_PATH')
    start=time.perf_counter();path=Path(a.dictionary)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    rows=[]
    for phrase in json.loads(path.read_text()):
        tokens=tokenizer.encode(' '+phrase,add_special_tokens=False)
        rows.append(dict(phrase=phrase,token_ids=tokens,token_count=len(tokens),passes=2<=len(tokens)<=5))
    result=dict(status='token_length_audit',all_pass=all(r['passes'] for r in rows) and bool(rows),
        model=Path(a.model).name,
        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        dictionary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows,
        seconds=time.perf_counter()-start,model_loaded=False,
        scope='Fixed dictionary only; evaluation-derived target phrases require a later audit. Leading space matches the phrase runner.')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(all_pass=result['all_pass'],phrases=len(rows),seconds=result['seconds'])))
    if not result['all_pass']:raise SystemExit(1)


if __name__=='__main__':main()
