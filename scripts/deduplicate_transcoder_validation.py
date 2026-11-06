"""Remove validation documents overlapping the exact selected training windows.

Never changes training records, trains a model, or inspects model outcomes.
The original corpus/audit are retained and the new corpus is independently audited.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
import yaml
from transformers import AutoTokenizer
from ilslens.transcoders.corpus import mixed_examples,examples,audit_budget
from ilslens.experiment import write_json


def digest(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/transcoders.yaml');p.add_argument('--length',type=int,default=128);a=p.parse_args()
    source=Path(a.corpus);output=Path(a.output);config=yaml.safe_load(Path(a.config).read_text());start=time.perf_counter()
    if output.exists():raise FileExistsError('Preserve previous corpus; choose a new output')
    output.parent.mkdir(parents=True,exist_ok=True);status=output.with_suffix('.repair-status.json')
    write_json(output.with_suffix('.estimate.json'),dict(estimated_seconds=600,estimated_disk_bytes=source.stat().st_size,
        scope='Training-window scan, validation-document filtering, and independent new-corpus audit; no model training'))
    def record(state,**kwargs):write_json(status,dict(state=state,seconds=time.perf_counter()-start,**kwargs))
    try:
        tokenizer=AutoTokenizer.from_pretrained(os.environ['ILS_MODEL_PATH'],local_files_only=True)
        budget=config['token_presentations'];length=a.length
        if budget%length:raise ValueError('Budget must be divisible by window length')
        record('collecting_selected_training_windows');selected=set()
        for _,meta in mixed_examples(source,tokenizer,'train',config['chat_every'],length,cycle=False,unique=True):
            selected.add(meta['sha256'])
            if len(selected)*length==budget:break
        if len(selected)*length!=budget:raise ValueError('Source cannot meet registered training budget')
        record('finding_validation_overlap');excluded=set();overlaps=set()
        for _,meta in examples(source,tokenizer,'validation',length):
            if meta['sha256'] in selected:excluded.add(meta['document_id']);overlaps.add(meta['sha256'])
        record('writing_filtered_corpus',excluded_documents=len(excluded),overlap_windows=len(overlaps))
        temporary=output.with_suffix('.tmp');counts={};before=hashlib.sha256();after=hashlib.sha256()
        with source.open('rb') as src,temporary.open('wb') as dst:
            for line in src:
                row=json.loads(line)
                if row['split']=='train':before.update(line)
                if row['split']=='validation' and row['id'] in excluded:continue
                dst.write(line);counts[row['split']]=counts.get(row['split'],0)+1
                if row['split']=='train':after.update(line)
        if before.digest()!=after.digest():raise AssertionError('Training records changed')
        temporary.replace(output)
        manifest=dict(source_corpus_sha256=digest(source),sha256=digest(output),document_counts=counts,
            excluded_validation_document_ids=sorted(excluded),overlapping_window_count=len(overlaps),
            training_records_sha256=before.hexdigest(),training_records_unchanged=True,token_budget=budget,
            length=length,chat_every=config['chat_every'],
            policy='Drop entire validation documents sharing any token window with the exact selected training prefix; preserve training lines byte-for-byte',
            seconds_before_independent_audit=time.perf_counter()-start)
        write_json(output.with_suffix('.manifest.json'),manifest);record('independent_budget_audit')
        audit=audit_budget(output,tokenizer,budget,config['chat_every'],length)
        audit.update(corpus_sha256=manifest['sha256'],seconds_total_repair_and_audit=time.perf_counter()-start)
        write_json(output.with_suffix('.budget.json'),audit)
        if not audit['passes']:raise RuntimeError('Filtered corpus failed independent budget audit')
        record('completed',scope='Data preparation only; no model training',excluded_documents=len(excluded),audit=audit)
    except Exception as error:record('failed',error=str(error));raise

if __name__=='__main__':main()
