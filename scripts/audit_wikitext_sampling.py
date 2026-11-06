"""Compare legacy prefix sampling with the fixed seeded permutation, without labels."""
import argparse,collections,json,re,time
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from ilslens.data import wikitext_sequences,SAMPLING_POLICY
from ilslens.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--model',required=True)
    p.add_argument('--legacy-manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    start=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    old=json.loads(Path(a.legacy_manifest).read_text())
    _,new=wikitext_sequences(a.dataset,tokenizer,'train',32)
    banks={'legacy_source':old['source_rows'],'legacy_donor':old['donor_rows'],
           'shuffled_source':new[:16],'shuffled_donor':new[16:]}
    wanted={r['row'] for rows in banks.values() for r in rows}
    titles={};title='unknown';dataset=load_from_disk(a.dataset)['train']
    for index,record in enumerate(dataset):
        text=record['text'].strip()
        if re.fullmatch(r'= [^=].*[^=] =',text):title=text.strip('= ').strip()
        if index in wanted:titles[index]=title
        if len(titles)==len(wanted):break
    results={}
    for name,rows in banks.items():
        counts=collections.Counter(titles[r['row']] for r in rows)
        results[name]=dict(rows=[dict(r,article=titles[r['row']]) for r in rows],
            article_counts=dict(counts),distinct_articles=len(counts),largest_article_fraction=max(counts.values())/len(rows))
    write_json(a.output,dict(sampling_policy=SAMPLING_POLICY,legacy_fingerprint=old['fingerprint'],
        legacy_policy='first eligible rows in original dataset order; minimum 600 characters and 128 tokens',
        dataset=old['dataset'],banks=results,selection_used_evaluation_labels=False,
        seconds=time.perf_counter()-start,
        limitation='32 rows remain a small corpus; shuffled rows are not balanced or stratified by article.'))


if __name__=='__main__':main()
