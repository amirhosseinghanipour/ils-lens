"""Audit saved capability decisions without changing scoring or selecting items."""
import argparse,hashlib,json
from pathlib import Path
from ilslens.eval.readout_tasks import exact_match,contains_concept


def audit(records):
    rows=[]
    for task in sorted({r['item']['task'] for r in records}):
        selected=[r for r in records if r['item']['task']==task]
        strict=mentions=draws=0
        for record in selected:
            answers=record['capability_answers'];aliases=record['item']['answers']
            observed=sum(exact_match(answer,aliases) for answer in answers)
            if observed!=record['correct']:raise ValueError('Saved correctness differs from exact-match recomputation')
            expected_retained=observed>=record['threshold'] and not record['prompt_leaks'] and not record['output_leaks']
            if record['retained']!=expected_retained:raise ValueError('Saved retention differs from correctness and recorded leakage gates')
            strict+=observed;draws+=len(answers)
            mentions+=sum(any(contains_concept(answer,alias) for alias in aliases) for answer in answers)
        rows.append(dict(task=task,items=len(selected),draws=draws,exact_matches=strict,
            expected_answer_mentions=mentions,items_passing_correctness=sum(r['correct']>=r['threshold'] for r in selected),
            prompt_leak_items=sum(bool(r['prompt_leaks']) for r in selected),
            output_leak_items=sum(bool(r['output_leaks']) for r in selected),
            retained=sum(r['retained'] for r in selected)))
    retained=sum(r['retained'] for r in records)
    protocols=sorted({r.get('config',{}).get('protocol','original-raw') for r in records})
    split_counts={split:dict(items=sum(r['item']['split']==split for r in records),
        retained=sum(r['retained'] and r['item']['split']==split for r in records)) for split in ('dev','test')}
    return dict(rows=rows,items=len(records),retained=retained,protocols=protocols,splits=split_counts,
        interpretation='Expected-answer mentions are formatting/content diagnostics, NOT correctness, capability, or a relaxed gate. Original exact-match and leakage criteria remain unchanged.',
        inference=('No items satisfy this protocol; this does not identify the cause or prove absence of latent knowledge.'
                   if retained==0 else 'Retained items satisfy this protocol’s capability and leakage gates; no lens readout or causal performance is established.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--gates',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();source=Path(a.gates);result=audit(json.loads(source.read_text()))
    result['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
