"""Explicit exact-match adaptation of selected WorkspaceBench banks.

This does not claim equivalence to the benchmark's LLM judge. Family-specific
renderers, gates and read positions must be preserved in each emitted item.
"""
import json,hashlib
import numpy as np
import sys
from pathlib import Path
from .readout_tasks import ReadoutItem,dev_split,normalize,exact_match

def load_reference(root):
    source=Path(root).resolve()/'src';sys.path.insert(0,str(source))
    import wsbench
    if not Path(wsbench.__file__).resolve().is_relative_to(source):raise ValueError('Unexpected WorkspaceBench import')
    from wsbench import readplan
    from wsbench.produce.render import render
    from wsbench.capable.questions import build,bank_items
    from wsbench.registry import bank_path
    return readplan,render,build,bank_items,bank_path

def load_family(root,family,seed=0):
    _,_,build,bank_items,bank_path=load_reference(root)
    path=bank_path(family);entries=bank_items(family);questions={q.id:q for q in build(family)}
    items=[]
    for index,row in enumerate(entries):
        identity=f"workspacebench:{family}:{row.get('id',row.get('name',index))}"
        if family=='relational_multihop':
            prompt=row['stimulus'];answers=[row['gold'],row['gold_alt']]
            question=prompt;concepts=[row['gold']]
        elif family=='role_bound_association':
            prompt=row['stimulus'];question=prompt+'\n'+row['asked_question']
            answers=[row['gold_label']];concepts=[row['gold_label']]
        elif family=='chain_intermediates':
            prompt=row['prompt'];question=prompt;answers=[str(row['answer'])]
            concepts=list(map(str,row['intermediates']))
        elif family in ('multihop','multilingual','typo','association','poetry'):
            prompt=row['prompt'];concepts=list(map(str,row['intermediates']))
            answers=[str(row['target'])] if 'target' in row else concepts
            if family=='typo':question='Correct the last word. Reply with only that word.\n'+prompt
            elif family=='association':question='Name the central concept in one word.\n'+prompt
            else:question=prompt
        else:raise ValueError(f'No verified adapter for {family}')
        upstream_question=questions[row['id']]
        question=upstream_question.ask;answers=upstream_question.golds
        readout=row.get('readout',{'kind':'final_prompt_token','offsets':[-1]})
        group_key=f"{family}:{row.get('pair_id',row.get('scene',identity))}"
        items.append(dict(item=ReadoutItem(identity,family,prompt,question,answers,concepts,dev_split(group_key,seed)),
            pair_id=row.get('pair_id'),split_group=group_key,
            source_file=str(path.relative_to(Path(root).resolve())),readout=readout,
            required_correct=10 if family=='chain_intermediates' else 8,
            require_greedy_correct=True,judge='exact_match',
            caveat='Poetry must be re-keyed to model commitments; strict leakage gate may exclude all such items' if family=='poetry' else None))
    return items

def paired_accuracy(records):
    groups={}
    for row in records:
        if row['pair_id'] is None:raise ValueError('A paired metric requires explicit pair IDs')
        groups.setdefault(row['pair_id'],[]).append(row)
    complete=[rows for rows in groups.values() if len(rows)==2]
    return dict(pairs=len(complete),incomplete_pairs=len(groups)-len(complete),
                accuracy=None if not complete else sum(all(row['correct'] for row in rows) for rows in complete)/len(complete))

def phrase_recall_report(rows,layers,seed=0,repeats=200):
    """Bootstrap items, or complete role-binding pairs, never individual read sites."""
    rng=np.random.default_rng(seed);results=[]
    for method in sorted({r['method'] for r in rows}):
        for family in sorted({r['family'] for r in rows if r['method']==method}):
            for layer in list(layers)+['any']:
                selected=[r for r in rows if r['method']==method and r['family']==family and (layer=='any' or r['layer']==layer)]
                by_item={}
                for row in selected:by_item.setdefault(row['item_id'],[]).append(row)
                for k in (1,10):
                    item_scores={identity:np.mean([min(r['rank'] for r in records if r['target']==target)<=k
                        for target in sorted({r['target'] for r in records})]) for identity,records in by_item.items()}
                    incomplete=0
                    if family=='role_bound_association':
                        pairs={}
                        for identity,records in by_item.items():
                            pair=records[0]['pair_id']
                            if pair is None:raise ValueError('Role-binding items require pair IDs')
                            pairs.setdefault(pair,[]).append(item_scores[identity])
                        incomplete=sum(len(values)!=2 for values in pairs.values())
                        values=np.array([all(value==1 for value in scores) for scores in pairs.values() if len(scores)==2],dtype=float)
                        unit='pair_both_directions_all_targets'
                    else:values=np.array(list(item_scores.values()));unit='item_mean_over_targets'
                    ci=[None,None] if not len(values) else np.quantile(values[rng.integers(len(values),size=(repeats,len(values)))].mean(1),[.025,.975]).tolist()
                    results.append(dict(method=method,family=family,layer=layer,k=k,items=len(by_item),units=len(values),unit=unit,
                        incomplete_pairs=incomplete,recall=None if not len(values) else float(values.mean()),ci=ci))
    return results

def overlaps_labeled_dev(item,metadata):
    """Exclude copied tasks across benchmark namespaces, not just identical IDs."""
    identity=item.id.removeprefix('workspacebench:')
    digest=hashlib.sha256(normalize(item.prompt).encode()).hexdigest()
    return identity in metadata.get('dev_item_ids',[]) or digest in metadata.get('dev_prompt_hashes',[])

def resolve_positions(tokenizer,prompt,readout):
    encoded=tokenizer(prompt,add_special_tokens=False,return_offsets_mapping=True)
    offsets=encoded['offset_mapping'];kind=readout.get('kind','final_prompt_token')
    if kind=='final_prompt_token':base=len(offsets)-1
    elif kind=='last_word_token':
        # Offsets in the bank are token offsets, not an invitation to guess a word boundary.
        base=len(offsets)
        return [base+int(offset) for offset in readout['offsets']]
    else:raise ValueError(f'Read-position convention requires an explicit adapter: {kind}')
    return [base]

def contradiction_check(readout,actual_answer,correct_answers,contrasting_answers):
    """Exact explicit contradictions only; absence is not a contradiction."""
    actual_correct=exact_match(actual_answer,correct_answers)
    read_correct=any(normalize(text)==normalize(readout) for text in correct_answers)
    read_contrast=any(normalize(text)==normalize(readout) for text in contrasting_answers)
    return dict(contradicts_actual_output=actual_correct and read_contrast,
                supports_actual_output=actual_correct and read_correct,
                judge='exact_match',unknown=not actual_correct or not (read_correct or read_contrast))
