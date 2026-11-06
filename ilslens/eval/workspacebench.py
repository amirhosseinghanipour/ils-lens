"""Explicit exact-match adaptation of selected WorkspaceBench banks.

This does not claim equivalence to the benchmark's LLM judge. Family-specific
renderers, gates and read positions must be preserved in each emitted item.
"""
import json
from pathlib import Path
from .readout_tasks import ReadoutItem,dev_split,normalize,exact_match

def load_family(root,family,seed=0):
    directory=Path(root)/'evals'/family
    # Use v2 only where the repository declares a v2 bank; record selected path.
    path=directory/('items_v2.json' if (directory/'items_v2.json').exists() else 'items.json')
    source=json.loads(path.read_text());entries=source['items'] if isinstance(source,dict) else source
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
        readout=row.get('readout',{'kind':'final_prompt_token','offsets':[-1]})
        group_key=f"{family}:{row.get('pair_id',row.get('scene',identity))}"
        items.append(dict(item=ReadoutItem(identity,family,prompt,question,answers,concepts,dev_split(group_key,seed)),
            pair_id=row.get('pair_id'),split_group=group_key,
            source_file=str(path.relative_to(root)),readout=readout,
            required_correct=10 if family=='chain_intermediates' else 8,
            require_greedy_correct=True,judge='exact_match',
            caveat='Poetry must be re-keyed to model commitments; strict leakage gate may exclude all such items' if family=='poetry' else None))
    return items

def paired_accuracy(records):
    groups={}
    for row in records:groups.setdefault(row['pair_id'],[]).append(row)
    complete=[rows for rows in groups.values() if len(rows)==2]
    return dict(pairs=len(complete),incomplete_pairs=len(groups)-len(complete),
                accuracy=None if not complete else sum(all(row['correct'] for row in rows) for rows in complete)/len(complete))

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
