"""Exact-match capability gates and item-level readout metrics.

Capability questions and natural completions are separate: an auxiliary question
may explicitly ask for a typo correction, while leakage is checked in the
original read prompt and its own continuations. No evaluator is an LLM judge.
"""
from dataclasses import dataclass,asdict
import hashlib,json,re,unicodedata
from pathlib import Path
import numpy as np
import torch

@dataclass
class ReadoutItem:
    id:str
    task:str
    prompt:str
    capability_prompt:str
    answers:list[str]
    intermediates:list[str]
    split:str

def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split()).strip(' \t\n.,!?;:\"\'')

def contains_concept(text,concept):
    # Whole word/phrase boundaries avoid matching "art" in "earth".
    return re.search(r'(?<!\w)'+re.escape(normalize(concept))+r'(?!\w)',normalize(text)) is not None

def exact_match(answer,aliases):
    return normalize(answer) in {normalize(alias) for alias in aliases}

def dev_split(item_id,seed=0,fraction=.2):
    value=int(hashlib.sha256(f'{seed}:{item_id}'.encode()).hexdigest()[:16],16)/2**64
    return 'dev' if value<fraction else 'test'

def load_reference_items(directory,seed=0):
    result=[]
    for task in ('multihop','multilingual','typo','association','poetry'):
        entries=json.loads((Path(directory)/f'lens-eval-{task}.json').read_text())['items']
        for entry in entries:
            prompt=entry['prompt']
            if not isinstance(prompt,str):raise ValueError('Chat-list prompt requires an explicit renderer')
            concepts=entry['intermediates'];answers=[entry['target']] if 'target' in entry else concepts
            if task=='typo':question=f'Correct the misspelled final word. Reply with only the corrected word.\n{prompt}'
            elif task=='association':question=f'Name the central concept in one word.\n{prompt}'
            else:question=prompt
            identity=f'{task}:{entry["name"]}'
            result.append(ReadoutItem(identity,task,prompt,question,answers,concepts,dev_split(identity,seed)))
    return result

def load_simplified_items(path,families,seed=0):
    """Fixed auxiliary tasks, separately named so results cannot be pooled silently."""
    result=[]
    for row in json.loads(Path(path).read_text())['items']:
        task=row['task']
        if task not in families:continue
        prompt=row['prompt'];question=prompt
        if task=='typo':question='Correct the misspelled final word. Reply with only the corrected word.\n'+prompt
        elif task=='association':question='Name the central concept in one word.\n'+prompt
        identity=f'simple:{task}:{row["id"]}'
        result.append(ReadoutItem(identity,task+':simplified',prompt,question,row['answers'],row['intermediates'],dev_split(identity,seed)))
    return result

def gate_item(item,capability_answers,natural_completions,draws=10,required=8,rendered_prompt=None):
    if len(capability_answers)!=draws or len(natural_completions)!=draws:
        raise ValueError('Capability gate requires exactly ten returned samples in both modes')
    correct=[exact_match(answer,item.answers) for answer in capability_answers]
    prompt_leaks=[c for c in item.intermediates if contains_concept(item.prompt if rendered_prompt is None else rendered_prompt,c)]
    output_leaks=[i for i,text in enumerate(natural_completions) if any(contains_concept(text,c) for c in item.intermediates)]
    return dict(item=asdict(item),correct=sum(correct),draws=draws,
        prompt_leaks=prompt_leaks,output_leaks=output_leaks,
        retained=sum(correct)>=required and not prompt_leaks and not output_leaks,
        capability_answers=capability_answers,natural_completions=natural_completions,
        judge='exact_match',threshold=required,temperature=.8)

def candidate_ids(tokenizer,concept):
    result=set()
    for spelling in (concept,concept.lower(),concept.capitalize()):
        for prefix in ('',' '):
            ids=tokenizer.encode(prefix+spelling,add_special_tokens=False)
            if len(ids)==1:result.add(ids[0])
    return sorted(result)

def recall_report(ranks,item_ids,repeats=200,seed=0):
    """Ranks [items,layers,intermediates], NaN for unscoreable intermediates.

    No unscoreable item is silently called a failure or removed from the count.
    Report their count and compute item-macro recall on scoreable items only.
    """
    ranks=np.asarray(ranks,dtype=float)
    if ranks.ndim!=3 or len(item_ids)!=len(ranks) or len(set(item_ids))!=len(item_ids):
        raise ValueError('Expected unique items and item/layer/concept ranks')
    rng=np.random.default_rng(seed);rows=[]
    for layer in list(range(ranks.shape[1]))+['any']:
        values=np.fmin.reduce(ranks,axis=1) if layer=='any' else ranks[:,layer]
        valid=np.isfinite(values);scoreable=valid.any(1)
        for k in (1,10):
            scores=((values<=k)&valid).sum(1)/np.maximum(valid.sum(1),1)
            scores=scores[scoreable]
            ci=[None,None] if not len(scores) else np.quantile(
                scores[rng.integers(len(scores),size=(repeats,len(scores)))].mean(1),[.025,.975]).tolist()
            rows.append(dict(layer=layer,k=k,recall=None if not len(scores) else float(scores.mean()),
                ci=ci,n=len(scores),unscoreable_items=int((~scoreable).sum())))
    return rows

@torch.no_grad()
def generate_answers(model,tokenizer,text,draws=10,max_new_tokens=24,seed=0,temperature=.8,prompt_format='raw',system_prompt=None):
    from .prompting import render_read_prompt
    text=render_read_prompt(tokenizer,text,prompt_format,system_prompt)
    ids=tokenizer(text,add_special_tokens=False).input_ids
    return [row['text'] for row in generate_token_completions(model,tokenizer,ids,draws,max_new_tokens,seed,temperature)]

def content_tokens(token_ids,special_ids):
    """Preserve the sampled segmentation, excluding terminal stop/special tokens."""
    result=list(token_ids);special=set(special_ids)
    while result and result[-1] in special:result.pop()
    return result

@torch.no_grad()
def generate_token_completions(model,tokenizer,ids,draws=10,max_new_tokens=24,seed=0,temperature=.8,greedy=False):
    device=model.model.embed_tokens.weight.device
    ids=torch.as_tensor(ids,device=device,dtype=torch.long).reshape(1,-1)
    outputs=[]
    for index in range(draws):
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(seed+index)
            parameters={} if greedy else dict(temperature=temperature,top_p=1.,top_k=0)
            answer=model.generate(ids,do_sample=not greedy,repetition_penalty=1.,**parameters,
                max_new_tokens=max_new_tokens,pad_token_id=tokenizer.eos_token_id)
        tokens=answer[0,ids.shape[1]:].tolist()
        outputs.append(dict(text=tokenizer.decode(tokens,skip_special_tokens=True),token_ids=tokens))
    return outputs


def validate_labeled_dev(gates, metadata, gate_sha256):
    """Reject renamed duplicate prompts and mismatched capability protocols."""
    if metadata.get('dev_gate_sha256') != gate_sha256:
        raise ValueError('J++ dev gates differ from this capability evaluation')
    if 'dev_prompt_hashes' not in metadata or 'dev_item_ids' not in metadata:
        raise ValueError('J++ dev provenance lacks prompt hashes or item IDs')
    dev_ids=set(metadata['dev_item_ids']);dev_hashes=set(metadata['dev_prompt_hashes'])
    for record in gates:
        item=record['item']
        if item['split']!='test':continue
        digest=hashlib.sha256(normalize(item['prompt']).encode()).hexdigest()
        if item['id'] in dev_ids or digest in dev_hashes:
            raise ValueError('J++ dev/test leakage by item ID or normalized prompt')
