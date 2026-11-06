"""Prompt-level preference measurements; generation draws are not independent items."""
import numpy as np
from .readout_tasks import contains_concept,generate_answers
from .contrastive import render_condition

def measure_preference(model,tokenizer,prompts,concept,draws=10,seed=0,max_new_tokens=32):
    records=[]
    for i,prompt in enumerate(prompts):
        text=tokenizer.decode(render_condition(tokenizer,None,prompt))
        completions=generate_answers(model,tokenizer,text,draws=draws,seed=seed+100*i,max_new_tokens=max_new_tokens)
        hits=[contains_concept(answer,concept) or contains_concept(answer,concept+'s') for answer in completions]
        records.append(dict(prompt=prompt,completions=completions,hits=hits,rate=float(np.mean(hits))))
    return records

def preference_difference(condition,control,seed=0,repeats=1000):
    if [r['prompt'] for r in condition]!=[r['prompt'] for r in control] or not condition:
        raise ValueError('Nonempty matched prompt records required')
    delta=np.array([a['rate']-b['rate'] for a,b in zip(condition,control)])
    rng=np.random.default_rng(seed)
    estimates=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(repeats)])
    return dict(difference=float(delta.mean()),ci=np.quantile(estimates,[.025,.975]).tolist(),
                prompts=len(delta),bootstrap_unit='prompt',statistically_detected=bool(np.quantile(estimates,.025)>0))
