"""Matched generated-token offsets for on- and off-policy contrasts."""
import torch
from ..hooks import capture_blocks

def render_condition(tokenizer,system,user):
    messages=([{'role':'system','content':system}] if system else [])+[{'role':'user','content':user}]
    return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)

@torch.no_grad()
def completion_states(model,prefix_ids,completion_ids,layers,offsets):
    if not offsets or min(offsets)<0 or max(offsets)>=len(completion_ids):raise ValueError('Unmatched or unavailable completion offset')
    device=model.model.embed_tokens.weight.device
    ids=torch.tensor([list(prefix_ids)+list(completion_ids)],device=device)
    with capture_blocks(model,layers) as states:model.model(ids,use_cache=False)
    positions=torch.tensor([len(prefix_ids)+offset for offset in offsets],device=device)
    return {layer:h[0,positions].clone() for layer,h in states.items()}

def matched_condition_states(model,tokenizer,users,condition,control,condition_completions,control_completions,layers,offsets):
    modes={mode:{arm:{layer:[] for layer in layers} for arm in ('condition','control')} for mode in ('on_policy','off_policy')}
    groups=[];excluded=[]
    for i,user in enumerate(users):
        first=condition_completions[i];second=control_completions[i]
        if isinstance(first,str) or isinstance(second,str):raise ValueError('Pass exact generated token IDs; on-policy text must not be retokenized')
        if min(len(first),len(second))<=max(offsets):excluded.append(i);continue
        prefixes={arm:render_condition(tokenizer,system,user) for arm,system in [('condition',condition),('control',control)]}
        for mode in modes:
            for arm in modes[mode]:
                tokens=first if mode=='on_policy' and arm=='condition' else second
                h=completion_states(model,prefixes[arm],tokens,layers,offsets)
                for layer in layers:modes[mode][arm][layer].append(h[layer])
        groups.extend([i]*len(offsets))
    if not groups:raise ValueError('No pairs have all requested token offsets')
    return {mode:{arm:{layer:torch.cat(values) for layer,values in states.items()} for arm,states in arms.items()} for mode,arms in modes.items()},torch.tensor(groups),excluded
