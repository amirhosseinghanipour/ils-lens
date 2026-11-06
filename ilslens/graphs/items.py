"""Preserve benchmark tokens and read sites when preparing local graph targets."""
import torch
from ..eval.readout_tasks import candidate_ids

def tokenized_item(item,tokenizer,device):
    ids=item.get('input_ids')
    if ids is None:ids=tokenizer.encode(item['prompt'],add_special_tokens=False)
    if not ids:raise ValueError('Empty graph context')
    positions=item.get('positions',[item.get('position',len(ids)-2)])
    if not positions or any(not 0<=p<len(ids) for p in positions):raise ValueError('Invalid graph read sites')
    return torch.tensor([ids],device=device),positions

def coverage_context(ids,tokenizer):
    """Append an excluded sentinel, preserving all original causal read states.

    Task contexts may be short and may read their last token. Their coverage
    diagnostic uses skip=0 and includes every original future token, excluding
    only this new sentinel. This is explicitly distinct from WikiText fitting.
    """
    sentinel=tokenizer.eos_token_id
    if sentinel is None:raise ValueError('A defined EOS token is required for the excluded coverage sentinel')
    return torch.cat([ids,ids.new_full((1,1),sentinel)],dim=1)

def target_vector(label,tokenizer,model,matrix,phrase_artifact=None):
    candidates=candidate_ids(tokenizer,label)
    if candidates:
        token=candidates[0]
        return (model.lm_head.weight[token]*model.model.norm.weight)@matrix,dict(kind='token_lens',token_id=token)
    if phrase_artifact and label in phrase_artifact['metadata']['phrases']:
        index=phrase_artifact['metadata']['phrases'].index(label)
        return phrase_artifact['ils'][index].to(matrix),dict(kind='phrase_ils',phrase_index=index,phrase_fingerprint=phrase_artifact['metadata']['fingerprint'])
    return None,dict(kind='unsupported_phrase',reason='No fitted 2–5-token ILS phrase row for this target')
