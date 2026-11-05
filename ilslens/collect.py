import torch
from .hooks import patch_site, final_state
from .sampling import horizon_weights

@torch.no_grad()
def antithetic(model, ids, layer, position, delta, gammas=(0., .5, .9, 1.), skip=16):
    outputs = []
    for sign in (1, -1):
        with patch_site(model, layer, position, sign*delta):
            outputs.append(final_state(model, ids))
    # This detects noncausal leakage; no tolerance relative to the intervention.
    torch.testing.assert_close(outputs[0][:,:position], outputs[1][:,:position], atol=1e-6, rtol=0)
    difference = (outputs[0]-outputs[1])*0.5
    return torch.stack([torch.einsum('t,btd->bd',
        horizon_weights(position, ids.shape[1], gamma, skip).to(difference.device),
        difference[:,position:-1]) for gamma in gammas])

def save_samples(path, delta, response, prompt_ids, metadata):
    assert delta.dtype == response.dtype == torch.float32
    assert delta.shape[0] == response.shape[0] == len(prompt_ids)
    torch.save(dict(delta=delta, response=response, prompt_ids=prompt_ids,
                    metadata=metadata), path)
