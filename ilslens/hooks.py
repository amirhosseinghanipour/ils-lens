"""Zero-based block OUTPUT hooks; final target is before final RMSNorm."""
from contextlib import contextmanager
import torch

def tensor_output(output):
    return output[0] if isinstance(output, tuple) else output

def replace_output(output, h):
    return (h, *output[1:]) if isinstance(output, tuple) else h

@contextmanager
def capture_blocks(model, indices):
    values, handles = {}, []
    def hook(i):
        def capture(module, args, output):
            values[i] = tensor_output(output)
        return capture
    try:
        for i in indices:
            handles.append(model.model.layers[i].register_forward_hook(hook(i)))
        yield values
    finally:
        for h in handles:
            h.remove()

@contextmanager
def patch_site(model, layer, position, delta):
    def patch(module, args, output):
        h = tensor_output(output).clone()
        h[:, position] = h[:, position] + delta
        return replace_output(output, h)
    handle = model.model.layers[layer].register_forward_hook(patch)
    try:
        yield
    finally:
        handle.remove()

def final_state(model, ids):
    with capture_blocks(model, [len(model.model.layers)-1]) as states:
        model.model(input_ids=ids, use_cache=False)
    return states[len(model.model.layers)-1]
