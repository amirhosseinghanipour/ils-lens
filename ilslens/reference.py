"""Adapters call upstream fitting unchanged; never relabel its target sum as a mean."""
from pathlib import Path
import sys

def load_reference(path):
    path=Path(path).resolve()
    if not (path/'jlens/fitting.py').is_file():
        raise FileNotFoundError(path/'jlens/fitting.py')
    sys.path.insert(0,str(path))
    import jlens
    if not Path(jlens.__file__).resolve().is_relative_to(path):
        raise RuntimeError('Another jlens implementation is already imported')
    return jlens

def token_adapter(reference,model,tokenizer,sequences):
    adapter=reference.from_hf(model,tokenizer,force_bos=False)
    # The text argument identifies an already-tokenized sequence. No decode /
    # re-tokenize round trip and no implicit BOS insertion is permitted.
    def encode(key,*,max_length=128):
        return sequences[int(key),:max_length][None].to(adapter.input_device)
    adapter.encode=encode
    return adapter
