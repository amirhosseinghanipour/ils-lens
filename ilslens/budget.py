"""Shared fitting-corpus exposure and prompt-fold prior membership."""
import torch


def baseline_count(config):
    policy=config.get('baseline_corpus','source_only_legacy')
    if policy=='source_only_legacy':return config['source_prompts']
    if policy!='source_and_donor':raise ValueError('Unknown baseline corpus policy')
    return config['source_prompts']+config['donor_prompts']


def prior_prompt_indices(config,training_sources):
    training_sources=torch.as_tensor(training_sources,dtype=torch.long).cpu()
    if training_sources.ndim!=1 or len(training_sources.unique())!=len(training_sources):
        raise ValueError('Training source indices must be a unique vector')
    if ((training_sources<0)|(training_sources>=config['source_prompts'])).any():
        raise ValueError('Invalid source prompt index')
    # Donors are a disjoint training bank, never held-out source prompts.
    donors=torch.arange(config['source_prompts'],baseline_count(config))
    return torch.cat((training_sources,donors))
