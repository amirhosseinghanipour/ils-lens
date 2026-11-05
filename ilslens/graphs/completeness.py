def completeness(features,embeddings,errors):
    """Absolute attribution mass; signed cancellation cannot inflate coverage."""
    explained = features.abs().sum()+embeddings.abs().sum()
    total = explained+errors.abs().sum()
    return (explained/total).item() if total>0 else float('nan')
