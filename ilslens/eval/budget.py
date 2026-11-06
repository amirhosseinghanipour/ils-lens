"""Itemwise budget closure, with missing graph measurements kept unknown."""
import numpy as np

def close_budget(items,coverage_threshold=.5,completeness_threshold=.5,repeats=200,seed=0):
    rows=[]
    for item in items:
        coverage=item.get('lens_r2');completeness=item.get('graph_completeness')
        for name,value in (('lens_r2',coverage),('graph_completeness',completeness)):
            if value is not None and not np.isfinite(value):
                raise ValueError(f'{name} must be finite or explicitly missing (None)')
        if completeness is not None and not 0<=completeness<=1:raise ValueError('Graph completeness must be in [0,1]')
        known=coverage is not None and completeness is not None
        rows.append(dict(item,unexplained=None if coverage is None else 1-coverage,
            dark_matter=(coverage<coverage_threshold and completeness<completeness_threshold) if known else None))
    ids=sorted(set(row['item_id'] for row in rows))
    rng=np.random.default_rng(seed);estimates=[]
    known=[row for row in rows if row['dark_matter'] is not None]
    for _ in range(repeats):
        sample=[row for item in rng.choice(ids,len(ids),replace=True) for row in rows
                if row['item_id']==item and row['dark_matter'] is not None] if ids else []
        if sample:estimates.append(np.mean([row['dark_matter'] for row in sample]))
    return dict(rows=rows,thresholds=dict(coverage=coverage_threshold,completeness=completeness_threshold),
        fraction=None if not known else float(np.mean([row['dark_matter'] for row in known])),
        ci=[None,None] if not estimates else np.quantile(estimates,[.025,.975]).tolist(),
        known_cases=len(known),unknown_cases=len(rows)-len(known),
        dark_cases=[row for row in rows if row['dark_matter'] is True])
