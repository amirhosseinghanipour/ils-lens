"""Fail-closed gate for collecting primary data after numerical validation."""
import math


def validate_numerical_result(result,manifest):
    config=manifest['protocol']['config']
    if result.get('fingerprint')!=manifest['fingerprint']:
        raise ValueError('Numerical result fingerprint mismatch')
    expected={(layer,float(epsilon)) for layer in config['layers'] for epsilon in config['epsilons']}
    rows=result.get('rows',[])
    keys=[(row['layer'],float(row['epsilon'])) for row in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('Missing or duplicated numerical validation cells')
    if config['layers']!=list(range(24)):
        raise ValueError('Primary Qwen validation must cover all 24 layers')
    smallest=min(config['epsilons'])
    for row in rows:
        for name in ('matrix_cosine','heldout_top10_overlap'):
            if not math.isfinite(row[name]):raise ValueError('Nonfinite numerical result')
        if row['epsilon']==smallest and (row['matrix_cosine']<.95 or row['heldout_top10_overlap']<.8):
            raise ValueError('Registered smallest-epsilon equivalence criterion failed')
    if result.get('smallest_epsilon_passes_all_layers') is not True:
        raise ValueError('Numerical validation did not report completion')
    return dict(layers=24,smallest_epsilon=smallest,cells=len(rows),fingerprint=manifest['fingerprint'])
