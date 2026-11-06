"""Analytic final-block diagnostic; not a model-quality measurement.

At the final block, a site intervention changes only that site's target state.
Normalized horizons therefore yield w_gamma(0 | t) * delta. Position-dependent
weights can make a position-independent lens incomplete even without curvature.
"""
import torch
from ilslens.sampling import valid_positions,horizon_weights


def final_horizon_diagnostic(length=128,skip=16,gammas=(0.,.5,.9,1.)):
    positions=valid_positions(length,skip)
    rows=[]
    for gamma in gammas:
        gains=torch.stack([horizon_weights(int(t),length,gamma,skip)[0] for t in positions])
        mean=gains.mean();second=gains.square().mean()
        rows.append(dict(gamma=gamma,mean_gain=mean.item(),minimum_gain=gains.min().item(),
            maximum_gain=gains.max().item(),optimal_global_isotropic_r2=(mean.square()/second).item(),
            upstream_identity_isotropic_r2=(1-(gains-1).square().mean()/second).item()))
    return dict(length=length,skip_first=skip,exclude_final=True,rows=rows,
        assumptions='Uniform source positions; zero-mean perturbations independent of position with the same nonzero covariance at every position; population moments.',
        caveat='Donor perturbations need not satisfy these assumptions. This is not a ceiling for their measured R2, nor evidence of curvature or absent workspace.')
