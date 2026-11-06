import torch
from ..fit import fit,r2

ABLATION_DIFFERENCE_ORDER = 'per-position-before-horizon-v2'

def effective_rank(matrix):
    s=torch.linalg.svdvals(matrix)
    if s.sum()==0:return 0.
    p=s/s.sum()
    return torch.exp(-(p*p.clamp_min(1e-30).log()).sum()).item()

def procrustes_experts(matrices):
    """Align polar rotations to expert zero, fixed before readout evaluation.

    This is a diagnostic coordinate change, not an equivalent causal map.
    """
    if matrices.ndim!=3 or matrices.shape[-1]!=matrices.shape[-2] or len(matrices)==0:
        raise ValueError('Expected nonempty square expert maps')
    if not torch.isfinite(matrices).all():raise ValueError('Nonfinite expert maps')
    u,s,vh=torch.linalg.svd(matrices)
    rotations=u@vh
    positive=vh.transpose(-1,-2)@torch.diag_embed(s)@vh
    reference=rotations[0]
    aligned=[]
    for rotation,scale in zip(rotations,positive):
        left,_,right=torch.linalg.svd(rotation.T@reference)
        q=left@right
        aligned.append(rotation@q@scale)
    plain=matrices.mean(0);aligned_maps=torch.stack(aligned);aligned=aligned_maps.mean(0)
    def dispersion(values):return (values-values.mean(0)).square().sum(dim=(-1,-2)).mean().item()
    total=matrices.square().sum(dim=(-1,-2)).mean().item()
    diagnostics=dict(raw_dispersion=dispersion(matrices),aligned_dispersion=dispersion(aligned_maps),
        positive_factor_dispersion=dispersion(positive),
        rotation_chordal_dispersion=dispersion(rotations),
        mean_expert_squared_norm=total,
        interpretation='Square polar rotations align to the reference by construction; residual positive-factor disagreement need not be curvature.')
    return dict(plain=plain,aligned=aligned,plain_rank=effective_rank(plain),
                aligned_rank=effective_rank(aligned),polar_diagnostics=diagnostics)

def learning_curve(x,y,prompt_ids,held_x,held_y,prior,lam,estimator='ridge'):
    results=[];d=x.shape[1]
    for n in (d,2*d,4*d,8*d):
        if n>len(x):continue
        indices=balanced_prefix(prompt_ids,n)
        matrix=fit(x[indices],y[indices],lam,prior,estimator,prompt_ids[indices])
        results.append(dict(N=n,r2=r2(held_y,held_x@matrix.T)))
    return results

def balanced_prefix(prompt_ids,n):
    """Nested round-robin prefixes retain prompt diversity at every sample budget."""
    if not 0<n<=len(prompt_ids):raise ValueError('Invalid learning-curve sample budget')
    rows=[torch.where(prompt_ids==group)[0].tolist() for group in prompt_ids.unique()]
    order=[row[offset] for offset in range(max(map(len,rows))) for row in rows if offset<len(row)]
    return torch.tensor(order[:n],device=prompt_ids.device,dtype=torch.long)
