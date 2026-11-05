import torch
from ..fit import fit,r2

def effective_rank(matrix):
    s=torch.linalg.svdvals(matrix)
    if s.sum()==0:return 0.
    p=s/s.sum()
    return torch.exp(-(p*p.clamp_min(1e-30).log()).sum()).item()

def procrustes_experts(matrices):
    """Align polar rotations to expert zero, fixed before readout evaluation.

    This is a diagnostic coordinate change, not an equivalent causal map.
    """
    u,s,vh=torch.linalg.svd(matrices)
    rotations=u@vh
    positive=vh.transpose(-1,-2)@torch.diag_embed(s)@vh
    reference=rotations[0]
    aligned=[]
    for rotation,scale in zip(rotations,positive):
        left,_,right=torch.linalg.svd(rotation.T@reference)
        q=left@right
        aligned.append(rotation@q@scale)
    plain=matrices.mean(0);aligned=torch.stack(aligned).mean(0)
    return dict(plain=plain,aligned=aligned,plain_rank=effective_rank(plain),
                aligned_rank=effective_rank(aligned))

def learning_curve(x,y,prompt_ids,held_x,held_y,prior,lam,estimator='ridge'):
    results=[];d=x.shape[1]
    for n in (d,2*d,4*d,8*d):
        if n>len(x):continue
        matrix=fit(x[:n],y[:n],lam,prior,estimator,prompt_ids[:n])
        results.append(dict(N=n,r2=r2(held_y,held_x@matrix.T)))
    return results
