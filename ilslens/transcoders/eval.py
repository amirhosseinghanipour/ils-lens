import torch

@torch.no_grad()
def metrics(model,x,y):
    denominator = (y-y.mean(0)).square().sum()
    return dict(variance_explained=(1-(model(x)-y).square().sum()/denominator).item(),
                dead_fraction=(model.features(x).sum(0)==0).float().mean().item())

def ce_recovered(clean,replaced,ablated):
    if ablated == clean:
        return float('nan')
    return (ablated-replaced)/(ablated-clean)
