import torch
from torch import nn

class SkipTranscoder(nn.Module):
    def __init__(self,d,width=None,k=8):
        super().__init__()
        width = width or 8*d
        self.encoder = nn.Linear(d,width)
        self.decoder = nn.Linear(width,d)
        self.skip = nn.Linear(d,d,bias=False)
        self.k = min(k,width)
    def features(self,x):
        values = self.encoder(x).relu()
        chosen = values.topk(self.k,dim=-1).indices
        return torch.zeros_like(values).scatter(-1,chosen,values.gather(-1,chosen))
    def forward(self,x):
        return self.decoder(self.features(x))+self.skip(x)

def train(x,y,steps=100,seed=0):
    torch.manual_seed(seed)
    model = SkipTranscoder(x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (model(x)-y).square().mean()
        loss.backward()
        optimizer.step()
    return model
