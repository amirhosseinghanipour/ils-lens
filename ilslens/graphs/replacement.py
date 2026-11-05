import torch
from torch import nn

class FrozenTranscoder(nn.Module):
    """Freeze TopK/ReLU support, preserving a locally affine feature graph."""
    def __init__(self,transcoder,clean_input,clean_output):
        super().__init__()
        self.transcoder = transcoder
        with torch.no_grad():
            self.register_buffer('support',(transcoder.features(clean_input)>0).float())
            self.register_buffer('error',clean_output-transcoder(clean_input))
    def forward(self,x):
        features = self.transcoder.encoder(x)*self.support
        return self.transcoder.decoder(features)+self.transcoder.skip(x)+self.error
