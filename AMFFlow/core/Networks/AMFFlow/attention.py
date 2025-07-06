import torch
from torch import nn, einsum
from einops import rearrange

class Attention(nn.Module):
    def __init__(
        self,
        input_dim,
        heads = 4,
        dim_head = 128,
        
    ):
        super().__init__()
       
        self.heads = heads
        
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        self.to_q = nn.Conv2d(input_dim, inner_dim ,1, bias=False)
        self.to_k = nn.Conv2d(input_dim, inner_dim ,1, bias=False)
        
    
    
    def forward(self, fmaps0,fmaps1,temp=None):
        assert fmaps0.shape == fmaps1.shape
        heads, b, c, h, w = self.heads, *fmaps0.shape

        q = self.to_q(fmaps0)
        k = self.to_k(fmaps1)
         
        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
       
        if temp is not None:
            scale = self.scale*temp
        else:
            scale = self.scale

        q = scale* q
        sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1).squeeze(1)

        return attn

