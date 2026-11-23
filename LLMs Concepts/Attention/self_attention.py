import torch
import torch.nn as nn 
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self,d_model:int):
        super().__init__()
        self.d_model=d_model
        self.query=nn.Linear(d_model,d_model)
        self.key=nn.Linear(d_model,d_model)
        self.value=nn.Linear(d_model,d_model)

        self.out=nn.Linear(d_model,d_model)

    def forward(self,x:torch.Tensor,mask:torch.Tensor | None =None):
        Q=self.query(x)
        K=self.key(x)
        V=self.value(x)

        scores=torch.matmul(Q,K.transpose(-2,-1))

        scores=scores/math.sqrt(self.d_model)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) # type: ignore

        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        out=self.out(context)

        return out ,weights

batch_size=2
seq_len=5
d_model=8
x=torch.randn(batch_size,seq_len,d_model)
model=SelfAttention(d_model)
out,attention=model(x)

print(f"input shpae:{x.shape}")
print(f"Attention shape : {attention.shape }")
print(f"out put shpae :{out.shape}")


        