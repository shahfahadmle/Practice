import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()

        assert d_model%num_heads==0

        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model // num_heads

        self.qkv=nn.Linear(d_model,3*d_model)

        self.out=nn.Linear(d_model,d_model)

    def forward(self, x:torch.Tensor , mask:torch.Tensor |None=None):
        B,T,D=x.shape

        qkv=self.qkv(x)

        qkv=qkv.reshape(B,T,3,self.num_heads,self.head_dim)
         # (3, B, heads, T, head_dim)
        qkv=qkv.permute(2,0,3,1,4)
        Q,K,V=qkv.unbind(0)
        scores=torch.matmul(Q,K.transpose(-2,-1))
        scores=scores/math.sqrt(self.head_dim)

        if mask is not None:
            scores=scores.masked_fill(mask==0,float('-inf'))
        attention=F.softmax(scores,dim=-1)
        context=torch.matmul(attention,V)
        context=context.transpose(1,2).contiguous().reshape(B,T,D)
        

        final_output=self.out(context)

        return final_output,attention

batch_size=2
seq_len=5
d_model=512
num_heads=8

x=torch.randn(batch_size,seq_len,d_model)
model=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
output,attention=model(x)

print(f"input shpae:{x.shape}")
print(f"Attention shape : {attention.shape }")
print(f"out put shpae :{output.shape}")



