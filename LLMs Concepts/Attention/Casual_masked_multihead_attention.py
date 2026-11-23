import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CasualSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.0):
        super().__init__()
        assert d_model%num_heads==0,"Num heads should be divisiable by d_model"
        
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads
        self.qkv=nn.Linear(d_model,3*d_model)
        self.out=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,padding_mask=None):
        B,T,D=x.shape
        qkv=self.qkv(x)
        qkv=qkv.reshape(B,T,3,self.num_heads,self.head_dim)
        qkv=qkv.permute(2,0,3,1,4)
        Q,K,V=qkv.unbind(0)

        scores=torch.matmul(Q,K.transpose(-2,-1))
        scores=scores/math.sqrt(self.head_dim)
        mask=torch.triu(torch.ones((T,T),device=x.device,dtype=torch.bool),diagonal=1)
        if padding_mask is not None:
            pad_mask=~(padding_mask.to(device=x.device,dtype=torch.bool))
            pad_mask=pad_mask.unsqueeze(1).unsqueeze(2)
            combined_mask=mask.unsqueeze(0).unsqueeze(0) |pad_mask
        else:
            combined_mask=mask

        scores=scores.masked_fill(combined_mask,float('-inf'))
        attention=F.softmax(scores,dim=-1)
        attention=torch.nan_to_num(attention,nan=0.0)
        attention=self.dropout(attention)
        context=torch.matmul(attention,V)
        context=context.transpose(1,2).contiguous().reshape(B,T,D)
        output=self.out(context)


        return output,attention



# --- Example Usage ---
batch_size = 2
seq_len = 5
d_model = 512
num_heads = 8

x = torch.randn(batch_size, seq_len, d_model)

# 1 = real token, 0 = [PAD] token
# Sequence 0 has 3 real tokens and 2 [PAD] tokens
# Sequence 1 has 5 real tokens and 0 [PAD] tokens
padding_mask = torch.tensor([
    [1, 1, 1, 1, 1]
])

model = CasualSelfAttention(d_model=d_model, num_heads=num_heads)
output, attention = model(x, padding_mask=padding_mask)

print(f"Input shape: {x.shape}")
print(f"Attention shape: {attention.shape}")
print(f"Output shape: {output.shape}")
print("\nAttention weights for Batch 0, Head 0 (last 2 columns are 0.0):")
print(attention[0, 0])