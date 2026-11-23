import torch 
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000,dropout=0.1):
        super().__init__()
        self.dropout=nn.Dropout(p=dropout)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(max_len,dtype=torch.float32).unsqueeze(1)
        div_term=torch.exp(
            torch.arange(0,d_model,2).float()
            *(-math.log(10000.0)/d_model)
        )
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position *div_term[:pe[:,1::2].shape[0]])
        pe=pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):
        seq_len=x.size(1)
        x=x+self.pe[: ,:seq_len,:]

        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self,max_len,d_model,device):
        super().__init__()
        self.device=device
        self.position_embedding=nn.Embedding(max_len,d_model)
    def forward(self,x):
        x=x.to(self.device)
        batch_size,seq_len,d_model=x.shape
        position=torch.arange(seq_len,device=self.device)
        positional_vecter=self.position_embedding(position)

        return x+positional_vecter.unsqueeze(0)


class RoPE(nn.Module):
    def __init__(self,dim,base=10000.0):
        super().__init__()
        if dim%2 !=0:
            raise ValueError("RoPE dimension must be even.")

        self.dim=dim
        self.bas=base
        in_freq=1.0(base**(torch.arange(0,dim,2,dtype=torch.float32)/dim))
        self.register_buffer(
            'inv_freq',
            in_freq,
            persistent=False
        )

    def _rotate(self,x,cos,sin):
        x_even=x[...,0::2].float()
        x_odd=x[...,1::2].flaot()

        roate_even=(x_even *cos-x_odd*sin)
        roate_odd=(x_even *sin +x_odd *cos)
        roated=torch.stack([roate_even,roate_odd],
                           dim=-1)
        return roated.flatten(-2).to(dtype=x.dtype)

    def forward(self,q,k,position_ids=None):
        if q.shape!=k.shape:
            raise ValueError("Q and K must ne the shape.")
        batch,heads,seq_len,head_dim=q.shape

        if head_dim!=self.dim:
            raise ValueError(
                f"Expected head_dims={self.dim} , got {head_dim}"
            )
        if position_ids is None:
            position_ids=torch.arange(seq_len,
                                      device=q.device)
        position_ids=position_ids.to(
            device=q.deivce,
            dtype=torch.float32
        )
        freqs=(
            position_ids.unsqueeze(-1)*self.inv_freq.to(q.device)
        )
        cos=freqs.cos()
        sin=freqs.sin()

        if position_ids.ndim==1:
            cos=cos[None,None,:,:]
            sin=sin[None,None,:,:]
        elif position_ids.ndim==2:
            sin=sin[:,None,:,:]
            cos=cos[:,None,:,:]
        else:
            raise ValueError(
                "position_ids must have shape [S] or [B, S]"
            )
        q_rot=self._rotate(q,cos,sin)
        k_rot=self._rotate(k,cos,sin)
        return q_rot,k_rot

class ALiBi:
    def __init__(self,dim,)
    