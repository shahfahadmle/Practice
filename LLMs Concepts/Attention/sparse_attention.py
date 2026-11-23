import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SparseAttentionBase(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0.0):
        super().__init__()
        assert embed_dim%num_heads==0
        self.embed_dim=embed_dim
        self.dropout=dropout
        self.num_heads=num_heads
        self.heads_dim=embed_dim//num_heads
        self.qkv=nn.Linear(embed_dim,3*embed_dim)
        self.output=nn.Linear(embed_dim,embed_dim)
    def project_qkv(self,x):
        B,T,D=x.shape
        qkv=self.qkv(x)
        qkv=qkv.reshape(B,T,3,self.num_heads,self.heads_dim)
        # (3, B, H, N, Dh)
        qkv=qkv.permute(2,0,3,1,4)
        Q,K,V=qkv.unbind(0)

        return Q,K,V

    def merge_heads(self,output):
        # (B, H, N, Dh) -> (B, N, H, Dh)
        output = output.transpose(1, 2).contiguous()

        B, N, H, Dh = output.shape

        return output.reshape(B, N, H * Dh)

    
    def forward_attention(self,x,mask):
        Q,K,V=self.project_qkv(x)
        scores=Q@K.transpose(-2,-1)
        scores=scores/(self.heads_dim*0.5)
        if mask.dim()==2:
            scores=scores.masked_fill(~mask[None,None,:,:],float('-inf'))
        elif mask.dim()==3:
            scores=scores.masked_fill(~mask[:,None,:,:],float('-inf'))
        weights=F.softmax(scores,dim=-1)
        weights=F.dropout(weights,p=self.dropout,training=self.training)
        output=weights@V
        output=self.merge_heads(output)
        return self.output(output)

class LocalSparseAttention(SparseAttentionBase):
    def __init__(self,embed_dim,num_heads,radius,dropout=0.0):
        super().__init__(embed_dim,num_heads,dropout)
        self.radius=radius

    def create_mask(self,N,device):
        positon=torch.arange(N,device=device)
        distance=(positon[:,None]-positon[None,:]).abs()

        return distance<=self.radius
    
    def forward(self,x):
        B,N,D=x.shape
        mask=self.create_mask(N,x.device)
        return self.forward_attention(x,mask=mask)

class CausalSparseAttention(SparseAttentionBase):
    def __init__(self,embed_dim,num_heads,window_size=3,dropout=0.0):
        super().__init__(embed_dim,num_heads,dropout)
        self.window_size=window_size
    def create_mask(self,N,device):
        position=torch.arange(N,device=device)
        distences=(position[None:]-position[:None]).abs()
        casual=distences>=0
        local=distences<self.window_size
        return casual & local
    def forward(self,x):
        B,N,D=x.shape
        mask=self.create_mask(N,device=x.device)

        return self.forward_attention(x,mask)



class GlobalSparseAttention(SparseAttentionBase):
    def __init__(self,embed_dims,num_heads,global_tokens=None,dropout=0.0):
        super().__init__(embed_dims,num_heads,dropout)
        self.global_tokens=(global_tokens if global_tokens is not None else [0])
    
    def create_mask(self,N,device):
        mask=torch.eye(N,dtype=torch.bool,device=device)
        for token in self.global_tokens:
            if not 0<=token<N:
                raise ValueError('Invalid global token index')
            mask[token,:]=True
            mask[:,token]=True
        
        return mask
    def forward(self,x):
        B,N,D=x.shape
        mask=self.create_mask(N,device=x.device)
        return self.forward_attention(x,mask)


class StridedSparseAttention(SparseAttentionBase):
    def __init__(self,embed_dim,num_heads,stride=2,casual=None,dropout=0.0):
        super().__init__(embed_dim,num_heads,dropout)
        self.dropout=dropout

        if stride<1:
            raise ValueError("Stride msut be greater or equal to zero.")
        self.stride=stride
        self.casual=casual
    def create_mask(self,N,device):
        positions=torch.arange(N,device=device)

        distence=(positions[:,None]-positions[None,:])
        mask=distence.remainder(self.stride).bool()
        if self.casual:
            mask=mask &(distence>=0)
        return mask
    def forward(self,x):
            B,N,D=x.shape
            mask=self.create_mask(N,device=x.device)
            return self.forward_attention(x,mask)


class RandomSparseAttention(SparseAttentionBase):
    def __init__(self,embed_dim,num_heads,random_connections=2,casual=False,dropout=0.0):
        super().__init__(embed_dim,num_heads,dropout)
        self.random_connections=random_connections
        self.casual=casual
    
    def create_mask(self,N,device):
        positons=torch.arange(N,device=device)
        mask=torch.eye(N,dtype=torch.bool,device=device)
        for i in range(N):
            if self.casual:
                condidates=positons[:i+1]
            else:
                condidates=positons
            
            condidates=condidates[condidates!=i]

            k=min(self.random_connections,len(condidates))

            if k>0:
                selected=condidates[torch.randperm(len(condidates),device=device)[:k]]
                mask[i,selected]=True
        return mask
    def forward(self,x):
        B,N,D=x.shape
        mask=self.create_mask(N,device=x.device)
        return self.forward_attention(x,mask=mask)


model=RandomSparseAttention(embed_dim=64,num_heads=4,random_connections=3)
x=torch.randn(2,8,64)
y=model(x)
print(y.shape)







# model=StridedSparseAttention(embed_dim=64,num_heads=4,stride=4)
# x=torch.randn(2,8,64)
# pred=model(x)
# print(pred.shape)





# model = GlobalSparseAttention(
#     embed_dims=64,
#     num_heads=4,
#     global_tokens=[0, 3]
# )

# x = torch.randn(2, 8, 64)

# y = model(x)

# print(y.shape)




# model = CausalSparseAttention(
#     embed_dim=64,
#     num_heads=4,
#     window_size=3
# )

# x = torch.randn(2, 8, 64)

# y = model(x)

# print(y.shape)




# model = LocalSparseAttention(
#     embed_dim=64,
#     num_heads=4,
#     radius=2
# )

# x = torch.randn(2, 8, 64)

# y = model(x)

# print(y.shape)
# # torch.Size([2, 8, 64])
