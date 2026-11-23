import torch
import torch.nn as nn

class HosingModel(nn.Module):
    def __init__(self,in_features,hidden_unit,out_features=1):
        super().__init__()
        self.linear1=nn.Linear(in_features,hidden_unit)
        self.linear2=nn.Linear(hidden_unit,hidden_unit)
        self.linear3=nn.Linear(hidden_unit,hidden_unit)
        self.out=nn.Linear(hidden_unit,out_features)
        self.relu=nn.ReLU()
        


    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        x=self.linear3(x)
        x=self.relu(x)
        x=self.out(x)

        return x
