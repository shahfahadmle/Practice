import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,num_classes=10):
        super(VGG,self).__init__()
        #(3,32,32)->(64,32,32)
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)#->(64,16,16)
        )
        #(64,16,116)->(128,16,16)
        self.block2=nn.Sequential(
                    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2,stride=2)#->(128,8,8)
                )
        self.block3=nn.Sequential(
                    nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2,stride=2)#->(256,4,4)
                )
        self.block4=nn.Sequential(
                    nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2,stride=2)#->(512,2,2)
                )
        self.block5=nn.Sequential(
                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2,stride=2)#->(512,1,1)
                )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512,out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024,out_features=num_classes)
            
        )
    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.classifier(x)
    

        return x
                
                
                