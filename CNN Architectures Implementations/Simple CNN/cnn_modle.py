import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,
                             out_channels=16,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        self.conv2=nn.Conv2d(in_channels=16,
                             out_channels=32,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu=nn.ReLU()
        self.fc=nn.Linear(32*7*7,num_classes)  # Assuming input images are 28x28

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)

        return x

    
if __name__ == "__main__":

    model = CNN(in_channels=1, out_channels=32)

    dummy_input = torch.randn(4, 1, 28, 28) 
    output = model(dummy_input)
    
    print(f"Success! Output shape: {output.shape}") # Should be [4, 10]