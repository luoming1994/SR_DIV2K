import torch
import torch.nn as nn

# common conv layer
def conv1x1(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding=0)

def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1)

def conv5x5(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=stride,padding=2)

        
class ResBlock_EDSR(nn.Module):
    '''
    EDSR Residual Block:
    in ->
    conv(channels,channels,stride) -> ReLU
    conv(out_channels,out_channels,1)
    -> out
    (downsample)in + out
    '''
    def __init__(self, channels, stride=1):
        super(ResBlock_EDSR,self).__init__()
        self.conv1 = conv3x3(channels,channels,stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels,channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + x
        #out = self.relu(out)
        return out  

class GenerateNet(nn.Module):
    """
    Generate network,input lr image ,output hr image
    """
    def __init__(self,channels=64):
        super(GenerateNet,self).__init__()
        self.conv1 = conv3x3(1,channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.block1 = ResBlock_EDSR(channels)
        self.block2 = ResBlock_EDSR(channels)
        self.block3 = ResBlock_EDSR(channels)
        self.block4 = ResBlock_EDSR(channels)
        
        self.conv2 = conv3x3(channels,channels//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(channels//2,1)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out