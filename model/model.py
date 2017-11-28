import torch
import torch.nn as nn
from torch.autograd import Function

# common conv layer
def conv1x1(in_channels,out_channels,stride=1,bias = False):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,padding=0,bias = False)

def conv3x3(in_channels,out_channels,stride=1,bias=False):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)

def conv5x5(in_channels,out_channels,stride=1,bias=False):
    return nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=stride,padding=2,bias=False)

def conv7x7(in_channels,out_channels,stride=1,bias=False):
    return nn.Conv2d(in_channels,out_channels,kernel_size=7,stride=stride,padding=3,bias=False)

def conv9x9(in_channels,out_channels,stride=1,bias=False):
    return nn.Conv2d(in_channels,out_channels,kernel_size=9,stride=stride,padding=4,bias=False)    

class ReLU01F(Function)：
    """
    f(x) = x ,when 0<x<1;
    f(x) = 0 ,when x<=0;
    f(x) = 1 ,when x>1;
    """
    def forward(self, input):
        self.save_for_backward(input)

        output = input.clamp(min = 0,max = 1)
        return output

    def backward(self, output_grad):
        input, = self.to_save

        input_grad = output_grad.clone()
        input_grad[input < 0] = 0
        input_grad[input > 0] = 0
        return input_grad
    
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

class multiResBlock(nn.Module):
    """
    multi scale Residual Block:
    in ->
    conv3x3 conv5x5 conv7x7 conv9x9 -> ReLU
    conv3x3(out_channels,out_channels,1)
    -> out
    in + out
    """
    def __init__(self, in_channels,out_channels, stride=1):
        super(multiResBlock,self).__init__()
        
        self.conv3 = conv3x3(in_channels,out_channels//4,stride)
        self.conv5 = conv5x5(in_channels,out_channels//4,stride)
        self.conv7 = conv7x7(in_channels,out_channels//4,stride)
        self.conv9 = conv9x9(in_channels,out_channels//4,stride)
        self.relu = nn.ReLU(inplace=True)
            
        self.conv = conv3x3(out_channels,out_channels)

    def forward(self, x):
        out3 = self.conv3(x)
        out3 = self.relu(out3)
        out5 = self.conv5(x)
        out5 = self.relu(out5)
        out7 = self.conv7(x)
        out7 = self.relu(out7)
        out9 = self.conv9(x)
        out9 = self.relu(out9)
        out  = torch.cat((out3,out5,out7,out9),dim=1)
        out  = self.relu(out)
        out  = self.conv(out)

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
        self.conv3 = conv1x1(channels//2,1)
    
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
        
        out = out + x
        return out

class MultiGenerateNet(nn.Module):
    """
    Generate network using multiResBlock,input lr image ,output hr image
    """
    def __init__(self,channels = 64):
        super(MultiGenerateNet,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(1,channels//2)        
        self.conv2 = conv3x3(channels//2,channels)
        
        self.mRB1 = multiResBlock(channels,channels)
        self.mRB2 = multiResBlock(channels,channels)
        self.mRB3 = multiResBlock(channels,channels)
        
        self.conv3 = conv3x3(channels,channels//2)
        self.conv4 = conv1x1(channels//2,1) 
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.mRB1(out)
        out = self.mRB2(out)
        out = self.mRB3(out)
        
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        
        out = out + x
        return out

from math import sqrt


##### VDSR
class Conv_ReLU_Block(nn.Module):
    """
        conv3x3(64 channels) => relu
    """
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class VDSR(nn.Module):
    """
    VDSR:<Accurate Image Super-Resolution Using Very Deep Convolutional Networks>
    """
    def __init__(self):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = conv3x3(1,64)
        self.output = conv3x3(64,1)
        self.relu = nn.ReLU(inplace=True)
        self.relu01 = ReLU01F()
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        out = self.relu01(out)  #  out clamp to [0,1]
        return out
        


class _basicBlock(nn.Module):
    """
    basic Block to build dense block
    """
    def __init__(self,in_planes,out_planes):
        super(_basicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1,
                               padding=2, bias=False)
    def forward(self, x):
        out = self.conv1(self.relu(x))
        
        return torch.cat([x, out], 1)        

        
    
class DenseBlock(nn.Module):
    """
    """
    def __init__(self, nb_layers, in_planes, growth_rate, block):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
        
        
class DenseNet(nn.Module):
    """
    """
    def __init__(self):
        super(DenseNet,self).__init__()
        self.block1 = DenseBlock(4,1,12,_basicBlock)
        self.block2 = DenseBlock(4,1+12*4,12,_basicBlock)
        self.block3 = DenseBlock(4,1+12*8,12,_basicBlock)
        
        self.conv1 = conv1x1(1+12*4,1)
        self.conv2 = conv1x1(1+12*8,1)
        self.conv3 = conv1x1(1+12*12,1)
    
    def forward(self,x):
        out = self.block1(x)
        #print out.size
        out1 = self.conv1(out)
        out = self.block2(out)
        #print out.size
        out2 = self.conv2(out)
        out = self.block3(out)
        #print out.size
        out3 = self.conv3(out)
        return out1,out2,out3        


