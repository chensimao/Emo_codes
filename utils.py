import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.functional import pad
from torch.nn.modules import Module
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import math


def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_rows = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] +
                        (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,
                  padding=(padding_rows // 2, padding_cols // 2),
                  dilation=dilation, groups=groups)

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class Conv2d(_ConvNd): 

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
    
    
    
    
    
class PairPuzzle_AE(nn.Module):

    def __init__(self,model=None,lr=0.0001,num_classes=1000):
        super(PairPuzzle_AE, self).__init__()
        
        self.feature = nn.Sequential(
            # conv_0
            Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # conv_1
            Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # conv_2
            Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # conv_3
            Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # conv_4
            Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(12800, 4608))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))
        self.classifier = nn.Sequential(
            nn.Linear(4608,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_classes)
        )
      
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer=torch.optim.SGD(self.parameters(),lr=lr,momentum=0.9,weight_decay = 5e-4)
        

    

class JigsawPuzzle_AE(nn.Module):

    def __init__(self, lr=0.001,classes=1000):
        super(JigsawPuzzle_AE, self).__init__()

        
        self.conv = nn.Sequential(
            # conv_0
            Conv2d(in_channels=target_shape[0], out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # conv_1
            Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # conv_2
            Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # conv_3
            Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # conv_4
            Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512)
        )
        
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(12800, 512))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))
        self.classifier = nn.Sequential(
            nn.Linear(4608,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer=torch.optim.SGD(self.parameters(),lr=lr,momentum=0.9,weight_decay = 5e-4)

        

class PairPuzzle(nn.Module):

    def __init__(self,lr=0.0001,num_classes=1000):
        super(PairPuzzle, self).__init__()
        
        alexnet = models.alexnet(pretrained=False)
        
        self.feature = alexnet.features
        self.classifier = nn.Sequential(
            nn.Linear(9472,4608),
            nn.ReLU(),
            nn.Linear(4608,4096),
            nn.ReLU(),
            nn.Linear(4096,100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss() 
        self.optimizer=torch.optim.Adam([p for p in self.parameters() if p.requires_grad] , lr=lr)


    
class JigsawPuzzle(nn.Module):

    def __init__(self, lr=0.001,classes=1000):
        super(JigsawPuzzle, self).__init__()
        alexnet = models.alexnet(pretrained=False)
        self.conv = alexnet.features
        
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(256, 512))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))
        
        self.classifier = nn.Sequential(
            nn.Linear(4608,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam([p for p in self.parameters() if p.requires_grad] , lr=lr)
