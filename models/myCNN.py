import torch
import  torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torchvision
from thop import profile
class SELayer_2d(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer_2d, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # shape = [32, 64, 2000, 80]

        y = self.avg_pool(x).view(b,c) # shape = [32, 64, 1, 1]
        y=self.fc(y).view(b,c,1,1)

        return x * y.expand_as(x)

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



class Cnn_v1(nn.Module):
    def __init__(self):
        super(Cnn_v1,self).__init__()
        layer1=nn.Sequential()

        layer1.add_module('conv1',nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1))
#         layer1.add_module('attention1',SELayer_2d(16))
       
        layer1.add_module('bn1',nn.BatchNorm2d(16))
        layer1.add_module('relu1',nn.ReLU(True))
        layer1.add_module('pool1',nn.MaxPool2d(2,2))
#         layer1.add_module('attention1',eca_layer(16))
        layer1.add_module('attention1',SELayer_2d(16))
        self.layer1=layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1))
#         layer2.add_module('attention2',SELayer_2d(32))
        layer2.add_module('bn2',nn.BatchNorm2d(32))
#         layer2.add_module('attention2',eca_layer(32))
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
#         layer2.add_module('attention2',eca_layer(32))
#         layer2.add_module('attention2',SELayer_2d(32))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1))
#         layer3.add_module('attention3',SELayer_2d(64))
        layer3.add_module('bn3',nn.BatchNorm2d(64))
#         layer3.add_module('attention3',eca_layer(64))
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))
#         layer3.add_module('attention3',eca_layer(64))
#         layer3.add_module('attention3',SELayer_2d(64))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('conv4', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
#         layer4.add_module('attention4',SELayer_2d(64))
        layer4.add_module('bn4',nn.BatchNorm2d(64))
#         layer4.add_module('attention4',eca_layer(64))
        layer4.add_module('relu4', nn.ReLU(True))
        layer4.add_module('pool4', nn.MaxPool2d(2, 2))
#         layer4.add_module('attention4',eca_layer(64))
#         layer4.add_module('attention4',SELayer_2d(64))
        self.layer4 = layer4

        layer5 = nn.Sequential()
        layer5.add_module('conv5', nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, stride=1, padding=1))
#         layer5.add_module('attention5',SELayer_2d(64))
        layer5.add_module('bn5',nn.BatchNorm2d(64))
        layer5.add_module('relu5', nn.ReLU(True))
        layer5.add_module('pool5', nn.MaxPool2d(2, 2))
#         layer5.add_module('attention5',eca_layer(64))
#         layer5.add_module('attention5',SELayer_2d(64))
        self.layer5 = layer5

        layer6 = nn.Sequential()
        layer6.add_module('fc1',nn.Linear(4 * 7 * 64, 512))
        layer6.add_module('dropout',nn.Dropout2d(0.5))
        layer6.add_module('fc1_relu', nn.ReLU(True))
        layer6.add_module('fc2', nn.Linear(512, 64))
        layer6.add_module('fc2_relu', nn.ReLU(True))
        layer6.add_module('fc3', nn.Linear(64, 4))
        self.layer6=layer6
    def forward(self,x):
        conv1=self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
#         conv3=self.eca(conv3)
        conv4 = self.layer4(conv3)
#         conv4=self.eca(conv4)
        conv5 = self.layer5(conv4)
#         conv5=self.eca(conv5)
        fc_input=conv5.view(conv5.size(0),-1)
        fc_out = self.layer6(fc_input)
        return fc_out
# def conv3x3(in_channels,out_channels,stride=1):
#     return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)

if __name__ == '__main__':
    model=Cnn_v1()
    input = torch.randn(1, 3, 128, 250)
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)




