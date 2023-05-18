import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
class Mel_CNN(nn.Module):
    def __init__(self):
        super(Mel_CNN,self).__init__()
        layer1=nn.Sequential()
        layer1.add_module('conv1',nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1))
        layer1.add_module('relu1',nn.ReLU(True))
        layer1.add_module('pool1',nn.MaxPool2d(2,2))
#         layer1.add_module('attention1',SELayer_2d(16))
        self.layer1=layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1))
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2))
#         layer2.add_module('attention2',SELayer_2d(32))
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1))
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2))
#         layer3.add_module('attention3',SELayer_2d(64))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('conv4', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        layer4.add_module('relu4', nn.ReLU(True))
        layer4.add_module('pool4', nn.MaxPool2d(2, 2))
#         layer4.add_module('attention4',SELayer_2d(64))
        self.layer4 = layer4

        layer5 = nn.Sequential()
        layer5.add_module('conv5', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        layer5.add_module('relu5', nn.ReLU(True))
        layer5.add_module('pool5', nn.MaxPool2d(2, 2))
#         layer5.add_module('attention5',eca_layer(64))
        self.layer5 = layer5

        layer6 = nn.Sequential()
        layer6.add_module('fc1',nn.Linear(4 * 7 * 64, 512))
        layer6.add_module('fc1_relu', nn.ReLU(True))
        layer6.add_module('fc2', nn.Linear(512, 64))
        layer6.add_module('fc2_relu', nn.ReLU(True))
        layer6.add_module('fc3', nn.Linear(64, 4))
        self.layer6=layer6
    def forward(self,x):
        conv1=self.layer1(x)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv4 = self.layer4(conv3)
        conv5 = self.layer5(conv4)
        fc_input=conv5.view(conv5.size(0),-1)
        fc_out = self.layer6(fc_input)
        return fc_out
if __name__ == '__main__':
    model=Mel_CNN()
    input = torch.randn(1, 1, 128, 250)
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)



