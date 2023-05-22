import torch
import torch.nn as nn
import torchvision.models as models


class DenseNet(nn.Module):
    def __init__(self, pretrained=False):
        super(DenseNet, self).__init__()
#         num_classes =4
        self.model = models.mobilenet_v2()
        print("mobilenet")
#         self.model = models.squeezenet1_1(pretrained=pretrained)
#         self.model = models.vgg16(pretrained=pretrained)
#         print("vgg16")
#         self.model.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        output = self.model(x)
        return output
