import torch
import torch.nn as nn
import torchvision.models as models
from torchstat import stat
from thop import profile
# from torchsummary import summary

class Inception(nn.Module):
	def __init__(self,pretrained=False):
		super(Inception, self).__init__()
		num_classes =4
		self.model = models.inception_v3(pretrained=pretrained, aux_logits=False)
		self.model.fc = nn.Linear(2048, num_classes)

	def forward(self, x):
		output = self.model(x)
		return output
if __name__ == '__main__':
	model = Inception()
	# y=model(torch.randn(1,3,128,250))
	# print(model)
	# net = RNN()  # 定义好的网络模型
	input = torch.randn(1, 3, 128, 250)
	flops, params = profile(model, (input,))
	print('flops: ', flops, 'params: ', params)
