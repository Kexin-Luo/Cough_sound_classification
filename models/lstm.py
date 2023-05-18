import torch
import torch.nn as nn
from torchstat import stat
from thop import profile
from models import inception
from models import resnet
from models import ResneXt
from models import Mel_CNN
from models import myCNN
from thop import profile

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        layer1=nn.Sequential()
        layer1.add_module('lstm1',nn.LSTM( input_size=250,
            hidden_size=128,
            num_layers=1,
            batch_first=True,))
        self.layer1=layer1
        layer2=nn.Sequential()
        layer2.add_module('lstm2',nn.LSTM(input_size=128,
            hidden_size=256,
            num_layers=1,
            batch_first=True))
        self.layer2=layer2
        layer3 = nn.Sequential()
        layer3.add_module('classifier',nn.Linear(256,4))
        layer3.add_module('relu',nn.Sigmoid())
        self.layer3=layer3

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # x = x.view(-1, 128, 250)
        x = x.view(-1, 128*3, 250)
        x, (h_n, h_c)=self.layer1(x)
        x, (h_n, h_c)=self.layer2(x)

        # None represents zero initial hidden state
        # r_out, (h_n, h_c) = self.rnn_1(r_out, None)
        # r_out, (h_n, h_c) = self.rnn_2(r_out, None)
        # choose r_out at the last time step
        # r_out = self.rnn(r_out)
        out = self.layer3(x[:, -1, :])
        return out


if __name__ == '__main__':
    model=()
    # y=model(torch.randn(1,3,128,250))
    # print(model)
    # net = RNN()  # 定义好的网络模型
    input = torch.randn(1, 1, 128, 250)
    flops, params = profile(model, (input,))
    print('flops: ', flops, 'params: ', params)



