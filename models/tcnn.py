import torch
import torch.nn as nn
import torch.nn.functional as F

from models.temp_tcn import TemporalConvNet


class TCNN_Old(nn.Module):
    def __init__(self, c_in, d_model):
        super(TCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                               kernel_size=3, padding=1, padding_mode='circular')
        # self.pool = nn.MaxPool1d(2,2)

        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                               kernel_size=3, padding=1, padding_mode='circular')

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        temp = x.permute(0, 2, 1)
        x = self.conv1(x.permute(0, 2, 1)).transpose(1, 2)
        x = x + nn.functional.relu(self.conv2(x.permute(0, 2, 1))).transpose(1, 2)

        return x


# 引入时序卷积神经网络（Temporal Convolutional Network），包括三个主要的模块：（1）因果卷积（Causal Convolution）；
# （2）扩展（膨胀）卷积（dialted Convolution））；（3）残差连接
# 具体分析可参考如下Blog：https://blog.csdn.net/qq_33331451/article/details/104810419
# 论文发表于2018年，CMU大学，题目：An Empirical Evaluation of Generic Convolutional and Recurrent Networks
# for Sequence Modeling
class TCNN(nn.Module):

    def __init__(self, input_size, output_size, num_channels,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        super(TCNN, self).__init__()

        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)

        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        # emb = self.drop(self.encoder(input))
        y = self.tcn(inputs.transpose(1, 2)).transpose(1, 2)
        y = self.linear(y[:, :, :])
        return y.contiguous()
