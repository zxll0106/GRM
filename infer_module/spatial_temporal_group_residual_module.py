import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from infer_module.temporal_group_residual_module import Temporal_Group_Residual_Module

class Spatial_Temporal_Group_Residual_Module(nn.Module):

    def __init__(self, v_len, sub_layers=3, global_layers=1):

        super(Spatial_Temporal_Group_Residual_Module, self).__init__()
        self.st_grms = ST_GRMs(layers_number=sub_layers, v_len=v_len,temporal_layers_number=global_layers)
        self.p_len = v_len * (2 ** sub_layers)

    def forward(self, boxes_features):

        B,T,N,D=boxes_features.shape
        boxes_features=self.st_grms(boxes_features)

        return boxes_features




class ST_GRMs(nn.Module):

    def __init__(self, v_len, layers_number,temporal_layers_number):

        super(ST_GRMs, self).__init__()
        self.layers = nn.Sequential()
        for i in range(layers_number):
            self.layers.add_module("sub{}".format(i), Spatial_GRM(v_len * (2 ** i)))
        self.v_len = v_len
        self.layers_number = layers_number

        self.linear_spatial=nn.Linear(v_len*(2 ** layers_number),v_len)

        self.temporal_grms=Temporal_Group_Residual_Module(v_len,temporal_layers_number)

    def forward(self, x):


        batch_size,t,n,_ = x.shape
        x = self.layers(x)  # [batch_size, v_number, p_len]
        x=x.permute(0,2,1,3)
        nums=torch.range(1,t)

        x=self.linear_spatial(x)
        output=self.temporal_grms(x)
        output=output.transpose(1,2)
        output = output.reshape(batch_size*t,n,-1).transpose(1,2) # [batch size, p_len, v_number]
        output = F.max_pool1d(output, kernel_size=output.shape[2]).squeeze(0)
        output = output.reshape(batch_size,t,-1)  # [batch size, 1, p_len]

        # assert x.shape == (batch_size, t, self.v_len * (2 ** self.layers_number))
        return output


class Spatial_GRM(nn.Module):

    def __init__(self, len):

        super(Spatial_GRM, self).__init__()
        self.g_enc = MLP(len, len,hidden_size=8192)

    def forward(self, x):

        x = self.g_enc(x)
        batch_size,t, n, length = x.shape

        x2=x.reshape(batch_size*t,n,length)
        x2 = x2.permute(0, 2,1)  # [batch_size, len, n]
        x2 = F.max_pool1d(x2, kernel_size=x2.shape[2])  # [batch_size, len, 1]
        x2 = torch.cat([x2] * n, dim=2)  # [batch_size, len, n]
        x2=x2.reshape(batch_size,t,length,n)

        y = torch.cat((x2.permute(0, 1, 3, 2), x), dim=3)
        assert y.shape == (batch_size, t, n, length * 2)
        return y


class MLP(nn.Module):
    r"""
    Construct a MLP, include a single fully-connected layer,
    followed by layer normalization and then ReLU.
    """

    def __init__(self, input_size, output_size, hidden_size=64):
        r"""
        self.norm is layer normalization.
        Args:
            input_size: the size of input layer.
            output_size: the size of output layer.
            hidden_size: the size of output layer.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        r"""
        Args:
            x: x.shape = [batch_size, ..., input_size]
        """
        x = self.fc1(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        r"""
        Layer normalization implemented by myself. 'feature' is the length of input.
        :param features: length of input.
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        r"""

        :param x: x.shape = [batch, feature]
        :return:
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2