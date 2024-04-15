import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_Group_Residual_Module(nn.Module):

    def __init__(self, v_len, sub_layers=3):

        super(Temporal_Group_Residual_Module, self).__init__()
        self.temporal_grms = Temporal_GRMs(layers_number=sub_layers, v_len=v_len)
        self.p_len = v_len * (2 ** sub_layers)

    def forward(self, boxes_features):

        boxes_features=self.temporal_grms(boxes_features)

        return boxes_features


class Temporal_GRMs(nn.Module):

    def __init__(self, v_len, layers_number):

        super(Temporal_GRMs, self).__init__()
        self.layers = nn.Sequential()
        for i in range(layers_number):
            self.layers.add_module("sub{}".format(i), GRM(v_len * (2 ** i)))
        self.v_len = v_len
        self.layers_number = layers_number

    def forward(self, x):

        # batch_size,t,n,_ = x.shape
        output = self.layers(x)  # [batch_size, v_number, p_len]


        # assert x.shape == (batch_size, t, self.v_len * (2 ** self.layers_number))
        return output


class GRM(nn.Module):

    def __init__(self, len):

        super(GRM, self).__init__()
        self.g_enc = MLP(len, len,hidden_size=8192)

    def forward(self, x):

        x = self.g_enc(x)
        batch_size,n,t, length = x.shape

        x=x.reshape(batch_size*n,t,length)

        x_square=torch.repeat_interleave(x.unsqueeze(1),t,dim=1)

        temporal_mask = torch.tril(torch.ones(t, t), diagonal=0).cuda()

        x2=torch.mul(torch.repeat_interleave(temporal_mask.unsqueeze(-1),length,dim=-1),x_square)
        x2=x2.sum(2)/torch.repeat_interleave(torch.range(1,t).unsqueeze(-1),length,dim=-1).cuda()


        y = torch.cat((x2, x), dim=-1)
        y=y.reshape(batch_size,n,t,-1)
        assert y.shape == (batch_size, n, t, length * 2)
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