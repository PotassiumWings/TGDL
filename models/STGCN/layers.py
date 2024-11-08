import torch
import torch.nn as nn

from models.STGCN.spatio_layers import SpatioConvLayer, SparseSpatioConvLayer, SpatioConvLayerGCN
from models.STGCN.utils import Align


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, kernel_size=(1, kt))
        else:
            self.conv = nn.Conv2d(c_in, c_out, kernel_size=(1, kt))

    def forward(self, x):
        """
        :param x: (batch_size, c_in, num_nodes, time_steps)
        :return: (batch_size, c_out, num_nodes, time_steps-kt+1)
        """
        # x_in: for residuals (batch_size, c_out, num_nodes, time_steps-kt+1)
        x_in = self.align(x)[:, :, :, self.kt - 1:]
        if self.act == "GLU":
            # x: (batch_size, c_in, num_nodes, time_steps)
            x_conv = self.conv(x)
            # x_conv: (batch_size, c_out * 2, num_nodes, time_steps-kt+1)  [P Q]
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            # return P * sigmoid(Q) shape: (batch_size, c_out, num_nodes, time_steps-kt+1)
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  # residual connection
        return torch.relu(self.conv(x) + x_in)  # residual connection


class STConvBlock(nn.Module):
    def __init__(self, ks, kt, n, adj, c, p, nh, sparse, batch_size, input_len, use_gcn, node_emb_len, device):
        super(STConvBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        if not sparse:
            if use_gcn:
                self.sconv = SpatioConvLayerGCN(c[1], c[1], n, node_emb_len, device)
            else:
                self.sconv = SpatioConvLayer(c[1], c[1], n, nh, adj, device)
        else:
            self.sconv = SparseSpatioConvLayer(c[1], c[1], n, nh, adj, batch_size, input_len - kt + 1, device)
        self.tconv2 = TemporalConvLayer(kt, c[1], c[2])
        self.layer_norm = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x, adj):  # x: (batch_size, c[0], num_nodes, time_steps)
        x_t1 = self.tconv1(x)  # (batch_size, c[1], num_nodes, time_steps-kt+1)
        x_s = self.sconv(x_t1, adj)  # (batch_size, c[1], num_nodes, time_steps-kt+1)
        x_t2 = self.tconv2(x_s)  # (batch_size, c[2], num_nodes, time_steps-kt+1-kt+1)
        x_ln = self.layer_norm(x_t2.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return self.dropout(x_ln)


class OutputLayer(nn.Module):
    def __init__(self, c, t, n, out_dim):
        super(OutputLayer, self).__init__()
        self.tconv1 = TemporalConvLayer(t, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = TemporalConvLayer(1, c, c, "sigmoid")  # kernel=1*1
        self.fc_1 = nn.Conv2d(c, out_dim, kernel_size=(1, 1))
        # self.fc_2 = nn.Conv2d(c, in_dim, kernel_size=(1, 1))

    def forward(self, x):
        # x, x_t1: (batch_size, input_dim(c), num_nodes, T)
        x_t1 = self.tconv1(x)
        # x_ln: (batch_size, input_dim(c), num_nodes, 1)
        x_ln = self.ln(x_t1.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # x_t2: (batch_size, input_dim(c), num_nodes, 1)
        x_t2 = self.tconv2(x_ln)
        # return: (batch_size, out_dim, num_nodes, 1)
        return self.fc_1(x_t2)
        # return self.fc_1(x_t2), self.fc_2(x_t2)


class OutputLayer2(nn.Module):  # Residual
    def __init__(self, c, n, t, input_len, out_dim):
        super(OutputLayer2, self).__init__()
        self.tconv1 = TemporalConvLayer(1, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = TemporalConvLayer(1, c, c, "sigmoid")  # kernel=1*1
        self.fc_1 = nn.Conv2d(t, input_len, kernel_size=(1, 1))
        self.fc_2 = nn.Conv2d(c, out_dim, kernel_size=(1, 1))

    def forward(self, x):
        # x, x_t1: (batch_size, input_dim(c), num_nodes, t)
        x_t1 = self.tconv1(x)
        # x_ln: (batch_size, input_dim(c), num_nodes, t)
        x_ln = self.ln(x_t1.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # x_t2: (batch_size, input_dim(c), num_nodes, t)
        x_t2 = self.tconv2(x_ln)
        # return: (batch_size, input_dim(c), num_nodes, input_len)
        fc1 = self.fc_1(x_t2.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return self.fc_2(fc1)
