import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.GraphWavenet_configs import GraphWavenetConfig
from models.abstract_st_encoder import AbstractSTEncoder


def calc_sym(adj):
    adj = adj + torch.eye(adj.size(0)).to(adj.device)
    row_sum = torch.sum(adj, dim=0)
    inv_rs = 1 / row_sum
    inv_rs_diag = torch.diag(inv_rs)
    return torch.mm(adj, inv_rs_diag)


class GraphWavenet(AbstractSTEncoder):
    def __init__(self, config: GraphWavenetConfig, gp_supports):
        super(GraphWavenet, self).__init__(config, gp_supports)
        assert (len(gp_supports) == 1)
        self.dropout = config.dropout
        self.blocks = config.blocks
        self.layers = config.layers
        self.num_nodes = config.num_nodes
        self.hidden_size = config.hidden_size
        self.kernel_size = config.kernel_size

        self.tconvs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        residual_channels = self.hidden_size
        dilation_channels = self.hidden_size
        skip_channels = self.hidden_size * 8
        self.end_channels = end_channels = self.hidden_size * 16

        self.start_conv = nn.Conv2d(in_channels=self.config.c_in,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = self.config.output_len

        self.supports_len = 3

        # adaptive node embedding
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device), requires_grad=True).to(
            self.device)
        self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device), requires_grad=True).to(
            self.device)

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.tconvs.append(nn.Conv2d(in_channels=residual_channels,
                                             out_channels=dilation_channels * 2,
                                             kernel_size=(1, self.kernel_size), dilation=new_dilation))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(
                    gcn(dilation_channels, residual_channels, self.dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels * self.config.output_len,
                                    out_channels=self.config.c_out * self.config.output_len,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2_res = nn.Conv2d(in_channels=end_channels * self.config.output_len,
                                        out_channels=self.config.c_in * self.config.input_len,
                                        kernel_size=(1, 1),
                                        bias=True)

        self.receptive_field = receptive_field
        logging.info(f"Receptive field: {receptive_field}")

    def forward(self, x, supports):
        adj = supports[0]
        # change 1: input format
        # x: (batch_size, feature_dim, num_nodes, input_window)
        # inputs = nn.functional.pad(x, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_window+1)
        inputs = x

        in_len = inputs.size(3)
        assert in_len < self.receptive_field
        x = nn.functional.pad(inputs, (self.receptive_field - in_len, 0, 0, 0))
        # else:
        #     x = inputs
        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        # change 2: supports calculation
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        supports = [calc_sym(adj), calc_sym(adj.transpose(0, 1)), adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # (dilation, init_dilation) = self.dilations[i]
            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x

            # (batch_size, dilation_channels * 2, num_nodes, receptive_field-kernel_size+1)
            x = self.tconvs[i](residual)  # x: NCVL
            filter, gate = torch.chunk(x, 2, dim=1)
            x = torch.tanh(filter) * torch.sigmoid(gate)

            # parametrized skip connection
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            s = self.skip_convs[i](x)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            # try:
            #     skip = skip[:, :, :, -s.size(3):]
            # except(Exception):
            #     if type(skip) != int:
            #         logging.info(f"Exception in skip: s {s.size(3)}, skip {skip.size(3)}")
            #     skip = 0
            # logging.info(f"s.shape {s.shape}")
            skip = s[:, :, :, -self.config.output_len:] + skip
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            x = self.gconv[i](x, supports)
            # residual: (batch_size, residual_channels, num_nodes, self.receptive_field)
            x = x + residual[:, :, :, -x.size(3):]
            # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            x = self.bn[i](x)
        # (batch_size, skip_channels, num_nodes, self.output_len)
        x = F.relu(skip)
        # (batch_size, end_channels, num_nodes, self.output_len)
        x = F.relu(self.end_conv_1(x))

        # (batch_size, num_nodes, self.output_len * end_channels, 1)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.num_nodes, self.config.output_len * self.end_channels, 1)
        # (batch_size, self.output_len * end_channels, num_nodes, 1)
        x = x.permute(0, 2, 1, 3)

        # (batch_size, self.c_out * output_len, num_nodes, 1)
        pred = self.end_conv_2(x).permute(0, 2, 3, 1) \
            .reshape(-1, self.num_nodes, self.config.c_out, self.config.output_len).permute(0, 2, 1, 3)
        # (batch_size, self.c_in * input_len, num_nodes, 1)
        residual = self.end_conv_2_res(x).permute(0, 2, 3, 1) \
            .reshape(-1, self.num_nodes, self.config.c_in, self.config.input_len).permute(0, 2, 1, 3)
        return pred, residual


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
