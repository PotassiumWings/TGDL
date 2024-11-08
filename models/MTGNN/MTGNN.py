import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.MTGNN_configs import MTGNNConfig
from models.MTGNN.layers import dilated_inception, mixprop, LayerNorm, graph_constructor
from models.abstract_st_encoder import AbstractSTEncoder


class MTGNN(AbstractSTEncoder):
    def __init__(self, config: MTGNNConfig, gp_supports):
        super(MTGNN, self).__init__(config, gp_supports)

        self.num_nodes = config.num_nodes
        self.dropout = config.dropout
        self.in_dim = config.c_in
        self.out_dim = config.c_out
        self.residual_channels = config.residual_channels
        self.skip_channels = config.skip_channels
        self.conv_channels = config.conv_channels
        self.end_channels = config.end_channels
        self.dilation_exponential = config.dilation_exponential
        self.gcn_depth = config.gcn_depth
        self.layers = config.layers
        self.propalpha = config.propalpha
        self.layer_norm_affline = config.layer_norm_affline
        self.use_graph_constructor = config.use_graph_constructor

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(self.num_nodes, config.subgraph_size, config.node_dim, self.device,
                                    alpha=config.tanhalpha, static_feat=None)

        self.seq_length = config.input_len

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(1 + (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1) / (
                    self.dilation_exponential - 1))
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + 1

        for i in range(1):
            if self.dilation_exponential > 1:
                rf_size_i = int(1 + i * (kernel_size - 1) * (self.dilation_exponential ** self.layers - 1) / (
                        self.dilation_exponential - 1))
            else:
                rf_size_i = i * self.layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, self.layers + 1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size - 1) * (self.dilation_exponential ** j - 1) / (
                            self.dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.skip_channels,
                                                     kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                self.gconv1.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout,
                                           self.propalpha))
                self.gconv2.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout,
                                           self.propalpha))

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm((self.residual_channels, self.num_nodes, self.seq_length - rf_size_j + 1),
                                  elementwise_affine=self.layer_norm_affline))
                else:
                    self.norm.append(
                        LayerNorm((self.residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1),
                                  elementwise_affine=self.layer_norm_affline))

                new_dilation *= self.dilation_exponential

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels * self.config.output_len,
                                    out_channels=self.config.c_out * self.config.output_len,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2_res = nn.Conv2d(in_channels=self.end_channels * self.config.output_len,
                                        out_channels=self.config.c_in * self.config.input_len,
                                        kernel_size=(1, 1),
                                        bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels,
                                   kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels,
                                   kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(self.device)

    def forward(self, x, supports):
        input_x = x
        seq_len = input_x.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input_x = nn.functional.pad(input_x, (self.receptive_field - self.seq_length, 0, 0, 0))

        if self.use_graph_constructor:
            adp = self.gc(self.idx)
        else:
            adp = supports[0]

        x = self.start_conv(input_x)
        skip = self.skip0(F.dropout(input_x, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter_ = self.filter_convs[i](x)
            filter_ = torch.tanh(filter_)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter_ * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))

            x = x + residual[:, :, :, -x.size(3):]

            x = self.norm[i](x, self.idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
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
