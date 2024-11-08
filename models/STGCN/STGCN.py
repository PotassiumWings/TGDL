from logging import getLogger

import torch.nn as nn

from configs.STGCN_configs import STGCNConfig
from models.STGCN.layers import OutputLayer, OutputLayer2, STConvBlock
from models.abstract_st_encoder import AbstractSTEncoder


class STGCN(AbstractSTEncoder):
    def __init__(self, config: STGCNConfig, gp_supports):
        super(STGCN, self).__init__(config, gp_supports)
        assert (len(gp_supports) == 1)
        self._logger = getLogger()

        self.num_nodes = self.config.num_nodes
        self.num_heads = self.config.num_heads

        self.Ks = config.Ks
        self.Kt = config.Kt

        if config.num_st_blocks == -1:
            blocks = list(map(int, config.blocks.replace(" ", "").split(",")))
            self.blocks = [blocks[0:3], blocks[3:6]]
            self.blocks[0][0] = self.config.c_in
        else:
            blocks = list(map(int, config.basic_block.replace(" ", "").split(",")))
            self.blocks = [blocks.copy() for i in range(config.num_st_blocks)]
            self.blocks[0][0] = self.config.c_in

        self.time_padding = config.time_padding
        self.input_window = self.config.input_len + self.time_padding
        self.drop_prob = self.config.dropout

        if self.input_window - len(self.blocks) * 2 * (self.Kt - 1) <= 0:
            raise ValueError('Input_window must bigger than 4*(Kt-1) for 2 STConvBlock'
                             ' have 4 kt-kernel convolutional layer.')

        # 模型结构
        self.st_convs = []
        for i in range(len(self.blocks)):
            conv = STConvBlock(ks=self.Ks, kt=self.Kt, n=self.num_nodes, adj=self.gp_supports[0],
                               c=self.blocks[i], p=self.drop_prob, nh=self.num_heads, sparse=self.config.sparse,
                               batch_size=self.config.batch_size, input_len=self.input_window - 2 * (self.Kt - 1) * i,
                               use_gcn=config.use_gcn, node_emb_len=config.adapt_node_emb_len,
                               device=self.device)
            self.st_convs.append(conv)
            self.register_module(f"st_conv_{i}", conv)

        self.output_1 = OutputLayer(self.blocks[-1][2], self.input_window - len(self.blocks) * 2
                                    * (self.Kt - 1), self.num_nodes, out_dim=self.config.c_out)
        self.output_2 = OutputLayer2(self.blocks[-1][2], self.num_nodes, self.input_window - len(self.blocks) * 2
                                     * (self.Kt - 1), self.config.input_len, out_dim=self.config.c_in)
        # self.output_2 = OutputLayer(self.blocks[1][2], self.input_window - len(self.blocks) * 2
        #                             * (self.Kt - 1), self.num_nodes, out_dim=c_in)

    def forward(self, x, gp_supports):
        # x: (batch_size, c_in, num_nodes, input_len)
        x = nn.functional.pad(x, (self.time_padding, 0, 0, 0, 0, 0))

        cur_x = x
        for i in range(len(self.blocks)):
            cur_x = self.st_convs[i](cur_x, gp_supports[0])
        self.cache_embedding = cur_x

        # method 1: 2 output layers
        outputs_1 = self.output_1(cur_x)  # (batch_size, c_out, num_nodes, output_len(1))
        outputs_2 = self.output_2(cur_x)  # (batch_size, c_in, num_nodes, input_len(19))

        # method 2: 2 fc
        # outputs_1, outputs_2 = self.output_1(cur_x)
        return outputs_1, outputs_2

    def get_embedding(self):
        return self.cache_embedding

    # def calculate_loss(self, x, y):
    #     y_predicted = self(x).squeeze(3).squeeze(1)  # (batch_size, num_nodes)
    #     return loss.masked_mse_torch(y_predicted, y)
