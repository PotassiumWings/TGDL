import torch.nn as nn

from configs.arguments import TrainingArguments
from models.GraphWavenet.GraphWavenet import GraphWavenet
from models.MSDR.MSDR import MSDR
from models.MTGNN.MTGNN import MTGNN
from models.STGCN.STGCN import STGCN
from models.STSSL.STSSL import STSSL
from models.abstract_st_encoder import AbstractSTEncoder


class MDBlock(nn.Module):
    def __init__(self, config: TrainingArguments, supports: list):
        super(MDBlock, self).__init__()
        self.conv: AbstractSTEncoder
        if config.st_encoder == "STGCN":
            self.conv = STGCN(config, supports)
        elif config.st_encoder == "GraphWavenet":
            self.conv = GraphWavenet(config, supports)
        elif config.st_encoder == "STSSL":
            self.conv = STSSL(config, supports)
        elif config.st_encoder == "MSDR":
            self.conv = MSDR(config, supports)
        elif config.st_encoder == "MTGNN":
            self.conv = MTGNN(config, supports)
        else:
            raise NotImplementedError(f"ST Encoder {config.st_encoder} not implemented.")

    def forward(self, x, subgraph):
        # x: (batch_size, c_in, num_nodes, input_len)
        pred, residual = self.conv(x, subgraph)
        return pred, residual

    def get_embedding(self):
        return self.conv.get_embedding()

    def get_forward_loss(self):
        return self.conv.get_forward_loss()
