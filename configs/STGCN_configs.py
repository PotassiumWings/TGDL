from pydantic import Field

from configs.arguments import TrainingArguments


class STGCNConfig(TrainingArguments):
    Ks: int = Field(3)
    Kt: int = Field(3)
    graph_conv_type: str = Field("chebconv")
    use_gcn: bool = Field(True)

    adapt_node_emb_len: int = Field(-1)  # -1 不使用自适应邻接矩阵，否则为自适应节点表征维度

    # blocks: str = Field("0, 32, 64, 64, 32, 128")
    blocks: str = Field("0,8,16,16,8,16")

    basic_block: str = Field("16, 8, 16")
    num_st_blocks: int = Field(-1)
    time_padding: int = Field(0)
