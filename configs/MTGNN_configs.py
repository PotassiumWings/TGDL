from pydantic import Field

from configs.arguments import TrainingArguments


class MTGNNConfig(TrainingArguments):
    layers: int = Field(3)
    residual_channels: int = Field(16)
    conv_channels: int = Field(16)
    skip_channels: int = Field(32)
    end_channels: int = Field(64)
    dilation_exponential: int = Field(2)
    gcn_depth: int = Field(2)
    propalpha: float = Field(0.05)
    layer_norm_affline: bool = Field(False)

    # dynamic graph
    use_graph_constructor: bool = Field(False)
    subgraph_size: int = Field(20)
    node_dim: int = Field(40)
    tanhalpha: float = Field(3)
